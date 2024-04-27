"""
Streaming-LLM: Triton Implementation
gmlwns2000, jeffwillette @ github
"""

import math
import time
import numpy as np
import torch
import triton
from torch.nn import functional as F
from torch import nn
import triton.language as tl
from torch import Tensor
from torch.autograd import Function
from typing import List, Optional, Tuple, Dict, Any
import warnings


@triton.jit
def _update_sink_cache(
        # input tensors
        KEY,
        VAL,
        stride_k_n,
        stride_k_h,
        stride_k_t,
        stride_k_hid,
        SINK_K,
        SINK_V,
        stride_sk_n,
        stride_sk_h,
        stride_sk_t,
        stride_sk_hid,
        SINK_MASK,
        stride_m_n,
        stride_m_h,
        stride_m_t,
        SINK_POS,
        stride_p_n,
        stride_p_h,
        stride_p_t,
        STORED_SINKS,
        stride_ss_n,
        stride_ss_h,

        # input variables
        N,
        K,
        HID,
        NUM_SINK,
        WINDOW_SIZE,

        # kernel constants
        BLOCK_HID: tl.constexpr):

    dtype = tl.float16
    idtype = tl.int64

    idx_hid = tl.arange(0, BLOCK_HID).to(idtype)
    mask_hid = idx_hid < HID

    idx_n = tl.program_id(0).to(idtype)
    idx_h = tl.program_id(1).to(idtype)
    idx_t = tl.program_id(2).to(idtype)

    kv_shift = idx_n.to(idtype) * stride_k_n + \
        idx_h.to(idtype) * stride_k_h + \
        idx_t.to(idtype) * stride_k_t + \
        idx_hid.to(idtype) * stride_k_hid

    # # load key
    key = tl.load(KEY + kv_shift, mask=mask_hid, other=0)
    val = tl.load(VAL + kv_shift, mask=mask_hid, other=0)

    stored_shift = idx_n * stride_ss_n + idx_h * stride_ss_h
    stored = tl.load(STORED_SINKS + stored_shift)

    kv_cshift = idx_n.to(idtype) * stride_sk_n + \
        idx_h.to(idtype) * stride_sk_h + \
        stored.to(idtype) * stride_sk_t + \
        idx_hid.to(idtype) * stride_sk_hid

    tl.store(SINK_K + kv_cshift, value=key.to(dtype), mask=mask_hid)
    tl.store(SINK_V + kv_cshift, value=val.to(dtype), mask=mask_hid)

    tl.store(
        SINK_POS + \
            idx_n.to(idtype) * stride_p_n + \
            idx_h.to(idtype) * stride_p_h + \
            stored.to(idtype) * stride_p_t,
        value=stored.to(idtype),
    )

    tl.store(
        SINK_MASK + \
            idx_n.to(idtype) * stride_m_n + \
            idx_h.to(idtype) * stride_m_h + \
            stored.to(idtype) * stride_m_t,
        value=0,
    )

    tl.store(STORED_SINKS + stored_shift, value=(stored + 1).to(idtype))


@triton.jit
def _update_positional_idx(
    POS,
    stride_p_n,
    stride_p_h,
    stride_p_t,
    idx_n,
    idx_h,
    u,
    l,
    segment_len,
    pos_ub,
    stored_tokens_i,
    start_idx_i,
    WINDOW_SIZE_CONST,
):

    u = min(u, l + stored_tokens_i)
    segment_len = min(segment_len, stored_tokens_i)

    pos = (tl.arange(0, WINDOW_SIZE_CONST) + (segment_len - start_idx_i)) % segment_len + \
        pos_ub - segment_len

    pos_idx = tl.arange(0, WINDOW_SIZE_CONST).to(tl.int64)
    tl.store(
        POS + \
            idx_n * stride_p_n + \
            idx_h * stride_p_h + \
            (l + pos_idx) * stride_p_t,
        value=pos
    )


@triton.jit
def _update_kv_cache(
    # input tensors
    KEY,
    VAL,
    stride_k_n,
    stride_k_h,
    stride_k_t,
    stride_k_hid,
    SCR,
    stride_s_n,
    stride_s_h,
    stride_s_t,
    CACHE_K,
    CACHE_V,
    stride_ck_n,
    stride_ck_h,
    stride_ck_t,
    stride_ck_hid,
    CACHE_S,
    stride_cs_n,
    stride_cs_h,
    stride_cs_t,
    MASK,
    stride_m_n,
    stride_m_h,
    stride_m_t,
    POS,
    stride_p_n,
    stride_p_h,
    stride_p_t,

    # tracker variables
    STORED_TOKENS,
    START_INDICES,
    stride_st_n,
    stride_st_h,
    stride_st_c,
    DO_CACHE,

    # input variables
    N,
    K,
    HID,
    NUM_SINK,
    WINDOW_SIZE,

    # kernel constants
    WINDOW_SIZE_CONST: tl.constexpr,
    CASCADES: tl.constexpr,
    BLOCK_HID: tl.constexpr,
):
    dtype = tl.float16
    idtype = tl.int64

    idx_hid = tl.arange(0, BLOCK_HID).to(idtype)
    mask_hid = idx_hid < HID

    cascades_idx = tl.arange(0, CASCADES).to(idtype)

    idx_n = tl.program_id(0).to(idtype)
    idx_h = tl.program_id(1).to(idtype)
    idx_t = tl.program_id(2).to(idtype)

    stored = tl.load(STORED_TOKENS + idx_n * CASCADES + cascades_idx)

    pos_ub = tl.sum(stored, axis=0)

    do_cache = tl.load(DO_CACHE + cascades_idx)
    add_to_cache = tl.sum(do_cache) * WINDOW_SIZE > pos_ub
    eager_add = tl.sum(do_cache) * WINDOW_SIZE == pos_ub
    if add_to_cache or eager_add:
        pos_ub = pos_ub + 1

    # LOAD KEY VALUE AND SCORE STATES
    kv_shift = idx_n.to(idtype) * stride_k_n + \
        idx_h.to(idtype) * stride_k_h + \
        idx_t.to(idtype) * stride_k_t + \
        idx_hid.to(idtype) * stride_k_hid

    key = tl.load(KEY + kv_shift, mask=mask_hid, other=0)
    value = tl.load(VAL + kv_shift, mask=mask_hid, other=0)

    score = tl.load(
        SCR + \
            idx_n.to(idtype) * stride_s_n + \
            idx_h.to(idtype) * stride_s_h + \
            idx_t.to(idtype) * stride_s_t,
    )

    do_break = False
    i = 0
    while i < CASCADES and not do_break:
        l = (i * WINDOW_SIZE).to(idtype)
        u = ((i + 1) * WINDOW_SIZE).to(idtype)
        segment_len = WINDOW_SIZE.to(idtype)

        # all N will be the same here, so there is no n dimension included
        do_cache_i = tl.load(DO_CACHE + i.to(idtype))

        stored_shift = idx_n.to(tl.int64) * stride_st_n + \
            idx_h.to(tl.int64) * stride_st_h + \
            i.to(tl.int64) * stride_st_c

        stored_tokens_i = tl.load(STORED_TOKENS + stored_shift)
        start_idx_i = tl.load(START_INDICES + stored_shift)

        if do_cache_i:
            # print("append first")
            if stored_tokens_i < segment_len:
                t = start_idx_i.to(idtype) + stored_tokens_i.to(idtype) + l.to(
                    idtype)

                kv_adds = idx_n.to(idtype) * stride_ck_n + \
                    idx_h.to(idtype) * stride_ck_h + \
                    t.to(idtype) * stride_ck_t + \
                    idx_hid.to(idtype) * stride_ck_hid

                # add in new values at the proper location, set mask
                tl.store(CACHE_K + kv_adds, value=key.to(dtype), mask=mask_hid)
                tl.store(CACHE_V + kv_adds,
                         value=value.to(dtype),
                         mask=mask_hid)

                tl.store(
                    CACHE_S + \
                         idx_n.to(idtype) * stride_s_n + \
                         idx_h.to(idtype) * stride_s_h + \
                         t.to(idtype) * stride_s_t,
                         value=score.to(dtype)
                )

                tl.store(
                    MASK + \
                         idx_n.to(idtype) * stride_m_n + \
                         idx_h.to(idtype) * stride_m_h + \
                         t.to(idtype) * stride_m_t,
                         value=0
                )

                # increment the stored tokens for this cascade
                tl.store(STORED_TOKENS + stored_shift,
                         value=(stored_tokens_i + 1).to(idtype))

                _update_positional_idx(
                    POS,
                    stride_p_n,
                    stride_p_h,
                    stride_p_t,
                    idx_n,
                    idx_h,
                    u,
                    l,
                    segment_len,
                    pos_ub,
                    stored_tokens_i + 1,
                    start_idx_i,
                    WINDOW_SIZE_CONST,
                )

                do_break = True

            else:
                # print("evict")
                t = start_idx_i.to(idtype) + l.to(idtype)

                # load the key value and score states which are going do be evicted
                kv_adds = idx_n.to(idtype) * stride_ck_n + \
                    idx_h.to(idtype) * stride_ck_h + \
                    t.to(idtype) * stride_ck_t + \
                    idx_hid.to(idtype) * stride_ck_hid

                # we need to evict
                # 1. find the oldest token (start point), remove it and
                #    set input_key_state at that location
                next_key = tl.load(CACHE_K + kv_adds, mask=mask_hid, other=0)
                next_value = tl.load(CACHE_V + kv_adds, mask=mask_hid, other=0)

                sc_shift = idx_n.to(idtype) * stride_cs_n + \
                    idx_h.to(idtype) * stride_cs_h + \
                    t.to(idtype) * stride_cs_t

                next_score = tl.load(CACHE_S + sc_shift)

                # store the new tokens in place of the evicted ones
                tl.store(CACHE_K + kv_adds, value=key.to(dtype), mask=mask_hid)
                tl.store(CACHE_V + kv_adds,
                         value=value.to(dtype),
                         mask=mask_hid)

                tl.store(CACHE_S + sc_shift, value=score.to(dtype))

                # set the evicted token variables for the next iteration
                key = next_key.to(dtype)
                value = next_value.to(dtype)
                score = next_score.to(dtype)

                # 2. rotate the start index.
                tl.store(
                    START_INDICES + \
                        idx_n.to(idtype) * stride_st_n + \
                        idx_h.to(idtype) * stride_st_h + \
                        i.to(idtype) * stride_st_c,
                     value=((start_idx_i + 1) % segment_len).to(idtype)
                )

                _update_positional_idx(
                    POS,
                    stride_p_n,
                    stride_p_h,
                    stride_p_t,
                    idx_n,
                    idx_h,
                    u,
                    l,
                    segment_len,
                    pos_ub,
                    stored_tokens_i,
                    (start_idx_i + 1) % segment_len,
                    WINDOW_SIZE_CONST,
                )
                pos_ub = pos_ub - segment_len

                i += 1
        else:
            if stored_tokens_i == 0:
                # print("eager add")
                # if we are not supposed to move the cache, but we were called
                # with states as an input. Then there are two possibilities:
                # 1. We are not supposed to do cache, but the length of this cache is zero.
                #    this may happen due to the do_cache input_values not lining up perfectly with powers of 2.
                #    In this case, we should add an element to the cache so it doesn't just get automatically evicted.

                t = start_idx_i.to(idtype) + stored_tokens_i.to(idtype) + l.to(
                    idtype)

                kv_adds = idx_n.to(idtype) * stride_ck_n + \
                    idx_h.to(idtype) * stride_ck_h + \
                    t.to(idtype) * stride_ck_t + \
                    idx_hid.to(idtype) * stride_ck_hid

                # add in new values at the proper location, set mask
                tl.store(CACHE_K + kv_adds, value=key.to(dtype), mask=mask_hid)
                tl.store(CACHE_V + kv_adds,
                         value=value.to(dtype),
                         mask=mask_hid)

                tl.store(
                    CACHE_S + \
                        idx_n.to(idtype) * stride_cs_n + \
                        idx_h.to(idtype) * stride_cs_h + \
                        t.to(idtype) * stride_cs_t,
                    value=score.to(dtype)
                )

                tl.store(
                    MASK + \
                        idx_n.to(idtype) * stride_m_n + \
                        idx_h.to(idtype) * stride_m_n + \
                        t.to(idtype) * stride_m_t,
                    value=0
                )

                # increment the stored tokens for this cascade
                tl.store(
                    STORED_TOKENS + \
                        idx_n.to(idtype) * stride_st_n + \
                        idx_h.to(idtype) * stride_st_h + \
                        i.to(idtype) * stride_st_c,
                    value=(stored_tokens_i + 1).to(idtype)
                )

                _update_positional_idx(
                    POS,
                    stride_p_n,
                    stride_p_h,
                    stride_p_t,
                    idx_n,
                    idx_h,
                    u,
                    l,
                    segment_len,
                    pos_ub,
                    stored_tokens_i + 1,
                    start_idx_i,
                    WINDOW_SIZE_CONST,
                )

                do_break = True

            else:
                # print("overwrite")
                # 2. Since we know this cache has something in it, and we are not to do caching,
                #    find the most recent thing in this cache, compare attention input_scores,
                #    and remove if needed.

                # not sure why all this typecasting is needed, but 0 - 1 evals to 2^32 - 1
                # and casting to a float fixes this.
                # t = ((start_idx_i - 1) % stored_tokens_i) + l

                t = (((start_idx_i.to(tl.float32) - 1) %
                      stored_tokens_i).to(idtype) + l.to(idtype)).to(idtype)

                cs_shift = idx_n.to(idtype) * stride_cs_n + \
                    idx_h.to(idtype) * stride_cs_h + \
                    t.to(idtype) * stride_cs_t

                old_score = tl.load(CACHE_S + cs_shift)

                if (old_score > score):
                    # old input_score is better, do nothing.
                    # increment cascade index for next iter
                    # break onstead of cotinue because this stops the cascade
                    # print("overwrite break")
                    do_break = True
                else:
                    # print("overwrite do")

                    kv_adds = idx_n.to(idtype) * stride_ck_n + \
                        idx_h.to(idtype) * stride_ck_h + \
                        t.to(idtype) * stride_ck_t + \
                        idx_hid.to(idtype) * stride_ck_hid

                    tl.store(CACHE_K + kv_adds,
                             value=key.to(dtype),
                             mask=mask_hid)
                    tl.store(CACHE_V + kv_adds,
                             value=value.to(dtype),
                             mask=mask_hid)

                    tl.store(CACHE_S + cs_shift, value=score)

                    _update_positional_idx(
                        POS,
                        stride_p_n,
                        stride_p_h,
                        stride_p_t,
                        idx_n,
                        idx_h,
                        u,
                        l,
                        segment_len,
                        pos_ub,
                        stored_tokens_i,
                        start_idx_i,
                        WINDOW_SIZE_CONST,
                    )

                do_break = True


class SinkCacheFunc(Function):

    @staticmethod
    def forward(
        ctx,
        k: Tensor,
        v: Tensor,
        s: Tensor,
        sink_k: Tensor,
        sink_v: Tensor,
        sink_mask: Tensor,
        sink_pos: Tensor,
        cache_k: Tensor,
        cache_v: Tensor,
        cache_s: Tensor,
        mask: Tensor,
        pos: Tensor,
        do_cache: Tensor,
        stored_tokens: Tensor,
        start_indices: Tensor,
        stored_sinks: Tensor,
        num_sink: int,
        window_size: int,
    ):
        assert k.ndim == 4
        assert v.ndim == 4
        N, H, K, HID = k.shape
        assert v.shape == (N, H, K, HID)
        assert k.stride() == v.stride()
        assert sink_k.stride() == sink_v.stride()
        assert cache_k.stride() == cache_v.stride()
        assert stored_tokens.stride() == start_indices.stride()

        device = k.device
        dtype = k.dtype

        BLOCK_HID = triton.next_power_of_2(HID)
        CASCADES = stored_tokens.size(-1)

        grid = (N, H, K)

        B, H, _, D = k.size()
        # print(f"{k.size()=} {k.stride()=}")
        # print(f"{v.size()=} {v.stride()=}")
        # print(f"{s.size()=} {s.stride()=}")
        # print(f"{s.size()=} {s.stride()=}")
        # print(f"{sink_k.size()=} {sink_k.stride()=}")
        # print(f"{sink_k.size()=} {sink_k.stride()=}")
        # print(f"{sink_v.size()=} {sink_v.stride()=}")
        # print(f"{sink_mask.size()=} {sink_mask.stride()=}")
        # print(f"{sink_pos.size()=} {sink_pos.stride()=}")
        # print(f"{cache_k.size()=} {cache_k.stride()=}")
        # print(f"{cache_v.size()=} {cache_v.stride()=}")
        # print(f"{cache_s.size()=} {cache_s.stride()=}")
        # print(f"{pos.size()=} {pos.stride()=}")
        # print(f"{mask.size()=} {mask.stride()=}")
        # print(f"{stored_tokens.size()=} {stored_tokens.stride()=}")
        # print(f"{start_indices.size()=} {start_indices.stride()=}")
        # print(f"{stored_sinks.size()=} {stored_sinks.stride()=}")
        # print(f"{HID=} {BLOCK_HID=}")

        _device = torch.cuda.current_device()
        torch.cuda.set_device(device)

        try:
            if stored_sinks[0, 0] < num_sink:

                _update_sink_cache[grid](k,
                                         v,
                                         *k.stride(),
                                         sink_k,
                                         sink_v,
                                         *sink_k.stride(),
                                         sink_mask,
                                         *sink_mask.stride(),
                                         sink_pos,
                                         *sink_pos.stride(),
                                         stored_sinks,
                                         *stored_sinks.stride(),
                                         N,
                                         K,
                                         HID,
                                         num_sink,
                                         window_size,
                                         BLOCK_HID,
                                         num_warps=1,
                                         num_stages=1)

            else:
                _update_kv_cache[grid](k,
                                       v,
                                       *k.stride(),
                                       s,
                                       *s.stride(),
                                       cache_k,
                                       cache_v,
                                       *cache_k.stride(),
                                       cache_s,
                                       *cache_s.stride(),
                                       mask,
                                       *mask.stride(),
                                       pos,
                                       *pos.stride(),
                                       stored_tokens,
                                       start_indices,
                                       *stored_tokens.stride(),
                                       do_cache,
                                       N,
                                       K,
                                       HID,
                                       num_sink,
                                       window_size,
                                       window_size,
                                       CASCADES,
                                       BLOCK_HID,
                                       num_warps=1,
                                       num_stages=1)

        except RuntimeError as ex:
            print(N, K, HID, BLOCK_HID,
                  num_sink, window_size, _device, k.shape, k.dtype,
                  k.is_contiguous(), k.device, k.shape, k.dtype,
                  v.is_contiguous(), v.device)
            raise Exception() from ex
        torch.cuda.set_device(_device)

        return stored_sinks, start_indices, stored_tokens

    @staticmethod
    def backward(ctx, grad_indices: Tensor, grad_values: Tensor):
        raise NotImplementedError("backward not implemented for sink cache")


def _sink_cache(
    k: Tensor,
    v: Tensor,
    s: Tensor,
    sink_k: Tensor,
    sink_v: Tensor,
    sink_mask: Tensor,
    sink_pos: Tensor,
    cache_k: Tensor,
    cache_v: Tensor,
    cache_s: Tensor,
    mask: Tensor,
    pos: Tensor,
    do_cache: Tensor,
    stored_tokens: Tensor,
    start_indices: Tensor,
    stored_sinks: Tensor,
    num_sink,
    window_size,
):
    N, H, K, HID = k.shape

    SinkCacheFunc.apply(
        k,
        v,
        s,
        sink_k,
        sink_v,
        sink_mask,
        sink_pos,
        cache_k,
        cache_v,
        cache_s,
        mask,
        pos,
        do_cache,
        stored_tokens,
        start_indices,
        stored_sinks,
        num_sink,
        window_size,
    )


def sink_cache(
    k: Tensor,
    v: Tensor,
    s: Tensor,
    sink_k: Tensor,
    sink_v: Tensor,
    sink_mask: Tensor,
    sink_pos: Tensor,
    cache_k: Tensor,
    cache_v: Tensor,
    cache_s: Tensor,
    mask: Tensor,
    pos: Tensor,
    do_cache: Tensor,
    stored_tokens: Tensor,
    start_indices: Tensor,
    stored_sinks: Tensor,
    num_sink,
    window_size,
    BENCHMARK: bool = False,
):
    if BENCHMARK:
        event_cache_start = torch.cuda.Event(enable_timing=True)
        event_cache_end = torch.cuda.Event(enable_timing=True)
        event_cache_start.record()

    _sink_cache(k,
                v,
                s,
                sink_k,
                sink_v,
                sink_mask,
                sink_pos,
                cache_k,
                cache_v,
                cache_s,
                mask,
                pos,
                do_cache,
                stored_tokens,
                start_indices,
                stored_sinks,
                num_sink=num_sink,
                window_size=window_size)

    if BENCHMARK:
        event_cache_end.record()

    if BENCHMARK:
        torch.cuda.synchronize()
        elapsed_cache = event_cache_start.elapsed_time(event_cache_end)

        print(elapsed_cache)

    return k, v


class SinkCache:
    pass


class CascadingSinkCacheTriton(SinkCache):

    def __init__(
        self,
        window_length: int = 8,
        num_sink_tokens: int = 4,
        max_batch_size: int = 1,
        heads: int = 16,
        dim: int = 128,
        max_seq_len: int = 32,
        device: torch.device = "cpu",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.heads = heads
        self.dim = dim
        self.device = device
        self.dtype = dtype

        self.key_cache: torch.Tensor
        self.value_cache: torch.Tensor
        self.score_cache: torch.Tensor
        self.mask: torch.Tensor
        self.sink_keys: torch.Tensor
        self.sink_values: torch.Tensor
        self.sink_mask: torch.Tensor

        self.bh = self.max_batch_size * self.heads

        self.cascades = max_seq_len // window_length
        self.do_cache = torch.tensor([True for _ in range(self.cascades)],
                                     device=device,
                                     dtype=torch.bool)
        self.do_cache_every_n = torch.tensor(
            [2**i for i in range(self.cascades)],
            device=device,
            dtype=torch.long)
        # self.do_cache_every_n = [i + 1 for i in range(self.cascades)]

        self.beta = np.exp(-np.log(100) / window_length)
        self.num_sink_tokens = num_sink_tokens

        self.window_length = window_length
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen

        # per layer, not per cascade
        self.stored_tokens = torch.zeros(self.max_batch_size,
                                         self.heads,
                                         self.cascades,
                                         device=device,
                                         dtype=torch.long)

        self.stored_sinks = torch.zeros(self.max_batch_size,
                                        self.heads,
                                        device=device,
                                        dtype=torch.long)

        # each cascade will have start indices which are considered the beginning of
        # the cascade cache to avoid excessive concatenation.
        self.start_indices = torch.zeros(self.max_batch_size,
                                         self.heads,
                                         self.cascades,
                                         device=device,
                                         dtype=torch.long)

        # index for positional encodings, this will be modified on
        # each return in order to grab the correct positional encoding indices.
        self.pos = torch.zeros(max_seq_len, device=device,
                               dtype=torch.long).reshape(1, -1).repeat(
                                   self.max_batch_size, self.heads, 1)

        self.sink_pos = torch.zeros(self.num_sink_tokens,
                                    device=device,
                                    dtype=torch.long).reshape(1, -1).repeat(
                                        self.max_batch_size, self.heads, 1)

        print("INIT NLOGN TRITON VERSION")
        self.init_static_cache()

    def init_static_cache(self):
        B, H, S, D = self.max_batch_size, self.heads, self.max_seq_len, self.dim
        nsink, dev, dtp = self.num_sink_tokens, self.device, self.dtype

        blank = torch.zeros(B, H, S, D, device=dev, dtype=dtp)
        blank_scores = torch.zeros(B,
                                   H,
                                   self.max_seq_len,
                                   device=dev,
                                   dtype=dtp)
        blank_sinks = torch.zeros(B, H, nsink, D, device=dev, dtype=dtp)

        self.key_cache = blank.clone()
        self.value_cache = blank.clone()
        self.score_cache = blank_scores.clone()
        self.sink_keys = blank_sinks.clone()
        self.sink_values = blank_sinks.clone()
        self.mask = torch.empty(B, H, S, device=dev, dtype=dtp)
        self.sink_mask = torch.empty(B,
                                     H,
                                     self.num_sink_tokens,
                                     device=dev,
                                     dtype=dtp)

        self.mask.fill_(torch.finfo(dtp).min)
        self.sink_mask.fill_(torch.finfo(dtp).min)

    def set_cache_bools(self):
        self.do_cache = ((self._seen_tokens - 1 - self.num_sink_tokens) %
                         self.do_cache_every_n) == 0

    def get_seq_length(self,
                       layer_idx: Optional[int] = 0,
                       cascade_idx: Optional[int] = -1) -> int:
        raise NotImplementedError()

    def get_max_length(self) -> Optional[int]:
        return self.max_seq_len

    def update_attention_scores(
        self,
        attention_scores: torch.Tensor,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.score_cache = self.beta * self.score_cache + (
            1 - self.beta) * attention_scores.mean(dim=1,
                                                   keepdim=True).squeeze(2)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # init new scores for this incoming kv which has none so far
        score_states = torch.zeros(self.max_batch_size,
                                   self.heads,
                                   1,
                                   device=key_states.device,
                                   dtype=key_states.dtype)

        self._seen_tokens += key_states.shape[-2]
        self.set_cache_bools()

        sink_cache(
            key_states,
            value_states,
            score_states,
            self.sink_keys,
            self.sink_values,
            self.sink_mask,
            self.sink_pos,
            self.key_cache,
            self.value_cache,
            self.score_cache,
            self.mask,
            self.pos,
            self.do_cache,
            self.stored_tokens,
            self.start_indices,
            self.stored_sinks,
            self.num_sink_tokens,
            self.window_length,
        )

        pos_shift = self.num_sink_tokens if self._seen_tokens > self.num_sink_tokens else 0

        return (
            self.sink_keys,
            self.sink_values,
            self.sink_pos[:1, 0],
            self.sink_mask,
            self.key_cache,
            self.value_cache,
            self.pos[:1, 0] + pos_shift,
            self.mask,
        )


def update_segment_pos(cascade_idx, pos, pos_ub, start_indices, stored_tokens,
                       l, u, seg_len, tmp_arange):
    u = torch.amin(
        torch.cat((u, l + torch.gather(stored_tokens, 0, cascade_idx))))

    seg_len = torch.amin(
        torch.cat((torch.gather(stored_tokens, 0,
                                cascade_idx).unsqueeze(0), seg_len)))
    start_idx = torch.gather(start_indices, 0, cascade_idx)

    tmp = (tmp_arange + (seg_len - start_idx)) % seg_len + (pos_ub - seg_len)
    pos.scatter_(1, l + tmp_arange.unsqueeze(0), tmp.unsqueeze(0))
    # pos[0, l:u] = (self.tmp_arange + (seg_len - start_idx)) % seg_len

    pos_ub.sub_(seg_len)
    return cascade_idx


def append_to_cache(cascade_idx, input_key_states, input_value_states,
                    input_score_states, keys, values, scores, mask, cache_idx,
                    score_idx, mask_idx, start_indices, stored_tokens, l, u,
                    seg_len, pos, pos_ub, tmp_arange):

    start_idx = torch.gather(start_indices, 0, cascade_idx)
    # we have empty room in this cache, so we need to shift the index
    # forward by the number of tokens already stored.
    stored = torch.gather(stored_tokens, 0, cascade_idx)
    s = start_idx + l + stored

    # we do not need to evict, find the end point and insert token
    # since this cache is not full, the insert point will be start + stored_tokens
    cache_idx_local = cache_idx * s
    score_idx_local = score_idx * s
    mask_idx_local = mask_idx * s

    keys.scatter_(2, cache_idx_local, input_key_states)
    values.scatter_(2, cache_idx_local, input_value_states)
    scores.scatter_(0, score_idx_local, input_score_states)
    mask.scatter_(3, mask_idx_local, 0)

    stored_tokens.add_(F.one_hot(cascade_idx, stored_tokens.size(0)))

    _ = update_segment_pos(cascade_idx, pos, pos_ub, start_indices,
                           stored_tokens, l, u, seg_len, tmp_arange)

    # move along cascade idx for the next iteration
    cascade_idx.add_(1)

    return cascade_idx


def evict_from_cache(cascade_idx, input_key_states, input_value_states,
                     input_score_states, keys, values, scores, start_indices,
                     cache_idx, score_idx, l, u, segment_len, pos, pos_ub,
                     stored_tokens, tmp_arange):

    start_idx = torch.gather(start_indices, 0, cascade_idx)
    s = start_idx + l

    # we need to evict
    # 1. find the oldest token (start point), remove it and
    #    set input_key_state at that location

    cache_idx_local = cache_idx * s
    score_idx_local = score_idx * s

    next_input_key_state = torch.gather(keys, 2, cache_idx_local).clone()
    next_input_value_state = torch.gather(values, 2, cache_idx_local).clone()
    next_input_score_state = torch.gather(scores, 0, score_idx_local).clone()

    keys.scatter_(2, cache_idx_local, input_key_states)
    values.scatter_(2, cache_idx_local, input_value_states)
    scores.scatter_(0, score_idx_local, input_score_states)

    # 2. rotate the start index.
    # new_start_idx = (start_idx + 1) % segment_len (vectorized version of this)
    # start_indices = (start_indices + F.one_hot(
    #     cascade_idx, start_indices.size(0))) % segment_len
    start_indices.add_(F.one_hot(cascade_idx, start_indices.size(0)))
    start_indices.fmod_(segment_len)

    _ = update_segment_pos(cascade_idx, pos, pos_ub, start_indices,
                           stored_tokens, l, u, segment_len, tmp_arange)

    # move along cascade idx for the next iteration
    cascade_idx.add_(1)

    # mask remains unchanged for this operation.
    return (cascade_idx, next_input_key_state, next_input_value_state,
            next_input_score_state)


def overwrite_cache(cascade_idx, input_key_states, input_value_states,
                    input_score_states, keys, values, scores, start_indices,
                    cache_idx, score_idx, l, u, seg_len, pos, pos_ub,
                    stored_tokens, tmp_arange):
    # print(
    #     f"{stored_tokens.size()=} {cascade_idx.size()=} {start_indices.size()=}"
    # )
    # print(f"{stored_tokens=} {cascade_idx=} {start_indices=}")
    start_idx = torch.gather(start_indices, 0, cascade_idx)
    # print("hit")
    stored = torch.gather(stored_tokens, 0, cascade_idx)
    # print("hit2")
    # print(f"{start_idx=} {stored=}")

    # s = start_idx + l
    # print(f"{start_idx - 1=} {(start_idx - 1) % stored=}")
    s = ((start_idx - 1) % stored) + l
    s = torch.amax(torch.cat((s, torch.zeros_like(s))))

    # print(f"{s=}")

    cache_idx_local = cache_idx * s
    score_idx_local = score_idx * s

    keys.scatter_(2, cache_idx_local, input_key_states)
    values.scatter_(2, cache_idx_local, input_value_states)
    scores.scatter_(0, score_idx_local, input_score_states)

    _ = update_segment_pos(cascade_idx, pos, pos_ub, start_indices,
                           stored_tokens, l, u, seg_len, tmp_arange)

    # move along cascade idx for the next iteration
    cascade_idx.add_(1)
    return cascade_idx


def add_sinks(input_key_states, input_value_states, sink_keys, sink_values,
              sink_pos, sink_mask, stored_sinks, cache_idx, pos_idx, mask_idx):

    cache_idx_local = cache_idx * stored_sinks
    pos_idx_local = pos_idx * stored_sinks
    mask_idx_local = mask_idx * stored_sinks

    sink_keys.scatter_(2, cache_idx_local, input_key_states)
    sink_values.scatter_(2, cache_idx_local, input_value_states)
    sink_pos.scatter_(1, pos_idx_local, stored_sinks.expand_as(pos_idx_local))
    sink_mask.scatter_(3, mask_idx_local, 0)

    return input_key_states


class CascadingSinkCache(SinkCache):

    def __init__(
        self,
        window_length: int = 8,
        num_sink_tokens: int = 4,
        max_batch_size: int = 1,
        heads: int = 16,
        dim: int = 128,
        n_layers: int = 1,  # need to know in advance for static cache
        max_seq_len: int = 32,
        device: torch.device = "cpu",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.heads = heads
        self.dim = dim
        self.n_layers = n_layers
        self.device = device
        self.dtype = dtype

        self.key_cache: torch.Tensor
        self.score_cache: torch.Tensor
        self.value_cache: torch.Tensor
        self.sink_keys: torch.Tensor
        self.sink_values: torch.Tensor

        self.cascades = max_seq_len // window_length
        self.do_cache_cpu = torch.tensor([True for _ in range(self.cascades)],
                                         dtype=torch.bool,
                                         requires_grad=False)
        self.do_cache = torch.tensor([True for _ in range(self.cascades)],
                                     device=device,
                                     dtype=torch.bool,
                                     requires_grad=False)

        print(f"{self.cascades=} {self.do_cache=}")
        self.do_cache_every_n = torch.tensor(
            [2**i for i in range(self.cascades)],
            dtype=torch.long,
            requires_grad=False,
        )

        self.beta = np.exp(-np.log(100) / window_length)
        self.num_sink_tokens = num_sink_tokens

        self.window_length = window_length
        self._seen_tokens = torch.tensor(
            0, dtype=torch.long, requires_grad=False
        )  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.stored_sinks = 0

        self.stored_tokens = torch.tensor([0 for _ in range(self.cascades)],
                                          device=device,
                                          dtype=torch.long,
                                          requires_grad=False)

        # each cascade will have start indices which are considered the beginning of
        # the cascade cache to avoid excessive concatenation.
        self.start_indices = torch.tensor([0 for _ in range(self.cascades)],
                                          device=device,
                                          dtype=torch.long,
                                          requires_grad=False)

        # index for positional encodings, this will be modified on
        # each return in order to grab the correct positional encoding indices.
        self.pos = torch.zeros(max_seq_len,
                               device=device,
                               dtype=torch.long,
                               requires_grad=False).view(1, -1)

        self.pos_ub = torch.tensor(max_seq_len,
                                   device=device,
                                   dtype=torch.long,
                                   requires_grad=False)

        self.tmp_arange = torch.arange(self.window_length,
                                       device=device,
                                       dtype=torch.long,
                                       requires_grad=False)
        self.sink_pos = torch.zeros(self.num_sink_tokens,
                                    device=device,
                                    dtype=torch.long,
                                    requires_grad=False).view(1, -1)
        print("INIT NLOGN FAST COMPILED VERSION")

        self.init_static_cache()

    def init_static_cache(self):
        B, H, S, D = self.max_batch_size, self.heads, self.max_seq_len, self.dim
        nsink, dev, dtp = self.num_sink_tokens, self.device, self.dtype

        blank = torch.zeros(B,
                            H,
                            S,
                            D,
                            device=dev,
                            dtype=dtp,
                            requires_grad=False)
        blank_scores = torch.zeros(self.max_seq_len,
                                   device=dev,
                                   dtype=dtp,
                                   requires_grad=False)
        blank_sinks = torch.zeros(B,
                                  H,
                                  nsink,
                                  D,
                                  device=dev,
                                  dtype=dtp,
                                  requires_grad=False)

        self.key_cache = blank.clone()
        self.value_cache = blank.clone()
        self.score_cache = blank_scores.clone()
        self.sink_keys = blank_sinks.clone()
        self.sink_values = blank_sinks.clone()

        self.scalar = torch.ones(1,
                                 device=self.device,
                                 dtype=torch.long,
                                 requires_grad=False)

        self.cascade_idx = torch.tensor(0,
                                        device=self.device,
                                        dtype=torch.long,
                                        requires_grad=False)

        self.cascade_bounds = []
        for i in range(self.cascades):
            self.cascade_bounds.append(
                (self.scalar * self.window_length * i,
                 self.scalar * self.window_length * (i + 1),
                 self.scalar * self.window_length, self.window_length))

        self.cache_idx = torch.ones(self.max_batch_size,
                                    self.heads,
                                    1,
                                    self.dim,
                                    device=self.device,
                                    dtype=torch.long,
                                    requires_grad=False)

        self.pos_idx = torch.ones(1,
                                  1,
                                  device=self.device,
                                  dtype=torch.long,
                                  requires_grad=False)

        self.mask_idx = torch.ones(1,
                                   1,
                                   1,
                                   1,
                                   device=self.device,
                                   dtype=torch.long,
                                   requires_grad=False)

        self.score_idx = torch.ones(1,
                                    device=self.device,
                                    dtype=torch.long,
                                    requires_grad=False)

        self.sink_pos_idx = torch.ones(1,
                                       self.num_sink_tokens,
                                       device=self.device,
                                       dtype=torch.long,
                                       requires_grad=False)

        self.sink_mask = torch.full((1, 1, 1, self.num_sink_tokens),
                                    torch.finfo(self.dtype).min,
                                    device=self.device,
                                    dtype=self.dtype,
                                    requires_grad=False)

        self.mask = torch.full((1, 1, 1, self.max_seq_len),
                               torch.finfo(self.dtype).min,
                               device=self.device,
                               dtype=self.dtype,
                               requires_grad=False)

        self.score_states = torch.zeros(1,
                                        device=self.device,
                                        dtype=self.dtype,
                                        requires_grad=False)

    def set_cache_bools(self):
        # minus one because seen tokens is incremented before tokens are really added. Therefore we need to subtract that one
        for i, _ in enumerate(self.do_cache_cpu):
            if (self._seen_tokens - 1 -
                    self.num_sink_tokens) % self.do_cache_every_n[i] == 0:
                self.do_cache_cpu[i] = True
                continue

            self.do_cache_cpu[i] = False
        self.do_cache.copy_(self.do_cache_cpu)

    def get_cascade_bounds(self, i):
        return self.cascade_bounds[i]

    def get_seq_length(self,
                       layer_idx: Optional[int] = 0,
                       cascade_idx: Optional[int] = -1) -> int:
        return sum([v for v in self.stored_tokens])

    def get_max_length(self) -> Optional[int]:
        return self.max_seq_len

    def update_attention_scores(self, scores, layer_idx) -> None:
        self.score_cache = self.beta * self.score_cache + (1 -
                                                           self.beta) * scores

    def warn(self, args):
        warnings.warn(
            "the cascading cache is full, evicted context from the last cascade will be dropped"
        )
        return args

    def add_keys(self, input_key_states, input_value_states):

        # in order to create the positional embeddings in teh same loop as
        # the main logic, we must know if we are going to add anything to the
        # cache or not which will change what happens to the positional embeddings.
        stored_tokens_cpu = self.stored_tokens.cpu()

        tmp_pos_ub = stored_tokens_cpu.sum()
        add_to_cache = self.do_cache_cpu.sum(
        ) * self.window_length > tmp_pos_ub
        eager_add = self.do_cache_cpu.sum() * self.window_length == tmp_pos_ub
        if add_to_cache or eager_add:
            tmp_pos_ub += 1

        self.pos_ub.fill_(tmp_pos_ub.item())

        input_score_states = self.score_states

        self.cascade_idx.zero_()
        for i in range(self.cascades):
            l, u, segment_len, segment_len_cpu = self.get_cascade_bounds(i)

            if self.do_cache_cpu[i]:
                if stored_tokens_cpu[i] < segment_len_cpu:

                    _ = self.append_to_cache(
                        self.cascade_idx, input_key_states, input_value_states,
                        input_score_states, self.key_cache, self.value_cache,
                        self.score_cache, self.mask, self.cache_idx,
                        self.score_idx, self.mask_idx, self.start_indices,
                        self.stored_tokens, l, u, segment_len, self.pos,
                        self.pos_ub, self.tmp_arange)
                    break
                else:
                    (_, input_key_states, input_value_states,
                     input_score_states) = self.evict_from_cache(
                         self.cascade_idx, input_key_states,
                         input_value_states, input_score_states,
                         self.key_cache, self.value_cache, self.score_cache,
                         self.start_indices, self.cache_idx, self.score_idx, l,
                         u, segment_len, self.pos, self.pos_ub,
                         self.stored_tokens, self.tmp_arange)

                    if i + 1 > (self.cascades - 1):
                        break
            else:
                if stored_tokens_cpu[i] == 0:
                    # if we are not supposed to move the cache, but we were called
                    # with states as an input. Then there are two possibilities:
                    # 1. We are not supposed to do cache, but the length of this cache is zero.
                    #    this may happen due to the do_cache input_values not lining up perfectly with powers of 2.
                    #    In this case, we should add an element to the cache so it doesn't just get automatically evicted.
                    _ = self.append_to_cache(
                        self.cascade_idx, input_key_states, input_value_states,
                        input_score_states, self.key_cache, self.value_cache,
                        self.score_cache, self.mask, self.cache_idx,
                        self.score_idx, self.mask_idx, self.start_indices,
                        self.stored_tokens, l, u, segment_len, self.pos,
                        self.pos_ub, self.tmp_arange)
                    break
                else:
                    # 2. Since we know this cache has something in it, and we are not to do caching,
                    #    find the oldest thing in this cache, compare attention input_scores,
                    #    and remove if needed.

                    s = self.start_indices[i].add(l)

                    # TODO: make this use cond or not use gpu access
                    score_idx = self.score_idx * s
                    old_input_score = torch.gather(self.score_cache, 0,
                                                   score_idx)
                    if old_input_score > input_score_states:
                        # old input_score is better, do nothing.
                        # increment cascade index for next iter
                        # break onstead of cotinue because this stops the cascade
                        break

                    _ = self.overwrite_cache(
                        self.cascade_idx, input_key_states, input_value_states,
                        input_score_states, self.key_cache, self.value_cache,
                        self.score_cache, self.start_indices, self.cache_idx,
                        self.score_idx, l, u, segment_len, self.pos,
                        self.pos_ub, self.stored_tokens, self.tmp_arange)

                    break

        return

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[Dict[str, Any]] = None,
        create_mask: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        self._seen_tokens += key_states.shape[-2]
        self.set_cache_bools()

        if self.stored_sinks < self.num_sink_tokens:
            _ = self.add_sinks(key_states, value_states, self.sink_keys,
                               self.sink_values, self.sink_pos, self.sink_mask,
                               self.stored_sinks * self.scalar, self.cache_idx,
                               self.pos_idx, self.mask_idx)

            self.stored_sinks += 1
            pos = self.pos
        else:
            self.add_keys(key_states, value_states)

            # self.pos[:, self.stored_tokens.sum():] = -self.num_sink_tokens
            pos = self.pos.add(self.num_sink_tokens)
        # print(f"\n\n\nbefore")
        # print(
        #     f"{self.sink_keys=}\n{self.sink_values=}\n{self.sink_pos=}\n{self.sink_mask=}"
        # )

        # print(f"\n\n\nafter")
        # print(
        #     f"{self.sink_keys=}\n{self.sink_values=}\n{self.sink_pos=}\n{self.sink_mask=}"
        # )

        return (
            self.sink_keys,
            self.sink_values,
            self.sink_pos,
            self.sink_mask,
            self.key_cache,
            self.value_cache,
            pos,
            self.mask,
        )


def test_batch():
    cache = CascadingSinkCacheTriton(window_length=WIND,
                                     num_sink_tokens=NSINK,
                                     max_batch_size=N,
                                     heads=HEAD,
                                     dim=HID,
                                     n_layers=1,
                                     max_seq_len=MAX_SEQ,
                                     device=DEVICE,
                                     dtype=DTYPE)

    with torch.no_grad():
        slow_times, fast_times = [], []
        for i in range(3000):
            print(f"{'='*50}")
            k, v = torch.ones(N, HEAD, 1, HID, device=DEVICE, dtype=DTYPE) * (
                i + 1), torch.ones(N, HEAD, 1, HID, device=DEVICE,
                                   dtype=DTYPE) * (i + 1)
            # k, v = torch.randn(1, 1, 1, HID, device=DEVICE,
            #                    dtype=DTYPE).repeat(N, HEAD, 1, 1), torch.randn(
            #                        1, 1, 1, HID, device=DEVICE,
            #                        dtype=DTYPE).repeat(N, HEAD, 1, 1)

            # print(f"\n\n{'='*100}\n\n")
            # ============================================================================================
            tic = time.perf_counter()
            k, v, pos, sink_mask, k_nosink, v_nosink, pos_nosink, mask = cache.update(
                k, v)
            fast_times.append(time.perf_counter() - tic)

            # print(f"{k=}\n{v=}\n{pos=}\n{sink_mask=}")
            k, v = torch.cat((k, k_nosink), dim=-2), torch.cat((v, v_nosink),
                                                               dim=-2)

            k0, k1 = k[0], k[1]
            print(f"{k.size()=}")
            for i in range(k0.size(0)):
                print(f"head: {i}")
                print(f"0: {k0[i, :, 0]=}")
                print(f"1: {k1[i, :, 0]=}")
                diff = (k0[i, :, 0] - k1[i, :, 0]).abs()
                print(
                    f"stored tokens: {cache.stored_tokens[0]=} {cache.stored_tokens[1]=}"
                )
                print(
                    f"start indices: {cache.start_indices[0]=} {cache.start_indices[1]=}"
                )
                print(f"{diff=}")

            equal = k0 == k1
            if not torch.all(equal):
                print(f"equal: ", k0 == k1)
                print(f"{(k0 - k1).abs().amax()=}")
                exit()
            pos = torch.cat((pos, pos_nosink), dim=-1).squeeze(0)

            # print(f"{k[0, 0, :,  0]=}")
            mask = torch.cat((sink_mask, mask), dim=-1)

            # n = (mask == 0).sum()
            # k, v = k[:, :, :n], v[:, :, :n]
            # argsort = torch.argsort(pos[:n])
            # # print(f"{mask=} {k=} {v=}")

            # print(f"before sort: \n{k.reshape(-1)=}\n{pos.reshape(-1)=}")

            # k, v = k[:, :, argsort], v[:, :, argsort]

            # print(f"after sort: {k.view(-1)}")
    fast_times = fast_times[100:]
    fast_times = sum(fast_times) / len(fast_times)
    print(f"{fast_times=}")


def test_against_reference():
    cache_slow = CascadingSinkCache(window_length=WIND,
                                    num_sink_tokens=NSINK,
                                    max_batch_size=1,
                                    heads=HEAD,
                                    dim=HID,
                                    n_layers=1,
                                    max_seq_len=MAX_SEQ,
                                    device=DEVICE,
                                    dtype=DTYPE)
    cache_slow.add_sinks = add_sinks
    cache_slow.append_to_cache = append_to_cache
    cache_slow.evict_from_cache = evict_from_cache
    cache_slow.overwrite_cache = overwrite_cache

    cache = CascadingSinkCacheTriton(window_length=WIND,
                                     num_sink_tokens=NSINK,
                                     max_batch_size=N,
                                     heads=HEAD,
                                     dim=HID,
                                     max_seq_len=MAX_SEQ,
                                     device=DEVICE,
                                     dtype=DTYPE)

    with torch.no_grad():
        slow_times, fast_times = [], []
        for i in range(6000):
            print(f"{'='*50}")
            k, v = torch.ones(N, HEAD, 1, HID, device=DEVICE, dtype=DTYPE) * (
                i + 1), torch.ones(N, HEAD, 1, HID, device=DEVICE,
                                   dtype=DTYPE) * (i + 1)
            # k, v = torch.randn(1, HEAD, 1, HID, device=DEVICE,
            #                    dtype=DTYPE), torch.randn(1,
            #                                              HEAD,
            #                                              1,
            #                                              HID,
            #                                              device=DEVICE,
            #                                              dtype=DTYPE)

            # print(f"\n\n{'='*100}\n\n")
            tic = time.perf_counter()
            (
                k_slow,
                v_slow,
                pos_slow,
                sink_mask_slow,
                k_nosink_slow,
                v_nosink_slow,
                pos_nosink_slow,
                mask_slow,
            ) = cache_slow.update(k[:1].clone(), v[:1].clone())

            slow_times.append(time.perf_counter() - tic)

            # print(f"{k=}\n{v=}\n{pos=}\n{sink_mask=}")
            k_slow, v_slow = torch.cat((k_slow, k_nosink_slow),
                                       dim=-2), torch.cat(
                                           (v_slow, v_nosink_slow), dim=-2)
            pos_slow = torch.cat((pos_slow, pos_nosink_slow),
                                 dim=-1).squeeze(0)

            # print(f"{k[0, 0, :,  0]=}")
            mask = torch.cat((sink_mask_slow, mask_slow), dim=-1)

            n = (mask == 0).sum()
            k_slow, v_slow = k_slow[:, :, :n], v_slow[:, :, :n]
            argsort = torch.argsort(pos_slow[:n])
            # print(
            #     f"before sort slow: \n{k_slow.reshape(-1)=}\n{pos_slow.reshape(-1)=}"
            # )

            # print(f"{k_nocomp[0, 0, :, 0]=}")
            k_slow, v_slow = k_slow[:, :, argsort], v_slow[:, :, argsort]

            # ============================================================================================
            tic = time.perf_counter()
            k, v, pos, sink_mask, k_nosink, v_nosink, pos_nosink, mask = cache.update(
                k, v)
            fast_times.append(time.perf_counter() - tic)

            idx = 0
            pos, mask, sink_mask, pos_nosink = pos[idx, 0], mask[
                idx, 0], sink_mask[idx, 0], pos_nosink[idx, 0]

            # print(f"{k.size()=}")
            # print(f"{k=}")
            # print(f"{k[idx:idx+1].view(-1)=}\n{pos=}\n{sink_mask=}")
            k, v = torch.cat((k, k_nosink), dim=-2), torch.cat((v, v_nosink),
                                                               dim=-2)
            pos = torch.cat((pos, pos_nosink), dim=-1)

            # print(f"{k[0, 0, :,  0]=}")
            mask = torch.cat((sink_mask, mask), dim=-1)

            n = (mask == 0).sum()
            k, v = k[idx:idx + 1, :, :n], v[idx:idx + 1, :, :n]
            argsort = torch.argsort(pos[:n])
            # print(
            #     f"fast: {mask=}\n{k[:, :].reshape(-1)=}\n{v[:, :].reshape(-1)=}"
            # )

            # print(f"before sort: \n{k.reshape(-1)=}\n{pos.reshape(-1)=}")

            k, v = k[:, :, argsort], v[:, :, argsort]

            # print(f"after sort: {k.view(-1)}")
            if not k_slow.size() == k.size():
                print(f"{k_slow.size()=} {k.size()=}")
                print(f"sizes not equal...\n{k_slow=} {k=}")
                exit()

            diff = (k_slow - k).abs().amax()
            print(f"k diff: {diff=} {i=}")
            if diff > 1e-6:
                print(
                    f"{k_slow.view(-1)=}\n{k.view(-1)=}\n{pos.view(-1)=}\n{(k - k_slow).abs().view(-1)=}"
                )
                exit("too big")

    slow_times = slow_times[100:]
    fast_times = fast_times[100:]
    slow_times = sum(slow_times) / len(slow_times)
    fast_times = sum(fast_times) / len(fast_times)
    print(f"{slow_times=} {fast_times=}")


if __name__ == '__main__':
    N = 128
    HID = 128
    NSINK = 4
    WIND = 32
    HEAD = 16
    MAX_SEQ = 128
    DEVICE = "cuda:0"
    DTYPE = torch.float16

    # k = torch.arange(27 * 3).reshape(3, 3, 3, 3).contiguous().cuda()
    # print(f"{k=}")
    # loaded = TestFunc.apply(k)
    # print(f"{loaded=}")

    test_against_reference()
    # test_batch()
