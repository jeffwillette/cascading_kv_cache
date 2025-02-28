from cascade.models.cascading_cache import SinkAttentionNaive, CascadingKVCache, CascadingKVCacheSlow, add_sinks, append_to_cache, evict_from_cache, overwrite_cache
import numpy as np
import unittest
import triton
import json
import gc
import time

from argparse import Namespace
from cascade.main.llama_eval import load_model
import torch
import time
from cascade.models.cascade_attention import sample_monkeypatch
from cascade.models.flash_attention import attention
from cascade.main.llama_eval import MODELS
import transformers
from transformers import pipeline
from minference import MInference, get_support_models


def get_cache(model, dtype=torch.float):
    mdl = model.model
    window = mdl.config._window // mdl.config._cascades
    max_seq_len = mdl.config._window

    return CascadingKVCache(
        window,
        num_sink_tokens=mdl.config._sinks,
        max_batch_size=mdl.config._batch_size,
        heads=mdl.config.num_key_value_heads // mdl.config.world_size,
        dim=mdl.config.hidden_size // mdl.config.num_attention_heads,
        max_seq_len=max_seq_len,
        dtype=dtype,  # use fp32 for testing so we can be sure numerical errors aren't from precision
        device=mdl.embed_tokens.weight.device,
        cascade_func=mdl.config._cascade_func,
        head_reduction=mdl.config._head_reduction,
        layers=len(mdl.layers),
        eager_fill=False,
    )


class TestTokenizer(unittest.TestCase):
    def test_tokenizer_chat_prompts(self):
        messages = [
            {"role": "system", "content": "You are a helpful chat bot."},
            {"role": "user", "content": "Who are you?"},
        ]
        for model in ["llama3.1-8b-instruct", "qwen2-7b-instruct"]:
            model_id = MODELS[model]
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
            out = tokenizer.apply_chat_template(messages, tokenize=True)
            print(out)
            self.assertIsNotNone(out)


class TestCascadeAttention(unittest.TestCase):
    def setUp(self):
        print('\n', unittest.TestCase.id(self))
        args = Namespace(
            batch_size=1,
            sinks=64,
            cascades=1,
            window=2048,
            world_size=1,
            cascade_func="pow2",
            head_reduction="mean",
            method="sink",
            use_fp32=True,   # use fp 32 for testing
            cascade_stride=128,
            job="test",
            lora_r=0,
            homogeneous_heads=True,
            do_og_pos=False,
        )
        self.args = args
        torch.set_grad_enabled(False)

    def tearDown(self):
        torch.set_grad_enabled(True)

    def test_cascade_attention_iterative_generate(self):
        args = self.args
        for model in ["llama3.1-8b", "qwen2-7b-instruct"]:
            args.model = model
            model, tokenizer, device = load_model(args)
            model = sample_monkeypatch(model)
            past_key_values = get_cache(model)
            past_key_values.force_eager = True

            stride = args.cascade_stride
            inputs = torch.randint(0, 1024, size=(args.batch_size, 2048), device=device)

            # test a manual iterative generation
            for i in range(0, inputs.size(1), stride):
                inp = inputs[:, i: i + args.cascade_stride]
                output = model(inp, past_key_values=past_key_values, use_cache=True)
                past_key_values = output.past_key_values
                del output

            past_key_values.reset(verbose=True)

            # test the generate method monkeypatch
            text = "Are you conscious? " * 300
            inputs = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

            output = model.generate(
                input_ids=inputs,
                max_new_tokens=128,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                use_cache=True,
                past_key_values=past_key_values,
            )[0]

            _ = tokenizer.decode(output[inputs.size(1):])

            del model
            del past_key_values
            gc.collect()
            torch.cuda.empty_cache()

    def test_cascade_attention_with_cache_iterative_equals_dense(self):
        """
        with a seqeuence length which gits inside the sliding window, it should
        be equivalent to dense attention.
        """
        for model in ["llama3.1-8b", "qwen2-7b-instruct"]:
            args = self.args
            args.cascades = 1
            args.model = model

            model, tokenizer, device = load_model(args)
            model = sample_monkeypatch(model)
            past_key_values = get_cache(model)
            past_key_values.force_eager = True

            stride = args.cascade_stride
            inputs = torch.randint(
                0, 1024, size=(args.batch_size, 1024 - (128 - 92)),
                device=device
            )

            # test a manual iterative generation
            cascade_logits = []
            for i in range(0, inputs.size(1), stride):
                inp = inputs[:, i: i + args.cascade_stride]
                output = model(inp, past_key_values=past_key_values, use_cache=True)
                cascade_logits += [output.logits]
                past_key_values = output.past_key_values
                del output

            past_key_values.reset(verbose=True)

            output = model(inputs, past_key_values=past_key_values, use_cache=True)
            dense_logits = output.logits
            dense_logits = torch.split(dense_logits, args.cascade_stride, dim=1)

            for c, d in zip(cascade_logits, dense_logits):
                diff = (c - d).abs().amax(dim=-1)
                self.assertTrue(torch.allclose(c, d, atol=1e-1, rtol=0), f"{args.model=} {diff=}")
                argmax_diff = c.argmax(dim=-1) != d.argmax(dim=-1)
                argmax_diff = argmax_diff.sum() / c.size(-2)
                self.assertTrue(argmax_diff == 0, f"{args.model=} {argmax_diff=}")
                # print(f"{diff=} {argmax_diff=}")

            del model
            del past_key_values
            gc.collect()
            torch.cuda.empty_cache()


class TestFlashAttention(unittest.TestCase):
    def setUp(self):
        print('\n', unittest.TestCase.id(self))

    def test_flash_attention_fwd_bwd(self):
        def test_op_with_backward(Z, H, N_CTX, N_KV, HEAD_DIM, causal, dtype=torch.float16):
            print(f"{Z=} {H=} {N_CTX=} {N_KV=}")
            torch.manual_seed(20)
            q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
            k = (torch.empty((Z, H, N_KV, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
            v = (torch.empty((Z, H, N_KV, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())

            if N_CTX == N_KV:
                M = torch.full((1, N_KV), 1, dtype=torch.bool, device="cuda")
                M_eager = M.repeat(N_CTX, 1).triu(1)
            elif N_CTX < N_KV:
                # M = torch.rand((1, N_KV - N_CTX), device="cuda") > 2 pass (none masked)
                # M = torch.rand((1, N_KV - N_CTX), device="cuda") > -1
                M = torch.rand((1, N_KV - N_CTX), device="cuda") > 0.5

                M2 = torch.full((N_CTX, N_CTX), 1, dtype=torch.bool, device="cuda").triu(1)
                M_eager = torch.cat((M.repeat(N_CTX, 1), M2), dim=-1)

            sm_scale = 0.5
            dout = torch.randn_like(q)
            # reference implementation
            p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
            if causal:
                p[:, :, M_eager == 1] = float("-inf")

            p = torch.softmax(p.float(), dim=-1).to(dtype)
            # p = torch.exp(p)
            ref_out = torch.matmul(p, v).to(dtype)
            ref_out.backward(dout)
            ref_dv, v.grad = v.grad.clone(), None
            ref_dk, k.grad = k.grad.clone(), None
            ref_dq, q.grad = q.grad.clone(), None
            # triton implementation
            tri_out, _ = attention(q, k, v, causal, sm_scale, M.squeeze(0), 1.0)
            tri_out = tri_out.to(dtype)

            # compare
            atol = 1e-2
            fwd_diff = (ref_out - tri_out).abs()
            close = torch.allclose(ref_out, tri_out, atol=atol, rtol=0)
            # print(f"fwd close: {close=}")
            self.assertTrue(close, f"{fwd_diff.amax()=}")

            tri_out.backward(dout, None)
            tri_dv, v.grad = v.grad.clone(), None
            tri_dk, k.grad = k.grad.clone(), None
            tri_dq, q.grad = q.grad.clone(), None

            rtol = 0.0
            # Relative tolerance workaround for known hardware limitation of MI200 GPU.
            # For details see https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
            if torch.version.hip is not None and triton.runtime.driver.active.get_current_target().arch == "gfx90a":
                rtol = 1e-2

            dv_close = torch.allclose(ref_dv, tri_dv, atol=atol, rtol=rtol)
            dk_close = torch.allclose(ref_dk, tri_dk, atol=atol, rtol=rtol)
            dq_close = torch.allclose(ref_dq, tri_dq, atol=atol, rtol=rtol)

            dv_diff = (ref_dv - tri_dv).abs()
            dk_diff = (ref_dk - tri_dk).abs()
            dq_diff = (ref_dq - tri_dq).abs()
            # print(f"{dv_diff=} {dk_diff=} {dq_diff=}")

            # print(f"{dv_diff.amax()=}")
            # print(f"{dk_diff.amax()=}")
            # print(f"{dq_diff.amax()=}")
            # print(f"{dk_close=} {dv_close=} {dq_close=}")

            self.assertTrue(dv_close, f"{dv_diff.amax()=}")
            self.assertTrue(dk_close, f"{dk_diff.amax()=}")
            self.assertTrue(dq_close, f"{dq_diff.amax()=}")

            del q, k, v, ref_dv, ref_dk, ref_dq, ref_out, tri_out, fwd_diff
            gc.collect()
            torch.cuda.empty_cache()

        # torch.set_printoptions(threshold=10_000)
        for Z, H, N_CTX, N_KV, HEAD_DIM in (
            (1, 2, 128, 128, 64),
            (1, 2, 256, 256, 64),
            (1, 2, 512, 512, 64),
            (1, 1, 128, 128 + 128, 64),
            (1, 2, 1024, 16384 + 1024, 64),
            (1, 2, 1024, 16384 + 1024, 64),
            (2, 32, 1024, 16384 + 1024, 128),
            (1, 2, 512, 2560, 64),
            (1, 2, 500, 2048 + 500, 64),
            (1, 2, 500, 2048 + 64 + 500, 64),
            (1, 2, 500, 2048 + 500, 64),

        ):
            test_op_with_backward(Z, H, N_CTX, N_KV, HEAD_DIM, True)

    def test_flash_kernel_scores_against_eager(self):
        """
        the flash attention kernel must approximate the scores since the full normalization
        constant is never available and cannot be stored withouta big increase in memory.
        Test that the approxmate score method is accurate enough.
        """

        with torch.no_grad():
            N_CTX, N_KV = 4096, 4096
            q = torch.randn(1, 1, N_CTX, 128, dtype=torch.float16, device="cuda")
            k = torch.randn(1, 1, N_KV, 128, dtype=torch.float16, device="cuda")
            v = torch.randn(1, 1, N_KV, 128, dtype=torch.float16, device="cuda")

            M = torch.full((1, N_KV - N_CTX), 0, dtype=torch.bool, device="cuda")
            M2 = torch.full((N_CTX, N_CTX), 1, dtype=torch.bool, device="cuda").triu(1)
            M_eager = torch.cat((M.repeat(N_CTX, 1), M2), dim=-1)

            beta = 0.9999

            # flash
            scale = 1 / np.sqrt(q.size(-1))
            _, score = attention(q, k, v, True, scale, M.squeeze(0), beta)
            print(f"kernel score: {score=}")

            # eager
            qkT = torch.einsum("bhqd,bgkd->bhqk", np.sqrt(scale) * q, np.sqrt(scale) * k)
            qkT = qkT + (M_eager * torch.finfo(qkT.dtype).min)
            eager_scores = qkT.softmax(dim=-1).half()

            exps = (1 - beta) * beta**torch.arange(
                eager_scores.size(2), device=eager_scores.device).flip(dims=(0, )).half()
            eager_scores = (eager_scores * exps.half()[None, None, :, None]).sum(dim=2)
            print(f"eager score: {eager_scores=}")

            score_rank = score.argsort(dim=-1, descending=True).argsort(dim=-1).squeeze()
            eager_rank = eager_scores.argsort(dim=-1, descending=True).argsort(dim=-1).squeeze()
            spearman = 1 - 6 * ((score_rank - eager_rank) ** 2).sum() / (score_rank.size(0) * (score_rank.size(0) ** 2 - 1))
            self.assertTrue(spearman.item() > 0.95, f"{spearman.item()=}")

            # scores will be exact if BLOCK_N is set to 64 in flash attention kernel, and N_CTX, N_KV
            # is also set to 64, 64 here. This is because there is no approximation over the softmax
            # normalization constant in this case.
            # score_diff = (score - eager_scores).abs().amax()
            # print(f"{score_diff=}")
            # self.assertTrue(torch.allclose(score, eager_scores, atol=1e-4), f"{score_diff=}")

        # del q, k, v, cache, eager_cache, qkT, M, M2, M_eager
        gc.collect()


class TestCascadingKVCache(unittest.TestCase):
    def setUp(self):
        print('\n', unittest.TestCase.id(self))

    def test_different_sized_layer_caches(self):
        # toy settings
        cache_sizes = [2**i for i in range(5, 10)]
        windows = [2**i for i in range(4)]

        for i in range(10):
            n_layers = np.random.randint(8, 33, size=(1,)).tolist()[0]
            N = 1
            HID = 128
            NSINK = 64
            HEAD = 1
            DEVICE = "cuda:0"
            DTYPE = torch.float

            MAX_SEQ = np.random.choice(np.array(cache_sizes), size=(n_layers,)).tolist()
            w = np.random.choice(windows, size=(n_layers,)).tolist()

            # MAX_SEQ = [s for _ in range(n_layers)]
            WIND = [ms // win for ms, win in zip(MAX_SEQ, w)]

            S = torch.randint(max(MAX_SEQ) + NSINK, max(MAX_SEQ) * 2, size=(1,)).item()
            # S = s + NSINK
            with torch.no_grad():

                cache = CascadingKVCache(
                    window_length=WIND,
                    num_sink_tokens=NSINK,
                    max_batch_size=N,
                    heads=HEAD,
                    dim=HID,
                    max_seq_len=MAX_SEQ,
                    device=DEVICE,
                    dtype=DTYPE,
                    layers=n_layers,
                    eager_fill=True,
                )

                for l in range(n_layers):
                    _k = torch.randn(N, HEAD, S, HID, device=DEVICE, dtype=DTYPE)
                    _v = torch.randn(N, HEAD, S, HID, device=DEVICE, dtype=DTYPE)

                    k, v, pos, sink_mask, k_nosink, v_nosink, pos_nosink, mask, of_pos = \
                        cache.update(_k, _v, l)

                    self.assertEqual(k_nosink.size(2), MAX_SEQ[l], f"{k.size()=} expected to match {MAX_SEQ[l]=} at dim 2")
                    self.assertEqual(v_nosink.size(2), MAX_SEQ[l], f"{v.size()=} expected to match {MAX_SEQ[l]=} at dim 2")
                    self.assertEqual(mask.sum().item(), 0,
                                     "expected nothing to be masked due to eager filling: " + \
                                     f"({MAX_SEQ[l]=}, {k_nosink.size()=}, {mask.size()=}, queries: {_k.size()=})\n{mask=}" + \
                                     f"{WIND[l]=} layer: {l=} {cache.stored_tokens[l]=}"
                                     )
            del cache, k, v, pos, sink_mask, k_nosink, v_nosink, pos_nosink, mask, of_pos
            gc.collect()

    def test_num_sinks(self):
        # toy settings
        N = 2
        HID = 128
        HEAD = 32
        MAX_SEQ = 4096
        DEVICE = "cuda:0"
        DTYPE = torch.float16

        # NOTE: ONLY TESTING THE CACHE HERE, BUT <64 sinks does not work with flash attention
        for NSINK in [4, 8, 16, 32, 64]:
            with torch.no_grad():
                # MAX_SEQ = 2 ** torch.randint(10, 10, size=(1,)).item()
                WIND = MAX_SEQ // 4
                S = 512

                for j in range(20):
                    cache = CascadingKVCache(
                        window_length=WIND,
                        num_sink_tokens=NSINK,
                        max_batch_size=N,
                        heads=HEAD,
                        dim=HID,
                        max_seq_len=MAX_SEQ,
                        device=DEVICE,
                        dtype=DTYPE,
                        layers=1,
                        eager_fill=False,
                    )

                    start = 0
                    for i in range(20):
                        print(f"{start=}")
                        _k = torch.arange(S, device=DEVICE, dtype=DTYPE) + start + 1
                        start += S
                        _k = _k.view(1, 1, -1, 1).repeat(N, HEAD, 1, HID)
                        _v = _k.clone()

                        out = cache.update(_k, _v, 0)
                    print(f"{NSINK=}")

                    del cache, _k, _v, out
                    gc.collect()

    def test_eager_add_same_as_one_cascade(self):
        # toy settings
        N = 1
        HID = 1
        NSINK = 64
        HEAD = 1
        MAX_SEQ = 32
        DEVICE = "cuda:0"
        DTYPE = torch.float16

        for i in range(10):
            with torch.no_grad():
                MAX_SEQ = 2 ** torch.randint(5, 10, size=(1,)).item()
                WIND = MAX_SEQ // 4
                S = MAX_SEQ

                cache_sllm = CascadingKVCache(
                    window_length=MAX_SEQ,
                    num_sink_tokens=NSINK,
                    max_batch_size=N,
                    heads=HEAD,
                    dim=HID,
                    max_seq_len=MAX_SEQ,
                    device=DEVICE,
                    dtype=DTYPE,
                    layers=1,
                    eager_fill=False,
                )

                cache = CascadingKVCache(
                    window_length=WIND,
                    num_sink_tokens=NSINK,
                    max_batch_size=N,
                    heads=HEAD,
                    dim=HID,
                    max_seq_len=MAX_SEQ,
                    device=DEVICE,
                    dtype=DTYPE,
                    layers=1,
                    eager_fill=True,
                )

                _k = torch.arange(S, device=DEVICE, dtype=DTYPE) + 1
                _k = _k.view(1, 1, -1, 1).repeat(N, HEAD, 1, HID)
                _v = _k.clone()

                for i in range(_k.size(2)):
                    # print(f"\n\n{'='*100}\n\n")
                    sllm_out = cache_sllm.update(_k[:, :, i:i + 1].clone(), _v[:, :, i:i + 1].clone(), 0)

                out = cache.update(_k, _v, 0)

                names = ("sink keys", "sink values", "sink pos", "sink mask", "keys", "values", "pos", "mask", "og pos")
                sizes = (4, 4, 2, 2, 4, 4, 2, 2, 3)

                for sllm_item, item, name, size in zip(sllm_out, out, names, sizes):
                    if size == 4:
                        sllm_list = sllm_item[0, 0, :, 0].tolist()
                        item_list = item[0, 0, :, 0].tolist()
                    elif size == 2:
                        sllm_list = sllm_item[0].tolist()
                        item_list = item[0].tolist()
                    elif size == 3:
                        sllm_list = sllm_item[0, 0].tolist()
                        item_list = item[0, 0].tolist()
                    passing = True
                    for v in item_list:
                        if v not in sllm_list:
                            passing = False
                            break

                    self.assertTrue(passing, f"failed {name=} Streaming LLM: {sllm_item=} Ours: {item=}")

            del cache, cache_sllm, sllm_out, out, _k, _v
            gc.collect()

    def test_against_naive_single_iter(self):
        # toy settings
        N = 1
        HID = 128
        NSINK = 64
        WIND = 4
        HEAD = 32
        MAX_SEQ = 32
        DEVICE = "cuda:0"
        DTYPE = torch.float16

        for i in range(10):
            with torch.no_grad():
                S = torch.randint(200, 500, size=(1,)).item()

                cache_slow = CascadingKVCacheSlow(
                    window_length=WIND,
                    num_sink_tokens=NSINK,
                    max_batch_size=N,
                    heads=HEAD,
                    dim=HID,
                    n_layers=1,
                    max_seq_len=MAX_SEQ,
                    device=DEVICE,
                    dtype=DTYPE,
                )
                cache_slow.add_sinks = add_sinks
                cache_slow.append_to_cache = append_to_cache
                cache_slow.evict_from_cache = evict_from_cache
                cache_slow.overwrite_cache = overwrite_cache

                cache = CascadingKVCache(
                    window_length=WIND,
                    num_sink_tokens=NSINK,
                    max_batch_size=N,
                    heads=HEAD,
                    dim=HID,
                    max_seq_len=MAX_SEQ,
                    device=DEVICE,
                    dtype=DTYPE,
                    layers=1,
                    eager_fill=False,
                )

                slow_times, fast_times = [], []

                # k = torch.arange(S, device=DEVICE, dtype=DTYPE) + 1
                # k = k.view(1, 1, -1, 1).repeat(N, HEAD, 1, HID)
                # v = k.clone()
                _k = torch.randn(N, HEAD, S, HID, device=DEVICE, dtype=DTYPE)
                _v = torch.randn(N, HEAD, S, HID, device=DEVICE, dtype=DTYPE)

                for i in range(_k.size(2)):
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
                    ) = cache_slow.update(_k[:, :, i:i + 1].clone(), _v[:, :, i:i + 1].clone())

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

                    tic = time.perf_counter()
                    k, v, pos, sink_mask, k_nosink, v_nosink, pos_nosink, mask, of_pos = \
                        cache.update(_k[:, :, i:i + 1], _v[:, :, i:i + 1], 0)

                    fast_times.append(time.perf_counter() - tic)

                    # print(
                    #     f"fast from mdoel out {pos.size()=} {pos_nosink.size()=} {k.size()=} {k_nosink.size()=}"
                    # )
                    # print(f"sink: {k}")
                    # print(f"nosink: {k_nosink}")
                    # print(f"og pos: {cache.og_pos[0, 0]=}")
                    idx = 0
                    pos, mask, sink_mask, pos_nosink = pos[idx], mask[
                        idx, 0], sink_mask[idx, 0], pos_nosink[idx]

                    # print(f"{k.size()=}")
                    # print(f"{k=}")
                    # print(f"{k[idx:idx+1].view(-1)=}\n{pos=}\n{sink_mask=}")
                    k, v = torch.cat((k, k_nosink), dim=-2), torch.cat((v, v_nosink), dim=-2)
                    pos = torch.cat((pos, pos_nosink), dim=-1)

                    # print(f"{k[0, 0, :,  0]=}")
                    mask = torch.cat((sink_mask, mask), dim=-1)

                    n = (mask != 1).sum()
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

                    diff = (k_slow - k).abs().sum()
                    pos_diff = (pos - pos_slow).abs()
                    # torch.set_printoptions(profile="default")
                    if diff > 1e-6 or pos_diff.sum() > 0:
                        print(f"{k_slow=}")
                        print(f"{k_slow[0, 0, -100:, 0]=}")
                        print(f"{k=}")
                        print(f"{(k - k_slow).abs()=}")
                        print(
                            f"{k_slow.view(-1)=}\n{k.view(-1)=}\n{pos.view(-1)=}\n{(k - k_slow).abs().view(-1)=}"
                        )
                        print(f"pos diff: {(pos - pos_slow).abs()} {(pos - pos_slow).sum()=}")
                        self.fail("failed difference test")

            del cache, cache_slow, k, v, k_slow, v_slow, pos_slow, sink_mask_slow
            del k_nosink_slow, v_nosink_slow, pos_nosink_slow, mask_slow
            del pos, sink_mask, k_nosink, v_nosink, pos_nosink, mask, of_pos
            gc.collect()
            slow_times = slow_times[100:]
            fast_times = fast_times[100:]
            slow_times = sum(slow_times) / len(slow_times)
            fast_times = sum(fast_times) / len(fast_times)
            print(f"{slow_times=} {fast_times=} ratio {slow_times / fast_times}")

    def test_against_naive_batched(self):
        # toy settings
        N = 1
        HID = 128
        NSINK = 64
        WIND = 512
        HEAD = 32
        MAX_SEQ = 2048
        DEVICE = "cuda:0"
        DTYPE = torch.float16

        for i in range(10):
            with torch.no_grad():
                S = torch.randint(1000, 20000, size=(1,)).item()

                cache_slow = CascadingKVCacheSlow(
                    window_length=WIND,
                    num_sink_tokens=NSINK,
                    max_batch_size=N,
                    heads=HEAD,
                    dim=HID,
                    n_layers=1,
                    max_seq_len=MAX_SEQ,
                    device=DEVICE,
                    dtype=DTYPE,
                )
                cache_slow.add_sinks = add_sinks
                cache_slow.append_to_cache = append_to_cache
                cache_slow.evict_from_cache = evict_from_cache
                cache_slow.overwrite_cache = overwrite_cache

                cache = CascadingKVCache(
                    window_length=WIND,
                    num_sink_tokens=NSINK,
                    max_batch_size=N,
                    heads=HEAD,
                    dim=HID,
                    max_seq_len=MAX_SEQ,
                    device=DEVICE,
                    dtype=DTYPE,
                    layers=1,
                    eager_fill=False,
                )

                slow_times, fast_times = [], []

                # k = torch.arange(S, device=DEVICE, dtype=DTYPE) + 1
                # k = k.view(1, 1, -1, 1).repeat(N, HEAD, 1, HID)
                # v = k.clone()
                k = torch.randn(N, HEAD, S, HID, device=DEVICE, dtype=DTYPE)
                v = torch.randn(N, HEAD, S, HID, device=DEVICE, dtype=DTYPE)

                for i in range(k.size(2)):
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
                    ) = cache_slow.update(k[:, :, i:i + 1].clone(), v[:, :, i:i + 1].clone())

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
                _ = cache.update(k, v, 0)
                cache.reset()

                tic = time.perf_counter()
                k, v, pos, sink_mask, k_nosink, v_nosink, pos_nosink, mask, of_pos = \
                    cache.update(k, v, 0)

                fast_times.append(time.perf_counter() - tic)

                # print(
                #     f"fast from mdoel out {pos.size()=} {pos_nosink.size()=} {k.size()=} {k_nosink.size()=}"
                # )
                # print(f"sink: {k}")
                # print(f"nosink: {k_nosink}")
                # print(f"og pos: {cache.og_pos[0, 0]=}")
                idx = 0
                pos, mask, sink_mask, pos_nosink = pos[idx], mask[
                    idx, 0], sink_mask[idx, 0], pos_nosink[idx]

                # print(f"{k.size()=}")
                # print(f"{k=}")
                # print(f"{k[idx:idx+1].view(-1)=}\n{pos=}\n{sink_mask=}")
                k, v = torch.cat((k, k_nosink), dim=-2), torch.cat((v, v_nosink), dim=-2)
                pos = torch.cat((pos, pos_nosink), dim=-1)

                # print(f"{k[0, 0, :,  0]=}")
                mask = torch.cat((sink_mask, mask), dim=-1)

                n = (mask != 1).sum()
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

                diff = (k_slow - k).abs().sum()
                if diff > 1e-6:
                    print(f"{k_slow=}")
                    print(f"{k_slow[0, 0, -100:, 0]=}")
                    print(f"{k=}")
                    print(f"{(k - k_slow).abs()=}")
                    print(
                        f"{k_slow.view(-1)=}\n{k.view(-1)=}\n{pos.view(-1)=}\n{(k - k_slow).abs().view(-1)=}"
                    )
                    print(f"pos diff: {(pos - pos_slow).abs()} {(pos - pos_slow).sum()=}")
                    exit("too big")

            del cache, cache_slow, k, v, k_slow, v_slow, pos_slow, sink_mask_slow
            del k_nosink_slow, v_nosink_slow, pos_nosink_slow, mask_slow
            del pos, sink_mask, k_nosink, v_nosink, pos_nosink, mask, of_pos
            gc.collect()

            slow_times = slow_times[100:]
            slow_times = sum(slow_times) / len(slow_times)
            fast_times = sum(fast_times) / len(fast_times) / S
            print(f"{slow_times=} {fast_times=} ratio {slow_times / fast_times}")

    def test_bench_cascade_single_vs_four(self):
        total_cache = 16384
        naive = SinkAttentionNaive(64, total_cache).cuda()

        dev = "cuda:0"
        cache = CascadingKVCache(
            window_length=total_cache // 1,
            num_sink_tokens=64,
            max_batch_size=8,
            heads=32,
            dim=128,
            max_seq_len=total_cache,
            device=dev,
            dtype=torch.float32,
            layers=1,
        )

        cache_cascade = CascadingKVCache(
            window_length=total_cache // 4,
            num_sink_tokens=64,
            max_batch_size=8,
            heads=32,
            dim=128,
            max_seq_len=total_cache,
            device=dev,
            dtype=torch.float32,
            layers=1,
            eager_fill=False,
        )

        naive_times, fast_times, cascade_times = [], [], []
        its = total_cache
        # testing the original version
        for i in range(its):
            k = torch.randn(8, 32, 1, 128).cuda()
            v = torch.randn(8, 32, 1, 128).cuda()
            attn_score = torch.randn(8, 32, 1024).cuda()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            sink_k, sink_v, k_cache, v_cache = naive(k, v)
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end)
            naive_times.append(elapsed)

        k = torch.randn(8, 32, its, 128).cuda()
        v = torch.randn(8, 32, its, 128).cuda()
        attn_score = torch.randn(8, 32, its).cuda()

        # testing the cascade fast version
        # burn in period for triton compile
        _ = cache_cascade.update(k, v, 0, attn_score)
        cache_cascade.reset()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _k, _v, pos, sink_mask, k_nosink, v_nosink, pos_nosink, mask, og_pos = cache_cascade.update(
            k, v, 0, attn_score)
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
        cascade_times.append(elapsed)

        # testing the fast version of SLLM
        # burn in period for triton compile
        _ = cache.update(k, v, 0, attn_score)
        cache.reset()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _k, _v, pos, sink_mask, k_nosink, v_nosink, pos_nosink, mask, og_pos = cache.update(
            k, v, 0, attn_score)
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
        fast_times.append(elapsed)

        naive_mean = sum(naive_times) / len(naive_times)
        fast_mean = sum(fast_times) / len(fast_times) / its
        cascade_mean = sum(cascade_times) / len(cascade_times) / its

        del cache, cache_cascade, _k, _v, pos, sink_mask, k_nosink, v_nosink, mask, og_pos
        gc.collect()
        print(f"{naive_mean=} {fast_mean=} {cascade_mean=}")
        print(f"{cascade_mean / naive_mean=}")
        print(f"{fast_mean / naive_mean=}")

        save = False
        if save:
            d = dict(naive_times=naive_times, fast_times=fast_times, cascade_times=cascade_times)
            with open("./plots/latency/cache_times.json", "w") as f:
                json.dump(d, f)


class RebuttalTest(unittest.TestCase):
    def setUp(self):
        print('\n', unittest.TestCase.id(self))
        args = Namespace(
            batch_size=1,
            sinks=64,
            cascades=4,
            window=65536,
            world_size=1,
            cascade_func="pow2",
            head_reduction="mean",
            method="sink",
            use_fp32=False,   # use fp 32 for testing
            cascade_stride=4096,
            job="test",
            lora_r=0,
            homogeneous_heads=True,
            do_og_pos=False,
        )
        self.args = args
        torch.set_grad_enabled(False)

    def tearDown(self):
        torch.set_grad_enabled(True)

    def test_minference(self):
        model_name = "/d1/dataset/llama/models/llama_v3.1/Meta-Llama-3.1-8B-Instruct"

        pipe = pipeline("text-generation", model=model_name, torch_dtype="auto", device_map="auto")

        for i in range(11, 21):
            prompt = pipe.tokenizer.decode(torch.randint(0, 32768, size=(2**i,)))

            # Patch MInference Module,
            # If you use the local path, please use the model_name from HF when initializing MInference.
            minference_patch = MInference(
                "streaming",
                model_name,
                n_local=65536 + 4096,
                n_init=64
            )
            pipe.model = minference_patch(pipe.model)

            tic = time.perf_counter()
            _ = pipe(prompt, max_new_tokens=1)
            torch.cuda.synchronize()
            print(f"minference time ({2**i}): {time.perf_counter() - tic=}")
            # print(out)

    def test_cascade(self):
        args = self.args
        args.model = "llama3.1-8b-instruct"

        model, tokenizer, device = load_model(args)
        model = sample_monkeypatch(model)

        for i in range(11, 21):
            if i < 12:
                continue
            inputs = torch.randint(0, 32768, size=(1, 2**i)).cuda()

            past_key_values = get_cache(model, dtype=torch.float16)
            tic = time.perf_counter()
            output = model.generate(
                input_ids=inputs,
                max_new_tokens=1,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                use_cache=True,
                past_key_values=past_key_values,
            )[0]
            torch.cuda.synchronize()
            print(f"cascade time ({2**i}): {time.perf_counter() - tic=}")

            _ = tokenizer.decode(output[inputs.size(1):])

        del model
        del past_key_values


if __name__ == "__main__":
    N = 128
    HID = 128
    NSINK = 64
    WIND = 2048 // 4
    HEAD = 32
    MAX_SEQ = 2048
    DEVICE = "cuda:0"
    DTYPE = torch.float16

    # for nsys profiling
    cache = CascadingKVCache(
        window_length=WIND,
        num_sink_tokens=NSINK,
        max_batch_size=N,
        heads=HEAD,
        dim=HID,
        max_seq_len=MAX_SEQ,
        device=DEVICE,
        dtype=DTYPE,
        layers=1,
        eager_fill=False,
    )

    ITS = 16384
    with torch.no_grad():
        s_in = torch.randn(N, HEAD, ITS, device=DEVICE, dtype=DTYPE)
        k_in = torch.randn(N, HEAD, ITS, HID, device=DEVICE, dtype=DTYPE)
        s_in = torch.randn(N, HEAD, ITS, device=DEVICE, dtype=DTYPE)
        v_in = torch.randn(N, HEAD, ITS, HID, device=DEVICE, dtype=DTYPE)

        for k_i, v_i, s_i in zip(k_in.chunk(1, dim=2), v_in.chunk(1, dim=2),
                                 s_in.chunk(1, dim=2)):
            _ = cache.update_batch(k_i, v_i, s_i, 0, )
