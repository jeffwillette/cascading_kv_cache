import torch

import triton
import triton.language as tl


def is_hip():
    return False
    # return triton.runtime.driver.active.get_current_target().backend == "hip"


DTYPE = tl.float16


@triton.jit
def _attn_fwd_inner(
    acc,  # accumulator for qk - exp(max)
    l_i,  # running normalization constant sum
    m_i,  # running maximum
    q,  #
    K_block_ptr,
    V_block_ptr,  #
    start_m,
    qk_scale,  #
    MASK,
    stride_mm,
    stride_mn,
    SCORES,
    stride_sn,
    beta: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,  #
    N_CTX: tl.constexpr,
    N_KV: tl.constexpr,
    fp8_v: tl.constexpr,
):
    lo, hi = 0, N_KV

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    Mask_block = offs_n * stride_mn

    # loop over k, v and update accumulator
    mask_vert = tl.full((BLOCK_M, 1), value=1, dtype=tl.int1)
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)

        N_CACHED = N_KV - N_CTX
        if start_n < N_CACHED:
            # load from the cache mask (sink, keyvals)
            mask = (mask_vert * \
                    tl.load(MASK + Mask_block)[None, :].to(tl.int1)).to(tl.int1)
        else:
            # load a regular causal mask for the leading block (sink, cached keyvals, ctx keyvals)
            mask = (offs_m[:, None] < (start_n - N_CACHED + offs_n[None, :])).to(tl.int1)

        qk = tl.dot(q, k)
        qk = qk * qk_scale + tl.where(mask, -1.0e6, 0)

        # ------------------------------
        # for sum accumulation
        # coeff = 1
        # ------------------------------

        # for EMA accumulation --------
        # exps = tl.flip(tl.arange(0, BLOCK_M))[:, None]  # original submission
        exps = (N_CTX - 1) - (start_m + (tl.arange(0, BLOCK_M))[:, None])  # bugfix
        # do beta ** exps * (1 - beta)
        unmasked = tl.where(mask, 0, 1)
        exps = tl.exp2(exps.to(DTYPE) * tl.log2(beta))
        coeff = exps * (1 - beta) * unmasked
        # ------------------------------

        score_offset = (start_n + tl.arange(0, BLOCK_N)).to(
            tl.int64) * stride_sn

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        # this will be incrementally more accurate as we get to the end of the sequence.
        # it is not exactly equivalent to the non-flash attention version.
        # print("qk term: ", tl.sum((p / l_ij[:, None])))
        # print("coeff term: ", coeff)
        steps_left = (hi - (start_n + BLOCK_N)) // BLOCK_N
        steps_done = (start_n + BLOCK_N) // BLOCK_N
        adj = steps_left / steps_done
        # print("adj: ", adj)
        tl.atomic_add(SCORES + score_offset, val=tl.sum((p / (l_i[:, None] + (l_i[:, None] * adj) + 1e-6)) * coeff, 0))

        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(DTYPE)

        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij

        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        Mask_block += BLOCK_N * stride_mn

    return acc, l_i, m_i


# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [64, 128]\
    for BN in [32, 64]\
    for s in ([1] if is_hip() else [3, 4, 7])\
    for w in [4, 8]\
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


# @triton.autotune(list(filter(keep, configs)), key=["N_CTX"])
@triton.jit
def _attn_fwd(
        Q,
        K,
        V,
        sm_scale,
        M,
        Out,  #
        MASK,
        SCORES,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,  #
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,  #
        stride_vz,
        stride_vh,
        stride_vk,
        stride_vn,  #
        stride_oz,
        stride_oh,
        stride_om,
        stride_on,  #
        stride_mm,
        stride_mn,
        stride_sz,
        stride_sh,
        stride_sn,
        Z,
        H,
        N_CTX: tl.constexpr,  #
        N_KV: tl.constexpr,
        beta: tl.constexpr,
        BLOCK_M: tl.constexpr,  #
        BLOCK_N: tl.constexpr,  #
        HEAD_DIM: tl.constexpr,  #
        STAGE: tl.constexpr  #
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    kv_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    score_offset = off_z.to(tl.int64) * stride_sz + off_h.to(
        tl.int64) * stride_sh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(N_KV, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(0 if V.dtype.element_ty == tl.float8e5 else 1,
               1 if V.dtype.element_ty == tl.float8e5 else 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(HEAD_DIM, N_KV),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )

    O_block_ptr = tl.make_block_ptr(
        base=Out + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)

    acc, l_i, m_i = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,
        K_block_ptr,
        V_block_ptr,  #
        start_m,
        qk_scale,  #
        MASK,
        stride_mm,
        stride_mn,
        SCORES + score_offset,
        stride_sn,
        beta,
        BLOCK_M,
        HEAD_DIM,
        BLOCK_N,  #
        offs_m,
        offs_n,
        N_CTX,
        N_KV,
        V.dtype.element_ty == tl.float8e5,  #
    )

    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


@triton.jit
def _attn_bwd_preprocess(O, DO,  #
                         Delta,  #
                         Z, H, N_CTX,  #
                         BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr  #
                         ):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    # load
    o = tl.load(O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :])
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv(dk, dv,  #
                   Q, k, v, sm_scale,  #
                   DO,  #
                   M, D,  #
                   # shared by Q/K/V/DO.
                   stride_tok, stride_d,  #
                   H, N_CTX, N_KV, BLOCK_M1: tl.constexpr,  #
                   BLOCK_N1: tl.constexpr,  #
                   HEAD_DIM: tl.constexpr,  #
                   # Filled in by the wrapper.
                   start_n, start_m, num_steps,  #
                   MASK_TNS,
                   stride_m_q, stride_m_k,
                   ):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)

    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1

    # determine if we are in the cached block or the causal block
    is_causal_block = start_n >= N_KV - N_CTX
    mask_vert = tl.full((1, BLOCK_M1), 1, dtype=tl.int1)
    N_CACHED = N_KV - N_CTX
    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m[None, :])
        # Autoregressive masking.

        if is_causal_block:
            mask = (offs_n[:, None] - N_CACHED > offs_m[None, :]).to(tl.int1)
        else:
            mask = tl.load(MASK_TNS + offs_n * stride_m_k).to(tl.int1)
            mask = (mask_vert.to(tl.int1) * mask[:, None]).to(tl.int1)

        pT = tl.where(mask, 0.0, pT)

        # this was from the original implementation
        # mask = (offs_m[None, :] >= offs_n[:, None])
        # pT = tl.where(mask, pT, 0.0)

        do = tl.load(do_ptrs)
        # Compute dV.
        ppT = pT
        ppT = ppT.to(DTYPE)
        dv += tl.dot(ppT, do)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(DTYPE)
        dk += tl.dot(dsT, tl.trans(qT))
        # Increment pointers.
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
    return dk, dv


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(dq, q, K, V,  #
                 do, m, D,
                 # shared by Q/K/V/DO.
                 stride_tok, stride_d,  #
                 H, N_CTX, N_KV,  #
                 MASK_TNS,
                 stride_m_q, stride_m_k,
                 BLOCK_M2: tl.constexpr,  #
                 BLOCK_N2: tl.constexpr,  #
                 HEAD_DIM: tl.constexpr,
                 # Filled in by the wrapper.
                 start_m, start_n, num_steps,  #
                 ):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2

    mask_vert = tl.full((BLOCK_M2,), 1, dtype=tl.int1)
    N_CACHED = N_KV - N_CTX
    for blk_idx in range(num_steps):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk - m)
        # Autoregressive masking.

        offs_n = curr_n + tl.arange(0, BLOCK_N2)
        is_causal_block = curr_n >= N_KV - N_CTX
        if is_causal_block:
            mask = (offs_m[:, None] < offs_n[None, :] - N_CACHED)
        else:
            mask = tl.load(MASK_TNS + offs_n * stride_m_k)
            mask = (mask[None, :] * mask_vert[:, None]).to(tl.int1)

        p = tl.where(mask, 0.0, p)

        # original implementation
        # offs_n = curr_n + tl.arange(0, BLOCK_N2)
        # mask = (offs_m[:, None] >= offs_n[None, :])
        # p = tl.where(mask, p, 0.0)

        # Compute dP and dS.
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(DTYPE)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT))
        # Increment pointers.
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
    return dq


@triton.jit
def _attn_bwd(Q, K, V, sm_scale,  #
              DO,  #
              DQ, DK, DV,  #
              M, D,
              # shared by Q/K/V/DO.
              stride_z, stride_h, stride_tok, stride_d,  #
              stride_kz, stride_kh, stride_ktok, stride_kd,  #
              MASK,
              stride_m_q, stride_m_k,
              H, N_CTX, N_KV,  #
              BLOCK_M1: tl.constexpr,  #
              BLOCK_N1: tl.constexpr,  #
              BLOCK_M2: tl.constexpr,  #
              BLOCK_N2: tl.constexpr,  #
              BLK_SLICE_FACTOR: tl.constexpr,  #
              HEAD_DIM: tl.constexpr):

    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    bhid = tl.program_id(2)
    # off_chz = (bhid * N_KV).to(tl.int64)
    off_chzq = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    adjk = (stride_kh * (bhid % H) + stride_kz * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adj
    K += adjk
    V += adjk
    DO += adj
    DQ += adj
    DK += adjk
    DV += adjk
    M += off_chzq
    D += off_chzq

    # load scales
    offs_k = tl.arange(0, HEAD_DIM)

    start_n = pid * BLOCK_N1
    start_m = 0

    # MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + offs_n[:, None] * stride_ktok + offs_k[None, :] * stride_kd)
    v = tl.load(V + offs_n[:, None] * stride_ktok + offs_k[None, :] * stride_kd)

    # dk_dv sets off_n anf then iterates over the m (queries to do the bwd)
    num_steps = N_CTX // BLOCK_M1
    dk, dv = _attn_bwd_dkdv(dk, dv,  #
                            Q, k, v, sm_scale,  #
                            DO,  #
                            M, D,  #
                            stride_tok, stride_d,  #
                            H, N_CTX, N_KV,  #
                            BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
                            start_n, start_m, num_steps,  #
                            MASK,
                            stride_m_q, stride_m_k,
                            )

    dv_ptrs = DV + offs_n[:, None] * stride_ktok + offs_k[None, :] * stride_kd
    tl.store(dv_ptrs, dv)

    # Write back dK.
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_ktok + offs_k[None, :] * stride_kd
    tl.store(dk_ptrs, dk)

    # THIS BLOCK DOES DQ:
    start_m = pid * BLOCK_M2
    start_n = 0

    if start_m < N_CTX:
        offs_m = start_m + tl.arange(0, BLOCK_M2)

        q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
        dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
        do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

        m = tl.load(M + offs_m)
        m = m[:, None]

        num_steps = N_KV // BLOCK_N2

        dq = _attn_bwd_dq(dq, q, K, V,  #
                          do, m, D,  #
                          stride_ktok, stride_kd,  #
                          H, N_CTX, N_KV,
                          MASK,
                          stride_m_q, stride_m_k,
                          BLOCK_M2, BLOCK_N2, HEAD_DIM,  #
                          start_m, start_n, num_steps,  #
                          )

        # Write back dQ.
        dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
        dq *= LN2
        tl.store(dq_ptrs, dq)


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, mask, beta):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-2] if v.dtype == torch.float8_e5m2 else v.shape[
            -1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        assert k.size(2) == v.size(2)
        N_KV = k.size(2)  # constant
        n_kv = N_KV       # variable

        # pad the key values because the BLOCK_N is always expecting some multiple of BLOCK_N
        # we will have to mask out the unwanted values in the attention_inner
        b, h, s, d = k.shape
        N_SCORES = N_KV
        if N_KV % 64 != 0:
            n = 64 - (N_KV % 64)  # + 32
            k = torch.cat(
                (k, torch.zeros(b, h, n, d, dtype=k.dtype, device=k.device)),
                dim=2)
            v = torch.cat(
                (v, torch.zeros(b, h, n, d, device=v.device, dtype=v.dtype)),
                dim=2)
            n_kv += n

            # mask will be taken care of by trating the leading edge of the
            # attention matrix as the causal portion

            N_SCORES += n

        scores = torch.zeros(q.size(0),
                             q.size(1),
                             N_SCORES,
                             device=q.device,
                             dtype=q.dtype)

        og_q = q.size(2)
        if q.size(2) % 64 != 0:
            n = 64 - (q.size(2) % 64)
            q = torch.cat((q, torch.zeros(b, h, n, d, dtype=q.dtype, device=q.device)), dim=2)

        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}
        # Tuning for AMD target
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {
                "waves_per_eu": waves_per_eu,
                "allow_flush_denorm": True
            }

        def grid(args):
            return (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]),
                        device=q.device,
                        dtype=torch.float32)

        # print(f"{q.size()=} {k.size()=} {v.size()=} {sm_scale=} {M.size()=} {o.size()=}")
        # print(f"{mask.size()=} {scores.size()=} {q.stride()=} {k.stride()=} {v.stride()=}")
        # print(f"{o.stride()=} {mask.stride()=} {scores.stride()=}")

        mask = mask[None, :].contiguous()
        assert len(mask.shape) == 2

        _attn_fwd[grid](
            q,
            k,
            v,
            sm_scale,
            M,
            o,  #
            mask,
            scores,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            mask.stride(0), mask.stride(1),
            scores.stride(0), scores.stride(1), scores.stride(2),
            q.shape[0],
            q.shape[1],  #
            N_CTX=q.shape[2],  #
            N_KV=n_kv,
            beta=beta,
            HEAD_DIM=HEAD_DIM_K,  #
            STAGE=stage,  #
            # temporary fix to hardcode
            BLOCK_M=64,
            BLOCK_N=32,
            num_warps=4,
            num_stages=3,
            **extra_kern_args)

        # print(f"{o.size()=} {scores.size()=}")

        ctx.save_for_backward(q, k, v, o, M, mask)
        ctx.og_q = og_q
        ctx.og_kv = N_KV
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal

        o = o[:, :, :og_q].contiguous()
        scores = scores[:, :, :N_KV]

        return o, scores

    @staticmethod
    def backward(ctx, do, d_scores):
        q, k, v, o, M, mask = ctx.saved_tensors
        if do.size() != q.size():
            n = q.size(2) - do.size(2)
            b, h, _, d = q.size()
            do = torch.cat((do, torch.zeros(b, h, n, d, device=do.device, dtype=do.dtype)), dim=-2)

        assert do.is_contiguous()
        assert q.stride() == o.stride() == do.stride()
        assert k.stride() == v.stride()

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        N_KV = k.shape[2]
        PRE_BLOCK = 128
        NUM_WARPS, NUM_STAGES = 4, 4
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 64, 64, 32
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)
        PRE_BLOCK = 128
        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M)
        _attn_bwd_preprocess[pre_grid](
            o, do,  #
            delta,  #
            BATCH, N_HEAD, N_CTX,  #
            BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM  #
        )

        grid = (N_KV // BLOCK_N1, 1, BATCH * N_HEAD)
        _attn_bwd[grid](
            q, arg_k, v, ctx.sm_scale, do, dq, dk, dv,  #
            M, delta,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            mask,
            mask.stride(0), mask.stride(1),
            N_HEAD, N_CTX, N_KV,  #
            BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,  #
            BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,  #
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
            HEAD_DIM=ctx.HEAD_DIM,  #
            num_warps=NUM_WARPS,  #
            num_stages=NUM_STAGES  #
        )

        dq = dq[:, :, :ctx.og_q]
        dk = dk[:, :, :ctx.og_kv]
        dv = dv[:, :, :ctx.og_kv]

        return dq, dk, dv, None, None, None, None


attention = _attention.apply

try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    # turned off due to GPU requirements, needed to test flash attention standalone
    # HAS_FLASH = True
    HAS_FLASH = False
except BaseException:
    HAS_FLASH = False

TORCH_HAS_FP8 = hasattr(torch, 'float8_e5m2')
BATCH, N_HEADS, HEAD_DIM = 4, 32, 128
# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd", "bwd"]:
    for causal in [True, False]:
        if mode == "bwd" and not causal:
            continue
        configs.append(
            triton.testing.Benchmark(
                x_names=["N_CTX"],
                x_vals=[2**i for i in range(9, 10)],
                line_arg="provider",
                line_vals=["triton-fp16"] +
                (["triton-fp8"] if TORCH_HAS_FP8 else []) +
                (["flash"] if HAS_FLASH else []),
                line_names=["Triton [FP16]"] +
                (["Triton [FP8]"] if TORCH_HAS_FP8 else []) +
                (["Flash-2"] if HAS_FLASH else []),
                styles=[("red", "-"), ("blue", "-")],
                ylabel="ms",
                plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal={causal}",
                args={
                    "H": N_HEADS,
                    "BATCH": BATCH,
                    "HEAD_DIM": HEAD_DIM,
                    "mode": mode,
                    "causal": causal,
                },
            ))


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH,
                          H,
                          N_CTX,
                          HEAD_DIM,
                          causal,
                          mode,
                          provider,
                          device="cuda"):
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100
    dtype = torch.float16
    if "triton" in provider:
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM),
                        dtype=dtype,
                        device=device,
                        requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM),
                        dtype=dtype,
                        device=device,
                        requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM),
                        dtype=dtype,
                        device=device,
                        requires_grad=True)
        if mode == "fwd" and "fp8" in provider:
            q = q.to(torch.float8_e5m2)
            k = k.to(torch.float8_e5m2)
            v = v.permute(0, 1, 3, 2)
            v = v.to(torch.float8_e5m2)
        sm_scale = 1.3
        def fn(): return attention(q, k, v, causal, sm_scale)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            def fn(): return o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "flash":
        qkv = torch.randn((BATCH, N_CTX, 3, H, HEAD_DIM),
                          dtype=dtype,
                          device=device,
                          requires_grad=True)

        def fn(): return flash_attn_func(qkv, causal=causal)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            def fn(): return o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops / ms * 1e-9


if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    bench_flash_attention.run(save_path=".", print_data=True)
