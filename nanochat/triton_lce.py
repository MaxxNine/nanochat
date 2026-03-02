"""
Triton kernels for memory-efficient Linear + Cross Entropy with softcapped logits.

This module intentionally focuses on bounded-workspace kernels:
- Forward computes per-row LSE without materializing full logits.
- Backward computes dH/dW directly from tiles with atomic accumulations.
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False

_WARNED_FALLBACK = False


def _warn_once(msg: str) -> None:
    global _WARNED_FALLBACK
    if not _WARNED_FALLBACK:
        _WARNED_FALLBACK = True
        print(f"WARNING: {msg}")


def triton_lce_available() -> bool:
    if not _HAS_TRITON:
        return False
    return torch.cuda.is_available()


def triton_lce_use_forward() -> bool:
    return os.environ.get("NANOCHAT_TLCE_USE_FWD", "1") == "1"


def triton_lce_use_backward() -> bool:
    # Current backward kernel uses atomics heavily and may be slower on some setups.
    return os.environ.get("NANOCHAT_TLCE_USE_BWD", "0") == "1"


def _can_use_tlce_tensors(h: torch.Tensor, w: torch.Tensor) -> bool:
    if not triton_lce_available():
        return False
    if (not h.is_cuda) or (not w.is_cuda):
        return False
    if h.ndim != 2 or w.ndim != 2:
        return False
    if h.size(1) != w.size(1):
        return False
    if h.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if w.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    # Triton pointer math below assumes dense inner dimension.
    if h.stride(1) != 1:
        return False
    if w.stride(1) != 1:
        return False
    return True


if _HAS_TRITON:
    @triton.jit
    def _stable_tanh(x):
        # Compatible tanh for Triton builds that don't expose tl.tanh.
        # Clamp first to avoid overflow in exp while preserving saturated behavior.
        x = tl.maximum(tl.minimum(x, 10.0), -10.0)
        e2x = tl.exp(2.0 * x)
        return (e2x - 1.0) / (e2x + 1.0)


    @triton.jit
    def _lce_lse_kernel(
        H_ptr, W_ptr, LSE_ptr,
        N, V, D,
        stride_hn, stride_hd,
        stride_wn, stride_wd,
        stride_lse,
        softcap,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < N

        m = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        s = tl.zeros((BLOCK_M,), tl.float32)

        n_start = 0
        while n_start < V:
            offs_n = n_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < V
            acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

            k_start = 0
            while k_start < D:
                offs_k = k_start + tl.arange(0, BLOCK_K)
                mask_k = offs_k < D
                h_ptrs = H_ptr + offs_m[:, None] * stride_hn + offs_k[None, :] * stride_hd
                w_ptrs = W_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wd
                h = tl.load(h_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0)
                w = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0)
                # tl.dot requires both operands to have the same dtype.
                w = w.to(h.dtype)
                acc += tl.dot(h, tl.trans(w))
                k_start += BLOCK_K

            z = softcap * _stable_tanh(acc / softcap)
            z = tl.where(mask_n[None, :], z, -float("inf"))
            z_max = tl.max(z, axis=1)
            new_m = tl.maximum(m, z_max)
            s = tl.exp(m - new_m) * s + tl.sum(tl.exp(z - new_m[:, None]), axis=1)
            m = new_m
            n_start += BLOCK_N

        out = m + tl.log(s)
        out_ptrs = LSE_ptr + offs_m * stride_lse
        tl.store(out_ptrs, out, mask=mask_m)


    @triton.jit
    def _lce_backward_kernel(
        H_ptr, W_ptr, Y_ptr, LSE_ptr, COEFF_ptr,
        GH_ptr, GW_ptr,
        N, V, D,
        stride_hn, stride_hd,
        stride_wn, stride_wd,
        stride_y,
        stride_lse,
        stride_coeff,
        stride_ghn, stride_ghd,
        stride_gwn, stride_gwd,
        softcap,
        HAS_GH: tl.constexpr,
        HAS_GW: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        grid_n = tl.cdiv(V, BLOCK_N)
        pid_m = pid // grid_n
        pid_n = pid % grid_n

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < N
        mask_n = offs_n < V

        # Build tile logits a[m, n] = h[m, :] @ w[n, :].T
        a = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        k_start = 0
        while k_start < D:
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < D
            h_ptrs = H_ptr + offs_m[:, None] * stride_hn + offs_k[None, :] * stride_hd
            w_ptrs = W_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wd
            h = tl.load(h_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0)
            w = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0)
            w = w.to(h.dtype)
            a += tl.dot(h, tl.trans(w))
            k_start += BLOCK_K

        tanh_u = _stable_tanh(a / softcap)
        dz_da = 1.0 - tanh_u * tanh_u
        lse = tl.load(LSE_ptr + offs_m * stride_lse, mask=mask_m, other=0.0)
        coeff = tl.load(COEFF_ptr + offs_m * stride_coeff, mask=mask_m, other=0.0)
        p = tl.exp((softcap * tanh_u) - lse[:, None])
        g = p * dz_da * coeff[:, None]

        # Target correction: subtract dz/da*coeff at the target class.
        y = tl.load(Y_ptr + offs_m * stride_y, mask=mask_m, other=0)
        hit = (y[:, None] == offs_n[None, :]) & mask_n[None, :]
        g -= tl.where(hit, dz_da * coeff[:, None], 0.0)
        g = tl.where(mask_m[:, None] & mask_n[None, :], g, 0.0)

        k_start = 0
        while k_start < D:
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < D
            h_ptrs = H_ptr + offs_m[:, None] * stride_hn + offs_k[None, :] * stride_hd
            w_ptrs = W_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wd
            h = tl.load(h_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0)
            w = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0)
            w = w.to(h.dtype)
            g_mm = g.to(h.dtype)

            if HAS_GH:
                gh_tile = tl.dot(g_mm, w)
                gh_ptrs = GH_ptr + offs_m[:, None] * stride_ghn + offs_k[None, :] * stride_ghd
                tl.atomic_add(gh_ptrs, gh_tile, mask=mask_m[:, None] & mask_k[None, :])
            if HAS_GW:
                gw_tile = tl.dot(tl.trans(g_mm), h)
                gw_ptrs = GW_ptr + offs_n[:, None] * stride_gwn + offs_k[None, :] * stride_gwd
                tl.atomic_add(gw_ptrs, gw_tile, mask=mask_n[:, None] & mask_k[None, :])
            k_start += BLOCK_K


def _default_block_m() -> int:
    return int(os.environ.get("NANOCHAT_TLCE_BLOCK_M", "16"))


def _default_block_n() -> int:
    return int(os.environ.get("NANOCHAT_TLCE_BLOCK_N", "64"))


def _default_block_k() -> int:
    return int(os.environ.get("NANOCHAT_TLCE_BLOCK_K", "32"))


def _default_num_warps() -> int:
    return int(os.environ.get("NANOCHAT_TLCE_NUM_WARPS", "4"))


def lse_forward_triton(h: torch.Tensor, w: torch.Tensor, softcap: float) -> torch.Tensor | None:
    """
    Return per-row LSE using Triton kernel, or None when unsupported.
    """
    if (not triton_lce_use_forward()) or (not _can_use_tlce_tensors(h, w)):
        return None
    try:
        h_ = h.contiguous()
        w_ = w.contiguous()
        n, d = h_.shape
        v = w_.shape[0]
        lse = torch.empty((n,), dtype=torch.float32, device=h_.device)
        bm = _default_block_m()
        bn = _default_block_n()
        bk = _default_block_k()
        grid = (triton.cdiv(n, bm),)
        _lce_lse_kernel[grid](
            h_, w_, lse,
            n, v, d,
            h_.stride(0), h_.stride(1),
            w_.stride(0), w_.stride(1),
            lse.stride(0),
            float(softcap),
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
            num_warps=_default_num_warps(),
            num_stages=2,
        )
        return lse
    except Exception as e:
        _warn_once(f"Triton LSE kernel failed ({type(e).__name__}: {e}); falling back to PyTorch path.")
        return None


def backward_triton(
    h: torch.Tensor,
    w: torch.Tensor,
    y_safe: torch.Tensor,
    lse: torch.Tensor,
    coeff: torch.Tensor,
    softcap: float,
    needs_h: bool,
    needs_w: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    Return (grad_h32, grad_w32) from Triton backward kernel, or (None, None) when unsupported.
    """
    if (not needs_h) and (not needs_w):
        return None, None
    if not triton_lce_use_backward():
        return None, None
    if not _can_use_tlce_tensors(h, w):
        return None, None
    if y_safe.dtype != torch.long:
        return None, None
    if lse.dtype != torch.float32 or coeff.dtype != torch.float32:
        return None, None
    try:
        h_ = h.contiguous()
        w_ = w.contiguous()
        y_ = y_safe.contiguous()
        lse_ = lse.contiguous()
        coeff_ = coeff.contiguous()

        n, d = h_.shape
        v = w_.shape[0]
        gh = torch.zeros((n, d), dtype=torch.float32, device=h_.device) if needs_h else torch.empty((1, 1), dtype=torch.float32, device=h_.device)
        gw = torch.zeros((v, d), dtype=torch.float32, device=w_.device) if needs_w else torch.empty((1, 1), dtype=torch.float32, device=w_.device)

        bm = _default_block_m()
        bn = _default_block_n()
        bk = _default_block_k()
        grid = (triton.cdiv(n, bm) * triton.cdiv(v, bn),)
        _lce_backward_kernel[grid](
            h_, w_, y_, lse_, coeff_,
            gh, gw,
            n, v, d,
            h_.stride(0), h_.stride(1),
            w_.stride(0), w_.stride(1),
            y_.stride(0),
            lse_.stride(0),
            coeff_.stride(0),
            gh.stride(0), gh.stride(1),
            gw.stride(0), gw.stride(1),
            float(softcap),
            HAS_GH=needs_h,
            HAS_GW=needs_w,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
            num_warps=_default_num_warps(),
            num_stages=2,
        )
        return (gh if needs_h else None), (gw if needs_w else None)
    except Exception as e:
        _warn_once(f"Triton backward kernel failed ({type(e).__name__}: {e}); falling back to PyTorch path.")
        return None, None
