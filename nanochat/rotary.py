"""
Fused rotary embeddings and fused RMSNorm+Rotary via Triton kernels.

Two public APIs:
  - apply_rotary_emb(x, cos, sin)       — standalone rotary (eager Triton / compiled PyTorch)
  - fused_norm_rotary(x, cos, sin)      — RMSNorm + rotary in one pass (works inside torch.compile)

The fused kernel is the main win: it combines a reduction (RMSNorm) with element-wise
rotation, which torch.compile's inductor cannot auto-fuse. This halves the number of
kernel launches (4 → 2) and memory passes for the QK-norm + rotary path.
"""

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False


def _is_real_tensor(x):
    """Return True if x has real storage (False for FakeTensor during torch.compile tracing)."""
    try:
        x.data_ptr()
        return True
    except RuntimeError:
        return False


# =============================================================================
# Standalone rotary kernel (unchanged from before)
# =============================================================================
if _HAS_TRITON:
    @triton.jit
    def _rotary_kernel(
        X_ptr, COS_ptr, SIN_ptr,
        stride_xb, stride_xt, stride_xh, stride_xd,
        stride_ct, stride_cd,
        seq_len, n_heads,
        HALF_D: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        head = pid % n_heads
        tmp = pid // n_heads
        seq = tmp % seq_len
        batch = tmp // seq_len
        x_base = batch * stride_xb + seq * stride_xt + head * stride_xh
        cs_base = seq * stride_ct
        cols = tl.arange(0, BLOCK_D)
        mask = cols < HALF_D
        x1 = tl.load(X_ptr + x_base + cols * stride_xd, mask=mask, other=0.0)
        x2 = tl.load(X_ptr + x_base + (cols + HALF_D) * stride_xd, mask=mask, other=0.0)
        cos = tl.load(COS_ptr + cs_base + cols * stride_cd, mask=mask, other=0.0)
        sin = tl.load(SIN_ptr + cs_base + cols * stride_cd, mask=mask, other=0.0)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        tl.store(X_ptr + x_base + cols * stride_xd, y1, mask=mask)
        tl.store(X_ptr + x_base + (cols + HALF_D) * stride_xd, y2, mask=mask)


# =============================================================================
# Fused RMSNorm + Rotary kernels
# =============================================================================
if _HAS_TRITON:
    @triton.jit
    def _fused_norm_rotary_fwd_kernel(
        X_ptr, OUT_ptr, RMS_ptr, COS_ptr, SIN_ptr,
        stride_xb, stride_xt, stride_xh, stride_xd,
        stride_ob, stride_ot, stride_oh, stride_od,
        stride_rb, stride_rt, stride_rh,
        stride_ct, stride_cd,
        seq_len, n_heads,
        EPS: tl.constexpr, D: tl.constexpr, HALF_D: tl.constexpr,
        BLOCK_HD: tl.constexpr,
    ):
        """Fused RMSNorm + rotary forward. One program per (batch, seq, head)."""
        pid = tl.program_id(0)
        head = pid % n_heads
        tmp = pid // n_heads
        seq = tmp % seq_len
        batch = tmp // seq_len

        x_base = batch * stride_xb + seq * stride_xt + head * stride_xh
        o_base = batch * stride_ob + seq * stride_ot + head * stride_oh
        r_off  = batch * stride_rb + seq * stride_rt + head * stride_rh
        cs_base = seq * stride_ct

        cols = tl.arange(0, BLOCK_HD)
        hmask = cols < HALF_D

        # Load both halves of head dim in float32 for precision
        x1 = tl.load(X_ptr + x_base + cols * stride_xd, mask=hmask, other=0.0).to(tl.float32)
        x2 = tl.load(X_ptr + x_base + (cols + HALF_D) * stride_xd, mask=hmask, other=0.0).to(tl.float32)

        # RMSNorm: variance over all D elements
        var = (tl.sum(x1 * x1, axis=0) + tl.sum(x2 * x2, axis=0)) / D
        rms = tl.sqrt(var + EPS)
        x1n = x1 / rms
        x2n = x2 / rms

        # Rotary
        cos_v = tl.load(COS_ptr + cs_base + cols * stride_cd, mask=hmask, other=0.0).to(tl.float32)
        sin_v = tl.load(SIN_ptr + cs_base + cols * stride_cd, mask=hmask, other=0.0).to(tl.float32)
        y1 = x1n * cos_v + x2n * sin_v
        y2 = x1n * (-sin_v) + x2n * cos_v

        # Store output (cast back to input dtype) + rms scalar
        tl.store(OUT_ptr + o_base + cols * stride_od, y1.to(x1.dtype), mask=hmask)
        tl.store(OUT_ptr + o_base + (cols + HALF_D) * stride_od, y2.to(x1.dtype), mask=hmask)
        tl.store(RMS_ptr + r_off, rms)

    @triton.jit
    def _fused_norm_rotary_bwd_kernel(
        GRAD_OUT_ptr, OUT_ptr, RMS_ptr, COS_ptr, SIN_ptr, GRAD_X_ptr,
        stride_gb, stride_gt, stride_gh, stride_gd,
        stride_ob, stride_ot, stride_oh, stride_od,
        stride_rb, stride_rt, stride_rh,
        stride_ct, stride_cd,
        stride_dxb, stride_dxt, stride_dxh, stride_dxd,
        seq_len, n_heads,
        D: tl.constexpr, HALF_D: tl.constexpr, BLOCK_HD: tl.constexpr,
    ):
        """Fused RMSNorm + rotary backward. One program per (batch, seq, head)."""
        pid = tl.program_id(0)
        head = pid % n_heads
        tmp = pid // n_heads
        seq = tmp % seq_len
        batch = tmp // seq_len

        g_base  = batch * stride_gb + seq * stride_gt + head * stride_gh
        o_base  = batch * stride_ob + seq * stride_ot + head * stride_oh
        r_off   = batch * stride_rb + seq * stride_rt + head * stride_rh
        cs_base = seq * stride_ct
        dx_base = batch * stride_dxb + seq * stride_dxt + head * stride_dxh

        cols = tl.arange(0, BLOCK_HD)
        hmask = cols < HALF_D

        # Load grad_output, saved output, cos, sin — all in float32
        g1 = tl.load(GRAD_OUT_ptr + g_base + cols * stride_gd, mask=hmask, other=0.0).to(tl.float32)
        g2 = tl.load(GRAD_OUT_ptr + g_base + (cols + HALF_D) * stride_gd, mask=hmask, other=0.0).to(tl.float32)
        y1 = tl.load(OUT_ptr + o_base + cols * stride_od, mask=hmask, other=0.0).to(tl.float32)
        y2 = tl.load(OUT_ptr + o_base + (cols + HALF_D) * stride_od, mask=hmask, other=0.0).to(tl.float32)
        cos_v = tl.load(COS_ptr + cs_base + cols * stride_cd, mask=hmask, other=0.0).to(tl.float32)
        sin_v = tl.load(SIN_ptr + cs_base + cols * stride_cd, mask=hmask, other=0.0).to(tl.float32)
        rms = tl.load(RMS_ptr + r_off)

        # Inverse rotary: recover grad_x_norm and x_norm from their rotated versions
        gxn1 = g1 * cos_v - g2 * sin_v
        gxn2 = g1 * sin_v + g2 * cos_v
        xn1 = y1 * cos_v - y2 * sin_v
        xn2 = y1 * sin_v + y2 * cos_v

        # RMSNorm backward: grad_x = (1/rms) * (grad_x_norm - x_norm * mean(grad_x_norm · x_norm))
        c = (tl.sum(gxn1 * xn1, axis=0) + tl.sum(gxn2 * xn2, axis=0)) / D
        dx1 = (gxn1 - xn1 * c) / rms
        dx2 = (gxn2 - xn2 * c) / rms

        # Store grad_x (cast back)
        tl.store(GRAD_X_ptr + dx_base + cols * stride_dxd, dx1.to(g1.dtype), mask=hmask)
        tl.store(GRAD_X_ptr + dx_base + (cols + HALF_D) * stride_dxd, dx2.to(g1.dtype), mask=hmask)


# =============================================================================
# Launch helpers
# =============================================================================

def _launch_rotary_kernel(x, cos, sin):
    """Launch standalone rotary kernel on x (in-place)."""
    B, T, H, D = x.shape
    half_d = D // 2
    BLOCK_D = triton.next_power_of_2(half_d)
    cos_2d = cos.view(T, half_d)
    sin_2d = sin.view(T, half_d)
    grid = (B * T * H,)
    _rotary_kernel[grid](
        x, cos_2d, sin_2d,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        cos_2d.stride(0), cos_2d.stride(1),
        T, H, HALF_D=half_d, BLOCK_D=BLOCK_D,
        num_warps=min(4, max(1, BLOCK_D // 32)), num_stages=1,
    )


def _launch_fused_fwd(x, cos, sin, out, rms):
    """Launch fused norm+rotary forward kernel."""
    B, T, H, D = x.shape
    half_d = D // 2
    BLOCK_HD = triton.next_power_of_2(half_d)
    cos_2d = cos.view(T, half_d)
    sin_2d = sin.view(T, half_d)
    eps = torch.finfo(x.dtype).eps
    grid = (B * T * H,)
    _fused_norm_rotary_fwd_kernel[grid](
        x, out, rms, cos_2d, sin_2d,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        rms.stride(0), rms.stride(1), rms.stride(2),
        cos_2d.stride(0), cos_2d.stride(1),
        T, H, EPS=eps, D=D, HALF_D=half_d, BLOCK_HD=BLOCK_HD,
        num_warps=min(4, max(1, BLOCK_HD // 32)), num_stages=1,
    )


def _launch_fused_bwd(grad_output, out, rms, cos, sin, grad_x):
    """Launch fused norm+rotary backward kernel."""
    B, T, H, D = grad_output.shape
    half_d = D // 2
    BLOCK_HD = triton.next_power_of_2(half_d)
    cos_2d = cos.view(T, half_d)
    sin_2d = sin.view(T, half_d)
    grid = (B * T * H,)
    _fused_norm_rotary_bwd_kernel[grid](
        grad_output, out, rms, cos_2d, sin_2d, grad_x,
        grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        rms.stride(0), rms.stride(1), rms.stride(2),
        cos_2d.stride(0), cos_2d.stride(1),
        grad_x.stride(0), grad_x.stride(1), grad_x.stride(2), grad_x.stride(3),
        T, H, D=D, HALF_D=half_d, BLOCK_HD=BLOCK_HD,
        num_warps=min(4, max(1, BLOCK_HD // 32)), num_stages=1,
    )


# =============================================================================
# Autograd wrappers
# =============================================================================

class _ApplyRotaryEmb(torch.autograd.Function):
    """Standalone rotary via Triton (eager mode only)."""
    @staticmethod
    def forward(ctx, x, cos, sin):
        ctx.save_for_backward(cos, sin)
        out = x.clone()
        _launch_rotary_kernel(out, cos, sin)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        cos, sin = ctx.saved_tensors
        grad_x = grad_output.clone()
        _launch_rotary_kernel(grad_x, cos, -sin)
        return grad_x, None, None


@torch._dynamo.allow_in_graph
class _FusedNormRotary(torch.autograd.Function):
    """Fused RMSNorm + rotary embedding. Works inside torch.compile via FakeTensor fallback."""

    @staticmethod
    def forward(ctx, x, cos, sin):
        B, T, H, D = x.shape
        out = torch.empty_like(x)
        rms = torch.empty(B, T, H, device=x.device, dtype=torch.float32)

        if _is_real_tensor(x) and x.is_cuda and _HAS_TRITON:
            _launch_fused_fwd(x, cos, sin, out, rms)
        else:
            # FakeTensor (torch.compile tracing) or CPU — use PyTorch ops
            eps = torch.finfo(x.dtype).eps
            x_f = x.float()
            variance = x_f.pow(2).mean(dim=-1)
            rms = (variance + eps).sqrt()
            x_norm = x_f / rms.unsqueeze(-1)
            d = D // 2
            x1, x2 = x_norm[..., :d], x_norm[..., d:]
            cos_f, sin_f = cos.float(), sin.float()
            y1 = x1 * cos_f + x2 * sin_f
            y2 = x1 * (-sin_f) + x2 * cos_f
            out = torch.cat([y1, y2], dim=-1).to(x.dtype)

        ctx.save_for_backward(out, rms, cos, sin)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, rms, cos, sin = ctx.saved_tensors
        grad_x = torch.empty_like(grad_output)

        if _is_real_tensor(grad_output) and grad_output.is_cuda and _HAS_TRITON:
            _launch_fused_bwd(grad_output, out, rms, cos, sin, grad_x)
        else:
            # PyTorch fallback backward
            D = grad_output.shape[-1]
            d = D // 2
            g = grad_output.float()
            y = out.float()
            cos_f, sin_f = cos.float(), sin.float()
            # Inverse rotary
            g1, g2 = g[..., :d], g[..., d:]
            y1, y2 = y[..., :d], y[..., d:]
            gxn1 = g1 * cos_f - g2 * sin_f
            gxn2 = g1 * sin_f + g2 * cos_f
            xn1 = y1 * cos_f - y2 * sin_f
            xn2 = y1 * sin_f + y2 * cos_f
            # RMSNorm backward
            gxn = torch.cat([gxn1, gxn2], dim=-1)
            xn = torch.cat([xn1, xn2], dim=-1)
            c = (gxn * xn).mean(dim=-1, keepdim=True)
            grad_x = ((gxn - xn * c) / rms.unsqueeze(-1)).to(grad_output.dtype)

        return grad_x, None, None


# =============================================================================
# Pure PyTorch fallbacks
# =============================================================================

def _apply_rotary_emb_torch(x, cos, sin):
    """Pure PyTorch rotary embedding."""
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def _fused_norm_rotary_torch(x, cos, sin):
    """Pure PyTorch RMSNorm + rotary (CPU/no-Triton fallback)."""
    x_norm = F.rms_norm(x, (x.size(-1),))
    return _apply_rotary_emb_torch(x_norm, cos, sin)


# =============================================================================
# Public API
# =============================================================================

def _can_use_triton(x):
    if not _HAS_TRITON:
        return False
    if not x.is_cuda:
        return False
    if x.ndim != 4:
        return False
    if x.shape[3] % 2 != 0:
        return False
    return True


def apply_rotary_emb(x, cos, sin):
    """Standalone rotary positional embeddings (eager Triton / compiled PyTorch)."""
    if _can_use_triton(x) and not torch.compiler.is_compiling():
        return _ApplyRotaryEmb.apply(x, cos, sin)
    return _apply_rotary_emb_torch(x, cos, sin)


def fused_norm_rotary(x, cos, sin):
    """
    Fused RMSNorm + rotary embedding in a single pass.

    Replaces the two-step ``norm(x)`` then ``apply_rotary_emb(x, cos, sin)``
    with one fused Triton kernel that does both in a single memory pass.
    This works inside torch.compile — the kernel is opaque to inductor but
    fuses a reduction + element-wise ops that inductor can't auto-fuse.

    Args:
        x:   (B, T, H, D) tensor — queries or keys (pre-norm)
        cos: (1, T, 1, D//2) precomputed cosines
        sin: (1, T, 1, D//2) precomputed sines

    Returns:
        Normalized + rotated tensor, same shape as x.
    """
    if _can_use_triton(x):
        return _FusedNormRotary.apply(x, cos, sin)
    return _fused_norm_rotary_torch(x, cos, sin)
