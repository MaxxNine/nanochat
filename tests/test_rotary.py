"""
Test rotary embeddings and fused RMSNorm+Rotary — Triton kernel vs pure PyTorch reference.

Run: python -m pytest tests/test_rotary.py -v -s

IMPORTANT: The fused norm+rotary backward uses inverse rotation (R^T) to recover
x_norm from the saved output. This requires cos²+sin²=1, which is always true for
real rotary embeddings but NOT for random cos/sin. All tests that exercise backward
must use real cos/sin computed from actual angles.
"""
import torch
import torch.nn.functional as F
import pytest


# Reference implementations
def _ref_rotary(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def _ref_norm_rotary(x, cos, sin):
    x_norm = F.rms_norm(x, (x.size(-1),))
    return _ref_rotary(x_norm, cos, sin)


def _make_inputs(B, T, H, D, device, dtype):
    """Create random x with REAL cos/sin (from actual rotary angles, cos²+sin²=1)."""
    x = torch.randn(B, T, H, D, device=device, dtype=dtype)
    half_d = D // 2
    # Compute real rotary embeddings (same as GPT._precompute_rotary_embeddings)
    channel_range = torch.arange(0, D, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (10000.0 ** (channel_range / D))
    t = torch.arange(T, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(dtype)[None, :, None, :]  # (1, T, 1, D//2)
    sin = freqs.sin().to(dtype)[None, :, None, :]
    return x, cos, sin


# =============================================================================
# Standalone rotary tests
# =============================================================================
class TestRotaryForward:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    @pytest.mark.parametrize("B,T,H,D", [
        (2, 64, 6, 128), (1, 128, 12, 64), (4, 32, 8, 128), (1, 1, 6, 128),
    ])
    def test_forward_shapes(self, B, T, H, D):
        from nanochat.rotary import apply_rotary_emb
        x, cos, sin = _make_inputs(B, T, H, D, self.DEVICE, self.DTYPE)
        result = apply_rotary_emb(x, cos, sin)
        expected = _ref_rotary(x, cos, sin)
        assert result.shape == expected.shape
        torch.testing.assert_close(result, expected, atol=2e-2, rtol=2e-2)

    def test_forward_gqa(self):
        from nanochat.rotary import apply_rotary_emb
        B, T, D = 2, 64, 128
        q, cos, sin = _make_inputs(B, T, 12, D, self.DEVICE, self.DTYPE)
        k, _, _ = _make_inputs(B, T, 3, D, self.DEVICE, self.DTYPE)
        torch.testing.assert_close(apply_rotary_emb(q, cos, sin), _ref_rotary(q, cos, sin), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(apply_rotary_emb(k, cos, sin), _ref_rotary(k, cos, sin), atol=2e-2, rtol=2e-2)


class TestRotaryBackward:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float32

    @pytest.mark.parametrize("B,T,H,D", [(2, 32, 6, 128), (1, 64, 4, 64)])
    def test_backward_gradients(self, B, T, H, D):
        from nanochat.rotary import apply_rotary_emb
        x_data, cos, sin = _make_inputs(B, T, H, D, self.DEVICE, self.DTYPE)
        x1 = x_data.clone().requires_grad_(True)
        apply_rotary_emb(x1, cos, sin).sum().backward()
        x2 = x_data.clone().requires_grad_(True)
        _ref_rotary(x2, cos, sin).sum().backward()
        torch.testing.assert_close(x1.grad, x2.grad, atol=1e-5, rtol=1e-5)


class TestRotaryCPUFallback:
    def test_cpu_forward(self):
        from nanochat.rotary import apply_rotary_emb
        x, cos, sin = _make_inputs(2, 32, 4, 64, "cpu", torch.float32)
        torch.testing.assert_close(apply_rotary_emb(x, cos, sin), _ref_rotary(x, cos, sin), atol=1e-6, rtol=1e-6)

    def test_cpu_backward(self):
        from nanochat.rotary import apply_rotary_emb
        x_data, cos, sin = _make_inputs(2, 16, 4, 64, "cpu", torch.float32)
        x1 = x_data.clone().requires_grad_(True)
        apply_rotary_emb(x1, cos, sin).sum().backward()
        x2 = x_data.clone().requires_grad_(True)
        _ref_rotary(x2, cos, sin).sum().backward()
        torch.testing.assert_close(x1.grad, x2.grad, atol=1e-6, rtol=1e-6)


# =============================================================================
# Fused RMSNorm + Rotary tests
# =============================================================================
class TestFusedNormRotaryForward:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    @pytest.mark.parametrize("B,T,H,D", [
        (2, 64, 6, 128), (1, 128, 12, 64), (4, 32, 8, 128), (1, 1, 6, 128),
    ])
    def test_forward_shapes(self, B, T, H, D):
        from nanochat.rotary import fused_norm_rotary
        x, cos, sin = _make_inputs(B, T, H, D, self.DEVICE, self.DTYPE)
        result = fused_norm_rotary(x, cos, sin)
        expected = _ref_norm_rotary(x, cos, sin)
        assert result.shape == expected.shape
        torch.testing.assert_close(result, expected, atol=3e-2, rtol=3e-2)

    def test_forward_gqa(self):
        from nanochat.rotary import fused_norm_rotary
        B, T, D = 2, 64, 128
        q, cos, sin = _make_inputs(B, T, 12, D, self.DEVICE, self.DTYPE)
        k, _, _ = _make_inputs(B, T, 3, D, self.DEVICE, self.DTYPE)
        torch.testing.assert_close(fused_norm_rotary(q, cos, sin), _ref_norm_rotary(q, cos, sin), atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(fused_norm_rotary(k, cos, sin), _ref_norm_rotary(k, cos, sin), atol=3e-2, rtol=3e-2)


class TestFusedNormRotaryBackward:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float32

    @pytest.mark.parametrize("B,T,H,D", [(2, 32, 6, 128), (1, 64, 4, 64)])
    def test_backward_gradients(self, B, T, H, D):
        from nanochat.rotary import fused_norm_rotary
        x_data, cos, sin = _make_inputs(B, T, H, D, self.DEVICE, self.DTYPE)
        x1 = x_data.clone().requires_grad_(True)
        fused_norm_rotary(x1, cos, sin).sum().backward()
        x2 = x_data.clone().requires_grad_(True)
        _ref_norm_rotary(x2, cos, sin).sum().backward()
        torch.testing.assert_close(x1.grad, x2.grad, atol=1e-4, rtol=1e-4)

    def test_backward_with_upstream_grad(self):
        from nanochat.rotary import fused_norm_rotary
        B, T, H, D = 2, 32, 4, 64
        x_data, cos, sin = _make_inputs(B, T, H, D, self.DEVICE, self.DTYPE)
        upstream = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        x1 = x_data.clone().requires_grad_(True)
        fused_norm_rotary(x1, cos, sin).backward(upstream)
        x2 = x_data.clone().requires_grad_(True)
        _ref_norm_rotary(x2, cos, sin).backward(upstream)
        torch.testing.assert_close(x1.grad, x2.grad, atol=1e-4, rtol=1e-4)


class TestFusedNormRotaryCPUFallback:
    def test_cpu_forward(self):
        from nanochat.rotary import fused_norm_rotary
        x, cos, sin = _make_inputs(2, 32, 4, 64, "cpu", torch.float32)
        torch.testing.assert_close(fused_norm_rotary(x, cos, sin), _ref_norm_rotary(x, cos, sin), atol=1e-5, rtol=1e-5)

    def test_cpu_backward(self):
        from nanochat.rotary import fused_norm_rotary
        x_data, cos, sin = _make_inputs(2, 16, 4, 64, "cpu", torch.float32)
        x1 = x_data.clone().requires_grad_(True)
        fused_norm_rotary(x1, cos, sin).sum().backward()
        x2 = x_data.clone().requires_grad_(True)
        _ref_norm_rotary(x2, cos, sin).sum().backward()
        torch.testing.assert_close(x1.grad, x2.grad, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    try:
        import triton
        print(f"Triton version: {triton.__version__}")
    except ImportError:
        print("Triton: not available")
    print()
    pytest.main([__file__, "-v", "-s"])
