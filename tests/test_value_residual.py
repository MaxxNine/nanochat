import torch

from nanochat.gpt import fused_value_residual


def _baseline_value_residual(v, ve, x_slice, w):
    gate = 2.0 * torch.sigmoid(torch.nn.functional.linear(x_slice, w))
    return v + gate.unsqueeze(-1) * ve


def test_fused_value_residual_matches_baseline_forward_and_backward():
    torch.manual_seed(42)
    B, T = 2, 16
    n_kv_head, head_dim = 4, 8
    gate_channels = 6

    v0 = torch.randn(B, T, n_kv_head, head_dim, dtype=torch.float32)
    ve0 = torch.randn(B, T, n_kv_head, head_dim, dtype=torch.float32)
    x0 = torch.randn(B, T, gate_channels, dtype=torch.float32)
    w0 = torch.randn(n_kv_head, gate_channels, dtype=torch.float32)
    grad_out = torch.randn(B, T, n_kv_head, head_dim, dtype=torch.float32)

    # Baseline
    v = v0.clone().requires_grad_(True)
    ve = ve0.clone().requires_grad_(True)
    x = x0.clone().requires_grad_(True)
    w = w0.clone().requires_grad_(True)
    y_ref = _baseline_value_residual(v, ve, x, w)
    y_ref.backward(grad_out)
    grads_ref = (v.grad.clone(), ve.grad.clone(), x.grad.clone(), w.grad.clone())

    # Fused custom autograd
    v2 = v0.clone().requires_grad_(True)
    ve2 = ve0.clone().requires_grad_(True)
    x2 = x0.clone().requires_grad_(True)
    w2 = w0.clone().requires_grad_(True)
    y_fused = fused_value_residual(v2, ve2, x2, w2)
    y_fused.backward(grad_out)
    grads_fused = (v2.grad.clone(), ve2.grad.clone(), x2.grad.clone(), w2.grad.clone())

    # Forward
    assert torch.allclose(y_fused, y_ref, rtol=1e-5, atol=1e-6)
    # Backward
    for g_fused, g_ref in zip(grads_fused, grads_ref):
        assert torch.allclose(g_fused, g_ref, rtol=1e-5, atol=1e-6)

