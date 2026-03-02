import torch
import torch.nn.functional as F

from nanochat.gpt import chunked_linear_cross_entropy


def _baseline_softcapped_ce(h, w, y, softcap=15.0, reduction="mean"):
    logits = h @ w.t()
    logits = softcap * torch.tanh(logits / softcap)
    return F.cross_entropy(logits, y, ignore_index=-1, reduction=reduction)


def test_chunked_ce_matches_baseline_loss_and_grads():
    torch.manual_seed(1234)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    n, d, v = 32, 48, 257
    h0 = torch.randn(n, d, device=device, dtype=dtype)
    w0 = torch.randn(v, d, device=device, dtype=torch.float32)
    y = torch.randint(0, v, (n,), device=device, dtype=torch.long)
    y[::7] = -1

    # Baseline
    h_base = h0.clone().detach().requires_grad_(True)
    w_base = w0.clone().detach().requires_grad_(True)
    loss_base = _baseline_softcapped_ce(h_base, w_base, y, reduction="mean")
    loss_base.backward()
    grad_h_base = h_base.grad.detach().float()
    grad_w_base = w_base.grad.detach().float()

    # Chunked
    h_chunk = h0.clone().detach().requires_grad_(True)
    w_chunk = w0.clone().detach().requires_grad_(True)
    loss_chunk = chunked_linear_cross_entropy(
        h_chunk, w_chunk, y, softcap=15.0, chunk_size=64, reduction="mean"
    )
    loss_chunk.backward()
    grad_h_chunk = h_chunk.grad.detach().float()
    grad_w_chunk = w_chunk.grad.detach().float()

    assert torch.allclose(loss_chunk.float(), loss_base.float(), rtol=1e-4, atol=2e-4)
    assert torch.allclose(grad_h_chunk, grad_h_base, rtol=2e-3, atol=2e-3)
    assert torch.allclose(grad_w_chunk, grad_w_base, rtol=2e-3, atol=2e-3)


def test_chunked_ce_compile_runs_multiple_steps():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    n, d, v = 16, 32, 128
    h = torch.randn(n, d, device=device, dtype=dtype, requires_grad=True)
    w = torch.randn(v, d, device=device, dtype=torch.float32, requires_grad=True)
    y = torch.randint(0, v, (n,), device=device, dtype=torch.long)

    def step(hh, ww, yy):
        loss = chunked_linear_cross_entropy(hh, ww, yy, chunk_size=32, reduction="mean")
        loss.backward()
        return loss

    compiled = torch.compile(step, dynamic=False)
    losses = []
    for _ in range(6):
        if h.grad is not None:
            h.grad.zero_()
        if w.grad is not None:
            w.grad.zero_()
        losses.append(float(compiled(h, w, y).detach()))

    assert all(torch.isfinite(torch.tensor(losses)))


def test_chunked_ce_all_ignore_targets_mean_is_nan_and_zero_grads():
    torch.manual_seed(7)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    n, d, v = 12, 24, 64
    h = torch.randn(n, d, device=device, dtype=dtype, requires_grad=True)
    w = torch.randn(v, d, device=device, dtype=torch.float32, requires_grad=True)
    y = torch.full((n,), -1, device=device, dtype=torch.long)

    loss = chunked_linear_cross_entropy(h, w, y, chunk_size=16, reduction="mean")
    assert torch.isnan(loss)

    loss.backward()
    assert torch.allclose(h.grad.float(), torch.zeros_like(h.grad.float()))
    assert torch.allclose(w.grad.float(), torch.zeros_like(w.grad.float()))
