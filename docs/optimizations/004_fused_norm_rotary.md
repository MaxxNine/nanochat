# Optimization 004: Fused QK-Norm + Rotary Embedding Kernel

Date: 2026-03-02
Status: Implemented and measured

## Objective

Fuse QK-norm (RMSNorm) and rotary embedding into a single Triton kernel, reducing kernel launches from 4 to 2 and eliminating intermediate activation memory. This fusion targets a reduction + element-wise composition that `torch.compile`'s inductor cannot auto-fuse.

## What Was Implemented

- Fused Triton kernels in `nanochat/rotary.py`:
  - **Forward**: `_fused_norm_rotary_fwd_kernel` — one program per (batch, seq, head) slice, does sum-of-squares reduction → RMSNorm → rotary rotation → store output + rms scalar.
  - **Backward**: `_fused_norm_rotary_bwd_kernel` — recovers x_norm via inverse rotation R^T, then applies fused RMSNorm backward.
- `torch.compile` compatibility via `@allow_in_graph` + FakeTensor detection (`data_ptr()` try/catch).
- CPU/no-Triton fallback using standard PyTorch ops.
- `gpt.py` integration: 4-line norm+rotary → 2 calls:
  ```python
  q = fused_norm_rotary(q, cos, sin)
  k = fused_norm_rotary(k, cos, sin)
  ```

## Measurement (`speedrun_small.sh`)

| Metric | Baseline | Fused Kernel | Delta |
|--------|----------|-------------|-------|
| `fwd peak alloc (MiB)` | 18477.90 | 17901.90 | **-576.0** (−3.1%) |
| `bwd peak alloc (MiB)` | 18735.43 | 18186.35 | **-549.1** (−2.9%) |
| `step peak alloc (MiB)` | 18735.43 | 18314.25 | **-421.2** (−2.2%) |
| Loss (step 9) | 8.521710 | 8.521724 | +0.000014 |
| Min val bpb | 1.798807 | 1.798807 | 0.000000 |

## Mathematical Foundation

### Forward: Fused RMSNorm + Rotary

For input vector $\mathbf{x} \in \mathbb{R}^D$ (one head slice), the fused operation computes:

1. **RMSNorm**: $\text{rms} = \sqrt{\frac{1}{D}\sum_{i=1}^{D} x_i^2 + \varepsilon}$, then $\hat{x}_i = x_i / \text{rms}$

2. **Rotary**: Split $\hat{\mathbf{x}}$ into halves $(\hat{x}_1, \hat{x}_2)$, apply per-pair rotation:

$$\begin{bmatrix} y_1 \\ y_2 \end{bmatrix} = R(\theta) \begin{bmatrix} \hat{x}_1 \\ \hat{x}_2 \end{bmatrix} = \begin{bmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} \hat{x}_1 \\ \hat{x}_2 \end{bmatrix}$$

The kernel does this in **1 read + 1 write** (plus one rms scalar write), with all computation in float32 registers.

### Backward: Inverse Rotation Trick

The backward needs $\hat{\mathbf{x}}$ (the normalized, pre-rotation activations) for the RMSNorm gradient. Instead of saving $\mathbf{x}$ (which costs ~576 MiB of activation memory), we exploit the **orthogonality of the rotation matrix**:

Since $\cos^2\theta + \sin^2\theta = 1$, the rotation matrix $R(\theta)$ is orthogonal, so $R^{-1} = R^T$:

$$\hat{\mathbf{x}} = R^T \mathbf{y}$$

This recovers $\hat{\mathbf{x}}$ from the saved output $\mathbf{y}$ **for free** — no extra memory needed.

The full backward then computes:
1. $\nabla_{\hat{x}} = R^T \nabla_y$ (rotary backward)
2. $c = \frac{1}{D}\sum_i (\nabla_{\hat{x}})_i \cdot \hat{x}_i$ (reduction for RMSNorm)
3. $\nabla_x = \frac{1}{\text{rms}}(\nabla_{\hat{x}} - \hat{\mathbf{x}} \cdot c)$ (RMSNorm backward)

### The cos/sin Testing Pitfall

> **Edge case caught during development**: The inverse rotation $R^T$ only recovers $\hat{\mathbf{x}}$ when $\cos^2 + \sin^2 = 1$. Our initial tests used **random** cos/sin values (not from actual angles), which broke this invariant and caused 99.7% of backward elements to be wrong by up to 57x.
>
> With real rotary embeddings — where cos/sin are computed as `freqs.cos()` and `freqs.sin()` from the same frequencies — the identity holds exactly, and the backward is correct. Tests were fixed to use real rotary angles.

### Why Saving `out` Instead of `x` Saves Memory

The output tensor $\mathbf{y}$ is already kept alive by PyTorch's autograd graph because it flows into the next operation (flash attention). Saving it in `ctx.save_for_backward` adds **zero** extra memory — it merely increments a reference count on an already-retained tensor.

In contrast, saving the input $\mathbf{x}$ forces the pre-norm Q/K activations to remain in memory through the entire backward pass. For this model:
- Per-layer Q tensor: $B \times T \times H \times D$ at bf16 = ~12 MiB
- Across 24 layers × 2 tensors (Q, K) = ~576 MiB total

This is exactly the 576 MiB reduction observed in the measurements.

## Why Inductor Can't Do This

`torch.compile`'s inductor backend can fuse element-wise ops (mul, add, cat) into a single kernel. But RMSNorm involves a **reduction** (sum of squares across the D dimension) followed by element-wise operations. Inductor's fusion heuristics don't cross the reduction boundary:

```
norm kernel:   read x → reduce → normalize → write x_norm     (separate launch)
rotary kernel: read x_norm → rotate → write y                  (separate launch)
```

Our fused kernel does both in one pass:
```
fused kernel:  read x → reduce → normalize → rotate → write y  (single launch)
```

## Tests

```bash
python -m pytest tests/test_rotary.py -v -s
# 19 passed in 1.80s
```

Covers: forward (4 shapes + GQA), backward (2 shapes + upstream grad), CPU fallback (fwd + bwd), for both standalone rotary and fused norm+rotary.
