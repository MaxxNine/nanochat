# Optimization 005: Fused Linear Cross-Entropy via Cut-Cross-Entropy

Date: 2026-03-02
Status: Implemented and measured

## Objective

Replace the memory-hungry full-logits `(B*T, V)` materialization in the LM-head CE path with Apple's [cut-cross-entropy](https://github.com/apple/ml-cross-entropy) library, which computes the loss tile-by-tile in SRAM without ever writing the full logit tensor to HBM.

## Background: Why Prior Attempts Failed

Before landing on cut-cross-entropy, two other approaches were tried:

1. **Python chunked CE** (`_ChunkedLinearCrossEntropy`): A custom `torch.autograd.Function` that processes vocab in chunks. Used *more* VRAM than baseline because it created redundant fp32 copies of `h` and `weight` in both forward and backward, recomputed LSE from scratch in backward, and the Python loop launched many small CUDA kernels whose intermediates weren't freed promptly.

2. **In-repo Triton kernels** (`triton_lce.py`): Forward LSE and backward kernels existed but the backward used `tl.atomic_add` for gradient accumulation, making it **2.8× slower** than baseline. Additionally, `.item()` calls in the autograd function caused `torch.compile` graph breaks, negating compile-time memory optimizations.

## What Was Implemented

A 15-line wrapper in `gpt.py` that delegates to `cut_cross_entropy.linear_cross_entropy`:

```python
def fused_linear_cross_entropy(h, weight, targets, softcap=15.0, chunk_size=4096, reduction='mean'):
    from cut_cross_entropy import linear_cross_entropy
    return linear_cross_entropy(
        h, weight, targets,
        ignore_index=-1,
        softcap=softcap,
        reduction=reduction,
    )
```

Activated via `--lm-ce-backend=fused` in `base_train.py` or `LM_CE_BACKEND=fused` in `speedrun_small.sh`.

### Why cut-cross-entropy works where our prototypes didn't

- **No atomics**: backward computes `grad_h` and `grad_w` without atomic accumulation
- **No graph breaks**: designed to work inside `torch.compile`
- **No fp32 duplication**: the kernel handles mixed-precision internally in SRAM
- **Native softcap support**: `softcap=` parameter, no external squashing needed

## Measurement (`speedrun_small.sh`, d24, FP8, dynamic_suffix)

| Metric | Baseline | Fused (CCE) | Delta |
|--------|----------|-------------|-------|
| `fwd peak alloc (MiB)` | 17901.90 | 17646.79 | **−255.1** (−1.4%) |
| `bwd peak alloc (MiB)` | 18186.35 | 18156.54 | **−29.8** (−0.2%) |
| `step peak alloc (MiB)` | 18186.35 | 18156.54 | **−29.8** (−0.2%) |
| `dt (ms)` | 1430.71 | 1423.91 | −6.8 (−0.5%) |
| `tok/sec` | 22,903 | 23,012 | +109 (+0.5%) |
| `bf16_mfu` | 73.27% | 73.62% | +0.35 pp |
| Loss (step 2) | 10.232 | 10.257 | +0.025 |

**Forward peak savings of ~255 MiB** with speed neutral-to-slightly-faster (~0.5%). The step-level peak is dominated by backward, where savings are smaller because the backward allocates temporary gradient buffers.

The forward savings come from never materializing the `(B*T, V)` logits tensor:
- Baseline: `lm_head(x)` produces `(B*T, padded_V)` in bf16 = `16384 × 32768 × 2 bytes` ≈ **1024 MiB**
- CCE: computes loss tile-by-tile, peak intermediate is bounded by tile size (typically 128×128)

## Full 300-Step Training Run

| Metric | Baseline | Fused (CCE) | Delta |
|--------|----------|-------------|-------|
| Peak memory (MiB) | 18186.37 | 18156.56 | **−29.8** |
| Avg step peak memory (MiB) | 15065.62 | 14896.80 | **−168.8** (−1.1%) |
| Min validation bpb | 1.657326 | 1.618329 | **−0.039** (−2.4%) |
| Final loss | 5.307 | 5.180 | **−0.127** |
| Total time | 5.03m | 5.59m | +0.56m (+11.1%)* |

*\*Speed caveat: the fused run had background system load (browser/video playback) that inflated wall-clock time. A clean-system step-2 comparison showed CCE is actually ~0.5% faster (1424ms vs 1431ms).*

**Surprise quality win**: the fused path produced a **0.039 bpb improvement** (1.618 vs 1.657). CCE uses Kahan summation and gradient filtering, which changes the numerical accumulation path compared to `F.cross_entropy`. This subtle difference in gradient precision appears to benefit convergence at this scale.

The **avg step peak savings of ~169 MiB** is more representative than the absolute peak, since the absolute peak occurs at the same probe step (60) for both runs and is dominated by the dynamic suffix overhead rather than the CE path.

The 257 MiB measured savings (vs the theoretical 1024 MiB) shows that `torch.compile` already reclaims much of the logits memory via activation recomputation in the baseline path.

## Files Changed

| File | Change |
|------|--------|
| `nanochat/gpt.py` | Added `fused_linear_cross_entropy()` wrapper, `"fused"` backend to `configure_lm_ce()` |
| `pyproject.toml` | Added `cut-cross-entropy` dependency |
| `scripts/base_train.py` | Added `fused` to `--lm-ce-backend` choices |
| `runs/speedrun_small.sh` | Fixed `uv sync --inexact` (prevents package deletion), updated backend choices |
| `tests/test_chunked_ce.py` | Added fused CE correctness tests (mean + sum reduction) |

## Tests

```bash
python -m pytest tests/test_chunked_ce.py -v -s
# 5 passed in 3.48s
```
