# Progress So Far (Math + Code)

## Goal Context
Target: get high-quality training on 1x RTX 4090 with strong wall-clock efficiency, while preserving home-trainable workflows.

Core optimization model:

`time_to_quality ~= required_flops / effective_flops_per_second + pipeline_overhead`

## What We Implemented (Code)
### Container A (Required FLOPs / approximation attempts)
- `attempts/container_a_required_flops/matrix_approx_bench.py`
- `attempts/container_a_required_flops/loss_aware_low_rank_recovery.py`
- `attempts/container_a_required_flops/data_aware_low_rank_plus_diag.py`
- `attempts/container_a_required_flops/structured_sparsity_mlp_bench.py`

### Container B (Effective FLOPs/s attempts)
- `attempts/container_b_effective_flops/attention_backend_bench.py`

### Container D (Convergence / selective updates)
- `attempts/container_d_convergence/dynamic_block_update_schedule_bench.py`
  - `full_update`, `static_suffix`, `dynamic_suffix`
  - equal-time budget mode
  - relevance EMA scheduling
  - adaptive active-layer budget
  - stability-gated freeze start
  - final full-update recovery phase

All attempt scripts support JSON outputs for run tracking.

## Mathematical Strategy (What We Tested)
### A) Low-rank approximation family
Initial hypothesis:
- reduce matrix cost with low-rank surrogates.

Tested:
1. plain truncated SVD low-rank
2. loss-aware recovery after SVD init
3. data-aware low-rank objective (`||X(W-W_hat)||`)
4. low-rank plus diagonal residual
5. structured MLP sparsity variants

Observed:
- large speedups generally came with too much approximation error.
- quality degradation at useful speedup levels was too strong for now.

### D) Dynamic block update scheduling
Main idea:
- use gradient relevance to choose a trainable top-suffix of layers.

Relevance score per block:

`s_l = ||g_l||_2 / (||W_l||_2 + eps)`

EMA update:

`ema_l <- beta * ema_l + (1-beta) * s_l`

Suffix selection:
- choose smallest top-suffix meeting cumulative relevance threshold,
- constrained by min/max active layers.

Additional controls:
- freeze start gating by step/fraction and optional loss stability:
  - relative mean-loss change threshold
  - coefficient-of-variation threshold
- adaptive budget with probe-loss EMA:
  - if loss rises beyond upper factor -> increase active budget
  - if loss falls below lower factor -> decrease active budget
- final full-update fraction for recovery.

## Key Experimental Results
### Early dynamic run (too aggressive quality drop)
File:
- `attempts/results/container_d/dynamic_block_schedule_20260225_160512.json`

Summary:
- `full_update`: val `0.1470`, mean ms `72.18`
- `static_suffix`: val `0.2728`, mean ms `38.32`, speedup `1.88x`
- `dynamic_suffix`: val `0.2619`, mean ms `55.27`, speedup `1.31x`

Issue:
- dynamic mostly sat at one reduced setting, quality too far from full.

### Tuned dynamic run (major improvement)
File:
- `attempts/results/container_d/dynamic_block_schedule_tuned_20260225_163732.json`

Summary:
- `full_update`: val `0.098275`, mean ms `72.98`
- `static_suffix`: val `0.248637`, mean ms `38.45`, speedup `1.90x`
- `dynamic_suffix`: val `0.099690`, mean ms `65.37`, speedup `1.12x`

Dynamic quality gap vs full:
- `+1.44%` (very close)

### Latest tuned v2 run (better speed, small extra quality cost)
File:
- `attempts/results/container_d/dynamic_block_schedule_tuned_v2_20260225_164614.json`

Summary:
- `full_update`: val `0.098275`, mean ms `71.46`
- `static_suffix`: val `0.248711`, mean ms `38.12`, speedup `1.87x`
- `dynamic_suffix`: val `0.103321`, mean ms `62.09`, speedup `1.15x`

Dynamic quality gap vs full:
- about `+5.1%`

Dynamic active-layer behavior:
- active suffix levels used: `12` and `18`
- mean active layers: `14.47`
- probes: `65`

## Interpretation
1. **Static aggressive freezing** is fast but degrades quality too much.
2. **Dynamic suffix scheduling works** and can stay near full quality.
3. Current dynamic gains are real (`~1.12x-1.15x`), but still moderate.
4. Main bottleneck now is scheduler shape:
   - when to enter reduced mode,
   - how long to stay there,
   - how often to probe,
   - how strong final recovery should be.

## Why This Matters
We now have evidence that:
- selective update scheduling is viable on 4090,
- we can trade speed for quality in a controllable way,
- we can get non-trivial speedup without catastrophic loss.

This is a concrete, measurable path to keep improving.

## Next Suggested Iterations
1. Replace relevance metric with saliency-like signal (`|g * p|`) and compare.
2. Allow more than two active levels (currently often toggles between 12 and 18).
3. Tune probe cadence to reduce overhead while preserving adaptation.
4. Optimize final-recovery window to recover quality with less full-update time.
