# Container A Attempts: Required FLOPs

## Goal
Probe whether approximate matrix compute can reduce required FLOPs while preserving acceptable output quality.

## Script
- `matrix_approx_bench.py`
- `loss_aware_low_rank_recovery.py`
- `data_aware_low_rank_plus_diag.py`
- `structured_sparsity_mlp_bench.py`

What it benchmarks:
- Dense linear projection (baseline) vs low-rank approximation.
- Dense MLP block (baseline) vs low-rank approximation for both MLP matrices.
- Recovery variant: low-rank starts from SVD, then receives short optimization against dense teacher outputs.
- Data-aware low-rank variants with and without diagonal residual.
- Structured sparsity MLP variants (neuron pruning and optional 2:4 masking).

Reported metrics:
- per-iteration latency (ms)
- estimated FLOPs/iteration
- estimated TFLOP/s
- relative L2 output error vs dense baseline
- speedup vs dense baseline
- For recovery: initial error vs final error, and error reduction percentage.

## Example
```bash
poetry run python attempts/container_a_required_flops/matrix_approx_bench.py \
  --device auto \
  --dtype bfloat16 \
  --tokens 16384 \
  --dmodel 1664 \
  --rank-fracs 1.0,0.75,0.5,0.375,0.25 \
  --iters 30 \
  --warmup 10 \
  --json-out attempts/results/container_a/matrix_approx_$(date +'%Y%m%d_%H%M%S').json
```

## Notes
- This script uses SVD to build best low-rank approximations, which can be expensive.
- If SVD is too slow, reduce dimensions first, validate trend, then scale up.

## Loss-aware recovery example
```bash
poetry run python attempts/container_a_required_flops/loss_aware_low_rank_recovery.py \
  --device auto \
  --dtype bfloat16 \
  --benchmark both \
  --dmodel 1664 \
  --rank-fracs 0.75,0.5,0.375,0.25 \
  --tokens-train 32768 \
  --tokens-val 8192 \
  --tokens-bench 16384 \
  --steps 200 \
  --batch-size 2048 \
  --lr 0.003 \
  --iters 30 \
  --warmup 10 \
  --json-out attempts/results/container_a/low_rank_recovery_$(date +'%Y%m%d_%H%M%S').json
```

## Data-aware low-rank + diag example
```bash
poetry run python attempts/container_a_required_flops/data_aware_low_rank_plus_diag.py \
  --device auto \
  --dtype bfloat16 \
  --dmodel 1664 \
  --rank-fracs 0.75,0.5,0.375,0.25 \
  --tokens-train 32768 \
  --tokens-val 8192 \
  --tokens-bench 16384 \
  --iters 30 \
  --warmup 10 \
  --json-out attempts/results/container_a/data_aware_lr_diag_$(date +'%Y%m%d_%H%M%S').json
```

## Structured sparsity MLP example
```bash
poetry run python attempts/container_a_required_flops/structured_sparsity_mlp_bench.py \
  --device auto \
  --dtype bfloat16 \
  --dmodel 1664 \
  --expansion 4 \
  --keep-fracs 0.75,0.5,0.375,0.25 \
  --tokens-val 8192 \
  --tokens-bench 16384 \
  --iters 30 \
  --warmup 10 \
  --json-out attempts/results/container_a/structured_sparse_mlp_$(date +'%Y%m%d_%H%M%S').json
```
