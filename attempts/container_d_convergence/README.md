# Container D Attempts: Convergence Efficiency

## Goal
Test whether we can reduce wall-clock by updating only relevant blocks over time, while preserving quality.

## Script
- `dynamic_block_update_schedule_bench.py`

Schedules compared:
1. `full_update`: all blocks updated every step.
2. `static_suffix`: only top K blocks updated every step.
3. `dynamic_suffix`: relevance-guided top-suffix updates, with periodic full probe/update steps.

Metrics:
- final validation loss
- train loss start/end
- mean/p50/p90 step latency
- speedup vs full update
- active layer statistics and probe count
- full traces in JSON

New feature flags:
- `--dyn-final-recovery-mode staged`
  - Replaces "all-full in final window" with a staged layer ramp for better quality/time tradeoff.
- `--dyn-relevance-metric saliency_abs_gp`
  - Uses saliency-like per-block relevance (`|g * p|`) instead of gradient/weight ratio.

## Example (quick 4090 run)
```bash
poetry run python attempts/container_d_convergence/dynamic_block_update_schedule_bench.py \
  --device auto \
  --dtype bfloat16 \
  --dmodel 1024 \
  --n-layers 18 \
  --hidden-mult 4 \
  --batch-size 8 \
  --seq-len 1024 \
  --steps 200 \
  --equal-time-budget \
  --max-steps-multiplier 3 \
  --lr 3e-4 \
  --static-active-layers 6 \
  --dyn-warmup-steps 25 \
  --dyn-probe-every 40 \
  --dyn-refresh-every 120 \
  --dyn-relevance-metric grad_ratio \
  --dyn-relevance-threshold 0.92 \
  --dyn-min-active-layers 12 \
  --dyn-max-active-layers 18 \
  --dyn-adapt-budget \
  --dyn-require-stable \
  --dyn-freeze-start-frac 0.55 \
  --dyn-stability-window 12 \
  --dyn-stability-rel-change 0.02 \
  --dyn-stability-cv 0.03 \
  --dyn-final-full-frac 0.25 \
  --dyn-final-recovery-mode full \
  --dyn-loss-upper 1.01 \
  --dyn-loss-lower 0.997 \
  --dyn-budget-step-up 2 \
  --dyn-budget-step-down 1 \
  --dyn-loss-ema-decay 0.9 \
  --json-out attempts/results/container_d/dynamic_block_schedule_$(date +'%Y%m%d_%H%M%S').json
```

## Example: staged final recovery
```bash
poetry run python attempts/container_d_convergence/dynamic_block_update_schedule_bench.py \
  --device auto --dtype bfloat16 --dmodel 1024 --n-layers 18 --hidden-mult 4 \
  --batch-size 8 --seq-len 1024 --steps 1050 --equal-time-budget --max-steps-multiplier 3 \
  --dyn-probe-every 24 --dyn-refresh-every 120 \
  --dyn-min-active-layers 12 --dyn-max-active-layers 15 \
  --dyn-adapt-budget --dyn-require-stable --dyn-freeze-start-frac 0.35 \
  --dyn-final-full-frac 0.12 \
  --dyn-final-recovery-mode staged \
  --dyn-final-recovery-stage1-progress 0.34 \
  --dyn-final-recovery-stage1-layers 14 \
  --dyn-final-recovery-stage2-progress 0.67 \
  --dyn-final-recovery-stage2-layers 16 \
  --json-out attempts/results/container_d/dynamic_block_schedule_staged_$(date +'%Y%m%d_%H%M%S').json
```

## Example: saliency relevance metric
```bash
poetry run python attempts/container_d_convergence/dynamic_block_update_schedule_bench.py \
  --device auto --dtype bfloat16 --dmodel 1024 --n-layers 18 --hidden-mult 4 \
  --batch-size 8 --seq-len 1024 --steps 1050 --equal-time-budget --max-steps-multiplier 3 \
  --dyn-relevance-metric saliency_abs_gp \
  --dyn-min-active-layers 12 --dyn-max-active-layers 15 \
  --dyn-adapt-budget --dyn-require-stable --dyn-freeze-start-frac 0.35 \
  --dyn-final-full-frac 0.10 --dyn-final-recovery-mode full \
  --json-out attempts/results/container_d/dynamic_block_schedule_saliency_$(date +'%Y%m%d_%H%M%S').json
```

## Interpretation guide
- Keep `dynamic_suffix` only if:
  - it yields clear speedup vs `full_update`, and
  - final val loss degradation is acceptable for the target.
- If `dynamic_suffix` quality drops too much:
  - increase `dyn-min-active-layers`,
  - increase `dyn-max-active-layers`,
  - reduce `dyn-probe-every` (probe more often),
  - reduce `dyn-relevance-threshold` less aggressively.
