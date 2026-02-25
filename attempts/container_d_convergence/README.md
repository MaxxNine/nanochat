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
  --dyn-loss-upper 1.01 \
  --dyn-loss-lower 0.997 \
  --dyn-budget-step-up 2 \
  --dyn-budget-step-down 1 \
  --dyn-loss-ema-decay 0.9 \
  --json-out attempts/results/container_d/dynamic_block_schedule_$(date +'%Y%m%d_%H%M%S').json
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
