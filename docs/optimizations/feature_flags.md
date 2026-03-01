# Feature Flags Source of Truth

Date: 2026-02-28
Scope: base model training (`scripts/base_train.py`) and local runners.

This file is the canonical list of optimization feature flags used for 1x4090 experiments.

## FP8 Flags
- `--fp8`
- `--fp8-backend {custom,torchao}`
- `--fp8-recipe {tensorwise,rowwise}` (`rowwise` is torchao-only)
- `--fp8-min-dim`
- `--fp8-skip-lm-head`
- `--fp8-skip-attn-qk`
- `--fp8-fast-accum` (custom backend)
- `--fp8-weight-grad-bf16` (custom backend; default is FP32 dWeight)
- `--fp8-no-allow-in-graph` (custom backend)
- `--fp8-include-regex`
- `--fp8-exclude-regex`
- `--fp8-log-modules`

## Dynamic Suffix Flags
- `--block-update-schedule {full_update,dynamic_suffix}`
- `--dyn-warmup-steps`
- `--dyn-probe-every`
- `--dyn-refresh-every`
- `--dyn-relevance-metric {grad_ratio,saliency_abs_gp}`
- `--dyn-relevance-threshold`
- `--dyn-min-active-layers`
- `--dyn-max-active-layers` (`0` means no cap)
- `--dyn-freeze-start-step` (`-1` uses fraction)
- `--dyn-freeze-start-frac`
- `--dyn-ema-decay`
- `--dyn-eps`
- `--dyn-log-every`

Notes:
- Current implementation is dynamic-only (no static schedule in training code).
- `dynamic_suffix` currently supports single-rank runs (`nproc_per_node=1`).
- `dynamic_suffix` can be combined with FP8.
- `dynamic_suffix` compile behavior:
  compiled graphs are cached per `active_start`; eager is fallback for per-suffix compile failures.

## Debug/Control Flags
- `--no-compile`
- `--debug-nan`
- `--debug-anomaly`
- `--debug-max-nonfinite`

## Runner Environment Mapping (`runs/speedrun_small.sh`)
FP8:
- `USE_FP8`, `FP8_BACKEND`, `FP8_RECIPE`

Dynamic suffix:
- `BLOCK_UPDATE_SCHEDULE`
- `DYN_WARMUP_STEPS`, `DYN_PROBE_EVERY`, `DYN_REFRESH_EVERY`
- `DYN_RELEVANCE_METRIC`, `DYN_RELEVANCE_THRESHOLD`
- `DYN_MIN_ACTIVE_LAYERS`, `DYN_MAX_ACTIVE_LAYERS`
- `DYN_FREEZE_START_STEP`, `DYN_FREEZE_START_FRAC`
- `DYN_EMA_DECAY`, `DYN_EPS`, `DYN_LOG_EVERY`
