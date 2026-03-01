# Optimization 002: Dynamic Suffix Block Updates (Feature-Flagged)

Date: 2026-02-28
Status: Implemented and measured

## Objective
Bring `dynamic_suffix` scheduling from the experiment bench into main training behind feature flags, so it can be toggled and combined with FP8.

## What Was Implemented
- `GPT.forward(..., active_start=0)` support in [`nanochat/gpt.py`](/home/gabrielmaxx/projects/gpt/nanochat/nanochat/gpt.py)
  - Prefix blocks `[0, active_start)` run under `no_grad`
  - Activations are detached at suffix boundary
  - Suffix blocks `[active_start, end)` train normally
- Dynamic suffix control logic in [`scripts/base_train.py`](/home/gabrielmaxx/projects/gpt/nanochat/scripts/base_train.py)
  - Probe/full-update steps: warmup + periodic probe + periodic refresh
  - Relevance EMA per block
  - Active suffix chosen by cumulative relevance threshold
  - Per-step logging: active layers + probe/suffix marker
- Optimizer hardening for sparse block gradients in [`nanochat/optim.py`](/home/gabrielmaxx/projects/gpt/nanochat/nanochat/optim.py)
  - Muon now tolerates missing grads in group params for this schedule
- Runner env wiring in [`runs/speedrun_small.sh`](/home/gabrielmaxx/projects/gpt/nanochat/runs/speedrun_small.sh)
  - Dynamic flags can be set via environment variables
- Schedule A/B runner in [`runs/dynamic_suffix_compare.sh`](/home/gabrielmaxx/projects/gpt/nanochat/runs/dynamic_suffix_compare.sh)
  - Compares `full_update` vs `dynamic_suffix` with optional FP8 in both runs

## Important Constraints
- `dynamic_suffix` is currently single-rank only (`nproc_per_node=1`).
- It can be used together with FP8.
- With compile enabled, dynamic suffix caches compiled graphs per `active_start` and reuses them; eager is fallback only if a specific suffix graph fails to compile.

## Quick Start
Dynamic suffix only:
```bash
BLOCK_UPDATE_SCHEDULE=dynamic_suffix \
SKIP_SETUP=1 PREPARE_DATA=0 \
bash runs/speedrun_small.sh
```

Dynamic suffix + FP8 custom:
```bash
BLOCK_UPDATE_SCHEDULE=dynamic_suffix \
USE_FP8=1 FP8_BACKEND=custom FP8_RECIPE=tensorwise \
SKIP_SETUP=1 PREPARE_DATA=0 \
bash runs/speedrun_small.sh
```

Conservative FP8 + dynamic suffix preset:
```bash
BLOCK_UPDATE_SCHEDULE=dynamic_suffix \
USE_FP8=1 FP8_BACKEND=custom \
BASE_TRAIN_EXTRA_ARGS="--fp8-skip-lm-head --fp8-skip-attn-qk --fp8-min-dim=256 --fp8-no-allow-in-graph" \
SKIP_SETUP=1 PREPARE_DATA=0 \
bash runs/speedrun_small.sh
```

Schedule comparison (full vs dynamic) with FP8:
```bash
USE_FP8_COMPARE=1 FP8_BACKEND_COMPARE=custom \
SKIP_SETUP=1 PREPARE_DATA=0 \
bash runs/dynamic_suffix_compare.sh
```

Fair-time comparison example (more dynamic steps in same wall-clock range):
```bash
USE_FP8_COMPARE=1 FP8_BACKEND_COMPARE=custom \
FULL_NUM_ITERATIONS=520 DYNAMIC_NUM_ITERATIONS=650 \
SKIP_SETUP=1 PREPARE_DATA=0 \
bash runs/dynamic_suffix_compare.sh
```

## Raw Result (Provided, Fair-Time Style)
```text
variant  schedule        num_iterations  elapsed_sec  max_tok_per_sec  avg_tok_per_sec  final_loss  min_val_bpb  avg_step_peak_vram_mib  peak_vram_mib  active_layers_mean  active_layers_range
dynamic  dynamic_suffix  650             198          137822           126324.83        4.913683    1.446937     3535.81                 3778.14        7.44                6-12
full     full_update     520             191          100619            99419.09        4.869418    1.458417     3778.05                 3778.14        NA                  NA
```

## Mathematical Rationale
Let each transformer block have forward cost `F_l` and backward+update cost `B_l`.

Full update per step:

`C_full = sum_l (F_l + B_l)`

Dynamic suffix with `active_start = a`:
- prefix `[0, a)` does forward only (`no_grad`)
- suffix `[a, n)` does forward + backward/update

`C_dyn(a) = sum_{l < a} F_l + sum_{l >= a} (F_l + B_l)`

Assuming `B_l ~= k * F_l` with `k ~= 2` (common training approximation), and `p` = frozen-prefix forward cost fraction:

`C_dyn / C_full ~= 1 - p * k / (1 + k)`

`Speedup ~= 1 / (1 - p * k / (1 + k))`

With observed `active_layers_mean = 7.44/12`, frozen prefix fraction is roughly:

`p ~= (12 - 7.44) / 12 = 0.38`

Then with `k=2`:

`Speedup_pred ~= 1 / (1 - 0.38 * 2/3) = 1.34x`

Observed max speedup was `1.37x`, very close to this model.

## Fair-Comparison Math (Equal Wall Time)
For wall-clock budget `T`, with batch tokens `B_tok` and throughput `nu`:

`steps ~= (nu * T) / B_tok`

So dynamic should run more steps than full in the same time if `nu_dynamic > nu_full`:

`steps_dynamic / steps_full ~= nu_dynamic / nu_full`

From the provided run:

`126324.83 / 99419.09 ~= 1.27`

`520 * 1.27 ~= 661` (close to used `650`), so your setup is already near time-fair.

## Interpretation
- This is a meaningful optimization: substantial throughput gain at similar quality trend.
- `avg_step_peak_vram_mib` dropped, which indicates lower per-step pressure even when global peak remains equal.
- Equal `peak_vram_mib` can happen because both variants still hit the same transient high-watermark at least once.
