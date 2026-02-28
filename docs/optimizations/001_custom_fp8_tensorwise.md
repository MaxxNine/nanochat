# Optimization 001: Custom FP8 (Tensorwise) for 1x RTX 4090

Date: 2026-02-28
Status: Implemented and measured

## Objective
Enable stable FP8 training on a single RTX 4090 behind feature flags, and measure speed/VRAM impact versus baseline BF16.

## What Was Implemented
- Small local runner for iterative experiments:
  - `runs/speedrun_small.sh`
- One-command baseline vs FP8 comparison:
  - `runs/fp8_compare.sh`
- One-command custom vs torchao backend comparison:
  - `runs/fp8_backend_compare.sh`
- 2-step NaN debug runner:
  - `runs/fpi_debug.sh`
- FP8 integration in training (`scripts/base_train.py`) with feature flags:
  - `--fp8`
  - `--fp8-backend {custom,torchao}`
  - `--fp8-recipe`
  - `--fp8-min-dim`
  - `--fp8-skip-lm-head`
  - `--fp8-skip-attn-qk`
  - `--fp8-fast-accum`
  - `--fp8-weight-grad-bf16`
  - `--fp8-no-allow-in-graph`
  - `--fp8-include-regex`
  - `--fp8-exclude-regex`
  - `--fp8-log-modules`
- Custom FP8 backend hardening (`nanochat/fp8.py`):
  - FP32 dWeight path by default (stability)
  - optional BF16 dWeight via flag for speed testing
- Compile/eval safety fix:
  - eval runs through uncompiled model path to avoid compiled graph topology mutation.

## Stable FP8 Preset Used in This Measurement
`--fp8-skip-lm-head --fp8-skip-attn-qk --fp8-min-dim=256 --fp8-no-allow-in-graph`

Converted FP8 layers: `48/79` (60.76%)

## Measurement Command
```bash
bash runs/fp8_compare.sh
```

## Raw Result (Provided)
```text
variant   use_fp8  fp8_backend  fp8_recipe  exit_code  status  elapsed_sec  max_tok_per_sec  avg_tok_per_sec  final_loss  min_val_bpb  peak_vram_mib  fp8_layers
fp8       1        custom       tensorwise  0          ok      83           100559           98905.82         5.427657    1.591045     5498.80        48/79
baseline  0        NA           NA          0          ok      79           96401            95277.12         5.426070    1.590747     5722.43        NA
```

## Gain vs Baseline
- Max throughput: `+4.31%` (`100559` vs `96401` tok/s)
- Avg throughput: `+3.81%` (`98905.82` vs `95277.12` tok/s)
- Peak VRAM: `-223.63 MiB` (`5498.80` vs `5722.43`, `-3.91%`)
- Final loss delta: `+0.001587` (`+0.029%`)
- Min val bpb delta: `+0.000298` (`+0.019%`)

## Interpretation
- This first 4090 optimization delivered a real speedup and VRAM reduction.
- Quality impact in this short run was very small (loss/bpb almost unchanged).
- This is a conservative FP8 subset (48/79 layers), chosen for stability under compile.

## Notes for Next Optimization Entries
For each new optimization, track:
- exact command
- flags changed
- throughput deltas
- VRAM delta
- loss/bpb delta
- stability notes (NaN/compile behavior)
