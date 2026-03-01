# Optimization 003: Muon Sparse Stacking + Chunking (Feature-Flagged)

Date: 2026-03-01
Status: Implemented and measured

## Objective
Reduce optimizer-step memory spikes (the dominant peak source) and improve throughput on dynamic-suffix training by updating only active Muon matrices and chunking stack size behind feature flags.

## Motivation (From Memory Attribution)
Phase-level memory debugging showed the largest spike came from optimizer step, not forward CE:

- `phase_peak_alloc_mib(fwd/bwd/opt/zero)` had `opt` as max during full/suffix steps.
- This indicated Muon stacking workspace was the right optimization target.

## What Was Implemented
- New Muon feature flags in training CLI (`scripts/base_train.py`):
  - `--muon-active-only-stack`
  - `--muon-stack-chunk-size`
- New Muon env mapping in runner (`runs/speedrun_small.sh`):
  - `MUON_ACTIVE_ONLY_STACK`
  - `MUON_STACK_CHUNK_SIZE`
- Optimizer implementation updates (`nanochat/optim.py`):
  - Active-only contiguous range stacking for Muon groups.
  - Optional stack chunking to cap temporary optimizer workspace.
  - Shape-stable padded short chunks to avoid `torch.compile` recompile-limit failures when active set changes.
  - Single-rank `torchrun` path avoids materializing full `grad_stack` when sparse/chunk mode is enabled.
  - Sparse-only mode: if all Muon params are active, auto-fallback to legacy one-shot Muon update (preserves full-step profile).

## New Default in Local Runner
`runs/speedrun_small.sh` now defaults to:

- `MUON_ACTIVE_ONLY_STACK=1`
- `MUON_STACK_CHUNK_SIZE=4`

This is now the default local 4090 optimization profile.

## Measurement (Provided)
Optimized run:

```text
step 00199/00200 ... dt: 744.07ms | tok/sec: 44,038 ...
Peak memory usage: 16170.12MiB
Peak memory step: 60
Average memory usage: 8293.65MiB
Average step peak memory usage: 13485.77MiB
Total training time: 2.78m
Minimum validation bpb: 1.782550
Dynamic suffix summary: probe_steps=48, active_layers_mean=13.64, active_layers_range=[11,22]
```

Previous run:

```text
step 00199/00200 ... dt: 862.02ms | tok/sec: 38,013 ...
Peak memory usage: 17271.55MiB
Peak memory step: 1
Average memory usage: 7716.33MiB
Average step peak memory usage: 16176.13MiB
Total training time: 3.00m
Minimum validation bpb: 1.783329
Dynamic suffix summary: probe_steps=48, active_layers_mean=13.64, active_layers_range=[11,22]
```

## Gain vs Previous
- End-step throughput: `+15.85%` (`44,038` vs `38,013` tok/s)
- Peak memory: `-1101.43 MiB` (`16170.12` vs `17271.55`, `-6.38%`)
- Average step peak memory: `-2690.36 MiB` (`13485.77` vs `16176.13`, `-16.63%`)
- Total training time: `-7.33%` (`2.78m` vs `3.00m`)
- Min val bpb: improved by `0.000779` (`1.782550` vs `1.783329`)
- Average memory usage increased (`+577.32 MiB`), which is acceptable here because peak and per-step peak dropped substantially.

## Mathematical Rationale
Let a Muon group contain `K` same-shape matrices. Let `K_active` be active ones on a suffix step.

Legacy Muon temporary stack cost scales with all matrices:

`C_mem_legacy ~ O(K * M)`

where `M` is per-matrix bytes in stacked workspace.

Active-only stacking changes this to:

`C_mem_active ~ O(K_active * M)`

Memory reduction factor:

`r_mem ~ K_active / K`

On suffix steps, `K_active < K`, so optimizer workspace drops proportionally.

Chunking with chunk size `c` further caps instantaneous stack size:

`C_mem_chunk_peak ~ O(min(c, K_active) * M)`

This reduces peaks further at cost of extra launch/loop overhead.

Throughput model per step:

`T_step = T_fwd+bwd + T_opt`

`T_opt_legacy ~ a*K + b`

`T_opt_new ~ a*K_active + b + delta_chunk`

So when `K_active << K` and `delta_chunk` is controlled, `T_step` improves on suffix steps. This matched measured `tok/sec` gains.

## Why It Needed the Compile Fix
`muon_step_fused` is compiled with fixed-shape assumptions. Dynamic suffix changes active set size across steps, causing leading-dimension shape changes and recompiles.

Padding short chunks to fixed leading dimension keeps compile shapes stable:

`shape_in = (c, rows, cols)` for all short-chunk calls.

This prevented recompile-limit crashes while preserving sparse update behavior.

## Commands
Default (now includes this optimization):

```bash
bash runs/speedrun_small.sh
```

Explicit:

```bash
MUON_ACTIVE_ONLY_STACK=1 \
MUON_STACK_CHUNK_SIZE=4 \
bash runs/speedrun_small.sh
```

## Interpretation
This optimization targets the actual bottleneck (optimizer peak), not output CE.
It gives a better speed/memory tradeoff for dynamic suffix on 4090 and is a strong default profile.
