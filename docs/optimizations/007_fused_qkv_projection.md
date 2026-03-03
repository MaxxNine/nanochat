# Optimization 007: Fused QKV Projection (`c_qkv`)

Date: 2026-03-03
Status: Implemented and measured

## Objective

Speed up attention projection by replacing three independent linears (`c_q`, `c_k`, `c_v`) with one fused linear (`c_qkv`) and zero-copy split.

## Motivation

In the split path:

- forward launches 3 GEMMs from the same `x`
- backward accumulates gradients through 3 projection branches

In the fused path:

- forward uses 1 larger GEMM
- output is split with `torch.split` views (`q`, `k`, `v`)
- fewer launch overheads and usually better tensor-core utilization

## What Was Implemented

- Attention layout toggle in [`nanochat/gpt.py`](/home/gabrielmaxx/projects/gpt/nanochat/nanochat/gpt.py):
  - New config field: `GPTConfig.fused_qkv` (default `False` in model config)
  - `CausalSelfAttention` now supports:
    - split path: `c_q`, `c_k`, `c_v`
    - fused path: `c_qkv` + `torch.split`
- Checkpoint compatibility in [`nanochat/gpt.py`](/home/gabrielmaxx/projects/gpt/nanochat/nanochat/gpt.py):
  - `GPT.load_state_dict(...)` now converts split<->fused keys on load
  - strict loading remains supported across old/new checkpoints
- CLI flag in [`scripts/base_train.py`](/home/gabrielmaxx/projects/gpt/nanochat/scripts/base_train.py):
  - `--fused-qkv`
- Speedrun default in [`runs/speedrun_small.sh`](/home/gabrielmaxx/projects/gpt/nanochat/runs/speedrun_small.sh):
  - `FUSED_QKV=1` by default

## FP8 Interaction (Auto-Fix)

Important interaction:

- `--fp8-skip-attn-qk` is safe in split mode (skips only `c_q/c_k`)
- in fused mode, that same flag would skip full `c_qkv` (Q+K+V), which hurts performance/memory

To keep `fused_qkv` easy to trigger correctly:

- `runs/speedrun_small.sh` automatically removes `--fp8-skip-attn-qk` from dynamic FP8 safe args when `FUSED_QKV=1`
- `scripts/base_train.py` also auto-disables `args.fp8_skip_attn_qk` if both `--fused-qkv` and FP8 are active

## Measurement (Provided)

Fused QKV run:

```text
alloc_mib(before/bwd/step/zero)=(9553.23/12155.12/12155.12/9553.23)
reserved_mib(before/bwd/step/zero)=(9874.00/19122.00/19122.00/19122.00)
phase_peak_alloc_mib(fwd/bwd/opt/zero)=(16095.79/16822.98/13037.12/12155.12)
step_peak_alloc_mib=16822.98 step_peak_reserved_mib=19122.00
step 00002/00300 (0.67%) | dt: 1442.57ms | tok/sec: 22,715 | bf16_mfu: 72.67
```

Baseline split-QKV run:

```text
alloc_mib(before/bwd/step/zero)=(9529.23/12121.25/12121.25/9529.23)
reserved_mib(before/bwd/step/zero)=(10304.00/19170.00/19170.00/19170.00)
phase_peak_alloc_mib(fwd/bwd/opt/zero)=(16205.04/16714.26/13003.25/12121.25)
step_peak_alloc_mib=16714.26 step_peak_reserved_mib=19170.00
step 00002/00300 (0.67%) | dt: 1468.91ms | tok/sec: 22,307 | bf16_mfu: 71.36
```

## Observed Tradeoff

- Throughput improved (`22,307 -> 22,715 tok/s`, ~`+1.8%`)
- Forward peak improved (`16205.04 -> 16095.79 MiB`, ~`-109 MiB`)
- Backward peak increased (`16714.26 -> 16822.98 MiB`, ~`+109 MiB`)
- Step peak increased by the same amount (~`+109 MiB`)

Interpretation: fused QKV is currently a speed optimization with a small backward-memory tax.

## How To Run

Default speedrun already enables it:

```bash
bash runs/speedrun_small.sh
```

Explicit toggles:

```bash
FUSED_QKV=1 bash runs/speedrun_small.sh
FUSED_QKV=0 bash runs/speedrun_small.sh
```

## Next Iteration

If we want speed gain without backward memory increase, the next step is a custom QKV autograd path that avoids materializing full `grad_qkv` buffers in backward.
