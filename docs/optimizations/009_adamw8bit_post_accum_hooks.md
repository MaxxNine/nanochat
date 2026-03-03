# Optimization 009: AdamW 8-bit + Post-Accumulate Hooks

Date: 2026-03-03  
Status: Implemented (experimental) and enabled by default in `speedrun_small`

## Objective

Lower AdamW memory footprint further by combining:

- step-in-backward grad lifetime reduction (`post-accum-hooks`)
- 8-bit AdamW optimizer states (`bitsandbytes.AdamW8bit`)

## Motivation

Optimization 006 reduced the backward peak by freeing AdamW grads early.  
After that, the next large AdamW memory component is optimizer state (`exp_avg`, `exp_avg_sq`) for:

- `lm_head`
- `wte`
- `value_embeds`
- scalar groups

Moving AdamW state to 8-bit can reduce persistent optimizer memory while keeping Muon unchanged.

## What Was Implemented

- New hybrid optimizer in [`nanochat/optim.py`](/home/gabrielmaxx/projects/gpt/nanochat/nanochat/optim.py):
  - `MuonAdamW8bit`
  - AdamW groups use `bitsandbytes.optim.AdamW8bit`
  - Muon groups keep existing `muon_step_fused` path (unchanged)
- Hook compatibility in [`nanochat/optim.py`](/home/gabrielmaxx/projects/gpt/nanochat/nanochat/optim.py):
  - `install_post_accum_hooks()` is implemented for 8-bit mode
  - hook path updates AdamW8bit immediately and frees grad (`param.grad = None`)
  - `step()` skips AdamW8bit update when hooks are active (avoids double-step)
- Hyperparameter sync safety:
  - LR/betas/eps/wd are synced from training param groups into bnb groups before each 8-bit update
  - this preserves scheduler behavior in both normal-step and hook-step paths
- Optimizer factory wiring in [`nanochat/gpt.py`](/home/gabrielmaxx/projects/gpt/nanochat/nanochat/gpt.py):
  - `setup_optimizer(..., adamw_8bit, adamw_8bit_min_size)`
  - single-rank selection:
    - `adamw_8bit=False` -> `MuonAdamW`
    - `adamw_8bit=True`  -> `MuonAdamW8bit`
  - distributed (`world_size > 1`) with 8-bit is currently rejected
- CLI flags in [`scripts/base_train.py`](/home/gabrielmaxx/projects/gpt/nanochat/scripts/base_train.py):
  - `--adamw-8bit`
  - `--adamw-8bit-min-size`
- Runner defaults in [`runs/speedrun_small.sh`](/home/gabrielmaxx/projects/gpt/nanochat/runs/speedrun_small.sh):
  - `POST_ACCUM_HOOKS=1` (already default)
  - `ADAMW_8BIT=1` (new default)

## Current Constraints

- Single-rank only for `--adamw-8bit` (distributed path not implemented yet).
- Requires `bitsandbytes` installed in the environment.
- In hook mode with `grad_accum_steps > 1`, AdamW still updates per micro-step (same caveat as Optimization 006).

## Default Behavior Now

Running:

```bash
bash runs/speedrun_small.sh
```

now enables both:

- `--post-accum-hooks`
- `--adamw-8bit`

by default.

## How To Toggle

Disable 8-bit AdamW:

```bash
ADAMW_8BIT=0 bash runs/speedrun_small.sh
```

Disable post-accum hooks:

```bash
POST_ACCUM_HOOKS=0 bash runs/speedrun_small.sh
```

Control bnb threshold:

```bash
ADAMW_8BIT=1 ADAMW_8BIT_MIN_SIZE=4096 bash runs/speedrun_small.sh
```

## Notes

- Checkpoint save/load includes bnb optimizer state via `_adamw8_state_dict`.
- This optimization is intentionally isolated to AdamW groups; Muon math and communication behavior are unchanged.
