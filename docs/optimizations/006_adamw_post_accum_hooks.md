# Optimization 006: AdamW Post-Accumulate Hooks (Step-In-Backward)

Date: 2026-03-03
Status: Implemented and measured

## Objective

Lower backward-phase peak VRAM by stepping AdamW parameters as soon as their gradients are produced, instead of waiting for `optimizer.step()` after full backward completes.

## Motivation (Memory Attribution)

Phase attribution showed the dominant peak in backward, where gradients accumulate and remain live until the later optimizer phase. For AdamW parameters (`lm_head` + `value_embeds`), this leaves large gradient buffers resident for most of backward.

Core idea: use `register_post_accumulate_grad_hook` to update each AdamW parameter immediately when its grad is finalized, then free that grad tensor right away.

## What Was Implemented

- Optimizer hooks in [`nanochat/optim.py`](/home/gabrielmaxx/projects/gpt/nanochat/nanochat/optim.py):
  - `MuonAdamW.install_post_accum_hooks()`
  - `MuonAdamW.remove_post_accum_hooks()`
  - Shared per-parameter AdamW fused update helper (`_adamw_update_param`)
  - Hook path runs fused `adamw_step_fused(...)` in `torch.no_grad()` and then sets `p.grad = None`
  - `_step_adamw()` becomes no-op while hooks are active (`self._adamw_hooked=True`)
- Training flag and schedule timing in [`scripts/base_train.py`](/home/gabrielmaxx/projects/gpt/nanochat/scripts/base_train.py):
  - New CLI flag: `--post-accum-hooks` (default off)
  - Installs hooks after optimizer creation
  - Moves LR/momentum/weight-decay assignment to **before** `loss.backward()`
  - Warns when `grad_accum_steps > 1` (AdamW steps once per micro-step in current hook design)
- Runner wiring in [`runs/speedrun_small.sh`](/home/gabrielmaxx/projects/gpt/nanochat/runs/speedrun_small.sh):
  - New env flag: `POST_ACCUM_HOOKS=0|1`
  - Pass-through to `--post-accum-hooks`
- Single-rank torchrun selection fix in [`nanochat/gpt.py`](/home/gabrielmaxx/projects/gpt/nanochat/nanochat/gpt.py):
  - With `torchrun --nproc_per_node=1`, use `MuonAdamW` (not `DistMuonAdamW`) so hook API is available.

## Important Constraints

- Current implementation is single-rank only (`world_size == 1`) when hooks are enabled.
- Dynamic suffix relevance probe (`block_relevance_scores`) reads Muon block grads, not embedding/lm-head grads, so early AdamW grad release is safe for this workflow.
- With `grad_accum_steps > 1`, AdamW updates happen per micro-step (not once per outer step). Use with awareness.

## Measurement (Provided)

Observed with hooks enabled:

```text
alloc_mib(before/bwd/step/zero)=(9529.23/12121.25/12121.25/9529.23)
reserved_mib(before/bwd/step/zero)=(10306.00/19170.00/19170.00/19170.00)
phase_peak_alloc_mib(fwd/bwd/opt/zero)=(16205.04/16714.16/13003.25/12121.25)
step_peak_alloc_mib=16714.16 step_peak_reserved_mib=19170.00
step 00002/00300 (0.67%) | loss: 9.032235 | lrm: 1.00 | dt: 1472.23ms | tok/sec: 22,257 | bf16_mfu: 71.20 | epoch: 1 | active_layers: 24/24 (P) | total time: 0.00m
```

Baseline reference (same optimization stack before this change): backward peak ~`18157 MiB`.

## Gain vs Baseline

- Backward peak alloc: `-1442.84 MiB` (`18157.00 -> 16714.16`, `-7.95%`)
- Step peak alloc: now aligned with reduced backward peak (`16714.16 MiB`)
- Optimizer-phase peak (`opt`) is no longer dominant in this measured step (`13003.25 MiB`)

## Why This Works (Math Intuition)

Let:
- `M_base` = memory not affected by AdamW-grad lifetime
- `G_adamw` = live AdamW grad memory

Before hooks (during most of backward):

`M_peak_old ~= M_base + G_adamw`

After hooks:
- each AdamW grad is consumed and freed immediately after accumulation
- only a much smaller transient subset of AdamW grads is live at a time

So peak becomes approximately:

`M_peak_new ~= M_base + G_adamw_transient`, with `G_adamw_transient << G_adamw`

Hence `M_peak_new < M_peak_old`, matching the measured drop.

## How To Run

```bash
POST_ACCUM_HOOKS=1 \
DEBUG_MEM_EVERY=1 \
bash runs/speedrun_small.sh
```

## Interpretation

This is a high-impact memory optimization: it attacks the true peak owner (backward grad lifetime), not just forward intermediates. It materially lowers peak allocation on 1x4090 runs while preserving the existing fused AdamW update path.
