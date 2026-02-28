# 08 - Checkpoint and Export Cost

## Scope
This file maps Step 8 (checkpoint selection/export) to storage and I/O cost.

Cost tags: `io`, `bandwidth`, `latency`, `memory`

## Variables
- `P`: parameters
- `b_model`: bytes/param for saved weights
- `b_master`: bytes/param for fp32 master weights (if present)
- `b_m`, `b_v`: bytes/param for Adam moments
- `BW_disk`: effective write/read bandwidth
- `N_ckpt`: number of checkpoints

## Size Equations
Model-only checkpoint:

`size_model ~= P * b_model`

Training-state checkpoint (typical Adam training):

`size_train ~= P * (b_model + b_master + b_m + b_v) + metadata`

Example at `P=1.5B`:
- model-only bf16 (`b_model=2`) -> `~3.0 GB`
- with fp32 master + Adam moments (`2+4+4+4=14`) -> `~21 GB`

## Time Equations
Write time:

`t_write ~= size / BW_disk`

Read/load time:

`t_read ~= size / BW_disk`

Total checkpoint overhead in a run:

`t_ckpt_total ~= N_ckpt * (t_write + optional_verify_time)`

## Practical Consequence
Too-frequent full-state checkpoints can consume meaningful wall-clock time and disk bandwidth.

## Optimization Targets
1. Keep frequent checkpoints model-only when possible.
2. Save full training state less frequently (recovery anchors).
3. Align checkpoint cadence with eval cadence and risk tolerance.
