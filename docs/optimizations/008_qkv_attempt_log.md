# Optimization 008: QKV Attempts Log (What We Tried, What Worked)

Date: 2026-03-03
Status: Closed (baseline default retained)

## Objective

Reduce memory peak around attention QKV while preserving or improving throughput.

## Baseline Reference

Split projections (`c_q`, `c_k`, `c_v`), no custom QKV autograd.

```text
phase_peak_alloc_mib(fwd/bwd/opt/zero)=(16205.04/16714.26/13003.25/12121.25)
step_peak_alloc_mib=16714.26
dt=1468.91ms | tok/sec=22,307 | bf16_mfu=71.36
```

## Attempt 1: Fused QKV Projection (`c_qkv`)

Change:

- Replace `c_q/c_k/c_v` with one `c_qkv` linear + `torch.split`

Observed:

```text
phase_peak_alloc_mib(fwd/bwd/opt/zero)=(16095.79/16822.98/13037.12/12155.12)
step_peak_alloc_mib=16822.98
dt=1442.57ms | tok/sec=22,715 | bf16_mfu=72.67
```

Interpretation:

- Faster (`+1.8% tok/sec`)
- Better forward memory (`-109 MiB`)
- Slightly worse backward memory (`+109 MiB`)

## Attempt 2: Split-QKV Custom Backward (`EfficientQKV`)

Change:

- Keep split projections
- custom `autograd.Function` to accumulate `dX` in one buffer

Observed:

```text
phase_peak_alloc_mib(fwd/bwd/opt/zero)=(15953.00/16749.10/13003.25/12121.25)
step_peak_alloc_mib=16749.10
dt=1584.58ms | tok/sec=20,679 | bf16_mfu=66.15
```

Interpretation:

- Memory gain too small to justify it
- Throughput regression was large

## Attempt 3: Fused-QKV Custom Backward (`EfficientFusedQKV`)

Change:

- Keep fused forward
- custom fused `autograd.Function` to avoid explicit `grad_qkv` assembly for `dX`

Observed:

- Similar pattern to Attempt 2 in this stack: very small memory movement, clear speed penalty.
- Not promoted.

## Final Decision

- Keep only two maintained paths:
  - Baseline split QKV (default)
  - Optional fused QKV (`--fused-qkv`)
- Remove experimental custom QKV autograd paths from training code.
- Keep baseline as default in `runs/speedrun_small.sh` (`FUSED_QKV=0`).

## Practical Guidance

- Use baseline split QKV if memory headroom is the priority.
- Try fused QKV (`--fused-qkv`) when throughput is priority and a small backward-memory increase is acceptable.
