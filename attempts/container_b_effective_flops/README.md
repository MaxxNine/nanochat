# Container B Attempts: Effective FLOPs/s

## Goal
Benchmark attention backend/kernel throughput on current hardware, especially for non-FA3 paths.

## Script
- `attention_backend_bench.py`

What it benchmarks:
- `nanochat.flash_attention` wrapper path (FA3 when available, SDPA fallback otherwise)
- raw `torch.nn.functional.scaled_dot_product_attention`
- optional forced SDPA backends (flash/efficient/math/cudnn), if PyTorch exposes them

Reported metrics:
- latency (ms/iter)
- estimated attention TFLOP/s (QK + AV only)
- speedup vs baseline backend
- relative L2 output error vs baseline backend

## Example
```bash
poetry run python attempts/container_b_effective_flops/attention_backend_bench.py \
  --device auto \
  --dtype bfloat16 \
  --batch 4 \
  --seq-len 2048 \
  --heads 13 \
  --head-dim 128 \
  --iters 30 \
  --warmup 10 \
  --json-out attempts/results/container_b/attention_backend_$(date +'%Y%m%d_%H%M%S').json
```

## Notes
- This script isolates attention compute and not full training-step overhead.
- Estimated TFLOP/s here is an attention-only metric, useful for backend comparisons.
