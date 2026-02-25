# Attempts

Quick, isolated experiments to measure potential wins before touching core training code.

## Structure
- `container_a_required_flops/`: experiments that try to reduce **required FLOPs** (usually via approximations).
- `container_b_effective_flops/`: experiments that try to increase **effective FLOPs/sec** (kernel/backend throughput).
- `container_d_convergence/`: experiments that try to reduce steps/tokens needed for a target quality.

## Ground Rule
For exact dense matmul, you usually cannot reduce required FLOPs without changing computation.  
So Container A experiments here are approximation-oriented and should always report both:
- speed/FLOP savings
- approximation error

## Typical workflow
1. Run microbench scripts.
2. Keep only candidates with meaningful speedup and acceptable error.
3. Promote winners into a design doc before integration into training code.

## Poetry
Run scripts with:
```bash
poetry run python <script> [args]
```
