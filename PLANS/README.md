# PLANS Index (Container View)

## Optimization Equation
`time_to_quality ~= required_flops / effective_flops_per_second + pipeline_overhead`

Containers are organized by which part of this equation (or constraint) they improve.

## Container A: Reduce Required FLOPs
Goal:
- Achieve target quality with fewer total training FLOPs.

Files:
- `PLANS/00-single-4090-24h-target-map.md`
- `PLANS/FLOP_DETAILMENT.md`
- `PLANS/01-compute-budget-and-target-architecture.md`
- `PLANS/02-data-strategy-for-low-compute.md` (planned)

## Container B: Increase Effective FLOPs/s
Goal:
- Convert 4090 hardware capability into higher real throughput.

Files:
- `PLANS/00-single-4090-24h-target-map.md`
- `PLANS/04-4090-systems-optimization-checklist.md` (planned)

## Container C: Memory/OOM Fit (24GB)
Goal:
- Keep runs stable in 24GB VRAM while preserving useful batch/model scale.

Files:
- `PLANS/00-single-4090-24h-target-map.md`
- `PLANS/01-compute-budget-and-target-architecture.md`
- `PLANS/04-4090-systems-optimization-checklist.md` (planned)

## Container D: Convergence Efficiency
Goal:
- Reduce steps/tokens needed to reach target quality.

Files:
- `PLANS/00-single-4090-24h-target-map.md`
- `PLANS/02-data-strategy-for-low-compute.md` (planned)
- `PLANS/03-self-bootstrapping-design.md` (planned)

## Container E: End-to-End Latency And Experiment Velocity
Goal:
- Reduce full pipeline latency and increase iteration speed.

Files:
- `PLANS/00-single-4090-24h-target-map.md`
- `PLANS/05-experiment-matrix-and-stop-rules.md` (planned)

## Notes
- Some targets intentionally span multiple containers (e.g., sequence curriculum affects throughput and memory).
- New planning files should list their primary and secondary containers near the top.
