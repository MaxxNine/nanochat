# Dynamic Scheduler Tasks

Goal: improve `dynamic_suffix` quality/speed tradeoff toward 4090-friendly training.

## Active Tasks
- [x] 1) Final-recovery optimization behind feature flags
  - Add staged final recovery (`12/14/16/18`-style ramp) so we recover quality with less full-update time.
  - Status: initial implementation added (`--dyn-final-recovery-*` flags).
- [x] 2) Relevance metric swap behind feature flags
  - Add saliency-like relevance signal (`|g * p|`) and keep current gradient-ratio baseline.
  - Status: initial implementation added (`--dyn-relevance-metric`).
- [ ] 3) Probe cadence optimization
  - Tune/reshape probe schedule to reduce probe overhead while preserving adaptation quality.
  - Candidate: denser probes early, sparser probes later.

## Next Experiments
- [ ] A/B: `grad_ratio` vs `saliency_abs_gp` under same seed/time budget.
- [ ] A/B: `final_recovery_mode=full` vs `final_recovery_mode=staged`.
- [ ] Joint sweep: staged recovery + saliency metric + cadence tuning.
