# Single 4090 in <24h: Target Map (Containerized v1)

## Mission
Train a nanochat model on **one RTX 4090 (24GB)** in **under 24 hours** while keeping capability near current GPT-2-grade outcomes.

## Principles
- Prefer approaches that are reproducible on home hardware without requiring paid external teacher models.
- Treat the 24h target as a directional optimization goal, while preserving long-term local-trainability.
- Optimize for open, iterative improvement loops that many individuals can run independently.

## Success Criteria
- Primary: reach or beat GPT-2-grade capability target (CORE ~= 0.2565 reference) in <=24h wall-clock on 1x4090.
- Secondary: stable chat behavior after SFT, reproducible run script, no OOM on 24GB VRAM.
- Tertiary: total cost low enough to keep iterative experimentation practical.

## Optimization Model
We optimize this equation:

`time_to_quality ~= required_flops / effective_flops_per_second + pipeline_overhead`

Subject to constraints:
- `VRAM fit` (no OOM on 24GB)
- `stability` (no divergence/collapse)
- `reproducibility` (home hardware repeatability)

## Target Catalog
| ID | Target | What to try first | Why this is a strong target | Success signal |
|---|---|---|---|---|
| T1 | **Re-budget compute with scaling laws** | Recompute optimal depth/width/tokens for the 4090 FLOP budget; stop assuming 8xH100-optimal ratios are valid. | The current recipe is tuned for much larger compute. Wrong scaling assumptions can waste most of the 24h budget. | New baseline config that improves CORE-per-FLOP and fits 24GB. |
| T2 | **Data quality over raw token count** | Build a higher-signal pretraining mix (dedup, quality filtering, domain balancing) instead of brute-force tokens. | On constrained compute, cleaner data can outperform much larger noisy corpora. | Better val_bpb/CORE at same token budget. |
| T3 | **Self-bootstrapping training methods** | Test self-distillation from earlier checkpoints, EMA/weight averaging, progressive curriculum transfer, and replay of high-learning-value samples. | Preserves “from-scratch at home” philosophy while still improving sample efficiency and stability. | Higher CORE at same compute budget without dependence on external teacher models. |
| T4 | **Short-horizon optimizer tuning** | Retune LR schedule, warmup, weight decay, batch schedule specifically for <=24h runs. | Defaults built for longer training often underperform in compressed schedules. | Faster early CORE gains and better final score under fixed wall-clock. |
| T5 | **Sequence-length curriculum** | Train with shorter contexts first for throughput, then extend context late. | 4090 throughput is the hard bottleneck; curriculum can buy more optimization steps per hour. | Higher tokens/sec without quality collapse; better final CORE in 24h. |
| T6 | **Memory-efficient architecture tweaks** | Evaluate GQA/MQA, KV-head reduction, tied embeddings, and lightweight FFN choices that preserve quality. | 24GB VRAM is a hard constraint; memory savings can unlock larger effective batch/model. | No OOM + higher effective batch and improved wall-clock efficiency. |
| T7 | **Single-GPU systems optimization** | Maximize kernels and runtime efficiency (FlashAttention path, compile/fused ops, dataloader overlap, I/O caching). | Even 10-20% throughput wins are huge when the time budget is fixed. | Sustained tokens/sec increase and higher MFU on 4090. |
| T8 | **Tokenizer strategy optimization** | Test keeping an existing strong tokenizer vs retraining, plus vocab-size tradeoffs for compression/speed. | Tokenization affects both compression and training dynamics; bad choices waste compute immediately. | Lower bpb and/or faster training to same quality. |
| T9 | **Stage compression (pretrain -> SFT pipeline)** | Shorten or partially merge stages; tune SFT to avoid overpaying in post-pretrain time. | The mission is end-to-end <24h, not only pretraining speed. | Full pipeline completion under 24h with acceptable chat quality. |
| T10 | **Fast proxy eval + ablation harness** | Define cheap proxy checkpoints/metrics to kill weak ideas early. | Fast feedback loops are mandatory for discovering a winning recipe quickly. | Reliable proxy that predicts final CORE ranking with low run cost. |

## Containers (What Belongs To What)
### Container A: Reduce Required FLOPs (numerator)
Targets:
- `T1` Re-budget compute with scaling laws
- `T2` Data quality over raw token count
- `T8` Tokenizer strategy optimization

Primary metrics:
- CORE-per-FLOP
- val_bpb at fixed FLOP budget
- FLOPs needed to hit target CORE

### Container B: Increase Effective FLOPs/s (denominator)
Targets:
- `T7` Single-GPU systems optimization
- `T5` Sequence-length curriculum (throughput side)

Primary metrics:
- tokens/sec
- effective_flops/sec
- MFU and step time stability

### Container C: Memory/OOM Fit (24GB constraint)
Targets:
- `T6` Memory-efficient architecture tweaks
- `T5` Sequence-length curriculum (memory side)

Primary metrics:
- peak VRAM
- OOM rate / stability
- effective batch unlocked by memory savings

### Container D: Convergence Efficiency (steps/tokens to quality)
Targets:
- `T3` Self-bootstrapping training methods
- `T4` Short-horizon optimizer tuning
- `T2` Data quality over raw token count

Primary metrics:
- CORE vs step
- CORE vs token
- early-epoch learning slope

### Container E: End-to-End Latency And Experiment Velocity
Targets:
- `T9` Stage compression (pretrain -> SFT)
- `T10` Fast proxy eval + ablation harness

Primary metrics:
- wall-clock to usable chat model
- number of completed experiments/week
- proxy-to-final ranking accuracy

## Suggested Next Planning Files (By Container)
1. `PLANS/FLOP_DETAILMENT.md` (A: cost model foundation)
2. `PLANS/01-compute-budget-and-target-architecture.md` (A/C foundation)
3. `PLANS/02-data-strategy-for-low-compute.md` (A/D)
4. `PLANS/03-self-bootstrapping-design.md` (D)
5. `PLANS/04-4090-systems-optimization-checklist.md` (B/C)
6. `PLANS/05-experiment-matrix-and-stop-rules.md` (E)
