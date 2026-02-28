# 05 - Pretrain Step Math

## Scope
This file maps Step 5 (pretraining loop) to training-time math.

Cost tags: `compute`, `memory`, `bandwidth`, `latency`

## Variables
- `TBS`: total batch size in tokens (global tokens per optimizer step)
- `S`: sequence length
- `f_tok`: FLOPs/token (from Step 4)
- `dt`: step time (seconds/step)
- `tok_s`: throughput in tokens/second
- `N_steps`: number of optimizer steps
- `N_tok`: total trained tokens

## Core Equations
Tokens per step:

`TBS = microbatch_tokens * grad_accum_steps * world_size`

Token throughput relation:

`tok_s ~= TBS / dt`

Compute per step:

`flops_step = f_tok * TBS`

Total compute:

`flops_total = f_tok * N_tok = flops_step * N_steps`

Total time:

`time_total ~= N_steps * dt = N_tok / tok_s`

## 24h Capacity Equation
For a fixed run time `H` hours:

`N_tok_24h = tok_s * (H*3600)`

This is the hard bridge from system speed to achievable training tokens.

## Example (from recent dense baseline)
Observed:
- `tok_s ~= 95,950` (seq1024, bs2, ckpt off)
- `dt ~= 0.357 s`

Then:
- steps/day `~= 86400 / 0.357 ~= 242k`
- tokens/day `~= 95,950 * 86400 ~= 8.29e9`

So any math improvement must either:
- raise `tok_s`, or
- reduce tokens needed for target quality.

## Where Time Is Lost In Practice
1. GPU compute kernels (forward/backward matmuls and attention)
2. Memory traffic (activation/gradient movement)
3. Launch/runtime overhead
4. Data input stalls

## Optimization Targets
1. Increase `tok_s` without quality drop.
2. Reduce `f_tok` without breaking convergence.
3. Improve quality per token so required `N_tok` drops.
