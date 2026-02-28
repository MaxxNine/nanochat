# 07 - Evaluation Math

## Scope
This file maps Step 7 (evaluation) to predictable runtime cost.

Cost tags: `compute`, `latency`, `io`

## Variables
- `N_eval_tok`: number of evaluation tokens
- `tok_s_eval`: eval throughput (tokens/s)
- `f_tok_train`: train FLOPs/token
- `f_tok_eval`: eval FLOPs/token

## Core Equations
No backward/optimizer in eval, so:

`f_tok_eval ~= f_tok_train / 3` (rule-of-thumb)

Eval time:

`t_eval ~= N_eval_tok / tok_s_eval`

Full periodic overhead:

`t_eval_total ~= n_evals * t_eval + checkpoint_io_overhead`

## Tradeoff
Frequent evals improve decision quality but steal training time.

If eval cadence is too aggressive:
- total wall-clock-to-quality can get worse even with better monitoring.

## Recommended Control
Use staged cadence:
1. sparse eval early,
2. denser eval near target quality.

This minimizes total interruption cost while preserving decision confidence.

## Optimization Targets
1. Keep eval representative but lightweight.
2. Track ranking stability of candidates, not only absolute metrics.
3. Budget eval explicitly as a fixed percentage of wall-clock.
