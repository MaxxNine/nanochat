# 06 - SFT Step Math

## Scope
This file maps Step 6 (supervised finetuning) to cost and signal math.

Cost tags: `compute`, `memory`, `latency`

## Variables
- `TBS`: total tokens per step
- `r_mask`: fraction of tokens that contribute to supervised loss
- `f_tok`: FLOPs/token during training (similar order to pretrain)
- `tok_s_sft`: SFT throughput
- `N_tok_sft`: total SFT tokens

## Core Equations
Compute is still paid on full tokens:

`flops_step_sft ~= f_tok * TBS`

But supervised signal tokens per step are:

`signal_tokens_step = r_mask * TBS`

Total supervised signal:

`signal_tokens_total = r_mask * N_tok_sft`

Interpretation:
- low `r_mask` wastes compute (many tokens with no gradient signal),
- packing/dialog formatting can increase effective training signal.

## Practical Consequence
If two SFT datasets have same raw token count but different `r_mask`:
- the one with higher `r_mask` gives more learning signal at same compute.

## Quality-vs-Time Lens
SFT objective is not "largest token count"; it is:
- maximize behavior gain per second,
- avoid overfitting or catastrophic drift from pretrained base.

## Optimization Targets
1. Increase `r_mask` through better conversation packing/masking rules.
2. Keep sequence lengths where throughput is healthy on 4090.
3. Use short eval loops to stop once alignment gain saturates.
