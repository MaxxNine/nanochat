# 04 - Model FLOPs and VRAM Budget

## Scope
This file maps Step 4 (model configuration) to math constraints:
- compute cost (FLOPs/token),
- static memory (weights + optimizer),
- dynamic memory (activations).

Cost tags: `compute`, `memory`

## Variables
- `P`: total parameters
- `P_train`: trainable parameters
- `d`: model width (`d_model` / `n_embd`)
- `L`: number of transformer layers
- `m`: FFN expansion factor (`hidden_mult`, usually 4)
- `V`: vocab size
- `S`: sequence length
- `Bmu`: microbatch size (device batch size)
- `b_w`: bytes per weight (bf16 -> 2)
- `b_g`: bytes per gradient (bf16 -> 2)
- `b_opt`: optimizer bytes/param (Adam moments in fp32 -> ~8)

## Parameter Budget (first-order)
For dense decoder-only transformer:

`P ~= V*d (wte) + V*d (lm_head) + L*(4*d^2 + 2*m*d^2) + small_terms`

With `m=4`:

`P ~= 2*V*d + L*(12*d^2) + small_terms`

Interpretation:
- width/depth dominate quickly via `d^2` term,
- vocab dominates embedding/head memory.

## FLOPs/Token (nanochat estimator)
Current code-level estimate (training):

`flops_token = 6*(P - wte - value_embeds - scalars) + attn_term`

`attn_term = sum_layers(12 * n_head * head_dim * effective_seq_layer)`

If `n_head*head_dim = d`:

`attn_term = sum_layers(12 * d * effective_seq_layer)`

Interpretation:
- matrix multiplications (especially MLP) are the dominant term,
- attention sequence term grows with effective sequence/window.

## VRAM Budget (training approximation)
Static memory:

`M_static ~= P*b_w + P_train*b_g + P_train*b_opt + misc`

Dynamic memory (activations):

`M_act ~= k_act * Bmu * S * d * L * b_w`

Peak:

`M_peak ~= M_static + M_act + fragmentation`

`k_act` depends on implementation details and checkpointing.

## 4090 Anchors From Our Runs
From recent dense short profiles:
- `seq=1024, bs=2, ckpt=off`: peak ~`5.83 GiB`, `~95k tok/s`
- `seq=1024, bs=2, ckpt=on`: peak ~`5.44 GiB`, `~80.8k tok/s`
- `seq=1536, bs=1, ckpt=off`: peak ~`5.57 GiB`, `~80.9k tok/s`
- `seq=2048, bs=1, ckpt=off`: peak ~`5.81 GiB`, `~90.4k tok/s`

So checkpointing helped memory by ~5-8%, but reduced throughput ~15-16%.

## Optimization Targets
1. Keep `M_peak` under 24GB with margin (target <= 21-22GB).
2. Maximize `tok/s` at fixed quality.
3. Reduce `flops_token` through math/architecture changes, not only by shrinking model.
