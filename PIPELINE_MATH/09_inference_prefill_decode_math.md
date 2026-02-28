# 09 - Inference Math (Prefill + Decode)

## Scope
This file maps Step 9 (inference engine) to latency and memory math.

Cost tags: `compute`, `memory`, `latency`, `bandwidth`

## Variables
- `T`: prompt length (tokens)
- `G`: generated tokens
- `L`: number of layers
- `d`: model width
- `n_kv`: KV heads
- `d_h`: head dimension
- `B`: batch size (requests batched together)
- `b`: bytes per KV value (bf16 -> 2)

## Latency Decomposition
Total response latency:

`t_total ~= t_prefill + t_decode + t_app`

Where:
- `t_prefill`: process full prompt once
- `t_decode`: autoregressive token-by-token generation
- `t_app`: non-model overhead (server/UI/streaming)

## Prefill Cost
Without cache reuse inside prefill, attention has quadratic prompt interaction:

`t_prefill ~ a*T + b*T^2`

Interpretation:
- long prompts strongly increase first-token latency.

## Decode Cost (with KV cache)
Per generated token:
- matrix part is roughly constant per layer,
- attention to past KV is linear in current context.

So decode time behaves like:

`t_token_decode(C) ~ c0 + c1*C`

with `C` current context length.

## KV Cache Memory
Approximate KV cache bytes:

`M_kv ~= 2 * L * B * C * n_kv * d_h * b`

(`2` for K and V)

Example (`L=18`, `B=1`, `C=2048`, `n_kv=8`, `d_h=128`, `b=2`):
- `M_kv ~= 151,000,000 bytes` (~144 MiB)

If `n_kv` is reduced (GQA/MQA), KV memory and decode bandwidth drop proportionally.

## Practical Consequence
Inference bottlenecks are usually:
1. prefill latency for long prompts,
2. decode tokens/s under KV bandwidth pressure,
3. batching/queue policy effects.

## Optimization Targets
1. Reduce prefill latency for common prompt sizes.
2. Improve decode tokens/s while respecting quality.
3. Keep KV cache within safe VRAM headroom for target concurrency.
