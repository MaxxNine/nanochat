# CODE OPTIMIZATIONS

## Scope
Analysis of:
- `scripts/base_train.py`
- `nanochat/gpt.py`

Focus: redundant structures, duplicated work, data movement/synchronization, and hardware-level efficiency (GPU utilization, memory traffic, kernel-launch overhead).

## Executive summary
The codebase is already strong on major performance practices (meta-device init, `to_empty`, bf16 autocast, `torch.compile`, gradient accumulation, `set_to_none=True`, chunked CE option). The largest remaining gains are from reducing GPU<->CPU sync points and eliminating repeated per-step Python/dtype overhead in hot paths.

## High-impact optimization findings

1. **Repeated GPU synchronization in block relevance scoring**
- Location: `scripts/base_train.py:443-465`
- Issue: `p.grad.float().pow(2).sum().item()` / `...sum().item()` inside nested loops triggers many device-to-host syncs during probe steps.
- Impact: stalls GPU pipeline; probe steps become disproportionately slow.
- Recommendation: accumulate relevance as device tensors per block and call `.item()` once per block (or once total after stacking). Avoid per-parameter `.item()`.

2. **Unbounded per-step history list growth**
- Location: `scripts/base_train.py:655`, `scripts/base_train.py:801`, summary at `1060-1067`
- Issue: `dyn_active_layers_hist.append(...)` stores one entry every step.
- Impact: memory growth for long runs; unnecessary Python heap pressure.
- Recommendation: keep running `min/max/sum/count` instead of storing full history.

3. **Value embedding lookup uses string keys per layer per forward**
- Location: `nanochat/gpt.py:615`, `620`, `625`
- Issue: repeated `str(i)` conversion + `ModuleDict` membership checks in the critical forward loop.
- Impact: avoidable Python overhead on every layer, every step.
- Recommendation: replace with indexable structure (e.g., `ModuleList` aligned to layers with `None` sentinel via plain list metadata) or precomputed per-layer callable/reference table.

4. **Chunked CE does repeated dtype conversion per chunk**
- Location: `nanochat/gpt.py:161-163`, `277-281`
- Issue: each chunk may do `w_chunk.to(gemm_dtype)` in both forward and backward.
- Impact: extra kernel launches and memory bandwidth overhead when weights are not already in GEMM dtype.
- Recommendation: keep LM head weight in the chosen GEMM dtype for this path, or pre-cast once per call outside chunk loop when safe.

5. **Autoregressive generation recomputes full prefix every token**
- Location: `nanochat/gpt.py:673-685`
- Issue: `self.forward(ids)` on growing `ids` each token gives O(T^2) inference compute.
- Impact: major generation slowdown, especially for long contexts.
- Recommendation: use KV cache during generation (single-token decode step after prompt prefill).

## Medium-impact findings

6. **`row_idx` reallocated every CE backward chunk**
- Location: `nanochat/gpt.py:290`
- Issue: `torch.arange(n, ...)` is inside chunk loop.
- Impact: repeated small allocation/kernel launch overhead.
- Recommendation: move `row_idx` allocation outside the loop.

7. **FP8 evaluation swap rebuilds modules repeatedly**
- Location: `scripts/base_train.py:293-338`, used in eval/sample blocks
- Issue: `disable_fp8` scans all modules and instantiates replacement `nn.Linear` modules on each invocation.
- Impact: CPU overhead and allocator churn for frequent eval/sample.
- Recommendation: cache swap metadata once (or implement reversible fast-path wrapper) and reuse across calls.

8. **Dynamic suffix compile lookup inside micro-step loop**
- Location: `scripts/base_train.py:824-846`
- Issue: `get_or_compile_suffix_fn(active_start_this_step)` is called each micro-step though `active_start_this_step` is constant per outer step.
- Impact: small but avoidable Python overhead.
- Recommendation: resolve compiled function once before entering `for micro_step in range(grad_accum_steps)`.

9. **Shape-grouping for optimizer is O(U*N) Python work**
- Location: `nanochat/gpt.py:573-575`
- Issue: for each unique shape, scans full matrix param list.
- Impact: initialization-only overhead (not training-step hot), but avoidable.
- Recommendation: build a `dict[shape] -> list[params]` in one pass.

## Low-impact / informational

10. **Prompt list recreated each sampling event**
- Location: `scripts/base_train.py:726-734`
- Recommendation: define static prompts once outside loop.

11. **Repeated final cast/softcap in both train baseline and inference branches**
- Location: `nanochat/gpt.py:644-656`
- Note: duplication is minor and acceptable; can be factored into helper for maintainability only.

## What is already well optimized
- Meta-device model construction + `to_empty` init path (`scripts/base_train.py:170-193`).
- BF16 autocast on CUDA and compile support (`scripts/base_train.py:128`, `349`).
- Memory-saving CE option via chunked exact CE (`nanochat/gpt.py:180-318`, `631-642`).
- `optimizer.zero_grad(set_to_none=True)` to reduce memory writes (`scripts/base_train.py:932`).
- Dataloader prefetch overlap pattern (`scripts/base_train.py:879`).
- Optional dynamic suffix and FP8 feature flags for controlled performance experiments.

## Suggested implementation order (highest ROI first)
1. Fix block relevance scoring sync pattern.
2. Add KV-cache generation path.
3. Remove `dyn_active_layers_hist` list growth.
4. Replace `ModuleDict` string-based value embedding lookup in forward.
5. Optimize chunked CE inner-loop allocations/conversions.
