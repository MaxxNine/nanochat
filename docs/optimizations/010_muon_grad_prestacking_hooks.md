# Optimization 010: Muon Gradient Pre-Stacking Hooks

Date: 2026-03-04
Status: Implemented

## Objective

Reduce backward-phase and optimizer-step peak VRAM by eliminating the `torch.stack()` copy for Muon gradients and freeing individual Muon grad tensors during backward instead of after `optimizer.step()`.

## Motivation (Memory Attribution)

Optimization 006 reduced backward peak by hooking **AdamW** parameters (embeddings, lm_head, scalars) to free their gradients immediately during backward. However, **Muon** parameters (all 2D matrices in transformer blocks: attention Q/K/V/O projections + MLP up/down) were left untouched. These are the bulk of the model (~45% of total params, ~71M params at 768 dim).

Without hooks, the Muon gradient memory lifecycle was:

1. **Backward:** Individual grad tensors accumulate one-by-one, all staying alive
2. **`_step_muon`:** `torch.stack([pp.grad for pp in params])` creates a **full duplicate** of all Muon grads
3. **`_step_muon`:** `torch.stack(params)` creates a copy of all Muon params
4. **Peak during step:** `params + grads + stacked_grads + stacked_params = 4x Muon param size`

The `torch.stack()` for grads is pure waste — it copies data that already exists in individual tensors, just to get a contiguous layout for the fused kernel.

## What Was Implemented

### Core: Pre-Allocated Contiguous Grad Buffer

New method `_install_muon_grad_hooks()` in `MuonAdamW` (inherited by `MuonAdamW8bit`):

- For each Muon param group (params grouped by shape), **pre-allocates** a contiguous buffer of shape `(N, *param_shape)` where N = number of params in the group
- Pre-allocates a boolean `active_mask` of shape `(N,)` to track which params received gradients
- Registers `register_post_accumulate_grad_hook` on each Muon parameter

Each hook does three things:

```python
def _muon_hook(param, idx=i, buf=grad_buffer, mask=active_mask):
    if param.grad is None:
        return
    buf[idx].add_(param.grad)   # Accumulate into contiguous buffer
    mask[idx] = True             # Mark as active
    param.grad = None            # Free individual grad immediately
```

Key design decisions:
- **`add_` not `copy_`**: Correctly handles `grad_accum_steps > 1` — multiple backward passes accumulate into the same buffer
- **`active_mask` tracking**: Supports dynamic/frozen layers — only active params get their weights updated after the fused kernel
- **`param.grad = None`**: Individual grad tensor is freed immediately, CUDA allocator can reuse that memory for the next layer's activation computation

### Hooked Fast Path in `_step_muon`

When the grad buffer is present (hooks installed), `_step_muon` takes a new fast path:

```python
hooked_grad_buffer = state.get('_hooked_grad_buffer')
if hooked_grad_buffer is not None:
    # No torch.stack for grads — buffer is already contiguous and stacked
    stacked_params = torch.stack(params)
    muon_step_fused(hooked_grad_buffer, stacked_params, ...)

    # Copy back only active params
    if len(active_indices) == num_params:
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))
    else:
        # Skip inactive params to avoid applying weight-decay-only updates
        active_params = [params[i] for i in active_indices]
        active_values = [stacked_params[i] for i in active_indices]
        torch._foreach_copy_(active_params, active_values)

    # Zero buffers for next accumulation cycle
    hooked_grad_buffer.zero_()
    hooked_active_mask.zero_()
    return
```

### Integration Points

- `MuonAdamW.install_post_accum_hooks()` calls `_install_muon_grad_hooks()` after setting up AdamW hooks
- `MuonAdamW8bit.install_post_accum_hooks()` also calls `_install_muon_grad_hooks()` (inherited)
- `remove_post_accum_hooks()` cleans up both buffer state dicts and resets `_muon_hooked` flag

Files modified:
- [`nanochat/optim.py`](/home/gabrielmaxx/projects/gpt/nanochat/nanochat/optim.py): `_install_muon_grad_hooks()`, hooked fast path in `_step_muon()`, cleanup in `remove_post_accum_hooks()`

## Why This Works (Math / Memory Analysis)

### Memory Model

Let `P_muon` = total Muon parameter memory (e.g., 71M params x 2 bytes bf16 = ~142 MB).

**Before (no Muon hooks):**

| Phase | Live Memory | Formula |
|-------|------------|---------|
| Backward | params + all grads | `P_muon + P_muon = 2 * P_muon` |
| `torch.stack(grads)` | params + grads + stacked_grads | `3 * P_muon` |
| `torch.stack(params)` | params + grads + stacked_grads + stacked_params | `4 * P_muon` (peak) |

**After (with Muon hooks):**

| Phase | Live Memory | Formula |
|-------|------------|---------|
| Backward | params + grad_buffer + 1 transient grad | `2 * P_muon + O(1)` |
| `_step_muon` entry | params + grad_buffer (already stacked) | `2 * P_muon` |
| `torch.stack(params)` | params + grad_buffer + stacked_params | `3 * P_muon` (peak) |

**Saving: `P_muon` (one full copy eliminated from peak)**

For a 768-dim model: ~142 MB saved. Scales linearly with model size.

### Mathematical Equivalence

The hook accumulates identically to what `torch.stack()` would produce:

```
# Without hooks:
stacked_grads[i] = params[i].grad       (via torch.stack)

# With hooks:
grad_buffer[i] += params[i].grad        (via add_ in hook, starting from zero)
```

Since the buffer is zeroed at the end of each `_step_muon` call, and `add_` in the hook is the same accumulation PyTorch's autograd performs, the result is mathematically identical.

With gradient accumulation (`grad_accum_steps > 1`):

```
# Micro-step 1: hook fires, buf[i] += grad_micro1, param.grad = None
# Micro-step 2: hook fires, buf[i] += grad_micro2, param.grad = None
# step(): muon_step_fused uses buf[i] = grad_micro1 + grad_micro2 (correct)
```

### CUDA Fragmentation Benefit

Without hooks, N individual grad tensors are scattered across GPU memory. With hooks, there is one contiguous pre-allocated buffer. This reduces fragmentation pressure during backward, where the CUDA allocator must interleave grad allocations with activation memory. Individual grads freed by hooks become immediately reusable by the allocator for subsequent layers' activations.

## Interaction with Existing Features

- **`active_only_stack`**: Not used in hooked path (buffer processes all params, only copies back active ones). Consistent with the legacy fast path behavior.
- **`stack_chunk_size`**: Not used in hooked path (buffer is already allocated as one contiguous block). Chunking is a speed/recompilation tradeoff that doesn't apply when the buffer is pre-allocated.
- **`grad_accum_steps > 1`**: Works correctly. Hooks use `add_` to accumulate across micro-steps. Buffer is zeroed once per `_step_muon` call (once per macro-step).
- **Dynamic layers (frozen/unfrozen params)**: Handled via `active_mask`. Only params whose hooks fired get copied back. Inactive params' buffer slots remain zero; their weights are untouched.

## How To Run

Muon hooks activate automatically with `--post-accum-hooks`:

```bash
POST_ACCUM_HOOKS=1 bash runs/speedrun_small.sh
```

No separate flag needed — if post-accum hooks are enabled, Muon grad hooks are installed alongside AdamW hooks.
