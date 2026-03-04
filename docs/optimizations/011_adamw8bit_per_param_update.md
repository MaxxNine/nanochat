# Optimization 011: AdamW 8-bit Per-Parameter Hook Update

Date: 2026-03-04
Status: Implemented

## Objective

Eliminate O(N^2) Python overhead in the `MuonAdamW8bit` post-accumulate hook path by using bitsandbytes' per-parameter `update_step` API instead of calling the full `.step()` from each hook.

## Motivation (Algorithmic Complexity)

The original `MuonAdamW8bit.install_post_accum_hooks()` implementation had a subtle inefficiency. Each parameter's hook called:

```python
def _hook(param):
    self._sync_adamw8_hparams()   # Syncs ALL groups
    self._adamw8.step()           # Iterates ALL params
    param.grad = None
```

`self._adamw8.step()` is the bnb optimizer's full `.step()` method, which internally iterates over **every parameter group and every parameter**:

```python
# Inside bnb Optimizer8bit.step():
for gindex, group in enumerate(self.param_groups):
    for pindex, p in enumerate(group["params"]):
        if p.grad is None:
            continue
        self.update_step(group, p, gindex, pindex)
```

Since each hook sets `param.grad = None` after stepping, subsequent hooks' `.step()` calls find that parameter already processed and skip it. But the iteration still happens. With N AdamW parameters:

- Hook 1 fires: `.step()` iterates N params, updates 1
- Hook 2 fires: `.step()` iterates N params, updates 1
- ...
- Hook N fires: `.step()` iterates N params, updates 1

Total Python iteration work: **O(N^2)** for O(N) actual updates.

Additionally, `_sync_adamw8_hparams()` was called N times per backward pass (once per hook), redundantly copying the same lr/betas/eps/weight_decay values.

## What Was Implemented

### Pre-Computed Parameter Index Map

In `MuonAdamW8bit.__init__()`, after creating the bnb optimizer:

```python
self._param_bnb_idx: dict[int, tuple[dict, int, int]] = {}
for gindex, bnb_group in enumerate(self._adamw8.param_groups):
    for pindex, p in enumerate(bnb_group["params"]):
        self._param_bnb_idx[id(p)] = (bnb_group, gindex, pindex)
```

Maps each parameter's `id()` to its bnb `(group, group_index, param_index)` triple. O(1) dict lookup per parameter.

### Per-Parameter Hook Using bnb Internal API

```python
def _hook(param):
    if param.grad is None:
        return
    with torch.no_grad():
        bnb_group, gindex, pindex = self._param_bnb_idx[id(param)]
        state = self._adamw8.state[param]
        if len(state) == 0:
            self._adamw8.init_state(bnb_group, param, gindex, pindex)
        self._adamw8.prefetch_state(param)
        self._adamw8.update_step(bnb_group, param, gindex, pindex)
        param.grad = None
```

Three bnb internal methods used:
- `init_state(group, p, gindex, pindex)` — lazy-initializes 8-bit optimizer state (quantization maps, absmax buffers) on first step
- `prefetch_state(p)` — ensures state is on the correct device (mirrors what bnb's `.step()` does before each update)
- `update_step(group, p, gindex, pindex)` — performs the actual Adam update for a single parameter (8-bit blockwise or 32-bit fallback depending on param size vs `min_8bit_size`)

### Hparam Sync Moved to `step()`

Instead of syncing inside each hook (N times), hparams are synced once:

```python
def step(self):
    if not self._adamw_hooked:
        self._sync_adamw8_hparams()
        self._adamw8.step()
    else:
        # Sync once so next backward's hooks use fresh values
        self._sync_adamw8_hparams()
    for group in self.param_groups:
        ...
```

The upfront sync in `install_post_accum_hooks()` ensures hooks have correct values for the first backward pass.

Files modified:
- [`nanochat/optim.py`](/home/gabrielmaxx/projects/gpt/nanochat/nanochat/optim.py): `MuonAdamW8bit.__init__()`, `install_post_accum_hooks()`, `step()`

## Why This Works (Complexity Analysis)

### Before

Per backward pass with N AdamW parameters:

| Operation | Per Hook | Total (N hooks) |
|-----------|----------|-----------------|
| `_sync_adamw8_hparams()` | O(G) where G = num groups | O(N * G) |
| `self._adamw8.step()` iteration | O(N) | O(N^2) |
| Actual bnb update work | O(1) | O(N) |
| **Total** | | **O(N^2 + N*G)** |

### After

| Operation | Per Hook | Total (N hooks) |
|-----------|----------|-----------------|
| Dict lookup `_param_bnb_idx[id(p)]` | O(1) | O(N) |
| `init_state` (first step only) | O(1) | O(N) once |
| `prefetch_state` + `update_step` | O(1) | O(N) |
| `_sync_adamw8_hparams()` (in step) | — | O(G) once |
| **Total** | | **O(N + G)** |

**Speedup: O(N^2) -> O(N)**

### Mathematical Equivalence

The bnb `update_step` performs exactly the same computation that `.step()` would perform for that parameter — same 8-bit blockwise quantization, same Adam math, same step counter increment. The only difference is that we call it directly instead of via the outer iteration loop. The result is bit-identical.

## bnb Internal API Reference

The bitsandbytes `AdamW8bit` class hierarchy:

```
torch.optim.Optimizer
  -> Optimizer8bit          # step(), prefetch_state()
    -> Optimizer2State      # init_state(), update_step()
      -> AdamW8bit          # configuration only
```

`update_step` dispatches to either:
- `F.optimizer_update_8bit_blockwise()` for params >= `min_8bit_size` (default 4096)
- `F.optimizer_update_32bit()` for smaller params (full precision fallback)

Both are CUDA kernel calls — the per-parameter invocation pattern has no additional kernel launch overhead compared to `.step()`, since `.step()` launches the same kernels one-by-one in its own loop.
