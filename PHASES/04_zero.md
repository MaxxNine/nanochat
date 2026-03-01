# Zero Phase (`zero`)

Source: `scripts/base_train.py` (around lines 914-920)

```python
if mem_debug_step:
    torch.cuda.reset_peak_memory_stats()
model.zero_grad(set_to_none=True)
if mem_debug_step:
    phase_zero_peak_bytes = float(get_max_memory())
mem_after_zero_alloc = get_cur_memory()
mem_after_zero_reserved = get_reserved_memory()
```
