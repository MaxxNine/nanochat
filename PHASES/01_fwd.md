# Forward Phase (`fwd`)

Source: `scripts/base_train.py` (training loop, around lines 808-834)

```python
for micro_step in range(grad_accum_steps):
    if mem_debug_step:
        torch.cuda.reset_peak_memory_stats()
    with autocast_ctx:
        if dynamic_suffix_enabled:
            # Compile-aware dynamic suffix path:
            # - active_start=0 uses the base compiled model
            # - active_start>0 uses cached compiled forward specialized to that suffix
            # - eager is only used as fallback if compilation for that active_start fails
            compiled_fn = get_or_compile_suffix_fn(active_start_this_step)
            if compiled_fn is not None:
                try:
                    loss = compiled_fn(x, y)
                except Exception as e:
                    compiled_suffix_failed.add(int(active_start_this_step))
                    print0(
                        f"WARNING: dynamic suffix compiled forward failed at active_start={active_start_this_step}; "
                        f"falling back to eager for this active_start. error={type(e).__name__}: {e}"
                    )
                    loss = orig_model(x, y, active_start=active_start_this_step)
            else:
                loss = orig_model(x, y, active_start=active_start_this_step)
        else:
            # Preserve the original full-update call path.
            loss = model(x, y)
    if mem_debug_step:
        phase_fwd_peak_bytes = max(phase_fwd_peak_bytes, float(get_max_memory()))
```
