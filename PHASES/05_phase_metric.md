# Phase Metric Formatter

Source: `scripts/base_train.py` (around lines 943-964)

```python
if mem_debug_step:
    print0(
        "MEM_DEBUG "
        f"step={step} "
        f"alloc_mib(before/bwd/step/zero)=("
        f"{mem_before_alloc/1024/1024:.2f}/"
        f"{mem_after_bwd_alloc/1024/1024:.2f}/"
        f"{mem_after_step_alloc/1024/1024:.2f}/"
        f"{mem_after_zero_alloc/1024/1024:.2f}) "
        f"reserved_mib(before/bwd/step/zero)=("
        f"{mem_before_reserved/1024/1024:.2f}/"
        f"{mem_after_bwd_reserved/1024/1024:.2f}/"
        f"{mem_after_step_reserved/1024/1024:.2f}/"
        f"{mem_after_zero_reserved/1024/1024:.2f}) "
        f"phase_peak_alloc_mib(fwd/bwd/opt/zero)=("
        f"{phase_fwd_peak_bytes/1024/1024:.2f}/"
        f"{phase_bwd_peak_bytes/1024/1024:.2f}/"
        f"{phase_opt_peak_bytes/1024/1024:.2f}/"
        f"{phase_zero_peak_bytes/1024/1024:.2f}) "
        f"step_peak_alloc_mib={step_peak_memory_bytes/1024/1024:.2f} "
        f"step_peak_reserved_mib={get_max_reserved_memory()/1024/1024:.2f}"
    )
```
