# Backward Phase (`bwd`)

Source: `scripts/base_train.py` (training loop, around lines 849-863)

```python
train_loss = loss.detach()  # for logging
loss = loss / grad_accum_steps  # each .backward() is a grad sum => normalize loss here
if mem_debug_step:
    torch.cuda.reset_peak_memory_stats()
loss.backward()
if mem_debug_step:
    phase_bwd_peak_bytes = max(phase_bwd_peak_bytes, float(get_max_memory()))
if args.debug_nan:
    bad_grads = debug_list_nonfinite_grads(orig_model, max_items=args.debug_max_nonfinite)
    if bad_grads:
        print0(f"DEBUG_NAN: non-finite gradients detected at step={step} micro_step={micro_step}")
        for name, dtype, shape, num_bad in bad_grads:
            print0(f"  grad {name}: dtype={dtype} shape={shape} bad={num_bad}")
        raise RuntimeError("Non-finite gradients detected")
x, y, dataloader_state_dict = next(train_loader)  # prefetch next batch while GPU is busy
```
