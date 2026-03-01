# Optimizer Phase (`opt`)

This phase has two layers: the training loop callsite and the optimizer internals.

## 1) Training loop callsite

Source: `scripts/base_train.py` (around lines 891-905)

```python
# step the optimizer
lrm = get_lr_multiplier(step)
muon_momentum = get_muon_momentum(step)
muon_weight_decay = get_weight_decay(step)
for group in optimizer.param_groups:
    group["lr"] = group["initial_lr"] * lrm
    if group['kind'] == 'muon':
        group["momentum"] = muon_momentum
        group["weight_decay"] = muon_weight_decay
if mem_debug_step:
    torch.cuda.reset_peak_memory_stats()
optimizer.step()
if mem_debug_step:
    phase_opt_peak_bytes = float(get_max_memory())
mem_after_step_alloc = get_cur_memory()
mem_after_step_reserved = get_reserved_memory()
```

## 2) Optimizer implementation used on 1x4090

Source: `nanochat/gpt.py` (around lines 528-575)

```python
def setup_optimizer(
    self,
    unembedding_lr=0.004,
    embedding_lr=0.2,
    matrix_lr=0.02,
    weight_decay=0.0,
    adam_betas=(0.8, 0.95),
    scalar_lr=0.5,
    muon_active_only_stack=False,
    muon_stack_chunk_size=0,
):
    model_dim = self.config.n_embd
    ddp, rank, local_rank, world_size = get_dist_info()
    matrix_params = list(self.transformer.h.parameters())
    value_embeds_params = list(self.value_embeds.parameters())
    embedding_params = list(self.transformer.wte.parameters())
    lm_head_params = list(self.lm_head.parameters())
    resid_params = [self.resid_lambdas]
    x0_params = [self.x0_lambdas]
    assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params)

    dmodel_lr_scale = (model_dim / 768) ** -0.5
    print0(f"Scaling the LR for the AdamW parameters 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")

    param_groups = [
        dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
    ]
    for shape in sorted({p.shape for p in matrix_params}):
        group_params = [p for p in matrix_params if p.shape == shape]
        param_groups.append(dict(
            kind='muon', params=group_params, lr=matrix_lr,
            momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            active_only_stack=bool(muon_active_only_stack),
            stack_chunk_size=int(muon_stack_chunk_size),
        ))
    Factory = DistMuonAdamW if ddp else MuonAdamW
    optimizer = Factory(param_groups)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    return optimizer
```

On single-GPU (`ddp=False`) this uses `MuonAdamW`.

Source: `nanochat/optim.py` (around lines 236-445)

```python
def _step_adamw(self, group: dict) -> None:
    for p in group['params']:
        if p.grad is None:
            continue
        grad = p.grad
        state = self.state[p]

        if not state:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(p)
            state['exp_avg_sq'] = torch.zeros_like(p)
        exp_avg = state['exp_avg']
        exp_avg_sq = state['exp_avg_sq']
        state['step'] += 1

        self._adamw_step_t.fill_(state['step'])
        self._adamw_lr_t.fill_(group['lr'])
        self._adamw_beta1_t.fill_(group['betas'][0])
        self._adamw_beta2_t.fill_(group['betas'][1])
        self._adamw_eps_t.fill_(group['eps'])
        self._adamw_wd_t.fill_(group['weight_decay'])

        adamw_step_fused(
            p, grad, exp_avg, exp_avg_sq,
            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
        )

def _step_muon(self, group: dict) -> None:
    params: list[Tensor] = group['params']
    if not params:
        return
    active_only_stack = bool(group.get("active_only_stack", False))
    stack_chunk_size = int(group.get("stack_chunk_size", 0) or 0)

    p = params[0]
    state = self.state[p]
    num_params = len(params)
    shape, device, dtype = p.shape, p.device, p.dtype

    if "momentum_buffer" not in state:
        state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
    momentum_buffer = state["momentum_buffer"]

    if "second_momentum_buffer" not in state:
        state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
        state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
    second_momentum_buffer = state["second_momentum_buffer"]
    red_dim = -1 if shape[-2] >= shape[-1] else -2

    self._muon_momentum_t.fill_(group["momentum"])
    self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
    self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
    self._muon_wd_t.fill_(group["weight_decay"])
    active_indices = [i for i, pp in enumerate(params) if pp.grad is not None]
    if not active_indices:
        return
    all_active = len(active_indices) == num_params

    if active_only_stack and all_active:
        stacked_grads = torch.stack([pp.grad for pp in params])  # type: ignore[arg-type]
        stacked_params = torch.stack(params)
        muon_step_fused(
            stacked_grads,
            stacked_params,
            momentum_buffer,
            second_momentum_buffer,
            self._muon_momentum_t,
            self._muon_lr_t,
            self._muon_wd_t,
            self._muon_beta2_t,
            group["ns_steps"],
            red_dim,
        )
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))
        return

    if not active_only_stack and stack_chunk_size <= 0:
        stacked_grads = torch.stack([pp.grad if pp.grad is not None else torch.zeros_like(pp) for pp in params])
        stacked_params = torch.stack(params)
        muon_step_fused(
            stacked_grads,
            stacked_params,
            momentum_buffer,
            second_momentum_buffer,
            self._muon_momentum_t,
            self._muon_lr_t,
            self._muon_wd_t,
            self._muon_beta2_t,
            group["ns_steps"],
            red_dim,
        )
        active_params = [params[i] for i in active_indices]
        active_values = [stacked_params[i] for i in active_indices]
        torch._foreach_copy_(active_params, active_values)
        return

    if active_only_stack and stack_chunk_size <= 0:
        stack_chunk_size = num_params
    ranges = _iter_muon_ranges(params, active_only_stack=active_only_stack, stack_chunk_size=stack_chunk_size)
    if not ranges:
        return
    for s, e in ranges:
        chunk_params = params[s:e]
        chunk_len = e - s
        stacked_params = torch.stack(chunk_params)
        if active_only_stack:
            stacked_grads = torch.stack([pp.grad for pp in chunk_params])  # type: ignore[arg-type]
            active_chunk = None
        else:
            stacked_grads = torch.stack([pp.grad if pp.grad is not None else torch.zeros_like(pp) for pp in chunk_params])
            active_chunk = torch.tensor([pp.grad is not None for pp in chunk_params], dtype=torch.bool, device=device)

        use_padded = stack_chunk_size > 0 and chunk_len < stack_chunk_size
        if use_padded:
            pad_key = f"_muon_pad_buffers_{stack_chunk_size}"
            pad = state.get(pad_key)
            if pad is None:
                second_shape = (stack_chunk_size, shape[-2], 1) if shape[-2] >= shape[-1] else (stack_chunk_size, 1, shape[-1])
                pad = dict(
                    params=torch.empty(stack_chunk_size, *shape, dtype=dtype, device=device),
                    grads=torch.zeros(stack_chunk_size, *shape, dtype=dtype, device=device),
                    mom=torch.zeros(stack_chunk_size, *shape, dtype=dtype, device=device),
                    second=torch.zeros(second_shape, dtype=dtype, device=device),
                )
                state[pad_key] = pad
            pad_params = pad["params"]
            pad_grads = pad["grads"]
            pad_mom = pad["mom"]
            pad_second = pad["second"]
            pad_params.zero_()
            pad_grads.zero_()
            pad_mom.zero_()
            pad_second.zero_()
            pad_params[:chunk_len].copy_(stacked_params)
            pad_grads[:chunk_len].copy_(stacked_grads)
            pad_mom[:chunk_len].copy_(momentum_buffer[s:e])
            pad_second[:chunk_len].copy_(second_momentum_buffer[s:e])
            muon_step_fused(
                pad_grads,
                pad_params,
                pad_mom,
                pad_second,
                self._muon_momentum_t,
                self._muon_lr_t,
                self._muon_wd_t,
                self._muon_beta2_t,
                group["ns_steps"],
                red_dim,
            )
            momentum_buffer[s:e].copy_(pad_mom[:chunk_len])
            second_momentum_buffer[s:e].copy_(pad_second[:chunk_len])
            updated_params = pad_params[:chunk_len]
        else:
            muon_step_fused(
                stacked_grads,
                stacked_params,
                momentum_buffer[s:e],
                second_momentum_buffer[s:e],
                self._muon_momentum_t,
                self._muon_lr_t,
                self._muon_wd_t,
                self._muon_beta2_t,
                group["ns_steps"],
                red_dim,
            )
            updated_params = stacked_params

        if active_chunk is None or bool(active_chunk.all()):
            torch._foreach_copy_(chunk_params, list(updated_params.unbind(0)))
        else:
            idx_local = torch.nonzero(active_chunk, as_tuple=False).squeeze(1).tolist()
            if idx_local:
                active_params = [chunk_params[i] for i in idx_local]
                active_values = [updated_params[i] for i in idx_local]
                torch._foreach_copy_(active_params, active_values)

@torch.no_grad()
def step(self):
    for group in self.param_groups:
        if group['kind'] == 'adamw':
            self._step_adamw(group)
        elif group['kind'] == 'muon':
            self._step_muon(group)
        else:
            raise ValueError(f"Unknown optimizer kind: {group['kind']}")
```
