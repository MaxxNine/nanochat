# FLOP_DETAILMENT

## Goal
Create a repeatable way to compute:
- FLOPs per token
- FLOPs per training step
- Total FLOPs per run
- Cost share by component (to identify hotspots)

All formulas here are aligned with current nanochat code (`gpt.py` + `base_train.py`).

## Source Formula (current code)
`nanochat/gpt.py::estimate_flops()` uses:

```text
flops_per_token =
  6 * (total_params - wte - value_embeds - scalars)
  + sum_over_layers(12 * n_head * head_dim * effective_seq_len_layer)
```

With default config (`n_kv_head = n_head`):
- `head_dim = n_embd / n_head`
- `n_head * head_dim = n_embd`
- attention term becomes: `sum_over_layers(12 * n_embd * effective_seq_len_layer)`

`scripts/base_train.py` then uses:

```text
flops_per_step  = flops_per_token * total_batch_size_tokens
flops_total_run = flops_per_step * num_iterations
```

## Step-by-step FLOP Calculation (worked example: d26, seq=2048, pattern=SSSL)
Inputs:
- `depth=26`
- `aspect_ratio=64` -> `base_dim=1664`
- `head_dim=128` -> `n_embd=1664`, `n_head=13`
- `vocab_size=32768` (already padded)
- `num_ve_layers=ceil(26/2)=13`

### 1) Parameter groups
- `wte = vocab * n_embd = 32,768 * 1,664 = 54,525,952`
- `lm_head = 54,525,952`
- per-layer attention projection params (`Q,K,V,O`) = `4 * n_embd^2 = 11,075,584`
- per-layer MLP params = `8 * n_embd^2 = 22,151,168`
- per-layer block total = `33,226,752`
- transformer block total = `26 * 33,226,752 = 863,895,552`
- VE gates (tiny) = `13 * (32 * 13) = 5,408`
- `transformer_matrices = 863,900,960`
- `value_embeds = 13 * vocab * n_embd = 708,837,376`
- `scalars = 2 * depth = 52`
- `total_params = 1,681,790,292`

### 2) Matrix FLOPs per token
Matrix params counted for FLOPs:
- `matmul_params = total_params - wte - value_embeds - scalars`
- `matmul_params = transformer_matrices + lm_head = 918,426,912`

So:
- `matrix_flops_per_token = 6 * 918,426,912 = 5,510,561,472`

### 3) Attention score/value FLOPs per token
For `SSSL` tiled across 26 layers, final layer forced to `L`:
- 19 short layers (`1024`) + 7 long layers (`2048`)
- `sum_effective_seq = 19*1024 + 7*2048 = 33,792`

So:
- `attn_flops_per_token = 12 * 1,664 * 33,792 = 674,758,656`

### 4) Total FLOPs per token
- `flops_per_token = 5,510,561,472 + 674,758,656 = 6,185,320,128`

### 5) FLOPs per step (example batch)
If `total_batch_size = 1,048,576` tokens:
- `flops_per_step = 6,185,320,128 * 1,048,576 = 6,485,778,238,537,728`

## Hotspot Breakdown (same d26 example)
Per-token contribution:

| Component | FLOPs/token | Share |
|---|---:|---:|
| MLP matrices (`c_fc`, `c_proj`) | 3,455,582,208 | 55.87% |
| Attention projection matrices (`Q,K,V,O`) | 1,727,791,104 | 27.93% |
| `lm_head` matrix | 327,155,712 | 5.29% |
| Attention score/value term | 674,758,656 | 10.91% |
| VE gates | 32,448 | ~0.00% |

Takeaway:
- Main cost is matrix math (especially MLP), not VE gates/scalars.
- Sequence-length changes mostly hit the attention score/value term.
- Width/depth changes hit almost everything (highest leverage).

## Metrics To Track Every Run
From logs and config, compute:

1. `flops_per_token` (model estimate)
2. `tok_per_sec` (training log)
3. `effective_flops_per_sec = flops_per_token * tok_per_sec`
4. `target_tokens` (from `target_param_data_ratio * scaling_params`, or explicit)
5. `num_iterations`
6. `flops_total_run = flops_per_token * total_batch_size * num_iterations`
7. `time_estimate = flops_total_run / effective_flops_per_sec`

## Important Caveat
This FLOP model is intentionally simplified and does not fully include:
- softmax/normalization elementwise ops
- optimizer/kernel overhead
- memory bandwidth limits
- dataloader/I/O overhead

So this is a **planning and hotspot model**, not exact wall-clock truth.  
For 4090 specifically, SDPA kernel behavior can dominate runtime even when FLOP counts look lower.

## Reusable Template (fill per experiment)
```text
Run ID:
GPU:
depth:
n_embd:
n_head / n_kv_head:
seq_len:
window_pattern:
target_param_data_ratio:
total_batch_size:

scaling_params:
flops_per_token:
tok_per_sec (measured):
effective_flops_per_sec:
num_iterations:
flops_total_run:
total_training_time (measured):

Top hotspots (%):
1)
2)
3)
```
