# ACTION_PLAN: Liger-Kernel Fit for nanochat

Date: 2026-03-02  
Scope: `scripts/base_train.py`, `nanochat/gpt.py`, and related kernel subfiles

## 1) Current nanochat hotspots

- LM-head loss path:
  - `scripts/base_train.py` selects `--lm-ce-backend` with `baseline|chunked` (`scripts/base_train.py:68-69`).
  - `nanochat/gpt.py` runs either full logits CE or custom chunked CE (`nanochat/gpt.py:621-641`).
  - Custom chunked CE autograd is implemented in-file (`nanochat/gpt.py:151-310`).
- Attention path:
  - Uses FA3 when available, SDPA fallback otherwise (`nanochat/flash_attention.py:23-42`, `:99-169`).
  - RoPE is currently implemented manually (`nanochat/gpt.py:51-57`, `:91-94`).
- Norm/MLP architecture:
  - Norm is parameter-free functional RMSNorm (`nanochat/gpt.py:42-44`).
  - MLP is `relu(x)^2`, not SwiGLU/GEGLU (`nanochat/gpt.py:127-131`).

## 2) Liger compatibility matrix

| Liger item | Can use here? | Decision | Why |
|---|---|---|---|
| `LigerFusedLinearCrossEntropyLoss` | Yes, directly in `gpt.forward` | **Use** | Best overlap with existing bottleneck; replaces custom CE path while preserving softcap/ignore-index/reduction semantics. |
| `LigerCrossEntropyLoss` | Yes, but lower leverage | Optional fallback | Useful as backup if fused linear CE is unstable in our env. |
| `LigerRopeFunction` | Yes, with adapter code | Experimental | Can replace manual RoPE math, but likely lower impact than CE and needs shape/in-place validation under `torch.compile`. |
| `LigerRMSNorm` | Not a drop-in for us | **Do not use now** | Our model intentionally uses parameter-free functional RMSNorm; adopting `LigerRMSNorm` changes architecture/params. |
| `LigerSwiGLUMLP` / `LigerGEGLUMLP` | No (without arch change) | **Do not use now** | Our MLP is `relu^2`; switching would change training dynamics and invalidate current scaling/tuning assumptions. |
| Liger model monkey patch (`apply_liger_kernel_to_*`) | No (for this repo) | **Do not use** | Patch helpers target HF model classes, while nanochat uses a custom GPT module. |
| `LigerMultiTokenAttention` | Not needed currently | Defer | We already rely on FA3/SDPA attention path; overlap is limited and integration risk is higher. |

## 3) Critical gate before implementation

Liger docs currently show official tested matrix up to PyTorch `2.8` and Triton `3.4`, while this repo/environment is:
- `torch==2.9.1` (`pyproject.toml`)
- local runtime reports `torch 2.9.1+cu128`, `triton 3.5.1`

Action:
- Treat integration as **experimental** until Liger explicitly supports this stack, or run a temporary compatibility branch with supported Torch/Triton versions.

## 4) Recommended phased implementation

## Phase A: Integrate fused linear CE (highest ROI)

1. Extend CE backend options:
   - `scripts/base_train.py`: change `--lm-ce-backend` choices to include `liger_flce`.
   - `nanochat/gpt.py`: allow backend `"liger_flce"` in `configure_lm_ce`.
2. Add optional Liger import guard in `nanochat/gpt.py`:
   - Lazy-import `LigerFusedLinearCrossEntropyLoss`.
   - If unavailable/import fails, raise clear error or fallback to `"chunked"` with warning.
3. Add forward path in `GPT.forward`:
   - For training with targets and backend `liger_flce`, call Liger fused CE using:
     - `lin_weight = self.lm_head.weight[:self.config.vocab_size]`
     - `_input = x.view(-1, x.size(-1))`
     - `target = targets.view(-1)`
     - `ignore_index=-1`, `softcap=15`, and current `loss_reduction`.
4. Keep `baseline` and `chunked` backends intact as fallback paths.

## Phase B: Validation and parity tests

1. Add tests (new file, e.g. `tests/test_liger_ce.py`):
   - Loss parity vs baseline full-logits CE (`mean`, `sum`, `none`).
   - Gradient parity (hidden + lm_head weight) for random targets with ignored tokens (`-1`).
   - Edge case: all targets ignored (behavior should match current semantics).
2. Add compile smoke test:
   - `torch.compile` path with repeated steps (similar to `tests/test_chunked_ce.py`).
3. Run short training smoke:
   - `python -m scripts.base_train --depth=4 ... --lm-ce-backend=liger_flce --num-iterations=20`
   - Verify no NaN/Inf regressions.

## Phase C: Benchmark and decide default

1. Benchmark `baseline` vs `chunked` vs `liger_flce` on representative GPU(s):
   - tokens/sec
   - peak memory
   - validation bpb drift over same short run length
2. Promote default only if:
   - numerical behavior is stable,
   - memory and/or throughput improve materially,
   - compile stability is at least equal to current backends.

## Phase D: Optional RoPE experiment (lower ROI)

1. Add opt-in flag (e.g. `--rope-backend baseline|liger`).
2. Replace manual `apply_rotary_emb` with Liger call under flag only.
3. Validate tensor shape expectations, in-place behavior, and `torch.compile` stability.
4. Keep disabled by default unless benchmark shows consistent gain.

## 5) What should NOT be done now

- Do not replace FA3/SDPA attention path with Liger attention primitives in this pass.
- Do not switch MLP activation to SwiGLU/GEGLU for kernel convenience.
- Do not introduce learnable RMSNorm weights only to fit `LigerRMSNorm`.

## 6) Definition of done for this integration track

- `liger_flce` backend implemented and guarded by optional dependency import.
- Test suite covers parity + compile smoke.
- Short train run succeeds without numerical regressions.
- Benchmark report shows clear win or documents why integration should remain optional.

## References

- Liger README: https://github.com/linkedin/Liger-Kernel
- Liger getting started / compatibility: https://linkedin.github.io/Liger-Kernel/getting_started/
- Liger API index: https://linkedin.github.io/Liger-Kernel/api/transformers/
