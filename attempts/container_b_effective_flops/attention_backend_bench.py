#!/usr/bin/env python3
"""
Attention backend microbenchmark.

Compares:
- nanochat flash_attention wrapper
- raw scaled_dot_product_attention
- optional forced SDPA backends (if available in current PyTorch)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from nanochat.flash_attention import flash_attn


@dataclass
class BenchResult:
    name: str
    ms: float
    attn_tflops: float
    speedup_vs_base: float
    rel_l2_error: float
    status: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Attention backend throughput benchmark")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--heads", type=int, default=13)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--window", type=int, default=-1, help="-1 for full context, else left sliding window")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--json-out", type=str, default="", help="Optional path to write JSON report")
    return p.parse_args()


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def pick_dtype(dtype_arg: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_arg]


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_fn(fn: Callable[[], torch.Tensor], warmup: int, iters: int, device: torch.device) -> tuple[float, torch.Tensor]:
    with torch.no_grad():
        for _ in range(warmup):
            _ = fn()
        sync(device)
        t0 = time.perf_counter()
        out = None
        for _ in range(iters):
            out = fn()
        sync(device)
        t1 = time.perf_counter()
    assert out is not None
    return (t1 - t0) * 1000.0 / iters, out


def rel_l2_error(a: torch.Tensor, b: torch.Tensor) -> float:
    a32 = a.float()
    b32 = b.float()
    denom = b32.norm().item()
    if denom == 0.0:
        return float("nan")
    return (a32 - b32).norm().item() / denom


def attention_flops_est(batch: int, heads: int, seq_len: int, head_dim: int) -> float:
    # QK^T + AV only (common planning approximation), softmax/etc omitted.
    return float(4 * batch * heads * seq_len * seq_len * head_dim)


def build_sliding_mask(seq_len: int, left_window: int, device: torch.device) -> torch.Tensor:
    row = torch.arange(seq_len, device=device).unsqueeze(1)
    col = torch.arange(seq_len, device=device).unsqueeze(0)
    mask = col <= row
    if left_window >= 0:
        mask = mask & ((row - col) <= left_window)
    return mask


def json_float(x: float) -> float | None:
    if not math.isfinite(x):
        return None
    return float(x)


def result_to_dict(r: BenchResult) -> dict:
    return {
        "name": r.name,
        "ms": json_float(r.ms),
        "attn_tflops": json_float(r.attn_tflops),
        "speedup_vs_base": json_float(r.speedup_vs_base),
        "rel_l2_error": json_float(r.rel_l2_error),
        "status": r.status,
    }


def write_json(path: str, payload: dict) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def get_sdpa_backend_wrappers(
    qh: torch.Tensor,
    kh: torch.Tensor,
    vh: torch.Tensor,
    causal: bool,
    attn_mask: torch.Tensor | None,
) -> list[tuple[str, Callable[[], torch.Tensor]]]:
    out: list[tuple[str, Callable[[], torch.Tensor]]] = []
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
    except Exception:
        return out

    backend_names = [
        ("sdpa_forced_flash", "FLASH_ATTENTION"),
        ("sdpa_forced_efficient", "EFFICIENT_ATTENTION"),
        ("sdpa_forced_math", "MATH"),
        ("sdpa_forced_cudnn", "CUDNN_ATTENTION"),
    ]

    for label, attr in backend_names:
        if not hasattr(SDPBackend, attr):
            continue
        backend = getattr(SDPBackend, attr)

        def _fn(backend=backend):
            with sdpa_kernel(backends=[backend]):
                y = F.scaled_dot_product_attention(
                    qh,
                    kh,
                    vh,
                    attn_mask=attn_mask,
                    is_causal=causal and attn_mask is None,
                )
            return y.transpose(1, 2).contiguous()

        out.append((label, _fn))
    return out


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype)
    causal = True

    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Shape: B={args.batch}, T={args.seq_len}, H={args.heads}, D={args.head_dim}")
    print(f"Window: {args.window} (-1 means full context)")
    print(f"Warmup/Iters: {args.warmup}/{args.iters}")

    q = torch.randn(args.batch, args.seq_len, args.heads, args.head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # SDPA expects [B, H, T, D]
    qh = q.transpose(1, 2).contiguous()
    kh = k.transpose(1, 2).contiguous()
    vh = v.transpose(1, 2).contiguous()

    attn_mask = None
    if 0 <= args.window < args.seq_len:
        attn_mask = build_sliding_mask(args.seq_len, args.window, device)

    candidates: list[tuple[str, Callable[[], torch.Tensor]]] = []

    # nanochat wrapper path
    window_size = (-1, 0) if args.window < 0 else (args.window, 0)
    candidates.append(
        (
            "nanochat_flash_wrapper",
            lambda: flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size),
        )
    )

    # Raw SDPA auto backend
    candidates.append(
        (
            "sdpa_auto",
            lambda: F.scaled_dot_product_attention(
                qh,
                kh,
                vh,
                attn_mask=attn_mask,
                is_causal=causal and attn_mask is None,
            ).transpose(1, 2).contiguous(),
        )
    )

    # Optional forced SDPA backends
    candidates.extend(get_sdpa_backend_wrappers(qh, kh, vh, causal, attn_mask))

    results: list[BenchResult] = []
    base_ms = None
    base_out = None
    attn_flops = attention_flops_est(args.batch, args.heads, args.seq_len, args.head_dim)

    for idx, (name, fn) in enumerate(candidates):
        try:
            ms, out = benchmark_fn(fn, args.warmup, args.iters, device)
            tflops = attn_flops / (ms / 1000.0) / 1e12

            if idx == 0:
                base_ms = ms
                base_out = out

            if base_ms is None:
                speedup = 1.0
            else:
                speedup = base_ms / ms

            if base_out is None or idx == 0:
                err = float("nan")
            else:
                err = rel_l2_error(out, base_out)

            results.append(
                BenchResult(
                    name=name,
                    ms=ms,
                    attn_tflops=tflops,
                    speedup_vs_base=speedup,
                    rel_l2_error=err,
                    status="ok",
                )
            )
        except Exception as e:
            results.append(
                BenchResult(
                    name=name,
                    ms=float("nan"),
                    attn_tflops=float("nan"),
                    speedup_vs_base=float("nan"),
                    rel_l2_error=float("nan"),
                    status=f"unsupported: {type(e).__name__}",
                )
            )

    print("\n=== Attention Backend Benchmark ===")
    print(f"{'Backend':30s} {'ms/iter':>10s} {'Attn TFLOP/s':>13s} {'Speedup':>10s} {'RelL2Err':>10s} {'Status':>16s}")
    for r in results:
        ms = "nan" if math.isnan(r.ms) else f"{r.ms:.3f}"
        tflops = "nan" if math.isnan(r.attn_tflops) else f"{r.attn_tflops:.3f}"
        speedup = "nan" if math.isnan(r.speedup_vs_base) else f"{r.speedup_vs_base:.3f}"
        err = "baseline" if math.isnan(r.rel_l2_error) else f"{r.rel_l2_error:.3e}"
        print(f"{r.name:30s} {ms:>10s} {tflops:>13s} {speedup:>10s} {err:>10s} {r.status:>16s}")

    if args.json_out:
        device_name = None
        if device.type == "cuda":
            device_idx = device.index if device.index is not None else 0
            device_name = torch.cuda.get_device_name(device_idx)

        payload = {
            "script": "attention_backend_bench.py",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": {
                "device_arg": args.device,
                "resolved_device": str(device),
                "dtype": args.dtype,
                "seed": args.seed,
                "batch": args.batch,
                "seq_len": args.seq_len,
                "heads": args.heads,
                "head_dim": args.head_dim,
                "window": args.window,
                "warmup": args.warmup,
                "iters": args.iters,
            },
            "system": {
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_name": device_name,
                "torch_version": torch.__version__,
            },
            "results": [result_to_dict(r) for r in results],
        }
        write_json(args.json_out, payload)
        print(f"\nJSON report saved to: {args.json_out}")

    print("\nDone. Promote only candidates that improve latency materially and remain numerically stable.")


if __name__ == "__main__":
    main()
