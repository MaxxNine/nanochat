#!/usr/bin/env python3
"""
Structured sparsity MLP benchmark.

Compares:
1) Dense MLP baseline
2) Structured neuron pruning (physically reduced hidden dimension)
3) 2:4 weight masking on dense kernels (theoretical sparse FLOP reduction; runtime may not improve)
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


@dataclass
class BenchResult:
    method: str
    keep_frac: float
    hidden_kept: int
    val_rel_l2_error: float
    ms: float
    flops: float
    tflops: float
    speedup_vs_dense: float
    nnz_frac_w1: float
    nnz_frac_w2: float
    notes: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Structured sparsity MLP benchmark")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dmodel", type=int, default=1664)
    p.add_argument("--expansion", type=int, default=4)
    p.add_argument("--tokens-val", type=int, default=8192)
    p.add_argument("--tokens-bench", type=int, default=16384)
    p.add_argument("--keep-fracs", type=str, default="0.75,0.5,0.375,0.25")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--skip-2to4", action="store_true", help="Skip 2:4 masked runs")
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


def parse_keep_fracs(keep_fracs: str) -> list[float]:
    vals: list[float] = []
    for raw in keep_fracs.split(","):
        raw = raw.strip()
        if not raw:
            continue
        v = float(raw)
        if v <= 0.0 or v >= 1.0:
            continue
        vals.append(v)
    vals = sorted(set(vals), reverse=True)
    if not vals:
        raise ValueError("keep-fracs must include at least one value in (0, 1).")
    return vals


def mlp_dense_flops(n_tokens: int, dmodel: int, hidden: int) -> float:
    # Two matmuls: X@W1 and H@W2
    return float(2 * n_tokens * dmodel * hidden + 2 * n_tokens * hidden * dmodel)


def apply_2to4_mask_rows(weight: torch.Tensor) -> torch.Tensor:
    """
    Apply 2:4 sparsity across rows for each column.
    For every group of 4 rows and each column, keep top-2 by magnitude.
    """
    rows, cols = weight.shape
    if rows % 4 != 0:
        raise ValueError(f"2:4 mask requires row count multiple of 4, got rows={rows}")
    w4 = weight.view(rows // 4, 4, cols)
    top2 = w4.abs().topk(k=2, dim=1).indices
    mask = torch.zeros_like(w4, dtype=torch.bool)
    mask.scatter_(1, top2, True)
    out = torch.where(mask, w4, torch.zeros_like(w4))
    return out.view(rows, cols)


def neuron_scores(w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    # Importance heuristic for hidden neurons.
    # w1: [d, h], w2: [h, d]
    s1 = torch.linalg.vector_norm(w1, dim=0)  # per hidden column
    s2 = torch.linalg.vector_norm(w2, dim=1)  # per hidden row
    return s1 * s2


def json_float(x: float) -> float | None:
    if not math.isfinite(x):
        return None
    return float(x)


def result_to_dict(r: BenchResult) -> dict:
    return {
        "method": r.method,
        "keep_frac": json_float(r.keep_frac),
        "hidden_kept": r.hidden_kept,
        "val_rel_l2_error": json_float(r.val_rel_l2_error),
        "ms": json_float(r.ms),
        "flops": json_float(r.flops),
        "tflops": json_float(r.tflops),
        "speedup_vs_dense": json_float(r.speedup_vs_dense),
        "nnz_frac_w1": json_float(r.nnz_frac_w1),
        "nnz_frac_w2": json_float(r.nnz_frac_w2),
        "notes": r.notes,
    }


def write_json(path: str, payload: dict) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_table(rows: list[BenchResult]) -> None:
    print("\n=== Structured Sparsity MLP Results ===")
    print(
        f"{'Method':28s} {'Keep':>6s} {'Hid':>6s} {'ValErr':>10s} {'ms':>8s} "
        f"{'TFLOP/s':>10s} {'Speedup':>9s} {'nnz(w1/w2)':>13s}"
    )
    for r in rows:
        nnz = f"{r.nnz_frac_w1:.2f}/{r.nnz_frac_w2:.2f}"
        print(
            f"{r.method:28s} {r.keep_frac:6.3f} {r.hidden_kept:6d} {r.val_rel_l2_error:10.4e} "
            f"{r.ms:8.3f} {r.tflops:10.3f} {r.speedup_vs_dense:9.3f} {nnz:>13s}"
        )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    bench_dtype = pick_dtype(args.dtype)
    keep_fracs = parse_keep_fracs(args.keep_fracs)

    print(f"Device: {device}")
    print(f"Benchmark dtype: {bench_dtype}")
    print(f"dmodel={args.dmodel}, expansion={args.expansion}")
    print(f"tokens(val/bench)=({args.tokens_val}, {args.tokens_bench})")
    print(f"keep_fracs={keep_fracs}")
    print(f"warmup/iters={args.warmup}/{args.iters}")

    # dtype/device sanity check
    try:
        _ = (
            torch.randn(128, 128, device=device, dtype=bench_dtype)
            @ torch.randn(128, 128, device=device, dtype=bench_dtype)
        )
        sync(device)
    except Exception as e:
        raise RuntimeError(f"Requested dtype/device combo failed ({bench_dtype} on {device}): {e}") from e

    d = args.dmodel
    h = args.expansion * d
    x_val = torch.randn(args.tokens_val, d, device=device, dtype=torch.float32)
    x_bench = torch.randn(args.tokens_bench, d, device=device, dtype=bench_dtype)
    w1 = torch.randn(d, h, device=device, dtype=torch.float32)
    w2 = torch.randn(h, d, device=device, dtype=torch.float32)

    def dense_mlp(x: torch.Tensor, w1_: torch.Tensor, w2_: torch.Tensor) -> torch.Tensor:
        z = x @ w1_
        z = F.relu(z).square()
        return z @ w2_

    with torch.no_grad():
        y_val_dense = dense_mlp(x_val, w1, w2)

    w1_b = w1.to(dtype=bench_dtype)
    w2_b = w2.to(dtype=bench_dtype)
    dense_fn = lambda: dense_mlp(x_bench, w1_b, w2_b)
    dense_ms, _ = benchmark_fn(dense_fn, args.warmup, args.iters, device)

    dense_flops = mlp_dense_flops(args.tokens_bench, d, h)
    results: list[BenchResult] = []

    # Method 1: structured neuron pruning (true reduced hidden size)
    scores = neuron_scores(w1, w2)
    for frac in keep_fracs:
        kept = max(1, int(round(h * frac)))
        idx = torch.topk(scores, kept, largest=True).indices
        idx_sorted = torch.sort(idx).values

        w1p = w1[:, idx_sorted]
        w2p = w2[idx_sorted, :]

        with torch.no_grad():
            y_val = dense_mlp(x_val, w1p, w2p)
            val_err = rel_l2_error(y_val, y_val_dense)

        w1p_b = w1p.to(dtype=bench_dtype)
        w2p_b = w2p.to(dtype=bench_dtype)
        approx_fn = lambda w1p_b=w1p_b, w2p_b=w2p_b: dense_mlp(x_bench, w1p_b, w2p_b)
        ms, _ = benchmark_fn(approx_fn, args.warmup, args.iters, device)
        flops = mlp_dense_flops(args.tokens_bench, d, kept)
        tflops = flops / (ms / 1000.0) / 1e12

        results.append(
            BenchResult(
                method="neuron_prune",
                keep_frac=frac,
                hidden_kept=kept,
                val_rel_l2_error=val_err,
                ms=ms,
                flops=flops,
                tflops=tflops,
                speedup_vs_dense=dense_ms / ms,
                nnz_frac_w1=float(kept) / float(h),
                nnz_frac_w2=float(kept) / float(h),
                notes="true structured reduction (reduced hidden dimension)",
            )
        )

    # Method 2: 2:4 masking (dense kernels)
    if not args.skip_2to4:
        try:
            w1_2to4 = apply_2to4_mask_rows(w1)
            w2_2to4 = apply_2to4_mask_rows(w2)

            with torch.no_grad():
                y_val = dense_mlp(x_val, w1_2to4, w2_2to4)
                val_err = rel_l2_error(y_val, y_val_dense)

            w1_2to4_b = w1_2to4.to(dtype=bench_dtype)
            w2_2to4_b = w2_2to4.to(dtype=bench_dtype)
            approx_fn = lambda: dense_mlp(x_bench, w1_2to4_b, w2_2to4_b)
            ms, _ = benchmark_fn(approx_fn, args.warmup, args.iters, device)

            nnz_frac_w1 = float((w1_2to4 != 0).float().mean().item())
            nnz_frac_w2 = float((w2_2to4 != 0).float().mean().item())
            effective_flops = dense_flops * 0.5 * (nnz_frac_w1 + nnz_frac_w2)
            tflops = effective_flops / (ms / 1000.0) / 1e12

            results.append(
                BenchResult(
                    method="mask_2to4_dense_kernel",
                    keep_frac=0.5,
                    hidden_kept=h,
                    val_rel_l2_error=val_err,
                    ms=ms,
                    flops=effective_flops,
                    tflops=tflops,
                    speedup_vs_dense=dense_ms / ms,
                    nnz_frac_w1=nnz_frac_w1,
                    nnz_frac_w2=nnz_frac_w2,
                    notes="runtime uses dense kernels; speedup may be limited",
                )
            )
        except Exception as e:
            print(f"Skipping 2:4 masked run: {type(e).__name__}: {e}")

    results.sort(key=lambda r: (r.method, -r.keep_frac))
    print_table(results)

    if args.json_out:
        device_name = None
        if device.type == "cuda":
            device_idx = device.index if device.index is not None else 0
            device_name = torch.cuda.get_device_name(device_idx)
        payload = {
            "script": "structured_sparsity_mlp_bench.py",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": {
                "device_arg": args.device,
                "resolved_device": str(device),
                "dtype": args.dtype,
                "seed": args.seed,
                "dmodel": args.dmodel,
                "expansion": args.expansion,
                "tokens_val": args.tokens_val,
                "tokens_bench": args.tokens_bench,
                "keep_fracs": keep_fracs,
                "warmup": args.warmup,
                "iters": args.iters,
                "skip_2to4": args.skip_2to4,
            },
            "system": {
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_name": device_name,
                "torch_version": torch.__version__,
            },
            "baseline": {
                "dense_ms": dense_ms,
                "dense_flops": dense_flops,
            },
            "results": [result_to_dict(r) for r in results],
        }
        write_json(args.json_out, payload)
        print(f"\nJSON report saved to: {args.json_out}")

    print("\nDone. Prefer variants with clear speedup and acceptable error.")


if __name__ == "__main__":
    main()
