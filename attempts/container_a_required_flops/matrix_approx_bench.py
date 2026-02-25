#!/usr/bin/env python3
"""
Benchmark dense vs low-rank approximations for matrix-heavy paths.

This is an approximation experiment harness:
- We report speed/FLOP effects
- We also report approximation error vs dense baseline
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


@dataclass
class BenchResult:
    name: str
    rank: int
    ms: float
    flops: float
    tflops: float
    speedup_vs_dense: float
    rel_l2_error: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dense vs low-rank matrix approximation microbench")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tokens", type=int, default=16384, help="Batch*Seq flattened token count (N)")
    p.add_argument("--dmodel", type=int, default=1664, help="Model width (e.g. d26 nanochat width)")
    p.add_argument("--expansion", type=int, default=4, help="MLP expansion factor")
    p.add_argument("--rank-fracs", type=str, default="1.0,0.75,0.5,0.375,0.25")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--benchmark", type=str, default="both", choices=["linear", "mlp", "both"])
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


def dense_linear_flops(n_tokens: int, in_dim: int, out_dim: int) -> float:
    return float(2 * n_tokens * in_dim * out_dim)


def low_rank_linear_flops(n_tokens: int, in_dim: int, out_dim: int, rank: int) -> float:
    # (X @ A) @ B where A:[in,rank], B:[rank,out]
    return float(2 * n_tokens * in_dim * rank + 2 * n_tokens * rank * out_dim)


def build_svd_factors(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Compute in float32 for numerical stability.
    u, s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
    return u, s, vh


def materialize_rank_factors(
    u: torch.Tensor,
    s: torch.Tensor,
    vh: torch.Tensor,
    rank: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    a = (u[:, :rank] * s[:rank]).to(device=device, dtype=dtype)  # [in, rank]
    b = vh[:rank, :].to(device=device, dtype=dtype)               # [rank, out]
    return a, b


def parse_rank_fracs(rank_fracs: str) -> list[float]:
    vals = []
    for raw in rank_fracs.split(","):
        raw = raw.strip()
        if not raw:
            continue
        v = float(raw)
        if v <= 0.0:
            continue
        vals.append(v)
    vals = sorted(set(vals), reverse=True)
    if 1.0 not in vals:
        vals.insert(0, 1.0)
    return vals


def json_float(x: float) -> float | None:
    if not math.isfinite(x):
        return None
    return float(x)


def result_to_dict(r: BenchResult) -> dict:
    return {
        "name": r.name,
        "rank": r.rank,
        "ms": json_float(r.ms),
        "flops": json_float(r.flops),
        "tflops": json_float(r.tflops),
        "speedup_vs_dense": json_float(r.speedup_vs_dense),
        "rel_l2_error": json_float(r.rel_l2_error),
    }


def write_json(path: str, payload: dict) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_table(title: str, rows: list[BenchResult]) -> None:
    print(f"\n=== {title} ===")
    print(f"{'Variant':34s} {'Rank':>6s} {'ms/iter':>10s} {'Est TFLOP/s':>12s} {'Speedup':>10s} {'RelL2Err':>10s}")
    for r in rows:
        rank_str = "-" if r.rank < 0 else str(r.rank)
        err = "baseline" if math.isnan(r.rel_l2_error) else f"{r.rel_l2_error:.4e}"
        print(
            f"{r.name:34s} {rank_str:>6s} {r.ms:10.3f} {r.tflops:12.3f} "
            f"{r.speedup_vs_dense:10.3f} {err:>10s}"
        )


def run_linear_bench(
    device: torch.device,
    dtype: torch.dtype,
    n_tokens: int,
    dmodel: int,
    rank_fracs: list[float],
    warmup: int,
    iters: int,
) -> list[BenchResult]:
    x = torch.randn(n_tokens, dmodel, device=device, dtype=dtype)
    w = torch.randn(dmodel, dmodel, device=device, dtype=dtype)

    dense_fn = lambda: x @ w
    dense_ms, dense_out = benchmark_fn(dense_fn, warmup, iters, device)
    dense_flops = dense_linear_flops(n_tokens, dmodel, dmodel)
    dense_tflops = dense_flops / (dense_ms / 1000.0) / 1e12

    results = [
        BenchResult(
            name="dense_linear",
            rank=-1,
            ms=dense_ms,
            flops=dense_flops,
            tflops=dense_tflops,
            speedup_vs_dense=1.0,
            rel_l2_error=float("nan"),
        )
    ]

    print("Computing SVD for linear weight...")
    u, s, vh = build_svd_factors(w)
    max_rank = min(w.shape)

    for frac in rank_fracs:
        if frac >= 1.0:
            continue
        rank = max(1, min(max_rank, int(round(max_rank * frac))))
        a, b = materialize_rank_factors(u, s, vh, rank, dtype, device)
        approx_fn = lambda a=a, b=b: (x @ a) @ b
        ms, out = benchmark_fn(approx_fn, warmup, iters, device)
        flops = low_rank_linear_flops(n_tokens, dmodel, dmodel, rank)
        tflops = flops / (ms / 1000.0) / 1e12
        results.append(
            BenchResult(
                name=f"low_rank_linear(frac={frac:.3f})",
                rank=rank,
                ms=ms,
                flops=flops,
                tflops=tflops,
                speedup_vs_dense=dense_ms / ms,
                rel_l2_error=rel_l2_error(out, dense_out),
            )
        )
    return results


def run_mlp_bench(
    device: torch.device,
    dtype: torch.dtype,
    n_tokens: int,
    dmodel: int,
    expansion: int,
    rank_fracs: list[float],
    warmup: int,
    iters: int,
) -> list[BenchResult]:
    hidden = expansion * dmodel
    x = torch.randn(n_tokens, dmodel, device=device, dtype=dtype)
    w1 = torch.randn(dmodel, hidden, device=device, dtype=dtype)
    w2 = torch.randn(hidden, dmodel, device=device, dtype=dtype)

    def dense_mlp() -> torch.Tensor:
        h = x @ w1
        h = torch.relu(h).square()
        return h @ w2

    dense_ms, dense_out = benchmark_fn(dense_mlp, warmup, iters, device)
    dense_flops = dense_linear_flops(n_tokens, dmodel, hidden) + dense_linear_flops(n_tokens, hidden, dmodel)
    dense_tflops = dense_flops / (dense_ms / 1000.0) / 1e12

    results = [
        BenchResult(
            name=f"dense_mlp(exp={expansion})",
            rank=-1,
            ms=dense_ms,
            flops=dense_flops,
            tflops=dense_tflops,
            speedup_vs_dense=1.0,
            rel_l2_error=float("nan"),
        )
    ]

    print("Computing SVD for MLP weights...")
    u1, s1, vh1 = build_svd_factors(w1)
    u2, s2, vh2 = build_svd_factors(w2)
    max_rank = dmodel  # min(dmodel, expansion*dmodel)

    for frac in rank_fracs:
        if frac >= 1.0:
            continue
        rank = max(1, min(max_rank, int(round(max_rank * frac))))
        a1, b1 = materialize_rank_factors(u1, s1, vh1, rank, dtype, device)  # d -> h
        a2, b2 = materialize_rank_factors(u2, s2, vh2, rank, dtype, device)  # h -> d

        def approx_mlp(a1=a1, b1=b1, a2=a2, b2=b2) -> torch.Tensor:
            h = (x @ a1) @ b1
            h = torch.relu(h).square()
            return (h @ a2) @ b2

        ms, out = benchmark_fn(approx_mlp, warmup, iters, device)
        flops = (
            low_rank_linear_flops(n_tokens, dmodel, hidden, rank)
            + low_rank_linear_flops(n_tokens, hidden, dmodel, rank)
        )
        tflops = flops / (ms / 1000.0) / 1e12
        results.append(
            BenchResult(
                name=f"low_rank_mlp(frac={frac:.3f})",
                rank=rank,
                ms=ms,
                flops=flops,
                tflops=tflops,
                speedup_vs_dense=dense_ms / ms,
                rel_l2_error=rel_l2_error(out, dense_out),
            )
        )
    return results


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype)
    rank_fracs = parse_rank_fracs(args.rank_fracs)

    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Tokens: {args.tokens:,}, dmodel: {args.dmodel}, expansion: {args.expansion}")
    print(f"Rank fractions: {rank_fracs}")
    print(f"Warmup/Iters: {args.warmup}/{args.iters}")

    # Quick capability check for dtype/device combo.
    try:
        _ = (torch.randn(128, 128, device=device, dtype=dtype) @ torch.randn(128, 128, device=device, dtype=dtype))
        sync(device)
    except Exception as e:
        raise RuntimeError(f"Requested dtype/device combo failed ({dtype} on {device}): {e}") from e

    linear_rows: list[BenchResult] = []
    mlp_rows: list[BenchResult] = []

    if args.benchmark in ("linear", "both"):
        linear_rows = run_linear_bench(
            device=device,
            dtype=dtype,
            n_tokens=args.tokens,
            dmodel=args.dmodel,
            rank_fracs=rank_fracs,
            warmup=args.warmup,
            iters=args.iters,
        )
        print_table("Linear Projection", linear_rows)

    if args.benchmark in ("mlp", "both"):
        mlp_rows = run_mlp_bench(
            device=device,
            dtype=dtype,
            n_tokens=args.tokens,
            dmodel=args.dmodel,
            expansion=args.expansion,
            rank_fracs=rank_fracs,
            warmup=args.warmup,
            iters=args.iters,
        )
        print_table("MLP Block", mlp_rows)

    if args.json_out:
        device_name = None
        if device.type == "cuda":
            device_idx = device.index if device.index is not None else 0
            device_name = torch.cuda.get_device_name(device_idx)

        payload = {
            "script": "matrix_approx_bench.py",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": {
                "device_arg": args.device,
                "resolved_device": str(device),
                "dtype": args.dtype,
                "seed": args.seed,
                "tokens": args.tokens,
                "dmodel": args.dmodel,
                "expansion": args.expansion,
                "rank_fracs": rank_fracs,
                "warmup": args.warmup,
                "iters": args.iters,
                "benchmark": args.benchmark,
            },
            "system": {
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_name": device_name,
                "torch_version": torch.__version__,
            },
            "results": {
                "linear": [result_to_dict(r) for r in linear_rows],
                "mlp": [result_to_dict(r) for r in mlp_rows],
            },
        }
        write_json(args.json_out, payload)
        print(f"\nJSON report saved to: {args.json_out}")

    print("\nDone. Keep candidates only if speedup is meaningful AND approximation error is acceptable.")


if __name__ == "__main__":
    main()
