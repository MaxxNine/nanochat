#!/usr/bin/env python3
"""
Data-aware low-rank + diagonal residual benchmark.

For square projection matrices W (d x d), compare:
1) Truncated SVD low-rank (rank-k)
2) Truncated SVD + diagonal residual
3) Data-aware low-rank under input covariance metric
4) Data-aware low-rank + diagonal residual

Data-aware objective approximates minimizing:
|| X(W - W_hat) ||_F^2
instead of ||W - W_hat||_F^2.
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
    method: str
    rank: int
    val_rel_l2_error: float
    ms: float
    flops: float
    tflops: float
    speedup_vs_dense: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Data-aware low-rank + diagonal benchmark")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dmodel", type=int, default=1664)
    p.add_argument("--tokens-train", type=int, default=32768, help="For input covariance in data-aware method")
    p.add_argument("--tokens-val", type=int, default=8192, help="For approximation error measurement")
    p.add_argument("--tokens-bench", type=int, default=16384, help="For latency benchmarking")
    p.add_argument("--rank-fracs", type=str, default="0.75,0.5,0.375,0.25")
    p.add_argument("--cov-eps", type=float, default=1e-4, help="Epsilon for covariance stabilization")
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


def parse_rank_fracs(rank_fracs: str) -> list[float]:
    vals: list[float] = []
    for raw in rank_fracs.split(","):
        raw = raw.strip()
        if not raw:
            continue
        v = float(raw)
        if v <= 0.0 or v >= 1.0:
            continue
        vals.append(v)
    vals = sorted(set(vals), reverse=True)
    if not vals:
        raise ValueError("rank-fracs must include at least one value in (0, 1).")
    return vals


def dense_linear_flops(n_tokens: int, in_dim: int, out_dim: int) -> float:
    return float(2 * n_tokens * in_dim * out_dim)


def low_rank_plus_diag_flops(n_tokens: int, dmodel: int, rank: int, with_diag: bool) -> float:
    # (X@A)@B plus optional X * dvec and add.
    flops = float(2 * n_tokens * dmodel * rank + 2 * n_tokens * rank * dmodel)
    if with_diag:
        flops += float(2 * n_tokens * dmodel)
    return flops


def svd_rank_factors(w: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    u, s, vh = torch.linalg.svd(w.float(), full_matrices=False)
    a = u[:, :rank] * s[:rank]
    b = vh[:rank, :]
    return a, b


def data_aware_rank_factors(
    w: torch.Tensor,
    x_train: torch.Tensor,
    rank: int,
    cov_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build rank-k factors minimizing weighted Frobenius:
      min_rank<=k || C^{1/2}(W - W_hat) ||_F
    where C = (X^T X)/N + eps*I.
    """
    dmodel = w.shape[0]
    n = x_train.shape[0]
    c = (x_train.T @ x_train) / float(n)
    c = c + cov_eps * torch.eye(dmodel, device=w.device, dtype=w.dtype)

    evals, evecs = torch.linalg.eigh(c.float())
    evals = torch.clamp(evals, min=cov_eps)
    sqrt_e = torch.sqrt(evals)
    invsqrt_e = torch.rsqrt(evals)

    # M = C^{1/2} W
    ut_w = evecs.T @ w.float()
    m = evecs @ (sqrt_e[:, None] * ut_w)

    um, sm, vhm = torch.linalg.svd(m, full_matrices=False)
    uk = um[:, :rank]
    sk = sm[:rank]
    vk = vhm[:rank, :]

    # A = C^{-1/2} (Uk * Sk), B = Vk
    tmp = uk * sk
    ut_tmp = evecs.T @ tmp
    a = evecs @ (invsqrt_e[:, None] * ut_tmp)
    b = vk
    return a, b


def json_float(x: float) -> float | None:
    if not math.isfinite(x):
        return None
    return float(x)


def result_to_dict(r: BenchResult) -> dict:
    return {
        "method": r.method,
        "rank": r.rank,
        "val_rel_l2_error": json_float(r.val_rel_l2_error),
        "ms": json_float(r.ms),
        "flops": json_float(r.flops),
        "tflops": json_float(r.tflops),
        "speedup_vs_dense": json_float(r.speedup_vs_dense),
    }


def write_json(path: str, payload: dict) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_table(rows: list[BenchResult]) -> None:
    print("\n=== Data-aware Low-rank + Diag Results ===")
    print(f"{'Method':30s} {'Rank':>6s} {'ValRelL2':>10s} {'ms/iter':>10s} {'TFLOP/s':>10s} {'Speedup':>10s}")
    for r in rows:
        print(
            f"{r.method:30s} {r.rank:6d} {r.val_rel_l2_error:10.4e} {r.ms:10.3f} "
            f"{r.tflops:10.3f} {r.speedup_vs_dense:10.3f}"
        )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    bench_dtype = pick_dtype(args.dtype)
    rank_fracs = parse_rank_fracs(args.rank_fracs)

    print(f"Device: {device}")
    print(f"Dtype (benchmark): {bench_dtype}")
    print(f"dmodel: {args.dmodel}")
    print(f"tokens(train/val/bench)=({args.tokens_train}, {args.tokens_val}, {args.tokens_bench})")
    print(f"rank_fracs={rank_fracs}, cov_eps={args.cov_eps}")
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
    x_train = torch.randn(args.tokens_train, d, device=device, dtype=torch.float32)
    x_val = torch.randn(args.tokens_val, d, device=device, dtype=torch.float32)
    x_bench = torch.randn(args.tokens_bench, d, device=device, dtype=bench_dtype)
    w = torch.randn(d, d, device=device, dtype=torch.float32)

    with torch.no_grad():
        y_val_dense = x_val @ w

    dense_w_bench = w.to(dtype=bench_dtype)
    dense_fn = lambda: x_bench @ dense_w_bench
    dense_ms, _ = benchmark_fn(dense_fn, args.warmup, args.iters, device)

    results: list[BenchResult] = []
    max_rank = d
    for frac in rank_fracs:
        rank = max(1, min(max_rank, int(round(max_rank * frac))))
        # Build factor sets in float32
        a_svd, b_svd = svd_rank_factors(w, rank)
        a_da, b_da = data_aware_rank_factors(w, x_train, rank, cov_eps=args.cov_eps)

        methods: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor | None]] = []
        methods.append(("svd_low_rank", a_svd, b_svd, None))
        methods.append(("data_aware_low_rank", a_da, b_da, None))

        # +diag residual initializations
        with torch.no_grad():
            r_svd = w - (a_svd @ b_svd)
            r_da = w - (a_da @ b_da)
            d_svd = torch.diagonal(r_svd).clone()
            d_da = torch.diagonal(r_da).clone()
        methods.append(("svd_low_rank_plus_diag", a_svd, b_svd, d_svd))
        methods.append(("data_aware_low_rank_plus_diag", a_da, b_da, d_da))

        for method_name, a32, b32, d32 in methods:
            with torch.no_grad():
                y_val = (x_val @ a32) @ b32
                with_diag = d32 is not None
                if with_diag:
                    y_val = y_val + x_val * d32
                val_err = rel_l2_error(y_val, y_val_dense)

            a = a32.to(dtype=bench_dtype)
            b = b32.to(dtype=bench_dtype)
            dvec = d32.to(dtype=bench_dtype) if d32 is not None else None

            def approx_fn(a=a, b=b, dvec=dvec) -> torch.Tensor:
                y = (x_bench @ a) @ b
                if dvec is not None:
                    y = y + x_bench * dvec
                return y

            ms, _ = benchmark_fn(approx_fn, args.warmup, args.iters, device)
            flops = low_rank_plus_diag_flops(args.tokens_bench, d, rank, with_diag=(dvec is not None))
            tflops = flops / (ms / 1000.0) / 1e12
            results.append(
                BenchResult(
                    method=method_name,
                    rank=rank,
                    val_rel_l2_error=val_err,
                    ms=ms,
                    flops=flops,
                    tflops=tflops,
                    speedup_vs_dense=dense_ms / ms,
                )
            )

    # Sort by rank then method for readability.
    results.sort(key=lambda r: (r.rank, r.method))
    print_table(results)

    if args.json_out:
        device_name = None
        if device.type == "cuda":
            device_idx = device.index if device.index is not None else 0
            device_name = torch.cuda.get_device_name(device_idx)
        payload = {
            "script": "data_aware_low_rank_plus_diag.py",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": {
                "device_arg": args.device,
                "resolved_device": str(device),
                "dtype": args.dtype,
                "seed": args.seed,
                "dmodel": args.dmodel,
                "tokens_train": args.tokens_train,
                "tokens_val": args.tokens_val,
                "tokens_bench": args.tokens_bench,
                "rank_fracs": rank_fracs,
                "cov_eps": args.cov_eps,
                "warmup": args.warmup,
                "iters": args.iters,
            },
            "system": {
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_name": device_name,
                "torch_version": torch.__version__,
            },
            "baseline": {
                "dense_ms": dense_ms,
                "dense_flops": dense_linear_flops(args.tokens_bench, d, d),
            },
            "results": [result_to_dict(r) for r in results],
        }
        write_json(args.json_out, payload)
        print(f"\nJSON report saved to: {args.json_out}")

    print("\nDone. Look for Pareto winners: lower error with meaningful speedup.")


if __name__ == "__main__":
    main()
