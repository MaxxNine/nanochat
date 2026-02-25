#!/usr/bin/env python3
"""
Loss-aware low-rank recovery benchmark.

Pipeline per rank:
1) Build dense "teacher" weights.
2) Initialize low-rank factors from SVD.
3) Measure initial approximation error.
4) Recover low-rank factors with short optimization on teacher outputs.
5) Measure final error + latency/FLOP tradeoff.
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
class RecoveryResult:
    name: str
    rank: int
    init_rel_l2_error: float
    final_rel_l2_error: float
    error_reduction_frac: float
    dense_ms: float
    approx_ms: float
    speedup_vs_dense: float
    dense_flops: float
    approx_flops: float
    approx_tflops: float
    train_loss_start: float
    train_loss_end: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Loss-aware low-rank recovery benchmark")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--benchmark", type=str, default="both", choices=["linear", "mlp", "both"])
    p.add_argument("--dmodel", type=int, default=1664)
    p.add_argument("--expansion", type=int, default=4)
    p.add_argument("--tokens-train", type=int, default=32768)
    p.add_argument("--tokens-val", type=int, default=8192)
    p.add_argument("--tokens-bench", type=int, default=16384)
    p.add_argument("--rank-fracs", type=str, default="0.75,0.5,0.375,0.25")
    p.add_argument("--steps", type=int, default=200, help="Recovery optimization steps")
    p.add_argument("--batch-size", type=int, default=2048, help="Recovery minibatch size")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
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


def build_svd_factors(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # SVD in float32 for stability.
    u, s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
    return u, s, vh


def materialize_rank_factors(
    u: torch.Tensor,
    s: torch.Tensor,
    vh: torch.Tensor,
    rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Keep training params in float32 for optimization quality.
    a = (u[:, :rank] * s[:rank]).to(device=device, dtype=torch.float32)
    b = vh[:rank, :].to(device=device, dtype=torch.float32)
    return a, b


def dense_linear_flops(n_tokens: int, in_dim: int, out_dim: int) -> float:
    return float(2 * n_tokens * in_dim * out_dim)


def low_rank_linear_flops(n_tokens: int, in_dim: int, out_dim: int, rank: int) -> float:
    return float(2 * n_tokens * in_dim * rank + 2 * n_tokens * rank * out_dim)


def json_float(x: float) -> float | None:
    if not math.isfinite(x):
        return None
    return float(x)


def result_to_dict(r: RecoveryResult) -> dict:
    return {
        "name": r.name,
        "rank": r.rank,
        "init_rel_l2_error": json_float(r.init_rel_l2_error),
        "final_rel_l2_error": json_float(r.final_rel_l2_error),
        "error_reduction_frac": json_float(r.error_reduction_frac),
        "dense_ms": json_float(r.dense_ms),
        "approx_ms": json_float(r.approx_ms),
        "speedup_vs_dense": json_float(r.speedup_vs_dense),
        "dense_flops": json_float(r.dense_flops),
        "approx_flops": json_float(r.approx_flops),
        "approx_tflops": json_float(r.approx_tflops),
        "train_loss_start": json_float(r.train_loss_start),
        "train_loss_end": json_float(r.train_loss_end),
    }


def write_json(path: str, payload: dict) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_table(title: str, rows: list[RecoveryResult]) -> None:
    print(f"\n=== {title} ===")
    print(
        f"{'Variant':30s} {'Rank':>6s} {'InitErr':>10s} {'FinalErr':>10s} "
        f"{'ErrRed%':>9s} {'ms':>8s} {'Speedup':>9s} {'TFLOP/s':>10s}"
    )
    for r in rows:
        err_red_pct = 100.0 * r.error_reduction_frac
        print(
            f"{r.name:30s} {r.rank:6d} {r.init_rel_l2_error:10.4e} {r.final_rel_l2_error:10.4e} "
            f"{err_red_pct:9.2f} {r.approx_ms:8.3f} {r.speedup_vs_dense:9.3f} {r.approx_tflops:10.3f}"
        )


def recover_linear(
    *,
    device: torch.device,
    bench_dtype: torch.dtype,
    dmodel: int,
    tokens_train: int,
    tokens_val: int,
    tokens_bench: int,
    rank_fracs: list[float],
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    warmup: int,
    iters: int,
) -> list[RecoveryResult]:
    x_train = torch.randn(tokens_train, dmodel, device=device, dtype=torch.float32)
    x_val = torch.randn(tokens_val, dmodel, device=device, dtype=torch.float32)
    x_bench = torch.randn(tokens_bench, dmodel, device=device, dtype=bench_dtype)
    w = torch.randn(dmodel, dmodel, device=device, dtype=torch.float32)

    with torch.no_grad():
        y_val_teacher = x_val @ w

    dense_bench_w = w.to(dtype=bench_dtype)
    dense_fn = lambda: x_bench @ dense_bench_w
    dense_ms, _ = benchmark_fn(dense_fn, warmup, iters, device)
    dense_flops = dense_linear_flops(tokens_bench, dmodel, dmodel)

    u, s, vh = build_svd_factors(w)
    max_rank = min(w.shape)
    rows: list[RecoveryResult] = []

    for frac in rank_fracs:
        rank = max(1, min(max_rank, int(round(max_rank * frac))))
        a0, b0 = materialize_rank_factors(u, s, vh, rank, device)

        with torch.no_grad():
            y_val_init = (x_val @ a0) @ b0
            init_err = rel_l2_error(y_val_init, y_val_teacher)

        a = torch.nn.Parameter(a0.clone())
        b = torch.nn.Parameter(b0.clone())
        opt = torch.optim.AdamW([a, b], lr=lr, weight_decay=weight_decay)

        first_loss = None
        last_loss = None
        for _ in range(steps):
            idx = torch.randint(0, tokens_train, (batch_size,), device=device)
            xb = x_train[idx]
            with torch.no_grad():
                yb_teacher = xb @ w
            yb_pred = (xb @ a) @ b
            loss = F.mse_loss(yb_pred, yb_teacher)
            if first_loss is None:
                first_loss = float(loss.item())
            last_loss = float(loss.item())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        with torch.no_grad():
            y_val_final = (x_val @ a) @ b
            final_err = rel_l2_error(y_val_final, y_val_teacher)

        a_bench = a.detach().to(dtype=bench_dtype)
        b_bench = b.detach().to(dtype=bench_dtype)
        approx_fn = lambda: (x_bench @ a_bench) @ b_bench
        approx_ms, _ = benchmark_fn(approx_fn, warmup, iters, device)

        approx_flops = low_rank_linear_flops(tokens_bench, dmodel, dmodel, rank)
        approx_tflops = approx_flops / (approx_ms / 1000.0) / 1e12
        err_red = (init_err - final_err) / init_err if init_err > 0 else 0.0

        rows.append(
            RecoveryResult(
                name=f"linear_low_rank(frac={frac:.3f})",
                rank=rank,
                init_rel_l2_error=init_err,
                final_rel_l2_error=final_err,
                error_reduction_frac=err_red,
                dense_ms=dense_ms,
                approx_ms=approx_ms,
                speedup_vs_dense=dense_ms / approx_ms,
                dense_flops=dense_flops,
                approx_flops=approx_flops,
                approx_tflops=approx_tflops,
                train_loss_start=first_loss if first_loss is not None else float("nan"),
                train_loss_end=last_loss if last_loss is not None else float("nan"),
            )
        )

    return rows


def recover_mlp(
    *,
    device: torch.device,
    bench_dtype: torch.dtype,
    dmodel: int,
    expansion: int,
    tokens_train: int,
    tokens_val: int,
    tokens_bench: int,
    rank_fracs: list[float],
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    warmup: int,
    iters: int,
) -> list[RecoveryResult]:
    hidden = expansion * dmodel
    x_train = torch.randn(tokens_train, dmodel, device=device, dtype=torch.float32)
    x_val = torch.randn(tokens_val, dmodel, device=device, dtype=torch.float32)
    x_bench = torch.randn(tokens_bench, dmodel, device=device, dtype=bench_dtype)
    w1 = torch.randn(dmodel, hidden, device=device, dtype=torch.float32)
    w2 = torch.randn(hidden, dmodel, device=device, dtype=torch.float32)

    def dense_mlp(x: torch.Tensor) -> torch.Tensor:
        h = x @ w1
        h = F.relu(h).square()
        return h @ w2

    with torch.no_grad():
        y_val_teacher = dense_mlp(x_val)

    dense_bench_w1 = w1.to(dtype=bench_dtype)
    dense_bench_w2 = w2.to(dtype=bench_dtype)
    def dense_fn() -> torch.Tensor:
        h = x_bench @ dense_bench_w1
        h = F.relu(h).square()
        return h @ dense_bench_w2

    dense_ms, _ = benchmark_fn(dense_fn, warmup, iters, device)
    dense_flops = dense_linear_flops(tokens_bench, dmodel, hidden) + dense_linear_flops(tokens_bench, hidden, dmodel)

    u1, s1, vh1 = build_svd_factors(w1)
    u2, s2, vh2 = build_svd_factors(w2)
    max_rank = dmodel
    rows: list[RecoveryResult] = []

    for frac in rank_fracs:
        rank = max(1, min(max_rank, int(round(max_rank * frac))))
        a1_0, b1_0 = materialize_rank_factors(u1, s1, vh1, rank, device)
        a2_0, b2_0 = materialize_rank_factors(u2, s2, vh2, rank, device)

        with torch.no_grad():
            h_init = (x_val @ a1_0) @ b1_0
            h_init = F.relu(h_init).square()
            y_val_init = (h_init @ a2_0) @ b2_0
            init_err = rel_l2_error(y_val_init, y_val_teacher)

        a1 = torch.nn.Parameter(a1_0.clone())
        b1 = torch.nn.Parameter(b1_0.clone())
        a2 = torch.nn.Parameter(a2_0.clone())
        b2 = torch.nn.Parameter(b2_0.clone())
        opt = torch.optim.AdamW([a1, b1, a2, b2], lr=lr, weight_decay=weight_decay)

        first_loss = None
        last_loss = None
        for _ in range(steps):
            idx = torch.randint(0, tokens_train, (batch_size,), device=device)
            xb = x_train[idx]
            with torch.no_grad():
                yb_teacher = dense_mlp(xb)
            h = (xb @ a1) @ b1
            h = F.relu(h).square()
            yb_pred = (h @ a2) @ b2
            loss = F.mse_loss(yb_pred, yb_teacher)
            if first_loss is None:
                first_loss = float(loss.item())
            last_loss = float(loss.item())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        with torch.no_grad():
            h = (x_val @ a1) @ b1
            h = F.relu(h).square()
            y_val_final = (h @ a2) @ b2
            final_err = rel_l2_error(y_val_final, y_val_teacher)

        a1_b = a1.detach().to(dtype=bench_dtype)
        b1_b = b1.detach().to(dtype=bench_dtype)
        a2_b = a2.detach().to(dtype=bench_dtype)
        b2_b = b2.detach().to(dtype=bench_dtype)
        def approx_fn() -> torch.Tensor:
            h = (x_bench @ a1_b) @ b1_b
            h = F.relu(h).square()
            return (h @ a2_b) @ b2_b

        approx_ms, _ = benchmark_fn(approx_fn, warmup, iters, device)
        approx_flops = (
            low_rank_linear_flops(tokens_bench, dmodel, hidden, rank)
            + low_rank_linear_flops(tokens_bench, hidden, dmodel, rank)
        )
        approx_tflops = approx_flops / (approx_ms / 1000.0) / 1e12
        err_red = (init_err - final_err) / init_err if init_err > 0 else 0.0

        rows.append(
            RecoveryResult(
                name=f"mlp_low_rank(frac={frac:.3f})",
                rank=rank,
                init_rel_l2_error=init_err,
                final_rel_l2_error=final_err,
                error_reduction_frac=err_red,
                dense_ms=dense_ms,
                approx_ms=approx_ms,
                speedup_vs_dense=dense_ms / approx_ms,
                dense_flops=dense_flops,
                approx_flops=approx_flops,
                approx_tflops=approx_tflops,
                train_loss_start=first_loss if first_loss is not None else float("nan"),
                train_loss_end=last_loss if last_loss is not None else float("nan"),
            )
        )

    return rows


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    bench_dtype = pick_dtype(args.dtype)
    rank_fracs = parse_rank_fracs(args.rank_fracs)

    print(f"Device: {device}")
    print(f"Bench dtype: {bench_dtype}")
    print(f"Benchmark: {args.benchmark}")
    print(f"dmodel={args.dmodel}, expansion={args.expansion}")
    print(f"tokens(train/val/bench)=({args.tokens_train}, {args.tokens_val}, {args.tokens_bench})")
    print(f"rank_fracs={rank_fracs}")
    print(f"recovery steps={args.steps}, batch_size={args.batch_size}, lr={args.lr}, wd={args.weight_decay}")
    print(f"warmup/iters={args.warmup}/{args.iters}")

    # Sanity check for dtype/device.
    try:
        _ = (
            torch.randn(128, 128, device=device, dtype=bench_dtype)
            @ torch.randn(128, 128, device=device, dtype=bench_dtype)
        )
        sync(device)
    except Exception as e:
        raise RuntimeError(f"Requested dtype/device combo failed ({bench_dtype} on {device}): {e}") from e

    linear_rows: list[RecoveryResult] = []
    mlp_rows: list[RecoveryResult] = []

    if args.benchmark in ("linear", "both"):
        print("\nRunning linear recovery...")
        linear_rows = recover_linear(
            device=device,
            bench_dtype=bench_dtype,
            dmodel=args.dmodel,
            tokens_train=args.tokens_train,
            tokens_val=args.tokens_val,
            tokens_bench=args.tokens_bench,
            rank_fracs=rank_fracs,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup=args.warmup,
            iters=args.iters,
        )
        print_table("Linear Recovery", linear_rows)

    if args.benchmark in ("mlp", "both"):
        print("\nRunning MLP recovery...")
        mlp_rows = recover_mlp(
            device=device,
            bench_dtype=bench_dtype,
            dmodel=args.dmodel,
            expansion=args.expansion,
            tokens_train=args.tokens_train,
            tokens_val=args.tokens_val,
            tokens_bench=args.tokens_bench,
            rank_fracs=rank_fracs,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup=args.warmup,
            iters=args.iters,
        )
        print_table("MLP Recovery", mlp_rows)

    if args.json_out:
        device_name = None
        if device.type == "cuda":
            device_idx = device.index if device.index is not None else 0
            device_name = torch.cuda.get_device_name(device_idx)

        payload = {
            "script": "loss_aware_low_rank_recovery.py",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": {
                "device_arg": args.device,
                "resolved_device": str(device),
                "dtype": args.dtype,
                "seed": args.seed,
                "benchmark": args.benchmark,
                "dmodel": args.dmodel,
                "expansion": args.expansion,
                "tokens_train": args.tokens_train,
                "tokens_val": args.tokens_val,
                "tokens_bench": args.tokens_bench,
                "rank_fracs": rank_fracs,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "warmup": args.warmup,
                "iters": args.iters,
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

    print("\nDone. Prefer candidates with strong error reduction and meaningful speedup.")


if __name__ == "__main__":
    main()
