#!/usr/bin/env python3
"""
ResFormer value-embedding gate microbenchmark.

Benchmarks the hot path:
    gate = 2 * sigmoid(W @ x_slice)
    v = v + gate * ve

Goal: estimate whether algebraic rewrites / compile materially improve latency or memory
before introducing a custom fused kernel in core model code.
"""

from __future__ import annotations

import argparse
import gc
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
    name: str
    mode: str
    ms: float
    peak_alloc_mib: float
    speedup_vs_baseline: float
    rel_l2_out: float
    rel_l2_grad: float
    status: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ResFormer value-gate microbenchmark")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--n-embd", type=int, default=1536)
    p.add_argument("--n-kv-head", type=int, default=12)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--ve-gate-channels", type=int, default=32)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=80)
    p.add_argument("--mode", type=str, default="both", choices=["forward", "backward", "both"])
    p.add_argument("--compile-candidates", action="store_true", help="also benchmark torch.compile variants")
    p.add_argument("--json-out", type=str, default="", help="optional path to write JSON report")
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


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a32 = a.float()
    b32 = b.float()
    denom = b32.norm().item()
    if denom == 0.0:
        return float("nan")
    return (a32 - b32).norm().item() / denom


def peak_alloc_mib(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)


def maybe_reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def maybe_clear_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def baseline_op(x: torch.Tensor, v: torch.Tensor, ve: torch.Tensor, w: torch.Tensor, c: int) -> torch.Tensor:
    gate = 2.0 * torch.sigmoid(F.linear(x[..., :c], w))
    return v + gate[..., None] * ve


def addcmul_op(x: torch.Tensor, v: torch.Tensor, ve: torch.Tensor, w: torch.Tensor, c: int) -> torch.Tensor:
    gate = 2.0 * torch.sigmoid(F.linear(x[..., :c], w))
    return torch.addcmul(v, ve, gate[..., None])


def mul_then_add_op(x: torch.Tensor, v: torch.Tensor, ve: torch.Tensor, w: torch.Tensor, c: int) -> torch.Tensor:
    gate = F.linear(x[..., :c], w)
    gate = torch.sigmoid(gate)
    gate = gate * 2.0
    return v + ve * gate[..., None]


def build_inputs(
    batch: int,
    seq_len: int,
    n_embd: int,
    n_kv_head: int,
    head_dim: int,
    ve_gate_channels: int,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.randn(batch, seq_len, n_embd, dtype=dtype, device=device, requires_grad=requires_grad)
    v = torch.randn(batch, seq_len, n_kv_head, head_dim, dtype=dtype, device=device, requires_grad=requires_grad)
    ve = torch.randn(batch, seq_len, n_kv_head, head_dim, dtype=dtype, device=device, requires_grad=requires_grad)
    w = torch.randn(n_kv_head, ve_gate_channels, dtype=dtype, device=device, requires_grad=requires_grad)
    grad_out = torch.randn(batch, seq_len, n_kv_head, head_dim, dtype=dtype, device=device)
    return x, v, ve, w, grad_out


def run_once_with_grads(
    fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    v: torch.Tensor,
    ve: torch.Tensor,
    w: torch.Tensor,
    grad_out: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    out = fn(x, v, ve, w)
    out.backward(grad_out)
    grads = [
        x.grad.detach().clone() if x.grad is not None else torch.zeros_like(x),
        v.grad.detach().clone() if v.grad is not None else torch.zeros_like(v),
        ve.grad.detach().clone() if ve.grad is not None else torch.zeros_like(ve),
        w.grad.detach().clone() if w.grad is not None else torch.zeros_like(w),
    ]
    return out.detach().clone(), grads


def bench_forward(
    fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    v: torch.Tensor,
    ve: torch.Tensor,
    w: torch.Tensor,
    warmup: int,
    iters: int,
    device: torch.device,
) -> tuple[float, float]:
    with torch.no_grad():
        for _ in range(warmup):
            _ = fn(x, v, ve, w)
        sync(device)
        maybe_reset_peak(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = fn(x, v, ve, w)
        sync(device)
        t1 = time.perf_counter()
    return ((t1 - t0) * 1000.0 / iters), peak_alloc_mib(device)


def bench_backward(
    fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    v: torch.Tensor,
    ve: torch.Tensor,
    w: torch.Tensor,
    grad_out: torch.Tensor,
    warmup: int,
    iters: int,
    device: torch.device,
) -> tuple[float, float]:
    for _ in range(warmup):
        x.grad = None
        v.grad = None
        ve.grad = None
        w.grad = None
        y = fn(x, v, ve, w)
        y.backward(grad_out)
    sync(device)
    maybe_reset_peak(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        x.grad = None
        v.grad = None
        ve.grad = None
        w.grad = None
        y = fn(x, v, ve, w)
        y.backward(grad_out)
    sync(device)
    t1 = time.perf_counter()
    return ((t1 - t0) * 1000.0 / iters), peak_alloc_mib(device)


def result_to_dict(r: BenchResult) -> dict:
    return {
        "name": r.name,
        "mode": r.mode,
        "ms": None if not math.isfinite(r.ms) else float(r.ms),
        "peak_alloc_mib": None if not math.isfinite(r.peak_alloc_mib) else float(r.peak_alloc_mib),
        "speedup_vs_baseline": None if not math.isfinite(r.speedup_vs_baseline) else float(r.speedup_vs_baseline),
        "rel_l2_out": None if not math.isfinite(r.rel_l2_out) else float(r.rel_l2_out),
        "rel_l2_grad": None if not math.isfinite(r.rel_l2_grad) else float(r.rel_l2_grad),
        "status": r.status,
    }


def print_table(rows: list[BenchResult]) -> None:
    if not rows:
        return
    print("\n=== ResFormer VE Gate Benchmark ===")
    print(f"{'name':32s} {'mode':9s} {'ms':>10s} {'peak_mib':>10s} {'speedup':>10s} {'rel_l2_out':>12s} {'rel_l2_grad':>12s} {'status':>10s}")
    for r in rows:
        ms = "NA" if not math.isfinite(r.ms) else f"{r.ms:.4f}"
        pm = "NA" if not math.isfinite(r.peak_alloc_mib) else f"{r.peak_alloc_mib:.2f}"
        su = "NA" if not math.isfinite(r.speedup_vs_baseline) else f"{r.speedup_vs_baseline:.4f}"
        eo = "NA" if not math.isfinite(r.rel_l2_out) else f"{r.rel_l2_out:.3e}"
        eg = "NA" if not math.isfinite(r.rel_l2_grad) else f"{r.rel_l2_grad:.3e}"
        print(f"{r.name:32s} {r.mode:9s} {ms:>10s} {pm:>10s} {su:>10s} {eo:>12s} {eg:>12s} {r.status:>10s}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype)
    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        # Keep CPU path functional in environments without CUDA BF16/FP16 speedups.
        dtype = torch.float32

    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(
        "Shape: "
        f"B={args.batch}, T={args.seq_len}, n_embd={args.n_embd}, "
        f"n_kv_head={args.n_kv_head}, head_dim={args.head_dim}, gate_channels={args.ve_gate_channels}"
    )
    print(f"Mode: {args.mode} | Warmup/Iters: {args.warmup}/{args.iters} | Compile: {args.compile_candidates}")

    base_fns: list[tuple[str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]] = [
        (
            "baseline_expr",
            lambda x, v, ve, w: baseline_op(x, v, ve, w, args.ve_gate_channels),
        ),
        (
            "addcmul_expr",
            lambda x, v, ve, w: addcmul_op(x, v, ve, w, args.ve_gate_channels),
        ),
        (
            "mul_then_add_expr",
            lambda x, v, ve, w: mul_then_add_op(x, v, ve, w, args.ve_gate_channels),
        ),
    ]

    candidates: list[tuple[str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]] = []
    for name, fn in base_fns:
        candidates.append((f"{name}_eager", fn))
        if args.compile_candidates:
            try:
                cfn = torch.compile(fn, dynamic=False)
                candidates.append((f"{name}_compiled", cfn))
            except Exception as e:
                print(f"WARNING: compile failed for {name}: {type(e).__name__}: {e}")

    # Accuracy reference uses baseline eager.
    x_ref, v_ref, ve_ref, w_ref, grad_out_ref = build_inputs(
        args.batch,
        args.seq_len,
        args.n_embd,
        args.n_kv_head,
        args.head_dim,
        args.ve_gate_channels,
        dtype,
        device,
        requires_grad=True,
    )
    y_ref, grads_ref = run_once_with_grads(candidates[0][1], x_ref, v_ref, ve_ref, w_ref, grad_out_ref)
    x_ref.grad = None
    v_ref.grad = None
    ve_ref.grad = None
    w_ref.grad = None

    rows: list[BenchResult] = []
    baseline_ms: dict[str, float] = {}

    modes = [args.mode] if args.mode != "both" else ["forward", "backward"]
    for mode in modes:
        for name, fn in candidates:
            gc.collect()
            maybe_clear_cache(device)
            try:
                # Accuracy check on fresh tensors.
                x_chk, v_chk, ve_chk, w_chk, grad_out_chk = build_inputs(
                    args.batch,
                    args.seq_len,
                    args.n_embd,
                    args.n_kv_head,
                    args.head_dim,
                    args.ve_gate_channels,
                    dtype,
                    device,
                    requires_grad=True,
                )
                # Copy from reference so candidate comparisons are exact.
                with torch.no_grad():
                    x_chk.copy_(x_ref.detach())
                    v_chk.copy_(v_ref.detach())
                    ve_chk.copy_(ve_ref.detach())
                    w_chk.copy_(w_ref.detach())
                    grad_out_chk.copy_(grad_out_ref.detach())
                y_chk, grads_chk = run_once_with_grads(fn, x_chk, v_chk, ve_chk, w_chk, grad_out_chk)
                rel_out = rel_l2(y_chk, y_ref)
                g_ref = torch.cat([g.float().reshape(-1) for g in grads_ref])
                g_chk = torch.cat([g.float().reshape(-1) for g in grads_chk])
                rel_grad = rel_l2(g_chk, g_ref)

                # Benchmark on separate tensors.
                x, v, ve, w, grad_out = build_inputs(
                    args.batch,
                    args.seq_len,
                    args.n_embd,
                    args.n_kv_head,
                    args.head_dim,
                    args.ve_gate_channels,
                    dtype,
                    device,
                    requires_grad=(mode == "backward"),
                )
                if mode == "forward":
                    ms, peak = bench_forward(fn, x, v, ve, w, args.warmup, args.iters, device)
                else:
                    ms, peak = bench_backward(fn, x, v, ve, w, grad_out, args.warmup, args.iters, device)

                if mode not in baseline_ms:
                    baseline_ms[mode] = ms
                speedup = baseline_ms[mode] / ms if ms > 0 else float("nan")

                rows.append(
                    BenchResult(
                        name=name,
                        mode=mode,
                        ms=ms,
                        peak_alloc_mib=peak,
                        speedup_vs_baseline=speedup,
                        rel_l2_out=rel_out,
                        rel_l2_grad=rel_grad,
                        status="ok",
                    )
                )
            except Exception as e:
                rows.append(
                    BenchResult(
                        name=name,
                        mode=mode,
                        ms=float("nan"),
                        peak_alloc_mib=float("nan"),
                        speedup_vs_baseline=float("nan"),
                        rel_l2_out=float("nan"),
                        rel_l2_grad=float("nan"),
                        status=f"fail:{type(e).__name__}",
                    )
                )

    print_table(rows)

    payload = {
        "script": "resformer_ve_gate_bench.py",
        "config": {
            "device": str(device),
            "dtype": str(dtype),
            "seed": args.seed,
            "batch": args.batch,
            "seq_len": args.seq_len,
            "n_embd": args.n_embd,
            "n_kv_head": args.n_kv_head,
            "head_dim": args.head_dim,
            "ve_gate_channels": args.ve_gate_channels,
            "warmup": args.warmup,
            "iters": args.iters,
            "mode": args.mode,
            "compile_candidates": bool(args.compile_candidates),
        },
        "results": [result_to_dict(r) for r in rows],
    }

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON: {out_path}")


if __name__ == "__main__":
    main()
