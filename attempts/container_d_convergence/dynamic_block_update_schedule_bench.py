#!/usr/bin/env python3
"""
Dynamic block update schedule benchmark.

Goal:
Test whether dynamic gradient-relevance-guided suffix updates can improve
speed/quality tradeoff vs full updates.

Schedules compared:
1) full_update: all blocks updated every step.
2) static_suffix: only top K blocks updated every step.
3) dynamic_suffix: relevance EMA from periodic probe/full steps; train top suffix otherwise.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RunResult:
    schedule: str
    steps_executed: int
    elapsed_ms: float
    final_val_loss: float
    train_loss_start: float
    train_loss_end: float
    mean_step_ms: float
    p50_step_ms: float
    p90_step_ms: float
    speedup_vs_full: float
    active_layers_mean: float
    active_layers_min: int
    active_layers_max: int
    probe_step_count: int
    notes: str


class TinyBlock(nn.Module):
    def __init__(self, dmodel: int, hidden_mult: int) -> None:
        super().__init__()
        hidden = dmodel * hidden_mult
        self.fc1 = nn.Linear(dmodel, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dmodel, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.rms_norm(x, (x.size(-1),))
        h = F.relu(self.fc1(h)).square()
        h = self.fc2(h)
        return x + h


class TinyStack(nn.Module):
    def __init__(self, dmodel: int, n_layers: int, hidden_mult: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([TinyBlock(dmodel, hidden_mult) for _ in range(n_layers)])
        self.head = nn.Linear(dmodel, dmodel, bias=False)

    def forward(self, x: torch.Tensor, active_start: int = 0) -> torch.Tensor:
        # If active_start > 0, prefix blocks are run in no_grad and detached.
        # This approximates compute savings from freezing a contiguous prefix.
        if active_start > 0:
            with torch.no_grad():
                for i in range(active_start):
                    x = self.blocks[i](x)
            x = x.detach()
            for i in range(active_start, len(self.blocks)):
                x = self.blocks[i](x)
        else:
            for b in self.blocks:
                x = b(x)
        return self.head(F.rms_norm(x, (x.size(-1),)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dynamic block update schedule benchmark")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dmodel", type=int, default=1024)
    p.add_argument("--n-layers", type=int, default=18)
    p.add_argument("--hidden-mult", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--equal-time-budget", action="store_true", help="Run static/dynamic with same wall-clock budget as full_update")
    p.add_argument("--max-steps-multiplier", type=float, default=3.0, help="Safety cap for equal-time mode")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--teacher-noise", type=float, default=0.02, help="init student near teacher")
    p.add_argument("--val-batches", type=int, default=4)
    p.add_argument("--warmup-ignore-steps", type=int, default=5, help="ignore first steps for time averaging")

    # Static suffix
    p.add_argument("--static-active-layers", type=int, default=6, help="K top layers active in static suffix")

    # Dynamic suffix
    p.add_argument("--dyn-warmup-steps", type=int, default=40, help="full-update warmup for relevance EMA")
    p.add_argument("--dyn-probe-every", type=int, default=20, help="full probe/update every K steps")
    p.add_argument("--dyn-refresh-every", type=int, default=80, help="forced full update every M steps")
    p.add_argument(
        "--dyn-relevance-metric",
        type=str,
        default="grad_ratio",
        choices=["grad_ratio", "saliency_abs_gp"],
        help="Per-block relevance metric on probe steps",
    )
    p.add_argument("--dyn-relevance-threshold", type=float, default=0.9, help="top suffix cumulative relevance fraction")
    p.add_argument("--dyn-min-active-layers", type=int, default=4)
    p.add_argument("--dyn-max-active-layers", type=int, default=0, help="0 means no cap; otherwise cap active suffix size")
    p.add_argument("--dyn-adapt-budget", action="store_true", help="Adapt max active suffix size based on probe loss")
    p.add_argument("--dyn-freeze-start-step", type=int, default=-1, help="Step when freezing can start (-1 uses frac)")
    p.add_argument("--dyn-freeze-start-frac", type=float, default=0.5, help="If freeze-start-step=-1, start after this fraction of run")
    p.add_argument("--dyn-require-stable", action="store_true", help="Require loss stability before enabling freezing")
    p.add_argument("--dyn-stability-window", type=int, default=12)
    p.add_argument("--dyn-stability-rel-change", type=float, default=0.02, help="relative mean-loss change threshold")
    p.add_argument("--dyn-stability-cv", type=float, default=0.03, help="coefficient-of-variation threshold")
    p.add_argument(
        "--dyn-final-full-frac",
        type=float,
        default=0.0,
        help="Enable final recovery window in final fraction of run/time budget",
    )
    p.add_argument(
        "--dyn-final-recovery-mode",
        type=str,
        default="full",
        choices=["full", "staged"],
        help="Behavior inside final recovery window (enabled when dyn-final-full-frac > 0)",
    )
    p.add_argument(
        "--dyn-final-recovery-stage1-progress",
        type=float,
        default=0.34,
        help="For staged mode: first boundary in [0,1] inside final window",
    )
    p.add_argument(
        "--dyn-final-recovery-stage2-progress",
        type=float,
        default=0.67,
        help="For staged mode: second boundary in [0,1] inside final window",
    )
    p.add_argument(
        "--dyn-final-recovery-stage1-layers",
        type=int,
        default=14,
        help="For staged mode: minimum active layers in stage 1",
    )
    p.add_argument(
        "--dyn-final-recovery-stage2-layers",
        type=int,
        default=16,
        help="For staged mode: minimum active layers in stage 2",
    )
    p.add_argument("--dyn-loss-upper", type=float, default=1.03, help="If probe loss > probe_loss_ema * upper, increase active budget")
    p.add_argument("--dyn-loss-lower", type=float, default=0.995, help="If probe loss < probe_loss_ema * lower, decrease active budget")
    p.add_argument("--dyn-budget-step-up", type=int, default=2)
    p.add_argument("--dyn-budget-step-down", type=int, default=1)
    p.add_argument("--dyn-loss-ema-decay", type=float, default=0.9, help="EMA decay for probe-loss adaptation signal")
    p.add_argument("--dyn-ema-decay", type=float, default=0.9)
    p.add_argument("--dyn-eps", type=float, default=1e-8)

    p.add_argument("--json-out", type=str, default="", help="Optional JSON output path")
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


def percentile(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    idx = int(round((len(s) - 1) * q))
    return s[idx]


def json_float(x: float) -> float | None:
    if not math.isfinite(x):
        return None
    return float(x)


def result_to_dict(r: RunResult) -> dict:
    return {
        "schedule": r.schedule,
        "steps_executed": r.steps_executed,
        "elapsed_ms": json_float(r.elapsed_ms),
        "final_val_loss": json_float(r.final_val_loss),
        "train_loss_start": json_float(r.train_loss_start),
        "train_loss_end": json_float(r.train_loss_end),
        "mean_step_ms": json_float(r.mean_step_ms),
        "p50_step_ms": json_float(r.p50_step_ms),
        "p90_step_ms": json_float(r.p90_step_ms),
        "speedup_vs_full": json_float(r.speedup_vs_full),
        "active_layers_mean": json_float(r.active_layers_mean),
        "active_layers_min": r.active_layers_min,
        "active_layers_max": r.active_layers_max,
        "probe_step_count": r.probe_step_count,
        "notes": r.notes,
    }


def write_json(path: str, payload: dict) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@torch.no_grad()
def init_student_from_teacher(student: TinyStack, teacher: TinyStack, noise_scale: float) -> None:
    sd = teacher.state_dict()
    ssd = student.state_dict()
    for k in ssd.keys():
        base = sd[k].clone()
        if noise_scale > 0:
            base = base + noise_scale * torch.randn_like(base)
        ssd[k].copy_(base)


def make_batch(
    batch_size: int,
    seq_len: int,
    dmodel: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.randn(batch_size, seq_len, dmodel, device=device, dtype=dtype)


@torch.no_grad()
def build_validation_set(
    teacher: TinyStack,
    val_batches: int,
    batch_size: int,
    seq_len: int,
    dmodel: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(val_batches):
        x = make_batch(batch_size, seq_len, dmodel, device, dtype)
        y = teacher(x, active_start=0)
        out.append((x, y))
    return out


@torch.no_grad()
def evaluate_val_loss(model: TinyStack, val_set: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
    losses = []
    for x, y in val_set:
        pred = model(x, active_start=0)
        losses.append(F.mse_loss(pred.float(), y.float()).item())
    return float(sum(losses) / max(1, len(losses)))


def block_relevance_scores(
    model: TinyStack,
    eps: float,
    metric: str,
) -> list[float]:
    scores: list[float] = []
    if metric not in {"grad_ratio", "saliency_abs_gp"}:
        raise ValueError(f"Unknown relevance metric: {metric}")
    for b in model.blocks:
        if metric == "grad_ratio":
            g2 = 0.0
            w2 = 0.0
            for p in b.parameters():
                if p.grad is not None:
                    g2 += p.grad.float().pow(2).sum().item()
                w2 += p.data.float().pow(2).sum().item()
            g = math.sqrt(max(g2, 0.0))
            w = math.sqrt(max(w2, 0.0))
            scores.append(g / (w + eps))
        else:
            saliency = 0.0
            for p in b.parameters():
                if p.grad is None:
                    continue
                saliency += (p.grad.float() * p.data.float()).abs().sum().item()
            scores.append(saliency)
    return scores


def choose_active_start_from_scores(
    scores: list[float],
    threshold: float,
    min_active_layers: int,
    max_active_layers: int,
) -> int:
    n = len(scores)
    min_active_layers = max(1, min(min_active_layers, n))
    if max_active_layers <= 0:
        max_active = n
    else:
        max_active = max(1, min(max_active_layers, n))
    if max_active < min_active_layers:
        max_active = min_active_layers

    total = sum(scores)
    if total <= 0:
        return n - max_active

    target = threshold * total
    running = 0.0
    active_start = n - max_active
    for i in range(n - 1, -1, -1):
        running += scores[i]
        active_layers = n - i
        if running >= target and min_active_layers <= active_layers <= max_active:
            active_start = i
            break
        if active_layers > max_active:
            break
    # Ensure minimum active layers always respected.
    active_start = min(active_start, n - min_active_layers)
    # Ensure maximum active layers cap respected.
    active_start = max(active_start, n - max_active)
    active_start = max(0, active_start)
    return active_start


def loss_is_stable(
    losses: list[float],
    window: int,
    rel_change_threshold: float,
    cv_threshold: float,
) -> tuple[bool, dict]:
    if window <= 0 or len(losses) < 2 * window:
        return False, {}
    prev = losses[-2 * window : -window]
    curr = losses[-window:]
    prev_mean = sum(prev) / window
    curr_mean = sum(curr) / window
    rel_change = abs(curr_mean - prev_mean) / (abs(prev_mean) + 1e-12)
    curr_var = sum((x - curr_mean) ** 2 for x in curr) / window
    curr_std = math.sqrt(max(curr_var, 0.0))
    cv = curr_std / (abs(curr_mean) + 1e-12)
    ok = rel_change <= rel_change_threshold and cv <= cv_threshold
    return ok, {
        "prev_mean": prev_mean,
        "curr_mean": curr_mean,
        "rel_change": rel_change,
        "cv": cv,
    }


def run_schedule(
    *,
    schedule: str,
    model: TinyStack,
    teacher: TinyStack,
    val_set: list[tuple[torch.Tensor, torch.Tensor]],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    steps_limit: int,
    time_budget_ms: float | None,
) -> tuple[RunResult, dict]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n = len(model.blocks)

    if schedule == "static_suffix":
        k = max(1, min(args.static_active_layers, n))
        static_active_start = n - k
    else:
        static_active_start = 0

    ema_scores = [0.0 for _ in range(n)]
    min_active = max(1, min(args.dyn_min_active_layers, n))
    max_active_cap = n if args.dyn_max_active_layers <= 0 else max(1, min(args.dyn_max_active_layers, n))
    if max_active_cap < min_active:
        max_active_cap = min_active
    active_budget = max_active_cap
    active_start_dyn = max(0, n - active_budget)
    best_probe_loss = float("inf")
    probe_loss_ema = None
    stage1_progress = float(min(1.0, max(0.0, args.dyn_final_recovery_stage1_progress)))
    stage2_progress = float(min(1.0, max(0.0, args.dyn_final_recovery_stage2_progress)))
    if stage2_progress < stage1_progress:
        stage2_progress = stage1_progress
    stage1_layers = max(min_active, min(args.dyn_final_recovery_stage1_layers, n))
    stage2_layers = max(stage1_layers, min(args.dyn_final_recovery_stage2_layers, n))

    freeze_enabled = schedule != "dynamic_suffix"
    if schedule == "dynamic_suffix":
        if args.dyn_freeze_start_step >= 0:
            freeze_start_step = max(0, args.dyn_freeze_start_step)
        else:
            freeze_start_step = max(0, int(round(args.dyn_freeze_start_frac * steps_limit)))
        freeze_start_time_ms = None if time_budget_ms is None else max(0.0, args.dyn_freeze_start_frac * time_budget_ms)
    else:
        freeze_start_step = 0
        freeze_start_time_ms = None

    step_times: list[float] = []
    train_losses: list[float] = []
    active_layers_hist: list[int] = []
    probe_steps = 0
    dynamic_trace: list[dict] = []
    prev_final_recovery_stage = None

    elapsed_ms = 0.0
    step = 0
    while step < steps_limit:
        x = make_batch(args.batch_size, args.seq_len, args.dmodel, device, dtype)
        with torch.no_grad():
            y = teacher(x, active_start=0)

        is_probe = False
        if schedule == "full_update":
            active_start = 0
        elif schedule == "static_suffix":
            active_start = static_active_start
        elif schedule == "dynamic_suffix":
            # Gate freezing: only start suffix-freezing after configured point and (optionally) stability.
            if not freeze_enabled:
                reached_start = False
                if freeze_start_time_ms is not None:
                    reached_start = elapsed_ms >= freeze_start_time_ms
                else:
                    reached_start = step >= freeze_start_step

                stable_ok, stable_stats = loss_is_stable(
                    train_losses,
                    window=args.dyn_stability_window,
                    rel_change_threshold=args.dyn_stability_rel_change,
                    cv_threshold=args.dyn_stability_cv,
                )
                if reached_start and ((not args.dyn_require_stable) or stable_ok):
                    freeze_enabled = True
                    dynamic_trace.append(
                        {
                            "step": step,
                            "freeze_enabled": True,
                            "freeze_trigger": {
                                "reached_start": reached_start,
                                "stable_ok": stable_ok,
                                **stable_stats,
                            },
                        }
                    )

            # Optional final recovery phase near the end.
            final_recovery_active_start = None
            final_recovery_stage = None
            final_recovery_progress = None
            if args.dyn_final_full_frac > 0:
                if time_budget_ms is not None and time_budget_ms > 0:
                    recovery_start = (1.0 - args.dyn_final_full_frac) * time_budget_ms
                    in_recovery = elapsed_ms >= recovery_start
                    if in_recovery:
                        denom = max(args.dyn_final_full_frac * time_budget_ms, 1e-9)
                        final_recovery_progress = float(min(1.0, max(0.0, (elapsed_ms - recovery_start) / denom)))
                else:
                    recovery_start = int((1.0 - args.dyn_final_full_frac) * steps_limit)
                    in_recovery = step >= recovery_start
                    if in_recovery:
                        denom = max(1, steps_limit - recovery_start)
                        final_recovery_progress = float(min(1.0, max(0.0, (step - recovery_start) / denom)))

                if in_recovery:
                    if args.dyn_final_recovery_mode == "full":
                        final_recovery_stage = "full"
                        final_recovery_active_start = 0
                    else:
                        current_active_layers = n - active_start_dyn
                        if final_recovery_progress is None:
                            progress = 1.0
                        else:
                            progress = final_recovery_progress
                        if progress >= stage2_progress:
                            final_recovery_stage = "stage3_full"
                            target_active_layers = n
                        elif progress >= stage1_progress:
                            final_recovery_stage = "stage2"
                            target_active_layers = max(current_active_layers, stage2_layers)
                        else:
                            final_recovery_stage = "stage1"
                            target_active_layers = max(current_active_layers, stage1_layers)
                        target_active_layers = min(n, max(min_active, target_active_layers))
                        final_recovery_active_start = max(0, n - target_active_layers)

            if schedule == "dynamic_suffix" and final_recovery_stage != prev_final_recovery_stage:
                if final_recovery_stage is not None:
                    dynamic_trace.append(
                        {
                            "step": step,
                            "final_recovery_mode": args.dyn_final_recovery_mode,
                            "final_recovery_stage": final_recovery_stage,
                            "final_recovery_progress": final_recovery_progress,
                            "final_recovery_active_layers": n - final_recovery_active_start,
                        }
                    )
                prev_final_recovery_stage = final_recovery_stage

            # Probe conditions: warmup, periodic probe, periodic refresh.
            is_warmup = step < args.dyn_warmup_steps
            is_periodic_probe = (args.dyn_probe_every > 0 and step % args.dyn_probe_every == 0)
            is_refresh = (args.dyn_refresh_every > 0 and step % args.dyn_refresh_every == 0)
            is_probe = (is_warmup or is_periodic_probe or is_refresh) and (final_recovery_active_start is None)

            # Before freeze is enabled, stay full-update (but still allow probes for relevance signals).
            if not freeze_enabled:
                active_start = 0
            elif final_recovery_active_start is not None:
                active_start = final_recovery_active_start
            else:
                active_start = 0 if is_probe else active_start_dyn
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        optimizer.zero_grad(set_to_none=True)
        sync(device)
        t0 = time.perf_counter()
        pred = model(x, active_start=active_start)
        loss = F.mse_loss(pred.float(), y.float())
        loss.backward()

        if schedule == "dynamic_suffix" and is_probe:
            probe_steps += 1
            probe_loss = float(loss.item())
            if probe_loss < best_probe_loss:
                best_probe_loss = probe_loss

            prev_probe_loss_ema = probe_loss_ema
            if probe_loss_ema is None:
                probe_loss_ema = probe_loss
            else:
                if args.dyn_adapt_budget:
                    if probe_loss > probe_loss_ema * args.dyn_loss_upper:
                        active_budget = min(max_active_cap, active_budget + args.dyn_budget_step_up)
                    elif probe_loss < probe_loss_ema * args.dyn_loss_lower:
                        active_budget = max(min_active, active_budget - args.dyn_budget_step_down)
                probe_loss_ema = (
                    args.dyn_loss_ema_decay * probe_loss_ema
                    + (1.0 - args.dyn_loss_ema_decay) * probe_loss
                )

            scores = block_relevance_scores(model, eps=args.dyn_eps, metric=args.dyn_relevance_metric)
            for i, s in enumerate(scores):
                ema_scores[i] = args.dyn_ema_decay * ema_scores[i] + (1.0 - args.dyn_ema_decay) * s

            active_start_dyn = choose_active_start_from_scores(
                ema_scores,
                threshold=args.dyn_relevance_threshold,
                min_active_layers=args.dyn_min_active_layers,
                max_active_layers=active_budget,
            )
            dynamic_trace.append(
                {
                    "step": step,
                    "probe": True,
                    "freeze_enabled": freeze_enabled,
                    "probe_loss": probe_loss,
                    "best_probe_loss": best_probe_loss,
                    "probe_loss_ema_prev": prev_probe_loss_ema,
                    "probe_loss_ema": probe_loss_ema,
                    "active_budget": active_budget,
                    "active_start_next": active_start_dyn,
                    "active_layers_next": n - active_start_dyn,
                    "ema_scores": ema_scores.copy(),
                }
            )

        optimizer.step()
        sync(device)
        t1 = time.perf_counter()

        dt_ms = (t1 - t0) * 1000.0
        step_times.append(dt_ms)
        elapsed_ms += dt_ms
        train_losses.append(float(loss.item()))
        active_layers_hist.append(n - active_start)

        step += 1
        if time_budget_ms is not None and elapsed_ms >= time_budget_ms and step >= max(1, args.warmup_ignore_steps):
            break

    val_loss = evaluate_val_loss(model, val_set)
    times_for_stats = step_times[args.warmup_ignore_steps :] if len(step_times) > args.warmup_ignore_steps else step_times
    mean_ms = float(sum(times_for_stats) / max(1, len(times_for_stats)))

    result = RunResult(
        schedule=schedule,
        steps_executed=step,
        elapsed_ms=elapsed_ms,
        final_val_loss=val_loss,
        train_loss_start=float(train_losses[0]) if train_losses else float("nan"),
        train_loss_end=float(train_losses[-1]) if train_losses else float("nan"),
        mean_step_ms=mean_ms,
        p50_step_ms=float(percentile(times_for_stats, 0.5)),
        p90_step_ms=float(percentile(times_for_stats, 0.9)),
        speedup_vs_full=float("nan"),  # filled later
        active_layers_mean=float(sum(active_layers_hist) / max(1, len(active_layers_hist))),
        active_layers_min=min(active_layers_hist) if active_layers_hist else 0,
        active_layers_max=max(active_layers_hist) if active_layers_hist else 0,
        probe_step_count=probe_steps,
        notes=(
            "full update" if schedule == "full_update" else
            f"static active suffix={n-static_active_start}" if schedule == "static_suffix" else
            (
                f"dynamic relevance-guided active suffix (metric={args.dyn_relevance_metric})"
                + (" + adaptive budget" if args.dyn_adapt_budget else "")
                + (" + stability-gated freeze" if args.dyn_require_stable else "")
                + (
                    f" + final_recovery={args.dyn_final_recovery_mode}"
                    f"@frac={args.dyn_final_full_frac}"
                    if args.dyn_final_full_frac > 0
                    else ""
                )
            )
        ),
    )
    extra = {
        "step_times_ms": step_times,
        "train_losses": train_losses,
        "active_layers_hist": active_layers_hist,
        "dynamic_trace": dynamic_trace,
    }
    return result, extra


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype)

    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Model: dmodel={args.dmodel}, n_layers={args.n_layers}, hidden_mult={args.hidden_mult}")
    print(f"Batch: batch_size={args.batch_size}, seq_len={args.seq_len}, steps={args.steps}")
    if args.equal_time_budget:
        print(f"Equal-time mode enabled (max_steps_multiplier={args.max_steps_multiplier})")
    print("Schedules: full_update, static_suffix, dynamic_suffix")

    # Quick dtype/device check.
    try:
        _ = (
            torch.randn(64, 64, device=device, dtype=dtype)
            @ torch.randn(64, 64, device=device, dtype=dtype)
        )
        sync(device)
    except Exception as e:
        raise RuntimeError(f"Requested dtype/device combo failed ({dtype} on {device}): {e}") from e

    teacher = TinyStack(args.dmodel, args.n_layers, args.hidden_mult).to(device=device, dtype=dtype)
    teacher.eval()

    val_set = build_validation_set(
        teacher=teacher,
        val_batches=args.val_batches,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        dmodel=args.dmodel,
        device=device,
        dtype=dtype,
    )

    # Base student initialization from teacher + noise for fair schedule comparison.
    base_student = TinyStack(args.dmodel, args.n_layers, args.hidden_mult).to(device=device, dtype=dtype)
    init_student_from_teacher(base_student, teacher, noise_scale=args.teacher_noise)
    base_state = copy.deepcopy(base_student.state_dict())

    results: list[RunResult] = []
    extras: dict[str, dict] = {}

    # Full update baseline first (also defines time budget in equal-time mode).
    print("\nRunning schedule: full_update")
    student = TinyStack(args.dmodel, args.n_layers, args.hidden_mult).to(device=device, dtype=dtype)
    student.load_state_dict(base_state, strict=True)
    full_res, full_extra = run_schedule(
        schedule="full_update",
        model=student,
        teacher=teacher,
        val_set=val_set,
        args=args,
        device=device,
        dtype=dtype,
        steps_limit=args.steps,
        time_budget_ms=None,
    )
    results.append(full_res)
    extras["full_update"] = full_extra
    print(
        f"{'full_update':14s} | val_loss={full_res.final_val_loss:.6f} | "
        f"mean_ms={full_res.mean_step_ms:.3f} | active_layers_mean={full_res.active_layers_mean:.2f} | "
        f"steps={full_res.steps_executed}"
    )

    baseline_time_budget_ms = full_res.elapsed_ms
    for schedule in ("static_suffix", "dynamic_suffix"):
        print(f"\nRunning schedule: {schedule}")
        student = TinyStack(args.dmodel, args.n_layers, args.hidden_mult).to(device=device, dtype=dtype)
        student.load_state_dict(base_state, strict=True)

        if args.equal_time_budget:
            steps_limit = max(args.steps, int(round(args.steps * args.max_steps_multiplier)))
            time_budget_ms = baseline_time_budget_ms
        else:
            steps_limit = args.steps
            time_budget_ms = None

        res, extra = run_schedule(
            schedule=schedule,
            model=student,
            teacher=teacher,
            val_set=val_set,
            args=args,
            device=device,
            dtype=dtype,
            steps_limit=steps_limit,
            time_budget_ms=time_budget_ms,
        )
        results.append(res)
        extras[schedule] = extra
        print(
            f"{schedule:14s} | val_loss={res.final_val_loss:.6f} | "
            f"mean_ms={res.mean_step_ms:.3f} | active_layers_mean={res.active_layers_mean:.2f} | "
            f"steps={res.steps_executed}"
        )

    # Fill speedups vs full.
    full_ms = next(r.mean_step_ms for r in results if r.schedule == "full_update")
    for r in results:
        r.speedup_vs_full = full_ms / r.mean_step_ms if r.mean_step_ms > 0 else float("nan")

    print("\n=== Summary ===")
    print(f"{'Schedule':14s} {'ValLoss':>10s} {'Mean ms':>10s} {'Speedup':>9s} {'Steps':>8s} {'ActMean':>9s} {'ActMin/Max':>11s} {'Probes':>8s}")
    for r in results:
        print(
            f"{r.schedule:14s} {r.final_val_loss:10.6f} {r.mean_step_ms:10.3f} {r.speedup_vs_full:9.3f} "
            f"{r.steps_executed:8d} {r.active_layers_mean:9.2f} {f'{r.active_layers_min}/{r.active_layers_max}':>11s} {r.probe_step_count:8d}"
        )

    if args.json_out:
        device_name = None
        if device.type == "cuda":
            device_idx = device.index if device.index is not None else 0
            device_name = torch.cuda.get_device_name(device_idx)
        payload = {
            "script": "dynamic_block_update_schedule_bench.py",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": {
                "device_arg": args.device,
                "resolved_device": str(device),
                "dtype": args.dtype,
                "seed": args.seed,
                "dmodel": args.dmodel,
                "n_layers": args.n_layers,
                "hidden_mult": args.hidden_mult,
                "batch_size": args.batch_size,
                "seq_len": args.seq_len,
                "steps": args.steps,
                "equal_time_budget": args.equal_time_budget,
                "max_steps_multiplier": args.max_steps_multiplier,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "teacher_noise": args.teacher_noise,
                "val_batches": args.val_batches,
                "warmup_ignore_steps": args.warmup_ignore_steps,
                "static_active_layers": args.static_active_layers,
                "dyn_warmup_steps": args.dyn_warmup_steps,
                "dyn_probe_every": args.dyn_probe_every,
                "dyn_refresh_every": args.dyn_refresh_every,
                "dyn_relevance_metric": args.dyn_relevance_metric,
                "dyn_relevance_threshold": args.dyn_relevance_threshold,
                "dyn_min_active_layers": args.dyn_min_active_layers,
                "dyn_max_active_layers": args.dyn_max_active_layers,
                "dyn_adapt_budget": args.dyn_adapt_budget,
                "dyn_freeze_start_step": args.dyn_freeze_start_step,
                "dyn_freeze_start_frac": args.dyn_freeze_start_frac,
                "dyn_require_stable": args.dyn_require_stable,
                "dyn_stability_window": args.dyn_stability_window,
                "dyn_stability_rel_change": args.dyn_stability_rel_change,
                "dyn_stability_cv": args.dyn_stability_cv,
                "dyn_final_full_frac": args.dyn_final_full_frac,
                "dyn_final_recovery_mode": args.dyn_final_recovery_mode,
                "dyn_final_recovery_stage1_progress": args.dyn_final_recovery_stage1_progress,
                "dyn_final_recovery_stage2_progress": args.dyn_final_recovery_stage2_progress,
                "dyn_final_recovery_stage1_layers": args.dyn_final_recovery_stage1_layers,
                "dyn_final_recovery_stage2_layers": args.dyn_final_recovery_stage2_layers,
                "dyn_loss_upper": args.dyn_loss_upper,
                "dyn_loss_lower": args.dyn_loss_lower,
                "dyn_budget_step_up": args.dyn_budget_step_up,
                "dyn_budget_step_down": args.dyn_budget_step_down,
                "dyn_loss_ema_decay": args.dyn_loss_ema_decay,
                "dyn_ema_decay": args.dyn_ema_decay,
                "dyn_eps": args.dyn_eps,
            },
            "system": {
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_name": device_name,
                "torch_version": torch.__version__,
            },
            "results": [result_to_dict(r) for r in results],
            "traces": {
                k: {
                    "step_times_ms": v["step_times_ms"],
                    "train_losses": v["train_losses"],
                    "active_layers_hist": v["active_layers_hist"],
                    "dynamic_trace": v["dynamic_trace"],
                }
                for k, v in extras.items()
            },
        }
        write_json(args.json_out, payload)
        print(f"\nJSON report saved to: {args.json_out}")

    print("\nDone. Compare dynamic_suffix against full_update for speedup at acceptable val loss delta.")


if __name__ == "__main__":
    main()
