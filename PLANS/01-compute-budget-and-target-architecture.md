# Compute Budget And Target Architecture (4090, <24h)

## Why this file exists
Before changing code, we need a **compute-first architecture plan** for 1x RTX 4090 (24GB).  
If this step is wrong, every later optimization is trying to rescue a bad starting point.

## Hard Constraints (Current nanochat stack)
- No FA3 on 4090 (Ada): training uses SDPA fallback.
- Current code warns that non-`L` window patterns can be inefficient without FA3.
- `--fp8` path is tuned for H100-class flow; for 4090 we should assume **bf16-first** planning.
- 24GB VRAM sets strict limits on model size, batch shape, and sequence length.

## Compute Budget Reality Check
Using current speedrun reference numbers (from repo docs/logs):
- GPT-2-grade reference run uses roughly `4.33e19` training FLOPs.
- On 8xH100 this is around `~2.76h` training time.

For 1x4090 in 24h, a practical budget is much smaller:
- Rough peak-ratio shortcut: `8*H100 / 1*4090` implies we have only a fraction of the baseline compute.
- Practical conclusion: expect around **15% to 25%** of the current speedrun FLOP budget in a 24h window.

Implication:
- We should not expect “same d26 recipe, just slower hardware” to work.
- We need a **different compute-optimal point** (model size, tokens, sequence curriculum, batch strategy).

## Core Strategy
1. Fix a realistic 24h FLOP envelope for 4090 using quick throughput probes.
2. Choose architecture on that envelope (not from 8xH100 defaults).
3. Allocate FLOPs across phases (pretrain + SFT) with pretraining dominant.
4. Use a staged context schedule to improve early token throughput.

## Candidate Architecture Bands (starting hypotheses)
These are not commitments; they are initial hypotheses to test quickly.

| Band | Goal | Candidate depth | Likely seq strategy | Why |
|---|---|---|---|---|
| A (safe) | guaranteed fit + high throughput | d12-d14 | mostly 512->1024 | Fastest iteration loop; lower risk of OOM/regression. |
| B (target) | best chance for strong quality/compute balance | d16-d18 | 512->1024->2048 (late) | Likely best tradeoff for 24h local budget. |
| C (stretch) | push capability ceiling | d20+ | 1024/2048 heavy | Might be too slow on 4090; use only if throughput surprises us. |

## Architecture Decisions To Evaluate Early
1. Keep `aspect-ratio=64` as baseline (current proven default), then only change if profiling says so.
2. Start with `window-pattern=L` for 4090 baseline runs.
3. Test `head-dim=128` baseline, then compare `head-dim=64` for speed/quality tradeoff.
4. Evaluate GQA (`n_kv_head < n_head`) as a memory/throughput lever, but only after baseline stability.
5. Treat embedding tying and similar param-saving tricks as secondary unless they clearly improve wall-clock quality.

## 24h FLOP Allocation Proposal (initial)
- 80% to 90%: base pretraining
- 10% to 20%: SFT + eval + safety margin

Rationale:
- Pretraining dominates capability.
- End-to-end target still needs chat behavior and minimal eval sanity checks.

## Measurement Plan (must happen before detailed tuning)
### M0: Throughput + VRAM map (single afternoon)
For depths `{12,14,16,18,20}` and seq `{512,1024,2048}`:
- Record `tok/sec`, `dt`, peak memory, gradient-accum steps, stability.
- Run short fixed-step jobs (e.g. 100-300 steps) with eval/sampling mostly disabled.

Output:
- A table converting measured throughput to estimated 24h token capacity for each config.

### M1: Pick one “target band”
- Select A/B/C based on observed throughput and memory headroom.
- Lock one primary config and one fallback config.

### M2: Pilot run for curve shape
- Run a medium pilot (e.g. 2-6h) and inspect loss/val_bpb trajectory.
- Extrapolate whether final 24h run can reach desired quality zone.

### M3: First full 24h recipe
- Execute full pretrain+SFT budget with tight logging.
- Use results to re-center architecture band (up/down).

## What we should decide together now
1. Are we okay using `window-pattern=L` as the default 4090 baseline assumption?
2. Do we prioritize a stronger first success (Band A/B) over aggressive capability stretch (Band C)?
3. Do we treat 24h as strict hard cap for v1, or allow one “quality-first” >24h reference run for calibration?

## Success Signal For This Target
We have a win for Target #1 when we can say:
- “Given 1x4090 and 24h, this is our best architecture band and batch/sequence schedule,”
- and that choice is backed by measured throughput + memory + early quality trajectory (not guesswork).
