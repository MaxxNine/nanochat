#!/bin/bash
set -euo pipefail

# ResFormer VE-gate microbenchmark runner.
# Uses attempts/container_b_effective_flops/resformer_ve_gate_bench.py
#
# Example baseline:
#   bash runs/resformer_ve_gate_bench.sh
#
# Example CUDA BF16 with compile candidates:
#   DEVICE=cuda DTYPE=bfloat16 COMPILE_CANDIDATES=1 bash runs/resformer_ve_gate_bench.sh
#
# Example fast smoke test:
#   SKIP_SETUP=1 DEVICE=cpu DTYPE=float32 BATCH=1 SEQ_LEN=64 N_EMBD=128 N_KV_HEAD=4 HEAD_DIM=32 VE_GATE_CHANNELS=16 WARMUP=2 ITERS=5 bash runs/resformer_ve_gate_bench.sh

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

SKIP_SETUP="${SKIP_SETUP:-0}"
if [ "$SKIP_SETUP" != "1" ]; then
    command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -d ".venv" ] || uv venv
    uv sync --extra gpu --inexact
fi
source .venv/bin/activate

# Benchmark config
DEVICE="${DEVICE:-auto}"                   # auto|cuda|cpu
DTYPE="${DTYPE:-bfloat16}"                 # float32|float16|bfloat16
MODE="${MODE:-both}"                       # forward|backward|both
COMPILE_CANDIDATES="${COMPILE_CANDIDATES:-1}"  # 0|1

BATCH="${BATCH:-2}"
SEQ_LEN="${SEQ_LEN:-2048}"
N_EMBD="${N_EMBD:-1536}"
N_KV_HEAD="${N_KV_HEAD:-12}"
HEAD_DIM="${HEAD_DIM:-128}"
VE_GATE_CHANNELS="${VE_GATE_CHANNELS:-32}"

WARMUP="${WARMUP:-20}"
ITERS="${ITERS:-80}"
SEED="${SEED:-42}"

RESULTS_DIR="${RESULTS_DIR:-$NANOCHAT_BASE_DIR/bench_results/resformer_ve_gate}"
mkdir -p "$RESULTS_DIR"
DEFAULT_JSON_OUT="$RESULTS_DIR/resformer_ve_gate_$(date +%Y%m%d_%H%M%S).json"
JSON_OUT="${JSON_OUT:-$DEFAULT_JSON_OUT}"

CMD=(
    python attempts/container_b_effective_flops/resformer_ve_gate_bench.py
    "--device=$DEVICE"
    "--dtype=$DTYPE"
    "--seed=$SEED"
    "--batch=$BATCH"
    "--seq-len=$SEQ_LEN"
    "--n-embd=$N_EMBD"
    "--n-kv-head=$N_KV_HEAD"
    "--head-dim=$HEAD_DIM"
    "--ve-gate-channels=$VE_GATE_CHANNELS"
    "--mode=$MODE"
    "--warmup=$WARMUP"
    "--iters=$ITERS"
)

if [ "$COMPILE_CANDIDATES" = "1" ]; then
    CMD+=("--compile-candidates")
fi

if [ -n "$JSON_OUT" ]; then
    CMD+=("--json-out=$JSON_OUT")
fi

# Optional extra args passthrough (space-delimited)
if [ -n "${BENCH_EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS_ARRAY=($BENCH_EXTRA_ARGS)
    CMD+=("${EXTRA_ARGS_ARRAY[@]}")
fi

echo "=== resformer_ve_gate_bench config ==="
echo "device=$DEVICE dtype=$DTYPE mode=$MODE compile_candidates=$COMPILE_CANDIDATES"
echo "shape: batch=$BATCH seq_len=$SEQ_LEN n_embd=$N_EMBD n_kv_head=$N_KV_HEAD head_dim=$HEAD_DIM ve_gate_channels=$VE_GATE_CHANNELS"
echo "timing: warmup=$WARMUP iters=$ITERS seed=$SEED"
echo "json_out=${JSON_OUT:-none}"

if [ -n "${LOG_FILE:-}" ]; then
    mkdir -p "$(dirname "$LOG_FILE")"
    "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
else
    "${CMD[@]}"
fi

