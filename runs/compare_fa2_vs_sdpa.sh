#!/bin/bash
set -euo pipefail

# Compare training throughput between FA2 and SDPA backends on non-Hopper GPUs.
# This script runs speedrun_small twice (or more): one forced to SDPA, one forced to FA2.
#
# Examples:
#   bash runs/compare_fa2_vs_sdpa.sh
#   RUNS=3 NUM_ITERATIONS=300 WINDOW_PATTERN=L bash runs/compare_fa2_vs_sdpa.sh

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
RESULTS_DIR="${RESULTS_DIR:-$NANOCHAT_BASE_DIR/fa2_vs_sdpa/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RESULTS_DIR"

RUNS="${RUNS:-2}"                    # number of runs per backend
TAIL_STEPS="${TAIL_STEPS:-20}"       # average tok/sec over last N step logs

# Use fast, stable throughput settings by default (caller can override any of these).
NUM_ITERATIONS="${NUM_ITERATIONS:-200}"
EVAL_EVERY="${EVAL_EVERY:--1}"
CORE_METRIC_EVERY="${CORE_METRIC_EVERY:--1}"
SAMPLE_EVERY="${SAMPLE_EVERY:--1}"
SAVE_EVERY="${SAVE_EVERY:--1}"

SUMMARY_TSV="$RESULTS_DIR/summary.tsv"
echo -e "impl\trun\tavg_tok_sec" > "$SUMMARY_TSV"

if [ -x ".venv/bin/python" ]; then
    HAS_FA2="$(
        ./.venv/bin/python - <<'PY'
from nanochat.flash_attention import HAS_FA2
print("1" if HAS_FA2 else "0")
PY
    )"
    if [ "$HAS_FA2" != "1" ]; then
        echo "FlashAttention-2 is not available in the current environment (.venv)."
        echo "Install FA2 first, then rerun this comparison."
        exit 1
    fi
else
    echo "note: .venv not found yet; FA2 availability will be validated during the fa2 run."
fi

extract_avg_tok_sec() {
    local log_file="$1"
    local values
    values="$(rg -o 'tok/sec: [0-9,]+' "$log_file" | sed -E 's/[^0-9]//g' | tail -n "$TAIL_STEPS" || true)"
    if [ -z "$values" ]; then
        echo "nan"
        return
    fi
    echo "$values" | awk '{s+=$1;n+=1} END {if (n==0) print "nan"; else printf "%.2f", s/n}'
}

run_impl() {
    local impl="$1"
    local run_idx="$2"
    local log_file="$RESULTS_DIR/${impl}_run${run_idx}.log"
    echo "=== Running impl=$impl run=$run_idx/$RUNS ==="
    ATTN_IMPL="$impl" \
    ENABLE_FA2=1 \
    SKIP_SETUP=1 \
    NUM_ITERATIONS="$NUM_ITERATIONS" \
    EVAL_EVERY="$EVAL_EVERY" \
    CORE_METRIC_EVERY="$CORE_METRIC_EVERY" \
    SAMPLE_EVERY="$SAMPLE_EVERY" \
    SAVE_EVERY="$SAVE_EVERY" \
    LOG_FILE="$log_file" \
    bash runs/speedrun_small.sh

    local run_avg
    run_avg="$(extract_avg_tok_sec "$log_file")"
    echo -e "${impl}\t${run_idx}\t${run_avg}" >> "$SUMMARY_TSV"
    echo "avg tok/sec (last ${TAIL_STEPS} steps): ${run_avg}"
}

for run_idx in $(seq 1 "$RUNS"); do
    run_impl "sdpa" "$run_idx"
done

for run_idx in $(seq 1 "$RUNS"); do
    run_impl "fa2" "$run_idx"
done

sdpa_mean="$(awk -F'\t' '$1=="sdpa" && $3!="nan" {s+=$3;n+=1} END {if (n==0) print "nan"; else printf "%.2f", s/n}' "$SUMMARY_TSV")"
fa2_mean="$(awk -F'\t' '$1=="fa2" && $3!="nan" {s+=$3;n+=1} END {if (n==0) print "nan"; else printf "%.2f", s/n}' "$SUMMARY_TSV")"

echo
echo "=== FA2 vs SDPA summary ==="
echo "results_dir: $RESULTS_DIR"
echo "sdpa_mean_tok_sec: $sdpa_mean"
echo "fa2_mean_tok_sec:  $fa2_mean"
if [ "$sdpa_mean" != "nan" ] && [ "$fa2_mean" != "nan" ]; then
    awk -v sdpa="$sdpa_mean" -v fa2="$fa2_mean" 'BEGIN {printf "fa2_vs_sdpa_speedup: %.4fx\n", fa2/sdpa}'
fi
echo "per-run: $SUMMARY_TSV"
