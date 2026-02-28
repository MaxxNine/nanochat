#!/bin/bash
set -euo pipefail

# Compare FP8 backends directly under the same training configuration.
#
# Cases:
#   1) FP8 custom backend
#   2) FP8 torchao backend
# Optional:
#   3) baseline bf16 (set INCLUDE_BASELINE=1)
#
# Usage:
#   bash runs/fp8_backend_compare.sh

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${RESULTS_DIR:-$NANOCHAT_BASE_DIR/fp8_backend_compare/$TIMESTAMP}"
mkdir -p "$RESULTS_DIR"
SUMMARY_CSV="$RESULTS_DIR/summary.csv"

RUN_PREFIX="${RUN_PREFIX:-fp8_backend_compare_${TIMESTAMP}}"
MODEL_TAG_PREFIX="${MODEL_TAG_PREFIX:-fp8_backend_compare_${TIMESTAMP}}"
WANDB_RUN_COMPARE="${WANDB_RUN_COMPARE:-dummy}"
INCLUDE_BASELINE="${INCLUDE_BASELINE:-0}"

# Shared training knobs (forwarded to speedrun_small.sh)
DEPTH="${DEPTH:-12}"
WINDOW_PATTERN="${WINDOW_PATTERN:-L}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-2}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-32768}"
NUM_ITERATIONS="${NUM_ITERATIONS:-200}"
TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:--1}"
TARGET_FLOPS="${TARGET_FLOPS:--1}"
EVAL_EVERY="${EVAL_EVERY:-50}"
EVAL_TOKENS="${EVAL_TOKENS:-65536}"
CORE_METRIC_EVERY="${CORE_METRIC_EVERY:--1}"
CORE_METRIC_MAX_PER_TASK="${CORE_METRIC_MAX_PER_TASK:-128}"
SAMPLE_EVERY="${SAMPLE_EVERY:--1}"
SAVE_EVERY="${SAVE_EVERY:--1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
PREPARE_DATA="${PREPARE_DATA:-auto}"
SKIP_SETUP="${SKIP_SETUP:-0}"

# Keep defaults stability-first and equivalent across backends where possible.
COMMON_FP8_ARGS_DEFAULT="--warmup-ratio=0.03 --matrix-lr=0.005 --scalar-lr=0.05 --embedding-lr=0.15 --unembedding-lr=0.002 --fp8-skip-lm-head --fp8-skip-attn-qk --fp8-min-dim=256"
COMMON_FP8_ARGS="${COMMON_FP8_ARGS:-$COMMON_FP8_ARGS_DEFAULT}"
CUSTOM_ONLY_ARGS="${CUSTOM_ONLY_ARGS:---fp8-no-allow-in-graph}"
TORCHAO_ONLY_ARGS="${TORCHAO_ONLY_ARGS:-}"

BASELINE_EXTRA_ARGS="${BASELINE_EXTRA_ARGS:-}"

echo "variant,use_fp8,fp8_backend,fp8_recipe,exit_code,status,elapsed_sec,max_tok_per_sec,avg_tok_per_sec,final_loss,min_val_bpb,peak_vram_mib,fp8_layers,log_file" > "$SUMMARY_CSV"

extract_max_tok_per_sec() {
    local log_file="$1"
    if [ ! -f "$log_file" ]; then
        echo "NA"
        return
    fi
    awk '
    {
        if (match($0, /tok\/sec: [0-9,]+/)) {
            value = substr($0, RSTART + 9, RLENGTH - 9)
            gsub(",", "", value)
            if (!found || value + 0 > max) {
                max = value + 0
                found = 1
            }
        }
    }
    END {
        if (found) print int(max)
        else print "NA"
    }' "$log_file"
}

extract_avg_tok_per_sec() {
    local log_file="$1"
    if [ ! -f "$log_file" ]; then
        echo "NA"
        return
    fi
    awk '
    {
        if (match($0, /tok\/sec: [0-9,]+/)) {
            value = substr($0, RSTART + 9, RLENGTH - 9)
            gsub(",", "", value)
            sum += value + 0
            n += 1
        }
    }
    END {
        if (n == 0) print "NA"
        else printf("%.2f", sum / n)
    }' "$log_file"
}

extract_final_loss() {
    local log_file="$1"
    local value
    if [ ! -f "$log_file" ]; then
        echo "NA"
        return
    fi
    value="$(grep -E 'step [0-9]+/[0-9]+' "$log_file" | tail -1 | sed -E 's/.*loss: ([0-9.]+).*/\1/' || true)"
    if [ -z "$value" ]; then
        echo "NA"
    else
        echo "$value"
    fi
}

extract_min_val_bpb() {
    local log_file="$1"
    local value
    if [ ! -f "$log_file" ]; then
        echo "NA"
        return
    fi
    value="$(grep -E 'Minimum validation bpb:' "$log_file" | tail -1 | sed -E 's/.*: ([0-9.]+).*/\1/' || true)"
    if [ -z "$value" ]; then
        echo "NA"
    else
        echo "$value"
    fi
}

extract_peak_vram_mib() {
    local log_file="$1"
    local value
    if [ ! -f "$log_file" ]; then
        echo "NA"
        return
    fi
    value="$(grep -E 'Peak memory usage:' "$log_file" | tail -1 | sed -E 's/.*: ([0-9.]+)MiB.*/\1/' || true)"
    if [ -z "$value" ]; then
        echo "NA"
    else
        echo "$value"
    fi
}

extract_fp8_layers() {
    local log_file="$1"
    local value
    if [ ! -f "$log_file" ]; then
        echo "NA"
        return
    fi
    value="$(grep -E 'FP8 training enabled' "$log_file" | tail -1 | sed -E 's/.*converted ([0-9]+\/[0-9]+).*/\1/' || true)"
    if [ -z "$value" ]; then
        echo "NA"
    else
        echo "$value"
    fi
}

run_case() {
    local variant="$1"
    local use_fp8="$2"
    local fp8_backend="$3"
    local fp8_recipe="$4"
    local extra_args="$5"
    local log_file="$RESULTS_DIR/${variant}.log"
    local start_ts end_ts elapsed_sec exit_code status
    local max_tok avg_tok final_loss min_val_bpb peak_vram fp8_layers case_run_name

    if [ "$WANDB_RUN_COMPARE" = "dummy" ]; then
        case_run_name="dummy"
    else
        case_run_name="${RUN_PREFIX}_${variant}"
    fi

    echo "=== running case: $variant (run=$case_run_name fp8=$use_fp8 backend=$fp8_backend recipe=$fp8_recipe extra_args='${extra_args:-<none>}') ==="

    start_ts="$(date +%s)"
    set +e
    RUN_NAME="$case_run_name" \
    MODEL_TAG="${MODEL_TAG_PREFIX}_${variant}" \
    NPROC_PER_NODE="$NPROC_PER_NODE" \
    DEPTH="$DEPTH" \
    WINDOW_PATTERN="$WINDOW_PATTERN" \
    MAX_SEQ_LEN="$MAX_SEQ_LEN" \
    DEVICE_BATCH_SIZE="$DEVICE_BATCH_SIZE" \
    TOTAL_BATCH_SIZE="$TOTAL_BATCH_SIZE" \
    NUM_ITERATIONS="$NUM_ITERATIONS" \
    TARGET_PARAM_DATA_RATIO="$TARGET_PARAM_DATA_RATIO" \
    TARGET_FLOPS="$TARGET_FLOPS" \
    EVAL_EVERY="$EVAL_EVERY" \
    EVAL_TOKENS="$EVAL_TOKENS" \
    CORE_METRIC_EVERY="$CORE_METRIC_EVERY" \
    CORE_METRIC_MAX_PER_TASK="$CORE_METRIC_MAX_PER_TASK" \
    SAMPLE_EVERY="$SAMPLE_EVERY" \
    SAVE_EVERY="$SAVE_EVERY" \
    PREPARE_DATA="$PREPARE_DATA" \
    SKIP_SETUP="$SKIP_SETUP" \
    USE_FP8="$use_fp8" \
    FP8_BACKEND="$fp8_backend" \
    FP8_RECIPE="$fp8_recipe" \
    BASE_TRAIN_EXTRA_ARGS="$extra_args" \
    LOG_FILE="$log_file" \
    bash "$SCRIPT_DIR/speedrun_small.sh"
    exit_code=$?
    set -e
    end_ts="$(date +%s)"
    elapsed_sec="$((end_ts - start_ts))"

    if [ "$exit_code" -eq 0 ]; then
        status="ok"
    else
        status="failed"
    fi

    max_tok="$(extract_max_tok_per_sec "$log_file")"
    avg_tok="$(extract_avg_tok_per_sec "$log_file")"
    final_loss="$(extract_final_loss "$log_file")"
    min_val_bpb="$(extract_min_val_bpb "$log_file")"
    peak_vram="$(extract_peak_vram_mib "$log_file")"
    fp8_layers="$(extract_fp8_layers "$log_file")"

    echo "$variant,$use_fp8,$fp8_backend,$fp8_recipe,$exit_code,$status,$elapsed_sec,$max_tok,$avg_tok,$final_loss,$min_val_bpb,$peak_vram,$fp8_layers,$log_file" >> "$SUMMARY_CSV"
}

# FP8 backends first for faster feedback
run_case "fp8_custom" "1" "custom" "tensorwise" "$COMMON_FP8_ARGS $CUSTOM_ONLY_ARGS"

# Setup/data already handled after first run
SKIP_SETUP=1
PREPARE_DATA=0

run_case "fp8_torchao" "1" "torchao" "tensorwise" "$COMMON_FP8_ARGS $TORCHAO_ONLY_ARGS"

if [ "$INCLUDE_BASELINE" = "1" ]; then
    run_case "baseline" "0" "NA" "NA" "$BASELINE_EXTRA_ARGS"
fi

echo
echo "=== FP8 backend comparison summary ==="
column -t -s',' "$SUMMARY_CSV"
echo
echo "CSV: $SUMMARY_CSV"
echo "Logs: $RESULTS_DIR"

custom_max="$(awk -F',' '$1=="fp8_custom"{print $8}' "$SUMMARY_CSV" | tail -1)"
torchao_max="$(awk -F',' '$1=="fp8_torchao"{print $8}' "$SUMMARY_CSV" | tail -1)"

if [[ "$custom_max" =~ ^[0-9]+(\.[0-9]+)?$ ]] && [[ "$torchao_max" =~ ^[0-9]+(\.[0-9]+)?$ ]] && [ "$torchao_max" != "0" ]; then
    speed_ratio="$(awk -v c="$custom_max" -v t="$torchao_max" 'BEGIN{printf "%.3f", c / t}')"
    echo "Max tok/sec ratio (custom / torchao): ${speed_ratio}x"
fi
