#!/bin/bash
set -euo pipefail

# Compare block update schedules under the same training config:
#   1) full_update
#   2) dynamic_suffix
#
# Optional FP8 can be enabled for both cases together.
#
# Usage:
#   bash runs/dynamic_suffix_compare.sh
#   USE_FP8_COMPARE=1 FP8_BACKEND_COMPARE=custom bash runs/dynamic_suffix_compare.sh

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${RESULTS_DIR:-$NANOCHAT_BASE_DIR/dynamic_suffix_compare/$TIMESTAMP}"
mkdir -p "$RESULTS_DIR"
SUMMARY_CSV="$RESULTS_DIR/summary.csv"

RUN_PREFIX="${RUN_PREFIX:-dyn_suffix_compare_${TIMESTAMP}}"
MODEL_TAG_PREFIX="${MODEL_TAG_PREFIX:-dyn_suffix_compare_${TIMESTAMP}}"
WANDB_RUN_COMPARE="${WANDB_RUN_COMPARE:-dummy}"

# Shared training knobs
DEPTH="${DEPTH:-12}"
WINDOW_PATTERN="${WINDOW_PATTERN:-L}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-2}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-32768}"
NUM_ITERATIONS="${NUM_ITERATIONS:-200}"
FULL_NUM_ITERATIONS="${FULL_NUM_ITERATIONS:-$NUM_ITERATIONS}"
DYNAMIC_NUM_ITERATIONS="${DYNAMIC_NUM_ITERATIONS:-$NUM_ITERATIONS}"
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

# Optional FP8 for both schedule variants
USE_FP8_COMPARE="${USE_FP8_COMPARE:-0}"
FP8_BACKEND_COMPARE="${FP8_BACKEND_COMPARE:-custom}"
FP8_RECIPE_COMPARE="${FP8_RECIPE_COMPARE:-tensorwise}"

# Extra args per case (space-delimited)
FULL_EXTRA_ARGS="${FULL_EXTRA_ARGS:-}"
# Short-run tuned defaults so dynamic_suffix actually activates within ~200 steps.
DYNAMIC_EXTRA_ARGS="${DYNAMIC_EXTRA_ARGS:---dyn-warmup-steps=16 --dyn-probe-every=12 --dyn-refresh-every=32 --dyn-relevance-metric=grad_ratio --dyn-relevance-threshold=0.85 --dyn-min-active-layers=4 --dyn-max-active-layers=6 --dyn-freeze-start-frac=0.15}"

# For custom FP8, reuse the same stability-first preset as fp8_compare unless overridden.
FP8_STABLE_EXTRA_ARGS="${FP8_STABLE_EXTRA_ARGS:---warmup-ratio=0.03 --matrix-lr=0.005 --scalar-lr=0.05 --embedding-lr=0.15 --unembedding-lr=0.002 --fp8-skip-lm-head --fp8-skip-attn-qk --fp8-min-dim=256 --fp8-no-allow-in-graph}"
if [ "$USE_FP8_COMPARE" = "1" ] && [ "$FP8_BACKEND_COMPARE" = "custom" ]; then
    if [ -z "$FULL_EXTRA_ARGS" ]; then
        FULL_EXTRA_ARGS="$FP8_STABLE_EXTRA_ARGS"
    fi
    if [ -n "$DYNAMIC_EXTRA_ARGS" ]; then
        DYNAMIC_EXTRA_ARGS="$DYNAMIC_EXTRA_ARGS $FP8_STABLE_EXTRA_ARGS"
    else
        DYNAMIC_EXTRA_ARGS="$FP8_STABLE_EXTRA_ARGS"
    fi
fi

echo "variant,schedule,num_iterations,use_fp8,fp8_backend,fp8_recipe,exit_code,status,elapsed_sec,max_tok_per_sec,avg_tok_per_sec,final_loss,min_val_bpb,avg_vram_mib,avg_step_peak_vram_mib,peak_vram_mib,active_layers_mean,active_layers_range,fp8_layers,log_file" > "$SUMMARY_CSV"

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

extract_avg_vram_mib() {
    local log_file="$1"
    local value
    if [ ! -f "$log_file" ]; then
        echo "NA"
        return
    fi
    value="$(grep -E 'Average memory usage:' "$log_file" | tail -1 | sed -E 's/.*: ([0-9.]+)MiB.*/\1/' || true)"
    if [ -z "$value" ]; then
        echo "NA"
    else
        echo "$value"
    fi
}

extract_avg_step_peak_vram_mib() {
    local log_file="$1"
    local value
    if [ ! -f "$log_file" ]; then
        echo "NA"
        return
    fi
    value="$(grep -E 'Average step peak memory usage:' "$log_file" | tail -1 | sed -E 's/.*: ([0-9.]+)MiB.*/\1/' || true)"
    if [ -z "$value" ]; then
        echo "NA"
    else
        echo "$value"
    fi
}

extract_active_layers_mean() {
    local log_file="$1"
    local value
    if [ ! -f "$log_file" ]; then
        echo "NA"
        return
    fi
    value="$(grep -E 'Dynamic suffix summary:' "$log_file" | tail -1 | sed -E 's/.*active_layers_mean=([0-9.]+).*/\1/' || true)"
    if [ -z "$value" ]; then
        echo "NA"
    else
        echo "$value"
    fi
}

extract_active_layers_range() {
    local log_file="$1"
    local value
    if [ ! -f "$log_file" ]; then
        echo "NA"
        return
    fi
    value="$(grep -E 'Dynamic suffix summary:' "$log_file" | tail -1 | sed -E 's/.*active_layers_range=\[([0-9]+),([0-9]+)\].*/\1-\2/' || true)"
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
    local schedule="$2"
    local extra_args="$3"
    local case_iterations="$4"
    local log_file="$RESULTS_DIR/${variant}.log"
    local start_ts end_ts elapsed_sec exit_code status
    local max_tok avg_tok final_loss min_val_bpb avg_vram avg_step_peak_vram peak_vram active_mean active_range fp8_layers case_run_name

    if [ "$WANDB_RUN_COMPARE" = "dummy" ]; then
        case_run_name="dummy"
    else
        case_run_name="${RUN_PREFIX}_${variant}"
    fi

    echo "=== running case: $variant (schedule=$schedule fp8=$USE_FP8_COMPARE backend=$FP8_BACKEND_COMPARE recipe=$FP8_RECIPE_COMPARE extra_args='${extra_args:-<none>}') ==="

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
    NUM_ITERATIONS="$case_iterations" \
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
    USE_FP8="$USE_FP8_COMPARE" \
    FP8_BACKEND="$FP8_BACKEND_COMPARE" \
    FP8_RECIPE="$FP8_RECIPE_COMPARE" \
    BLOCK_UPDATE_SCHEDULE="$schedule" \
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
    avg_vram="$(extract_avg_vram_mib "$log_file")"
    avg_step_peak_vram="$(extract_avg_step_peak_vram_mib "$log_file")"
    peak_vram="$(extract_peak_vram_mib "$log_file")"
    active_mean="$(extract_active_layers_mean "$log_file")"
    active_range="$(extract_active_layers_range "$log_file")"
    fp8_layers="$(extract_fp8_layers "$log_file")"

    echo "$variant,$schedule,$case_iterations,$USE_FP8_COMPARE,$FP8_BACKEND_COMPARE,$FP8_RECIPE_COMPARE,$exit_code,$status,$elapsed_sec,$max_tok,$avg_tok,$final_loss,$min_val_bpb,$avg_vram,$avg_step_peak_vram,$peak_vram,$active_mean,$active_range,$fp8_layers,$log_file" >> "$SUMMARY_CSV"
}

# dynamic first, then full baseline
run_case "dynamic" "dynamic_suffix" "$DYNAMIC_EXTRA_ARGS" "$DYNAMIC_NUM_ITERATIONS"

# setup/data already handled by the first run
SKIP_SETUP=1
PREPARE_DATA=0

run_case "full" "full_update" "$FULL_EXTRA_ARGS" "$FULL_NUM_ITERATIONS"

echo
echo "=== Dynamic suffix comparison summary ==="
column -t -s',' "$SUMMARY_CSV"
echo
echo "CSV: $SUMMARY_CSV"
echo "Logs: $RESULTS_DIR"

full_max="$(awk -F',' '$1=="full"{print $10}' "$SUMMARY_CSV" | tail -1)"
dyn_max="$(awk -F',' '$1=="dynamic"{print $10}' "$SUMMARY_CSV" | tail -1)"
if [[ "$full_max" =~ ^[0-9]+(\.[0-9]+)?$ ]] && [[ "$dyn_max" =~ ^[0-9]+(\.[0-9]+)?$ ]] && [ "$full_max" != "0" ]; then
    speedup="$(awk -v f="$full_max" -v d="$dyn_max" 'BEGIN{printf "%.3f", d / f}')"
    echo "Max tok/sec speedup (dynamic / full): ${speedup}x"
fi
