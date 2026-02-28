#!/bin/bash
set -euo pipefail

# Small, fast pretraining entrypoint for local iteration (e.g. single 4090).
# Feature flags are exposed via env vars so experiments stay reproducible.
#
# Example baseline:
#   bash runs/speedrun_small.sh
#
# Example FP8 (custom backend, no wandb):
#   USE_FP8=1 FP8_BACKEND=custom bash runs/speedrun_small.sh
#
# Optional wandb:
#   RUN_NAME=my_run bash runs/speedrun_small.sh

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

SKIP_SETUP="${SKIP_SETUP:-0}"
PREPARE_DATA="${PREPARE_DATA:-auto}"  # auto|0|1

# Setup
if [ "$SKIP_SETUP" != "1" ]; then
    command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -d ".venv" ] || uv venv
    uv sync --extra gpu
fi
source .venv/bin/activate

# Optional report reset
if [ "${RESET_REPORT:-0}" = "1" ]; then
    python -m nanochat.report reset
fi

# Optional data/tokenizer bootstrap (only if missing unless PREPARE_DATA=1)
DATASET_SHARDS_MIN="${DATASET_SHARDS_MIN:-8}"
TOKENIZER_MAX_CHARS="${TOKENIZER_MAX_CHARS:-250000000}"
TOKENIZER_VOCAB_SIZE="${TOKENIZER_VOCAB_SIZE:-32768}"

DATA_DIR="$NANOCHAT_BASE_DIR/base_data"
TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer"
mkdir -p "$DATA_DIR"

shard_count="$(find "$DATA_DIR" -maxdepth 1 -name 'shard_*.parquet' 2>/dev/null | wc -l | tr -d ' ')"
tokenizer_ready=0
if [ -f "$TOKENIZER_DIR/tokenizer.pkl" ] && [ -f "$TOKENIZER_DIR/token_bytes.pt" ]; then
    tokenizer_ready=1
fi

need_prepare=0
case "$PREPARE_DATA" in
    1|true|TRUE|yes|YES)
        need_prepare=1
        ;;
    auto|AUTO)
        if [ "$shard_count" -lt "$DATASET_SHARDS_MIN" ] || [ "$tokenizer_ready" -ne 1 ]; then
            need_prepare=1
        fi
        ;;
    0|false|FALSE|no|NO)
        need_prepare=0
        ;;
    *)
        echo "Invalid PREPARE_DATA value: $PREPARE_DATA (use auto|0|1)"
        exit 1
        ;;
esac

if [ "$need_prepare" -eq 1 ]; then
    if [ "$shard_count" -lt "$DATASET_SHARDS_MIN" ]; then
        echo "Preparing dataset shards: have $shard_count, need at least $DATASET_SHARDS_MIN"
        python -m nanochat.dataset -n "$DATASET_SHARDS_MIN"
    fi
    if [ "$tokenizer_ready" -ne 1 ]; then
        echo "Tokenizer missing, training a local tokenizer"
        python -m scripts.tok_train --max-chars="$TOKENIZER_MAX_CHARS" --vocab-size="$TOKENIZER_VOCAB_SIZE"
    fi
fi

# Training config (defaults tuned for quick local iteration)
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
RUN_NAME="${RUN_NAME:-dummy}"          # "dummy" disables wandb logging
MODEL_TAG="${MODEL_TAG:-small_baseline}"
DEPTH="${DEPTH:-18}"
WINDOW_PATTERN="${WINDOW_PATTERN:-L}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
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

# FP8 feature flags
USE_FP8="${USE_FP8:-0}"
FP8_BACKEND="${FP8_BACKEND:-custom}"      # custom|torchao
FP8_RECIPE="${FP8_RECIPE:-tensorwise}"    # tensorwise|rowwise (rowwise requires torchao)

BASE_TRAIN_ARGS=(
    "--depth=$DEPTH"
    "--window-pattern=$WINDOW_PATTERN"
    "--max-seq-len=$MAX_SEQ_LEN"
    "--device-batch-size=$DEVICE_BATCH_SIZE"
    "--total-batch-size=$TOTAL_BATCH_SIZE"
    "--num-iterations=$NUM_ITERATIONS"
    "--target-param-data-ratio=$TARGET_PARAM_DATA_RATIO"
    "--target-flops=$TARGET_FLOPS"
    "--eval-every=$EVAL_EVERY"
    "--eval-tokens=$EVAL_TOKENS"
    "--core-metric-every=$CORE_METRIC_EVERY"
    "--core-metric-max-per-task=$CORE_METRIC_MAX_PER_TASK"
    "--sample-every=$SAMPLE_EVERY"
    "--save-every=$SAVE_EVERY"
    "--run=$RUN_NAME"
    "--model-tag=$MODEL_TAG"
)

if [ "$USE_FP8" = "1" ]; then
    BASE_TRAIN_ARGS+=(
        "--fp8"
        "--fp8-backend=$FP8_BACKEND"
        "--fp8-recipe=$FP8_RECIPE"
    )
fi

# Optional extra args passthrough (space-delimited)
if [ -n "${BASE_TRAIN_EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS_ARRAY=($BASE_TRAIN_EXTRA_ARGS)
    BASE_TRAIN_ARGS+=("${EXTRA_ARGS_ARRAY[@]}")
fi

echo "=== speedrun_small config ==="
echo "run=$RUN_NAME model_tag=$MODEL_TAG nproc=$NPROC_PER_NODE depth=$DEPTH seq=$MAX_SEQ_LEN dbs=$DEVICE_BATCH_SIZE tbs=$TOTAL_BATCH_SIZE steps=$NUM_ITERATIONS fp8=$USE_FP8 backend=$FP8_BACKEND recipe=$FP8_RECIPE"

TORCHRUN_CMD=(
    torchrun
    --standalone
    --nproc_per_node="$NPROC_PER_NODE"
    -m scripts.base_train
    --
    "${BASE_TRAIN_ARGS[@]}"
)

if [ -n "${LOG_FILE:-}" ]; then
    mkdir -p "$(dirname "$LOG_FILE")"
    "${TORCHRUN_CMD[@]}" 2>&1 | tee "$LOG_FILE"
else
    "${TORCHRUN_CMD[@]}"
fi
