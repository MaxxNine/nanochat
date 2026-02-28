#!/bin/bash
set -euo pipefail

# FP8 NaN debug runner: executes only a couple of training steps and enables
# detailed non-finite checks to identify where values first blow up.
#
# Usage:
#   bash runs/fpi_debug.sh
#
# Optional:
#   FP8_BACKEND=torchao bash runs/fpi_debug.sh

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
DEPTH="${DEPTH:-12}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-2}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-32768}"
NUM_ITERATIONS="${NUM_ITERATIONS:-2}"
WINDOW_PATTERN="${WINDOW_PATTERN:-L}"

USE_FP8="${USE_FP8:-1}"
FP8_BACKEND="${FP8_BACKEND:-custom}"
FP8_RECIPE="${FP8_RECIPE:-tensorwise}"

# Keep this run local-only and deterministic-ish for debugging.
RUN_NAME="${RUN_NAME:-dummy}"
MODEL_TAG="${MODEL_TAG:-fpi_debug_${FP8_BACKEND}}"

# Aggressive debug flags with a safer default FP8 filter for custom backend.
# Keep anomaly detection off by default; explicit non-finite checks are the
# most actionable signal.
DEBUG_ARGS="${DEBUG_ARGS:---debug-nan --debug-max-nonfinite=16 --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --save-every=-1 --fp8-skip-lm-head --fp8-skip-attn-qk --fp8-min-dim=256 --fp8-no-allow-in-graph --fp8-log-modules}"

echo "=== fpi_debug ==="
echo "fp8=$USE_FP8 backend=$FP8_BACKEND recipe=$FP8_RECIPE depth=$DEPTH seq=$MAX_SEQ_LEN dbs=$DEVICE_BATCH_SIZE tbs=$TOTAL_BATCH_SIZE steps=$NUM_ITERATIONS"
echo "debug_args: $DEBUG_ARGS"

RUN_NAME="$RUN_NAME" \
MODEL_TAG="$MODEL_TAG" \
NPROC_PER_NODE="$NPROC_PER_NODE" \
DEPTH="$DEPTH" \
WINDOW_PATTERN="$WINDOW_PATTERN" \
MAX_SEQ_LEN="$MAX_SEQ_LEN" \
DEVICE_BATCH_SIZE="$DEVICE_BATCH_SIZE" \
TOTAL_BATCH_SIZE="$TOTAL_BATCH_SIZE" \
NUM_ITERATIONS="$NUM_ITERATIONS" \
USE_FP8="$USE_FP8" \
FP8_BACKEND="$FP8_BACKEND" \
FP8_RECIPE="$FP8_RECIPE" \
BASE_TRAIN_EXTRA_ARGS="$DEBUG_ARGS" \
bash runs/speedrun_small.sh
