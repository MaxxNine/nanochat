#!/bin/bash
set -euo pipefail

# Profile CUDA memory allocations using torch.cuda.memory._record_memory_history().
# Generates a JSON snapshot that can be visualized at https://pytorch.org/memory_viz
#
# Usage:
#   bash runs/mem_profile.sh
#   # Then open the .json.gz file at https://pytorch.org/memory_viz

PROFILE_STEPS="${PROFILE_STEPS:-15}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-.}"
SNAPSHOT_FILE="${SNAPSHOT_DIR}/cuda_mem_snapshot.json.gz"

# Run a short training with memory profiling enabled via env var
export MEM_PROFILE_STEPS="$PROFILE_STEPS"
export MEM_PROFILE_SNAPSHOT="$SNAPSHOT_FILE"

# Use existing speedrun with minimal steps and memory debug
NUM_ITERATIONS="$PROFILE_STEPS" \
DEBUG_MEM_EVERY=1 \
EVAL_EVERY=0 \
SAVE_EVERY=-1 \
SAMPLE_EVERY=-1 \
CORE_METRIC_EVERY=-1 \
    bash runs/speedrun_small.sh

echo ""
echo "=== Memory profile complete ==="
echo "Snapshot: $SNAPSHOT_FILE"
echo "Visualize at: https://pytorch.org/memory_viz"
echo "  1. Open the URL above"
echo "  2. Drag and drop $SNAPSHOT_FILE into the page"
