FP8_EXTRA_ARGS="--fp8-skip-lm-head --fp8-skip-attn-qk --fp8-min-dim=256 --fp8-no-allow-in-graph" \
  BASELINE_EXTRA_ARGS="" \
  bash runs/fp8_compare.sh
