#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
GPU=${GPU:-0}
N_FILES=${N_FILES:-20}
MAX_VALID_OBJECTS=${MAX_VALID_OBJECTS:-1000000}
BATCH_SIZE=${BATCH_SIZE:-2048}
NUM_WORKERS=${NUM_WORKERS:-0}
FORCE=${FORCE:-0}

OUTPUT_DIR=${OUTPUT_DIR:-${ROOT}/results/atlas_electron_sampling_study/comparison}
LOG_DIR=${LOG_DIR:-${ROOT}/results/logs/atlas_electron_sampling_study}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/comparison.log}

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "${ROOT}/results/matplotlib_cache"
cd "$ROOT"

ARGS=(
  --mc-dir /home/zephyr/Data/viviana/bnl-treasure/data/h5
  --data-dir /home/zephyr/Data/viviana/bnl-treasure/data/h5/realdata
  --output-dir "$OUTPUT_DIR"
  --device cuda
  --n-files "$N_FILES"
  --max-valid-objects "$MAX_VALID_OBJECTS"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
)

if [ "$FORCE" -eq 1 ]; then
  ARGS+=(--force)
fi

echo "Comparing completed electron sampling runs"
echo "  GPU: $GPU"
echo "  files per sample: $N_FILES"
echo "  objects per sample/model: $MAX_VALID_OBJECTS"
echo "  output: $OUTPUT_DIR"

export MPLCONFIGDIR="${ROOT}/results/matplotlib_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES="$GPU" \
  /root/.pixi/bin/pixi run python scripts/compare_electron_sampling_study.py \
  "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"

echo
echo "Report: ${OUTPUT_DIR}/comparison_report.md"
echo "Plots:  ${OUTPUT_DIR}/*.png"
