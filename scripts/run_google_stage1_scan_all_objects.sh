#!/usr/bin/env bash
set -euo pipefail

cd "${REPO:-/home/magaras/heptok_fork/heptokens}"

OBJECTS="${OBJECTS:-jets electrons muons photons taus tracks}"
GOOGLE_BASE_ROOT="${GOOGLE_BASE_ROOT:-google_results}"
OUTPUT_BASE="${OUTPUT_BASE:-results/google_stage1_scan_plots}"
MC_DIR="${MC_DIR:-/home/zephyr/Data/viviana/bnl-treasure/data_new/h5}"
DATA_DIR="${DATA_DIR:-/home/zephyr/Data/viviana/bnl-treasure/data_new/h5/realdata}"
EXCLUDE_MC_PATTERN="${EXCLUDE_MC_PATTERN:-DAOD_PHYSLITE.370016*}"
N_FILES="${N_FILES:-20}"
DEVICE="${DEVICE:-cuda}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-last.ckpt}"
MAX_VALID_OBJECTS="${MAX_VALID_OBJECTS:-1000000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LINE_FEATURES="${LINE_FEATURES:-auto}"
LOG_Y="${LOG_Y:-1}"
CACHE_ONLY="${CACHE_ONLY:-0}"
FORCE="${FORCE:-0}"
SKIP_MISSING_RUNS="${SKIP_MISSING_RUNS:-0}"
INCLUDE_FULL_REFERENCE="${INCLUDE_FULL_REFERENCE:-0}"
SCAN_SET="${SCAN_SET:-original}"
PIXI="${PIXI:-/root/.pixi/bin/pixi}"

if [[ "${INCLUDE_FULL_REFERENCE}" == "1" ]]; then
  FULL_REFERENCE_ARGS=(--full-reference-run auto --full-reference-label auto)
else
  FULL_REFERENCE_ARGS=(--full-reference-run none)
fi

COMMON_ARGS=(
  --mc-dir "${MC_DIR}"
  --data-dir "${DATA_DIR}"
  --exclude-mc-pattern "${EXCLUDE_MC_PATTERN}"
  --n-files "${N_FILES}"
  --device "${DEVICE}"
  --checkpoint-name "${CHECKPOINT_NAME}"
  --max-valid-objects "${MAX_VALID_OBJECTS}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --line-features "${LINE_FEATURES}"
)

if [[ "${LOG_Y}" == "1" ]]; then
  COMMON_ARGS+=(--log-y)
else
  COMMON_ARGS+=(--no-log-y)
fi

if [[ "${FORCE}" == "1" ]]; then
  COMMON_ARGS+=(--force)
fi

if [[ "${CACHE_ONLY}" == "1" ]]; then
  COMMON_ARGS+=(--cache-only)
fi

if [[ "${SKIP_MISSING_RUNS}" == "1" ]]; then
  COMMON_ARGS+=(--skip-missing-runs)
fi

echo "Google stage-1 scan plotting"
echo "  objects: ${OBJECTS}"
echo "  MC:      ${MC_DIR}"
echo "  data:    ${DATA_DIR}"
echo "  exclude: ${EXCLUDE_MC_PATTERN}"
echo "  n files: ${N_FILES}"
echo "  line features: ${LINE_FEATURES}"
echo "  log y: ${LOG_Y}"
echo "  cache only: ${CACHE_ONLY}"
echo "  scan set: ${SCAN_SET}"
echo "  output:  ${OUTPUT_BASE}"
echo

for OBJECT in ${OBJECTS}; do
  echo "Plotting ${OBJECT}"
  "${PIXI}" run --frozen python scripts/compare_object_google_stage1_scan.py \
    --object "${OBJECT}" \
    --google-base "${GOOGLE_BASE_ROOT}/${OBJECT}" \
    --output-dir "${OUTPUT_BASE}/${OBJECT}" \
    --scan-set "${SCAN_SET}" \
    "${FULL_REFERENCE_ARGS[@]}" \
    "${COMMON_ARGS[@]}"
done

echo
echo "Done. Outputs are under ${OUTPUT_BASE}/<object>/"
