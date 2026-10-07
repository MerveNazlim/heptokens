#!/usr/bin/env bash
set -euo pipefail

cd "${REPO:-/home/magaras/heptok_fork/heptokens}"

OBJECTS="${OBJECTS:-jets electrons muons photons taus}"
RUN_SOURCE="${RUN_SOURCE:-stage1}"
OUTPUT_BASE="${OUTPUT_BASE:-results/google_stage1_scan_plots}"
CACHE_BASE="${CACHE_BASE:-${OUTPUT_BASE}}"
MC_DIR="${MC_DIR:-/home/zephyr/Data/viviana/bnl-treasure/data_new/h5}"
DATA_DIR="${DATA_DIR:-/home/zephyr/Data/viviana/bnl-treasure/data_new/h5/realdata}"
EXCLUDE_MC_PATTERN="${EXCLUDE_MC_PATTERN:-DAOD_PHYSLITE.370016*}"
N_FILES="${N_FILES:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LOG_Y="${LOG_Y:-0}"
PIXI="${PIXI:-/root/.pixi/bin/pixi}"

LOG_Y_ARG="--log-y"
if [[ "${LOG_Y}" == "0" ]]; then
  LOG_Y_ARG="--no-log-y"
fi

"${PIXI}" run --frozen python scripts/plot_stage1_pt_eta_phi_all_objects.py \
  --objects ${OBJECTS} \
  --run-source "${RUN_SOURCE}" \
  --output-base "${OUTPUT_BASE}" \
  --cache-base "${CACHE_BASE}" \
  --mc-dir "${MC_DIR}" \
  --data-dir "${DATA_DIR}" \
  --exclude-mc-pattern "${EXCLUDE_MC_PATTERN}" \
  --n-files "${N_FILES}" \
  --batch-size "${BATCH_SIZE}" \
  ${LOG_Y_ARG}
