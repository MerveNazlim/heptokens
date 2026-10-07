#!/usr/bin/env bash
set -euo pipefail

# Submit two transient services: signal/background sequentially on GPU 0,
# and real data independently on GPU 3.

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
RESULTS=${RESULTS:-${ROOT}/results}
TMP_DIR=${TMP_DIR:-${RESULTS}/tmp/event_tokens_grouped_cls_final_new_mcdata}
DRY_RUN=${DRY_RUN:-0}
OVERWRITE=${OVERWRITE:-0}
SIGNAL_BACKGROUND_GPU=${SIGNAL_BACKGROUND_GPU:-0}
DATA_GPU=${DATA_GPU:-3}
STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)}
SMOKE_TEST=${SMOKE_TEST:-0}
NUM_EVENTS=${NUM_EVENTS:-all}
MAX_SIGNAL_FILES=${MAX_SIGNAL_FILES:-all}
MAX_BACKGROUND_FILES=${MAX_BACKGROUND_FILES:-all}
MAX_DATA_FILES=${MAX_DATA_FILES:-all}

if [ "$SMOKE_TEST" -eq 1 ]; then
  NUM_EVENTS=${SMOKE_NUM_EVENTS:-2000}
  MAX_SIGNAL_FILES=${SMOKE_MAX_SIGNAL_FILES:-1}
  MAX_BACKGROUND_FILES=${SMOKE_MAX_BACKGROUND_FILES:-1}
  MAX_DATA_FILES=${SMOKE_MAX_DATA_FILES:-1}
  OUT_DIR=${OUT_DIR:-${RESULTS}/smoke_event_tokens_grouped_cls_final_new_mcdata/${STAMP}}
  LOGDIR=${LOGDIR:-${RESULTS}/logs/smoke_event_tokens_grouped_cls_final_new_mcdata/${STAMP}}
else
  OUT_DIR=${OUT_DIR:-${RESULTS}/event_tokens_grouped_cls_final_new_mcdata}
  LOGDIR=${LOGDIR:-${RESULTS}/logs/event_tokens_grouped_cls_final_new_mcdata}
fi

mkdir -p "$TMP_DIR"
cd "$ROOT"

write_worker () {
  local worker=$1
  local gpu=$2
  local samples=$3

  apply_worker_template "$worker" "$gpu" "$samples"
  chmod +x "$worker"
}

apply_worker_template () {
  local worker=$1
  local gpu=$2
  local samples=$3

  printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -euo pipefail' \
    "export GPU=${gpu}" \
    "export SIGNAL_GPU=${gpu}" \
    "export BACKGROUND_GPU=${gpu}" \
    "export DATA_GPU=${gpu}" \
    "export SAMPLES=${samples}" \
    "export OVERWRITE=${OVERWRITE}" \
    "export NUM_EVENTS=${NUM_EVENTS}" \
    "export MAX_SIGNAL_FILES=${MAX_SIGNAL_FILES}" \
    "export MAX_BACKGROUND_FILES=${MAX_BACKGROUND_FILES}" \
    "export MAX_DATA_FILES=${MAX_DATA_FILES}" \
    "export OUT_DIR=${OUT_DIR}" \
    "export LOGDIR=${LOGDIR}" \
    "exec bash ${ROOT}/scripts/tokenize_grouped_final_new_mcdata.sh" \
    > "$worker"
}

submit_service () {
  local unit=$1
  local description=$2
  local worker=$3
  local gpu=$4
  local samples=$5

  write_worker "$worker" "$gpu" "$samples"

  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY RUN: would submit ${unit}"
    echo "  GPU: ${gpu}"
    echo "  samples: ${samples}"
    echo "  worker: ${worker}"
    return 0
  fi

  systemd-run \
    --unit="$unit" \
    --description="$description" \
    --collect \
    --property=WorkingDirectory="$ROOT" \
    /bin/bash "$worker"

  echo "Submitted ${unit}"
  echo "  GPU: ${gpu}"
  echo "  samples: ${samples}"
  echo "  progress: journalctl -u ${unit}.service -f"
  echo "  worker: ${worker}"
}

if [ "$SMOKE_TEST" -eq 1 ]; then
  SB_UNIT="atlas-tokenize-grouped-cls-smoke-signal-background-${STAMP}"
  DATA_UNIT="atlas-tokenize-grouped-cls-smoke-data-${STAMP}"
else
  SB_UNIT="atlas-tokenize-grouped-cls-signal-background-${STAMP}"
  DATA_UNIT="atlas-tokenize-grouped-cls-data-${STAMP}"
fi

echo "Grouped full parquet export with CLS"
echo "  signal/background: GPU ${SIGNAL_BACKGROUND_GPU}, sequential"
echo "  data:              GPU ${DATA_GPU}"
echo "  events/file:       ${NUM_EVENTS}"
echo "  files:             signal=${MAX_SIGNAL_FILES} background=${MAX_BACKGROUND_FILES} data=${MAX_DATA_FILES}"
echo "  output:            ${OUT_DIR}"
echo "  existing outputs:  skipped unless OVERWRITE=1 is exported"
echo

submit_service \
  "$SB_UNIT" \
  "Grouped signal/background parquet export on GPU ${SIGNAL_BACKGROUND_GPU}" \
  "${TMP_DIR}/${SB_UNIT}.sh" \
  "$SIGNAL_BACKGROUND_GPU" \
  "signal,background"

echo

submit_service \
  "$DATA_UNIT" \
  "Grouped real-data parquet export on GPU ${DATA_GPU}" \
  "${TMP_DIR}/${DATA_UNIT}.sh" \
  "$DATA_GPU" \
  "data"
