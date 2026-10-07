#!/usr/bin/env bash
set -euo pipefail

# Submit data on GPU 0 and signal/background sequentially on GPU 3.

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
RESULTS=${RESULTS:-${ROOT}/results}
TMP_DIR=${TMP_DIR:-${RESULTS}/tmp/event_tokens_grouped_cls_q4_final_new_mcdata}
DRY_RUN=${DRY_RUN:-0}
OVERWRITE=${OVERWRITE:-0}
DATA_GPU=${DATA_GPU:-0}
SIGNAL_BACKGROUND_GPU=${SIGNAL_BACKGROUND_GPU:-3}
STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)}

mkdir -p "$TMP_DIR"
cd "$ROOT"

write_worker () {
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
    "exec bash ${ROOT}/scripts/tokenize_grouped_cls_q4_final_new_mcdata.sh" \
    > "$worker"
  chmod +x "$worker"
}

submit_service () {
  local unit=$1
  local description=$2
  local worker=$3
  local gpu=$4
  local samples=$5

  write_worker "$worker" "$gpu" "$samples"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY RUN: would submit ${unit}.service"
    echo "  GPU: ${gpu}"
    echo "  samples: ${samples}"
    echo "  worker: ${worker}"
    return
  fi

  systemd-run \
    --unit="$unit" \
    --description="$description" \
    --collect \
    --property=WorkingDirectory="$ROOT" \
    /bin/bash "$worker"

  echo "Submitted ${unit}.service"
  echo "Follow: journalctl -u ${unit}.service -f"
}

DATA_UNIT="atlas-tokenize-grouped-cls-q4-data-${STAMP}"
SB_UNIT="atlas-tokenize-grouped-cls-q4-signal-background-${STAMP}"

echo "Grouped CLS q4 full parquet export"
echo "  data:              GPU ${DATA_GPU}"
echo "  signal/background: GPU ${SIGNAL_BACKGROUND_GPU}, sequential"
echo "  selections:        identical to completed q8 production parquets"
echo "  output:            ${RESULTS}/event_tokens_grouped_cls_q4_final_new_mcdata"
echo "  existing outputs:  skipped unless OVERWRITE=1 is exported"
echo

submit_service \
  "$DATA_UNIT" \
  "Grouped CLS q4 real-data parquet export on GPU ${DATA_GPU}" \
  "${TMP_DIR}/${DATA_UNIT}.sh" \
  "$DATA_GPU" \
  "data"

echo

submit_service \
  "$SB_UNIT" \
  "Grouped CLS q4 signal/background parquet export on GPU ${SIGNAL_BACKGROUND_GPU}" \
  "${TMP_DIR}/${SB_UNIT}.sh" \
  "$SIGNAL_BACKGROUND_GPU" \
  "signal,background"
