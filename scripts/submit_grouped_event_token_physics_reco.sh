#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
RESULTS=${RESULTS:-${ROOT}/results}
MAX_EVENTS=${MAX_EVENTS:-50000}
DEVICE=${DEVICE:-cpu}
GPU=${GPU:-0}
DRY_RUN=${DRY_RUN:-0}
STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)}
UNIT=${UNIT:-atlas-grouped-event-physics-reco-${STAMP}}
OUTPUT_DIR=${OUTPUT_DIR:-${RESULTS}/grouped_event_token_physics_reco/${STAMP}}
TMP_DIR=${TMP_DIR:-${RESULTS}/tmp/grouped_event_token_physics_reco}
WORKER=${TMP_DIR}/${UNIT}.sh

mkdir -p "$TMP_DIR" "$OUTPUT_DIR"

printf '%s\n' \
  '#!/usr/bin/env bash' \
  'set -euo pipefail' \
  "export ROOT=${ROOT}" \
  "export RESULTS=${RESULTS}" \
  "export MAX_EVENTS=${MAX_EVENTS}" \
  "export DEVICE=${DEVICE}" \
  "export GPU=${GPU}" \
  "export OUTPUT_DIR=${OUTPUT_DIR}" \
  "exec bash ${ROOT}/scripts/run_grouped_event_token_physics_reco.sh" \
  > "$WORKER"
chmod +x "$WORKER"

echo "Grouped event-token physics reconstruction"
echo "  events: ${MAX_EVENTS}"
echo "  device: ${DEVICE}"
if [ "$DEVICE" = "cuda" ]; then
  echo "  GPU: ${GPU}"
fi
echo "  output: ${OUTPUT_DIR}"

if [ "$DRY_RUN" -eq 1 ]; then
  echo "DRY RUN: would submit ${UNIT}"
  echo "  worker: ${WORKER}"
  exit 0
fi

systemd-run \
  --unit="$UNIT" \
  --description="Grouped event-token physics reconstruction (${MAX_EVENTS} events, ${DEVICE})" \
  --collect \
  --property=WorkingDirectory="$ROOT" \
  /bin/bash "$WORKER"

echo "Submitted ${UNIT}.service"
echo "Follow: journalctl -u ${UNIT}.service -f"
