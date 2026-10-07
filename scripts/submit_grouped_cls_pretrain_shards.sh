#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
RESULTS=${RESULTS:-${ROOT}/results}
INPUT_DIR=${INPUT_DIR:-${RESULTS}/event_tokens_grouped_cls_final_new_mcdata}
OUTPUT_DIR=${OUTPUT_DIR:-${RESULTS}/event_tokens_grouped_cls_final_new_mcdata_shards}
TMP_DIR=${TMP_DIR:-${RESULTS}/tmp/event_tokens_grouped_cls_final_new_mcdata_shards}
INPUT_VARIANT=${INPUT_VARIANT:-grouped_cls_full}
PIXI=${PIXI:-/root/.pixi/bin/pixi}
TRAIN_FRAC=${TRAIN_FRAC:-0.9}
SEED=${SEED:-42}
READ_BATCH_SIZE=${READ_BATCH_SIZE:-4096}
SHARD_ROWS=${SHARD_ROWS:-50000}
COMPRESSION=${COMPRESSION:-snappy}
OVERWRITE=${OVERWRITE:-0}
DRY_RUN=${DRY_RUN:-0}
STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)}
UNIT=${UNIT:-atlas-grouped-cls-pretrain-shards-${STAMP}}
WORKER=${TMP_DIR}/${UNIT}.sh

SIGNAL=${INPUT_DIR}/event_tokens_signal_${INPUT_VARIANT}_seq256.parquet
BACKGROUND=${INPUT_DIR}/event_tokens_background_${INPUT_VARIANT}_seq256.parquet
DATA=${INPUT_DIR}/event_tokens_data_${INPUT_VARIANT}_seq256.parquet

for path in "$SIGNAL" "$BACKGROUND" "$DATA"; do
  if [ ! -s "$path" ]; then
    echo "Missing input parquet: ${path}" >&2
    exit 1
  fi
done
if [ -e "$OUTPUT_DIR" ] && [ "$OVERWRITE" -ne 1 ]; then
  echo "Output already exists: ${OUTPUT_DIR}" >&2
  echo "Choose a different OUTPUT_DIR. Nothing was changed." >&2
  exit 1
fi

mkdir -p "$TMP_DIR"

cat > "$WORKER" <<EOF
#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT}
PIXI=${PIXI}
CACHE_ROOT=\${RATTLER_CACHE_ROOT:-/home/magaras/.cache/rattler/cache/pkgs}

if ! "\$PIXI" run python -c 'import pyarrow' >/dev/null 2>&1; then
  PYARROW_SITE=\$(find "\$CACHE_ROOT" -path '*/site-packages/pyarrow' -type d | sort -V | tail -n 1)
  [ -n "\$PYARROW_SITE" ] || { echo "Cached pyarrow was not found" >&2; exit 1; }
  PYARROW_SITE=\${PYARROW_SITE%/pyarrow}
  CACHE_LIBS=\$(find "\$CACHE_ROOT" -maxdepth 2 -type d -name lib | paste -sd: -)
  export PYTHONPATH="\${PYARROW_SITE}\${PYTHONPATH:+:\${PYTHONPATH}}"
  export LD_LIBRARY_PATH="\${CACHE_LIBS}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
fi

cd "\$ROOT"

ARGS=(
  --input-parquets
    "${SIGNAL}"
    "${BACKGROUND}"
    "${DATA}"
  --output-dir "${OUTPUT_DIR}"
  --train-frac "${TRAIN_FRAC}"
  --seed "${SEED}"
  --read-batch-size "${READ_BATCH_SIZE}"
  --shard-rows "${SHARD_ROWS}"
  --compression "${COMPRESSION}"
)
if [ "${OVERWRITE}" -eq 1 ]; then
  ARGS+=(--overwrite)
fi

exec "\$PIXI" run python scripts/prepare_token_parquet_pretrain_shards.py "\${ARGS[@]}"
EOF
chmod +x "$WORKER"

echo "Grouped CLS pretraining shard preparation"
echo "  signal:          ${SIGNAL}"
echo "  background:      ${BACKGROUND}"
echo "  data:            ${DATA}"
echo "  output:          ${OUTPUT_DIR}"
echo "  train fraction:  ${TRAIN_FRAC}"
echo "  shard rows:      ${SHARD_ROWS}"
echo "  read batch size: ${READ_BATCH_SIZE}"
echo "  compression:     ${COMPRESSION}"
echo "  worker:          ${WORKER}"

if [ "$DRY_RUN" -eq 1 ]; then
  echo "DRY RUN: would submit ${UNIT}.service"
  exit 0
fi

systemd-run \
  --unit="$UNIT" \
  --description="Prepare grouped CLS foundation-pretraining shards" \
  --collect \
  --property=WorkingDirectory="$ROOT" \
  /bin/bash "$WORKER"

echo "Submitted ${UNIT}.service"
echo "Follow: journalctl -u ${UNIT}.service -f"
