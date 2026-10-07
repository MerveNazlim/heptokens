#!/usr/bin/env bash
set -euo pipefail

# Prepare q1 grouped-CLS pretraining and HZZ classification shards sequentially.

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
RESULTS=${RESULTS:-${ROOT}/results}
INPUT_DIR=${INPUT_DIR:-${RESULTS}/event_tokens_grouped_cls_q1_final_new_mcdata}
INPUT_VARIANT=${INPUT_VARIANT:-grouped_cls_q1_full}
PRETRAIN_OUTPUT=${PRETRAIN_OUTPUT:-${RESULTS}/event_tokens_grouped_cls_q1_final_new_mcdata_shards}
CLASSIFICATION_OUTPUT=${CLASSIFICATION_OUTPUT:-${INPUT_DIR}/hzz_ggf_vs_zz_cls_classification_shards}
TMP_DIR=${TMP_DIR:-${RESULTS}/tmp/q1_grouped_cls_shards}
LOG_DIR=${LOG_DIR:-${RESULTS}/logs/q1_grouped_cls_shards}
PIXI=${PIXI:-/root/.pixi/bin/pixi}

RUN_SET=${RUN_SET:-both}
TRAIN_FRAC=${TRAIN_FRAC:-0.90}
CLASSIFICATION_TRAIN_FRAC=${CLASSIFICATION_TRAIN_FRAC:-0.70}
CLASSIFICATION_VAL_FRAC=${CLASSIFICATION_VAL_FRAC:-0.15}
SIGNAL_DSID=${SIGNAL_DSID:-345060}
BACKGROUND_DSID=${BACKGROUND_DSID:-700600}
SEED=${SEED:-42}
READ_BATCH_SIZE=${READ_BATCH_SIZE:-4096}
SHARD_ROWS=${SHARD_ROWS:-50000}
COMPRESSION=${COMPRESSION:-snappy}
OVERWRITE=${OVERWRITE:-0}
DRY_RUN=${DRY_RUN:-0}
STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)}
UNIT=${UNIT:-atlas-q1-grouped-cls-shards-${STAMP}}
WORKER=${TMP_DIR}/${UNIT}.sh
LOG_FILE=${LOG_DIR}/${UNIT}.log

SIGNAL=${INPUT_DIR}/event_tokens_signal_${INPUT_VARIANT}_seq256.parquet
BACKGROUND=${INPUT_DIR}/event_tokens_background_${INPUT_VARIANT}_seq256.parquet
DATA=${INPUT_DIR}/event_tokens_data_${INPUT_VARIANT}_seq256.parquet

case "$RUN_SET" in
  both|pretrain|classification) ;;
  *)
    echo "RUN_SET must be one of: both, pretrain, classification" >&2
    exit 1
    ;;
esac

for path in "$SIGNAL" "$BACKGROUND" "$DATA"; do
  if [[ ! -s "$path" ]]; then
    echo "Missing input Parquet: ${path}" >&2
    exit 1
  fi
done

check_output() {
  local label=$1
  local path=$2
  if [[ -e "$path" && "$OVERWRITE" -ne 1 ]]; then
    echo "${label} output already exists: ${path}" >&2
    echo "Use a different output path or set OVERWRITE=1 deliberately." >&2
    exit 1
  fi
}

if [[ "$RUN_SET" == both || "$RUN_SET" == pretrain ]]; then
  check_output "Pretraining" "$PRETRAIN_OUTPUT"
fi
if [[ "$RUN_SET" == both || "$RUN_SET" == classification ]]; then
  check_output "Classification" "$CLASSIFICATION_OUTPUT"
fi

mkdir -p "$TMP_DIR" "$LOG_DIR"
cd "$ROOT"

cat > "$WORKER" <<EOF
#!/usr/bin/env bash
set -euo pipefail

ROOT='${ROOT}'
PIXI='${PIXI}'
CACHE_ROOT=\${RATTLER_CACHE_ROOT:-/home/magaras/.cache/rattler/cache/pkgs}
exec > >(tee -a '${LOG_FILE}') 2>&1

if ! "\$PIXI" run python -c 'import pyarrow' >/dev/null 2>&1; then
  PYARROW_SITE=\$(find "\$CACHE_ROOT" -path '*/site-packages/pyarrow' -type d | sort -V | tail -n 1)
  [[ -n "\$PYARROW_SITE" ]] || { echo "Cached pyarrow was not found" >&2; exit 1; }
  PYARROW_SITE=\${PYARROW_SITE%/pyarrow}
  CACHE_LIBS=\$(find "\$CACHE_ROOT" -maxdepth 2 -type d -name lib | paste -sd: -)
  export PYTHONPATH="\${PYARROW_SITE}\${PYTHONPATH:+:\${PYTHONPATH}}"
  export LD_LIBRARY_PATH="\${CACHE_LIBS}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
fi

cd "\$ROOT"

echo "Validating q1 grouped Parquets"
"\$PIXI" run python scripts/check_grouped_event_parquets.py \
  '${SIGNAL}' '${BACKGROUND}' '${DATA}' --rows 2048

"\$PIXI" run python -c '
import json
import sys
import pyarrow.parquet as pq

key = b"heptokens_token_vocabulary"
reference_vocabulary = None
for path in sys.argv[1:]:
    metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
    if key not in metadata:
        raise RuntimeError(f"{path}: missing token vocabulary metadata")
    vocabulary = json.loads(metadata[key])
    if reference_vocabulary is None:
        reference_vocabulary = vocabulary
    elif vocabulary != reference_vocabulary:
        raise RuntimeError(f"{path}: token vocabulary differs from the first input")
    maximum = int(vocabulary["max_quantizers"])
    object_quantizers = {
        name: int(spec["num_quantizers"])
        for name, spec in vocabulary["objects"].items()
    }
    if maximum != 1 or any(value != 1 for value in object_quantizers.values()):
        raise RuntimeError(
            f"{path}: expected q1, found max_quantizers={maximum}, "
            f"objects={object_quantizers}"
        )
    if not vocabulary.get("includes_cls", False):
        raise RuntimeError(f"{path}: expected grouped-CLS input")
    print(f"q1 metadata OK: {path}")
' '${SIGNAL}' '${BACKGROUND}' '${DATA}'

if [[ '${RUN_SET}' == both || '${RUN_SET}' == pretrain ]]; then
  echo "Preparing q1 pretraining shards"
  PRETRAIN_ARGS=(
    --input-parquets '${SIGNAL}' '${BACKGROUND}' '${DATA}'
    --output-dir '${PRETRAIN_OUTPUT}'
    --train-frac '${TRAIN_FRAC}'
    --seed '${SEED}'
    --read-batch-size '${READ_BATCH_SIZE}'
    --shard-rows '${SHARD_ROWS}'
    --compression '${COMPRESSION}'
  )
  if [[ '${OVERWRITE}' -eq 1 ]]; then
    PRETRAIN_ARGS+=(--overwrite)
  fi
  "\$PIXI" run python scripts/prepare_token_parquet_pretrain_shards.py \
    "\${PRETRAIN_ARGS[@]}"
fi

if [[ '${RUN_SET}' == both || '${RUN_SET}' == classification ]]; then
  echo "Preparing q1 HZZ classification shards"
  CLASSIFICATION_ARGS=(
    --signal-parquet '${SIGNAL}'
    --background-parquet '${BACKGROUND}'
    --signal-dsid '${SIGNAL_DSID}'
    --background-dsid '${BACKGROUND_DSID}'
    --output-dir '${CLASSIFICATION_OUTPUT}'
    --train-frac '${CLASSIFICATION_TRAIN_FRAC}'
    --val-frac '${CLASSIFICATION_VAL_FRAC}'
    --seed '${SEED}'
    --read-batch-size '${READ_BATCH_SIZE}'
    --shard-rows '${SHARD_ROWS}'
    --compression '${COMPRESSION}'
  )
  if [[ '${OVERWRITE}' -eq 1 ]]; then
    CLASSIFICATION_ARGS+=(--overwrite)
  fi
  "\$PIXI" run python scripts/prepare_grouped_hzz_classification_shards.py \
    "\${CLASSIFICATION_ARGS[@]}"
fi

echo "q1 shard preparation completed"
EOF
chmod +x "$WORKER"

echo "q1 grouped-CLS shard preparation"
echo "  run set:              ${RUN_SET}"
echo "  signal:               ${SIGNAL}"
echo "  background:           ${BACKGROUND}"
echo "  data:                 ${DATA}"
echo "  pretraining output:   ${PRETRAIN_OUTPUT}"
echo "  classification output:${CLASSIFICATION_OUTPUT}"
echo "  pretraining train fraction: ${TRAIN_FRAC} (remainder is validation)"
echo "  classification fractions:   train=${CLASSIFICATION_TRAIN_FRAC}, val=${CLASSIFICATION_VAL_FRAC} (remainder is test)"
echo "  classification DSIDs: ${SIGNAL_DSID} versus ${BACKGROUND_DSID}"
echo "  seed:                 ${SEED}"
echo "  shard rows:           ${SHARD_ROWS}"
echo "  worker:               ${WORKER}"
echo "  log:                  ${LOG_FILE}"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY RUN: would submit ${UNIT}.service"
  exit 0
fi

systemd-run \
  --unit="$UNIT" \
  --description="Prepare q1 grouped-CLS pretraining and HZZ classification shards" \
  --collect \
  --property=WorkingDirectory="$ROOT" \
  /bin/bash "$WORKER"

echo "Submitted ${UNIT}.service"
echo "Follow: journalctl -u ${UNIT}.service -f -o cat"
