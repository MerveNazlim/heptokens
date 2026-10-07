#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
RESULTS=${RESULTS:-${ROOT}/results}
RUN_DIR=${RUN_DIR:-${RESULTS}/atlas_hzz_grouped_cls_multiclass/q8_perlmutter_pretrained_finetuned_multiclass}
PREPARED_DIR=${PREPARED_DIR:-}
CHECKPOINT=${CHECKPOINT:-${RUN_DIR}/checkpoints/best.ckpt}
OUTPUT_DIR=${OUTPUT_DIR:-${RUN_DIR}/test_evaluation}
GPU=${GPU:-0}
BATCH_SIZE=${BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-0}
PIXI=${PIXI:-/root/.pixi/bin/pixi}
PYARROW_PKG=${PYARROW_PKG:-/home/magaras/.cache/rattler/cache/pkgs/pyarrow-core-16.1.0-py311hf1d6e26_2_cpu}

if [[ -z "$PREPARED_DIR" ]]; then
  mapfile -t manifests < <(
    find "$RESULTS" -type f -path '*multiclass*/manifest.json' -print | sort
  )
  if [[ ${#manifests[@]} -ne 1 ]]; then
    echo "Expected exactly one multiclass manifest, found ${#manifests[@]}." >&2
    echo "Set PREPARED_DIR explicitly." >&2
    exit 1
  fi
  PREPARED_DIR=$(dirname "${manifests[0]}")
fi
if [[ ! -s "$CHECKPOINT" ]]; then
  echo "Missing classifier checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -s "${PREPARED_DIR}/manifest.json" ]]; then
  echo "Missing multiclass manifest: ${PREPARED_DIR}/manifest.json" >&2
  exit 1
fi

STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)}
UNIT=${UNIT:-atlas-hzz-multiclass-eval-gpu${GPU}-${STAMP}}
WORKER=${RESULTS}/tmp/atlas_hzz_grouped_cls_multiclass/${UNIT}.sh
LOG_FILE=${RESULTS}/logs/atlas_hzz_grouped_cls_multiclass/multiclass_evaluation.log
mkdir -p "$(dirname "$WORKER")" "$(dirname "$LOG_FILE")" "$OUTPUT_DIR"

cat > "$WORKER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '${ROOT}'
exec > >(tee -a '${LOG_FILE}') 2>&1
CACHE_LIBS=\$(find /home/magaras/.cache/rattler/cache/pkgs -type d -path '*/lib' -print | paste -sd: -)
CUDA_VISIBLE_DEVICES='${GPU}' \
MPLBACKEND=Agg \
PYTHONPATH='${ROOT}/src:${PYARROW_PKG}/lib/python3.11/site-packages'\${PYTHONPATH:+:\${PYTHONPATH}} \
LD_LIBRARY_PATH="\${CACHE_LIBS}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}" \
'${PIXI}' run python scripts/evaluate_grouped_hzz_multiclass.py \
  --prepared-dir '${PREPARED_DIR}' \
  --checkpoint '${CHECKPOINT}' \
  --output-dir '${OUTPUT_DIR}' \
  --batch-size '${BATCH_SIZE}' \
  --num-workers '${NUM_WORKERS}' \
  --device cuda
EOF
chmod +x "$WORKER"

echo "Grouped HZZ multiclass test evaluation"
echo "  GPU:        ${GPU}"
echo "  checkpoint: ${CHECKPOINT}"
echo "  test data:  ${PREPARED_DIR}"
echo "  output:     ${OUTPUT_DIR}"

systemd-run \
  --unit="$UNIT" \
  --description="Evaluate grouped-CLS HZZ multiclass classifier" \
  --collect \
  --property=WorkingDirectory="$ROOT" \
  /bin/bash "$WORKER"

echo "Submitted ${UNIT}.service"
echo "Follow: journalctl -u ${UNIT}.service -f -o cat"
