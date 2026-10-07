#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
RESULTS=${RESULTS:-${ROOT}/results}
GPU=${GPU:-0}
VALIDATION_EVENTS=${VALIDATION_EVENTS:-50000}
MAX_OBJECTS_PER_TYPE=${MAX_OBJECTS_PER_TYPE:-10000}
SAMPLES_PER_OBJECT=${SAMPLES_PER_OBJECT:-4}
TEMPERATURE=${TEMPERATURE:-1.0}
DRY_RUN=${DRY_RUN:-0}

FOUNDATION_RUN=${FOUNDATION_RUN:-${RESULTS}/atlas_event_foundation_pretrain/grouped_new_mcdata_seq256_ddp_equalized_v2}
PREPARED_DIR=${PREPARED_DIR:-${RESULTS}/event_tokens_grouped_final_new_mcdata/pretrain_shards_train90_val10_full}
PROBE_STATE=${PROBE_STATE:-${FOUNDATION_RUN}/teacher_forced_quantizer_probe_200k/probe_state.pt}
RUN_BASE=${RUN_BASE:-${RESULTS}/atlas_object_final_tokenizers_new_mcdata}
OUTPUT_DIR=${OUTPUT_DIR:-${FOUNDATION_RUN}/quantizer_sampling_evaluation_50k}
LOG_DIR=${LOG_DIR:-${RESULTS}/logs/atlas_quantizer_sampling_evaluation}
TMP_DIR=${TMP_DIR:-${RESULTS}/tmp/atlas_quantizer_sampling_evaluation}

JETS_RUN=${JETS_RUN:-${RUN_BASE}/jets_full_dim8_cb2048_q8_e20_new_mcdata}
ELECTRONS_RUN=${ELECTRONS_RUN:-${RUN_BASE}/electrons_full_dim8_cb2048_q8_e20_new_mcdata}
MUONS_RUN=${MUONS_RUN:-${RUN_BASE}/muons_full_dim8_cb2048_q8_e20_new_mcdata}
PHOTONS_RUN=${PHOTONS_RUN:-${RUN_BASE}/photons_full_dim8_cb2048_q8_e20_new_mcdata}
TAUS_RUN=${TAUS_RUN:-${RUN_BASE}/taus_full_dim8_cb4096_q8_e20_new_mcdata}
TRACKS_RUN=${TRACKS_RUN:-${RUN_BASE}/tracks_full_dim8_cb4096_q8_e20_new_mcdata}

for path in \
  "${FOUNDATION_RUN}/checkpoints/last.ckpt" \
  "${PREPARED_DIR}/manifest.json" \
  "${PROBE_STATE}" \
  "${ROOT}/scripts/evaluate_grouped_quantizer_sampling.py" \
  "${JETS_RUN}/full_config.yaml" \
  "${ELECTRONS_RUN}/full_config.yaml" \
  "${MUONS_RUN}/full_config.yaml" \
  "${PHOTONS_RUN}/full_config.yaml" \
  "${TAUS_RUN}/full_config.yaml" \
  "${TRACKS_RUN}/full_config.yaml"
do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required input: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${LOG_DIR}" "${TMP_DIR}"
TS=$(date +%Y%m%d-%H%M%S)
UNIT="atlas-quantizer-sampling-eval-gpu${GPU}-${TS}"
WORKER="${TMP_DIR}/${UNIT}.sh"
LOG_FILE="${LOG_DIR}/${UNIT}.log"

cat > "${WORKER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"

PYARROW_PKG=/home/magaras/.cache/rattler/cache/pkgs/pyarrow-core-16.1.0-py311hf1d6e26_2_cpu
CACHE_LIBS=\$(find /home/magaras/.cache/rattler/cache/pkgs -type d -path '*/lib' -print | paste -sd: -)

echo "\$(date): starting grouped quantizer sampling evaluation"
echo "GPU=${GPU} validation_events=${VALIDATION_EVENTS} max_objects_per_type=${MAX_OBJECTS_PER_TYPE} samples_per_object=${SAMPLES_PER_OBJECT}"

CUDA_VISIBLE_DEVICES="${GPU}" \
MPLBACKEND=Agg \
PYTHONPATH="\${PYARROW_PKG}/lib/python3.11/site-packages\${PYTHONPATH:+:\$PYTHONPATH}" \
LD_LIBRARY_PATH="\${CACHE_LIBS}\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}" \
/usr/bin/time -v \
/root/.pixi/bin/pixi run python \
  scripts/evaluate_grouped_quantizer_sampling.py \
  --run-dir "${FOUNDATION_RUN}" \
  --prepared-dir "${PREPARED_DIR}" \
  --probe-state "${PROBE_STATE}" \
  --tokenizer-run "jets=${JETS_RUN}" \
  --tokenizer-run "electrons=${ELECTRONS_RUN}" \
  --tokenizer-run "muons=${MUONS_RUN}" \
  --tokenizer-run "photons=${PHOTONS_RUN}" \
  --tokenizer-run "taus=${TAUS_RUN}" \
  --tokenizer-run "tracks=${TRACKS_RUN}" \
  --device cuda \
  --batch-size 32 \
  --stream-batch-size 1024 \
  --validation-events "${VALIDATION_EVENTS}" \
  --max-objects-per-type "${MAX_OBJECTS_PER_TYPE}" \
  --samples-per-object "${SAMPLES_PER_OBJECT}" \
  --temperature "${TEMPERATURE}" \
  --decode-batch-size 2048 \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee "${LOG_FILE}"

echo "\$(date): evaluation completed"
EOF
chmod +x "${WORKER}"

echo "Grouped quantizer sampling evaluation"
echo "  GPU: ${GPU}"
echo "  validation events: ${VALIDATION_EVENTS}"
echo "  maximum objects/type: ${MAX_OBJECTS_PER_TYPE}"
echo "  samples/object: ${SAMPLES_PER_OBJECT}"
echo "  output: ${OUTPUT_DIR}"
echo "  worker: ${WORKER}"
echo "  log: ${LOG_FILE}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY RUN: would submit ${UNIT}.service"
  exit 0
fi

systemd-run \
  --unit="${UNIT}" \
  --property=WorkingDirectory="${ROOT}" \
  /bin/bash "${WORKER}"

echo
echo "Submitted ${UNIT}.service"
echo "  progress: journalctl -u ${UNIT}.service -f -o cat"
echo "  status:   systemctl status ${UNIT}.service --no-pager"
