#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
STAMP=${STAMP:-20260826-052254}
GPU=${GPU:-0}
VALIDATION_EVENTS=${VALIDATION_EVENTS:-50000}
MAX_OBJECTS_PER_TYPE=${MAX_OBJECTS_PER_TYPE:-10000}
SAMPLES_PER_OBJECT=${SAMPLES_PER_OBJECT:-4}

AR_RUN=${AR_RUN:-${ROOT}/results/atlas_grouped_decoder_comparison/grouped_cls_autoregressive_3epochs_${STAMP}}
PARALLEL_RUN=${PARALLEL_RUN:-${ROOT}/results/atlas_grouped_decoder_comparison/grouped_cls_parallel_3epochs_${STAMP}}
PREPARED_DIR=${PREPARED_DIR:-${ROOT}/results/event_tokens_grouped_cls_final_new_mcdata_shards}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT}/results/atlas_grouped_decoder_comparison/evaluation_${STAMP}_50k}
LOG_DIR=${LOG_DIR:-${ROOT}/results/logs/atlas_grouped_decoder_comparison/${STAMP}}

TOKENIZER_BASE=${TOKENIZER_BASE:-${ROOT}/results/atlas_object_final_tokenizers_new_mcdata}
JETS_RUN=${JETS_RUN:-${TOKENIZER_BASE}/jets_full_dim8_cb2048_q8_e20_new_mcdata}
ELECTRONS_RUN=${ELECTRONS_RUN:-${TOKENIZER_BASE}/electrons_full_dim8_cb2048_q8_e20_new_mcdata}
MUONS_RUN=${MUONS_RUN:-${TOKENIZER_BASE}/muons_full_dim8_cb2048_q8_e20_new_mcdata}
PHOTONS_RUN=${PHOTONS_RUN:-${TOKENIZER_BASE}/photons_full_dim8_cb2048_q8_e20_new_mcdata}
TAUS_RUN=${TAUS_RUN:-${TOKENIZER_BASE}/taus_full_dim8_cb4096_q8_e20_new_mcdata}
TRACKS_RUN=${TRACKS_RUN:-${TOKENIZER_BASE}/tracks_full_dim8_cb4096_q8_e20_new_mcdata}

for path in \
  "${AR_RUN}/checkpoints/last.ckpt" \
  "${PARALLEL_RUN}/checkpoints/last.ckpt" \
  "${PREPARED_DIR}/val"; do
  test -e "${path}" || { echo "Missing required path: ${path}" >&2; exit 1; }
done

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

PYARROW_PKG=${PYARROW_PKG:-/home/magaras/.cache/rattler/cache/pkgs/pyarrow-core-16.1.0-py311hf1d6e26_2_cpu}
CACHE_LIBS=${CACHE_LIBS:-$(find /home/magaras/.cache/rattler/cache/pkgs -type d -path '*/lib' -print | paste -sd: -)}
PYTHONPATH_VALUE="${PYARROW_PKG}/lib/python3.11/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
LD_LIBRARY_PATH_VALUE="${CACHE_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

SUBMIT_TIME=$(date +%Y%m%d-%H%M%S)
UNIT="atlas-grouped-decoder-eval-gpu${GPU}-${SUBMIT_TIME}"
systemd-run \
  --unit="${UNIT}" \
  --collect \
  --description="Matched grouped parallel vs autoregressive decoder evaluation" \
  --property=WorkingDirectory="${ROOT}" \
  --setenv="CUDA_VISIBLE_DEVICES=${GPU}" \
  --setenv="MPLBACKEND=Agg" \
  --setenv="PYTHONPATH=${PYTHONPATH_VALUE}" \
  --setenv="LD_LIBRARY_PATH=${LD_LIBRARY_PATH_VALUE}" \
  /bin/bash -c 'exec /root/.pixi/bin/pixi "$@"' bash run python \
    scripts/evaluate_grouped_decoder_comparison.py \
    --autoregressive-run "${AR_RUN}" \
    --parallel-run "${PARALLEL_RUN}" \
    --prepared-dir "${PREPARED_DIR}" \
    --device cuda \
    --batch-size 16 \
    --stream-batch-size 1024 \
    --validation-events "${VALIDATION_EVENTS}" \
    --mask-prob 0.15 \
    --max-objects-per-type "${MAX_OBJECTS_PER_TYPE}" \
    --samples-per-object "${SAMPLES_PER_OBJECT}" \
    --temperature 1.0 \
    --decode-batch-size 2048 \
    --seed 42 \
    --tokenizer-run "jets=${JETS_RUN}" \
    --tokenizer-run "electrons=${ELECTRONS_RUN}" \
    --tokenizer-run "muons=${MUONS_RUN}" \
    --tokenizer-run "photons=${PHOTONS_RUN}" \
    --tokenizer-run "taus=${TAUS_RUN}" \
    --tokenizer-run "tracks=${TRACKS_RUN}" \
    --output-dir "${OUTPUT_DIR}"

echo "Submitted ${UNIT}.service"
echo "Follow: journalctl -u ${UNIT}.service -f -o cat"
echo "Output: ${OUTPUT_DIR}"
