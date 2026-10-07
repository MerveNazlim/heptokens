#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
STAMP=${STAMP:-20260826-052254}
GPU=${GPU:-0}
BATCH_SIZE=${BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-3}
PROJECT_DIR=${PROJECT_DIR:-${ROOT}/results/atlas_grouped_decoder_hzz_comparison_${STAMP}}
PREPARED_DIR=${PREPARED_DIR:-${ROOT}/results/event_tokens_grouped_cls_final_new_mcdata/hzz_ggf_vs_zz_cls_classification_shards}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_DIR}/quantizer_input_ablation}
PYARROW_PKG=${PYARROW_PKG:-/home/magaras/.cache/rattler/cache/pkgs/pyarrow-core-16.1.0-py311hf1d6e26_2_cpu}

for path in "${PROJECT_DIR}" "${PREPARED_DIR}/manifest.json"; do
  test -e "${path}" || { echo "Missing required path: ${path}" >&2; exit 1; }
done

mkdir -p "${OUTPUT_DIR}"
CACHE_LIBS=${CACHE_LIBS:-$(find /home/magaras/.cache/rattler/cache/pkgs -type d -path '*/lib' -print | paste -sd: -)}
PYTHONPATH_VALUE="${PYARROW_PKG}/lib/python3.11/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
LD_LIBRARY_PATH_VALUE="${CACHE_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

SUBMIT_TIME=$(date +%Y%m%d-%H%M%S)
UNIT="atlas-decoder-hzz-q-ablation-gpu${GPU}-${SUBMIT_TIME}"
systemd-run \
  --unit="${UNIT}" \
  --collect \
  --description="HZZ inference-only residual-quantizer input ablation" \
  --property=WorkingDirectory="${ROOT}" \
  --setenv="CUDA_VISIBLE_DEVICES=${GPU}" \
  --setenv="MPLBACKEND=Agg" \
  --setenv="PYTHONPATH=${PYTHONPATH_VALUE}" \
  --setenv="LD_LIBRARY_PATH=${LD_LIBRARY_PATH_VALUE}" \
  /bin/bash -c 'exec /root/.pixi/bin/pixi "$@"' bash run python \
    scripts/evaluate_grouped_decoder_hzz_comparison.py \
    --project-dir "${PROJECT_DIR}" \
    --prepared-dir "${PREPARED_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --device cuda \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --quantizer-cutoffs 0 1 3 7

echo "Submitted ${UNIT}.service"
echo "Follow: journalctl -u ${UNIT}.service -f -o cat"
echo "Output: ${OUTPUT_DIR}"
