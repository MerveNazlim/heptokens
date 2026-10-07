#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
RESULTS=${RESULTS:-${ROOT}/results}
GPU=${GPU:-3}
BASE=${BASE:-${RESULTS}/atlas_hzz_q1_q8_continuous_mlp_comparison}
NEW_Q1_RUN=${NEW_Q1_RUN:-${RESULTS}/atlas_hzz_q1_200m_data_pretrained_comparison/q1_200m_data_pretrained_hzz_finetuned_cls_mlp}
NEW_Q8_RUN=${NEW_Q8_RUN:-${RESULTS}/atlas_hzz_q8_200m_data_pretrained_comparison/q8_200m_data_pretrained_hzz_finetuned_cls_mlp}
Q1_PREPARED_DIR=${Q1_PREPARED_DIR:-${RESULTS}/event_tokens_grouped_cls_q1_final_new_mcdata/hzz_ggf_vs_zz_cls_classification_shards}
PAIRED_PREPARED_DIR=${PAIRED_PREPARED_DIR:-${RESULTS}/grouped_representation_full_prepared/20260907-060234/hzz_classification}
CONTINUOUS_RUN=${CONTINUOUS_RUN:-${RESULTS}/atlas_hzz_flat_continuous_classification/perlmutter_pretrained_finetuned_cls}
OUT=${OUT:-${BASE}/roc_comparison_q1_q8_200m_data_pretrain_perlmutter_continuous}
PYTHON_BIN=${PYTHON_BIN:-${ROOT}/.pixi/envs/default/bin/python}
PYARROW_PKG=${PYARROW_PKG:-/home/magaras/.cache/rattler/cache/pkgs/pyarrow-core-16.1.0-py311hf1d6e26_2_cpu}

pick_ckpt() {
  if [[ -s "$1/checkpoints/best.ckpt" ]]; then
    printf '%s\n' "$1/checkpoints/best.ckpt"
  elif [[ -s "$1/checkpoints/last.ckpt" ]]; then
    printf '%s\n' "$1/checkpoints/last.ckpt"
  else
    echo "Missing classifier checkpoint under $1" >&2
    return 1
  fi
}

Q1=$(pick_ckpt "${BASE}/q1_pretrained_finetuned_cls_mlp")
Q1_DATA_ONLY=$(pick_ckpt "$NEW_Q1_RUN")
Q8=$(pick_ckpt "${BASE}/q8_pretrained_finetuned_cls_mlp")
Q8_DATA_ONLY=$(pick_ckpt "$NEW_Q8_RUN")
CONTINUOUS=$(pick_ckpt "$CONTINUOUS_RUN")
CACHE_LIBS=$(find /home/magaras/.cache/rattler/cache/pkgs -type d -path '*/lib' -print | paste -sd: -)

cd "$ROOT"
CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH="${ROOT}/src:${PYARROW_PKG}/lib/python3.11/site-packages${PYTHONPATH:+:${PYTHONPATH}}" \
LD_LIBRARY_PATH="${CACHE_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
"$PYTHON_BIN" scripts/evaluate_q1_q8_continuous_hzz.py \
  --q1-checkpoint "$Q1" \
  --q1-prepared-dir "$Q1_PREPARED_DIR" \
  --q1-data-only-checkpoint "$Q1_DATA_ONLY" \
  --q1-data-only-prepared-dir "$Q1_PREPARED_DIR" \
  --q8-checkpoint "$Q8" \
  --q8-prepared-dir "$PAIRED_PREPARED_DIR" \
  --q8-data-only-checkpoint "$Q8_DATA_ONLY" \
  --q8-data-only-prepared-dir "$PAIRED_PREPARED_DIR" \
  --continuous-checkpoint "$CONTINUOUS" \
  --continuous-prepared-dir "$PAIRED_PREPARED_DIR" \
  --output-dir "$OUT" \
  --device cuda \
  --batch-size 256 \
  --num-workers 0

echo "Plots: $OUT"
