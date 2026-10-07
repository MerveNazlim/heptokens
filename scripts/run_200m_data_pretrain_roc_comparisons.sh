#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
RESULTS=${RESULTS:-${ROOT}/results}
GPU=${GPU:-3}
PRIMARY_Q1=${PRIMARY_Q1:-${RESULTS}/atlas_hzz_q1_200m_data_pretrained_comparison/q1_200m_data_pretrained_hzz_finetuned_cls_mlp}
PRIMARY_Q8=${PRIMARY_Q8:-${RESULTS}/atlas_hzz_q8_200m_data_pretrained_comparison/q8_200m_data_pretrained_hzz_finetuned_cls_mlp}
PRIMARY_CONTINUOUS=${PRIMARY_CONTINUOUS:-${RESULTS}/atlas_hzz_continuous_200m_data_pretrained_comparison/continuous_200m_data_pretrained_hzz_finetuned_cls_mlp}
CONTROLS=${CONTROLS:-${RESULTS}/atlas_hzz_200m_data_pretrain_four_setup}
Q1_DATA=${Q1_DATA:-${RESULTS}/event_tokens_grouped_cls_q1_final_new_mcdata/hzz_ggf_vs_zz_cls_classification_shards}
PAIRED_DATA=${PAIRED_DATA:-${RESULTS}/grouped_representation_full_prepared/20260907-060234/hzz_classification}
OUT=${OUT:-${RESULTS}/atlas_hzz_200m_data_pretrain_comparison}
PYTHON_BIN=${PYTHON_BIN:-${ROOT}/.pixi/envs/default/bin/python}
PYARROW_PKG=${PYARROW_PKG:-/home/magaras/.cache/rattler/cache/pkgs/pyarrow-core-16.1.0-py311hf1d6e26_2_cpu}

pick_ckpt() {
  if [[ -s "$1/checkpoints/best.ckpt" ]]; then
    printf '%s\n' "$1/checkpoints/best.ckpt"
  elif [[ -s "$1/checkpoints/last.ckpt" ]]; then
    printf '%s\n' "$1/checkpoints/last.ckpt"
  else
    echo "Missing checkpoint under $1" >&2
    return 1
  fi
}

Q1=$(pick_ckpt "$PRIMARY_Q1")
Q8=$(pick_ckpt "$PRIMARY_Q8")
CONTINUOUS=$(pick_ckpt "$PRIMARY_CONTINUOUS")
CACHE_LIBS=$(find /home/magaras/.cache/rattler/cache/pkgs -type d -path '*/lib' -print | paste -sd: -)
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="${ROOT}/src:${PYARROW_PKG}/lib/python3.11/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${CACHE_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "$ROOT"

# Plot 1: only the three 200M data-only-pretrained, fine-tuned models.
"$PYTHON_BIN" scripts/evaluate_q1_q8_continuous_hzz.py \
  --q1-checkpoint "$Q1" --q1-prepared-dir "$Q1_DATA" \
  --q8-checkpoint "$Q8" --q8-prepared-dir "$PAIRED_DATA" \
  --continuous-checkpoint "$CONTINUOUS" --continuous-prepared-dir "$PAIRED_DATA" \
  --output-dir "$OUT/pretrained_finetuned_q1_q8_continuous" \
  --device cuda --batch-size 256 --num-workers 0

run_four_setup() {
  local representation=$1
  local title=$2
  local prepared_dir=$3
  local pretrained_finetuned=$4

  "$PYTHON_BIN" scripts/evaluate_hzz_four_setup_comparison.py \
    --representation "$representation" \
    --title "$title" \
    --prepared-dir "$prepared_dir" \
    --pretrained-frozen-checkpoint "$(pick_ckpt "$CONTROLS/${representation}_pretrained_frozen_cls_mlp")" \
    --random-frozen-checkpoint "$(pick_ckpt "$CONTROLS/${representation}_random_frozen_cls_mlp")" \
    --pretrained-finetuned-checkpoint "$pretrained_finetuned" \
    --random-finetuned-checkpoint "$(pick_ckpt "$CONTROLS/${representation}_random_finetuned_cls_mlp")" \
    --output-dir "$OUT/four_setup_${representation}" \
    --device cuda --batch-size 256 --num-workers 0
}

run_four_setup q1 "Q1 tokens: 200M data-only pretraining" "$Q1_DATA" "$Q1"
run_four_setup q8 "Q8 tokens: 200M data-only pretraining" "$PAIRED_DATA" "$Q8"
run_four_setup continuous "Continuous: 200M data-only pretraining" "$PAIRED_DATA" "$CONTINUOUS"

echo "All plots: $OUT"
