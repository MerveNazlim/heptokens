#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
RESULTS=${RESULTS:-${ROOT}/results}
GPU=${GPU:-3}
Q1_RUN=${Q1_RUN:-${RESULTS}/atlas_hzz_q1_200m_data_pretrained_comparison/q1_200m_data_pretrained_hzz_finetuned_cls_mlp}
HIERARCHICAL_RUN=${HIERARCHICAL_RUN:-${RESULTS}/atlas_hzz_continuous_hierarchical_200m_data_pretrained_comparison/continuous_hierarchical_200m_data_pretrained_hzz_finetuned_cls_mlp}
FLAT_RUN=${FLAT_RUN:-${RESULTS}/atlas_hzz_continuous_200m_data_pretrained_comparison/continuous_200m_data_pretrained_hzz_finetuned_cls_mlp}
Q1_PREPARED_DIR=${Q1_PREPARED_DIR:-${RESULTS}/event_tokens_grouped_cls_q1_final_new_mcdata/hzz_ggf_vs_zz_cls_classification_shards}
CONTINUOUS_PREPARED_DIR=${CONTINUOUS_PREPARED_DIR:-${RESULTS}/grouped_representation_full_prepared/20260907-060234/hzz_classification}
OUTPUT_DIR=${OUTPUT_DIR:-${RESULTS}/atlas_hzz_q1_hierarchical_flat_200m_comparison}
PYTHON_BIN=${PYTHON_BIN:-${ROOT}/.pixi/envs/default/bin/python}
PYARROW_PKG=${PYARROW_PKG:-/home/magaras/.cache/rattler/cache/pkgs/pyarrow-core-16.1.0-py311hf1d6e26_2_cpu}
DRY_RUN=${DRY_RUN:-0}

pick_ckpt () {
  if [[ -s "$1/checkpoints/best.ckpt" ]]; then
    printf '%s\n' "$1/checkpoints/best.ckpt"
  elif [[ -s "$1/checkpoints/last.ckpt" ]]; then
    printf '%s\n' "$1/checkpoints/last.ckpt"
  else
    echo "Missing classifier checkpoint under $1" >&2
    return 1
  fi
}

Q1_CKPT=$(pick_ckpt "$Q1_RUN")
HIERARCHICAL_CKPT=$(pick_ckpt "$HIERARCHICAL_RUN")
FLAT_CKPT=$(pick_ckpt "$FLAT_RUN")
test -s "$Q1_PREPARED_DIR/manifest.json"
test -s "$CONTINUOUS_PREPARED_DIR/manifest.json"

STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)}
UNIT=${UNIT:-atlas-hzz-q1-hier-flat-eval-gpu${GPU}-${STAMP}}
WORKER=${RESULTS}/tmp/atlas_hzz_q1_hierarchical_flat_200m_comparison/${UNIT}.sh
LOG_FILE=${RESULTS}/logs/atlas_hzz_q1_hierarchical_flat_200m_comparison/evaluation.log
mkdir -p "$(dirname "$WORKER")" "$(dirname "$LOG_FILE")" "$OUTPUT_DIR"

cat > "$WORKER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '${ROOT}'
exec > >(tee -a '${LOG_FILE}') 2>&1
CACHE_LIBS=\$(find /home/magaras/.cache/rattler/cache/pkgs -type d -path '*/lib' -print | paste -sd: -)
CUDA_VISIBLE_DEVICES='${GPU}' \
MPLBACKEND=Agg \
PYTHONPATH='${ROOT}/src:${ROOT}/scripts:${PYARROW_PKG}/lib/python3.11/site-packages'\${PYTHONPATH:+:\${PYTHONPATH}} \
LD_LIBRARY_PATH="\${CACHE_LIBS}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}" \
'${PYTHON_BIN}' scripts/evaluate_q1_hierarchical_flat_hzz.py \
  --q1-checkpoint '${Q1_CKPT}' \
  --q1-prepared-dir '${Q1_PREPARED_DIR}' \
  --hierarchical-checkpoint '${HIERARCHICAL_CKPT}' \
  --flat-checkpoint '${FLAT_CKPT}' \
  --continuous-prepared-dir '${CONTINUOUS_PREPARED_DIR}' \
  --output-dir '${OUTPUT_DIR}' \
  --device cuda \
  --batch-size 256 \
  --num-workers 0
EOF
chmod +x "$WORKER"

echo "Q1 vs hierarchical vs flat continuous HZZ evaluation"
echo "  GPU:          ${GPU}"
echo "  Q1:           ${Q1_CKPT}"
echo "  hierarchical: ${HIERARCHICAL_CKPT}"
echo "  flat:         ${FLAT_CKPT}"
echo "  output:       ${OUTPUT_DIR}"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY RUN: would submit ${UNIT}.service"
  exit 0
fi

systemd-run \
  --unit="$UNIT" \
  --description="Evaluate Q1, hierarchical, and flat continuous HZZ classifiers" \
  --collect \
  --property=WorkingDirectory="$ROOT" \
  /bin/bash "$WORKER"

echo "Submitted ${UNIT}.service"
echo "Follow: journalctl -u ${UNIT}.service -f -o cat"

