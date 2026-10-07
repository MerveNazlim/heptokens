#!/usr/bin/env bash
set -euo pipefail

# Train the three missing downstream controls for one representation. The
# pretrained/fine-tuned configuration is produced by its dedicated submitter.

REPRESENTATION=${REPRESENTATION:?Set REPRESENTATION=q1, q8, or continuous}
ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
RESULTS=${RESULTS:-${ROOT}/results}
GPU=${GPU:-3}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-0}
LOGGER_OFFLINE=${LOGGER_OFFLINE:-true}
DRY_RUN=${DRY_RUN:-0}
PROJECT_NAME=${PROJECT_NAME:-atlas_hzz_200m_data_pretrain_four_setup}
PYTHON_BIN=${PYTHON_BIN:-${ROOT}/.pixi/envs/default/bin/python}
PYARROW_PKG=${PYARROW_PKG:-/home/magaras/.cache/rattler/cache/pkgs/pyarrow-core-16.1.0-py311hf1d6e26_2_cpu}

case "$REPRESENTATION" in
  q1)
    MODEL_REPRESENTATION=vq
    MAX_QUANTIZERS=1
    PRETRAIN_CKPT=${PRETRAIN_CKPT:-${RESULTS}/atlas_event_foundation_pretrain_data16_200m/q1_b448_8gpu_e10_seed42/checkpoints/last.ckpt}
    PREPARED_DIR=${PREPARED_DIR:-${RESULTS}/event_tokens_grouped_cls_q1_final_new_mcdata/hzz_ggf_vs_zz_cls_classification_shards}
    FINETUNED_RUN=${FINETUNED_RUN:-${RESULTS}/atlas_hzz_q1_200m_data_pretrained_comparison/q1_200m_data_pretrained_hzz_finetuned_cls_mlp}
    ;;
  q8)
    MODEL_REPRESENTATION=vq
    MAX_QUANTIZERS=8
    PRETRAIN_CKPT=${PRETRAIN_CKPT:-${RESULTS}/atlas_event_foundation_pretrain_data16_200m/q8_b448_8gpu_e10_seed42/checkpoints/last.ckpt}
    PREPARED_DIR=${PREPARED_DIR:-${RESULTS}/grouped_representation_full_prepared/20260907-060234/hzz_classification}
    FINETUNED_RUN=${FINETUNED_RUN:-${RESULTS}/atlas_hzz_q8_200m_data_pretrained_comparison/q8_200m_data_pretrained_hzz_finetuned_cls_mlp}
    ;;
  continuous)
    MODEL_REPRESENTATION=flat
    MAX_QUANTIZERS=
    PRETRAIN_CKPT=${PRETRAIN_CKPT:-${RESULTS}/atlas_event_foundation_pretrain_data16_200m/continuous_flat_b448_8gpu_e10_seed42/checkpoints/last.ckpt}
    PREPARED_DIR=${PREPARED_DIR:-${RESULTS}/grouped_representation_full_prepared/20260907-060234/hzz_classification}
    FINETUNED_RUN=${FINETUNED_RUN:-${RESULTS}/atlas_hzz_continuous_200m_data_pretrained_comparison/continuous_200m_data_pretrained_hzz_finetuned_cls_mlp}
    ;;
  *)
    echo "REPRESENTATION must be q1, q8, or continuous" >&2
    exit 2
    ;;
esac

[[ -s "$PRETRAIN_CKPT" ]] || { echo "Missing checkpoint: $PRETRAIN_CKPT" >&2; exit 1; }
[[ -s "$PREPARED_DIR/manifest.json" ]] || { echo "Missing manifest: $PREPARED_DIR/manifest.json" >&2; exit 1; }
if [[ ! -s "$FINETUNED_RUN/SUCCESS.txt" ]]; then
  echo "Missing completed pretrained/fine-tuned run: $FINETUNED_RUN" >&2
  echo "Complete the dedicated fine-tuning run first." >&2
  exit 1
fi

STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)}
UNIT=${UNIT:-atlas-hzz-${REPRESENTATION}-four-controls-gpu${GPU}-${STAMP}}
TMP_DIR=${TMP_DIR:-${RESULTS}/tmp/${PROJECT_NAME}}
LOG_DIR=${LOG_DIR:-${RESULTS}/logs/${PROJECT_NAME}}
WORKER=${TMP_DIR}/${UNIT}.sh
LOG_FILE=${LOG_DIR}/${REPRESENTATION}_controls.log
mkdir -p "$TMP_DIR" "$LOG_DIR"

cat > "$WORKER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '${ROOT}'
exec > >(tee -a '${LOG_FILE}') 2>&1

CACHE_LIBS=\$(find /home/magaras/.cache/rattler/cache/pkgs -type d -path '*/lib' -print | paste -sd: -)
export CUDA_VISIBLE_DEVICES='${GPU}'
export HYDRA_FULL_ERROR=1
export MPLBACKEND=Agg
export WANDB_MODE=\$([[ '${LOGGER_OFFLINE}' == true ]] && echo offline || echo online)
export WANDB_INIT_TIMEOUT=300
export WANDB_SILENT=true
export PYTHONPATH='${ROOT}/src:${PYARROW_PKG}/lib/python3.11/site-packages'\${PYTHONPATH:+:\${PYTHONPATH}}
export LD_LIBRARY_PATH="\${CACHE_LIBS}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"

run_control() {
  local network_name=\$1
  local stage=\$2
  local freeze=\$3
  local checkpoint=\${4:-}
  local run_dir='${RESULTS}/${PROJECT_NAME}'/\$network_name

  if [[ -s "\$run_dir/SUCCESS.txt" ]]; then
    echo "Already complete: \$run_dir"
    return 0
  fi

  local -a command=(env
    REPRESENTATION='${MODEL_REPRESENTATION}'
    STAGE="\$stage"
    FREEZE_BACKBONE="\$freeze"
    MAX_QUANTIZERS='${MAX_QUANTIZERS}'
    PREPARED_CLASSIFICATION_DIR='${PREPARED_DIR}'
    OUTPUT_DIR='${RESULTS}'
    PROJECT_NAME='${PROJECT_NAME}'
    NETWORK_NAME="\$network_name"
    EPOCHS='${EPOCHS}'
    BATCH_SIZE='${BATCH_SIZE}'
    NUM_WORKERS='${NUM_WORKERS}'
    DEVICES=1
    ACCELERATOR=gpu
    LOGGER_OFFLINE='${LOGGER_OFFLINE}'
    PYTHON_BIN='${PYTHON_BIN}'
  )
  if [[ -n "\$checkpoint" ]]; then
    command+=(PRETRAIN_CKPT="\$checkpoint")
  fi
  command+=(bash scripts/run_grouped_representation_benchmark.sh)
  "\${command[@]}"
}

run_control '${REPRESENTATION}_pretrained_frozen_cls_mlp' finetune true '${PRETRAIN_CKPT}'
run_control '${REPRESENTATION}_random_frozen_cls_mlp' scratch true
run_control '${REPRESENTATION}_random_finetuned_cls_mlp' scratch false
EOF
chmod +x "$WORKER"

echo "200M data-pretraining four-setup controls"
echo "  representation:       $REPRESENTATION"
echo "  GPU:                  $GPU"
echo "  pretrained checkpoint:$PRETRAIN_CKPT"
echo "  HZZ MC data:          $PREPARED_DIR"
echo "  completed reference:  $FINETUNED_RUN"
echo "  controls output:      ${RESULTS}/${PROJECT_NAME}"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY RUN: would submit ${UNIT}.service"
  exit 0
fi

systemd-run \
  --unit="$UNIT" \
  --description="${REPRESENTATION} four-setup HZZ controls" \
  --collect \
  --property=WorkingDirectory="$ROOT" \
  /bin/bash "$WORKER"

echo "Submitted ${UNIT}.service"
echo "Follow: journalctl -u ${UNIT}.service -f -o cat"
