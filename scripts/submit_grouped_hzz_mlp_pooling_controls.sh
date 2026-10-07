#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
GPU=${GPU:-0}
POOLING=${POOLING:-all}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-3}
LOGGER_OFFLINE=${LOGGER_OFFLINE:-true}
CONTINUE_ON_FAILURE=${CONTINUE_ON_FAILURE:-0}
DRY_RUN=${DRY_RUN:-0}

PROJECT_NAME=${PROJECT_NAME:-atlas_hzz_grouped_mlp_pooling_controls}
PREPARED_DIR=${PREPARED_DIR:-${ROOT}/results/event_tokens_grouped_cls_final_new_mcdata/hzz_ggf_vs_zz_cls_classification_shards}
FOUNDATION_RUN=${FOUNDATION_RUN:-${ROOT}/results/google_foundation_grouped_cls_cluster_60}
BACKBONE_CKPT=${BACKBONE_CKPT:-}
PYARROW_PKG=${PYARROW_PKG:-/home/magaras/.cache/rattler/cache/pkgs/pyarrow-core-16.1.0-py311hf1d6e26_2_cpu}

if [[ "${POOLING}" != all && "${POOLING}" != cls && "${POOLING}" != mean ]]; then
  echo "POOLING must be all, cls, or mean" >&2
  exit 1
fi
if [[ -z "${BACKBONE_CKPT}" ]]; then
  for candidate in "${FOUNDATION_RUN}/checkpoints/best.ckpt" "${FOUNDATION_RUN}/checkpoints/last.ckpt"; do
    if [[ -f "${candidate}" ]]; then BACKBONE_CKPT=${candidate}; break; fi
  done
fi
for path in "${PREPARED_DIR}/manifest.json" "${BACKBONE_CKPT}"; do
  if [[ ! -f "${path}" ]]; then echo "Missing required input: ${path}" >&2; exit 1; fi
done

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
UNIT="atlas-hzz-mlp-${POOLING}-gpu${GPU}-${TIMESTAMP}"
TMP_DIR="${ROOT}/results/tmp/${PROJECT_NAME}"
LOG_DIR="${ROOT}/results/logs/${PROJECT_NAME}"
WORKER="${TMP_DIR}/${UNIT}.sh"
mkdir -p "${TMP_DIR}" "${LOG_DIR}"

cat >"${WORKER}" <<EOF
#!/usr/bin/env bash
set -u
cd '${ROOT}'
PYARROW_PKG='${PYARROW_PKG}'
CACHE_LIBS=\$(find /home/magaras/.cache/rattler/cache/pkgs -type d -path '*/lib' -print | paste -sd: -)
STATUS=0

run_control() {
  local name=\$1 model_config=\$2 checkpoint=\$3 frozen=\$4
  local run_dir='${ROOT}/results/${PROJECT_NAME}'/\${name}
  local log='${LOG_DIR}'/\${name}.log
  if [[ -f "\${run_dir}/SUCCESS.txt" ]]; then
    echo "\$(date): skipping completed \${name}"
    return 0
  fi
  local -a checkpoint_override
  if [[ -n "\${checkpoint}" ]]; then
    checkpoint_override=("model.backbone_ckpt_path=\${checkpoint}")
  else
    checkpoint_override=("model.backbone_ckpt_path=null")
  fi
  echo "\$(date): starting \${name} on GPU ${GPU}"
  CUDA_VISIBLE_DEVICES='${GPU}' HYDRA_FULL_ERROR=1 MPLBACKEND=Agg \
  WANDB_MODE=\$([[ '${LOGGER_OFFLINE}' == true ]] && echo offline || echo online) \
  WANDB_INIT_TIMEOUT=300 WANDB_SILENT=true \
  PYTHONPATH="\${PYARROW_PKG}/lib/python3.11/site-packages\${PYTHONPATH:+:\${PYTHONPATH}}" \
  LD_LIBRARY_PATH="\${CACHE_LIBS}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}" \
  /root/.pixi/bin/pixi run python scripts/train.py \
    datamodule=token_parquet_grouped_classification \
    model="\${model_config}" callbacks=grouped_classification \
    project_name='${PROJECT_NAME}' network_name="\${name}" \
    output_dir='${ROOT}/results' logger.offline='${LOGGER_OFFLINE}' \
    datamodule.prepared_dir='${PREPARED_DIR}' \
    datamodule.batch_size='${BATCH_SIZE}' datamodule.num_workers='${NUM_WORKERS}' \
    datamodule.stream_batch_size=4096 datamodule.shuffle_buffer_size=8192 \
    datamodule.persistent_workers=false \
    "\${checkpoint_override[@]}" model.freeze_backbone="\${frozen}" \
    trainer.max_epochs='${EPOCHS}' trainer.check_val_every_n_epoch=1 \
    trainer.val_check_interval=1.0 trainer.limit_val_batches=1.0 \
    +trainer.log_every_n_steps=50 trainer.devices=1 \
    2>&1 | tee "\${log}"
  rc=\${PIPESTATUS[0]}
  if [[ \${rc} -ne 0 ]]; then
    STATUS=1
    echo "\$(date): FAILED \${name} with exit code \${rc}"
    [[ '${CONTINUE_ON_FAILURE}' == 1 ]] || exit "\${rc}"
  else
    echo "\$(date): finished \${name}"
  fi
}

if [[ '${POOLING}' == all || '${POOLING}' == cls ]]; then
  run_control pretrained_frozen_cls_mlp foundation_grouped_cls_mlp_classifier '${BACKBONE_CKPT}' true
  run_control random_frozen_cls_mlp foundation_grouped_cls_mlp_classifier '' true
  run_control pretrained_finetuned_cls_mlp foundation_grouped_cls_mlp_classifier '${BACKBONE_CKPT}' false
  run_control random_finetuned_cls_mlp foundation_grouped_cls_mlp_classifier '' false
fi
if [[ '${POOLING}' == all || '${POOLING}' == mean ]]; then
  run_control pretrained_frozen_mean_mlp foundation_grouped_mean_mlp_classifier '${BACKBONE_CKPT}' true
  run_control random_frozen_mean_mlp foundation_grouped_mean_mlp_classifier '' true
  run_control pretrained_finetuned_mean_mlp foundation_grouped_mean_mlp_classifier '${BACKBONE_CKPT}' false
  run_control random_finetuned_mean_mlp foundation_grouped_mean_mlp_classifier '' false
fi
exit "\${STATUS}"
EOF
chmod +x "${WORKER}"

echo "Grouped HZZ two-layer MLP pooling controls"
echo "  GPU: ${GPU}"
echo "  pooling: ${POOLING}"
echo "  epochs: ${EPOCHS} (full train and validation sets)"
echo "  dataset: ${PREPARED_DIR}"
echo "  checkpoint: ${BACKBONE_CKPT}"
echo "  worker: ${WORKER}"
if [[ "${DRY_RUN}" == 1 ]]; then echo "DRY RUN: would submit ${UNIT}.service"; exit 0; fi

systemd-run --unit="${UNIT}" --collect --property=WorkingDirectory="${ROOT}" /bin/bash "${WORKER}"
echo "Submitted ${UNIT}.service"
echo "  progress: journalctl -u ${UNIT}.service -f -o cat"
