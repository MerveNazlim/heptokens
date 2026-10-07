#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/magaras/heptok_fork/heptokens}
GPU=${GPU:-3}
RUN_SET=${RUN_SET:-all}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-3}
STREAM_BATCH_SIZE=${STREAM_BATCH_SIZE:-4096}
SHUFFLE_BUFFER_SIZE=${SHUFFLE_BUFFER_SIZE:-8192}
LEARNING_RATE=${LEARNING_RATE:-1.0e-4}
LOGGER_OFFLINE=${LOGGER_OFFLINE:-true}
CONTINUE_ON_FAILURE=${CONTINUE_ON_FAILURE:-0}
DRY_RUN=${DRY_RUN:-0}

PROJECT_NAME=${PROJECT_NAME:-atlas_hzz_grouped_mean_pool_classification}
PREPARED_DIR=${PREPARED_DIR:-${PROJECT_ROOT}/results/event_tokens_grouped_final_new_mcdata/hzz_ggf_vs_zz_classification_shards}
BACKBONE_CKPT=${BACKBONE_CKPT:-${PROJECT_ROOT}/results/atlas_event_foundation_pretrain/grouped_new_mcdata_seq256_ddp_equalized_v2/checkpoints/last.ckpt}
PYARROW_PKG=${PYARROW_PKG:-/home/magaras/.cache/rattler/cache/pkgs/pyarrow-core-16.1.0-py311hf1d6e26_2_cpu}

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
UNIT_NAME="atlas-hzz-mean-classification-gpu${GPU}-${TIMESTAMP}"
TMP_DIR="${PROJECT_ROOT}/results/tmp/${PROJECT_NAME}"
LOG_DIR="${PROJECT_ROOT}/results/logs/${PROJECT_NAME}"
WORKER="${TMP_DIR}/${UNIT_NAME}.sh"

if [ "${RUN_SET}" != all ]; then
  IFS=',' read -r -a REQUESTED_RUNS <<<"${RUN_SET}"
  for RUN_NAME in "${REQUESTED_RUNS[@]}"; do
    case "${RUN_NAME}" in
      pretrained_frozen_mean_pool | random_frozen_mean_pool | \
        pretrained_finetuned_mean_pool | random_finetuned_mean_pool) ;;
      *)
        echo "Unknown classification control in RUN_SET: ${RUN_NAME}" >&2
        exit 1
        ;;
    esac
  done
fi

if [ ! -f "${PREPARED_DIR}/manifest.json" ]; then
  echo "Missing prepared classification dataset: ${PREPARED_DIR}" >&2
  exit 1
fi
if [ ! -f "${BACKBONE_CKPT}" ]; then
  echo "Missing foundation checkpoint: ${BACKBONE_CKPT}" >&2
  exit 1
fi

mkdir -p "${TMP_DIR}" "${LOG_DIR}"

cat >"${WORKER}" <<EOF
#!/usr/bin/env bash
set -u

cd '${PROJECT_ROOT}'
mkdir -p '${LOG_DIR}'

PYARROW_PKG='${PYARROW_PKG}'
CACHE_LIBS=\$(
  find /home/magaras/.cache/rattler/cache/pkgs \
    -type d -path '*/lib' -print | paste -sd: -
)

STATUS=0
RUN_SET='${RUN_SET}'

should_run() {
  local RUN_NAME=\$1
  [ "\${RUN_SET}" = all ] || [[ ",\${RUN_SET}," == *",\${RUN_NAME},"* ]]
}

run_control() {
  local RUN_NAME=\$1
  local BACKBONE_PATH=\$2
  local FREEZE_BACKBONE=\$3
  local RUN_LOG='${LOG_DIR}'/\${RUN_NAME}.log
  local RUN_DIR='${PROJECT_ROOT}/results/${PROJECT_NAME}'/\${RUN_NAME}

  echo
  echo "============================================================"
  echo "\$(date): starting \${RUN_NAME} on GPU ${GPU}"
  echo "backbone=\${BACKBONE_PATH:-random initialization}"
  echo "freeze_backbone=\${FREEZE_BACKBONE}"
  echo "log=\${RUN_LOG}"

  if [ -f "\${RUN_DIR}/SUCCESS.txt" ]; then
    echo "\$(date): skipping completed \${RUN_NAME}"
    return 0
  fi

  local -a BACKBONE_OVERRIDE
  if [ -n "\${BACKBONE_PATH}" ]; then
    BACKBONE_OVERRIDE=("model.backbone_ckpt_path=\${BACKBONE_PATH}")
  else
    BACKBONE_OVERRIDE=("model.backbone_ckpt_path=null")
  fi

  CUDA_VISIBLE_DEVICES='${GPU}' \
  HYDRA_FULL_ERROR=1 \
  MPLBACKEND=Agg \
  WANDB_MODE=\$([ '${LOGGER_OFFLINE}' = true ] && echo offline || echo online) \
  WANDB_INIT_TIMEOUT=300 \
  WANDB_SILENT=true \
  PYTHONPATH="\${PYARROW_PKG}/lib/python3.11/site-packages\${PYTHONPATH:+:\${PYTHONPATH}}" \
  LD_LIBRARY_PATH="\${CACHE_LIBS}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}" \
  /root/.pixi/bin/pixi run python scripts/train.py \
    datamodule=token_parquet_grouped_classification \
    model=foundation_grouped_mean_classifier \
    callbacks=grouped_classification \
    project_name='${PROJECT_NAME}' \
    network_name="\${RUN_NAME}" \
    output_dir='${PROJECT_ROOT}/results' \
    logger.offline='${LOGGER_OFFLINE}' \
    datamodule.prepared_dir='${PREPARED_DIR}' \
    datamodule.batch_size='${BATCH_SIZE}' \
    datamodule.num_workers='${NUM_WORKERS}' \
    datamodule.stream_batch_size='${STREAM_BATCH_SIZE}' \
    datamodule.shuffle_buffer_size='${SHUFFLE_BUFFER_SIZE}' \
    datamodule.persistent_workers=false \
    "\${BACKBONE_OVERRIDE[@]}" \
    model.freeze_backbone="\${FREEZE_BACKBONE}" \
    model.learning_rate='${LEARNING_RATE}' \
    model.optimizer.lr='${LEARNING_RATE}' \
    trainer.max_epochs='${EPOCHS}' \
    trainer.check_val_every_n_epoch=1 \
    trainer.val_check_interval=1.0 \
    trainer.limit_val_batches=1.0 \
    +trainer.log_every_n_steps=50 \
    trainer.devices=1 \
    2>&1 | tee "\${RUN_LOG}"

  local RC=\${PIPESTATUS[0]}
  if [ "\${RC}" -ne 0 ]; then
    echo "\$(date): FAILED \${RUN_NAME} with exit code \${RC}"
    STATUS=1
    if [ '${CONTINUE_ON_FAILURE}' -ne 1 ]; then
      exit "\${RC}"
    fi
  else
    echo "\$(date): finished \${RUN_NAME}"
  fi
}

if should_run pretrained_frozen_mean_pool; then
  run_control pretrained_frozen_mean_pool '${BACKBONE_CKPT}' true
fi
if should_run random_frozen_mean_pool; then
  run_control random_frozen_mean_pool '' true
fi
if should_run pretrained_finetuned_mean_pool; then
  run_control pretrained_finetuned_mean_pool '${BACKBONE_CKPT}' false
fi
if should_run random_finetuned_mean_pool; then
  run_control random_finetuned_mean_pool '' false
fi

echo
echo "\$(date): all grouped HZZ classification controls finished with status \${STATUS}"
exit "\${STATUS}"
EOF

chmod +x "${WORKER}"

echo "Grouped HZZ masked-mean classification controls"
echo "  GPU: ${GPU}"
echo "  run set: ${RUN_SET}"
echo "  epochs: ${EPOCHS}"
echo "  batch/workers: ${BATCH_SIZE}/${NUM_WORKERS}"
echo "  learning rate: ${LEARNING_RATE}"
echo "  dataset: ${PREPARED_DIR}"
echo "  checkpoint: ${BACKBONE_CKPT}"
echo "  W&B offline: ${LOGGER_OFFLINE}"
echo "  worker: ${WORKER}"

if [ "${DRY_RUN}" -eq 1 ]; then
  echo "DRY RUN: would submit ${UNIT_NAME}.service"
  exit 0
fi

systemd-run \
  --unit="${UNIT_NAME}" \
  --description="Grouped HZZ masked-mean classification controls on GPU ${GPU}" \
  --collect \
  --property=WorkingDirectory="${PROJECT_ROOT}" \
  /bin/bash "${WORKER}"

echo
echo "Submitted ${UNIT_NAME}.service"
echo "  progress: journalctl -u ${UNIT_NAME}.service -f"
