#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/magaras/heptok_fork/heptokens}
AUTOREGRESSIVE_GPU=${AUTOREGRESSIVE_GPU:-0}
PARALLEL_GPU=${PARALLEL_GPU:-3}
EPOCHS=${EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-3}
STREAM_BATCH_SIZE=${STREAM_BATCH_SIZE:-4096}
SHUFFLE_BUFFER_SIZE=${SHUFFLE_BUFFER_SIZE:-8192}
LIMIT_VAL_BATCHES=${LIMIT_VAL_BATCHES:-500}
PREPARED_DIR=${PREPARED_DIR:-${PROJECT_ROOT}/results/event_tokens_grouped_cls_final_new_mcdata_shards}
PROJECT_NAME=${PROJECT_NAME:-atlas_grouped_decoder_comparison}
LOGGER_OFFLINE=${LOGGER_OFFLINE:-true}
PYARROW_PKG=${PYARROW_PKG:-/home/magaras/.cache/rattler/cache/pkgs/pyarrow-core-16.1.0-py311hf1d6e26_2_cpu}
DRY_RUN=${DRY_RUN:-0}

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
TMP_DIR=${PROJECT_ROOT}/results/tmp/${PROJECT_NAME}/${TIMESTAMP}
LOG_DIR=${PROJECT_ROOT}/results/logs/${PROJECT_NAME}/${TIMESTAMP}
mkdir -p "${TMP_DIR}" "${LOG_DIR}"

if [ ! -f "${PREPARED_DIR}/manifest.json" ]; then
  echo "Missing prepared dataset manifest: ${PREPARED_DIR}/manifest.json" >&2
  exit 1
fi

submit_control() {
  local decoder=$1
  local gpu=$2
  local model_config=foundation_grouped_pretrain
  if [ "${decoder}" = autoregressive ]; then
    model_config=foundation_grouped_autoregressive_pretrain
  fi
  local unit="atlas-grouped-${decoder}-gpu${gpu}-${TIMESTAMP}"
  local network_name="grouped_cls_${decoder}_${EPOCHS}epochs_${TIMESTAMP}"
  local worker="${TMP_DIR}/${unit}.sh"
  local log_file="${LOG_DIR}/${decoder}.log"

  cat >"${worker}" <<EOF
#!/usr/bin/env bash
set -euo pipefail

cd '${PROJECT_ROOT}'

PYARROW_PKG='${PYARROW_PKG}'
CACHE_LIBS=\$(
  find /home/magaras/.cache/rattler/cache/pkgs \
    -type d -path '*/lib' -print | paste -sd: -
)

echo "\$(date): starting ${decoder} grouped pretraining on GPU ${gpu}"

CUDA_VISIBLE_DEVICES='${gpu}' \
WANDB_MODE=\$([ '${LOGGER_OFFLINE}' = true ] && echo offline || echo online) \
WANDB_INIT_TIMEOUT=300 \
WANDB_SILENT=true \
HYDRA_FULL_ERROR=1 \
PYTHONPATH='${PROJECT_ROOT}/src':"\${PYARROW_PKG}/lib/python3.11/site-packages\${PYTHONPATH:+:\${PYTHONPATH}}" \
LD_LIBRARY_PATH="\${CACHE_LIBS}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}" \
/usr/bin/time -v \
/root/.pixi/bin/pixi run python scripts/train.py \
  datamodule=token_parquet_pretrain \
  model='${model_config}' \
  callbacks=pretrain \
  project_name='${PROJECT_NAME}' \
  network_name='${network_name}' \
  output_dir='${PROJECT_ROOT}/results' \
  datamodule.prepared_dir='${PREPARED_DIR}' \
  datamodule.batch_size='${BATCH_SIZE}' \
  datamodule.num_workers='${NUM_WORKERS}' \
  datamodule.stream_batch_size='${STREAM_BATCH_SIZE}' \
  datamodule.shuffle_buffer_size='${SHUFFLE_BUFFER_SIZE}' \
  datamodule.persistent_workers=false \
  model.max_seq_length=256 \
  model.max_quantizers=8 \
  trainer.max_epochs='${EPOCHS}' \
  trainer.check_val_every_n_epoch=1 \
  trainer.val_check_interval=1.0 \
  trainer.limit_val_batches='${LIMIT_VAL_BATCHES}' \
  trainer.enable_progress_bar=false \
  +trainer.log_every_n_steps=50 \
  trainer.devices=1 \
  logger.offline='${LOGGER_OFFLINE}' \
  2>&1 | tee '${log_file}'

echo "\$(date): finished ${decoder} grouped pretraining"
EOF

  chmod +x "${worker}"

  if [ "${DRY_RUN}" -eq 1 ]; then
    echo "DRY RUN: would submit ${unit}.service"
  else
    systemd-run \
      --unit="${unit}" \
      --description="Grouped ${decoder} decoder comparison on GPU ${gpu}" \
      --collect \
      /bin/bash "${worker}"
    echo "Submitted ${unit}.service"
  fi

  echo "  decoder: ${decoder}"
  echo "  GPU: ${gpu}"
  echo "  run: ${PROJECT_ROOT}/results/${PROJECT_NAME}/${network_name}"
  echo "  log: ${log_file}"
  echo "  worker: ${worker}"
  echo "  journal: journalctl -u ${unit}.service -f -o cat"
  echo
}

echo "Matched grouped-decoder comparison"
echo "  epochs: ${EPOCHS}"
echo "  batch size: ${BATCH_SIZE} per model"
echo "  prepared dataset: ${PREPARED_DIR}"
echo "  autoregressive GPU: ${AUTOREGRESSIVE_GPU}"
echo "  parallel GPU: ${PARALLEL_GPU}"
echo "  W&B offline: ${LOGGER_OFFLINE}"
echo "  logs: ${LOG_DIR}"
echo

submit_control autoregressive "${AUTOREGRESSIVE_GPU}"
submit_control parallel "${PARALLEL_GPU}"

echo "Both controls submitted independently."
echo "Progress metrics are written to each run's live_metrics.csv."
