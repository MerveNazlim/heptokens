#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/magaras/heptok_fork/heptokens}
WAIT_FOR_UNIT=${WAIT_FOR_UNIT-}
GPU=${GPU:-3}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-3}
VOCAB_SIZE=${VOCAB_SIZE:-131588}
PROJECT_NAME=${PROJECT_NAME:-atlas_event_foundation_pretrain}
NETWORK_NAME=${NETWORK_NAME:-mcdata_seq256_native_full_offline}
LOGGER_OFFLINE=${LOGGER_OFFLINE:-true}
PYARROW_PKG=${PYARROW_PKG:-/home/magaras/.cache/rattler/cache/pkgs/pyarrow-core-16.1.0-py311hf1d6e26_2_cpu}

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
UNIT_NAME="atlas-event-foundation-pretrain-gpu${GPU}-${TIMESTAMP}"
TMP_DIR="${PROJECT_ROOT}/results/tmp/atlas_event_foundation_pretrain"
WORKER="${TMP_DIR}/${UNIT_NAME}.sh"

mkdir -p "${TMP_DIR}"

cat >"${WORKER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail

WAIT_FOR_UNIT='${WAIT_FOR_UNIT}'
if [ -n "\${WAIT_FOR_UNIT}" ]; then
  while systemctl is-active --quiet "\${WAIT_FOR_UNIT}"; do
    echo "\$(date): waiting for \${WAIT_FOR_UNIT}"
    sleep 60
  done
  echo "\$(date): \${WAIT_FOR_UNIT} is no longer active"
fi

echo "\$(date): starting full pretraining on GPU ${GPU}"
cd '${PROJECT_ROOT}'

PYARROW_PKG='${PYARROW_PKG}'
CACHE_LIBS=\$(
  find /home/magaras/.cache/rattler/cache/pkgs \
    -type d -path '*/lib' -print |
  paste -sd: -
)

CUDA_VISIBLE_DEVICES='${GPU}' \
HYDRA_FULL_ERROR=1 \
WANDB_MODE=\$([ '${LOGGER_OFFLINE}' = true ] && echo offline || echo online) \
WANDB_INIT_TIMEOUT=300 \
WANDB_SILENT=true \
PYTHONPATH="\${PYARROW_PKG}/lib/python3.11/site-packages\${PYTHONPATH:+:\${PYTHONPATH}}" \
LD_LIBRARY_PATH="\${CACHE_LIBS}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}" \
/root/.pixi/bin/pixi run python scripts/train.py \
  datamodule=token_parquet_pretrain \
  model=foundation_pretrain \
  callbacks=pretrain \
  project_name='${PROJECT_NAME}' \
  network_name='${NETWORK_NAME}' \
  logger.offline='${LOGGER_OFFLINE}' \
  output_dir='${PROJECT_ROOT}/results' \
  'datamodule.parquet_files=[${PROJECT_ROOT}/results/event_tokens_mc_data_event_context/event_tokens_signal_full_seq256.parquet,${PROJECT_ROOT}/results/event_tokens_mc_data_event_context/event_tokens_background_full_seq256.parquet,${PROJECT_ROOT}/results/event_tokens_mc_data_event_context/event_tokens_data_full_seq256.parquet]' \
  datamodule.batch_size='${BATCH_SIZE}' \
  datamodule.num_workers='${NUM_WORKERS}' \
  model.max_seq_length=256 \
  model.vocab_size='${VOCAB_SIZE}' \
  trainer.max_epochs='${EPOCHS}' \
  trainer.check_val_every_n_epoch=1 \
  trainer.val_check_interval=1.0 \
  trainer.limit_val_batches=500 \
  +trainer.log_every_n_steps=50 \
  trainer.devices=1
EOF

chmod +x "${WORKER}"

systemd-run \
  --unit="${UNIT_NAME}" \
  --description="Full event foundation pretraining on GPU ${GPU}" \
  --collect \
  /bin/bash "${WORKER}"

echo
echo "Submitted ${UNIT_NAME}.service"
if [ -n "${WAIT_FOR_UNIT}" ]; then
  echo "  waits for: ${WAIT_FOR_UNIT}"
else
  echo "  starts: immediately"
fi
echo "  GPU: ${GPU}"
echo "  W&B offline: ${LOGGER_OFFLINE}"
echo "  worker: ${WORKER}"
echo "  progress: journalctl -u ${UNIT_NAME}.service -f"
