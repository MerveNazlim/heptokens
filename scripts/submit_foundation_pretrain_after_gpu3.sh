#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/magaras/heptok_fork/heptokens}
WAIT_FOR_UNIT=${WAIT_FOR_UNIT-}
GPU=${GPU:-3}
GPUS=${GPUS:-${GPU}}
NUM_DEVICES=$(awk -F, '{print NF}' <<<"${GPUS}")
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-3}
VOCAB_SIZE=${VOCAB_SIZE:-131588}
PROJECT_NAME=${PROJECT_NAME:-atlas_event_foundation_pretrain}
NETWORK_NAME=${NETWORK_NAME:-grouped_new_mcdata_seq256_full_sharded_offline}
LOGGER_OFFLINE=${LOGGER_OFFLINE:-true}
TOKEN_PARQUET_DIR=${TOKEN_PARQUET_DIR:-${PROJECT_ROOT}/results/event_tokens_grouped_final_new_mcdata}
SIGNAL_PARQUET=${SIGNAL_PARQUET:-${TOKEN_PARQUET_DIR}/event_tokens_signal_grouped_full_seq256.parquet}
BACKGROUND_PARQUET=${BACKGROUND_PARQUET:-${TOKEN_PARQUET_DIR}/event_tokens_background_grouped_full_seq256.parquet}
DATA_PARQUET=${DATA_PARQUET:-${TOKEN_PARQUET_DIR}/event_tokens_data_grouped_full_seq256.parquet}
PREPARED_DIR=${PREPARED_DIR:-${TOKEN_PARQUET_DIR}/pretrain_shards_train90_val10_full}
READ_BATCH_SIZE=${READ_BATCH_SIZE:-4096}
SHARD_ROWS=${SHARD_ROWS:-50000}
STREAM_BATCH_SIZE=${STREAM_BATCH_SIZE:-4096}
SHUFFLE_BUFFER_SIZE=${SHUFFLE_BUFFER_SIZE:-8192}
MAX_QUANTIZERS=${MAX_QUANTIZERS:-8}
LIMIT_VAL_BATCHES=${LIMIT_VAL_BATCHES:-500}
REBUILD_SHARDS=${REBUILD_SHARDS:-false}
PREPARE_ONLY=${PREPARE_ONLY:-false}
DRY_RUN=${DRY_RUN:-0}
PYARROW_PKG=${PYARROW_PKG:-/home/magaras/.cache/rattler/cache/pkgs/pyarrow-core-16.1.0-py311hf1d6e26_2_cpu}

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
GPU_TAG=${GPUS//,/-}
UNIT_NAME="atlas-event-foundation-pretrain-gpu${GPU_TAG}-${TIMESTAMP}"
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

echo "\$(date): starting sharded foundation-pretraining pipeline"
cd '${PROJECT_ROOT}'

PYARROW_PKG='${PYARROW_PKG}'
CACHE_LIBS=\$(
  find /home/magaras/.cache/rattler/cache/pkgs \
    -type d -path '*/lib' -print |
  paste -sd: -
)

if [ '${REBUILD_SHARDS}' = true ]; then
  PREPARE_OVERWRITE=--overwrite
else
  PREPARE_OVERWRITE=
fi

if [ ! -f '${PREPARED_DIR}/manifest.json' ] || [ '${REBUILD_SHARDS}' = true ]; then
  echo "\$(date): preparing stratified train/validation parquet shards"
  PYTHONPATH="\${PYARROW_PKG}/lib/python3.11/site-packages\${PYTHONPATH:+:\${PYTHONPATH}}" \
  LD_LIBRARY_PATH="\${CACHE_LIBS}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}" \
  /root/.pixi/bin/pixi run python scripts/prepare_token_parquet_pretrain_shards.py \
    --input-parquets \
      '${SIGNAL_PARQUET}' \
      '${BACKGROUND_PARQUET}' \
      '${DATA_PARQUET}' \
    --output-dir '${PREPARED_DIR}' \
    --train-frac 0.9 \
    --seed 42 \
    --read-batch-size '${READ_BATCH_SIZE}' \
    --shard-rows '${SHARD_ROWS}' \
    \${PREPARE_OVERWRITE}
else
  echo "\$(date): reusing prepared shards from ${PREPARED_DIR}"
fi

if [ '${PREPARE_ONLY}' = true ]; then
  echo "\$(date): shard preparation complete; PREPARE_ONLY=true"
  exit 0
fi

echo "\$(date): starting full pretraining on GPUs ${GPUS}"

CUDA_VISIBLE_DEVICES='${GPUS}' \
HYDRA_FULL_ERROR=1 \
WANDB_MODE=\$([ '${LOGGER_OFFLINE}' = true ] && echo offline || echo online) \
WANDB_INIT_TIMEOUT=300 \
WANDB_SILENT=true \
PYTHONPATH="\${PYARROW_PKG}/lib/python3.11/site-packages\${PYTHONPATH:+:\${PYTHONPATH}}" \
LD_LIBRARY_PATH="\${CACHE_LIBS}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}" \
/root/.pixi/bin/pixi run python scripts/train.py \
  datamodule=token_parquet_pretrain \
  model=foundation_grouped_pretrain \
  callbacks=pretrain \
  project_name='${PROJECT_NAME}' \
  network_name='${NETWORK_NAME}' \
  logger.offline='${LOGGER_OFFLINE}' \
  output_dir='${PROJECT_ROOT}/results' \
  datamodule.prepared_dir='${PREPARED_DIR}' \
  datamodule.batch_size='${BATCH_SIZE}' \
  datamodule.num_workers='${NUM_WORKERS}' \
  datamodule.stream_batch_size='${STREAM_BATCH_SIZE}' \
  datamodule.shuffle_buffer_size='${SHUFFLE_BUFFER_SIZE}' \
  datamodule.persistent_workers=false \
  model.max_seq_length=256 \
  model.vocab_size='${VOCAB_SIZE}' \
  model.max_quantizers='${MAX_QUANTIZERS}' \
  trainer.max_epochs='${EPOCHS}' \
  trainer.check_val_every_n_epoch=1 \
  trainer.val_check_interval=1.0 \
  trainer.limit_val_batches='${LIMIT_VAL_BATCHES}' \
  +trainer.log_every_n_steps=50 \
  trainer.devices='${NUM_DEVICES}'
EOF

chmod +x "${WORKER}"

if [ "${DRY_RUN}" -eq 1 ]; then
  echo "DRY RUN: would submit ${UNIT_NAME}.service"
  echo "  worker: ${WORKER}"
  echo "  prepared shards: ${PREPARED_DIR}"
  echo "  prepare only: ${PREPARE_ONLY}"
  exit 0
fi

systemd-run \
  --unit="${UNIT_NAME}" \
  --description="Full grouped event foundation pretraining on GPUs ${GPUS}" \
  --collect \
  /bin/bash "${WORKER}"

echo
echo "Submitted ${UNIT_NAME}.service"
if [ -n "${WAIT_FOR_UNIT}" ]; then
  echo "  waits for: ${WAIT_FOR_UNIT}"
else
  echo "  starts: immediately"
fi
echo "  GPUs: ${GPUS} (${NUM_DEVICES} DDP processes)"
echo "  W&B offline: ${LOGGER_OFFLINE}"
echo "  batch/workers: ${BATCH_SIZE}/${NUM_WORKERS}"
echo "  Arrow batch/shuffle buffer: ${STREAM_BATCH_SIZE}/${SHUFFLE_BUFFER_SIZE}"
echo "  prepared shards: ${PREPARED_DIR}"
echo "  prepare only: ${PREPARE_ONLY}"
echo "  worker: ${WORKER}"
echo "  progress: journalctl -u ${UNIT_NAME}.service -f"
