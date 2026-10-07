#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/magaras/heptok_fork/heptokens}
GPU=${GPU:-3}
BATCH_SIZES=${BATCH_SIZES:-"64 128 256"}
TRAIN_BATCHES=${TRAIN_BATCHES:-200}
NUM_WORKERS=${NUM_WORKERS:-3}
STREAM_BATCH_SIZE=${STREAM_BATCH_SIZE:-4096}
SHUFFLE_BUFFER_SIZE=${SHUFFLE_BUFFER_SIZE:-8192}
VOCAB_SIZE=${VOCAB_SIZE:-131588}
MAX_QUANTIZERS=${MAX_QUANTIZERS:-8}
PREPARED_DIR=${PREPARED_DIR:-${PROJECT_ROOT}/results/event_tokens_grouped_final_new_mcdata/pretrain_shards_train90_val10_full}
PROJECT_NAME=${PROJECT_NAME:-atlas_foundation_batch_benchmark}
PYARROW_PKG=${PYARROW_PKG:-/home/magaras/.cache/rattler/cache/pkgs/pyarrow-core-16.1.0-py311hf1d6e26_2_cpu}

STAMP=$(date +%Y%m%d-%H%M%S)
LOG_DIR=${LOG_DIR:-${PROJECT_ROOT}/results/logs/${PROJECT_NAME}/${STAMP}}
mkdir -p "${LOG_DIR}"

cd "${PROJECT_ROOT}"

CACHE_LIBS=$(
  find /home/magaras/.cache/rattler/cache/pkgs \
    -type d -path '*/lib' -print | paste -sd: -
)
export PYTHONPATH="${PROJECT_ROOT}/src:${PYARROW_PKG}/lib/python3.11/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${CACHE_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

if [ ! -f "${PREPARED_DIR}/manifest.json" ]; then
  echo "Missing prepared dataset manifest: ${PREPARED_DIR}/manifest.json" >&2
  exit 1
fi

echo "Grouped foundation batch-size benchmark"
echo "  GPU: ${GPU}"
echo "  batch sizes: ${BATCH_SIZES}"
echo "  train batches per run: ${TRAIN_BATCHES}"
echo "  prepared dataset: ${PREPARED_DIR}"
echo "  logs: ${LOG_DIR}"

for batch_size in ${BATCH_SIZES}; do
  run_name="grouped_batch${batch_size}_${TRAIN_BATCHES}steps_${STAMP}"
  train_log="${LOG_DIR}/${run_name}.log"
  gpu_log="${LOG_DIR}/${run_name}_gpu_memory.csv"

  echo
  echo "$(date): testing batch size ${batch_size}"

  (
    while true; do
      printf '%s,' "$(date +%s)"
      nvidia-smi \
        --id="${GPU}" \
        --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits
      sleep 1
    done
  ) >"${gpu_log}" 2>/dev/null &
  monitor_pid=$!

  start_time=$(date +%s)
  set +e
  CUDA_VISIBLE_DEVICES="${GPU}" \
  WANDB_MODE=offline \
  HYDRA_FULL_ERROR=1 \
  /usr/bin/time -v \
  /root/.pixi/bin/pixi run python scripts/train.py \
    datamodule=token_parquet_pretrain \
    model=foundation_grouped_pretrain \
    callbacks=pretrain \
    project_name="${PROJECT_NAME}" \
    network_name="${run_name}" \
    output_dir="${PROJECT_ROOT}/results" \
    datamodule.prepared_dir="${PREPARED_DIR}" \
    datamodule.batch_size="${batch_size}" \
    datamodule.num_workers="${NUM_WORKERS}" \
    datamodule.stream_batch_size="${STREAM_BATCH_SIZE}" \
    datamodule.shuffle_buffer_size="${SHUFFLE_BUFFER_SIZE}" \
    datamodule.persistent_workers=false \
    model.max_seq_length=256 \
    model.max_quantizers="${MAX_QUANTIZERS}" \
    model.vocab_size="${VOCAB_SIZE}" \
    trainer.max_epochs=1 \
    +trainer.limit_train_batches="${TRAIN_BATCHES}" \
    trainer.limit_val_batches=0 \
    +trainer.log_every_n_steps=50 \
    +trainer.enable_checkpointing=false \
    '~callbacks.checkpoint_per_epoch' \
    trainer.devices=1 \
    logger.offline=true \
    2>&1 | tee "${train_log}"
  status=${PIPESTATUS[0]}
  set -e
  end_time=$(date +%s)

  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true

  elapsed=$((end_time - start_time))
  peak_gpu_mib=$(awk -F, 'BEGIN { max=0 } { gsub(/ /, "", $2); if ($2+0 > max) max=$2+0 } END { print max }' "${gpu_log}")

  if [ "${status}" -eq 0 ]; then
    events_per_second=$(awk -v b="${batch_size}" -v n="${TRAIN_BATCHES}" -v t="${elapsed}" 'BEGIN { if (t > 0) printf "%.1f", b*n/t; else print "n/a" }')
    echo "PASS batch=${batch_size} elapsed=${elapsed}s events/s=${events_per_second} peak_gpu=${peak_gpu_mib}MiB"
  else
    echo "FAIL batch=${batch_size} status=${status} elapsed=${elapsed}s peak_gpu=${peak_gpu_mib}MiB"
    echo "  inspect: ${train_log}"
  fi

  sleep 10
done

echo
echo "Benchmark complete. Logs: ${LOG_DIR}"
