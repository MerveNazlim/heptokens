#!/usr/bin/env bash
set -euo pipefail

export HOME=/root
export USER=root
export LOGNAME=root
export WANDB_DIR=/home/magaras/heptok_fork/heptokens/results/wandb_tmp
export WANDB_CACHE_DIR=/home/magaras/heptok_fork/heptokens/results/wandb_cache
export WANDB_CONFIG_DIR=/root/.config/wandb
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"

DATA_DIR=/home/zephyr/Data/viviana/bnl-treasure/data/h5
RESULTS=/home/magaras/heptok_fork/heptokens/results
LOGDIR=${RESULTS}/logs

mkdir -p "$LOGDIR"

mapfile -t H5_FILES < <(
  find "$DATA_DIR" -maxdepth 1 -type f \
    \( -name 'DAOD_PHYSLITE.381*.h5' \
    -o -name 'DAOD_PHYSLITE.371*.h5' \
    -o -name 'DAOD_PHYSLITE.376*.h5' \) \
    | sort
)

echo "Found ${#H5_FILES[@]} H5 files"
printf '%s\n' "${H5_FILES[@]}" | head
printf '%s\n' "${H5_FILES[@]}" | tail

if [ "${#H5_FILES[@]}" -eq 0 ]; then
  echo "No H5 files found. Check DATA_DIR=$DATA_DIR"
  exit 1
fi

H5_LIST=$(IFS=,; echo "[${H5_FILES[*]}]")

# Keep jets alone on GPU 3 because it is the longest job.
OBJECTS=(electrons muons photons taus jets)
GPUS=(0 1 2 0 3)
MAX_JOBS=5

CODEBOOK_DIM=8
VAL_BATCHES=1000
PROJECT_NAME=atlas_event_tokenizers_0506_allfeat_cd8_val1000

run_one () {
  OBJ=$1
  GPU=$2

  case "$OBJ" in
    jets)
      CODEBOOK_SIZE=32768
      NUM_QUANTIZERS=4
      ;;
    electrons|muons|taus)
      CODEBOOK_SIZE=32768
      NUM_QUANTIZERS=3
      ;;
    photons)
      CODEBOOK_SIZE=8192
      NUM_QUANTIZERS=3
      ;;
    *)
      echo "Unknown object type: $OBJ"
      exit 1
      ;;
  esac

  RUN_NAME="${OBJ}_cb${CODEBOOK_SIZE}_q${NUM_QUANTIZERS}_cd${CODEBOOK_DIM}"

  echo "Starting ${RUN_NAME} on GPU ${GPU}"
  echo "  validation batches=${VAL_BATCHES}"

  CUDA_VISIBLE_DEVICES=$GPU /root/.pixi/bin/pixi run python scripts/train.py \
    datamodule=atlas_event_object \
    model=vqvae \
    callbacks=event_tokenizer \
    output_dir="$RESULTS" \
    project_name="$PROJECT_NAME" \
    network_name="$RUN_NAME" \
    logger.offline=false \
    trainer.max_epochs=20 \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    trainer.limit_val_batches=$VAL_BATCHES \
    datamodule.object_type="${OBJ}" \
    model.codebook_size=$CODEBOOK_SIZE \
    model.num_quantizers=$NUM_QUANTIZERS \
    model.codebook_dim=$CODEBOOK_DIM \
    "datamodule.data_paths=${H5_LIST}" \
    > "${LOGDIR}/${RUN_NAME}.log" 2>&1
}

for i in "${!OBJECTS[@]}"; do
  run_one "${OBJECTS[$i]}" "${GPUS[$i]}" &

  while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
    wait -n
  done
done

wait
echo "All tokenizers finished"