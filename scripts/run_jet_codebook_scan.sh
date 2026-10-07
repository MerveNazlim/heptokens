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

CODEBOOKS=(16384 32768)
GPUS=(0 1)
NUM_QUANTIZERS=4
CODEBOOK_DIM=16
OBJ=jets
PROJECT=atlas_event_tokenizers_0106_jets_capacity_scan

run_one () {
  CODEBOOK_SIZE=$1
  GPU=$2

  echo "Starting jets cb${CODEBOOK_SIZE} q${NUM_QUANTIZERS} on GPU ${GPU}"

  CUDA_VISIBLE_DEVICES=$GPU /root/.pixi/bin/pixi run python scripts/train.py \
    datamodule=atlas_event_object \
    model=vqvae \
    callbacks=event_tokenizer \
    output_dir="$RESULTS" \
    project_name="$PROJECT" \
    network_name="${OBJ}_cb${CODEBOOK_SIZE}_q${NUM_QUANTIZERS}" \
    logger.offline=false \
    trainer.max_epochs=20 \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    datamodule.object_type="${OBJ}" \
    model.codebook_size=$CODEBOOK_SIZE \
    model.num_quantizers=$NUM_QUANTIZERS \
    model.codebook_dim=$CODEBOOK_DIM \
    "datamodule.data_paths=${H5_LIST}" \
    > "${LOGDIR}/${OBJ}_cb${CODEBOOK_SIZE}_q${NUM_QUANTIZERS}.log" 2>&1
}

for i in "${!CODEBOOKS[@]}"; do
  run_one "${CODEBOOKS[$i]}" "${GPUS[$i]}" &
done

wait
echo "All jet q4 scan tokenizers finished"


