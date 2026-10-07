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

OBJ=taus
PROJECT=atlas_event_tokenizers_0206_taus_capacity_scan
CODEBOOK_DIM=16

RUN_NAMES=(
  taus_cb8192_q3
  taus_cb16384_q3
  taus_cb16384_q4
  taus_cb32768_q3
)

CODEBOOKS=(8192 16384 16384 32768)
QUANTIZERS=(3 3 4 3)
GPUS=(0 1 2 3)

run_one () {
  NAME=$1
  CODEBOOK_SIZE=$2
  NUM_QUANTIZERS=$3
  GPU=$4

  echo "Starting ${NAME} on GPU ${GPU}"
  echo "  codebook_size=${CODEBOOK_SIZE}, num_quantizers=${NUM_QUANTIZERS}, codebook_dim=${CODEBOOK_DIM}"

  CUDA_VISIBLE_DEVICES=$GPU /root/.pixi/bin/pixi run python scripts/train.py \
    datamodule=atlas_event_object \
    model=vqvae \
    callbacks=event_tokenizer \
    output_dir="$RESULTS" \
    project_name="$PROJECT" \
    network_name="$NAME" \
    logger.offline=false \
    trainer.max_epochs=20 \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    datamodule.object_type="${OBJ}" \
    model.codebook_size=$CODEBOOK_SIZE \
    model.num_quantizers=$NUM_QUANTIZERS \
    model.codebook_dim=$CODEBOOK_DIM \
    "datamodule.data_paths=${H5_LIST}" \
    > "${LOGDIR}/${NAME}.log" 2>&1
}

for i in "${!RUN_NAMES[@]}"; do
  run_one "${RUN_NAMES[$i]}" "${CODEBOOKS[$i]}" "${QUANTIZERS[$i]}" "${GPUS[$i]}" &
done

wait
echo "All tau capacity-scan tokenizers finished"
