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
if [ "${#H5_FILES[@]}" -eq 0 ]; then
  echo "No H5 files found. Check DATA_DIR=$DATA_DIR"
  exit 1
fi

H5_LIST=$(IFS=,; echo "[${H5_FILES[*]}]")

OBJ=jets
GPU=2
CODEBOOK_SIZE=16384
NUM_QUANTIZERS=4
CODEBOOK_DIM=16
RUN_NAME="jets_cb16384_q4_cd16_datainit_only"

echo "Starting ${RUN_NAME} on GPU ${GPU}"

CUDA_VISIBLE_DEVICES=$GPU /root/.pixi/bin/pixi run python scripts/train.py \
  datamodule=atlas_event_object \
  model=vqvae \
  callbacks=event_tokenizer \
  output_dir="$RESULTS" \
  project_name=atlas_event_tokenizers_datainit_tests \
  network_name="$RUN_NAME" \
  logger.offline=false \
  trainer.max_epochs=20 \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.limit_val_batches=100 \
  datamodule.object_type="$OBJ" \
  model.codebook_size=$CODEBOOK_SIZE \
  model.num_quantizers=$NUM_QUANTIZERS \
  model.codebook_dim=$CODEBOOK_DIM \
  model.dead_code_reset=false \
  model.data_codebook_init=true \
  model.data_codebook_init_samples=65536 \
  "datamodule.data_paths=${H5_LIST}" \
  > "${LOGDIR}/${RUN_NAME}.log" 2>&1

echo "Finished ${RUN_NAME}"
