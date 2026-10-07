#!/usr/bin/env bash
set -euo pipefail

export HOME=/root
export USER=root
export LOGNAME=root
export WANDB_DIR=/home/magaras/heptok_fork/heptokens/results/wandb_tmp
export WANDB_CACHE_DIR=/home/magaras/heptok_fork/heptokens/results/wandb_cache
export WANDB_CONFIG_DIR=/root/.config/wandb
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_DIR=/home/zephyr/Data/viviana/bnl-treasure/data/h5
REALDATA_DIR=/home/zephyr/Data/viviana/bnl-treasure/data/h5/realdata
LOGDIR=/home/magaras/heptok_fork/heptokens/results/logs/atlas_event_tokenizers_0107_logstd_mc_realdata
mkdir -p "$LOGDIR" "$WANDB_DIR" "$WANDB_CACHE_DIR"

mapfile -t H5_FILES < <(
  {
    find "$DATA_DIR" -maxdepth 1 -type f -name "*.h5" -size +0c
    find "$REALDATA_DIR" -maxdepth 1 -type f -name "*.h5" -size +0c
  } | sort -u
)

echo "Found ${#H5_FILES[@]} H5 files"

if [ "${#H5_FILES[@]}" -eq 0 ]; then
  echo "No H5 files found"
  exit 1
fi

H5_LIST=$(IFS=,; echo "[${H5_FILES[*]}]")

CUDA_VISIBLE_DEVICES=2 /root/.pixi/bin/pixi run python scripts/train.py \
  datamodule=atlas_event_object \
  model=vqvae \
  callbacks=event_tokenizer \
  output_dir=/home/magaras/heptok_fork/heptokens/results \
  project_name=atlas_event_tokenizers_0107_logstd_mc_realdata \
  network_name=tracks_logstd_dim8_cb8192_q4 \
  logger.offline=false \
  trainer.max_epochs=20 \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.limit_val_batches=1000 \
  datamodule.object_type=tracks \
  datamodule.batch_size=256 \
  model.codebook_dim=8 \
  model.codebook_size=8192 \
  model.num_quantizers=4 \
  model.dead_code_reset=false \
  model.data_codebook_init=false \
  +datamodule.transforms.preprocess._target_=heptokens.data.collation.preprocess_objects_batch \
  +datamodule.transforms.preprocess._partial_=true \
  +datamodule.transforms.preprocess.cst_fn._target_=joblib.load \
  +datamodule.transforms.preprocess.cst_fn.filename=/home/magaras/heptok_fork/heptokens/results/preprocessing/atlas_event_tokenizers_0107_logstd_mc_realdata/tracks_log_standard_no_ndoflog.joblib \
  "datamodule.data_paths=${H5_LIST}" \
  > "$LOGDIR/tracks_logstd_dim8_cb8192_q4_rerun.log" 2>&1
