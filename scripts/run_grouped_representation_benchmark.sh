#!/usr/bin/env bash
set -euo pipefail

# Controlled representation benchmark:
#   REPRESENTATION=vq|flat|hierarchical|decoded_q8
#   STAGE=pretrain|scratch|finetune
# Required directories are prepared paired Parquet shards. Continuous runs need
# rows exported with --write-continuous-features.

REPRESENTATION=${REPRESENTATION:?Set REPRESENTATION=vq, flat, hierarchical, or decoded_q8}
STAGE=${STAGE:?Set STAGE=pretrain, scratch, or finetune}
OUTPUT_DIR=${OUTPUT_DIR:-results}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-3}
DEVICES=${DEVICES:-1}
ACCELERATOR=${ACCELERATOR:-auto}
LOGGER_OFFLINE=${LOGGER_OFFLINE:-true}
FREEZE_BACKBONE=${FREEZE_BACKBONE:-false}
PROJECT_NAME=${PROJECT_NAME:-grouped_representation_benchmark}
PYTHON_BIN=${PYTHON_BIN:-python}
SEED=${SEED:-42}
PRETRAIN_CALLBACKS=${PRETRAIN_CALLBACKS:-representation_pretrain}

# Hydra changes into ${full_path} before training. Keep full_path absolute so
# callback paths are not resolved relative to the run directory a second time.
if [[ "$OUTPUT_DIR" != /* ]]; then
  OUTPUT_DIR="$PWD/$OUTPUT_DIR"
fi

case "$REPRESENTATION" in
  vq)
    PRETRAIN_MODEL=foundation_grouped_pretrain
    CLASSIFIER_MODEL=foundation_grouped_cls_mlp_classifier
    PRETRAIN_DM=token_parquet_pretrain
    CLASSIFIER_DM=token_parquet_grouped_classification
    ;;
  flat)
    PRETRAIN_MODEL=foundation_continuous_flat_pretrain
    CLASSIFIER_MODEL=foundation_continuous_flat_cls_mlp_classifier
    PRETRAIN_DM=token_parquet_continuous_pretrain
    CLASSIFIER_DM=token_parquet_grouped_continuous_classification
    ;;
  hierarchical)
    PRETRAIN_MODEL=foundation_continuous_hierarchical_pretrain
    CLASSIFIER_MODEL=foundation_continuous_hierarchical_cls_mlp_classifier
    PRETRAIN_DM=token_parquet_continuous_pretrain
    CLASSIFIER_DM=token_parquet_grouped_continuous_classification
    ;;
  decoded_q8)
    PRETRAIN_MODEL=foundation_continuous_flat_pretrain
    CLASSIFIER_MODEL=foundation_continuous_flat_cls_mlp_classifier
    PRETRAIN_DM=token_parquet_decoded_q8_pretrain
    CLASSIFIER_DM=token_parquet_grouped_decoded_q8_classification
    ;;
  *)
    echo "Unknown REPRESENTATION=$REPRESENTATION" >&2
    exit 2
    ;;
esac

COMMON=(
  "output_dir=$OUTPUT_DIR"
  "logger.offline=$LOGGER_OFFLINE"
  "trainer.max_epochs=$EPOCHS"
  "trainer.devices=$DEVICES"
  "trainer.accelerator=$ACCELERATOR"
  "seed=$SEED"
)
if [[ -n "${CKPT_PATH:-}" ]]; then
  COMMON+=("ckpt_path=$CKPT_PATH")
fi
if [[ -n "${MAX_SEQUENCES:-}" ]]; then
  COMMON+=("datamodule.max_sequences=$MAX_SEQUENCES")
fi
if [[ -n "${LIMIT_TRAIN_BATCHES:-}" ]]; then
  COMMON+=("+trainer.limit_train_batches=$LIMIT_TRAIN_BATCHES")
fi
if [[ -n "${LIMIT_VAL_BATCHES:-}" ]]; then
  COMMON+=("trainer.limit_val_batches=$LIMIT_VAL_BATCHES")
fi
if [[ -n "${VAL_CHECK_INTERVAL:-}" ]]; then
  COMMON+=("trainer.val_check_interval=$VAL_CHECK_INTERVAL")
fi
if [[ -n "${CHECK_VAL_EVERY_N_EPOCH:-}" ]]; then
  COMMON+=("trainer.check_val_every_n_epoch=$CHECK_VAL_EVERY_N_EPOCH")
fi
if [[ -n "${LOG_EVERY_N_STEPS:-}" ]]; then
  COMMON+=("+trainer.log_every_n_steps=$LOG_EVERY_N_STEPS")
fi
if [[ -n "${MAX_QUANTIZERS:-}" ]]; then
  COMMON+=("model.max_quantizers=$MAX_QUANTIZERS")
fi
if [[ "$STAGE" == "pretrain" && -n "${CHECKPOINT_EVERY_N_STEPS:-}" ]]; then
  COMMON+=(
    "callbacks.periodic_checkpoint.every_n_train_steps=$CHECKPOINT_EVERY_N_STEPS"
  )
fi
if [[ "$DEVICES" -gt 1 ]]; then
  # Object/group-specific projections may be absent from an individual batch.
  COMMON+=("+trainer.strategy=ddp_find_unused_parameters_true")
fi

case "$STAGE" in
  pretrain)
    PREPARED_PRETRAIN_DIR=${PREPARED_PRETRAIN_DIR:?Set PREPARED_PRETRAIN_DIR}
    NETWORK_NAME=${NETWORK_NAME:-${REPRESENTATION}_masked_pretrain}
    exec "$PYTHON_BIN" scripts/benchmark_train.py \
      "datamodule=$PRETRAIN_DM" \
      "model=$PRETRAIN_MODEL" \
      "callbacks=$PRETRAIN_CALLBACKS" \
      "project_name=$PROJECT_NAME" \
      "network_name=$NETWORK_NAME" \
      "datamodule.prepared_dir=$PREPARED_PRETRAIN_DIR" \
      "datamodule.batch_size=$BATCH_SIZE" \
      "datamodule.num_workers=$NUM_WORKERS" \
      "${COMMON[@]}"
    ;;
  scratch|finetune)
    PREPARED_CLASSIFICATION_DIR=${PREPARED_CLASSIFICATION_DIR:?Set PREPARED_CLASSIFICATION_DIR}
    NETWORK_NAME=${NETWORK_NAME:-${REPRESENTATION}_${STAGE}_cls_mlp}
    MODEL_OVERRIDES=("model.freeze_backbone=$FREEZE_BACKBONE")
    if [[ "$STAGE" == "finetune" ]]; then
      PRETRAIN_CKPT=${PRETRAIN_CKPT:?Set PRETRAIN_CKPT for finetune}
      MODEL_OVERRIDES+=("model.backbone_ckpt_path=$PRETRAIN_CKPT")
    else
      MODEL_OVERRIDES+=("model.backbone_ckpt_path=null")
    fi
    exec "$PYTHON_BIN" scripts/benchmark_train.py \
      "datamodule=$CLASSIFIER_DM" \
      "model=$CLASSIFIER_MODEL" \
      "callbacks=grouped_classification" \
      "project_name=$PROJECT_NAME" \
      "network_name=$NETWORK_NAME" \
      "datamodule.prepared_dir=$PREPARED_CLASSIFICATION_DIR" \
      "datamodule.batch_size=$BATCH_SIZE" \
      "datamodule.num_workers=$NUM_WORKERS" \
      "${MODEL_OVERRIDES[@]}" \
      "${COMMON[@]}"
    ;;
  *)
    echo "Unknown STAGE=$STAGE" >&2
    exit 2
    ;;
esac
