#!/usr/bin/env bash
set -euo pipefail

# One-seed, short end-to-end pilot for all four representations. Run this from
# the repository root inside the same Pixi environment used for production.

PREPARED_PRETRAIN_DIR=${PREPARED_PRETRAIN_DIR:?Set PREPARED_PRETRAIN_DIR}
PREPARED_CLASSIFICATION_DIR=${PREPARED_CLASSIFICATION_DIR:?Set PREPARED_CLASSIFICATION_DIR}
OUTPUT_DIR=${OUTPUT_DIR:-results}
PROJECT_NAME=${PROJECT_NAME:-grouped_representation_pilot}
SEED=${SEED:-42}
EPOCHS=${EPOCHS:-1}
MAX_SEQUENCES=${MAX_SEQUENCES:-20000}
LIMIT_TRAIN_BATCHES=${LIMIT_TRAIN_BATCHES:-100}
LIMIT_VAL_BATCHES=${LIMIT_VAL_BATCHES:-20}
BATCH_SIZE=${BATCH_SIZE:-256}
# Tiny prepared pilots can contain only one row group. One worker ensures that
# the complete capped dataset is consumed instead of assigning no work to
# additional workers.
NUM_WORKERS=${NUM_WORKERS:-1}
DEVICES=${DEVICES:-1}
VAL_CHECK_INTERVAL=${VAL_CHECK_INTERVAL:-1.0}
CHECK_VAL_EVERY_N_EPOCH=${CHECK_VAL_EVERY_N_EPOCH:-1}
LOG_EVERY_N_STEPS=${LOG_EVERY_N_STEPS:-1}
ALLOW_PRETRAIN_DOWNSTREAM_OVERLAP=${ALLOW_PRETRAIN_DOWNSTREAM_OVERLAP:-false}
PYTHON_BIN=${PYTHON_BIN:-python}
AUDIT_TEMP_DIR=${AUDIT_TEMP_DIR:-$OUTPUT_DIR/$PROJECT_NAME/audit_tmp}

mkdir -p "$AUDIT_TEMP_DIR"

"$PYTHON_BIN" scripts/smoke_test_continuous_foundation.py
"$PYTHON_BIN" scripts/validate_paired_representation_shards.py \
  "$PREPARED_PRETRAIN_DIR" "$PREPARED_CLASSIFICATION_DIR" \
  --max-files 12 --rows-per-file 256 --require-decoded-q8

AUDIT_ARGS=(
  --pretrain-dir "$PREPARED_PRETRAIN_DIR"
  --classification-dir "$PREPARED_CLASSIFICATION_DIR"
  --output "$OUTPUT_DIR/$PROJECT_NAME/data_audit.json"
  --temp-dir "$AUDIT_TEMP_DIR"
)
if [[ "$ALLOW_PRETRAIN_DOWNSTREAM_OVERLAP" == "true" ]]; then
  AUDIT_ARGS+=(--allow-pretrain-downstream-overlap)
fi
"$PYTHON_BIN" scripts/audit_representation_benchmark_data.py "${AUDIT_ARGS[@]}"

common_env=(
  "OUTPUT_DIR=$OUTPUT_DIR"
  "PROJECT_NAME=$PROJECT_NAME"
  "SEED=$SEED"
  "EPOCHS=$EPOCHS"
  "MAX_SEQUENCES=$MAX_SEQUENCES"
  "LIMIT_TRAIN_BATCHES=$LIMIT_TRAIN_BATCHES"
  "LIMIT_VAL_BATCHES=$LIMIT_VAL_BATCHES"
  "BATCH_SIZE=$BATCH_SIZE"
  "NUM_WORKERS=$NUM_WORKERS"
  "DEVICES=$DEVICES"
  "VAL_CHECK_INTERVAL=$VAL_CHECK_INTERVAL"
  "CHECK_VAL_EVERY_N_EPOCH=$CHECK_VAL_EVERY_N_EPOCH"
  "LOG_EVERY_N_STEPS=$LOG_EVERY_N_STEPS"
  "LOGGER_OFFLINE=true"
)

checkpoint_for() {
  local network=$1
  local preferred=$2
  local path="$OUTPUT_DIR/$PROJECT_NAME/$network/checkpoints/$preferred.ckpt"
  if [[ ! -s "$path" ]]; then
    path="$OUTPUT_DIR/$PROJECT_NAME/$network/checkpoints/last.ckpt"
  fi
  test -s "$path" || {
    echo "Missing checkpoint for $network" >&2
    return 1
  }
  printf '%s\n' "$path"
}

representations=(vq flat hierarchical decoded_q8)
evaluation_args=()
for representation in "${representations[@]}"; do
  pretrain_network="${representation}_pilot_pretrain"
  scratch_network="${representation}_pilot_no_event_pretraining"
  finetune_network="${representation}_pilot_pretrained_finetune"

  env "${common_env[@]}" \
    REPRESENTATION="$representation" \
    STAGE=pretrain \
    NETWORK_NAME="$pretrain_network" \
    PREPARED_PRETRAIN_DIR="$PREPARED_PRETRAIN_DIR" \
    bash scripts/run_grouped_representation_benchmark.sh
  pretrain_checkpoint=$(checkpoint_for "$pretrain_network" last)

  env "${common_env[@]}" \
    REPRESENTATION="$representation" \
    STAGE=scratch \
    NETWORK_NAME="$scratch_network" \
    PREPARED_CLASSIFICATION_DIR="$PREPARED_CLASSIFICATION_DIR" \
    bash scripts/run_grouped_representation_benchmark.sh
  scratch_checkpoint=$(checkpoint_for "$scratch_network" best)

  env "${common_env[@]}" \
    REPRESENTATION="$representation" \
    STAGE=finetune \
    NETWORK_NAME="$finetune_network" \
    PREPARED_CLASSIFICATION_DIR="$PREPARED_CLASSIFICATION_DIR" \
    PRETRAIN_CKPT="$pretrain_checkpoint" \
    bash scripts/run_grouped_representation_benchmark.sh
  finetune_checkpoint=$(checkpoint_for "$finetune_network" best)

  evaluation_args+=(
    --run "${representation}_no_event_pretraining=${representation}=${scratch_checkpoint}"
    --run "${representation}_masked_pretraining=${representation}=${finetune_checkpoint}"
  )
done

"$PYTHON_BIN" scripts/evaluate_grouped_representation_benchmark.py \
  --prepared-dir "$PREPARED_CLASSIFICATION_DIR" \
  --output-dir "$OUTPUT_DIR/$PROJECT_NAME/evaluation" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --max-sequences "$MAX_SEQUENCES" \
  "${evaluation_args[@]}"

echo "PASS: one-seed representation pilot completed"
