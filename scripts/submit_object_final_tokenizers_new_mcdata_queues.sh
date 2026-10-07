#!/usr/bin/env bash
set -euo pipefail

# Final candidate object-tokenizer queues on the new MC+data H5 samples.
#
# Each object gets two candidates:
#   - compact:        q4, larger codebook, codebook_dim=16
#   - compositional:  q8, smaller codebook, codebook_dim=8
#
# The q1_full matrix is a single-code control with cb16384/dim8. It keeps the
# codebook embedding parameter count equal to q8/cb2048/dim8 per object.
#
# The two GPU workers run sequential queues. A file lock protects the fresh
# new-sample preprocessing fits when both candidates for one object start
# around the same time.

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
NEW_H5_DIR=${NEW_H5_DIR:-/home/zephyr/Data/viviana/bnl-treasure/data_new/h5}
PROJECT=${PROJECT:-atlas_object_final_tokenizers_new_mcdata}
RESULTS_DIR=${RESULTS_DIR:-$ROOT/results}
LOG_DIR=${LOG_DIR:-$RESULTS_DIR/logs/$PROJECT}
TMP_DIR=${TMP_DIR:-$RESULTS_DIR/tmp/$PROJECT}
PREPROCESS_DIR=${PREPROCESS_DIR:-$RESULTS_DIR/preprocessing/$PROJECT}
CONFIG=${CONFIG:-$ROOT/configs/datamodule/atlas_event_object.yaml}
DATAMODULE=${DATAMODULE:-atlas_event_object}

GPUS=${GPUS:-0,3}
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-1024}
MAX_PREPROCESS_OBJECTS=${MAX_PREPROCESS_OBJECTS:-1000000}
LOGGER_OFFLINE=${LOGGER_OFFLINE:-true}
DRY_RUN=${DRY_RUN:-0}
CONTINUE_ON_FAILURE=${CONTINUE_ON_FAILURE:-1}
REFIT_PREPROCESSORS=${REFIT_PREPROCESSORS:-false}
MATRIX_MODE=${MATRIX_MODE:-all}
UNIT_PREFIX=${UNIT_PREFIX:-atlas-object-final-tokenizers-new-mcdata}

mkdir -p "$LOG_DIR" "$TMP_DIR" "$PREPROCESS_DIR"
cd "$ROOT"

MIXED_FILELIST="$TMP_DIR/new_mcdata_files.txt"

mapfile -t H5_FILES < <(
  {
    find "$NEW_H5_DIR" \
      -maxdepth 1 -type f -name '*.h5' \
      ! -name 'DAOD_PHYSLITE.370016*' \
      -size +0c
    find "$NEW_H5_DIR/realdata" \
      -maxdepth 1 -type f -name '*.h5' \
      -size +0c
  } | sort -u
)

if [ "${#H5_FILES[@]}" -eq 0 ]; then
  echo "No non-empty H5 files found under $NEW_H5_DIR" >&2
  exit 1
fi

printf '%s\n' "${H5_FILES[@]}" > "$MIXED_FILELIST"
N_MC=$(find "$NEW_H5_DIR" \
  -maxdepth 1 -type f -name '*.h5' \
  ! -name 'DAOD_PHYSLITE.370016*' \
  -size +0c | wc -l)
N_DATA=$(find "$NEW_H5_DIR/realdata" \
  -maxdepth 1 -type f -name '*.h5' \
  -size +0c | wc -l)

echo "New MC+data H5 files: ${#H5_FILES[@]}"
echo "  MC files:        $N_MC"
echo "  real-data files: $N_DATA"
echo "  excluded MC:     $NEW_H5_DIR/DAOD_PHYSLITE.370016*"
echo "  file list:       $MIXED_FILELIST"

if [ "$N_MC" -eq 0 ] || [ "$N_DATA" -eq 0 ]; then
  echo "Missing MC or real-data H5 files in new-sample file list" >&2
  exit 1
fi

split_csv() {
  local value=$1
  tr ',' '\n' <<< "$value" | sed '/^$/d'
}

object_preprocessor_name() {
  case "$1" in
    tracks) echo "tracks_log_standard_no_ndoflog" ;;
    jets|electrons|muons|photons|taus) echo "$1_log_standard" ;;
    *)
      echo "Unknown object: $1" >&2
      return 1
      ;;
  esac
}

object_log_features() {
  case "$1" in
    jets) echo "pt,mass,n_trk,QG_nTracks" ;;
    electrons) echo "pt,ptvarcone30" ;;
    muons) echo "pt,ptvarcone30" ;;
    photons) echo "pt,ptcone20" ;;
    taus) echo "pt" ;;
    tracks) echo "pt,chiSquared" ;;
    *)
      echo "Unknown object: $1" >&2
      return 1
      ;;
  esac
}

run_name_for() {
  local object=$1
  local codebook_dim=$2
  local codebook_size=$3
  local num_quantizers=$4
  echo "${object}_full_dim${codebook_dim}_cb${codebook_size}_q${num_quantizers}_e${EPOCHS}_new_mcdata"
}

mapfile -t GPU_LIST < <(split_csv "$GPUS")
if [ "${#GPU_LIST[@]}" -eq 0 ]; then
  echo "No GPUs configured. Set GPUS=0,3 for example." >&2
  exit 1
fi

case "$MATRIX_MODE" in
  all)
    MATRIX=(
      "jets 16 16384 4 [256,512,1024]"
      "jets 8 2048 8 [128,256,512]"
      "electrons 16 8192 4 [256,512,1024]"
      "electrons 8 2048 8 [128,256,512]"
      "muons 16 8192 4 [256,512,1024]"
      "muons 8 2048 8 [128,256,512]"
      "photons 16 8192 4 [256,512,1024]"
      "photons 8 2048 8 [128,256,512]"
      "taus 16 8192 4 [256,512,1024]"
      "taus 8 4096 8 [128,256,512]"
      "tracks 16 16384 4 [256,512,1024]"
      "tracks 8 4096 8 [128,256,512]"
    )
    ;;
  q4_taus_tracks)
    MATRIX=(
      "taus 16 8192 4 [256,512,1024]"
      "tracks 16 16384 4 [256,512,1024]"
    )
    ;;
  q1_full)
    MATRIX=(
      "jets 8 16384 1 [128,256,512]"
      "electrons 8 16384 1 [128,256,512]"
      "muons 8 16384 1 [128,256,512]"
      "photons 8 16384 1 [128,256,512]"
      "taus 8 16384 1 [128,256,512]"
      "tracks 8 16384 1 [128,256,512]"
    )
    ;;
  jets_q8_cb4096)
    MATRIX=(
      "jets 8 4096 8 [128,256,512]"
    )
    ;;
  *)
    echo "Unknown MATRIX_MODE=$MATRIX_MODE. Valid options: all, q4_taus_tracks, q1_full, jets_q8_cb4096" >&2
    exit 1
    ;;
esac

declare -a QUEUE_FILES
for gpu in "${GPU_LIST[@]}"; do
  queue_file="$TMP_DIR/final_tokenizer_queue_gpu${gpu}.tsv"
  : > "$queue_file"
  QUEUE_FILES+=("$queue_file")
done

idx=0
for row in "${MATRIX[@]}"; do
  queue_file="${QUEUE_FILES[$((idx % ${#QUEUE_FILES[@]}))]}"
  read -r object codebook_dim codebook_size num_quantizers hidden_dims <<< "$row"
  prep_name=$(object_preprocessor_name "$object")
  preprocessor="$PREPROCESS_DIR/${prep_name}.joblib"
  log_features=$(object_log_features "$object")
  run_name=$(run_name_for "$object" "$codebook_dim" "$codebook_size" "$num_quantizers")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$object" "$codebook_dim" "$codebook_size" "$num_quantizers" \
    "$hidden_dims" "$preprocessor" "$log_features" "$run_name" \
    >> "$queue_file"
  idx=$((idx + 1))
done

write_worker() {
  local gpu=$1
  local queue_file=$2
  local worker_script=$3
  local wandb_mode=online
  if [ "$LOGGER_OFFLINE" = "true" ]; then
    wandb_mode=offline
  fi

  cat > "$worker_script" <<EOF
#!/usr/bin/env bash
set -u

ROOT="$ROOT"
PROJECT="$PROJECT"
RESULTS_DIR="$RESULTS_DIR"
LOG_DIR="$LOG_DIR"
TMP_DIR="$TMP_DIR"
PREPROCESS_DIR="$PREPROCESS_DIR"
CONFIG="$CONFIG"
DATAMODULE="$DATAMODULE"
QUEUE_FILE="$queue_file"
MIXED_FILELIST="$MIXED_FILELIST"
GPU="$gpu"
EPOCHS="$EPOCHS"
BATCH_SIZE="$BATCH_SIZE"
MAX_PREPROCESS_OBJECTS="$MAX_PREPROCESS_OBJECTS"
LOGGER_OFFLINE="$LOGGER_OFFLINE"
WANDB_MODE_VALUE="$wandb_mode"
CONTINUE_ON_FAILURE="$CONTINUE_ON_FAILURE"
REFIT_PREPROCESSORS="$REFIT_PREPROCESSORS"

cd "\$ROOT"
mkdir -p "\$LOG_DIR" "\$TMP_DIR" "\$PREPROCESS_DIR"

H5_LIST=\$(/root/.pixi/bin/pixi run python - "\$MIXED_FILELIST" <<'PY'
from pathlib import Path
import sys

files = [line.strip() for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
print("[" + ",".join(files) + "]")
PY
)

fit_preprocessor_if_needed() {
  local object=\$1
  local preprocessor=\$2
  local log_features=\$3
  local output_name
  output_name=\$(basename "\$preprocessor" .joblib)
  local lock_file="\$PREPROCESS_DIR/\${output_name}.lock"
  local preprocess_log="\$LOG_DIR/preprocess_\${output_name}.log"

  (
    flock 9
    if [ "\$REFIT_PREPROCESSORS" != "true" ] && [ -f "\$preprocessor" ]; then
      echo "\$(date): reusing preprocessor \$preprocessor"
      exit 0
    fi

    echo "\$(date): fitting preprocessor for \$object -> \$preprocessor"
    mapfile -t H5_FILES < "\$MIXED_FILELIST"

    /root/.pixi/bin/pixi run python scripts/get_atlas_object_preprocessing.py \\
      --h5-files "\${H5_FILES[@]}" \\
      --datamodule-config "\$CONFIG" \\
      --object-type "\$object" \\
      --mode log_standard \\
      --log-features "\$log_features" \\
      --max-objects "\$MAX_PREPROCESS_OBJECTS" \\
      --output-dir "\$PREPROCESS_DIR" \\
      --output-name "\$output_name" \\
      2>&1 | tee "\$preprocess_log"
  ) 9>"\$lock_file"

  if [ ! -f "\$preprocessor" ]; then
    echo "Preprocessor was not created: \$preprocessor" >&2
    return 1
  fi
}

STATUS=0
JOB_INDEX=0

while IFS=\$'\t' read -r OBJECT CODEBOOK_DIM CODEBOOK_SIZE NUM_QUANTIZERS HIDDEN_DIMS PREPROCESSOR LOG_FEATURES RUN_NAME; do
  JOB_INDEX=\$((JOB_INDEX + 1))
  RUN_DIR="\$RESULTS_DIR/\$PROJECT/\$RUN_NAME"
  RUN_LOG="\$LOG_DIR/\$RUN_NAME.log"

  echo
  echo "============================================================"
  echo "\$(date): GPU \$GPU job \$JOB_INDEX: \$RUN_NAME"
  echo "object=\$OBJECT codebook_dim=\$CODEBOOK_DIM codebook_size=\$CODEBOOK_SIZE num_quantizers=\$NUM_QUANTIZERS"
  echo "hidden_dims=\$HIDDEN_DIMS"
  echo "preprocessor=\$PREPROCESSOR"
  echo "log_features=\$LOG_FEATURES"
  echo "run_dir=\$RUN_DIR"
  echo "log=\$RUN_LOG"

  if [ -f "\$RUN_DIR/SUCCESS.txt" ]; then
    echo "\$(date): skipping completed \$RUN_NAME"
    continue
  fi

  fit_preprocessor_if_needed "\$OBJECT" "\$PREPROCESSOR" "\$LOG_FEATURES"
  prep_rc=\$?
  if [ "\$prep_rc" -ne 0 ]; then
    echo "\$(date): FAILED preprocessing for \$RUN_NAME with exit code \$prep_rc"
    STATUS=1
    if [ "\$CONTINUE_ON_FAILURE" -ne 1 ]; then
      exit "\$prep_rc"
    fi
    continue
  fi

  CUDA_VISIBLE_DEVICES="\$GPU" \\
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
  WANDB_MODE="\$WANDB_MODE_VALUE" \\
  WANDB_SILENT=true \\
  /root/.pixi/bin/pixi run python scripts/train.py \\
    datamodule="\$DATAMODULE" \\
    model=vqvae \\
    callbacks=event_tokenizer \\
    output_dir="\$RESULTS_DIR" \\
    project_name="\$PROJECT" \\
    network_name="\$RUN_NAME" \\
    logger.offline="\$LOGGER_OFFLINE" \\
    trainer.max_epochs="\$EPOCHS" \\
    trainer.accelerator=gpu \\
    trainer.devices=1 \\
    trainer.limit_val_batches=1000 \\
    datamodule.object_type="\$OBJECT" \\
    datamodule.batch_size="\$BATCH_SIZE" \\
    model.codebook_dim="\$CODEBOOK_DIM" \\
    "model.encoder.model.hidden_dims=\$HIDDEN_DIMS" \\
    "model.decoder.model.hidden_dims=\$HIDDEN_DIMS" \\
    model.codebook_size="\$CODEBOOK_SIZE" \\
    model.num_quantizers="\$NUM_QUANTIZERS" \\
    'model.feature_loss_weights=null' \\
    model.dead_code_reset=false \\
    model.data_codebook_init=false \\
    +datamodule.transforms.preprocess._target_=heptokens.data.collation.preprocess_objects_batch \\
    +datamodule.transforms.preprocess._partial_=true \\
    +datamodule.transforms.preprocess.cst_fn._target_=joblib.load \\
    +datamodule.transforms.preprocess.cst_fn.filename="\$PREPROCESSOR" \\
    "datamodule.data_paths=\${H5_LIST}" \\
    2>&1 | tee "\$RUN_LOG"

  rc=\${PIPESTATUS[0]}
  if [ "\$rc" -ne 0 ]; then
    echo "\$(date): FAILED \$RUN_NAME with exit code \$rc"
    STATUS=1
    if [ "\$CONTINUE_ON_FAILURE" -ne 1 ]; then
      exit "\$rc"
    fi
  else
    echo "\$(date): finished \$RUN_NAME"
  fi
done < "\$QUEUE_FILE"

echo
echo "\$(date): queue finished on GPU \$GPU with status \$STATUS"
exit "\$STATUS"
EOF

  chmod +x "$worker_script"
  {
    echo "gpu=$gpu"
    echo "queue_file=$queue_file"
    echo "worker_script=$worker_script"
    echo "project=$PROJECT"
    echo "new_h5_dir=$NEW_H5_DIR"
    echo "mixed_filelist=$MIXED_FILELIST"
    echo "preprocess_dir=$PREPROCESS_DIR"
    echo "datamodule=$DATAMODULE"
    echo "epochs=$EPOCHS"
    echo "batch_size=$BATCH_SIZE"
    echo "logger_offline=$LOGGER_OFFLINE"
    echo "continue_on_failure=$CONTINUE_ON_FAILURE"
    echo "refit_preprocessors=$REFIT_PREPROCESSORS"
  } > "$worker_script.manifest.txt"
}

echo
echo "Final object tokenizer new MC+data queues"
echo "  project: $PROJECT"
echo "  matrix mode: $MATRIX_MODE"
echo "  unit prefix: $UNIT_PREFIX"
echo "  GPUs: ${GPU_LIST[*]}"
echo "  jobs: ${#MATRIX[@]}"
echo "  epochs: $EPOCHS"
echo "  batch size: $BATCH_SIZE"
echo "  datamodule: $DATAMODULE"
echo "  preprocessors: $PREPROCESS_DIR"
echo "  refit preprocessors: $REFIT_PREPROCESSORS"
echo "  continue on failure: $CONTINUE_ON_FAILURE"
echo "  logs: $LOG_DIR"
echo

for i in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$i]}"
  queue_file="${QUEUE_FILES[$i]}"
  echo "GPU $gpu queue:"
  nl -ba "$queue_file" | sed 's/^/  /'
done

for i in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$i]}"
  queue_file="${QUEUE_FILES[$i]}"
  stamp=$(date +%Y%m%d-%H%M%S)
  unit="${UNIT_PREFIX}-gpu${gpu}-${stamp}"
  worker_script="$TMP_DIR/${unit}.sh"
  write_worker "$gpu" "$queue_file" "$worker_script"

  if [ "$DRY_RUN" -eq 1 ]; then
    echo
    echo "DRY RUN: would submit $unit"
    echo "  worker: $worker_script"
    echo "  queue:  $queue_file"
    continue
  fi

  systemd-run \
    --unit="$unit" \
    --description="Final object tokenizers on new MC+data, GPU $gpu" \
    --collect \
    --property=WorkingDirectory="$ROOT" \
    /bin/bash "$worker_script"

  echo
  echo "Submitted $unit"
  echo "  GPU: $gpu"
  echo "  progress: journalctl -u $unit -f"
  echo "  worker: $worker_script"
  echo "  queue: $queue_file"
done
