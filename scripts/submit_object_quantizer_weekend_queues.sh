#!/usr/bin/env bash
set -euo pipefail

# Weekend quantizer-control queues for object tokenizers.
#
# This creates one systemd service per GPU. Each service runs its assigned
# trainings sequentially, so a GPU only trains one tokenizer at a time.
#
# Default matrix:
#   jets/muons/photons: q6@4096, q8@2048, q8@4096
#   taus/tracks:        q6@8192, q6@4096, q8@4096, q8@8192

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
PROJECT=${PROJECT:-atlas_object_quantizer_weekend_controls}
RESULTS_DIR=${RESULTS_DIR:-$ROOT/results}
LOG_DIR=${LOG_DIR:-$RESULTS_DIR/logs/$PROJECT}
TMP_DIR=${TMP_DIR:-$RESULTS_DIR/tmp/$PROJECT}

REFERENCE_RUN=${REFERENCE_RUN:-$RESULTS_DIR/atlas_event_tokenizers_0107_logstd_mc_realdata/electrons_logstd_dim8_cb4096_q4}
REFERENCE_CONFIG=${REFERENCE_CONFIG:-$REFERENCE_RUN/full_config.yaml}
PREPROCESS_DIR=${PREPROCESS_DIR:-$RESULTS_DIR/preprocessing/atlas_event_tokenizers_0107_logstd_mc_realdata}

GPUS=${GPUS:-0,3}
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-1024}
LOGGER_OFFLINE=${LOGGER_OFFLINE:-true}
DRY_RUN=${DRY_RUN:-0}
CONTINUE_ON_FAILURE=${CONTINUE_ON_FAILURE:-1}
MATRIX_FILE=${MATRIX_FILE:-}

mkdir -p "$LOG_DIR" "$TMP_DIR"
cd "$ROOT"

MIXED_FILELIST="$TMP_DIR/mixed_files.txt"

if [ ! -f "$REFERENCE_CONFIG" ]; then
  echo "Missing reference configuration: $REFERENCE_CONFIG" >&2
  exit 1
fi

/root/.pixi/bin/pixi run python - "$REFERENCE_CONFIG" "$MIXED_FILELIST" <<'PY'
from pathlib import Path
import sys

from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
files = [str(path) for path in cfg.datamodule.data_paths]
Path(sys.argv[2]).write_text("\n".join(files) + "\n")
PY

N_MC=$(grep -vc '/realdata/' "$MIXED_FILELIST" || true)
N_DATA=$(grep -c '/realdata/' "$MIXED_FILELIST" || true)
echo "MC H5 files: $N_MC"
echo "real-data H5 files: $N_DATA"

if [ "$N_MC" -eq 0 ] || [ "$N_DATA" -eq 0 ]; then
  echo "Missing MC or real-data H5 files in reference file list" >&2
  exit 1
fi

split_csv() {
  local value=$1
  tr ',' '\n' <<< "$value" | sed '/^$/d'
}

object_preprocessor() {
  case "$1" in
    jets) echo "$PREPROCESS_DIR/jets_log_standard.joblib" ;;
    muons) echo "$PREPROCESS_DIR/muons_log_standard.joblib" ;;
    photons) echo "$PREPROCESS_DIR/photons_log_standard.joblib" ;;
    taus) echo "$PREPROCESS_DIR/taus_log_standard.joblib" ;;
    tracks) echo "$PREPROCESS_DIR/tracks_log_standard_no_ndoflog.joblib" ;;
    *)
      echo "Unknown object: $1" >&2
      return 1
      ;;
  esac
}

run_name_for() {
  local object=$1
  local codebook_size=$2
  local num_quantizers=$3
  echo "${object}_full_dim8_cb${codebook_size}_q${num_quantizers}_e${EPOCHS}_mcdata"
}

mapfile -t GPU_LIST < <(split_csv "$GPUS")
if [ "${#GPU_LIST[@]}" -eq 0 ]; then
  echo "No GPUs configured. Set GPUS=0,3 for example." >&2
  exit 1
fi

declare -a MATRIX
if [ -n "$MATRIX_FILE" ]; then
  if [ ! -f "$MATRIX_FILE" ]; then
    echo "Missing MATRIX_FILE: $MATRIX_FILE" >&2
    exit 1
  fi
  mapfile -t MATRIX < <(grep -v '^[[:space:]]*#' "$MATRIX_FILE" | sed '/^[[:space:]]*$/d')
else
  MATRIX=(
    "jets 4096 6"
    "jets 2048 8"
    "jets 4096 8"
    "muons 4096 6"
    "muons 2048 8"
    "muons 4096 8"
    "photons 4096 6"
    "photons 2048 8"
    "photons 4096 8"
    "taus 8192 6"
    "taus 4096 6"
    "taus 4096 8"
    "taus 8192 8"
    "tracks 8192 6"
    "tracks 4096 6"
    "tracks 4096 8"
    "tracks 8192 8"
  )
fi

for row in "${MATRIX[@]}"; do
  read -r object codebook_size num_quantizers extra <<< "$row"
  if [ -n "${extra:-}" ]; then
    echo "Bad matrix row, expected: <object> <codebook_size> <num_quantizers>: $row" >&2
    exit 1
  fi
  preprocessor=$(object_preprocessor "$object")
  if [ ! -f "$preprocessor" ]; then
    echo "Missing preprocessor for $object: $preprocessor" >&2
    exit 1
  fi
done

declare -a QUEUE_FILES
for gpu in "${GPU_LIST[@]}"; do
  queue_file="$TMP_DIR/weekend_queue_gpu${gpu}.tsv"
  : > "$queue_file"
  QUEUE_FILES+=("$queue_file")
done

idx=0
for row in "${MATRIX[@]}"; do
  queue_file="${QUEUE_FILES[$((idx % ${#QUEUE_FILES[@]}))]}"
  read -r object codebook_size num_quantizers <<< "$row"
  preprocessor=$(object_preprocessor "$object")
  run_name=$(run_name_for "$object" "$codebook_size" "$num_quantizers")
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$object" "$codebook_size" "$num_quantizers" "$preprocessor" "$run_name" \
    >> "$queue_file"
  idx=$((idx + 1))
done

write_worker() {
  local gpu=$1
  local queue_file=$2
  local worker_script=$3
  local worker_log=$4
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
QUEUE_FILE="$queue_file"
MIXED_FILELIST="$MIXED_FILELIST"
GPU="$gpu"
EPOCHS="$EPOCHS"
BATCH_SIZE="$BATCH_SIZE"
LOGGER_OFFLINE="$LOGGER_OFFLINE"
WANDB_MODE_VALUE="$wandb_mode"
CONTINUE_ON_FAILURE="$CONTINUE_ON_FAILURE"

cd "\$ROOT"
mkdir -p "\$LOG_DIR"

H5_LIST=\$(/root/.pixi/bin/pixi run python - "\$MIXED_FILELIST" <<'PY'
from pathlib import Path
import sys

files = [line.strip() for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
print("[" + ",".join(files) + "]")
PY
)

STATUS=0
JOB_INDEX=0

while IFS=\$'\t' read -r OBJECT CODEBOOK_SIZE NUM_QUANTIZERS PREPROCESSOR RUN_NAME; do
  JOB_INDEX=\$((JOB_INDEX + 1))
  RUN_DIR="\$RESULTS_DIR/\$PROJECT/\$RUN_NAME"
  RUN_LOG="\$LOG_DIR/\$RUN_NAME.log"

  echo
  echo "============================================================"
  echo "\$(date): GPU \$GPU job \$JOB_INDEX: \$RUN_NAME"
  echo "object=\$OBJECT codebook_size=\$CODEBOOK_SIZE num_quantizers=\$NUM_QUANTIZERS"
  echo "preprocessor=\$PREPROCESSOR"
  echo "run_dir=\$RUN_DIR"
  echo "log=\$RUN_LOG"

  if [ -f "\$RUN_DIR/SUCCESS.txt" ]; then
    echo "\$(date): skipping completed \$RUN_NAME"
    continue
  fi

  CUDA_VISIBLE_DEVICES="\$GPU" \\
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
  WANDB_MODE="\$WANDB_MODE_VALUE" \\
  WANDB_SILENT=true \\
  /root/.pixi/bin/pixi run python scripts/train.py \\
    datamodule=atlas_event_object \\
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
    model.codebook_dim=8 \\
    'model.encoder.model.hidden_dims=[128,256,512]' \\
    'model.decoder.model.hidden_dims=[128,256,512]' \\
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
    echo "worker_log=$worker_log"
    echo "project=$PROJECT"
    echo "epochs=$EPOCHS"
    echo "batch_size=$BATCH_SIZE"
    echo "logger_offline=$LOGGER_OFFLINE"
    echo "continue_on_failure=$CONTINUE_ON_FAILURE"
  } > "$worker_script.manifest.txt"
}

echo
echo "Weekend object quantizer queues"
echo "  project: $PROJECT"
echo "  GPUs: ${GPU_LIST[*]}"
echo "  jobs: ${#MATRIX[@]}"
echo "  epochs: $EPOCHS"
echo "  batch size: $BATCH_SIZE"
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
  unit="atlas-object-quantizer-weekend-gpu${gpu}-${stamp}"
  worker_script="$TMP_DIR/${unit}.sh"
  worker_log="$LOG_DIR/${unit}.log"
  write_worker "$gpu" "$queue_file" "$worker_script" "$worker_log"

  if [ "$DRY_RUN" -eq 1 ]; then
    echo
    echo "DRY RUN: would submit $unit"
    echo "  worker: $worker_script"
    echo "  queue:  $queue_file"
    continue
  fi

  systemd-run \
    --unit="$unit" \
    --description="Weekend object quantizer queue on GPU $gpu" \
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
