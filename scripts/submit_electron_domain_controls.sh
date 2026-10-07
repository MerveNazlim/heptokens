#!/usr/bin/env bash
set -euo pipefail

# Two controlled electron-tokenizer studies:
#   1. Natural MC+data training with only pt, eta, phi, charge.
#   2. Data-only training with the standard full electron feature set.

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
MC_DIR=${MC_DIR:-/home/zephyr/Data/viviana/bnl-treasure/data/h5}
DATA_DIR=${DATA_DIR:-/home/zephyr/Data/viviana/bnl-treasure/data/h5/realdata}

PROJECT=${PROJECT:-atlas_electron_domain_controls}
RESULTS_DIR=${RESULTS_DIR:-$ROOT/results}
LOG_DIR="$RESULTS_DIR/logs/$PROJECT"
TMP_DIR="$RESULTS_DIR/tmp/$PROJECT"
PREPROCESS_DIR="$RESULTS_DIR/preprocessing/$PROJECT"

KIN_GPU=${KIN_GPU:-1}
DATA_GPU=${DATA_GPU:-2}
MAX_EPOCHS=${MAX_EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-1024}
MAX_PREPROCESS_OBJECTS=${MAX_PREPROCESS_OBJECTS:-1000000}
LOGGER_OFFLINE=${LOGGER_OFFLINE:-true}
FORCE_PREPROCESSOR=${FORCE_PREPROCESSOR:-0}
DRY_RUN=${DRY_RUN:-0}
# Comma-separated controls: kinematics-mcdata,dataonly-full
RUNS=${RUNS:-kinematics-mcdata,dataonly-full}

KIN_RUN=${KIN_RUN:-electrons_kinematics_dim8_cb4096_q4_mcdata}
DATA_RUN=${DATA_RUN:-electrons_full_dim8_cb4096_q4_dataonly}

KIN_CONFIG="$ROOT/configs/datamodule/atlas_event_electron_kinematics.yaml"
KIN_PREPROCESSOR="$PREPROCESS_DIR/electrons_kinematics_log_standard.joblib"
FULL_PREPROCESSOR=${FULL_PREPROCESSOR:-$RESULTS_DIR/preprocessing/atlas_event_tokenizers_0107_logstd_mc_realdata/electrons_log_standard.joblib}

mkdir -p "$LOG_DIR" "$TMP_DIR" "$PREPROCESS_DIR"
cd "$ROOT"

MC_FILELIST="$TMP_DIR/mc_files.txt"
DATA_FILELIST="$TMP_DIR/data_files.txt"
MIXED_FILELIST="$TMP_DIR/mixed_interleaved_files.txt"

find "$MC_DIR" -maxdepth 1 -type f -name "*.h5" -size +0c | sort > "$MC_FILELIST"
find "$DATA_DIR" -maxdepth 1 -type f -name "*.h5" -size +0c | sort > "$DATA_FILELIST"

N_MC=$(wc -l < "$MC_FILELIST" | tr -d ' ')
N_DATA=$(wc -l < "$DATA_FILELIST" | tr -d ' ')
echo "MC H5 files: $N_MC"
echo "real-data H5 files: $N_DATA"

if [ "$N_MC" -eq 0 ] || [ "$N_DATA" -eq 0 ]; then
  echo "Missing MC or real-data H5 files"
  exit 1
fi

# Interleave domains for fitting the kinematics-only preprocessor. The fitting
# helper stops after MAX_PREPROCESS_OBJECTS, so MC-first concatenation would
# otherwise risk learning its statistics almost entirely from MC.
/root/.pixi/bin/pixi run python - "$MC_FILELIST" "$DATA_FILELIST" "$MIXED_FILELIST" <<'PY'
from itertools import zip_longest
from pathlib import Path
import sys

mc_path, data_path, output_path = map(Path, sys.argv[1:])
mc = [line for line in mc_path.read_text().splitlines() if line]
data = [line for line in data_path.read_text().splitlines() if line]

mixed = []
for mc_file, data_file in zip_longest(mc, data):
    if mc_file is not None:
        mixed.append(mc_file)
    if data_file is not None:
        mixed.append(data_file)

Path(output_path).write_text("\n".join(mixed) + "\n")
PY

if [ ! -f "$FULL_PREPROCESSOR" ]; then
  echo "Missing full-feature MC+data preprocessor: $FULL_PREPROCESSOR"
  exit 1
fi

is_selected() {
  case ",$RUNS," in
    *",$1,"*) return 0 ;;
    *) return 1 ;;
  esac
}

if is_selected kinematics-mcdata; then
  if [ ! -f "$KIN_CONFIG" ]; then
    echo "Missing kinematics datamodule config: $KIN_CONFIG"
    exit 1
  fi

  if [ "$FORCE_PREPROCESSOR" -eq 1 ] || [ ! -f "$KIN_PREPROCESSOR" ]; then
    echo "Fitting kinematics-only MC+data preprocessor..."
    mapfile -t MIXED_FILES < "$MIXED_FILELIST"
    /root/.pixi/bin/pixi run python scripts/get_atlas_object_preprocessing.py \
      --h5-files "${MIXED_FILES[@]}" \
      --datamodule-config "$KIN_CONFIG" \
      --object-type electrons \
      --mode log_standard \
      --log-features pt \
      --max-objects "$MAX_PREPROCESS_OBJECTS" \
      --output-dir "$PREPROCESS_DIR" \
      --output-name electrons_kinematics_log_standard
  else
    echo "Using existing kinematics preprocessor: $KIN_PREPROCESSOR"
  fi
fi

submit_run() {
  local tag=$1
  local gpu=$2
  local run_name=$3
  local datamodule_name=$4
  local filelist=$5
  local preprocessor=$6
  local description=$7

  local stamp unit job_script run_log wandb_mode
  stamp=$(date +%Y%m%d-%H%M%S)
  unit="atlas-electron-control-${stamp}-${tag}"
  job_script="$TMP_DIR/${unit}.sh"
  run_log="$LOG_DIR/${run_name}.log"
  wandb_mode=online
  if [ "$LOGGER_OFFLINE" = "true" ]; then
    wandb_mode=offline
  fi

  cat > "$job_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail

ROOT="$ROOT"
FILELIST="$filelist"
RUN_LOG="$run_log"

mapfile -t H5_FILES < "\$FILELIST"
N_FILES=\${#H5_FILES[@]}
echo "Using \$N_FILES H5 files"

if [ "\$N_FILES" -eq 0 ]; then
  echo "No H5 files found"
  exit 1
fi

H5_LIST=\$(/root/.pixi/bin/pixi run python - "\$FILELIST" <<'PY'
from pathlib import Path
import sys

files = [line.strip() for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
print("[" + ",".join(files) + "]")
PY
)

cd "\$ROOT"

CUDA_VISIBLE_DEVICES="$gpu" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_MODE="$wandb_mode" \
WANDB_SILENT=true \
/root/.pixi/bin/pixi run python scripts/train.py \
  datamodule="$datamodule_name" \
  model=vqvae \
  callbacks=event_tokenizer \
  output_dir="$RESULTS_DIR" \
  project_name="$PROJECT" \
  network_name="$run_name" \
  logger.offline="$LOGGER_OFFLINE" \
  trainer.max_epochs="$MAX_EPOCHS" \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.limit_val_batches=1000 \
  datamodule.object_type=electrons \
  datamodule.batch_size="$BATCH_SIZE" \
  model.codebook_dim=8 \
  model.codebook_size=4096 \
  model.num_quantizers=4 \
  model.dead_code_reset=false \
  model.data_codebook_init=false \
  +datamodule.transforms.preprocess._target_=heptokens.data.collation.preprocess_objects_batch \
  +datamodule.transforms.preprocess._partial_=true \
  +datamodule.transforms.preprocess.cst_fn._target_=joblib.load \
  +datamodule.transforms.preprocess.cst_fn.filename="$preprocessor" \
  "datamodule.data_paths=\${H5_LIST}" \
  2>&1 | tee "\$RUN_LOG"
EOF

  chmod +x "$job_script"

  {
    echo "unit=$unit"
    echo "description=$description"
    echo "gpu=$gpu"
    echo "run_name=$run_name"
    echo "datamodule=$datamodule_name"
    echo "filelist=$filelist"
    echo "preprocessor=$preprocessor"
    echo "max_epochs=$MAX_EPOCHS"
    echo "batch_size=$BATCH_SIZE"
    echo "wandb_mode=$wandb_mode"
  } > "$TMP_DIR/${unit}.manifest.txt"

  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY RUN: would submit $unit on physical GPU $gpu"
    echo "  job: $job_script"
    return
  fi

  systemd-run \
    --unit="$unit" \
    --description="$description" \
    --collect \
    --property=WorkingDirectory="$ROOT" \
    /bin/bash "$job_script"

  echo "Submitted $unit on physical GPU $gpu"
  echo "  progress: journalctl -u $unit -f"
  echo "  training log: $run_log"
}

if is_selected kinematics-mcdata; then
  submit_run \
    kinematics-mcdata \
    "$KIN_GPU" \
    "$KIN_RUN" \
    atlas_event_electron_kinematics \
    "$MIXED_FILELIST" \
    "$KIN_PREPROCESSOR" \
    "Electron tokenizer: natural MC+data, kinematics only"
fi

if is_selected dataonly-full; then
  submit_run \
    dataonly-full \
    "$DATA_GPU" \
    "$DATA_RUN" \
    atlas_event_object \
    "$DATA_FILELIST" \
    "$FULL_PREPROCESSOR" \
    "Electron tokenizer: real data only, full feature set"
fi

echo
echo "Selected controls: $RUNS"
if is_selected kinematics-mcdata; then
  echo "Kinematics MC+data: GPU $KIN_GPU, run $KIN_RUN"
fi
if is_selected dataonly-full; then
  echo "Data-only full:     GPU $DATA_GPU, run $DATA_RUN"
fi
