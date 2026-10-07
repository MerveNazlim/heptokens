#!/usr/bin/env bash
set -euo pipefail

# Controlled full-feature electron training study:
#   - longer: same baseline capacity, trained for more epochs
#   - q8: same baseline architecture, more residual quantizers

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
PROJECT=${PROJECT:-atlas_electron_full_training_controls}
RESULTS_DIR=${RESULTS_DIR:-$ROOT/results}
LOG_DIR=${LOG_DIR:-$RESULTS_DIR/logs/$PROJECT}
TMP_DIR=${TMP_DIR:-$RESULTS_DIR/tmp/$PROJECT}
REFERENCE_RUN=${REFERENCE_RUN:-$RESULTS_DIR/atlas_event_tokenizers_0107_logstd_mc_realdata/electrons_logstd_dim8_cb4096_q4}
REFERENCE_CONFIG=${REFERENCE_CONFIG:-$REFERENCE_RUN/full_config.yaml}
PREPROCESSOR=${PREPROCESSOR:-$RESULTS_DIR/preprocessing/atlas_event_tokenizers_0107_logstd_mc_realdata/electrons_log_standard.joblib}

GPU_LONGER=${GPU_LONGER:-0}
GPU_Q8=${GPU_Q8:-3}
LONGER_EPOCHS=${LONGER_EPOCHS:-60}
Q8_EPOCHS=${Q8_EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-1024}
LOGGER_OFFLINE=${LOGGER_OFFLINE:-true}
DRY_RUN=${DRY_RUN:-0}
RUNS=${RUNS:-longer,q8}

mkdir -p "$LOG_DIR" "$TMP_DIR"
cd "$ROOT"

MIXED_FILELIST="$TMP_DIR/mixed_files.txt"

if [ ! -f "$REFERENCE_CONFIG" ]; then
  echo "Missing reference configuration: $REFERENCE_CONFIG" >&2
  exit 1
fi
if [ ! -f "$PREPROCESSOR" ]; then
  echo "Missing full-feature MC+data preprocessor: $PREPROCESSOR" >&2
  exit 1
fi

# Preserve the exact ordered file list used by the existing full MC+data run.
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

is_selected() {
  case ",$RUNS," in
    *",$1,"*) return 0 ;;
    *) return 1 ;;
  esac
}

submit_run() {
  local tag=$1
  local gpu=$2
  local max_epochs=$3
  local num_quantizers=$4
  local run_name=$5
  local description=$6
  local stamp unit job_script run_log wandb_mode

  stamp=$(date +%Y%m%d-%H%M%S)
  unit="atlas-electron-full-${tag}-${stamp}"
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
FILELIST="$MIXED_FILELIST"
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

CUDA_VISIBLE_DEVICES="$gpu" \\
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
WANDB_MODE="$wandb_mode" \\
WANDB_SILENT=true \\
/root/.pixi/bin/pixi run python scripts/train.py \\
  datamodule=atlas_event_object \\
  model=vqvae \\
  callbacks=event_tokenizer \\
  output_dir="$RESULTS_DIR" \\
  project_name="$PROJECT" \\
  network_name="$run_name" \\
  logger.offline="$LOGGER_OFFLINE" \\
  trainer.max_epochs="$max_epochs" \\
  trainer.accelerator=gpu \\
  trainer.devices=1 \\
  trainer.limit_val_batches=1000 \\
  datamodule.object_type=electrons \\
  datamodule.batch_size="$BATCH_SIZE" \\
  model.codebook_dim=8 \\
  'model.encoder.model.hidden_dims=[128,256,512]' \\
  'model.decoder.model.hidden_dims=[128,256,512]' \\
  model.codebook_size=4096 \\
  model.num_quantizers="$num_quantizers" \\
  'model.feature_loss_weights=null' \\
  model.dead_code_reset=false \\
  model.data_codebook_init=false \\
  +datamodule.transforms.preprocess._target_=heptokens.data.collation.preprocess_objects_batch \\
  +datamodule.transforms.preprocess._partial_=true \\
  +datamodule.transforms.preprocess.cst_fn._target_=joblib.load \\
  +datamodule.transforms.preprocess.cst_fn.filename="$PREPROCESSOR" \\
  "datamodule.data_paths=\${H5_LIST}" \\
  2>&1 | tee "\$RUN_LOG"
EOF

  chmod +x "$job_script"

  {
    echo "unit=$unit"
    echo "gpu=$gpu"
    echo "run_name=$run_name"
    echo "description=$description"
    echo "input=natural MC+data, full electron features"
    echo "feature_order=pt,eta,phi,charge,LHMedium,LHTight,ptvarcone30,topoetcone20"
    echo "reference_run=$REFERENCE_RUN"
    echo "reference_config=$REFERENCE_CONFIG"
    echo "preprocessor=$PREPROCESSOR"
    echo "codebook_dim=8"
    echo "hidden_dims=[128,256,512]"
    echo "codebook_size=4096"
    echo "num_quantizers=$num_quantizers"
    echo "max_epochs=$max_epochs"
    echo "batch_size=$BATCH_SIZE"
    echo "feature_loss_weights=none"
    echo "wandb_mode=$wandb_mode"
  } > "$TMP_DIR/${unit}.manifest.txt"

  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY RUN: would submit $unit on physical GPU $gpu"
    echo "  job: $job_script"
    return
  fi

  systemd-run \
    --unit="$unit" \
    --description="Electron full-feature training control: $description" \
    --collect \
    --property=WorkingDirectory="$ROOT" \
    /bin/bash "$job_script"

  echo "Submitted $unit on physical GPU $gpu"
  echo "  progress: journalctl -u $unit -f"
  echo "  training log: $run_log"
}

if is_selected longer; then
  submit_run \
    longer \
    "$GPU_LONGER" \
    "$LONGER_EPOCHS" \
    4 \
    "electrons_full_dim8_cb4096_q4_e${LONGER_EPOCHS}_mcdata" \
    "baseline q4 trained for ${LONGER_EPOCHS} epochs"
fi

if is_selected q8; then
  submit_run \
    q8 \
    "$GPU_Q8" \
    "$Q8_EPOCHS" \
    8 \
    "electrons_full_dim8_cb4096_q8_e${Q8_EPOCHS}_mcdata" \
    "baseline width with 8 residual quantizers for ${Q8_EPOCHS} epochs"
fi

echo
echo "Selected runs: $RUNS"
echo "Longer arm: GPU $GPU_LONGER, q4, epochs=$LONGER_EPOCHS"
echo "Q8 arm:     GPU $GPU_Q8, q8, epochs=$Q8_EPOCHS"
echo "Project:    $PROJECT"
echo "Logs:       $LOG_DIR"
