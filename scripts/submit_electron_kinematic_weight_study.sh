#!/usr/bin/env bash
set -euo pipefail

# Controlled full-feature electron study:
#   - capacity: larger latent/network capacity with the original unweighted loss
#   - kinw2/kinw4: baseline capacity with only the kinematic loss weight changed

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}

PROJECT=${PROJECT:-atlas_electron_kinematic_weight_study}
RESULTS_DIR=${RESULTS_DIR:-$ROOT/results}
LOG_DIR="$RESULTS_DIR/logs/$PROJECT"
TMP_DIR="$RESULTS_DIR/tmp/$PROJECT"
REFERENCE_RUN=${REFERENCE_RUN:-$RESULTS_DIR/atlas_event_tokenizers_0107_logstd_mc_realdata/electrons_logstd_dim8_cb4096_q4}
REFERENCE_CONFIG="$REFERENCE_RUN/full_config.yaml"
PREPROCESSOR=${PREPROCESSOR:-$RESULTS_DIR/preprocessing/atlas_event_tokenizers_0107_logstd_mc_realdata/electrons_log_standard.joblib}

GPU_CAPACITY=${GPU_CAPACITY:-1}
GPU_KINW2=${GPU_KINW2:-2}
GPU_KINW4=${GPU_KINW4:-3}
MAX_EPOCHS=${MAX_EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-1024}
LOGGER_OFFLINE=${LOGGER_OFFLINE:-true}
DRY_RUN=${DRY_RUN:-0}
# Comma-separated selection: capacity,kinw2,kinw4. The existing natural MC+data model
# is the unweighted (1x) reference and is not retrained by this launcher.
RUNS=${RUNS:-capacity,kinw2,kinw4}

mkdir -p "$LOG_DIR" "$TMP_DIR"
cd "$ROOT"

MIXED_FILELIST="$TMP_DIR/mixed_files.txt"

if [ ! -f "$REFERENCE_CONFIG" ]; then
  echo "Missing reference configuration: $REFERENCE_CONFIG"
  exit 1
fi

# Preserve the exact ordered file list used by the existing 1x reference.
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
  echo "Missing MC or real-data H5 files"
  exit 1
fi
if [ ! -f "$PREPROCESSOR" ]; then
  echo "Missing full-feature MC+data preprocessor: $PREPROCESSOR"
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
  local kinematic_weight=$3
  local codebook_dim=$4
  local hidden_dims=$5
  local run_name=$6
  local weights=""
  local weight_override="'model.feature_loss_weights=null'"
  local study_description="larger capacity, unweighted reconstruction loss"
  local stamp unit job_script run_log wandb_mode

  if [ "$kinematic_weight" != "1" ]; then
    weights="[${kinematic_weight},${kinematic_weight},${kinematic_weight},${kinematic_weight},1,1,1,1]"
    weight_override="'model.feature_loss_weights=$weights'"
    study_description="${kinematic_weight}x kinematic loss weight, baseline capacity"
  fi

  stamp=$(date +%Y%m%d-%H%M%S)
  unit="atlas-electron-lossweight-${stamp}-${tag}"
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

CUDA_VISIBLE_DEVICES="$gpu" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_MODE="$wandb_mode" \
WANDB_SILENT=true \
/root/.pixi/bin/pixi run python scripts/train.py \
  datamodule=atlas_event_object \
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
  model.codebook_dim="$codebook_dim" \
  'model.encoder.model.hidden_dims=$hidden_dims' \
  'model.decoder.model.hidden_dims=$hidden_dims' \
  model.codebook_size=4096 \
  model.num_quantizers=4 \
  $weight_override \
  model.dead_code_reset=false \
  model.data_codebook_init=false \
  +datamodule.transforms.preprocess._target_=heptokens.data.collation.preprocess_objects_batch \
  +datamodule.transforms.preprocess._partial_=true \
  +datamodule.transforms.preprocess.cst_fn._target_=joblib.load \
  +datamodule.transforms.preprocess.cst_fn.filename="$PREPROCESSOR" \
  "datamodule.data_paths=\${H5_LIST}" \
  2>&1 | tee "\$RUN_LOG"
EOF

  chmod +x "$job_script"

  {
    echo "unit=$unit"
    echo "gpu=$gpu"
    echo "run_name=$run_name"
    echo "input=natural MC+data, full electron features"
    echo "feature_order=pt,eta,phi,charge,LHMedium,LHTight,ptvarcone30,topoetcone20"
    echo "study_change=$study_description"
    echo "raw_feature_loss_weights=${weights:-none}"
    echo "weights_are_normalized_to_mean_one=true"
    echo "reference_run=$REFERENCE_RUN"
    echo "reference_config=$REFERENCE_CONFIG"
    echo "codebook_dim=$codebook_dim"
    echo "hidden_dims=$hidden_dims"
    echo "codebook_size=4096"
    echo "num_quantizers=4"
    echo "preprocessor=$PREPROCESSOR"
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
    --description="Electron full-feature study: $study_description" \
    --collect \
    --property=WorkingDirectory="$ROOT" \
    /bin/bash "$job_script"

  echo "Submitted $unit on physical GPU $gpu"
  echo "  progress: journalctl -u $unit -f"
  echo "  training log: $run_log"
}

if is_selected capacity; then
  submit_run capacity "$GPU_CAPACITY" 1 16 '[256,512,1024]' \
    electrons_full_dim16_cb4096_q4_larger_capacity_mcdata
fi
if is_selected kinw2; then
  submit_run kinw2 "$GPU_KINW2" 2 8 '[128,256,512]' \
    electrons_full_dim8_cb4096_q4_kinw2_mcdata
fi
if is_selected kinw4; then
  submit_run kinw4 "$GPU_KINW4" 4 8 '[128,256,512]' \
    electrons_full_dim8_cb4096_q4_kinw4_mcdata
fi

echo
echo "Selected runs: $RUNS"
echo "Reference: existing natural MC+data model with 1x feature weights."
echo "Capacity arm: codebook_dim=16, hidden_dims=[256,512,1024], unweighted loss."
echo "Weight arms: codebook_dim=8, hidden_dims=[128,256,512], 2x/4x kinematics."
echo "All runs use codebook_size=4096 and num_quantizers=4."
