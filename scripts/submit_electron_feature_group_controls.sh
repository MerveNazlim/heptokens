#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
MC_DIR=${MC_DIR:-/home/zephyr/Data/viviana/bnl-treasure/data/h5}
DATA_DIR=${DATA_DIR:-/home/zephyr/Data/viviana/bnl-treasure/data/h5/realdata}
PROJECT=${PROJECT:-atlas_electron_feature_group_controls}
PREPROCESS_DIR=${PREPROCESS_DIR:-$ROOT/results/preprocessing/$PROJECT}
LOG_DIR=${LOG_DIR:-$ROOT/results/logs/$PROJECT}
TMP_DIR=${TMP_DIR:-$ROOT/results/tmp/$PROJECT}

ID_GPU=${ID_GPU:-2}
ISO_GPU=${ISO_GPU:-3}
MAX_EPOCHS=${MAX_EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-1024}
MAX_PREPROCESS_OBJECTS=${MAX_PREPROCESS_OBJECTS:-1000000}
LOGGER_OFFLINE=${LOGGER_OFFLINE:-true}
FORCE_PREPROCESSOR=${FORCE_PREPROCESSOR:-0}
DRY_RUN=${DRY_RUN:-0}
RUNS=${RUNS:-kinematics-id,kinematics-isolation}

mkdir -p "$PREPROCESS_DIR" "$LOG_DIR" "$TMP_DIR"

MC_LIST="$TMP_DIR/mc_h5_files.txt"
DATA_LIST="$TMP_DIR/data_h5_files.txt"
MIXED_LIST="$TMP_DIR/mc_data_interleaved_h5_files.txt"

find "$MC_DIR" -maxdepth 1 -type f -name '*.h5' -size +0c | sort > "$MC_LIST"
find "$DATA_DIR" -maxdepth 1 -type f -name '*.h5' -size +0c | sort > "$DATA_LIST"

N_MC=$(wc -l < "$MC_LIST" | tr -d ' ')
N_DATA=$(wc -l < "$DATA_LIST" | tr -d ' ')
if [ "$N_MC" -eq 0 ] || [ "$N_DATA" -eq 0 ]; then
  echo "Missing MC or real-data H5 files (MC=$N_MC, data=$N_DATA)" >&2
  exit 1
fi

cd "$ROOT"

# Interleave domains so preprocessing's object cap does not make its statistics
# depend on an MC-first or data-first file ordering.
/root/.pixi/bin/pixi run python - "$MC_LIST" "$DATA_LIST" "$MIXED_LIST" <<'PY'
from itertools import zip_longest
from pathlib import Path
import sys

mc = [x for x in Path(sys.argv[1]).read_text().splitlines() if x]
data = [x for x in Path(sys.argv[2]).read_text().splitlines() if x]
mixed = [x for pair in zip_longest(mc, data) for x in pair if x is not None]
Path(sys.argv[3]).write_text("\n".join(mixed) + "\n")
PY

N_MIXED=$(wc -l < "$MIXED_LIST" | tr -d ' ')
echo "MC files: $N_MC"
echo "real-data files: $N_DATA"
echo "interleaved training files: $N_MIXED"

H5_LIST=$(/root/.pixi/bin/pixi run python - "$MIXED_LIST" <<'PY'
from pathlib import Path
import sys

files = [x for x in Path(sys.argv[1]).read_text().splitlines() if x]
print("[" + ",".join(files) + "]")
PY
)

selected_run() {
  case ",$RUNS," in
    *",$1,"*) return 0 ;;
    *) return 1 ;;
  esac
}

submit_run() {
  local tag=$1
  local gpu=$2
  local config_name=$3
  local run_name=$4
  local preprocessor=$5
  local unit="atlas-electron-feature-${tag}-$(date +%Y%m%d-%H%M%S)"
  local job_script="$TMP_DIR/${unit}.sh"
  local train_log="$LOG_DIR/${run_name}.log"

  cat > "$job_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="$gpu"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=$([ "$LOGGER_OFFLINE" = true ] && echo offline || echo online)

/root/.pixi/bin/pixi run python scripts/train.py \\
  datamodule="$config_name" \\
  model=vqvae \\
  callbacks=event_tokenizer \\
  output_dir="$ROOT/results" \\
  project_name="$PROJECT" \\
  network_name="$run_name" \\
  logger.offline="$LOGGER_OFFLINE" \\
  trainer.max_epochs="$MAX_EPOCHS" \\
  trainer.accelerator=gpu \\
  trainer.devices=1 \\
  trainer.limit_val_batches=1000 \\
  datamodule.object_type=electrons \\
  datamodule.batch_size="$BATCH_SIZE" \\
  model.codebook_dim=8 \\
  model.codebook_size=4096 \\
  model.num_quantizers=4 \\
  model.dead_code_reset=false \\
  model.data_codebook_init=false \\
  +datamodule.transforms.preprocess._target_=heptokens.data.collation.preprocess_objects_batch \\
  +datamodule.transforms.preprocess._partial_=true \\
  +datamodule.transforms.preprocess.cst_fn._target_=joblib.load \\
  +datamodule.transforms.preprocess.cst_fn.filename="$preprocessor" \\
  "datamodule.data_paths=$H5_LIST" \\
  2>&1 | tee "$train_log"
EOF

  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY RUN: /bin/bash $job_script"
    return
  fi

  systemd-run \
    --unit="$unit" \
    --description="Electron feature-group control: $tag" \
    --collect \
    --property="WorkingDirectory=$ROOT" \
    /bin/bash "$job_script"

  echo "Submitted $tag on physical GPU $gpu"
  echo "  journalctl -u $unit -f"
  echo "  training log: $train_log"
}

fit_preprocessor() {
  local datamodule_config=$1
  local output_file=$2
  local log_features=$3
  local output_dir=${output_file%/*}
  local output_name=${output_file##*/}
  output_name=${output_name%.joblib}

  if [ -f "$output_file" ] && [ "$FORCE_PREPROCESSOR" -ne 1 ]; then
    echo "Using existing preprocessor: $output_file"
    return
  fi

  echo "Fitting preprocessor: $output_file"
  /root/.pixi/bin/pixi run python scripts/get_atlas_object_preprocessing.py \
    --datamodule-config "$datamodule_config" \
    --h5-files $(tr '\n' ' ' < "$MIXED_LIST") \
    --object-type electrons \
    --mode log_standard \
    --log-features "$log_features" \
    --max-objects "$MAX_PREPROCESS_OBJECTS" \
    --output-dir "$output_dir" \
    --output-name "$output_name"
}

if selected_run kinematics-id; then
  ID_PREPROCESSOR=${ID_PREPROCESSOR:-$PREPROCESS_DIR/electrons_kinematics_id_log_standard.joblib}
  fit_preprocessor \
    "$ROOT/configs/datamodule/atlas_event_electron_kinematics_id.yaml" \
    "$ID_PREPROCESSOR" \
    pt
  submit_run \
    kinematics-id "$ID_GPU" atlas_event_electron_kinematics_id \
    electrons_kinematics_id_dim8_cb4096_q4_mcdata "$ID_PREPROCESSOR"
fi

if selected_run kinematics-isolation; then
  ISO_PREPROCESSOR=${ISO_PREPROCESSOR:-$PREPROCESS_DIR/electrons_kinematics_isolation_log_standard.joblib}
  fit_preprocessor \
    "$ROOT/configs/datamodule/atlas_event_electron_kinematics_isolation.yaml" \
    "$ISO_PREPROCESSOR" \
    pt,ptvarcone30
  submit_run \
    kinematics-isolation "$ISO_GPU" atlas_event_electron_kinematics_isolation \
    electrons_kinematics_isolation_dim8_cb4096_q4_mcdata "$ISO_PREPROCESSOR"
fi
