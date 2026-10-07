#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
MC_DIR=${MC_DIR:-/home/zephyr/Data/viviana/bnl-treasure/data/h5}
PROJECT=${PROJECT:-atlas_electron_feature_group_controls_mconly}
PREPROCESS_DIR=${PREPROCESS_DIR:-$ROOT/results/preprocessing/$PROJECT}
LOG_DIR=${LOG_DIR:-$ROOT/results/logs/$PROJECT}
TMP_DIR=${TMP_DIR:-$ROOT/results/tmp/$PROJECT}

# With the current zephyr layout, GPUs 0 and 3 are free. The default queues run
# kinematics-only then kinematics+isolation on GPU 0, and kinematics+ID on GPU 3.
GPU0_RUNS=${GPU0_RUNS:-kinematics,kinematics-isolation}
GPU3_RUNS=${GPU3_RUNS:-kinematics-id}
MAX_EPOCHS=${MAX_EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-1024}
MAX_PREPROCESS_OBJECTS=${MAX_PREPROCESS_OBJECTS:-1000000}
LOGGER_OFFLINE=${LOGGER_OFFLINE:-true}
FORCE_PREPROCESSOR=${FORCE_PREPROCESSOR:-0}
DRY_RUN=${DRY_RUN:-0}

mkdir -p "$PREPROCESS_DIR" "$LOG_DIR" "$TMP_DIR"
cd "$ROOT"

MC_LIST="$TMP_DIR/mc_h5_files.txt"
find "$MC_DIR" -maxdepth 1 -type f -name '*.h5' -size +0c | sort > "$MC_LIST"

N_MC=$(wc -l < "$MC_LIST" | tr -d ' ')
if [ "$N_MC" -eq 0 ]; then
  echo "Missing MC H5 files in $MC_DIR" >&2
  exit 1
fi
echo "MC files: $N_MC"

is_selected() {
  local needle=$1
  local list=$2
  case ",$list," in
    *",$needle,"*) return 0 ;;
    *) return 1 ;;
  esac
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

  echo "Fitting MC-only preprocessor: $output_file"
  /root/.pixi/bin/pixi run python scripts/get_atlas_object_preprocessing.py \
    --datamodule-config "$datamodule_config" \
    --h5-files $(tr '\n' ' ' < "$MC_LIST") \
    --object-type electrons \
    --mode log_standard \
    --log-features "$log_features" \
    --max-objects "$MAX_PREPROCESS_OBJECTS" \
    --output-dir "$output_dir" \
    --output-name "$output_name"
}

run_spec_field() {
  local run=$1
  local field=$2
  case "$run:$field" in
    kinematics:tag) echo "kinematics" ;;
    kinematics:config) echo "atlas_event_electron_kinematics" ;;
    kinematics:config_path) echo "$ROOT/configs/datamodule/atlas_event_electron_kinematics.yaml" ;;
    kinematics:run_name) echo "electrons_kinematics_dim8_cb4096_q4_mconly" ;;
    kinematics:preprocessor) echo "$PREPROCESS_DIR/electrons_kinematics_log_standard.joblib" ;;
    kinematics:log_features) echo "pt" ;;

    kinematics-id:tag) echo "kinematics-id" ;;
    kinematics-id:config) echo "atlas_event_electron_kinematics_id" ;;
    kinematics-id:config_path) echo "$ROOT/configs/datamodule/atlas_event_electron_kinematics_id.yaml" ;;
    kinematics-id:run_name) echo "electrons_kinematics_id_dim8_cb4096_q4_mconly" ;;
    kinematics-id:preprocessor) echo "$PREPROCESS_DIR/electrons_kinematics_id_log_standard.joblib" ;;
    kinematics-id:log_features) echo "pt" ;;

    kinematics-isolation:tag) echo "kinematics-isolation" ;;
    kinematics-isolation:config) echo "atlas_event_electron_kinematics_isolation" ;;
    kinematics-isolation:config_path) echo "$ROOT/configs/datamodule/atlas_event_electron_kinematics_isolation.yaml" ;;
    kinematics-isolation:run_name) echo "electrons_kinematics_isolation_dim8_cb4096_q4_mconly" ;;
    kinematics-isolation:preprocessor) echo "$PREPROCESS_DIR/electrons_kinematics_isolation_log_standard.joblib" ;;
    kinematics-isolation:log_features) echo "pt,ptvarcone30" ;;

    *) echo "Unknown run spec: $run:$field" >&2; exit 1 ;;
  esac
}

ALL_RUNS="$GPU0_RUNS,$GPU3_RUNS"
for run in kinematics kinematics-id kinematics-isolation; do
  if is_selected "$run" "$ALL_RUNS"; then
    fit_preprocessor \
      "$(run_spec_field "$run" config_path)" \
      "$(run_spec_field "$run" preprocessor)" \
      "$(run_spec_field "$run" log_features)"
  fi
done

write_queue_script() {
  local gpu=$1
  local queue=$2
  local unit="atlas-electron-feature-mconly-gpu${gpu}-$(date +%Y%m%d-%H%M%S)"
  local job_script="$TMP_DIR/${unit}.sh"

  cat > "$job_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="$gpu"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=$([ "$LOGGER_OFFLINE" = true ] && echo offline || echo online)
export WANDB_SILENT=true

H5_LIST=\$(/root/.pixi/bin/pixi run python - "$MC_LIST" <<'PY'
from pathlib import Path
import sys

files = [x for x in Path(sys.argv[1]).read_text().splitlines() if x]
print("[" + ",".join(files) + "]")
PY
)

run_one() {
  local tag=\$1
  local config_name=\$2
  local run_name=\$3
  local preprocessor=\$4
  local train_log="$LOG_DIR/\${run_name}.log"

  echo "\$(date): starting \$run_name on physical GPU $gpu"
  /root/.pixi/bin/pixi run python scripts/train.py \\
    datamodule="\$config_name" \\
    model=vqvae \\
    callbacks=event_tokenizer \\
    output_dir="$ROOT/results" \\
    project_name="$PROJECT" \\
    network_name="\$run_name" \\
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
    +datamodule.transforms.preprocess.cst_fn.filename="\$preprocessor" \\
    "datamodule.data_paths=\${H5_LIST}" \\
    2>&1 | tee "\$train_log"
  echo "\$(date): finished \$run_name"
}

EOF

  IFS=',' read -r -a queued_runs <<< "$queue"
  for run in "${queued_runs[@]}"; do
    if [ -z "$run" ]; then
      continue
    fi
    cat >> "$job_script" <<EOF
run_one \\
  "$(run_spec_field "$run" tag)" \\
  "$(run_spec_field "$run" config)" \\
  "$(run_spec_field "$run" run_name)" \\
  "$(run_spec_field "$run" preprocessor)"

EOF
  done

  chmod +x "$job_script"

  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY RUN: would submit $unit on physical GPU $gpu"
    echo "  queue: $queue"
    echo "  job: $job_script"
    return
  fi

  systemd-run \
    --unit="$unit" \
    --description="Electron MC-only feature-group controls on GPU $gpu" \
    --collect \
    --property="WorkingDirectory=$ROOT" \
    /bin/bash "$job_script"

  echo "Submitted GPU $gpu queue: $queue"
  echo "  journalctl -u $unit -f"
}

if [ -n "$GPU0_RUNS" ]; then
  write_queue_script 0 "$GPU0_RUNS"
fi
if [ -n "$GPU3_RUNS" ]; then
  write_queue_script 3 "$GPU3_RUNS"
fi

echo
echo "Project: $PROJECT"
echo "GPU 0 queue: $GPU0_RUNS"
echo "GPU 3 queue: $GPU3_RUNS"
echo "Logs: $LOG_DIR"
