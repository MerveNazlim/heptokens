#!/usr/bin/env bash
# Historical launch reference only; not a runnable recipe.
set -euo pipefail

CLUSTER_ID=${1:?Missing Condor ClusterId}
PROC_ID=${2:?Missing Condor ProcId}

GCS_PREPARED_PREFIX=${GCS_PREPARED_PREFIX:-gs://EXAMPLE-PROJECT-BUCKET/bucket_results/foundation_pretrain/data_grl_data16_200m_v1/continuous}
GCS_OUTPUT_PREFIX=${GCS_OUTPUT_PREFIX:-gs://EXAMPLE-PROJECT-BUCKET/bucket_results/foundation_pretrain/data_grl_data16_200m_v1_pretrain/hierarchical_kin_aux_v1/smoke}
PROJECT_ARCHIVE=${PROJECT_ARCHIVE:-heptokens-q4-latest-hier-seed-base-2be1e53c.tar.gz}
SERVICE_ACCOUNT_KEY=${SERVICE_ACCOUNT_KEY:-example-worker-service-account.json}
B300_PIXI_MANIFEST=${B300_PIXI_MANIFEST:-q4-latest-hier-seed42-pixi.toml}
B300_PIXI_LOCK=${B300_PIXI_LOCK:-q4-latest-hier-seed42-pixi.lock}
EXPECTED_ARCHIVE_SHA256=${EXPECTED_ARCHIVE_SHA256:-2be1e53caa4eabe65a878d5bbea3d84a1118618585b5a6a6e9b52327869f4b2f}
GROUPING_UPDATE_ARCHIVE=${GROUPING_UPDATE_ARCHIVE:-heptokens-q4-latest-hier-seed-hierarchical_kin_aux-original-overlay.tar.gz}
EXPECTED_GROUPING_SHA256=497ac79701f35f355d62c6fe2cfcc5098b8b9536858b128b8e45d98c6f609b39
EXPECTED_MANIFEST_SHA256=${EXPECTED_MANIFEST_SHA256:-6e4d519a003efd3fc8e4d79a01d761323dd25dd0affa7b48be01f2dce1a14628}

EXPECTED_TOTAL_ROWS=${EXPECTED_TOTAL_ROWS:-216095647}
EXPECTED_TRAIN_ROWS=${EXPECTED_TRAIN_ROWS:-194491678}
EXPECTED_VAL_ROWS=${EXPECTED_VAL_ROWS:-21603969}
EXPECTED_TRAIN_SHARDS=${EXPECTED_TRAIN_SHARDS:-3898}
EXPECTED_VAL_SHARDS=${EXPECTED_VAL_SHARDS:-440}
EXPECTED_EXCLUDED_ROWS=${EXPECTED_EXCLUDED_ROWS:-none}
EXPECTED_INVALID_INPUT_FILE_COUNT=${EXPECTED_INVALID_INPUT_FILE_COUNT:-0}

DATASET_MODE=${DATASET_MODE:-smoke}
case "$DATASET_MODE" in
  smoke|full) ;;
  *) echo "DATASET_MODE must be smoke or full" >&2; exit 2 ;;
esac

DOWNLOAD_TRAIN_SHARDS=${DOWNLOAD_TRAIN_SHARDS:-32}
DOWNLOAD_VAL_SHARDS=${DOWNLOAD_VAL_SHARDS:-8}
SEED=${SEED:?Missing pretraining seed}
[[ "$SEED" == 43 || "$SEED" == 44 ]] || { echo "Only seeds 43/44 are allowed" >&2; exit 2; }
EPOCHS=${EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-597}
NUM_WORKERS=${NUM_WORKERS:-3}
NUM_DEVICES=${NUM_DEVICES:-6}
TRAINER_STRATEGY=${TRAINER_STRATEGY:-ddp}
STREAM_BATCH_SIZE=${STREAM_BATCH_SIZE:-4096}
SHUFFLE_BUFFER_SIZE=${SHUFFLE_BUFFER_SIZE:-8192}
CONTINUOUS_FEATURE_COLUMN=${CONTINUOUS_FEATURE_COLUMN:-continuous_features}
MODEL_CONFIG=${MODEL_CONFIG:-foundation_continuous_hierarchical_kin_aux_pretrain}
CONTINUOUS_ARCHITECTURE=${CONTINUOUS_ARCHITECTURE:-hierarchical}
LIMIT_TRAIN_BATCHES=${LIMIT_TRAIN_BATCHES:-200}
LIMIT_VAL_BATCHES=${LIMIT_VAL_BATCHES:-20}
CHECKPOINT_EVERY_N_STEPS=${CHECKPOINT_EVERY_N_STEPS:-2000}
CHECKPOINT_SYNC_INTERVAL=${CHECKPOINT_SYNC_INTERVAL:-300}
RESUME_CHECKPOINT_URI=${RESUME_CHECKPOINT_URI:-}
PROJECT_NAME=${PROJECT_NAME:-atlas_event_foundation_continuous}
NETWORK_NAME=${NETWORK_NAME:-hierarchical_kin_aux_b597_6gpu_200step_seed42}

# Full runs must download the entire prepared dataset and use every loader
# batch. In Lightning, the floating-point 1.0 means 100%; integer 1 means one.
if [[ "$DATASET_MODE" == full ]]; then
  if [[ "$DOWNLOAD_TRAIN_SHARDS" -ne "$EXPECTED_TRAIN_SHARDS" || \
        "$DOWNLOAD_VAL_SHARDS" -ne "$EXPECTED_VAL_SHARDS" || \
        "$LIMIT_TRAIN_BATCHES" != 1.0 ]]; then
    echo "Full mode requires all expected train/val shards and LIMIT_TRAIN_BATCHES=1.0" >&2
    exit 2
  fi
fi

SCRATCH_DIR=${_CONDOR_SCRATCH_DIR:-$PWD}
WORK_DIR="$SCRATCH_DIR/work"
PREPARED_DIR="$WORK_DIR/prepared"
RESULTS_DIR="$WORK_DIR/results"
PIXI_HOME="$WORK_DIR/.pixi"
KEY_FILE="$SCRATCH_DIR/$SERVICE_ACCOUNT_KEY"
GROUPING_UPDATE_FILE="$SCRATCH_DIR/$GROUPING_UPDATE_ARCHIVE"
B300_PIXI_MANIFEST_FILE="$SCRATCH_DIR/$B300_PIXI_MANIFEST"
B300_PIXI_LOCK_FILE="$SCRATCH_DIR/$B300_PIXI_LOCK"
DESTINATION="$GCS_OUTPUT_PREFIX/cluster_${CLUSTER_ID}_proc_${PROC_ID}"
# Retain the original wrapper's same-job restart, never another seed/job.
[[ -z "$RESUME_CHECKPOINT_URI" ]] || { echo "External starting checkpoints are not allowed" >&2; exit 2; }
AUTO_RESUME_CHECKPOINT_URI=${AUTO_RESUME_CHECKPOINT_URI:-$DESTINATION/checkpoints/resume.ckpt}
[[ "$AUTO_RESUME_CHECKPOINT_URI" == "$DESTINATION/checkpoints/resume.ckpt" ]] || { echo "External auto-resume path is not allowed" >&2; exit 2; }
GPU_MONITOR_PID=
CHECKPOINT_SYNC_PID=
ACTUAL_MANIFEST_SHA256=unavailable
ACTUAL_ARCHIVE_SHA256=unavailable
RESUME_SOURCE=

export XDG_CACHE_HOME="$WORK_DIR/cache"
export XDG_CONFIG_HOME="$WORK_DIR/config"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export CLOUDSDK_CONFIG="$WORK_DIR/gcloud-config"
mkdir -p "$PREPARED_DIR/train" "$PREPARED_DIR/val" "$RESULTS_DIR" \
  "$WORK_DIR/home" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$UV_CACHE_DIR" \
  "$CLOUDSDK_CONFIG"

upload_results() {
  local rc=$?
  trap - EXIT
  set +e
  if [[ -n "$GPU_MONITOR_PID" ]]; then
    kill "$GPU_MONITOR_PID" 2>/dev/null
    wait "$GPU_MONITOR_PID" 2>/dev/null
  fi
  if [[ -n "$CHECKPOINT_SYNC_PID" ]]; then
    kill "$CHECKPOINT_SYNC_PID" 2>/dev/null
    wait "$CHECKPOINT_SYNC_PID" 2>/dev/null
  fi
  {
    echo "exit_code=$rc"
    echo "finished=$(date --iso-8601=seconds)"
    echo "worker=$(hostname --fqdn 2>/dev/null || hostname)"
    echo "seed=$SEED"
    echo "pretraining_family=hierarchical_kin_aux"
    echo "seed42_reference_job=287.0"
    echo "fine_tuning_seed=42"
    echo "epochs=$EPOCHS"
    echo "expected_global_steps=${EXPECTED_GLOBAL_STEPS:-unavailable}"
    echo "batch_size_per_gpu=$BATCH_SIZE"
    echo "num_devices=$NUM_DEVICES"
    echo "global_batch_size=$((BATCH_SIZE * NUM_DEVICES))"
    echo "limit_train_batches=$LIMIT_TRAIN_BATCHES"
    echo "limit_val_batches=$LIMIT_VAL_BATCHES"
    echo "downloaded_train_shards=$DOWNLOAD_TRAIN_SHARDS"
    echo "downloaded_val_shards=$DOWNLOAD_VAL_SHARDS"
    echo "dataset_mode=$DATASET_MODE"
    echo "include_tokens=false"
    echo "prepared_prefix=$GCS_PREPARED_PREFIX"
    echo "continuous_feature_column=$CONTINUOUS_FEATURE_COLUMN"
    echo "model_config=$MODEL_CONFIG"
    echo "continuous_architecture=$CONTINUOUS_ARCHITECTURE"
    echo "resume_checkpoint_uri=${RESUME_SOURCE:-none}"
    echo "auto_resume_checkpoint_uri=$AUTO_RESUME_CHECKPOINT_URI"
    echo "manifest_sha256=$ACTUAL_MANIFEST_SHA256"
    echo "archive_sha256=$ACTUAL_ARCHIVE_SHA256"
    echo "feature_grouping=kinematics_auxiliary_v1"
    echo "grouping_overlay_sha256=$EXPECTED_GROUPING_SHA256"
  } > "$RESULTS_DIR/job_status.txt"

  echo "Uploading results to $DESTINATION"
  gcloud storage rsync --recursive "$RESULTS_DIR" "$DESTINATION"
  local upload_rc=$?
  [[ "$CLOUDSDK_CONFIG" == "$WORK_DIR/gcloud-config" ]] && rm -rf "$CLOUDSDK_CONFIG"
  if [[ $rc -eq 0 && $upload_rc -ne 0 ]]; then rc=$upload_rc; fi
  exit "$rc"
}
trap upload_results EXIT

for required in "$KEY_FILE" "$B300_PIXI_MANIFEST_FILE" \
  "$B300_PIXI_LOCK_FILE" "$PROJECT_ARCHIVE" "$GROUPING_UPDATE_FILE" \
  "$SCRATCH_DIR/q4-latest-hier-seed42-nccl.py"; do
  [[ -s "$required" ]] || {
    echo "Missing required input: $required" >&2
    exit 2
  }
done

chmod 600 "$KEY_FILE"
gcloud auth activate-service-account --key-file="$KEY_FILE"

echo "=== Six-GPU hierarchical continuous pretraining ($DATASET_MODE) ==="
echo "Started: $(date --iso-8601=seconds)"
echo "Worker: $(hostname --fqdn 2>/dev/null || hostname)"
echo "Prepared input: $GCS_PREPARED_PREFIX"
echo "Results: $DESTINATION"
echo "Per-GPU/global batch: $BATCH_SIZE/$((BATCH_SIZE * NUM_DEVICES))"
echo "Downloaded shards: train=$DOWNLOAD_TRAIN_SHARDS val=$DOWNLOAD_VAL_SHARDS"
echo "Continuous column: $CONTINUOUS_FEATURE_COLUMN"
echo "Model configuration: $MODEL_CONFIG"
echo "Continuous architecture: $CONTINUOUS_ARCHITECTURE"
echo "include_tokens=false"
nvidia-smi -L

# Respect Condor's CUDA_VISIBLE_DEVICES; never choose host GPU indices.
GPU_SELECTION_ARGS=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPU_SELECTION_ARGS=("--id=$CUDA_VISIBLE_DEVICES")
fi

GPU_LOG="$RESULTS_DIR/gpu_memory.csv"
echo "timestamp,index,name,memory_used_MiB,memory_total_MiB,gpu_utilization_percent" \
  > "$GPU_LOG"
(
  while true; do
    nvidia-smi "${GPU_SELECTION_ARGS[@]}" \
      --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits >> "$GPU_LOG"
    sleep 5
  done
) &
GPU_MONITOR_PID=$!

gcloud storage cp "$GCS_PREPARED_PREFIX/manifest.json" \
  "$PREPARED_DIR/manifest.json"
ACTUAL_MANIFEST_SHA256=$(sha256sum "$PREPARED_DIR/manifest.json" | awk '{print $1}')
if [[ -n "$EXPECTED_MANIFEST_SHA256" && \
      "$ACTUAL_MANIFEST_SHA256" != "$EXPECTED_MANIFEST_SHA256" ]]; then
  echo "Manifest checksum mismatch: expected=$EXPECTED_MANIFEST_SHA256 actual=$ACTUAL_MANIFEST_SHA256" >&2
  exit 2
fi

python3 - "$PREPARED_DIR/manifest.json" \
  "$EXPECTED_TOTAL_ROWS" "$EXPECTED_TRAIN_ROWS" "$EXPECTED_VAL_ROWS" \
  "$EXPECTED_TRAIN_SHARDS" "$EXPECTED_VAL_SHARDS" \
  "$EXPECTED_EXCLUDED_ROWS" "$EXPECTED_INVALID_INPUT_FILE_COUNT" <<'PY'
import json
import sys

(
    path,
    total,
    train,
    valid,
    train_shards,
    val_shards,
    excluded,
    invalid_input_file_count,
) = sys.argv[1:]
manifest = json.load(open(path))
expected = {
    "total_rows": int(total),
    "train_rows": int(train),
    "val_rows": int(valid),
    "train_shards": int(train_shards),
    "val_shards": int(val_shards),
}
if excluded.lower() not in {"", "none", "null", "skip"}:
    expected["excluded_rows"] = int(excluded)
if invalid_input_file_count.lower() not in {"", "none", "null", "skip"}:
    expected["invalid_input_file_count"] = int(invalid_input_file_count)
errors = [
    f"{key}: observed={manifest.get(key)!r}, expected={value!r}"
    for key, value in expected.items()
    if manifest.get(key) != value
]
if errors:
    raise SystemExit("Continuous manifest mismatch:\n  " + "\n  ".join(errors))
print("PASS: continuous production manifest matches NERSC")
print(json.dumps(expected, indent=2))
PY

download_split() {
  local split=$1
  local requested=$2
  local listing="$WORK_DIR/${split}_objects.txt"
  local selected="$WORK_DIR/${split}_selected.txt"

  gcloud storage ls --recursive "$GCS_PREPARED_PREFIX/$split/" \
    | grep -E '\.parquet$' | LC_ALL=C sort > "$listing"
  local available
  available=$(wc -l < "$listing")
  if [[ "$DATASET_MODE" == full && "$available" -ne "$requested" ]]; then
    echo "Full dataset mismatch: $available $split shards available; expected exactly $requested" >&2
    exit 2
  fi
  if [[ "$available" -lt "$requested" ]]; then
    echo "Only $available $split shards are currently available; need $requested" >&2
    exit 2
  fi
  head -n "$requested" "$listing" > "$selected"
  mapfile -t objects < "$selected"
  gcloud storage cp "${objects[@]}" "$PREPARED_DIR/$split/"
}

download_split train "$DOWNLOAD_TRAIN_SHARDS"
download_split val "$DOWNLOAD_VAL_SHARDS"

ACTUAL_TRAIN_SHARDS=$(find "$PREPARED_DIR/train" -type f -name '*.parquet' | wc -l)
ACTUAL_VAL_SHARDS=$(find "$PREPARED_DIR/val" -type f -name '*.parquet' | wc -l)
if [[ "$ACTUAL_TRAIN_SHARDS" -ne "$DOWNLOAD_TRAIN_SHARDS" || \
      "$ACTUAL_VAL_SHARDS" -ne "$DOWNLOAD_VAL_SHARDS" ]]; then
  echo "Downloaded shard-count mismatch: train=$ACTUAL_TRAIN_SHARDS val=$ACTUAL_VAL_SHARDS" >&2
  exit 2
fi

ACTUAL_ARCHIVE_SHA256=$(sha256sum "$PROJECT_ARCHIVE" | awk '{print $1}')
if [[ -n "$EXPECTED_ARCHIVE_SHA256" && \
      "$ACTUAL_ARCHIVE_SHA256" != "$EXPECTED_ARCHIVE_SHA256" ]]; then
  echo "Archive checksum mismatch: expected=$EXPECTED_ARCHIVE_SHA256 actual=$ACTUAL_ARCHIVE_SHA256" >&2
  exit 2
fi

mkdir -p "$WORK_DIR/project"
tar -xzf "$PROJECT_ARCHIVE" -C "$WORK_DIR/project"
PYPROJECT_PATH=$(find "$WORK_DIR/project" -maxdepth 3 -type f \
  -name pyproject.toml -print -quit)
[[ -n "$PYPROJECT_PATH" ]] || {
  echo "pyproject.toml not found in project archive" >&2
  exit 2
}
REPO_DIR=$(dirname "$PYPROJECT_PATH")
ACTUAL_GROUPING_SHA256=$(sha256sum "$GROUPING_UPDATE_FILE" | awk '{print $1}')
[[ "$ACTUAL_GROUPING_SHA256" == "$EXPECTED_GROUPING_SHA256" ]] || {
  echo "Grouping overlay checksum mismatch" >&2; exit 2;
}
tar -xzf "$GROUPING_UPDATE_FILE" -C "$REPO_DIR"

if command -v pixi >/dev/null 2>&1; then
  PIXI=$(command -v pixi)
else
  export PIXI_HOME
  curl --retry 5 --retry-all-errors --retry-delay 5 -fsSL \
    https://pixi.sh/install.sh -o "$WORK_DIR/install-pixi.sh"
  bash "$WORK_DIR/install-pixi.sh"
  PIXI="$PIXI_HOME/bin/pixi"
fi

cd "$REPO_DIR"
[[ ! -f pixi.lock ]] || mv pixi.lock "$RESULTS_DIR/pixi.lock.original"
cp "$B300_PIXI_MANIFEST_FILE" pixi.toml
cp "$B300_PIXI_LOCK_FILE" pixi.lock
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

"$PIXI" install --manifest-path pixi.toml --platform linux-64 --locked \
  2>&1 | tee "$RESULTS_DIR/pixi_install.log"

PIXI_ENV_LIB="$REPO_DIR/.pixi/envs/default/lib"
PIXI_LD_LIBRARY_PATH="$PIXI_ENV_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PIXI_RUN=(
  env "LD_LIBRARY_PATH=$PIXI_LD_LIBRARY_PATH"
  "$PIXI" run --manifest-path pixi.toml --platform linux-64 --locked
)

# Validate the actual Parquet metadata. Full jobs retain the source manifest
# byte-for-byte; only smoke jobs need a manifest describing their subset.
cp "$PREPARED_DIR/manifest.json" "$PREPARED_DIR/source_manifest.json"
"${PIXI_RUN[@]}" python - "$PREPARED_DIR" "$DATASET_MODE" <<'PY' \
  | tee "$RESULTS_DIR/dataset_manifest_report.txt"
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

root = Path(sys.argv[1])
mode = sys.argv[2]
source_path = root / "source_manifest.json"
manifest = json.loads(source_path.read_text())
train = sorted((root / "train").glob("*.parquet"))
valid = sorted((root / "val").glob("*.parquet"))
train_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in train)
val_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in valid)
observed = {
    "total_rows": train_rows + val_rows,
    "train_rows": train_rows,
    "val_rows": val_rows,
    "train_shards": len(train),
    "val_shards": len(valid),
}
if mode == "full":
    errors = [
        f"{key}: downloaded={value}, manifest={manifest.get(key)!r}"
        for key, value in observed.items()
        if manifest.get(key) != value
    ]
    if manifest.get("smoke_subset"):
        errors.append("Production manifest is marked smoke_subset")
    if errors:
        raise SystemExit("Full dataset mismatch:\n  " + "\n  ".join(errors))
    # Keep the original bytes, including excluded_rows and split provenance.
else:
    manifest.update(observed)
    manifest.update({"excluded_rows": 0, "smoke_subset": True})
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(
    f"PASS: {mode} manifest train={train_rows} rows/{len(train)} shards "
    f"val={val_rows} rows/{len(valid)} shards"
)
PY
cp "$PREPARED_DIR/source_manifest.json" \
  "$RESULTS_DIR/production_source_manifest.json"
cp "$PREPARED_DIR/manifest.json" "$RESULTS_DIR/dataset_manifest.json"
if [[ "$DATASET_MODE" == smoke ]]; then
  cp "$PREPARED_DIR/manifest.json" "$RESULTS_DIR/smoke_subset_manifest.json"
fi

"${PIXI_RUN[@]}" python - "$NUM_DEVICES" <<'PY' | tee "$RESULTS_DIR/pytorch_runtime.txt"
import sys
import torch

print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"nccl={torch.cuda.nccl.version()}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
for index in range(torch.cuda.device_count()):
    print(
        f"gpu[{index}]={torch.cuda.get_device_name(index)} "
        f"capability={torch.cuda.get_device_capability(index)}"
    )
if not torch.cuda.is_available() or torch.cuda.device_count() != int(sys.argv[1]):
    raise SystemExit(f"Expected {sys.argv[1]} assigned CUDA devices; found {torch.cuda.device_count()}")
if not torch.__version__.startswith("2.11.0") or torch.version.cuda != "13.0":
    raise SystemExit("Unexpected B300 PyTorch/CUDA runtime")
PY

echo "Running six-GPU NCCL all-reduce preflight"
NCCL_DEBUG=INFO "${PIXI_RUN[@]}" torchrun --standalone \
  --nproc_per_node="$NUM_DEVICES" "$SCRATCH_DIR/q4-latest-hier-seed42-nccl.py" \
  2>&1 | tee "$RESULTS_DIR/nccl_preflight.log"

"${PIXI_RUN[@]}" python - "$PREPARED_DIR" \
  "$CONTINUOUS_FEATURE_COLUMN" <<'PY' \
  | tee "$RESULTS_DIR/continuous_schema_preflight.txt"
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

root = Path(sys.argv[1])
feature_column = sys.argv[2]
reference_schema = None
for split in ("train", "val"):
    path = sorted((root / split).glob("*.parquet"))[0]
    parquet = pq.ParquetFile(path)
    names = set(parquet.schema_arrow.names)
    required = {
        "mask",
        "type_ids",
        feature_column,
        "continuous_feature_mask",
        "position_role_ids",
    }
    missing = required - names
    if missing:
        raise SystemExit(f"{split} shard missing columns: {sorted(missing)}")
    metadata = parquet.schema_arrow.metadata or {}
    encoded = metadata.get(b"heptokens_continuous_schema")
    if encoded is None:
        raise SystemExit(f"{split} shard has no continuous schema")
    schema = json.loads(encoded)
    if reference_schema is None:
        reference_schema = schema
    elif schema != reference_schema:
        raise SystemExit("Train and validation continuous schemas differ")
    batch = next(parquet.iter_batches(batch_size=2, columns=[feature_column]))
    values = np.asarray(batch.column(0).to_pylist(), dtype=np.float32)
    if values.shape[1:] != (256, 14):
        raise SystemExit(f"Unexpected continuous shape: {values.shape}")
print("PASS: continuous train/validation schemas and [256,14] shape verified")
print("include_tokens=false")
PY

cp pixi.lock "$RESULTS_DIR/pixi.lock.resolved"

"${PIXI_RUN[@]}" python -m unittest discover -s tests \
  -p test_hierarchical_kin_aux_grouping.py -v \
  2>&1 | tee "$RESULTS_DIR/grouping_unit_tests.log"
"${PIXI_RUN[@]}" python scripts/validate_hierarchical_kin_aux_run.py \
  --prepared-dir "$PREPARED_DIR" --output-dir "$RESULTS_DIR" \
  2>&1 | tee "$RESULTS_DIR/grouping_preflight.log"
"${PIXI_RUN[@]}" python "$SCRATCH_DIR/q4_latest_hier_seed_validate.py" epoch-plan \
  --family hierarchical_kin_aux --prepared-dir "$PREPARED_DIR" --seed "$SEED" \
  --batch "$BATCH_SIZE" --workers "$NUM_WORKERS" --devices "$NUM_DEVICES" --epochs "$EPOCHS" \
  --output "$RESULTS_DIR/epoch_plan.json" \
  2>&1 | tee "$RESULTS_DIR/epoch_plan_check.txt"
EXPECTED_GLOBAL_STEPS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["expected_global_steps"])' "$RESULTS_DIR/epoch_plan.json")

export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
export WANDB_INIT_TIMEOUT=300
export WANDB_SILENT=true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

RESUME_ARGS=()
if [[ -n "$RESUME_CHECKPOINT_URI" ]]; then
  RESUME_SOURCE="$RESUME_CHECKPOINT_URI"
elif gcloud storage ls "$AUTO_RESUME_CHECKPOINT_URI" >/dev/null 2>&1; then
  RESUME_SOURCE="$AUTO_RESUME_CHECKPOINT_URI"
  echo "Found automatic restart checkpoint: $RESUME_SOURCE"
fi
if [[ -n "$RESUME_SOURCE" ]]; then
  RESUME_DIR="$WORK_DIR/resume"
  RESUME_CHECKPOINT="$RESUME_DIR/resume.ckpt"
  mkdir -p "$RESUME_DIR"
  echo "Downloading resume checkpoint: $RESUME_SOURCE"
  gcloud storage cp "$RESUME_SOURCE" "$RESUME_CHECKPOINT"
  [[ -s "$RESUME_CHECKPOINT" ]] || {
    echo "Resume checkpoint is missing or empty" >&2
    exit 2
  }
  RESUME_ARGS+=("ckpt_path=$RESUME_CHECKPOINT")
fi

(
  while sleep "$CHECKPOINT_SYNC_INTERVAL"; do
    [[ -f "$RESULTS_DIR/training.log" ]] && \
      gcloud storage cp "$RESULTS_DIR/training.log" \
        "$DESTINATION/training.log" || true
    [[ -f "$GPU_LOG" ]] && gcloud storage cp "$GPU_LOG" \
      "$DESTINATION/gpu_memory.csv" || true
    live_metrics=$(find "$RESULTS_DIR" -type f -name live_metrics.csv -print -quit)
    [[ -n "$live_metrics" ]] && gcloud storage cp "$live_metrics" \
      "$DESTINATION/live_metrics.csv" || true
    checkpoint=$(
      find "$RESULTS_DIR" -type f -name '*.ckpt' -printf '%T@ %p\n' \
        | sort -nr | head -n 1 | cut -d' ' -f2-
    )
    [[ -n "$checkpoint" ]] || continue
    first=$(stat -c '%s:%Y' "$checkpoint" 2>/dev/null) || continue
    sleep 5
    second=$(stat -c '%s:%Y' "$checkpoint" 2>/dev/null) || continue
    [[ "$first" == "$second" ]] || continue
    gcloud storage cp "$checkpoint" \
      "$AUTO_RESUME_CHECKPOINT_URI" || true
  done
) &
CHECKPOINT_SYNC_PID=$!

"${PIXI_RUN[@]}" python -u scripts/benchmark_train.py \
  seed="$SEED" \
  datamodule=token_parquet_continuous_pretrain \
  datamodule.seed="$SEED" \
  model="$MODEL_CONFIG" \
  callbacks=nersc_grouped_pretrain \
  output_dir="$RESULTS_DIR" \
  project_name="$PROJECT_NAME" \
  network_name="$NETWORK_NAME" \
  logger.offline=true \
  datamodule.prepared_dir="$PREPARED_DIR" \
  datamodule.continuous_feature_column="$CONTINUOUS_FEATURE_COLUMN" \
  datamodule.include_tokens=false \
  datamodule.batch_size="$BATCH_SIZE" \
  datamodule.num_workers="$NUM_WORKERS" \
  datamodule.stream_batch_size="$STREAM_BATCH_SIZE" \
  datamodule.shuffle_buffer_size="$SHUFFLE_BUFFER_SIZE" \
  model.architecture="$CONTINUOUS_ARCHITECTURE" \
  model.hidden_dim=256 \
  model.num_heads=8 \
  model.num_layers=4 \
  model.inner_num_heads=4 \
  model.inner_num_layers=1 \
  model.dropout=0.1 \
  model.max_seq_length=256 \
  model.num_type_ids=12 \
  model.mask_prob=0.15 \
  model.use_type_embedding=true \
  model.use_position_embedding=true \
  model.learning_rate=1.0e-4 \
  model.optimizer.lr=1.0e-4 \
  model.optimizer.weight_decay=1.0e-2 \
  model.scheduler.warmup_epochs=1 \
  model.scheduler.min_lr=1.0e-6 \
  trainer.max_epochs="$EPOCHS" \
  trainer.check_val_every_n_epoch=1 \
  trainer.val_check_interval=1.0 \
  trainer.limit_val_batches="$LIMIT_VAL_BATCHES" \
  +trainer.limit_train_batches="$LIMIT_TRAIN_BATCHES" \
  +trainer.log_every_n_steps=20 \
  trainer.accelerator=gpu \
  trainer.devices="$NUM_DEVICES" \
  trainer.num_nodes=1 \
  +trainer.strategy="$TRAINER_STRATEGY" \
  callbacks.periodic_checkpoint.every_n_train_steps="$CHECKPOINT_EVERY_N_STEPS" \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee "$RESULTS_DIR/training.log"

RUN_DIR="$RESULTS_DIR/$PROJECT_NAME/$NETWORK_NAME"
[[ -s "$RUN_DIR/SUCCESS.txt" ]] || {
  echo "Training returned without creating $RUN_DIR/SUCCESS.txt" >&2
  exit 2
}
"${PIXI_RUN[@]}" python scripts/validate_hierarchical_kin_aux_run.py \
  --run-dir "$RUN_DIR" --output-dir "$RESULTS_DIR" \
  --epochs "$EPOCHS" --mode "$DATASET_MODE" \
  2>&1 | tee "$RESULTS_DIR/grouping_completion_check.log"
"${PIXI_RUN[@]}" python "$SCRATCH_DIR/q4_latest_hier_seed_validate.py" completed \
  --family hierarchical_kin_aux --run-dir "$RUN_DIR" --seed "$SEED" \
  --epoch-plan "$RESULTS_DIR/epoch_plan.json" --output "$RESULTS_DIR/seed_training_validation.json" \
  2>&1 | tee "$RESULTS_DIR/seed_training_validation.log"
echo "Latest hierarchical kin/aux seed-replicate pretraining completed successfully"
