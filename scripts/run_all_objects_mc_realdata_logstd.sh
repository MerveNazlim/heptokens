#!/usr/bin/env bash
set -euo pipefail

export HOME=/root
export USER=root
export LOGNAME=root
export WANDB_DIR=/home/magaras/heptok_fork/heptokens/results/wandb_tmp
export WANDB_CACHE_DIR=/home/magaras/heptok_fork/heptokens/results/wandb_cache
export WANDB_CONFIG_DIR=/root/.config/wandb

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
DATA_DIR=${DATA_DIR:-/home/zephyr/Data/viviana/bnl-treasure/data/h5}
REALDATA_DIR=${REALDATA_DIR:-/home/zephyr/Data/viviana/bnl-treasure/data/h5/realdata}
INCLUDE_REALDATA=${INCLUDE_REALDATA:-true}
EXTRA_H5_DIRS=${EXTRA_H5_DIRS:-}
RESULTS=${ROOT}/results
PROJECT=${PROJECT:-atlas_event_tokenizers_0107_logstd_mc_realdata}
LOGDIR=${RESULTS}/logs/${PROJECT}
PREP_DIR=${RESULTS}/preprocessing/${PROJECT}
CONFIG=configs/datamodule/atlas_event_object.yaml

mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$LOGDIR" "$PREP_DIR"
cd "$ROOT"

mapfile -t H5_FILES < <(
  {
    find "$DATA_DIR" -maxdepth 1 -type f -name '*.h5' -size +0c
    if [ "$INCLUDE_REALDATA" = "true" ]; then
      find "$REALDATA_DIR" -maxdepth 1 -type f -name '*.h5' -size +0c
    fi
    for extra_dir in $EXTRA_H5_DIRS; do
      find "$extra_dir" -maxdepth 1 -type f -name '*.h5' -size +0c
    done
  } | sort -u
)

echo "Found ${#H5_FILES[@]} non-empty H5 files"
printf '%s\n' "${H5_FILES[@]:0:10}"
if [ "${#H5_FILES[@]}" -gt 10 ]; then
  echo "..."
  printf '%s\n' "${H5_FILES[@]: -10}"
fi

if [ "${#H5_FILES[@]}" -eq 0 ]; then
  echo "No non-empty H5 files found. DATA_DIR=$DATA_DIR REALDATA_DIR=$REALDATA_DIR EXTRA_H5_DIRS=$EXTRA_H5_DIRS"
  exit 1
fi

echo "Checking for zero-byte H5 files"
ZERO_BYTE_DIRS=("$DATA_DIR")
if [ "$INCLUDE_REALDATA" = "true" ]; then
  ZERO_BYTE_DIRS+=("$REALDATA_DIR")
fi
for extra_dir in $EXTRA_H5_DIRS; do
  ZERO_BYTE_DIRS+=("$extra_dir")
done
find "${ZERO_BYTE_DIRS[@]}" -type f -name '*.h5' -size 0 -print || true

echo "Validating H5 files against $CONFIG"
/root/.pixi/bin/pixi run python - "$CONFIG" "${H5_FILES[@]}" <<'PY'
import sys
import h5py
from omegaconf import OmegaConf

config_path = sys.argv[1]
h5_files = sys.argv[2:]
cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
collections = cfg.get("object_collections") or []
required = []
for collection in collections:
    if collection.get("mask_input"):
        required.append(collection["mask_input"])
    required.extend(collection.get("inputs") or [])

bad = []
for path in h5_files:
    try:
        with h5py.File(path, "r") as handle:
            missing = [key for key in required if key not in handle]
    except Exception as exc:
        bad.append((path, f"cannot open: {exc}"))
        continue
    if missing:
        bad.append((path, "missing " + ", ".join(missing[:12])))

if bad:
    print("Validation failed. First bad files:")
    for path, reason in bad[:20]:
        print(f"  {path}: {reason}")
    raise SystemExit(1)

print(f"Validation passed for {len(h5_files)} files and {len(required)} required paths")
PY

fit_preprocessing () {
  OBJ=$1
  LOG_FEATURES=$2
  OUT_NAME=$3

  if [ -f "${PREP_DIR}/${OUT_NAME}.joblib" ]; then
    echo "Skipping existing preprocessing ${OUT_NAME}"
    return 0
  fi

  echo "Fitting preprocessing for ${OBJ}: log features ${LOG_FEATURES}"
  /root/.pixi/bin/pixi run python scripts/get_atlas_object_preprocessing.py \
    --h5-files "${H5_FILES[@]}" \
    --datamodule-config "$CONFIG" \
    --object-type "$OBJ" \
    --mode log_standard \
    --log-features "$LOG_FEATURES" \
    --max-objects 1000000 \
    --output-dir "$PREP_DIR" \
    --output-name "$OUT_NAME" \
    > "${LOGDIR}/preprocess_${OUT_NAME}.log" 2>&1
}

fit_preprocessing jets "pt,mass,n_trk,QG_nTracks" jets_log_standard
fit_preprocessing electrons "pt,ptvarcone30" electrons_log_standard
fit_preprocessing muons "pt,ptvarcone30" muons_log_standard
fit_preprocessing photons "pt,ptcone20" photons_log_standard
fit_preprocessing taus "pt" taus_log_standard
fit_preprocessing tracks "pt,chiSquared" tracks_log_standard_no_ndoflog

H5_LIST=$(IFS=,; echo "[${H5_FILES[*]}]")

run_one () {
  OBJ=$1
  GPU=$2
  CODEBOOK_DIM=$3
  CODEBOOK_SIZE=$4
  NUM_QUANTIZERS=$5
  PREPROCESSOR=$6

  RUN_NAME="${OBJ}_logstd_dim${CODEBOOK_DIM}_cb${CODEBOOK_SIZE}_q${NUM_QUANTIZERS}"
  RUN_DIR="${RESULTS}/${PROJECT}/${RUN_NAME}"

  if [ -f "${RUN_DIR}/SUCCESS.txt" ]; then
    echo "Skipping completed ${RUN_NAME}"
    return 0
  fi

  echo "$(date): Starting ${RUN_NAME} on GPU ${GPU}"
  if ! CUDA_VISIBLE_DEVICES=$GPU /root/.pixi/bin/pixi run python scripts/train.py \
    datamodule=atlas_event_object \
    model=vqvae \
    callbacks=event_tokenizer \
    output_dir="$RESULTS" \
    project_name="$PROJECT" \
    network_name="$RUN_NAME" \
    logger.offline=false \
    trainer.max_epochs=20 \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    trainer.limit_val_batches=1000 \
    datamodule.object_type="$OBJ" \
    model.codebook_dim=$CODEBOOK_DIM \
    model.codebook_size=$CODEBOOK_SIZE \
    model.num_quantizers=$NUM_QUANTIZERS \
    model.dead_code_reset=false \
    model.data_codebook_init=false \
    +datamodule.transforms.preprocess._target_=heptokens.data.collation.preprocess_objects_batch \
    +datamodule.transforms.preprocess._partial_=true \
    +datamodule.transforms.preprocess.cst_fn._target_=joblib.load \
    +datamodule.transforms.preprocess.cst_fn.filename="$PREPROCESSOR" \
    "datamodule.data_paths=${H5_LIST}" \
    > "${LOGDIR}/${RUN_NAME}.log" 2>&1; then
    echo "$(date): FAILED ${RUN_NAME}. Check ${LOGDIR}/${RUN_NAME}.log"
    return 1
  fi

  echo "$(date): Finished ${RUN_NAME}"
}

STATUS=0

run_one jets 0 8 4096 4 "${PREP_DIR}/jets_log_standard.joblib" &
PID0=$!

run_one electrons 1 8 4096 4 "${PREP_DIR}/electrons_log_standard.joblib" &
PID1=$!

run_one muons 2 8 4096 4 "${PREP_DIR}/muons_log_standard.joblib" &
PID2=$!

run_one photons 3 8 4096 4 "${PREP_DIR}/photons_log_standard.joblib" &
PID3=$!

for PID in "$PID0" "$PID1" "$PID2" "$PID3"; do
  if ! wait "$PID"; then
    STATUS=1
  fi
done

if [ "$STATUS" -ne 0 ]; then
  echo "First tokenizer wave failed. Check ${LOGDIR}"
  exit "$STATUS"
fi

run_one taus 0 8 4096 4 "${PREP_DIR}/taus_log_standard.joblib" &
PID4=$!

run_one taus 1 8 8192 4 "${PREP_DIR}/taus_log_standard.joblib" &
PID5=$!

run_one tracks 2 8 8192 4 "${PREP_DIR}/tracks_log_standard_no_ndoflog.joblib" &
PID6=$!

for PID in "$PID4" "$PID5" "$PID6"; do
  if ! wait "$PID"; then
    STATUS=1
  fi
done

if [ "$STATUS" -ne 0 ]; then
  echo "Tokenizer scan finished with failures. Check ${LOGDIR}"
  exit "$STATUS"
fi

echo "All preprocessing and tokenizers finished successfully"
