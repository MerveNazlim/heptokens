#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
MC_DIR=${MC_DIR:-/home/zephyr/Data/viviana/bnl-treasure/data/h5}
DATA_DIR=${DATA_DIR:-/home/zephyr/Data/viviana/bnl-treasure/data/h5/realdata}
RESULTS=${RESULTS:-${ROOT}/results}

PROJECT=${PROJECT:-atlas_electron_sampling_study}
PREPROCESSOR=${PREPROCESSOR:-${RESULTS}/preprocessing/atlas_event_tokenizers_0107_logstd_mc_realdata/electrons_log_standard.joblib}
MAX_EPOCHS=${MAX_EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-1024}
LIMIT_VAL_BATCHES=${LIMIT_VAL_BATCHES:-1000}
LOGGER_OFFLINE=${LOGGER_OFFLINE:-false}
UNIT_PREFIX=${UNIT_PREFIX:-atlas-electron-sampling-$(date +%Y%m%d-%H%M%S)}
DRY_RUN=${DRY_RUN:-0}
RUNS=${RUNS:-natural,balanced50,mcheavy75}

LOGDIR=${RESULTS}/logs/${PROJECT}
TMPDIR=${RESULTS}/tmp/${PROJECT}
MC_FILE_LIST=${TMPDIR}/mc_files.txt
DATA_FILE_LIST=${TMPDIR}/data_files.txt
ALL_FILE_LIST=${TMPDIR}/all_files.txt
DOMAIN_LIST_FILE=${TMPDIR}/domains.txt

mkdir -p "$LOGDIR" "$TMPDIR"
cd "$ROOT"

if [ ! -f "$PREPROCESSOR" ]; then
  echo "Missing electron preprocessor: $PREPROCESSOR"
  exit 1
fi

find "$MC_DIR" -maxdepth 1 -type f -name '*.h5' -size +0c | sort -u > "$MC_FILE_LIST"
find "$DATA_DIR" -maxdepth 1 -type f -name '*.h5' -size +0c | sort -u > "$DATA_FILE_LIST"

N_MC=$(wc -l < "$MC_FILE_LIST" | tr -d ' ')
N_DATA=$(wc -l < "$DATA_FILE_LIST" | tr -d ' ')
if [ "$N_MC" -eq 0 ] || [ "$N_DATA" -eq 0 ]; then
  echo "Need non-empty MC and real-data H5 inputs; found MC=$N_MC data=$N_DATA"
  exit 1
fi

cat "$MC_FILE_LIST" "$DATA_FILE_LIST" > "$ALL_FILE_LIST"
{
  awk '{print "mc"}' "$MC_FILE_LIST"
  awk '{print "data"}' "$DATA_FILE_LIST"
} > "$DOMAIN_LIST_FILE"

N_ALL=$(wc -l < "$ALL_FILE_LIST" | tr -d ' ')
N_DOMAINS=$(wc -l < "$DOMAIN_LIST_FILE" | tr -d ' ')
if [ "$N_ALL" -ne "$N_DOMAINS" ]; then
  echo "Internal file/domain alignment error: files=$N_ALL domains=$N_DOMAINS"
  exit 1
fi

echo "Electron sampling study"
echo "  MC files: $N_MC"
echo "  data files: $N_DATA"
echo "  preprocessor: $PREPROCESSOR"
echo "  project: $PROJECT"

submit_run () {
  local TAG=$1
  local GPU=$2
  local FRACTIONS=$3
  local RUN_NAME="electrons_logstd_dim8_cb4096_q4_${TAG}"
  local UNIT="${UNIT_PREFIX}-${TAG}"
  local JOB_SCRIPT="${TMPDIR}/${UNIT}.sh"
  local LOG_FILE="${LOGDIR}/${RUN_NAME}.log"

  {
    printf '%s\n' '#!/usr/bin/env bash' 'set -euo pipefail'
    printf 'export HOME=%q\n' /root
    printf 'export USER=%q\n' root
    printf 'export LOGNAME=%q\n' root
    printf 'export HYDRA_FULL_ERROR=%q\n' 1
    printf 'export PYTORCH_CUDA_ALLOC_CONF=%q\n' expandable_segments:True
    printf 'export WANDB_DIR=%q\n' "${RESULTS}/wandb_tmp"
    printf 'export WANDB_CACHE_DIR=%q\n' "${RESULTS}/wandb_cache"
    printf 'mkdir -p %q %q\n' "${RESULTS}/wandb_tmp" "${RESULTS}/wandb_cache"
    printf 'cd %q\n' "$ROOT"
    printf 'H5_LIST=$(python -c %q %q)\n' \
      'import sys; from pathlib import Path; print("[" + ",".join(x.strip() for x in Path(sys.argv[1]).read_text().splitlines() if x.strip()) + "]")' \
      "$ALL_FILE_LIST"
    printf 'DOMAIN_LIST=$(python -c %q %q)\n' \
      'import sys; from pathlib import Path; print("[" + ",".join(x.strip() for x in Path(sys.argv[1]).read_text().splitlines() if x.strip()) + "]")' \
      "$DOMAIN_LIST_FILE"
    printf 'echo %q\n' "Starting $RUN_NAME on physical GPU $GPU"
    printf 'CUDA_VISIBLE_DEVICES=%q /root/.pixi/bin/pixi run python scripts/train.py \\\n' "$GPU"
    printf '  datamodule=atlas_event_object \\\n'
    printf '  model=vqvae \\\n'
    printf '  callbacks=event_tokenizer \\\n'
    printf '  output_dir=%q \\\n' "$RESULTS"
    printf '  project_name=%q \\\n' "$PROJECT"
    printf '  network_name=%q \\\n' "$RUN_NAME"
    printf '  logger.offline=%q \\\n' "$LOGGER_OFFLINE"
    printf '  trainer.max_epochs=%q \\\n' "$MAX_EPOCHS"
    printf '  trainer.accelerator=gpu \\\n'
    printf '  trainer.devices=1 \\\n'
    printf '  trainer.limit_val_batches=%q \\\n' "$LIMIT_VAL_BATCHES"
    printf '  datamodule.object_type=electrons \\\n'
    printf '  datamodule.batch_size=%q \\\n' "$BATCH_SIZE"
    printf '  datamodule.sampling_balance_by=valid_objects \\\n'
    if [ -n "$FRACTIONS" ]; then
      printf '  datamodule.sampling_domain_fractions=%q \\\n' "$FRACTIONS"
    fi
    printf '  model.codebook_dim=8 \\\n'
    printf '  model.codebook_size=4096 \\\n'
    printf '  model.num_quantizers=4 \\\n'
    printf '  model.dead_code_reset=false \\\n'
    printf '  model.data_codebook_init=false \\\n'
    printf '  +datamodule.transforms.preprocess._target_=heptokens.data.collation.preprocess_objects_batch \\\n'
    printf '  +datamodule.transforms.preprocess._partial_=true \\\n'
    printf '  +datamodule.transforms.preprocess.cst_fn._target_=joblib.load \\\n'
    printf '  +datamodule.transforms.preprocess.cst_fn.filename=%q \\\n' "$PREPROCESSOR"
    printf '  "datamodule.data_paths=${H5_LIST}" \\\n'
    printf '  "datamodule.data_domains=${DOMAIN_LIST}" \\\n'
    printf '  > %q 2>&1\n' "$LOG_FILE"
  } > "$JOB_SCRIPT"

  chmod +x "$JOB_SCRIPT"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "  dry run: wrote $JOB_SCRIPT"
  else
    systemd-run \
      --unit="$UNIT" \
      --description="Electron tokenizer sampling study: $TAG" \
      --collect \
      --property=WorkingDirectory="$ROOT" \
      /bin/bash "$JOB_SCRIPT"
  fi

  echo "  submitted $TAG on GPU $GPU: $UNIT"
  echo "    journalctl -u $UNIT -f"
  echo "    tail -120 $LOG_FILE"
}

# Natural mixture uses the existing shuffled, without-replacement dataloader.
if [[ ",$RUNS," == *",natural,"* ]]; then
  submit_run natural 1 ''
fi

# These use domain-aware sampling with replacement. Fractions target expected
# valid-electron contributions to the reconstruction loss.
if [[ ",$RUNS," == *",balanced50,"* ]]; then
  submit_run balanced50 2 '{mc:0.50,data:0.50}'
fi
if [[ ",$RUNS," == *",mcheavy75,"* ]]; then
  submit_run mcheavy75 3 '{mc:0.75,data:0.25}'
fi

echo "All three jobs submitted. Unit prefix: $UNIT_PREFIX"
