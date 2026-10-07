#!/usr/bin/env bash
set -euo pipefail

# Tokenize the additional H->4l production-mode H5 files with the final Q1 and
# Q8 object tokenizers. The existing ggF file is deliberately not re-exported:
# the full event Parquets already contain DSID 345060.

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
RESULTS=${RESULTS:-${ROOT}/results}
H5_DIR=${H5_DIR:-/home/zephyr/Data/viviana/bnl-treasure/data_new/h5_new}
GPU_Q8=${GPU_Q8:-0}
GPU_Q1=${GPU_Q1:-3}
DRY_RUN=${DRY_RUN:-1}
OVERWRITE=${OVERWRITE:-0}

NEW_H5_FILES=(
  "${H5_DIR}/DAOD_PHYSLITE.37622300._000012.pool.root.h5" # ZH, 346645
  "${H5_DIR}/DAOD_PHYSLITE.38191629._000001.pool.root.h5" # ggZH, 345066
  "${H5_DIR}/DAOD_PHYSLITE.37622304._000001.pool.root.h5" # W+H, 346646
  "${H5_DIR}/DAOD_PHYSLITE.37622304._000002.pool.root.h5" # W+H, 346646
  "${H5_DIR}/DAOD_PHYSLITE.37622308._000001.pool.root.h5" # W-H, 346647
  "${H5_DIR}/DAOD_PHYSLITE.38191734._000001.pool.root.h5" # tHjb, 346414
  "${H5_DIR}/DAOD_PHYSLITE.38191308._000001.pool.root.h5" # tWH, 346511
)

for path in "${NEW_H5_FILES[@]}"; do
  [[ -s "$path" ]] || { echo "Missing H5 file: $path" >&2; exit 1; }
done

STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)}
TMP_DIR=${RESULTS}/tmp/atlas_hzz_production_mode_tokenization/${STAMP}
mkdir -p "$TMP_DIR"

submit_one () {
  local representation=$1
  local gpu=$2
  local unit="atlas-hzz-production-tokenize-${representation}-gpu${gpu}-${STAMP}"
  local worker="${TMP_DIR}/${unit}.sh"
  local out_dir="${RESULTS}/event_tokens_hzz_production_modes/${representation}"
  local log_dir="${RESULTS}/logs/event_tokens_hzz_production_modes/${representation}"
  local run_base="${RESULTS}/atlas_object_final_tokenizers_new_mcdata"
  local preprocess_base="${RESULTS}/preprocessing/atlas_object_final_tokenizers_new_mcdata"
  local files_text="${NEW_H5_FILES[*]}"

  local jets_run electrons_run muons_run photons_run taus_run tracks_run variant
  if [[ "$representation" == q1 ]]; then
    jets_run="${run_base}/jets_full_dim8_cb16384_q1_e20_new_mcdata"
    electrons_run="${run_base}/electrons_full_dim8_cb16384_q1_e20_new_mcdata"
    muons_run="${run_base}/muons_full_dim8_cb16384_q1_e20_new_mcdata"
    photons_run="${run_base}/photons_full_dim8_cb16384_q1_e20_new_mcdata"
    taus_run="${run_base}/taus_full_dim8_cb16384_q1_e20_new_mcdata"
    tracks_run="${run_base}/tracks_full_dim8_cb16384_q1_e20_new_mcdata"
    variant=hzz_production_modes_q1
  else
    jets_run="${run_base}/jets_full_dim8_cb2048_q8_e20_new_mcdata"
    electrons_run="${run_base}/electrons_full_dim8_cb2048_q8_e20_new_mcdata"
    muons_run="${run_base}/muons_full_dim8_cb2048_q8_e20_new_mcdata"
    photons_run="${run_base}/photons_full_dim8_cb2048_q8_e20_new_mcdata"
    taus_run="${run_base}/taus_full_dim8_cb4096_q8_e20_new_mcdata"
    tracks_run="${run_base}/tracks_full_dim8_cb4096_q8_e20_new_mcdata"
    variant=hzz_production_modes_q8
  fi

  mkdir -p "$out_dir" "$log_dir"
  cat > "$worker" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '${ROOT}'

export OUT_DIR='${out_dir}'
export LOGDIR='${log_dir}'
export TOKENIZE_SCRIPT=scripts/tokenize_objects_to_grouped_parquet.py
export OUTPUT_VARIANT='${variant}'
export WRITE_LEGACY_COLUMNS=0
export SIGNAL_H5_FILES='${files_text}'
# The shared wrapper requires all three variables, although only signal runs.
export BACKGROUND_H5_FILES='${NEW_H5_FILES[0]}'
export DATA_H5_FILES='${NEW_H5_FILES[0]}'
export SAMPLES=signal
export SIGNAL_GPU='${gpu}'
export RUN_PARALLEL=0
export OVERWRITE='${OVERWRITE}'
export NUM_EVENTS=all
export MAX_SEQ_LENGTH=256
export BATCH_SIZE=1024
export METADATA_SOURCE=atlasopenmagic
export ATLASOPENMAGIC_RELEASE=2024r-pp

export JETS_RUN='${jets_run}'
export ELECTRONS_RUN='${electrons_run}'
export MUONS_RUN='${muons_run}'
export PHOTONS_RUN='${photons_run}'
export TAUS_RUN='${taus_run}'
export TRACKS_RUN='${tracks_run}'

export JETS_PREPROCESSOR='${preprocess_base}/jets_log_standard.joblib'
export ELECTRONS_PREPROCESSOR='${preprocess_base}/electrons_log_standard.joblib'
export MUONS_PREPROCESSOR='${preprocess_base}/muons_log_standard.joblib'
export PHOTONS_PREPROCESSOR='${preprocess_base}/photons_log_standard.joblib'
export TAUS_PREPROCESSOR='${preprocess_base}/taus_log_standard.joblib'
export TRACKS_PREPROCESSOR='${preprocess_base}/tracks_log_standard_no_ndoflog.joblib'

exec bash scripts/smoke_tokenize_mc_data_event_context.sh
EOF
  chmod +x "$worker"

  echo "${representation^^}: GPU ${gpu}"
  echo "  output: ${out_dir}/event_tokens_signal_${variant}_seq256.parquet"
  echo "  worker: ${worker}"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "  DRY RUN: would submit ${unit}.service"
    return
  fi
  systemd-run \
    --unit="$unit" \
    --description="Tokenize added HZZ modes with ${representation^^}" \
    --collect \
    --property=WorkingDirectory="$ROOT" \
    /bin/bash "$worker"
  echo "  Follow: journalctl -u ${unit}.service -f -o cat"
}

submit_one q8 "$GPU_Q8"
submit_one q1 "$GPU_Q1"

