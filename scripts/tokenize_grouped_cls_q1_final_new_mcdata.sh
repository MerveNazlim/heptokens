#!/usr/bin/env bash
set -euo pipefail

# Export the same selected events as the final q8 dataset using the final q1 tokenizers.

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
RESULTS=${RESULTS:-${ROOT}/results}
RUN_BASE=${RUN_BASE:-${RESULTS}/atlas_object_final_tokenizers_new_mcdata}
PREPROCESS_BASE=${PREPROCESS_BASE:-${RESULTS}/preprocessing/atlas_object_final_tokenizers_new_mcdata}
SELECTION_DIR=${SELECTION_DIR:-${RESULTS}/event_tokens_grouped_cls_final_new_mcdata}

export OUT_DIR=${OUT_DIR:-${RESULTS}/event_tokens_grouped_cls_q1_final_new_mcdata}
export LOGDIR=${LOGDIR:-${RESULTS}/logs/event_tokens_grouped_cls_q1_final_new_mcdata}
export TOKENIZE_SCRIPT=${TOKENIZE_SCRIPT:-scripts/tokenize_objects_to_grouped_parquet.py}
export OUTPUT_VARIANT=${OUTPUT_VARIANT:-grouped_cls_q1_full}
export WRITE_LEGACY_COLUMNS=${WRITE_LEGACY_COLUMNS:-0}

export JETS_RUN=${JETS_RUN:-${RUN_BASE}/jets_full_dim8_cb16384_q1_e20_new_mcdata}
export ELECTRONS_RUN=${ELECTRONS_RUN:-${RUN_BASE}/electrons_full_dim8_cb16384_q1_e20_new_mcdata}
export MUONS_RUN=${MUONS_RUN:-${RUN_BASE}/muons_full_dim8_cb16384_q1_e20_new_mcdata}
export PHOTONS_RUN=${PHOTONS_RUN:-${RUN_BASE}/photons_full_dim8_cb16384_q1_e20_new_mcdata}
export TAUS_RUN=${TAUS_RUN:-${RUN_BASE}/taus_full_dim8_cb16384_q1_e20_new_mcdata}
export TRACKS_RUN=${TRACKS_RUN:-${RUN_BASE}/tracks_full_dim8_cb16384_q1_e20_new_mcdata}

export JETS_PREPROCESSOR=${JETS_PREPROCESSOR:-${PREPROCESS_BASE}/jets_log_standard.joblib}
export ELECTRONS_PREPROCESSOR=${ELECTRONS_PREPROCESSOR:-${PREPROCESS_BASE}/electrons_log_standard.joblib}
export MUONS_PREPROCESSOR=${MUONS_PREPROCESSOR:-${PREPROCESS_BASE}/muons_log_standard.joblib}
export PHOTONS_PREPROCESSOR=${PHOTONS_PREPROCESSOR:-${PREPROCESS_BASE}/photons_log_standard.joblib}
export TAUS_PREPROCESSOR=${TAUS_PREPROCESSOR:-${PREPROCESS_BASE}/taus_log_standard.joblib}
export TRACKS_PREPROCESSOR=${TRACKS_PREPROCESSOR:-${PREPROCESS_BASE}/tracks_log_standard_no_ndoflog.joblib}

SIGNAL_LIST=${SIGNAL_LIST:-${SELECTION_DIR}/selected_signal_files.txt}
BACKGROUND_LIST=${BACKGROUND_LIST:-${SELECTION_DIR}/selected_background_files.txt}
DATA_LIST=${DATA_LIST:-${SELECTION_DIR}/selected_data_files.txt}
for path in "$SIGNAL_LIST" "$BACKGROUND_LIST" "$DATA_LIST"; do
  if [ ! -s "$path" ]; then
    echo "Missing selected-file list: ${path}" >&2
    exit 1
  fi
done

export SIGNAL_H5_FILES
export BACKGROUND_H5_FILES
export DATA_H5_FILES
SIGNAL_H5_FILES=$(paste -sd' ' "$SIGNAL_LIST")
BACKGROUND_H5_FILES=$(paste -sd' ' "$BACKGROUND_LIST")
DATA_H5_FILES=$(paste -sd' ' "$DATA_LIST")

export BATCH_SIZE=${BATCH_SIZE:-1024}
export MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-256}
export NUM_EVENTS=${NUM_EVENTS:-all}
export METADATA_SOURCE=${METADATA_SOURCE:-atlasopenmagic}
export ATLASOPENMAGIC_RELEASE=${ATLASOPENMAGIC_RELEASE:-2024r-pp}
export RUN_PARALLEL=0

exec bash "${ROOT}/scripts/smoke_tokenize_mc_data_event_context.sh"
