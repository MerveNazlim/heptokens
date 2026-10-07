#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
RESULTS=${RESULTS:-${ROOT}/results}
RUN_BASE=${RUN_BASE:-${RESULTS}/atlas_object_final_tokenizers_new_mcdata}
PREPROCESS_BASE=${PREPROCESS_BASE:-${RESULTS}/preprocessing/atlas_object_final_tokenizers_new_mcdata}
PARQUET=${PARQUET:-${RESULTS}/event_tokens_grouped_final_new_mcdata/event_tokens_signal_grouped_full_seq256.parquet}
H5_DIR=${H5_DIR:-/home/zephyr/Data/viviana/bnl-treasure/data_new/h5}
OUTPUT_DIR=${OUTPUT_DIR:-${RESULTS}/grouped_event_token_physics_reco}
MAX_EVENTS=${MAX_EVENTS:-50000}
GPU=${GPU:-0}
DEVICE=${DEVICE:-cpu}
PIXI=${PIXI:-/root/.pixi/bin/pixi}

pick_checkpoint () {
  local run_dir=$1
  if [ -f "${run_dir}/checkpoints/best.ckpt" ]; then
    printf '%s\n' "${run_dir}/checkpoints/best.ckpt"
  elif [ -f "${run_dir}/checkpoints/last.ckpt" ]; then
    printf '%s\n' "${run_dir}/checkpoints/last.ckpt"
  else
    echo "No checkpoint found under ${run_dir}" >&2
    return 1
  fi
}

configure_cached_pyarrow () {
  if "$PIXI" run python -c 'import pyarrow' >/dev/null 2>&1; then
    return
  fi
  local cache_root=${RATTLER_CACHE_ROOT:-/home/magaras/.cache/rattler/cache/pkgs}
  local pyarrow_site cache_libs
  pyarrow_site=$(find "$cache_root" -path '*/site-packages/pyarrow' -type d | sort -V | tail -n 1)
  [ -n "$pyarrow_site" ] || { echo "Cached pyarrow was not found" >&2; exit 1; }
  pyarrow_site=${pyarrow_site%/pyarrow}
  cache_libs=$(find "$cache_root" -maxdepth 2 -type d -name lib | paste -sd: -)
  export PYTHONPATH="${pyarrow_site}${PYTHONPATH:+:${PYTHONPATH}}"
  export LD_LIBRARY_PATH="${cache_libs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
}

cd "$ROOT"
configure_cached_pyarrow

ELECTRONS_RUN=${RUN_BASE}/electrons_full_dim8_cb2048_q8_e20_new_mcdata
MUONS_RUN=${RUN_BASE}/muons_full_dim8_cb2048_q8_e20_new_mcdata
JETS_RUN=${RUN_BASE}/jets_full_dim8_cb2048_q8_e20_new_mcdata

RUN_COMMAND=("$PIXI" run python scripts/analyze_event_token_physics_reco.py)
if [ "$DEVICE" = "cuda" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi

"${RUN_COMMAND[@]}" \
  --parquet "$PARQUET" \
  --h5-dir "$H5_DIR" \
  --datamodule-config configs/datamodule/atlas_event_object.yaml \
  --tokenizer-checkpoints \
    "electrons=$(pick_checkpoint "$ELECTRONS_RUN")" \
    "muons=$(pick_checkpoint "$MUONS_RUN")" \
    "jets=$(pick_checkpoint "$JETS_RUN")" \
  --preprocess-transformers \
    "electrons=${PREPROCESS_BASE}/electrons_log_standard.joblib" \
    "muons=${PREPROCESS_BASE}/muons_log_standard.joblib" \
    "jets=${PREPROCESS_BASE}/jets_log_standard.joblib" \
  --objects electrons muons jets \
  --output-dir "$OUTPUT_DIR" \
  --max-events "$MAX_EVENTS" \
  --device "$DEVICE"
