#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
MC_DIR=${MC_DIR:-/home/zephyr/Data/viviana/bnl-treasure/data/h5}
DATA_DIR=${DATA_DIR:-/home/zephyr/Data/viviana/bnl-treasure/data/h5/realdata}

OUT_ROOT=${OUT_ROOT:-${ROOT}/results/fair_mc_vs_mcdata_eval/mixed_triptychs}
LOGDIR=${LOGDIR:-${ROOT}/results/logs/fair_mc_vs_mcdata_eval/mixed_triptychs}

GPU=${GPU:-0}
N_MC_FILES=${N_MC_FILES:-20}
N_DATA_FILES=${N_DATA_FILES:-20}
MAX_VALID_OBJECTS=${MAX_VALID_OBJECTS:-1000000}
BATCH_SIZE=${BATCH_SIZE:-2048}
NUM_WORKERS=${NUM_WORKERS:-0}
DEVICE=${DEVICE:-cuda}

mkdir -p "$OUT_ROOT" "$LOGDIR"
cd "$ROOT"

mapfile -t MC_FILES < <(
  find "$MC_DIR" -maxdepth 1 -type f -name "*.h5" -size +0c | sort | head -n "$N_MC_FILES"
)

mapfile -t DATA_FILES < <(
  find "$DATA_DIR" -maxdepth 1 -type f -name "*.h5" -size +0c | sort | head -n "$N_DATA_FILES"
)

if [ "${#MC_FILES[@]}" -eq 0 ]; then
  echo "No MC H5 files found in $MC_DIR" >&2
  exit 1
fi
if [ "${#DATA_FILES[@]}" -eq 0 ]; then
  echo "No real-data H5 files found in $DATA_DIR" >&2
  exit 1
fi

MIXED_FILES=("${MC_FILES[@]}" "${DATA_FILES[@]}")
echo "MC files: ${#MC_FILES[@]}"
echo "real-data files: ${#DATA_FILES[@]}"
echo "mixed files: ${#MIXED_FILES[@]}"

# object|MC-only run|MC+data run
COMPARISONS=(
  "jets|results/atlas_event_tokenizers_1606_jets_logstd_capacity_scan/jets_logstd_dim8_cb4096_q4|results/atlas_event_tokenizers_0107_logstd_mc_realdata/jets_logstd_dim8_cb4096_q4"
  "electrons|results/atlas_event_tokenizers_1806_nonjet_logstd_capacity_scan/electrons_logstd_dim8_cb4096_q4|results/atlas_event_tokenizers_0107_logstd_mc_realdata/electrons_logstd_dim8_cb4096_q4"
  "muons|results/atlas_event_tokenizers_1806_nonjet_logstd_capacity_scan/muons_logstd_dim8_cb4096_q4|results/atlas_event_tokenizers_0107_logstd_mc_realdata/muons_logstd_dim8_cb4096_q4"
  "photons|results/atlas_event_tokenizers_1806_nonjet_logstd_capacity_scan/photons_logstd_dim8_cb4096_q4|results/atlas_event_tokenizers_0107_logstd_mc_realdata/photons_logstd_dim8_cb4096_q4"
  "taus|results/atlas_event_tokenizers_1806_nonjet_logstd_capacity_scan/taus_logstd_dim8_cb8192_q4|results/atlas_event_tokenizers_0107_logstd_mc_realdata/taus_logstd_dim8_cb8192_q4"
  "tracks|results/atlas_event_tokenizers_1906_tracks_logstd_tests/tracks_logstd_no_ndoflog_dim8_cb8192_q4|results/atlas_event_tokenizers_0107_logstd_mc_realdata/tracks_logstd_dim8_cb8192_q4"
)

run_diagnostics () {
  local object_name=$1
  local tag=$2
  local run_dir=$3
  local output_dir="${OUT_ROOT}/${object_name}_${tag}"
  local log_file="${LOGDIR}/${object_name}_${tag}.log"

  if [ ! -f "$run_dir/full_config.yaml" ]; then
    echo "Missing run config: $run_dir" >&2
    return 1
  fi
  if [ ! -f "$run_dir/checkpoints/best.ckpt" ] && [ ! -f "$run_dir/checkpoints/last.ckpt" ]; then
    echo "Missing checkpoint: $run_dir" >&2
    return 1
  fi

  mkdir -p "$output_dir"
  echo "$(date): ${object_name}/${tag} -> ${output_dir}"
  CUDA_VISIBLE_DEVICES="$GPU" /root/.pixi/bin/pixi run python scripts/analyze_vqvae_tokenizer.py \
    --run-dir "$run_dir" \
    --h5-files "${MIXED_FILES[@]}" \
    --device "$DEVICE" \
    --split val \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --max-valid-objects "$MAX_VALID_OBJECTS" \
    --output-dir "$output_dir" \
    > "$log_file" 2>&1
}

STATUS=0
for row in "${COMPARISONS[@]}"; do
  IFS="|" read -r object_name mc_only_run mcdata_run <<< "$row"
  run_diagnostics "$object_name" "mc_only" "$mc_only_run" || STATUS=1
  run_diagnostics "$object_name" "mcdata" "$mcdata_run" || STATUS=1
done

if [ "$STATUS" -ne 0 ]; then
  echo "Some diagnostics failed. Check logs in $LOGDIR" >&2
  exit "$STATUS"
fi

echo "Mixed-sample triptych diagnostics finished:"
echo "$OUT_ROOT"
