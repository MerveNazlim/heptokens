#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
RUN_BASE=${RUN_BASE:-${ROOT}/results/atlas_object_final_tokenizers_new_mcdata}
OUTPUT_BASE=${OUTPUT_BASE:-${ROOT}/results/final_object_full_triptychs}
LOGDIR=${LOGDIR:-${ROOT}/results/logs/final_object_full_triptychs}
TMP_DIR=${TMP_DIR:-${ROOT}/results/tmp/final_object_full_triptychs}

MC_DIR=${MC_DIR:-/home/zephyr/Data/viviana/bnl-treasure/data_new/h5}
DATA_DIR=${DATA_DIR:-/home/zephyr/Data/viviana/bnl-treasure/data_new/h5/realdata}
EXCLUDE_MC_PATTERN=${EXCLUDE_MC_PATTERN:-DAOD_PHYSLITE.370016*}

OBJECTS=${OBJECTS:-jets,electrons,muons,photons,taus}
SAMPLES=${SAMPLES:-mc,data}
GPUS=${GPUS:-0,3}
N_FILES=${N_FILES:-20}
CHECKPOINT_NAME=${CHECKPOINT_NAME:-last.ckpt}
EXPECTED_NUM_QUANTIZERS=${EXPECTED_NUM_QUANTIZERS:-}
EXPECTED_CODEBOOK_SIZE=${EXPECTED_CODEBOOK_SIZE:-}
EXPECTED_CODEBOOK_DIM=${EXPECTED_CODEBOOK_DIM:-}
MAX_VALID_OBJECTS=${MAX_VALID_OBJECTS:-1000000}
BATCH_SIZE=${BATCH_SIZE:-2048}
NUM_WORKERS=${NUM_WORKERS:-0}
NUM_EVENTS_PER_FILE=${NUM_EVENTS_PER_FILE:-}
DEVICE=${DEVICE:-cuda}
DRY_RUN=${DRY_RUN:-0}
PIXI=${PIXI:-/root/.pixi/bin/pixi}

split_csv() {
  local value=$1
  tr ',' '\n' <<< "$value" | sed '/^[[:space:]]*$/d'
}

is_selected() {
  local needle=$1
  local value
  while IFS= read -r value; do
    [[ "$value" == "$needle" ]] && return 0
  done < <(split_csv "$SAMPLES")
  return 1
}

run_dir_for_object() {
  local object=$1
  case "$object" in
    jets)
      echo "${JETS_RUN:-${RUN_BASE}/jets_full_dim8_cb2048_q8_e20_new_mcdata}"
      ;;
    electrons)
      echo "${ELECTRONS_RUN:-${RUN_BASE}/electrons_full_dim8_cb2048_q8_e20_new_mcdata}"
      ;;
    muons)
      echo "${MUONS_RUN:-${RUN_BASE}/muons_full_dim8_cb2048_q8_e20_new_mcdata}"
      ;;
    photons)
      echo "${PHOTONS_RUN:-${RUN_BASE}/photons_full_dim8_cb2048_q8_e20_new_mcdata}"
      ;;
    taus)
      echo "${TAUS_RUN:-${RUN_BASE}/taus_full_dim8_cb4096_q8_e20_new_mcdata}"
      ;;
    tracks)
      echo "${TRACKS_RUN:-${RUN_BASE}/tracks_full_dim8_cb4096_q8_e20_new_mcdata}"
      ;;
    *)
      echo "Unknown object: $object" >&2
      return 1
      ;;
  esac
}

read_config_scalar() {
  local config=$1
  local key=$2
  sed -nE "s/^[[:space:]]*${key}:[[:space:]]*([^#[:space:]]+).*/\1/p" "$config" \
    | tail -n 1
}

validate_run_config() {
  local object=$1
  local run_dir=$2
  local config="${run_dir}/full_config.yaml"
  local expected actual

  for entry in \
    "num_quantizers:${EXPECTED_NUM_QUANTIZERS}" \
    "codebook_size:${EXPECTED_CODEBOOK_SIZE}" \
    "codebook_dim:${EXPECTED_CODEBOOK_DIM}"; do
    expected=${entry#*:}
    [[ -z "$expected" ]] && continue
    actual=$(read_config_scalar "$config" "${entry%%:*}")
    if [[ "$actual" != "$expected" ]]; then
      echo "Configuration mismatch for ${object}: ${config}" >&2
      echo "  expected ${entry%%:*}=${expected}, found ${actual:-<missing>}" >&2
      exit 1
    fi
  done
}

make_mc_filelist() {
  local output=$1
  local all_files="${output}.all"
  find "$MC_DIR" -maxdepth 1 -type f -name "*.h5" -size +0c \
    ! -name "$EXCLUDE_MC_PATTERN" | sort > "$all_files"
  sed -n "1,${N_FILES}p" "$all_files" > "$output"
  rm -f "$all_files"
}

make_data_filelist() {
  local output=$1
  local all_files="${output}.all"
  find "$DATA_DIR" -maxdepth 1 -type f -name "*.h5" -size +0c \
    | sort > "$all_files"
  sed -n "1,${N_FILES}p" "$all_files" > "$output"
  rm -f "$all_files"
}

mkdir -p "$OUTPUT_BASE" "$LOGDIR" "$TMP_DIR"
cd "$ROOT"

MC_FILELIST="${TMP_DIR}/mc_files.txt"
DATA_FILELIST="${TMP_DIR}/data_files.txt"

if is_selected mc; then
  make_mc_filelist "$MC_FILELIST"
  if [[ ! -s "$MC_FILELIST" ]]; then
    echo "No MC files selected from $MC_DIR" >&2
    exit 1
  fi
fi

if is_selected data; then
  make_data_filelist "$DATA_FILELIST"
  if [[ ! -s "$DATA_FILELIST" ]]; then
    echo "No data files selected from $DATA_DIR" >&2
    exit 1
  fi
fi

mapfile -t OBJECT_LIST < <(split_csv "$OBJECTS")
mapfile -t GPU_LIST < <(split_csv "$GPUS")
if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
  echo "No GPUs configured. Set GPUS=0,3 for example." >&2
  exit 1
fi

for gpu in "${GPU_LIST[@]}"; do
  : > "${TMP_DIR}/queue_gpu${gpu}.tsv"
done

job_index=0
for object in "${OBJECT_LIST[@]}"; do
  run_dir=$(run_dir_for_object "$object")
  if [[ ! -f "${run_dir}/full_config.yaml" ]]; then
    echo "Missing full_config.yaml for ${object}: ${run_dir}" >&2
    exit 1
  fi
  if [[ ! -f "${run_dir}/checkpoints/${CHECKPOINT_NAME}" ]]; then
    echo "Missing checkpoint for ${object}: ${run_dir}/checkpoints/${CHECKPOINT_NAME}" >&2
    exit 1
  fi
  validate_run_config "$object" "$run_dir"

  for sample in mc data; do
    if ! is_selected "$sample"; then
      continue
    fi
    gpu=${GPU_LIST[$((job_index % ${#GPU_LIST[@]}))]}
    filelist=$MC_FILELIST
    [[ "$sample" == "data" ]] && filelist=$DATA_FILELIST
    output_dir="${OUTPUT_BASE}/${object}/${sample}"
    log_file="${LOGDIR}/${object}_${sample}.log"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$object" "$sample" "$run_dir" "$filelist" "$output_dir" "$log_file" \
      >> "${TMP_DIR}/queue_gpu${gpu}.tsv"
    job_index=$((job_index + 1))
  done
done

echo
echo "Final full-tokenizer triptych diagnostics"
echo "  objects: ${OBJECTS}"
echo "  samples: ${SAMPLES}"
echo "  GPUs: ${GPU_LIST[*]}"
echo "  files per sample: ${N_FILES}"
echo "  checkpoint: ${CHECKPOINT_NAME}"
if [[ -n "$EXPECTED_NUM_QUANTIZERS$EXPECTED_CODEBOOK_SIZE$EXPECTED_CODEBOOK_DIM" ]]; then
  echo "  required config: q=${EXPECTED_NUM_QUANTIZERS:-any}, cb=${EXPECTED_CODEBOOK_SIZE:-any}, dim=${EXPECTED_CODEBOOK_DIM:-any}"
fi
echo "  batch size: ${BATCH_SIZE}"
echo "  max valid objects: ${MAX_VALID_OBJECTS}"
echo "  output: ${OUTPUT_BASE}"
echo "  logs: ${LOGDIR}"
echo

for gpu in "${GPU_LIST[@]}"; do
  queue="${TMP_DIR}/queue_gpu${gpu}.tsv"
  echo "GPU ${gpu} queue:"
  nl -ba "$queue"
done

stamp=$(date +%Y%m%d-%H%M%S)
for gpu in "${GPU_LIST[@]}"; do
  queue="${TMP_DIR}/queue_gpu${gpu}.tsv"
  if [[ ! -s "$queue" ]]; then
    continue
  fi

  unit="atlas-final-object-full-triptychs-gpu${gpu}-${stamp}"
  worker="${TMP_DIR}/${unit}.sh"
  cat > "$worker" <<'WORKER'
#!/usr/bin/env bash
set -euo pipefail

GPU="$1"
QUEUE="$2"
CHECKPOINT_NAME="$3"
BATCH_SIZE="$4"
MAX_VALID_OBJECTS="$5"
NUM_WORKERS="$6"
NUM_EVENTS_PER_FILE="$7"
DEVICE="$8"
PIXI="$9"

while IFS=$'\t' read -r object sample run_dir filelist output_dir log_file; do
  mkdir -p "$output_dir" "$(dirname "$log_file")"
  mapfile -t H5_FILES < "$filelist"
  echo "$(date): ${object}/${sample} -> ${output_dir}" | tee "$log_file"

  cmd=(
    "$PIXI" run --frozen python scripts/analyze_vqvae_tokenizer.py
    --run-dir "$run_dir"
    --checkpoint "$run_dir/checkpoints/$CHECKPOINT_NAME"
    --h5-files "${H5_FILES[@]}"
    --device "$DEVICE"
    --split val
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --max-valid-objects "$MAX_VALID_OBJECTS"
    --output-dir "$output_dir"
  )
  if [[ -n "$NUM_EVENTS_PER_FILE" ]]; then
    cmd+=(--num-events-per-file "$NUM_EVENTS_PER_FILE")
  fi

  CUDA_VISIBLE_DEVICES="$GPU" "${cmd[@]}" >> "$log_file" 2>&1
  echo "$(date): finished ${object}/${sample}" | tee -a "$log_file"
done < "$QUEUE"
WORKER
  chmod +x "$worker"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo
    echo "DRY RUN: would submit ${unit}"
    echo "  worker: ${worker}"
    echo "  queue:  ${queue}"
    continue
  fi

  systemd-run \
    --unit="$unit" \
    --description="Final full object tokenizer triptych diagnostics, GPU ${gpu}" \
    --collect \
    --property=WorkingDirectory="$ROOT" \
    /bin/bash "$worker" \
      "$gpu" "$queue" "$CHECKPOINT_NAME" "$BATCH_SIZE" "$MAX_VALID_OBJECTS" \
      "$NUM_WORKERS" "$NUM_EVENTS_PER_FILE" "$DEVICE" "$PIXI"

  echo
  echo "Submitted ${unit}"
  echo "  GPU: ${gpu}"
  echo "  progress: journalctl -u ${unit} -f"
  echo "  worker: ${worker}"
  echo "  queue: ${queue}"
done
