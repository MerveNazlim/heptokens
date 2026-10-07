#!/usr/bin/env bash
set -euo pipefail

# Foreground smoke test for one-position-per-object tokenization and pretraining.
# No systemd, Condor, Slurm, or background jobs are used.

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
RESULTS=${RESULTS:-${ROOT}/results}
MC_DIR=${MC_DIR:-/home/zephyr/Data/viviana/bnl-treasure/data/h5}
PIXI=${PIXI:-/root/.pixi/bin/pixi}
GPU=${GPU:-0}
NUM_EVENTS=${NUM_EVENTS:-300}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-128}
BATCH_SIZE=${BATCH_SIZE:-64}
TRAIN_BATCHES=${TRAIN_BATCHES:-5}
VAL_BATCHES=${VAL_BATCHES:-2}
CONFIG=${CONFIG:-configs/datamodule/atlas_event_object.yaml}
STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)}
SMOKE_DIR=${SMOKE_DIR:-${RESULTS}/smoke_grouped_event_pretrain/${STAMP}}
PYARROW_PKG=${PYARROW_PKG:-/home/magaras/.cache/rattler/cache/pkgs/pyarrow-core-16.1.0-py311hf1d6e26_2_cpu}
CACHE_ROOT=${CACHE_ROOT:-/home/magaras/.cache/rattler/cache/pkgs}

cd "$ROOT"
mkdir -p "$SMOKE_DIR"

if ! "$PIXI" run python -c 'import pyarrow' >/dev/null 2>&1; then
  if [ ! -d "$PYARROW_PKG/lib/python3.11/site-packages/pyarrow" ]; then
    echo "Cached PyArrow package not found: ${PYARROW_PKG}" >&2
    exit 1
  fi
  CACHE_LIBS=$(find "$CACHE_ROOT" -type d -path '*/lib' -print | paste -sd: -)
  export PYTHONPATH="${PYARROW_PKG}/lib/python3.11/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
  export LD_LIBRARY_PATH="${CACHE_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

"$PIXI" run python -c \
  'import pyarrow; print("pyarrow", pyarrow.__version__, pyarrow.__file__)'

pick_ckpt () {
  local run_dir=$1
  if [ -f "${run_dir}/checkpoints/best.ckpt" ]; then
    printf '%s\n' "${run_dir}/checkpoints/best.ckpt"
  elif [ -f "${run_dir}/checkpoints/last.ckpt" ]; then
    printf '%s\n' "${run_dir}/checkpoints/last.ckpt"
  else
    echo "Missing checkpoint in ${run_dir}" >&2
    exit 1
  fi
}

require_file () {
  if [ ! -f "$1" ]; then
    echo "Missing file: $1" >&2
    exit 1
  fi
}

if [ -n "${H5_FILES:-}" ]; then
  read -r -a INPUT_FILES <<< "$H5_FILES"
else
  mapfile -t INPUT_FILES < <(
    find "$MC_DIR" -maxdepth 1 -type f -name '*.h5' -size +0c | sort | head -n 1
  )
fi
if [ "${#INPUT_FILES[@]}" -eq 0 ]; then
  echo "No H5 input files found in ${MC_DIR}" >&2
  exit 1
fi

JETS_RUN=${JETS_RUN:-${RESULTS}/atlas_event_tokenizers_1606_jets_logstd_capacity_scan/jets_logstd_dim8_cb4096_q4}
ELECTRONS_RUN=${ELECTRONS_RUN:-${RESULTS}/atlas_event_tokenizers_1806_nonjet_logstd_capacity_scan/electrons_logstd_dim8_cb4096_q4}
MUONS_RUN=${MUONS_RUN:-${RESULTS}/atlas_event_tokenizers_1806_nonjet_logstd_capacity_scan/muons_logstd_dim8_cb4096_q4}
PHOTONS_RUN=${PHOTONS_RUN:-${RESULTS}/atlas_event_tokenizers_1806_nonjet_logstd_capacity_scan/photons_logstd_dim8_cb4096_q4}
TAUS_RUN=${TAUS_RUN:-${RESULTS}/atlas_event_tokenizers_1806_nonjet_logstd_capacity_scan/taus_logstd_dim8_cb8192_q4}
TRACKS_RUN=${TRACKS_RUN:-${RESULTS}/atlas_event_tokenizers_1906_tracks_logstd_tests/tracks_logstd_no_ndoflog_dim8_cb8192_q4}

JETS_PREPROCESSOR=${JETS_PREPROCESSOR:-${RESULTS}/preprocessing/jets_log_standard.joblib}
ELECTRONS_PREPROCESSOR=${ELECTRONS_PREPROCESSOR:-${RESULTS}/preprocessing/electrons_log_standard.joblib}
MUONS_PREPROCESSOR=${MUONS_PREPROCESSOR:-${RESULTS}/preprocessing/muons_log_standard.joblib}
PHOTONS_PREPROCESSOR=${PHOTONS_PREPROCESSOR:-${RESULTS}/preprocessing/photons_log_standard.joblib}
TAUS_PREPROCESSOR=${TAUS_PREPROCESSOR:-${RESULTS}/preprocessing/taus_log_standard.joblib}
TRACKS_PREPROCESSOR=${TRACKS_PREPROCESSOR:-${RESULTS}/preprocessing/tracks_log_standard_no_ndof.joblib}

for path in \
  "$JETS_PREPROCESSOR" \
  "$ELECTRONS_PREPROCESSOR" \
  "$MUONS_PREPROCESSOR" \
  "$PHOTONS_PREPROCESSOR" \
  "$TAUS_PREPROCESSOR" \
  "$TRACKS_PREPROCESSOR"
do
  require_file "$path"
done

TOKENIZER_CHECKPOINTS=(
  "electrons=$(pick_ckpt "$ELECTRONS_RUN")"
  "muons=$(pick_ckpt "$MUONS_RUN")"
  "taus=$(pick_ckpt "$TAUS_RUN")"
  "photons=$(pick_ckpt "$PHOTONS_RUN")"
  "jets=$(pick_ckpt "$JETS_RUN")"
  "tracks=$(pick_ckpt "$TRACKS_RUN")"
)
PREPROCESS_TRANSFORMERS=(
  "electrons=${ELECTRONS_PREPROCESSOR}"
  "muons=${MUONS_PREPROCESSOR}"
  "taus=${TAUS_PREPROCESSOR}"
  "photons=${PHOTONS_PREPROCESSOR}"
  "jets=${JETS_PREPROCESSOR}"
  "tracks=${TRACKS_PREPROCESSOR}"
)

GROUPED_PARQUET=${SMOKE_DIR}/grouped_events.parquet
PREPARED_DIR=${SMOKE_DIR}/prepared

echo "Smoke directory: ${SMOKE_DIR}"
echo "Input H5: ${INPUT_FILES[*]}"
echo "Events: ${NUM_EVENTS}"

CUDA_VISIBLE_DEVICES="$GPU" "$PIXI" run python \
  scripts/tokenize_objects_to_grouped_parquet.py \
  --h5-files "${INPUT_FILES[@]}" \
  --output "$GROUPED_PARQUET" \
  --datamodule-config "$CONFIG" \
  --tokenizer-checkpoints "${TOKENIZER_CHECKPOINTS[@]}" \
  --preprocess-transformers "${PREPROCESS_TRANSFORMERS[@]}" \
  --object-order electrons muons taus photons jets tracks \
  --event-token-inputs common/event/mu common/met/pt common/met/phi common/met/sumet \
  --event-token-ranges common/event/mu=0,100 common/met/pt=0,500 common/met/phi=-3.141592653589793,3.141592653589793 common/met/sumet=0,5000 \
  --event-bins 128 \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --batch-size "$BATCH_SIZE" \
  --num-events "$NUM_EVENTS" \
  --device cuda \
  --metadata-source auto \
  --atlasopenmagic-release 2024r-pp

"$PIXI" run python - "$GROUPED_PARQUET" <<'PY'
import json
import sys

import pyarrow.parquet as pq

path = sys.argv[1]
table = pq.read_table(path, columns=["tokens", "mask", "type_ids"])
metadata = table.schema.metadata or {}
vocab = json.loads(metadata[b"heptokens_token_vocabulary"])
tokens = table["tokens"][0].as_py()
mask = table["mask"][0].as_py()
assert vocab["sequence_layout"] == "one_position_per_object"
assert len(tokens) == len(mask)
assert all(len(position) == vocab["max_quantizers"] for position in tokens)
print("Grouped parquet OK")
print("  rows:", table.num_rows)
print("  token shape:", (len(tokens), len(tokens[0])))
print("  valid object positions in first event:", sum(mask))
print("  vocab_size:", vocab["vocab_size"])
print("  max_quantizers:", vocab["max_quantizers"])
PY

"$PIXI" run python scripts/prepare_token_parquet_pretrain_shards.py \
  --input-parquets "$GROUPED_PARQUET" \
  --output-dir "$PREPARED_DIR" \
  --train-frac 0.9 \
  --read-batch-size 128 \
  --shard-rows 256

read -r VOCAB_SIZE MAX_QUANTIZERS < <(
  "$PIXI" run python - "${GROUPED_PARQUET%.parquet}.vocab.json" <<'PY'
import json
import sys

vocab = json.load(open(sys.argv[1]))
print(vocab["vocab_size"], vocab["max_quantizers"])
PY
)

CUDA_VISIBLE_DEVICES="$GPU" HYDRA_FULL_ERROR=1 "$PIXI" run python scripts/train.py \
  datamodule=token_parquet_pretrain \
  model=foundation_grouped_pretrain \
  callbacks=pretrain \
  project_name=smoke_grouped_event_pretrain \
  network_name="grouped_${STAMP}" \
  output_dir="$SMOKE_DIR" \
  datamodule.prepared_dir="$PREPARED_DIR" \
  datamodule.batch_size=16 \
  datamodule.num_workers=0 \
  model.max_seq_length="$MAX_SEQ_LENGTH" \
  model.vocab_size="$VOCAB_SIZE" \
  model.max_quantizers="$MAX_QUANTIZERS" \
  trainer.max_epochs=1 \
  trainer.val_check_interval="$TRAIN_BATCHES" \
  trainer.limit_val_batches="$VAL_BATCHES" \
  +trainer.limit_train_batches="$TRAIN_BATCHES" \
  trainer.devices=1 \
  logger.offline=true

echo "Grouped tokenization and pretraining smoke test passed."
echo "Outputs: ${SMOKE_DIR}"
