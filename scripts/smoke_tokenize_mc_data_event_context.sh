#!/usr/bin/env bash
set -euo pipefail

# Tokenize ATLAS H5 files into three event-token parquet samples:
#   signal:     MC files whose DSID is in SIGNAL_DSIDS
#   background: all other MC files
#   data:       real-data H5 files
#
# By default this runs over all available files. Signal and background run
# sequentially on GPU 0, while data runs on GPU 1. For a quick smoke test, set
# MAX_*_FILES and/or NUM_EVENTS, for example:
#   MAX_SIGNAL_FILES=1 MAX_BACKGROUND_FILES=1 MAX_DATA_FILES=1 NUM_EVENTS=2000 bash ...

export HOME=${HOME:-/root}
export USER=${USER:-root}
export LOGNAME=${LOGNAME:-root}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
MC_DIR=${MC_DIR:-/home/zephyr/Data/viviana/bnl-treasure/data/h5}
REALDATA_DIR=${REALDATA_DIR:-/home/zephyr/Data/viviana/bnl-treasure/data/h5/realdata}
RESULTS=${RESULTS:-${ROOT}/results}
CONFIG=${CONFIG:-configs/datamodule/atlas_event_object.yaml}
PIXI=${PIXI:-/root/.pixi/bin/pixi}

OUT_DIR=${OUT_DIR:-${RESULTS}/event_tokens_mc_data_event_context}
LOGDIR=${LOGDIR:-${RESULTS}/logs/event_tokens_mc_data_event_context}
GPU=${GPU:-0}
SIGNAL_GPU=${SIGNAL_GPU:-0}
BACKGROUND_GPU=${BACKGROUND_GPU:-0}
DATA_GPU=${DATA_GPU:-1}
DEVICE=${DEVICE:-cuda}
BATCH_SIZE=${BATCH_SIZE:-1024}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-256}
EVENT_BINS=${EVENT_BINS:-128}
NUM_EVENTS=${NUM_EVENTS:-all}
MAX_SIGNAL_FILES=${MAX_SIGNAL_FILES:-all}
MAX_BACKGROUND_FILES=${MAX_BACKGROUND_FILES:-all}
MAX_DATA_FILES=${MAX_DATA_FILES:-all}
SIGNAL_DSIDS=${SIGNAL_DSIDS:-345060,344235}
SAMPLES=${SAMPLES:-signal,background,data}
OVERWRITE=${OVERWRITE:-0}
DRY_RUN=${DRY_RUN:-0}
RUN_PARALLEL=${RUN_PARALLEL:-0}
METADATA_SOURCE=${METADATA_SOURCE:-atlasopenmagic}
ATLASOPENMAGIC_RELEASE=${ATLASOPENMAGIC_RELEASE:-2024r-pp}
TOKENIZE_SCRIPT=${TOKENIZE_SCRIPT:-scripts/tokenize_objects_to_parquet_with_atlasopenmagic_metadata.py}
OUTPUT_VARIANT=${OUTPUT_VARIANT:-full}
MC_EXCLUDE_PATTERN=${MC_EXCLUDE_PATTERN:-}
WRITE_LEGACY_COLUMNS=${WRITE_LEGACY_COLUMNS:-1}

mkdir -p "$OUT_DIR" "$LOGDIR"
cd "$ROOT"

configure_cached_pyarrow () {
  if "$PIXI" run python - <<'PY' >/dev/null 2>&1
import pyarrow
PY
  then
    return 0
  fi

  CACHE_ROOT=${RATTLER_CACHE_ROOT:-/home/magaras/.cache/rattler/cache/pkgs}
  if [ ! -d "$CACHE_ROOT" ]; then
    echo "pyarrow is not importable and rattler cache was not found at ${CACHE_ROOT}" >&2
    return 1
  fi

  PYARROW_SITE=$(find "$CACHE_ROOT" -path '*/site-packages/pyarrow' -type d 2>/dev/null | sort -V | tail -n 1)
  if [ -z "$PYARROW_SITE" ]; then
    echo "pyarrow is not importable and no cached pyarrow package was found in ${CACHE_ROOT}" >&2
    return 1
  fi

  PYARROW_SITE=${PYARROW_SITE%/pyarrow}
  CACHE_LIBS=$(find "$CACHE_ROOT" -maxdepth 2 -type d -name lib 2>/dev/null | paste -sd: -)
  export PYTHONPATH="${PYARROW_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
  export LD_LIBRARY_PATH="${CACHE_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

  "$PIXI" run python - <<'PY'
import pyarrow
print("Using pyarrow", pyarrow.__version__, "from", pyarrow.__file__)
PY
}

pick_ckpt () {
  RUN_DIR=$1
  if [ -f "${RUN_DIR}/checkpoints/best.ckpt" ]; then
    echo "${RUN_DIR}/checkpoints/best.ckpt"
    return 0
  fi
  if [ -f "${RUN_DIR}/checkpoints/last.ckpt" ]; then
    echo "${RUN_DIR}/checkpoints/last.ckpt"
    return 0
  fi
  echo "Missing checkpoint in ${RUN_DIR}" >&2
  return 1
}

require_file () {
  FILE=$1
  LABEL=$2
  if [ ! -f "$FILE" ]; then
    echo "Missing ${LABEL}: ${FILE}" >&2
    echo "Set the corresponding environment variable before running this script." >&2
    exit 1
  fi
}

should_run_sample () {
  SAMPLE=$1
  case ",${SAMPLES}," in
    *",${SAMPLE},"*) return 0 ;;
    *) return 1 ;;
  esac
}

configure_cached_pyarrow

if [ "$METADATA_SOURCE" != "h5" ] && { should_run_sample signal || should_run_sample background; }; then
  "$PIXI" run python - "$ATLASOPENMAGIC_RELEASE" <<'PY'
import sys

release = sys.argv[1]
try:
    import atlasopenmagic as atom
except ImportError as exc:
    raise SystemExit(
        "atlasopenmagic is required for this parquet export but is not installed. "
        "Install it with: pixi add --pypi atlasopenmagic"
    ) from exc

if release:
    atom.set_release(release)
print(f"Using atlasopenmagic metadata release {release or '(default)'}")
PY
elif [ "$METADATA_SOURCE" != "h5" ]; then
  echo "Skipping atlasopenmagic preflight because only real data is selected"
fi

SPLIT_DIR=$(mktemp -d)
trap 'rm -rf "$SPLIT_DIR"' EXIT

if [ -n "${SIGNAL_H5_FILES:-}" ] || [ -n "${BACKGROUND_H5_FILES:-}" ] || [ -n "${DATA_H5_FILES:-}" ]; then
  if [ -z "${SIGNAL_H5_FILES:-}" ] || [ -z "${BACKGROUND_H5_FILES:-}" ] || [ -z "${DATA_H5_FILES:-}" ]; then
    echo "When using explicit files, set SIGNAL_H5_FILES, BACKGROUND_H5_FILES, and DATA_H5_FILES."
    exit 1
  fi
  read -r -a SIGNAL_FILES <<< "$SIGNAL_H5_FILES"
  read -r -a BACKGROUND_FILES <<< "$BACKGROUND_H5_FILES"
  read -r -a DATA_FILES <<< "$DATA_H5_FILES"
else
  mapfile -t MC_FILES < <(
    find "$MC_DIR" -maxdepth 1 -type f -name '*.h5' -size +0c | sort |
      while IFS= read -r path; do
        if [ -n "$MC_EXCLUDE_PATTERN" ] && [[ $(basename "$path") == $MC_EXCLUDE_PATTERN ]]; then
          continue
        fi
        printf '%s\n' "$path"
      done
  )
  mapfile -t ALL_DATA_FILES < <(find "$REALDATA_DIR" -maxdepth 1 -type f -name '*.h5' -size +0c | sort)

  if [ "${#MC_FILES[@]}" -eq 0 ]; then
    echo "No MC H5 files found in ${MC_DIR}" >&2
    exit 1
  fi
  if [ "${#ALL_DATA_FILES[@]}" -eq 0 ]; then
    echo "No real-data H5 files found in ${REALDATA_DIR}" >&2
    exit 1
  fi

  "$PIXI" run python - \
    "$SIGNAL_DSIDS" "$MAX_SIGNAL_FILES" "$MAX_BACKGROUND_FILES" "$MAX_DATA_FILES" "$SPLIT_DIR" \
    --mc "${MC_FILES[@]}" --data "${ALL_DATA_FILES[@]}" <<'PY'
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

signal_dsids = {int(item) for item in sys.argv[1].split(",") if item.strip()}
max_signal = sys.argv[2]
max_background = sys.argv[3]
max_data = sys.argv[4]
split_dir = Path(sys.argv[5])
separator = sys.argv.index("--data")
mc_paths = sys.argv[7:separator]
data_paths = sys.argv[separator + 1:]

def limit_reached(items, limit):
    return limit != "all" and len(items) >= int(limit)

signal = []
background = []
bad = []
dsid_counts = Counter()

def dsid_from_h5(path):
    with h5py.File(path, "r") as handle:
        if "metadata" in handle and "dsid" in handle["metadata"].attrs:
            return int(handle["metadata"].attrs["dsid"])
        if "atlas/event/mcChannelNumber" in handle:
            mc = np.asarray(handle["atlas/event/mcChannelNumber"][:])
            unique = np.unique(mc[mc > 0])
            if len(unique) == 1:
                return int(unique[0])
    return 0

for path in mc_paths:
    try:
        dsid = dsid_from_h5(path)
    except Exception as exc:
        bad.append((path, str(exc)))
        continue
    dsid_counts[dsid] += 1
    if dsid in signal_dsids:
        if not limit_reached(signal, max_signal):
            signal.append(path)
    elif dsid > 0:
        if not limit_reached(background, max_background):
            background.append(path)
    else:
        bad.append((path, "missing DSID"))

if max_data == "all":
    data = data_paths
else:
    data = data_paths[: int(max_data)]

(split_dir / "signal.txt").write_text("\n".join(signal) + ("\n" if signal else ""))
(split_dir / "background.txt").write_text("\n".join(background) + ("\n" if background else ""))
(split_dir / "data.txt").write_text("\n".join(data) + ("\n" if data else ""))
(split_dir / "bad_mc.txt").write_text("\n".join(f"{p}\t{e}" for p, e in bad) + ("\n" if bad else ""))

print(f"signal MC files: {len(signal)}")
print(f"background MC files: {len(background)}")
print(f"real-data files: {len(data)}")
print(f"bad/skipped MC files: {len(bad)}")
print("MC DSID counts:")
for dsid, count in dsid_counts.most_common():
    print(f"  {dsid}: {count}")
if bad:
    print("First skipped MC files:")
    for path, reason in bad[:10]:
        print(f"  {path}: {reason}")
PY

  mapfile -t SIGNAL_FILES < "${SPLIT_DIR}/signal.txt"
  mapfile -t BACKGROUND_FILES < "${SPLIT_DIR}/background.txt"
  mapfile -t DATA_FILES < "${SPLIT_DIR}/data.txt"
fi

printf '%s\n' "${SIGNAL_FILES[@]}" > "${OUT_DIR}/selected_signal_files.txt"
printf '%s\n' "${BACKGROUND_FILES[@]}" > "${OUT_DIR}/selected_background_files.txt"
printf '%s\n' "${DATA_FILES[@]}" > "${OUT_DIR}/selected_data_files.txt"

if [ "${#SIGNAL_FILES[@]}" -eq 0 ] || [ "${#BACKGROUND_FILES[@]}" -eq 0 ] || [ "${#DATA_FILES[@]}" -eq 0 ]; then
  echo "Missing inputs:"
  echo "  signal=${#SIGNAL_FILES[@]} background=${#BACKGROUND_FILES[@]} data=${#DATA_FILES[@]}"
  echo "Selected-file lists are in ${OUT_DIR}"
  exit 1
fi

echo "Selected files:"
echo "  signal:     ${#SIGNAL_FILES[@]}"
echo "  background: ${#BACKGROUND_FILES[@]}"
echo "  data:       ${#DATA_FILES[@]}"
echo "  lists:      ${OUT_DIR}/selected_*_files.txt"
echo "NUM_EVENTS per input file: ${NUM_EVENTS}"
echo "MAX_SEQ_LENGTH: ${MAX_SEQ_LENGTH}"
echo "SAMPLES: ${SAMPLES}"
echo "GPUs: signal=${SIGNAL_GPU} background=${BACKGROUND_GPU} data=${DATA_GPU}"
echo "RUN_PARALLEL: ${RUN_PARALLEL}"

JETS_RUN=${JETS_RUN:-${RESULTS}/atlas_event_tokenizers_1606_jets_logstd_capacity_scan/jets_logstd_dim8_cb4096_q4}
ELECTRONS_RUN=${ELECTRONS_RUN:-${RESULTS}/atlas_event_tokenizers_1806_nonjet_logstd_capacity_scan/electrons_logstd_dim8_cb4096_q4}
MUONS_RUN=${MUONS_RUN:-${RESULTS}/atlas_event_tokenizers_1806_nonjet_logstd_capacity_scan/muons_logstd_dim8_cb4096_q4}
PHOTONS_RUN=${PHOTONS_RUN:-${RESULTS}/atlas_event_tokenizers_1806_nonjet_logstd_capacity_scan/photons_logstd_dim8_cb4096_q4}
TAUS_RUN=${TAUS_RUN:-${RESULTS}/atlas_event_tokenizers_1806_nonjet_logstd_capacity_scan/taus_logstd_dim8_cb8192_q4}
TRACKS_RUN=${TRACKS_RUN:-${RESULTS}/atlas_event_tokenizers_1906_tracks_logstd_tests/tracks_logstd_no_ndoflog_dim8_cb8192_q4}

JETS_CKPT=$(pick_ckpt "$JETS_RUN")
ELECTRONS_CKPT=$(pick_ckpt "$ELECTRONS_RUN")
MUONS_CKPT=$(pick_ckpt "$MUONS_RUN")
PHOTONS_CKPT=$(pick_ckpt "$PHOTONS_RUN")
TAUS_CKPT=$(pick_ckpt "$TAUS_RUN")
TRACKS_CKPT=$(pick_ckpt "$TRACKS_RUN")

JETS_PREPROCESSOR=${JETS_PREPROCESSOR:-${RESULTS}/preprocessing/jets_log_standard.joblib}
ELECTRONS_PREPROCESSOR=${ELECTRONS_PREPROCESSOR:-${RESULTS}/preprocessing/electrons_log_standard.joblib}
MUONS_PREPROCESSOR=${MUONS_PREPROCESSOR:-${RESULTS}/preprocessing/muons_log_standard.joblib}
PHOTONS_PREPROCESSOR=${PHOTONS_PREPROCESSOR:-${RESULTS}/preprocessing/photons_log_standard.joblib}
TAUS_PREPROCESSOR=${TAUS_PREPROCESSOR:-${RESULTS}/preprocessing/taus_log_standard.joblib}
TRACKS_PREPROCESSOR=${TRACKS_PREPROCESSOR:-${RESULTS}/preprocessing/tracks_log_standard_no_ndoflog.joblib}

require_file "$JETS_PREPROCESSOR" "jets preprocessing transformer"
require_file "$ELECTRONS_PREPROCESSOR" "electrons preprocessing transformer"
require_file "$MUONS_PREPROCESSOR" "muons preprocessing transformer"
require_file "$PHOTONS_PREPROCESSOR" "photons preprocessing transformer"
require_file "$TAUS_PREPROCESSOR" "taus preprocessing transformer"
require_file "$TRACKS_PREPROCESSOR" "tracks preprocessing transformer"

TOKENIZER_CHECKPOINTS=(
  "electrons=${ELECTRONS_CKPT}"
  "muons=${MUONS_CKPT}"
  "taus=${TAUS_CKPT}"
  "photons=${PHOTONS_CKPT}"
  "jets=${JETS_CKPT}"
  "tracks=${TRACKS_CKPT}"
)

PREPROCESS_TRANSFORMERS=(
  "electrons=${ELECTRONS_PREPROCESSOR}"
  "muons=${MUONS_PREPROCESSOR}"
  "taus=${TAUS_PREPROCESSOR}"
  "photons=${PHOTONS_PREPROCESSOR}"
  "jets=${JETS_PREPROCESSOR}"
  "tracks=${TRACKS_PREPROCESSOR}"
)

COMMON_ARGS=(
  --datamodule-config "$CONFIG"
  --tokenizer-checkpoints "${TOKENIZER_CHECKPOINTS[@]}"
  --preprocess-transformers "${PREPROCESS_TRANSFORMERS[@]}"
  --object-order electrons muons taus photons jets tracks
  --event-token-inputs common/event/mu common/met/pt common/met/phi common/met/sumet
  --event-token-ranges common/event/mu=0,100 common/met/pt=0,500 common/met/phi=-3.141592653589793,3.141592653589793 common/met/sumet=0,5000
  --event-bins "$EVENT_BINS"
  --max-seq-length "$MAX_SEQ_LENGTH"
  --batch-size "$BATCH_SIZE"
  --device "$DEVICE"
  --metadata-source "$METADATA_SOURCE"
  --atlasopenmagic-release "$ATLASOPENMAGIC_RELEASE"
)

if [ "$WRITE_LEGACY_COLUMNS" -eq 1 ]; then
  COMMON_ARGS+=(--write-legacy-columns)
fi

if [ "$NUM_EVENTS" != "all" ] && [ -n "$NUM_EVENTS" ]; then
  COMMON_ARGS+=(--num-events "$NUM_EVENTS")
fi

run_tokenization () {
  SAMPLE=$1
  OUTPUT=$2
  SAMPLE_GPU=$3
  shift 3
  FILES=("$@")

  if [ -f "$OUTPUT" ] && [ "$OVERWRITE" -ne 1 ]; then
    echo "Skipping existing ${OUTPUT} (set OVERWRITE=1 to recreate)"
    return 0
  fi

  echo "$(date): tokenizing ${SAMPLE} -> ${OUTPUT}"
  echo "  files: ${#FILES[@]}"
  echo "  GPU: ${SAMPLE_GPU}"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY RUN only; command not executed."
    return 0
  fi

  rm -f "$OUTPUT" "${OUTPUT%.parquet}.vocab.json"
  SAMPLE_LOG="${LOGDIR}/tokenize_${SAMPLE}.log"
  if ! CUDA_VISIBLE_DEVICES="$SAMPLE_GPU" "$PIXI" run python \
    "$TOKENIZE_SCRIPT" \
    --h5-files "${FILES[@]}" \
    --output "$OUTPUT" \
    "${COMMON_ARGS[@]}" \
    > "$SAMPLE_LOG" 2>&1; then
    echo "$(date): FAILED ${SAMPLE}. Last lines from ${SAMPLE_LOG}:" >&2
    tail -n 80 "$SAMPLE_LOG" >&2 || true
    return 1
  fi
  echo "$(date): finished ${SAMPLE}"
}

SIGNAL_OUTPUT="${OUT_DIR}/event_tokens_signal_${OUTPUT_VARIANT}_seq${MAX_SEQ_LENGTH}.parquet"
BACKGROUND_OUTPUT="${OUT_DIR}/event_tokens_background_${OUTPUT_VARIANT}_seq${MAX_SEQ_LENGTH}.parquet"
DATA_OUTPUT="${OUT_DIR}/event_tokens_data_${OUTPUT_VARIANT}_seq${MAX_SEQ_LENGTH}.parquet"

PIDS=()
GPU_GROUPS=()
RUN_SIGNAL=0
RUN_BACKGROUND=0
RUN_DATA=0

if should_run_sample signal; then
  RUN_SIGNAL=1
else
  echo "Skipping signal because SAMPLES=${SAMPLES}"
fi

if should_run_sample background; then
  RUN_BACKGROUND=1
else
  echo "Skipping background because SAMPLES=${SAMPLES}"
fi

if should_run_sample data; then
  RUN_DATA=1
else
  echo "Skipping data because SAMPLES=${SAMPLES}"
fi

add_gpu_group () {
  GROUP_GPU=$1
  for EXISTING_GPU in "${GPU_GROUPS[@]}"; do
    if [ "$EXISTING_GPU" = "$GROUP_GPU" ]; then
      return 0
    fi
  done
  GPU_GROUPS+=("$GROUP_GPU")
}

run_gpu_group () {
  GROUP_GPU=$1
  echo "$(date): starting GPU ${GROUP_GPU} queue"
  if [ "$RUN_SIGNAL" -eq 1 ] && [ "$SIGNAL_GPU" = "$GROUP_GPU" ]; then
    run_tokenization signal "$SIGNAL_OUTPUT" "$SIGNAL_GPU" "${SIGNAL_FILES[@]}"
  fi
  if [ "$RUN_BACKGROUND" -eq 1 ] && [ "$BACKGROUND_GPU" = "$GROUP_GPU" ]; then
    run_tokenization background "$BACKGROUND_OUTPUT" "$BACKGROUND_GPU" "${BACKGROUND_FILES[@]}"
  fi
  if [ "$RUN_DATA" -eq 1 ] && [ "$DATA_GPU" = "$GROUP_GPU" ]; then
    run_tokenization data "$DATA_OUTPUT" "$DATA_GPU" "${DATA_FILES[@]}"
  fi
  echo "$(date): finished GPU ${GROUP_GPU} queue"
}

if [ "$RUN_PARALLEL" -eq 1 ]; then
  if [ "$RUN_SIGNAL" -eq 1 ]; then
    add_gpu_group "$SIGNAL_GPU"
  fi
  if [ "$RUN_BACKGROUND" -eq 1 ]; then
    add_gpu_group "$BACKGROUND_GPU"
  fi
  if [ "$RUN_DATA" -eq 1 ]; then
    add_gpu_group "$DATA_GPU"
  fi

  for GROUP_GPU in "${GPU_GROUPS[@]}"; do
    run_gpu_group "$GROUP_GPU" &
    PIDS+=("$!")
  done
else
  if [ "$RUN_SIGNAL" -eq 1 ]; then
    run_tokenization signal "$SIGNAL_OUTPUT" "$SIGNAL_GPU" "${SIGNAL_FILES[@]}"
  fi
  if [ "$RUN_BACKGROUND" -eq 1 ]; then
    run_tokenization background "$BACKGROUND_OUTPUT" "$BACKGROUND_GPU" "${BACKGROUND_FILES[@]}"
  fi
  if [ "$RUN_DATA" -eq 1 ]; then
    run_tokenization data "$DATA_OUTPUT" "$DATA_GPU" "${DATA_FILES[@]}"
  fi
fi

if [ "$RUN_PARALLEL" -eq 1 ] && [ "${#PIDS[@]}" -gt 0 ]; then
  STATUS=0
  for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
      STATUS=1
    fi
  done
  if [ "$STATUS" -ne 0 ]; then
    echo "At least one tokenization job failed. Check logs in ${LOGDIR}." >&2
    exit "$STATUS"
  fi
fi

if [ "$DRY_RUN" -ne 1 ]; then
  "$PIXI" run python - "$MAX_SEQ_LENGTH" "$SIGNAL_OUTPUT" "$BACKGROUND_OUTPUT" "$DATA_OUTPUT" <<'PY'
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

seq = sys.argv[1]
for path_text in sys.argv[2:]:
    path = Path(path_text)
    if not path.exists():
        continue
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    metadata = schema.metadata or {}
    vocab = json.loads(metadata[b"heptokens_token_vocabulary"])
    columns = set(schema.names)
    print(f"\n{path.name}")
    print(f"  rows: {parquet.metadata.num_rows:,}")
    print(f"  columns: {schema.names}")
    print(f"  vocab_size: {vocab['vocab_size']}")
    print(f"  event inputs: {vocab['event']['inputs']}")
    if "labels" in columns:
        raise RuntimeError(f"{path.name}: unexpected labels column")
    if parquet.metadata.num_rows:
        first_batch = next(parquet.iter_batches(batch_size=1, columns=["mask"]))
        first_mask = first_batch["mask"][0].as_py()
        print(f"  first row length: {sum(first_mask)} / {seq}")
PY
fi

echo
echo "Finished. Outputs:"
echo "  signal:     ${SIGNAL_OUTPUT}"
echo "  background: ${BACKGROUND_OUTPUT}"
echo "  data:       ${DATA_OUTPUT}"
echo "Foundation-model labels are assigned later from --signal-parquet and --background-parquet."
