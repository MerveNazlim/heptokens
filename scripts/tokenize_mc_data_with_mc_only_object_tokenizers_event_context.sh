#!/usr/bin/env bash
set -euo pipefail

export HOME=${HOME:-/root}
export USER=${USER:-root}
export LOGNAME=${LOGNAME:-root}

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
MC_DIR=${MC_DIR:-/home/zephyr/Data/viviana/bnl-treasure/data/h5}
REALDATA_DIR=${REALDATA_DIR:-/home/zephyr/Data/viviana/bnl-treasure/data/h5/realdata}
RESULTS=${RESULTS:-${ROOT}/results}
CONFIG=${CONFIG:-configs/datamodule/atlas_event_object.yaml}
PIXI=${PIXI:-/root/.pixi/bin/pixi}

OUT_DIR=${OUT_DIR:-${RESULTS}/event_tokens_mc_only_object_tokenizers_event_context}
LOGDIR=${LOGDIR:-${RESULTS}/logs/event_tokens_mc_only_object_tokenizers_event_context}
GPU=${GPU:-0}
BATCH_SIZE=${BATCH_SIZE:-1024}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-256}
EVENT_BINS=${EVENT_BINS:-128}
DEVICE=${DEVICE:-cuda}
SIGNAL_DSIDS=${SIGNAL_DSIDS:-345060,344235}
SAMPLES=${SAMPLES:-signal,background,data}

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

configure_cached_pyarrow

mapfile -t MC_FILES < <(
  find "$MC_DIR" -maxdepth 1 -type f -name '*.h5' -size +0c | sort
)

mapfile -t REALDATA_FILES < <(
  find "$REALDATA_DIR" -maxdepth 1 -type f -name '*.h5' -size +0c | sort
)

echo "MC files: ${#MC_FILES[@]}"
echo "real-data files: ${#REALDATA_FILES[@]}"
echo "signal DSIDs: ${SIGNAL_DSIDS}"
echo "samples to tokenize: ${SAMPLES}"

if [ "${#MC_FILES[@]}" -eq 0 ]; then
  echo "No MC H5 files found in ${MC_DIR}"
  exit 1
fi

if [ "${#REALDATA_FILES[@]}" -eq 0 ]; then
  echo "No real-data H5 files found in ${REALDATA_DIR}"
  exit 1
fi

SPLIT_DIR=$(mktemp -d)
trap 'rm -rf "$SPLIT_DIR"' EXIT

"$PIXI" run python - "$SIGNAL_DSIDS" "$SPLIT_DIR" "${MC_FILES[@]}" <<'PY'
import sys
from pathlib import Path

import h5py
import numpy as np

signal_dsids = {int(item) for item in sys.argv[1].split(",") if item.strip()}
split_dir = Path(sys.argv[2])
paths = sys.argv[3:]

signal = []
background = []
bad = []

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

for path in paths:
    try:
        dsid = dsid_from_h5(path)
    except Exception as exc:
        bad.append((path, str(exc)))
        continue
    if dsid in signal_dsids:
        signal.append(path)
    elif dsid > 0:
        background.append(path)
    else:
        bad.append((path, "missing DSID"))

(split_dir / "signal.txt").write_text("\n".join(signal) + ("\n" if signal else ""))
(split_dir / "background.txt").write_text("\n".join(background) + ("\n" if background else ""))
(split_dir / "bad.txt").write_text("\n".join(f"{p}\t{e}" for p, e in bad) + ("\n" if bad else ""))

print(f"signal MC files: {len(signal)}")
print(f"background MC files: {len(background)}")
print(f"bad/skipped MC files: {len(bad)}")
if bad:
    print("First skipped files:")
    for path, reason in bad[:10]:
        print(f"  {path}: {reason}")
PY

mapfile -t SIGNAL_FILES < "${SPLIT_DIR}/signal.txt"
mapfile -t BACKGROUND_FILES < "${SPLIT_DIR}/background.txt"

if [ "${#SIGNAL_FILES[@]}" -eq 0 ]; then
  echo "No signal MC files found for SIGNAL_DSIDS=${SIGNAL_DSIDS}"
  exit 1
fi

if [ "${#BACKGROUND_FILES[@]}" -eq 0 ]; then
  echo "No background MC files found"
  exit 1
fi

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

# MC-only object tokenizer runs selected from the capacity studies.
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

# These must match the transformers used when the MC-only object tokenizers were trained.
JETS_PREPROCESSOR=${JETS_PREPROCESSOR:-${RESULTS}/preprocessing/jets_log_standard.joblib}
ELECTRONS_PREPROCESSOR=${ELECTRONS_PREPROCESSOR:-${RESULTS}/preprocessing/electrons_log_standard.joblib}
MUONS_PREPROCESSOR=${MUONS_PREPROCESSOR:-${RESULTS}/preprocessing/muons_log_standard.joblib}
PHOTONS_PREPROCESSOR=${PHOTONS_PREPROCESSOR:-${RESULTS}/preprocessing/photons_log_standard.joblib}
TAUS_PREPROCESSOR=${TAUS_PREPROCESSOR:-${RESULTS}/preprocessing/taus_log_standard.joblib}
TRACKS_PREPROCESSOR=${TRACKS_PREPROCESSOR:-${RESULTS}/preprocessing/tracks_log_standard_no_ndof.joblib}

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
  --write-legacy-columns
)

run_tokenization () {
  SAMPLE=$1
  OUTPUT=$2
  shift 2
  FILES=("$@")

  if [ -f "$OUTPUT" ]; then
    echo "Skipping existing ${OUTPUT}"
    return 0
  fi

  echo "$(date): tokenizing ${SAMPLE} -> ${OUTPUT}"
  CUDA_VISIBLE_DEVICES="$GPU" "$PIXI" run python \
    scripts/tokenize_objects_to_parquet_with_atlasopenmagic_metadata.py \
    --h5-files "${FILES[@]}" \
    --output "$OUTPUT" \
    "${COMMON_ARGS[@]}" \
    > "${LOGDIR}/tokenize_${SAMPLE}.log" 2>&1
  echo "$(date): finished ${SAMPLE}"
}

should_run_sample () {
  SAMPLE=$1
  case ",${SAMPLES}," in
    *",${SAMPLE},"*) return 0 ;;
    *) return 1 ;;
  esac
}

if should_run_sample signal; then
  run_tokenization signal "${OUT_DIR}/event_tokens_signal_seq${MAX_SEQ_LENGTH}.parquet" "${SIGNAL_FILES[@]}"
else
  echo "Skipping signal because SAMPLES=${SAMPLES}"
fi

if should_run_sample background; then
  run_tokenization background "${OUT_DIR}/event_tokens_background_seq${MAX_SEQ_LENGTH}.parquet" "${BACKGROUND_FILES[@]}"
else
  echo "Skipping background because SAMPLES=${SAMPLES}"
fi

if should_run_sample data; then
  run_tokenization data "${OUT_DIR}/event_tokens_data_seq${MAX_SEQ_LENGTH}.parquet" "${REALDATA_FILES[@]}"
else
  echo "Skipping data because SAMPLES=${SAMPLES}"
fi

echo "Done. Outputs:"
echo "  ${OUT_DIR}/event_tokens_signal_seq${MAX_SEQ_LENGTH}.parquet"
echo "  ${OUT_DIR}/event_tokens_background_seq${MAX_SEQ_LENGTH}.parquet"
echo "  ${OUT_DIR}/event_tokens_data_seq${MAX_SEQ_LENGTH}.parquet"
echo "Vocabulary JSON files are written next to each parquet file."
