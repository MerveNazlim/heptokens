#!/usr/bin/env bash
set -euo pipefail

# Pilot-only paired export. The established production tokenization wrappers
# remain unchanged; this entry point selects one H5 file per required sample
# and injects the raw/decoded-Q8 export flags through a dedicated Python script.

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
RESULTS=${RESULTS:-${ROOT}/results}
MC_DIR=${MC_DIR:-/home/zephyr/Data/viviana/bnl-treasure/data_new/h5}
REALDATA_DIR=${REALDATA_DIR:-${MC_DIR}/realdata}
PIXI=${PIXI:-/root/.pixi/bin/pixi}
PYTHON_BIN=${PYTHON_BIN:-${ROOT}/.pixi/envs/default/bin/python}
GPU=${GPU:-0}
NUM_EVENTS=${NUM_EVENTS:-2000}
SIGNAL_DSID=${SIGNAL_DSID:-345060}
BACKGROUND_DSID=${BACKGROUND_DSID:-700600}
STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)}
OUT_DIR=${OUT_DIR:-${RESULTS}/event_tokens_grouped_representation_pilot/${STAMP}}
LOGDIR=${LOGDIR:-${RESULTS}/logs/event_tokens_grouped_representation_pilot/${STAMP}}
MC_EXCLUDE_PATTERN=${MC_EXCLUDE_PATTERN:-DAOD_PHYSLITE.370016*}

cd "$ROOT"

find_dsid_h5 () {
  local dsid=$1
  "$PYTHON_BIN" - "$MC_DIR" "$dsid" "$MC_EXCLUDE_PATTERN" <<'PY'
import fnmatch
import sys
from pathlib import Path

import h5py
import numpy as np

root = Path(sys.argv[1])
target = int(sys.argv[2])
exclude = sys.argv[3]

for path in sorted(root.glob("*.h5")):
    if not path.stat().st_size:
        continue
    if exclude and fnmatch.fnmatch(path.name, exclude):
        continue
    try:
        with h5py.File(path, "r") as handle:
            if "metadata" in handle and "dsid" in handle["metadata"].attrs:
                dsid = int(handle["metadata"].attrs["dsid"])
            elif "atlas/event/mcChannelNumber" in handle:
                values = np.asarray(handle["atlas/event/mcChannelNumber"][:])
                unique = np.unique(values[values > 0])
                dsid = int(unique[0]) if len(unique) == 1 else 0
            else:
                dsid = 0
    except Exception:
        continue
    if dsid == target:
        print(path)
        raise SystemExit(0)

raise SystemExit(f"No non-empty H5 file found for DSID {target} under {root}")
PY
}

SIGNAL_H5=$(find_dsid_h5 "$SIGNAL_DSID")
BACKGROUND_H5=$(find_dsid_h5 "$BACKGROUND_DSID")
mapfile -t DATA_FILES < <(
  find "$REALDATA_DIR" -maxdepth 1 -type f -name '*.h5' -size +0c | sort
)
if [ "${#DATA_FILES[@]}" -eq 0 ]; then
  echo "No non-empty real-data H5 file found under ${REALDATA_DIR}" >&2
  exit 1
fi
DATA_H5=${DATA_FILES[0]}

echo "Paired representation pilot inputs"
echo "  signal DSID ${SIGNAL_DSID}: ${SIGNAL_H5}"
echo "  background DSID ${BACKGROUND_DSID}: ${BACKGROUND_H5}"
echo "  data: ${DATA_H5}"
echo "  events per file: ${NUM_EVENTS}"
echo "  output: ${OUT_DIR}"

export ROOT RESULTS MC_DIR REALDATA_DIR PIXI
export SIGNAL_H5_FILES="$SIGNAL_H5"
export BACKGROUND_H5_FILES="$BACKGROUND_H5"
export DATA_H5_FILES="$DATA_H5"
export OUT_DIR LOGDIR GPU
export SIGNAL_GPU="$GPU"
export BACKGROUND_GPU="$GPU"
export DATA_GPU="$GPU"
export NUM_EVENTS
export SAMPLES=signal,background,data
export RUN_PARALLEL=0
export OVERWRITE=0
export OUTPUT_VARIANT=grouped_representation_pilot
export TOKENIZE_SCRIPT=scripts/benchmark_tokenize_objects_to_grouped_parquet.py
export WRITE_LEGACY_COLUMNS=0

exec bash "$ROOT/scripts/tokenize_grouped_final_new_mcdata.sh"
