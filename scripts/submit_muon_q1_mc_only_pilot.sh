#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
PYTHON=${PYTHON:-$ROOT/.pixi/envs/default/bin/python}
BASELINE=${BASELINE:-$ROOT/results/atlas_muon_q1_initialization_pilot_e3_v1}
DATA_DIR=${DATA_DIR:-/home/zephyr/Data/viviana/bnl-treasure/data_new/h5/realdata}
OUT=${OUT:-$ROOT/results/atlas_muon_q1_mc_only_trainfit_pilot_e3_v1}
GPU=${GPU:-3}
DRY_RUN=${DRY_RUN:-0}

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/heptokens-matplotlib}
"$PYTHON" scripts/muon_q1_mc_only_pilot.py prepare \
  --root "$ROOT" --output "$OUT" --baseline "$BASELINE" --data-dir "$DATA_DIR"

if [[ "$DRY_RUN" == 1 ]]; then
  echo "DRY RUN: config/audit prepared; no GPU job submitted"
  exit 0
fi

UNIT="atlas-muon-q1-mc-only-pilot-$(date +%Y%m%d-%H%M%S)"
systemd-run --unit="$UNIT" --collect --working-directory="$ROOT" \
  --setenv="CUDA_VISIBLE_DEVICES=$GPU" \
  --setenv="PYTHONPATH=$PYTHONPATH" \
  --setenv="MPLCONFIGDIR=$MPLCONFIGDIR" \
  --setenv="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  --setenv="WANDB_SILENT=true" \
  /bin/bash -c 'exec "$@"' muon-q1-mc-only-pilot \
  "$PYTHON" -u "$ROOT/scripts/muon_q1_mc_only_pilot.py" run --output "$OUT"
echo "Submitted $UNIT.service on GPU $GPU"
echo "Follow: journalctl -u $UNIT.service -f -o cat"
echo "Training log: $OUT/data_init.log"
