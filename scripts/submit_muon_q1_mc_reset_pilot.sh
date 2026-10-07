#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
PYTHON=${PYTHON:-$ROOT/.pixi/envs/default/bin/python}
BASELINE=${BASELINE:-$ROOT/results/atlas_muon_q1_mc_only_trainfit_pilot_e3_v1}
OUT=${OUT:-$ROOT/results/atlas_muon_q1_mc_only_reset_pilot_e3_v1}
GPU=${GPU:-0}
DRY_RUN=${DRY_RUN:-0}
WAIT_FIT_SECONDS=${WAIT_FIT_SECONDS:-3600}

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/heptokens-matplotlib}
"$PYTHON" scripts/muon_q1_mc_reset_pilot.py prepare \
  --root "$ROOT" --output "$OUT" --baseline "$BASELINE"
if [[ "$DRY_RUN" == 1 ]]; then
  echo "DRY RUN: reset config/audit prepared; no GPU job submitted"
  exit 0
fi

UNIT="atlas-muon-q1-mc-reset-pilot-$(date +%Y%m%d-%H%M%S)"
systemd-run --unit="$UNIT" --collect --working-directory="$ROOT" \
  --setenv="CUDA_VISIBLE_DEVICES=$GPU" \
  --setenv="PYTHONPATH=$PYTHONPATH" \
  --setenv="MPLCONFIGDIR=$MPLCONFIGDIR" \
  --setenv="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  --setenv="WANDB_SILENT=true" \
  /bin/bash -c 'exec "$@"' muon-q1-mc-reset-pilot \
  "$PYTHON" -u "$ROOT/scripts/muon_q1_mc_reset_pilot.py" run \
  --output "$OUT" --wait-fit-seconds "$WAIT_FIT_SECONDS"
echo "Submitted $UNIT.service on GPU $GPU"
echo "Follow: journalctl -u $UNIT.service -f -o cat"
echo "Training log: $OUT/data_init.log"
