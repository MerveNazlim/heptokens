#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
PYTHON=${PYTHON:-$ROOT/.pixi/envs/default/bin/python}
BASELINE=${BASELINE:-$ROOT/results/atlas_muon_q1_mc_only_trainfit_pilot_e3_v1}
OUT=${OUT:-$ROOT/results/atlas_muon_q2_cb8192_mc_reset_pilot_e3_v1}
GPU=${GPU:-3}
ARM=${ARM:-both}
DRY_RUN=${DRY_RUN:-0}

case "$ARM" in
  both|reset_off|reset_on) ;;
  *) echo "ARM must be both, reset_off or reset_on" >&2; exit 1 ;;
esac
if [[ ! "$GPU" =~ ^[0-9]+$ ]]; then
  echo "GPU must be one physical GPU index" >&2
  exit 1
fi

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/heptokens-matplotlib}
"$PYTHON" scripts/muon_q2_mc_reset_pilot.py prepare \
  --root "$ROOT" --output "$OUT" --baseline "$BASELINE"
echo "Physical GPU: $GPU; requested arms: $ARM"
if [[ "$DRY_RUN" == 1 ]]; then
  echo "DRY RUN: Q2 configs/audit prepared; no GPU job submitted"
  exit 0
fi

UNIT="atlas-muon-q2-cb8192-mc-$ARM-gpu$GPU-$(date +%Y%m%d-%H%M%S)"
systemd-run --unit="$UNIT" --collect --working-directory="$ROOT" \
  --setenv="CUDA_VISIBLE_DEVICES=$GPU" \
  --setenv="PYTHONPATH=$PYTHONPATH" \
  --setenv="MPLCONFIGDIR=$MPLCONFIGDIR" \
  --setenv="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  --setenv="WANDB_SILENT=true" \
  /bin/bash -c 'exec "$@"' muon-q2-mc-reset-pilot \
  "$PYTHON" -u "$ROOT/scripts/muon_q2_mc_reset_pilot.py" run \
  --output "$OUT" --arm "$ARM"
echo "Submitted $UNIT.service on GPU $GPU"
echo "Follow: journalctl -u $UNIT.service -f -o cat"
echo "Training logs: $OUT/reset_off.log and $OUT/reset_on.log"
