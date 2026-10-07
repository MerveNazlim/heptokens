#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
PYTHON=${PYTHON:-$ROOT/.pixi/envs/default/bin/python}
PILOT_EPOCHS=${PILOT_EPOCHS:-0}
case "$PILOT_EPOCHS" in
  0) DEFAULT_OUT="$ROOT/results/atlas_muon_q1_initialization_control_v1" ;;
  [1-9]|1[0-9]) DEFAULT_OUT="$ROOT/results/atlas_muon_q1_initialization_pilot_e${PILOT_EPOCHS}_v1" ;;
  *) echo "PILOT_EPOCHS must be 0 (full run) or 1-19" >&2; exit 1 ;;
esac
OUT=${OUT:-$DEFAULT_OUT}
REFERENCE_CONFIG=${REFERENCE_CONFIG:-$ROOT/results/atlas_object_final_tokenizers_new_mcdata/muons_full_dim8_cb16384_q1_e20_new_mcdata/full_config.yaml}
GPU=${GPU:-0}
DRY_RUN=${DRY_RUN:-0}

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/heptokens-matplotlib}
"$PYTHON" scripts/muon_q1_init_control.py prepare \
  --root "$ROOT" --output "$OUT" --reference "$REFERENCE_CONFIG" --pilot-epochs "$PILOT_EPOCHS"

echo "Physical GPU: $GPU; control then data_init, sequentially"
if [[ "$DRY_RUN" == 1 ]]; then
  echo "DRY RUN: configs/audit prepared; no GPU job submitted"
  exit 0
fi

UNIT="atlas-muon-q1-init-control-$(date +%Y%m%d-%H%M%S)"
# Match the existing Zephyr services: systemd starts the system shell, which
# executes the same frozen Python runtime. Pass argv separately to retain quoting.
systemd-run --unit="$UNIT" --collect --working-directory="$ROOT" \
  --setenv="CUDA_VISIBLE_DEVICES=$GPU" \
  --setenv="PYTHONPATH=$PYTHONPATH" \
  --setenv="MPLCONFIGDIR=$MPLCONFIGDIR" \
  --setenv="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  --setenv="WANDB_SILENT=true" \
  /bin/bash -c 'exec "$@"' muon-q1-init-control \
  "$PYTHON" -u "$ROOT/scripts/muon_q1_init_control.py" run --output "$OUT"
echo "Submitted $UNIT.service"
echo "Queue status: journalctl -u $UNIT.service -f -o cat"
echo "Training logs: $OUT/control.log and $OUT/data_init.log"
