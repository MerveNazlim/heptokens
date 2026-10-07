#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
RESULTS=${RESULTS:-${ROOT}/results}
GPU=${GPU:-3}
TASK=${TASK:-both}
DRY_RUN=${DRY_RUN:-0}
BATCH_SIZE=${BATCH_SIZE:-256}
PYTHON_BIN=${PYTHON_BIN:-${ROOT}/.pixi/envs/default/bin/python}
PYARROW_PKG=${PYARROW_PKG:-/home/magaras/.cache/rattler/cache/pkgs/pyarrow-core-16.1.0-py311hf1d6e26_2_cpu}
STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)}
OUT=${OUT:-${RESULTS}/atlas_hzz_q8_parallel_vs_autoregressive_200m/${STAMP}}
BINARY_PREPARED=${BINARY_PREPARED:-${RESULTS}/grouped_representation_full_prepared/20260907-060234/hzz_classification}
MULTICLASS_PREPARED=${MULTICLASS_PREPARED:-${RESULTS}/grouped_hzz_production_multiclass_prepared/q8}
PARALLEL_BINARY_RUN=${PARALLEL_BINARY_RUN:-${RESULTS}/atlas_hzz_q8_200m_data_pretrained_comparison/q8_200m_data_pretrained_hzz_finetuned_cls_mlp}
AR_BINARY_RUN=${AR_BINARY_RUN:-${RESULTS}/atlas_hzz_q8_autoregressive_200m/q8_autoregressive_finetuned_cls_mlp}
PARALLEL_MULTICLASS_RUN=${PARALLEL_MULTICLASS_RUN:-${RESULTS}/atlas_hzz_production_multiclass_200m_data_pretrained/q8_200m_data_pretrained_finetuned_cls}
AR_MULTICLASS_RUN=${AR_MULTICLASS_RUN:-${RESULTS}/atlas_hzz_production_multiclass_q8_autoregressive_200m/q8_autoregressive_finetuned_cls}

case "$TASK" in
  both) tasks=(binary multiclass) ;;
  binary|multiclass) tasks=("$TASK") ;;
  *) echo "TASK must be both, binary, or multiclass" >&2; exit 2 ;;
esac
[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 1; }
for script in evaluate_q8_decoder_downstream_comparison.py evaluate_q1_hierarchical_flat_hzz.py evaluate_grouped_hzz_multiclass.py; do
  [[ -s "$ROOT/scripts/$script" ]] || { echo "Missing evaluator dependency: $script" >&2; exit 1; }
done

choose_checkpoint() {
  local run=$1
  if [[ -s "$run/checkpoints/best.ckpt" ]]; then
    printf '%s\n' "$run/checkpoints/best.ckpt"
  elif [[ -s "$run/checkpoints/last.ckpt" ]]; then
    printf '%s\n' "$run/checkpoints/last.ckpt"
  else
    echo "No fine-tuned checkpoint in $run/checkpoints" >&2
    return 1
  fi
}

prepared_dirs=()
parallel_checkpoints=()
ar_checkpoints=()
for task in "${tasks[@]}"; do
  if [[ "$task" == binary ]]; then
    prepared=$BINARY_PREPARED
    parallel=${PARALLEL_BINARY_CKPT:-$(choose_checkpoint "$PARALLEL_BINARY_RUN")}
    autoregressive=${AR_BINARY_CKPT:-$(choose_checkpoint "$AR_BINARY_RUN")}
  else
    prepared=$MULTICLASS_PREPARED
    parallel=${PARALLEL_MULTICLASS_CKPT:-$(choose_checkpoint "$PARALLEL_MULTICLASS_RUN")}
    autoregressive=${AR_MULTICLASS_CKPT:-$(choose_checkpoint "$AR_MULTICLASS_RUN")}
  fi
  for path in "$prepared/manifest.json" "$parallel" "$autoregressive"; do
    [[ -s "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
  done
  prepared_dirs+=("$prepared")
  parallel_checkpoints+=("$parallel")
  ar_checkpoints+=("$autoregressive")
  echo "$task: parallel=$parallel"
  echo "$task: autoregressive=$autoregressive"
  echo "$task: test dataset=$prepared"
done

[[ ! -e "$OUT" ]] || { echo "Output already exists; choose a new OUT: $OUT" >&2; exit 1; }
mkdir -p "$OUT"
WORKER="$OUT/run_comparison.sh"
UNIT=${UNIT:-atlas-hzz-q8-parallel-vs-ar-gpu${GPU}-${STAMP}}
CACHE_LIBS=${CACHE_LIBS:-$(find /home/magaras/.cache/rattler/cache/pkgs -type d -path '*/lib' -print | paste -sd: -)}
{
  printf '#!/usr/bin/env bash\nset -euo pipefail\n'
  printf 'cd %q\n' "$ROOT"
  printf 'export CUDA_VISIBLE_DEVICES=%q MPLBACKEND=Agg\n' "$GPU"
  printf 'export PYTHONPATH=%q\n' "$ROOT/src:$ROOT/scripts:$PYARROW_PKG/lib/python3.11/site-packages${PYTHONPATH:+:$PYTHONPATH}"
  printf 'export LD_LIBRARY_PATH=%q\n' "$CACHE_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  printf 'exec > >(tee -a %q) 2>&1\n' "$OUT/evaluation.log"
  for index in "${!tasks[@]}"; do
    printf '%q ' "$PYTHON_BIN" "$ROOT/scripts/evaluate_q8_decoder_downstream_comparison.py" \
      --task "${tasks[$index]}" --prepared-dir "${prepared_dirs[$index]}" \
      --parallel-checkpoint "${parallel_checkpoints[$index]}" \
      --autoregressive-checkpoint "${ar_checkpoints[$index]}" \
      --output-dir "$OUT/${tasks[$index]}" --device cuda --batch-size "$BATCH_SIZE"
    printf '\n'
  done
} > "$WORKER"
bash -n "$WORKER"
echo "Output: $OUT"
if [[ "$DRY_RUN" == 1 ]]; then
  echo "DRY RUN: would submit $UNIT.service on GPU $GPU"
  exit 0
fi
systemd-run --unit="$UNIT" --description="Compare parallel and AR Q8 downstream classifiers" \
  --collect --property=WorkingDirectory="$ROOT" /bin/bash "$WORKER"
echo "Follow: journalctl -u $UNIT.service -f -o cat"
