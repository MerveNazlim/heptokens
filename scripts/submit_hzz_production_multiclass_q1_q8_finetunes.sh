#!/usr/bin/env bash
set -euo pipefail

# Fine-tune the latest 200M real-data-pretrained Q1 and Q8 CLS backbones on the
# same five-class H->4l production-mode dataset.

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
RESULTS=${RESULTS:-${ROOT}/results}
REPRESENTATION=${REPRESENTATION:-both}
PREPARED_BASE=${PREPARED_BASE:-${RESULTS}/grouped_hzz_production_multiclass_prepared}
Q1_PRETRAIN_CKPT=${Q1_PRETRAIN_CKPT:-${RESULTS}/atlas_event_foundation_pretrain_data16_200m/q1_b448_8gpu_e10_seed42/checkpoints/last.ckpt}
Q8_PRETRAIN_CKPT=${Q8_PRETRAIN_CKPT:-${RESULTS}/atlas_event_foundation_pretrain_data16_200m/q8_b448_8gpu_e10_seed42/checkpoints/last.ckpt}
GPU_Q8=${GPU_Q8:-0}
GPU_Q1=${GPU_Q1:-3}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-0}
LEARNING_RATE=${LEARNING_RATE:-1.0e-4}
LOGGER_OFFLINE=${LOGGER_OFFLINE:-true}
DRY_RUN=${DRY_RUN:-1}
PROJECT_NAME=${PROJECT_NAME:-atlas_hzz_production_multiclass_200m_data_pretrained}
Q1_NETWORK_NAME=${Q1_NETWORK_NAME:-q1_200m_data_pretrained_finetuned_cls}
Q8_NETWORK_NAME=${Q8_NETWORK_NAME:-q8_200m_data_pretrained_finetuned_cls}
PYTHON_BIN=${PYTHON_BIN:-${ROOT}/.pixi/envs/default/bin/python}
PYARROW_PKG=${PYARROW_PKG:-/home/magaras/.cache/rattler/cache/pkgs/pyarrow-core-16.1.0-py311hf1d6e26_2_cpu}

case "$REPRESENTATION" in
  both) representations=(q1 q8) ;;
  q1|q8) representations=("$REPRESENTATION") ;;
  *) echo "REPRESENTATION must be both, q1, or q8" >&2; exit 2 ;;
esac

manifest_dirs=()
for representation in "${representations[@]}"; do
  manifest_dirs+=("${PREPARED_BASE}/${representation}")
  checkpoint=$Q1_PRETRAIN_CKPT
  [[ "$representation" == q8 ]] && checkpoint=$Q8_PRETRAIN_CKPT
  for path in "${PREPARED_BASE}/${representation}/manifest.json" "$checkpoint"; do
    [[ -s "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
  done
done

manifest_values=$(
  "$PYTHON_BIN" - "${manifest_dirs[@]}" <<'PY'
import json
import sys
from pathlib import Path

manifests = [json.loads((Path(path) / "manifest.json").read_text()) for path in sys.argv[1:]]
classes = [
    [item["name"] for item in sorted(manifest["classes"], key=lambda item: int(item["label"]))]
    for manifest in manifests
]
if any(names != classes[0] for names in classes[1:]):
    raise RuntimeError(f"Q1/Q8 class definitions differ: {classes}")
if any(manifest.get("split_counts") != manifests[0].get("split_counts")
       for manifest in manifests[1:]):
    raise RuntimeError("Q1/Q8 split counts differ")
for path, manifest in zip(sys.argv[1:], manifests, strict=True):
    expected = [256, 1 if Path(path).name == "q1" else 8]
    if list(manifest.get("token_shape") or []) != list(expected):
        raise RuntimeError(
            f"Expected token_shape={list(expected)}, found {manifest.get('token_shape')}"
        )
    labels = sorted(int(item["label"]) for item in manifest["classes"])
    if labels != list(range(len(labels))) or manifest.get("n_classes") != len(labels):
        raise RuntimeError(f"Inconsistent class labels/count in {path}")
print(len(classes[0]))
print(",".join(classes[0]))
PY
)
N_CLASSES=$(printf '%s\n' "$manifest_values" | sed -n '1p')
CLASS_NAMES=$(printf '%s\n' "$manifest_values" | sed -n '2p')

STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)}
TMP_DIR=${RESULTS}/tmp/${PROJECT_NAME}/${STAMP}
LOG_DIR=${RESULTS}/logs/${PROJECT_NAME}
mkdir -p "$TMP_DIR" "$LOG_DIR"

submit_one () {
  local representation=$1
  local max_quantizers=$2
  local gpu=$3
  local prepared_dir=$4
  local checkpoint=$5
  local network_name=$Q1_NETWORK_NAME
  [[ "$representation" == q8 ]] && network_name=$Q8_NETWORK_NAME
  local unit="atlas-hzz-production-multiclass-${representation}-gpu${gpu}-${STAMP}"
  local worker="${TMP_DIR}/${unit}.sh"
  local log_file="${LOG_DIR}/${network_name}.log"

  cat > "$worker" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '${ROOT}'
exec > >(tee -a '${log_file}') 2>&1

CACHE_LIBS=\$(find /home/magaras/.cache/rattler/cache/pkgs -type d -path '*/lib' -print | paste -sd: -)
export CUDA_VISIBLE_DEVICES='${gpu}'
export HYDRA_FULL_ERROR=1
export MPLBACKEND=Agg
export WANDB_MODE=\$([[ '${LOGGER_OFFLINE}' == true ]] && echo offline || echo online)
export WANDB_INIT_TIMEOUT=300
export WANDB_SILENT=true
export PYTHONPATH='${ROOT}/src:${PYARROW_PKG}/lib/python3.11/site-packages'\${PYTHONPATH:+:\${PYTHONPATH}}
export LD_LIBRARY_PATH="\${CACHE_LIBS}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"

'${PYTHON_BIN}' scripts/train.py \
  datamodule=token_parquet_grouped_classification \
  model=foundation_grouped_cls_classifier \
  callbacks=grouped_classification \
  project_name='${PROJECT_NAME}' \
  network_name='${network_name}' \
  output_dir='${RESULTS}' \
  logger.offline='${LOGGER_OFFLINE}' \
  ckpt_path=null \
  datamodule.prepared_dir='${prepared_dir}' \
  datamodule.n_classes='${N_CLASSES}' \
  datamodule.batch_size='${BATCH_SIZE}' \
  datamodule.num_workers='${NUM_WORKERS}' \
  datamodule.stream_batch_size=4096 \
  datamodule.shuffle_buffer_size=8192 \
  datamodule.persistent_workers=false \
  model.backbone_ckpt_path='${checkpoint}' \
  model.freeze_backbone=false \
  model.max_quantizers='${max_quantizers}' \
  model.learning_rate='${LEARNING_RATE}' \
  model.optimizer.lr='${LEARNING_RATE}' \
  trainer.max_epochs='${EPOCHS}' \
  trainer.check_val_every_n_epoch=1 \
  trainer.val_check_interval=1.0 \
  trainer.limit_val_batches=1.0 \
  +trainer.log_every_n_steps=50 \
  trainer.devices=1
EOF
  chmod +x "$worker"

  echo "${representation^^}: GPU ${gpu}"
  echo "  prepared:  ${prepared_dir}"
  echo "  backbone:  ${checkpoint}"
  echo "  output:    ${RESULTS}/${PROJECT_NAME}/${network_name}"
  echo "  worker:    ${worker}"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "  DRY RUN: would submit ${unit}.service"
    return
  fi
  systemd-run \
    --unit="$unit" \
    --description="Fine-tune ${representation^^} on five HZZ production classes" \
    --collect \
    --property=WorkingDirectory="$ROOT" \
    /bin/bash "$worker"
  echo "  Follow: journalctl -u ${unit}.service -f -o cat"
}

echo "HZZ production-mode multiclass fine-tuning"
echo "  classes (${N_CLASSES}): ${CLASS_NAMES}"
echo "  epochs: ${EPOCHS}"
if [[ "$REPRESENTATION" == both || "$REPRESENTATION" == q8 ]]; then
  submit_one q8 8 "$GPU_Q8" "${PREPARED_BASE}/q8" "$Q8_PRETRAIN_CKPT"
fi
if [[ "$REPRESENTATION" == both || "$REPRESENTATION" == q1 ]]; then
  submit_one q1 1 "$GPU_Q1" "${PREPARED_BASE}/q1" "$Q1_PRETRAIN_CKPT"
fi
