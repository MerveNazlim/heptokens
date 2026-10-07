#!/usr/bin/env bash
set -euo pipefail

# Fine-tune the pretrained grouped-CLS q8 backbone on four HZZ classes.

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
RESULTS=${RESULTS:-${ROOT}/results}
PREPARED_DIR=${PREPARED_DIR:-}
FOUNDATION_RUN=${FOUNDATION_RUN:-${RESULTS}/perlmutter_foundation_grouped_cls_16gpu}
BACKBONE_CKPT=${BACKBONE_CKPT:-}
RESUME_CKPT=${RESUME_CKPT:-}
GPU=${GPU:-0}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-0}
LEARNING_RATE=${LEARNING_RATE:-1.0e-4}
LOGGER_OFFLINE=${LOGGER_OFFLINE:-true}
DRY_RUN=${DRY_RUN:-0}
PROJECT_NAME=${PROJECT_NAME:-atlas_hzz_grouped_cls_multiclass}
NETWORK_NAME=${NETWORK_NAME:-q8_perlmutter_pretrained_finetuned_multiclass}
PIXI=${PIXI:-/root/.pixi/bin/pixi}
PYARROW_PKG=${PYARROW_PKG:-/home/magaras/.cache/rattler/cache/pkgs/pyarrow-core-16.1.0-py311hf1d6e26_2_cpu}

if [[ -z "$PREPARED_DIR" ]]; then
  mapfile -t manifests < <(
    find "$RESULTS" -type f -path '*multiclass*/manifest.json' -print | sort
  )
  if [[ ${#manifests[@]} -ne 1 ]]; then
    echo "Expected exactly one multiclass manifest, found ${#manifests[@]}:" >&2
    printf '  %s\n' "${manifests[@]}" >&2
    echo "Set PREPARED_DIR explicitly." >&2
    exit 1
  fi
  PREPARED_DIR=$(dirname "${manifests[0]}")
fi

if [[ ! -s "${PREPARED_DIR}/manifest.json" ]]; then
  echo "Missing multiclass manifest: ${PREPARED_DIR}/manifest.json" >&2
  exit 1
fi

if [[ -z "$BACKBONE_CKPT" ]]; then
  for candidate in \
    "${FOUNDATION_RUN}/checkpoints/backbone_only_epoch9.ckpt" \
    "${FOUNDATION_RUN}/checkpoints/best.ckpt" \
    "${FOUNDATION_RUN}/checkpoints/last.ckpt"; do
    if [[ -s "$candidate" ]]; then
      BACKBONE_CKPT=$candidate
      break
    fi
  done
fi
if [[ ! -s "$BACKBONE_CKPT" ]]; then
  echo "Missing pretrained q8 backbone checkpoint under ${FOUNDATION_RUN}" >&2
  exit 1
fi

TRAINER_CKPT_PATH=null
if [[ -n "$RESUME_CKPT" ]]; then
  if [[ ! -s "$RESUME_CKPT" ]]; then
    echo "Missing resume checkpoint: ${RESUME_CKPT}" >&2
    exit 1
  fi
  TRAINER_CKPT_PATH=$RESUME_CKPT
fi

manifest_values=$(
  "$PIXI" run python -c '
import json
import sys
from pathlib import Path

manifest = json.loads((Path(sys.argv[1]) / "manifest.json").read_text())
manifest_n_classes = int(manifest.get("n_classes", -1))
if manifest_n_classes != 4:
    raise RuntimeError(f"Expected four classes, found {manifest_n_classes}")
classes = sorted(manifest["classes"], key=lambda item: int(item["label"]))
if [int(item["label"]) for item in classes] != list(range(4)):
    raise RuntimeError("Class labels must be contiguous 0..3")
shape = manifest.get("token_shape") or []
if len(shape) != 2 or int(shape[1]) != 8:
    raise RuntimeError(f"Expected grouped q8 tokens, found token_shape={shape}")
print(",".join(str(item["name"]) for item in classes))
print(",".join(str(manifest["recommended_cross_entropy_weights"][item["name"]]) for item in classes))
' "$PREPARED_DIR"
)
CLASS_NAMES=$(printf '%s\n' "$manifest_values" | sed -n '1p')
CLASS_WEIGHTS=$(printf '%s\n' "$manifest_values" | sed -n '2p')
if [[ -z "$CLASS_NAMES" || -z "$CLASS_WEIGHTS" ]]; then
  echo "Could not read classes and weights from ${PREPARED_DIR}/manifest.json" >&2
  exit 1
fi

STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)}
UNIT=${UNIT:-atlas-hzz-multiclass-finetune-gpu${GPU}-${STAMP}}
TMP_DIR=${TMP_DIR:-${RESULTS}/tmp/${PROJECT_NAME}}
LOG_DIR=${LOG_DIR:-${RESULTS}/logs/${PROJECT_NAME}}
WORKER=${TMP_DIR}/${UNIT}.sh
LOG_FILE=${LOG_DIR}/${NETWORK_NAME}.log
mkdir -p "$TMP_DIR" "$LOG_DIR"

cat > "$WORKER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '${ROOT}'
exec > >(tee -a '${LOG_FILE}') 2>&1

CACHE_LIBS=\$(find /home/magaras/.cache/rattler/cache/pkgs -type d -path '*/lib' -print | paste -sd: -)
CUDA_VISIBLE_DEVICES='${GPU}' \
HYDRA_FULL_ERROR=1 MPLBACKEND=Agg \
WANDB_MODE=\$([[ '${LOGGER_OFFLINE}' == true ]] && echo offline || echo online) \
WANDB_INIT_TIMEOUT=300 WANDB_SILENT=true \
PYTHONPATH='${PYARROW_PKG}/lib/python3.11/site-packages'\${PYTHONPATH:+:\${PYTHONPATH}} \
LD_LIBRARY_PATH="\${CACHE_LIBS}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}" \
'${PIXI}' run python scripts/train.py \
  datamodule=token_parquet_grouped_classification \
  model=foundation_grouped_cls_classifier \
  callbacks=grouped_classification \
  project_name='${PROJECT_NAME}' \
  network_name='${NETWORK_NAME}' \
  output_dir='${RESULTS}' \
  logger.offline='${LOGGER_OFFLINE}' \
  ckpt_path='${TRAINER_CKPT_PATH}' \
  datamodule.prepared_dir='${PREPARED_DIR}' \
  datamodule.n_classes=4 \
  datamodule.batch_size='${BATCH_SIZE}' \
  datamodule.num_workers='${NUM_WORKERS}' \
  datamodule.stream_batch_size=4096 \
  datamodule.shuffle_buffer_size=8192 \
  datamodule.persistent_workers=false \
  model.backbone_ckpt_path='${BACKBONE_CKPT}' \
  model.freeze_backbone=false \
  model.max_quantizers=8 \
  model.learning_rate='${LEARNING_RATE}' \
  model.optimizer.lr='${LEARNING_RATE}' \
  trainer.max_epochs='${EPOCHS}' \
  trainer.check_val_every_n_epoch=1 \
  trainer.val_check_interval=1.0 \
  trainer.limit_val_batches=1.0 \
  +trainer.log_every_n_steps=50 \
  trainer.devices=1
EOF
chmod +x "$WORKER"

echo "Grouped HZZ multiclass q8 fine-tuning"
echo "  GPU:              ${GPU}"
echo "  prepared dataset: ${PREPARED_DIR}"
echo "  classes:          ${CLASS_NAMES}"
echo "  CE weights:       ${CLASS_WEIGHTS}"
echo "  backbone:         ${BACKBONE_CKPT}"
echo "  resume checkpoint:${RESUME_CKPT:- none}"
echo "  epochs:           ${EPOCHS}"
echo "  batch size:       ${BATCH_SIZE}"
echo "  learning rate:    ${LEARNING_RATE}"
echo "  output:           ${RESULTS}/${PROJECT_NAME}/${NETWORK_NAME}"
echo "  worker:           ${WORKER}"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY RUN: would submit ${UNIT}.service"
  exit 0
fi

systemd-run \
  --unit="$UNIT" \
  --description="Fine-tune grouped-CLS q8 backbone for four-class HZZ classification" \
  --collect \
  --property=WorkingDirectory="$ROOT" \
  /bin/bash "$WORKER"

echo "Submitted ${UNIT}.service"
echo "Follow: journalctl -u ${UNIT}.service -f -o cat"
