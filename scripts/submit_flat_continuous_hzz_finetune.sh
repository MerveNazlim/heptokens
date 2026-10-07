#!/usr/bin/env bash
set -euo pipefail

# Fine-tune the completed Perlmutter flat-continuous event backbone on the
# matched binary H->ZZ*->4l versus continuum-ZZ classification dataset.

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
RESULTS=${RESULTS:-${ROOT}/results}
PRETRAIN_RUN=${PRETRAIN_RUN:-${RESULTS}/perlmutter_grouped_representation_benchmark/grouped_cls_flat_continuous_16gpu_seed42}
PRETRAIN_CKPT=${PRETRAIN_CKPT:-${PRETRAIN_RUN}/checkpoints/last.ckpt}
PREPARED_DIR=${PREPARED_DIR:-${RESULTS}/grouped_representation_full_prepared/20260907-060234/hzz_classification}
GPU=${GPU:-0}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-0}
LOGGER_OFFLINE=${LOGGER_OFFLINE:-true}
DRY_RUN=${DRY_RUN:-0}
PROJECT_NAME=${PROJECT_NAME:-atlas_hzz_flat_continuous_classification}
NETWORK_NAME=${NETWORK_NAME:-perlmutter_pretrained_finetuned_cls}
PYTHON_BIN=${PYTHON_BIN:-${ROOT}/.pixi/envs/default/bin/python}
PYARROW_PKG=${PYARROW_PKG:-/home/magaras/.cache/rattler/cache/pkgs/pyarrow-core-16.1.0-py311hf1d6e26_2_cpu}

if [[ ! -s "$PRETRAIN_CKPT" ]]; then
  echo "Missing flat-continuous pretraining checkpoint under ${PRETRAIN_RUN}" >&2
  exit 1
fi
if [[ ! -s "${PREPARED_DIR}/manifest.json" ]]; then
  echo "Missing paired HZZ classification manifest: ${PREPARED_DIR}/manifest.json" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python environment: ${PYTHON_BIN}" >&2
  exit 1
fi

STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)}
UNIT=${UNIT:-atlas-hzz-flat-continuous-gpu${GPU}-${STAMP}}
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
export CUDA_VISIBLE_DEVICES='${GPU}'
export HYDRA_FULL_ERROR=1
export MPLBACKEND=Agg
export WANDB_MODE=\$([[ '${LOGGER_OFFLINE}' == true ]] && echo offline || echo online)
export WANDB_INIT_TIMEOUT=300
export WANDB_SILENT=true
export PYTHONPATH='${ROOT}/src:${PYARROW_PKG}/lib/python3.11/site-packages'\${PYTHONPATH:+:\${PYTHONPATH}}
export LD_LIBRARY_PATH="\${CACHE_LIBS}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"

REPRESENTATION=flat \
STAGE=finetune \
PRETRAIN_CKPT='${PRETRAIN_CKPT}' \
PREPARED_CLASSIFICATION_DIR='${PREPARED_DIR}' \
OUTPUT_DIR='${RESULTS}' \
PROJECT_NAME='${PROJECT_NAME}' \
NETWORK_NAME='${NETWORK_NAME}' \
EPOCHS='${EPOCHS}' \
BATCH_SIZE='${BATCH_SIZE}' \
NUM_WORKERS='${NUM_WORKERS}' \
DEVICES=1 \
ACCELERATOR=gpu \
LOGGER_OFFLINE='${LOGGER_OFFLINE}' \
PYTHON_BIN='${PYTHON_BIN}' \
bash scripts/run_grouped_representation_benchmark.sh
EOF
chmod +x "$WORKER"

echo "Flat-continuous HZZ fine-tuning"
echo "  GPU:                ${GPU}"
echo "  pretrained backbone:${PRETRAIN_CKPT}"
echo "  classification data:${PREPARED_DIR}"
echo "  epochs:             ${EPOCHS}"
echo "  batch size:         ${BATCH_SIZE}"
echo "  workers:            ${NUM_WORKERS}"
echo "  learning rate:      1.0e-4"
echo "  output:             ${RESULTS}/${PROJECT_NAME}/${NETWORK_NAME}"
echo "  worker:             ${WORKER}"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY RUN: would submit ${UNIT}.service"
  exit 0
fi

systemd-run \
  --unit="$UNIT" \
  --description="Fine-tune flat-continuous CLS backbone for HZZ classification" \
  --collect \
  --property=WorkingDirectory="$ROOT" \
  /bin/bash "$WORKER"

echo "Submitted ${UNIT}.service"
echo "Follow: journalctl -u ${UNIT}.service -f -o cat"
