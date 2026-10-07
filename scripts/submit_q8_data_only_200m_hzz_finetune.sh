#!/usr/bin/env bash
set -euo pipefail

# Fine-tune the Q8 foundation model pretrained on 200M collision-data events.
# Fine-tuning uses the same labeled HZZ signal/background MC split as the
# existing Q1/Q8/continuous comparison.

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
RESULTS=${RESULTS:-${ROOT}/results}
GPU=${GPU:-3}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-0}
LOGGER_OFFLINE=${LOGGER_OFFLINE:-true}
DRY_RUN=${DRY_RUN:-0}
PROJECT_NAME=${PROJECT_NAME:-atlas_hzz_q8_200m_data_pretrained_comparison}
NETWORK_NAME=${NETWORK_NAME:-q8_200m_data_pretrained_hzz_finetuned_cls_mlp}
PRETRAIN_CKPT=${PRETRAIN_CKPT:-${RESULTS}/atlas_event_foundation_pretrain_data16_200m/q8_b448_8gpu_e10_seed42/checkpoints/last.ckpt}
PREPARED_DIR=${PREPARED_DIR:-${RESULTS}/grouped_representation_full_prepared/20260907-060234/hzz_classification}
PYTHON_BIN=${PYTHON_BIN:-${ROOT}/.pixi/envs/default/bin/python}
PYARROW_PKG=${PYARROW_PKG:-/home/magaras/.cache/rattler/cache/pkgs/pyarrow-core-16.1.0-py311hf1d6e26_2_cpu}

[[ -s "$PRETRAIN_CKPT" ]] || { echo "Missing checkpoint: $PRETRAIN_CKPT" >&2; exit 1; }
[[ -s "$PREPARED_DIR/manifest.json" ]] || { echo "Missing manifest: $PREPARED_DIR/manifest.json" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 1; }

STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)}
UNIT=${UNIT:-atlas-hzz-q8-200m-data-pretrain-gpu${GPU}-${STAMP}}
TMP_DIR=${TMP_DIR:-${RESULTS}/tmp/${PROJECT_NAME}}
LOG_DIR=${LOG_DIR:-${RESULTS}/logs/${PROJECT_NAME}}
RUN_DIR=${RESULTS}/${PROJECT_NAME}/${NETWORK_NAME}
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

REPRESENTATION=vq \
STAGE=finetune \
MAX_QUANTIZERS=8 \
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

echo "Q8 200M data-pretrained model: HZZ MC fine-tuning"
echo "  GPU:        $GPU"
echo "  checkpoint: $PRETRAIN_CKPT"
echo "  data:       $PREPARED_DIR"
echo "  epochs:     $EPOCHS"
echo "  batch size: $BATCH_SIZE"
echo "  output:     $RUN_DIR"
echo "  worker:     $WORKER"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY RUN: would submit ${UNIT}.service"
  exit 0
fi

systemd-run \
  --unit="$UNIT" \
  --description="Fine-tune 200M data-pretrained Q8 backbone on HZZ MC" \
  --collect \
  --property=WorkingDirectory="$ROOT" \
  /bin/bash "$WORKER"

echo "Submitted ${UNIT}.service"
echo "Follow: journalctl -u ${UNIT}.service -f -o cat"
