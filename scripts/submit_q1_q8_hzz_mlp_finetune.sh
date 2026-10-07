#!/usr/bin/env bash
set -euo pipefail

# Fine-tune matched Q1, Q8, and flat-continuous models with the same MLP head.
# Q1 and continuous run sequentially on Q1_GPU; Q8 runs concurrently on Q8_GPU.

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
RESULTS=${RESULTS:-${ROOT}/results}
Q1_GPU=${Q1_GPU:-0}
Q8_GPU=${Q8_GPU:-3}
SINGLE_GPU=${SINGLE_GPU:-}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-0}
LOGGER_OFFLINE=${LOGGER_OFFLINE:-true}
DRY_RUN=${DRY_RUN:-0}
PROJECT_NAME=${PROJECT_NAME:-atlas_hzz_q1_q8_continuous_mlp_comparison}
PYTHON_BIN=${PYTHON_BIN:-${ROOT}/.pixi/envs/default/bin/python}
PYARROW_PKG=${PYARROW_PKG:-/home/magaras/.cache/rattler/cache/pkgs/pyarrow-core-16.1.0-py311hf1d6e26_2_cpu}

Q1_CKPT=${Q1_CKPT:-/home/magaras/heptokens_external_checkpoints/q1/checkpoints/last.ckpt}
Q8_CKPT=${Q8_CKPT:-${RESULTS}/perlmutter_foundation_grouped_cls_16gpu/checkpoints/backbone_only_epoch9.ckpt}
CONTINUOUS_CKPT=${CONTINUOUS_CKPT:-/home/magaras/heptokens_external_checkpoints/continuous/checkpoints/last.ckpt}
Q1_PREPARED_DIR=${Q1_PREPARED_DIR:-${RESULTS}/event_tokens_grouped_cls_q1_final_new_mcdata/hzz_ggf_vs_zz_cls_classification_shards}
Q8_PREPARED_DIR=${Q8_PREPARED_DIR:-${RESULTS}/grouped_representation_full_prepared/20260907-060234/hzz_classification}
CONTINUOUS_PREPARED_DIR=${CONTINUOUS_PREPARED_DIR:-${RESULTS}/grouped_representation_full_prepared/20260907-060234/hzz_classification}

for path in "$Q1_CKPT" "$Q8_CKPT" "$CONTINUOUS_CKPT"; do
  [[ -s "$path" ]] || { echo "Missing checkpoint: $path" >&2; exit 1; }
done
for path in "$Q1_PREPARED_DIR" "$Q8_PREPARED_DIR" "$CONTINUOUS_PREPARED_DIR"; do
  [[ -s "$path/manifest.json" ]] || { echo "Missing manifest: $path/manifest.json" >&2; exit 1; }
done
[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 1; }

STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)}
TMP_DIR=${TMP_DIR:-${RESULTS}/tmp/${PROJECT_NAME}}
LOG_DIR=${LOG_DIR:-${RESULTS}/logs/${PROJECT_NAME}}
mkdir -p "$TMP_DIR" "$LOG_DIR"

submit_run() {
  local representation=$1
  local gpu=$2
  local max_quantizers=$3
  local checkpoint=$4
  local prepared_dir=$5
  local network_name="${representation}_pretrained_finetuned_cls_mlp"
  local run_dir="${RESULTS}/${PROJECT_NAME}/${network_name}"
  local unit="atlas-hzz-${representation}-mlp-gpu${gpu}-${STAMP}"
  local worker="${TMP_DIR}/${unit}.sh"
  local log_file="${LOG_DIR}/${network_name}.log"

  if [[ -s "${run_dir}/SUCCESS.txt" ]]; then
    echo "Already complete: ${run_dir}"
    return 0
  fi

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

REPRESENTATION=vq \
STAGE=finetune \
MAX_QUANTIZERS='${max_quantizers}' \
PRETRAIN_CKPT='${checkpoint}' \
PREPARED_CLASSIFICATION_DIR='${prepared_dir}' \
OUTPUT_DIR='${RESULTS}' \
PROJECT_NAME='${PROJECT_NAME}' \
NETWORK_NAME='${network_name}' \
EPOCHS='${EPOCHS}' \
BATCH_SIZE='${BATCH_SIZE}' \
NUM_WORKERS='${NUM_WORKERS}' \
DEVICES=1 \
ACCELERATOR=gpu \
LOGGER_OFFLINE='${LOGGER_OFFLINE}' \
PYTHON_BIN='${PYTHON_BIN}' \
bash scripts/run_grouped_representation_benchmark.sh
EOF

  if [[ "$representation" == q1 ]]; then
    cat >> "$worker" <<EOF

echo "Q1 fine-tuning complete; starting Google continuous fine-tuning on GPU ${gpu}"
REPRESENTATION=flat \
STAGE=finetune \
PRETRAIN_CKPT='${CONTINUOUS_CKPT}' \
PREPARED_CLASSIFICATION_DIR='${CONTINUOUS_PREPARED_DIR}' \
OUTPUT_DIR='${RESULTS}' \
PROJECT_NAME='${PROJECT_NAME}' \
NETWORK_NAME='continuous_google_pretrained_finetuned_cls_mlp' \
EPOCHS='${EPOCHS}' \
BATCH_SIZE='${BATCH_SIZE}' \
NUM_WORKERS='${NUM_WORKERS}' \
DEVICES=1 \
ACCELERATOR=gpu \
LOGGER_OFFLINE='${LOGGER_OFFLINE}' \
PYTHON_BIN='${PYTHON_BIN}' \
bash scripts/run_grouped_representation_benchmark.sh
EOF
    if [[ -n "$SINGLE_GPU" ]]; then
      cat >> "$worker" <<EOF

echo "Continuous fine-tuning complete; starting Q8 fine-tuning on GPU ${gpu}"
REPRESENTATION=vq \
STAGE=finetune \
MAX_QUANTIZERS=8 \
PRETRAIN_CKPT='${Q8_CKPT}' \
PREPARED_CLASSIFICATION_DIR='${Q8_PREPARED_DIR}' \
OUTPUT_DIR='${RESULTS}' \
PROJECT_NAME='${PROJECT_NAME}' \
NETWORK_NAME='q8_pretrained_finetuned_cls_mlp' \
EPOCHS='${EPOCHS}' \
BATCH_SIZE='${BATCH_SIZE}' \
NUM_WORKERS='${NUM_WORKERS}' \
DEVICES=1 \
ACCELERATOR=gpu \
LOGGER_OFFLINE='${LOGGER_OFFLINE}' \
PYTHON_BIN='${PYTHON_BIN}' \
bash scripts/run_grouped_representation_benchmark.sh
EOF
    fi
  fi
  chmod +x "$worker"

  echo "${representation^^} fine-tuning"
  echo "  GPU:        ${gpu}"
  echo "  Q:          ${max_quantizers}"
  echo "  checkpoint: ${checkpoint}"
  echo "  data:       ${prepared_dir}"
  echo "  output:     ${run_dir}"
  if [[ "$representation" == q1 ]]; then
    echo "  then:       continuous Google fine-tuning on the same GPU"
    echo "  continuous checkpoint: ${CONTINUOUS_CKPT}"
    if [[ -n "$SINGLE_GPU" ]]; then
      echo "  then:       Q8 fine-tuning on the same GPU"
    fi
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "DRY RUN: would submit ${unit}.service"
    return 0
  fi

  systemd-run \
    --unit="$unit" \
    --description="Matched ${representation^^} MLP HZZ fine-tuning" \
    --collect \
    --property=WorkingDirectory="$ROOT" \
    /bin/bash "$worker"
  echo "Follow: journalctl -u ${unit}.service -f -o cat"
}

if [[ -n "$SINGLE_GPU" ]]; then
  submit_run q1 "$SINGLE_GPU" 1 "$Q1_CKPT" "$Q1_PREPARED_DIR"
else
  submit_run q1 "$Q1_GPU" 1 "$Q1_CKPT" "$Q1_PREPARED_DIR"
  submit_run q8 "$Q8_GPU" 8 "$Q8_CKPT" "$Q8_PREPARED_DIR"
fi
