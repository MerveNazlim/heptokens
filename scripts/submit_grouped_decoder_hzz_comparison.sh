#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
STAMP=${STAMP:-20260826-052254}
GPU=${GPU:-0}
DECODER=${DECODER:-all}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-3}
LOGGER_OFFLINE=${LOGGER_OFFLINE:-true}
PROJECT_NAME=${PROJECT_NAME:-atlas_grouped_decoder_hzz_comparison_${STAMP}}
PREPARED_DIR=${PREPARED_DIR:-${ROOT}/results/event_tokens_grouped_cls_final_new_mcdata/hzz_ggf_vs_zz_cls_classification_shards}
AR_CKPT=${AR_CKPT:-${ROOT}/results/atlas_grouped_decoder_comparison/grouped_cls_autoregressive_3epochs_${STAMP}/checkpoints/last.ckpt}
PARALLEL_CKPT=${PARALLEL_CKPT:-${ROOT}/results/atlas_grouped_decoder_comparison/grouped_cls_parallel_3epochs_${STAMP}/checkpoints/last.ckpt}
PYARROW_PKG=${PYARROW_PKG:-/home/magaras/.cache/rattler/cache/pkgs/pyarrow-core-16.1.0-py311hf1d6e26_2_cpu}

if [[ "${DECODER}" != all && "${DECODER}" != autoregressive && "${DECODER}" != parallel ]]; then
  echo "DECODER must be all, autoregressive, or parallel" >&2
  exit 1
fi
for path in "${PREPARED_DIR}/manifest.json" "${AR_CKPT}" "${PARALLEL_CKPT}"; do
  test -f "${path}" || { echo "Missing required input: ${path}" >&2; exit 1; }
done

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
UNIT="atlas-decoder-hzz-${DECODER}-gpu${GPU}-${TIMESTAMP}"
TMP_DIR="${ROOT}/results/tmp/${PROJECT_NAME}"
LOG_DIR="${ROOT}/results/logs/${PROJECT_NAME}"
WORKER="${TMP_DIR}/${UNIT}.sh"
mkdir -p "${TMP_DIR}" "${LOG_DIR}"

cat >"${WORKER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '${ROOT}'
PYARROW_PKG='${PYARROW_PKG}'
CACHE_LIBS=\$(find /home/magaras/.cache/rattler/cache/pkgs -type d -path '*/lib' -print | paste -sd: -)

run_classifier() {
  local name=\$1 checkpoint=\$2 frozen=\$3
  local run_dir='${ROOT}/results/${PROJECT_NAME}'/\${name}
  local log='${LOG_DIR}'/\${name}.log
  if [[ -s "\${run_dir}/SUCCESS.txt" ]]; then
    echo "\$(date): skipping completed \${name}"
    return
  fi
  echo "\$(date): starting \${name} on GPU ${GPU}"
  CUDA_VISIBLE_DEVICES='${GPU}' HYDRA_FULL_ERROR=1 MPLBACKEND=Agg \
  WANDB_MODE=\$([[ '${LOGGER_OFFLINE}' == true ]] && echo offline || echo online) \
  WANDB_INIT_TIMEOUT=300 WANDB_SILENT=true \
  PYTHONPATH="\${PYARROW_PKG}/lib/python3.11/site-packages\${PYTHONPATH:+:\${PYTHONPATH}}" \
  LD_LIBRARY_PATH="\${CACHE_LIBS}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}" \
  /root/.pixi/bin/pixi run python scripts/train.py \
    datamodule=token_parquet_grouped_classification \
    model=foundation_grouped_cls_mlp_classifier callbacks=grouped_classification \
    project_name='${PROJECT_NAME}' network_name="\${name}" \
    output_dir='${ROOT}/results' logger.offline='${LOGGER_OFFLINE}' \
    datamodule.prepared_dir='${PREPARED_DIR}' \
    datamodule.batch_size='${BATCH_SIZE}' datamodule.num_workers='${NUM_WORKERS}' \
    datamodule.stream_batch_size=4096 datamodule.shuffle_buffer_size=8192 \
    datamodule.persistent_workers=false \
    model.backbone_ckpt_path="\${checkpoint}" model.freeze_backbone="\${frozen}" \
    trainer.max_epochs='${EPOCHS}' trainer.check_val_every_n_epoch=1 \
    trainer.val_check_interval=1.0 trainer.limit_val_batches=1.0 \
    +trainer.log_every_n_steps=50 trainer.devices=1 \
    2>&1 | tee "\${log}"
  echo "\$(date): finished \${name}"
}

if [[ '${DECODER}' == all || '${DECODER}' == autoregressive ]]; then
  run_classifier autoregressive_frozen_cls_mlp '${AR_CKPT}' true
  run_classifier autoregressive_finetuned_cls_mlp '${AR_CKPT}' false
fi
if [[ '${DECODER}' == all || '${DECODER}' == parallel ]]; then
  run_classifier parallel_frozen_cls_mlp '${PARALLEL_CKPT}' true
  run_classifier parallel_finetuned_cls_mlp '${PARALLEL_CKPT}' false
fi
EOF
chmod +x "${WORKER}"

systemd-run \
  --unit="${UNIT}" \
  --collect \
  --property=WorkingDirectory="${ROOT}" \
  /bin/bash "${WORKER}"

echo "Submitted ${UNIT}.service"
echo "Decoder selection: ${DECODER}"
echo "Full training set, ${EPOCHS} epochs per classifier"
echo "Follow: journalctl -u ${UNIT}.service -f -o cat"
echo "Runs: ${ROOT}/results/${PROJECT_NAME}"
