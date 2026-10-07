#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
NEW_H5_DIR=${NEW_H5_DIR:-/home/zephyr/Data/viviana/bnl-treasure/data_new/h5}
WAIT_FOR_UNIT=${WAIT_FOR_UNIT-}
GPU=${GPU:-0}

PROJECT=${PROJECT:-atlas_electron_new_samples_mcdata}
RUN_NAME=${RUN_NAME:-electrons_full_dim8_cb4096_q4_e20_mcdata_new_samples_offline}
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-1024}
MAX_PREPROCESS_OBJECTS=${MAX_PREPROCESS_OBJECTS:-1000000}
REFIT_PREPROCESSOR=${REFIT_PREPROCESSOR:-false}
LOGGER_OFFLINE=${LOGGER_OFFLINE:-true}
DRY_RUN=${DRY_RUN:-0}

RESULTS=${ROOT}/results
LOG_DIR=${RESULTS}/logs/${PROJECT}
TMP_DIR=${RESULTS}/tmp/${PROJECT}
PREPROCESS_DIR=${RESULTS}/preprocessing/${PROJECT}
PREPROCESSOR=${PREPROCESS_DIR}/electrons_log_standard.joblib
PREPROCESS_METADATA=${PREPROCESS_DIR}/electrons_log_standard.json
FILELIST=${TMP_DIR}/new_samples_h5_files.txt
CONFIG=${ROOT}/configs/datamodule/atlas_event_object.yaml

STAMP=$(date +%Y%m%d-%H%M%S)
UNIT="atlas-electron-new-samples-gpu${GPU}-${STAMP}"
WORKER=${TMP_DIR}/${UNIT}.sh
RUN_LOG=${LOG_DIR}/${RUN_NAME}.log

mkdir -p "${LOG_DIR}" "${TMP_DIR}" "${PREPROCESS_DIR}"

cat >"${WORKER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail

WAIT_FOR_UNIT='${WAIT_FOR_UNIT}'
if [ -n "\${WAIT_FOR_UNIT}" ]; then
  while systemctl is-active --quiet "\${WAIT_FOR_UNIT}"; do
    echo "\$(date): waiting for \${WAIT_FOR_UNIT}"
    sleep 60
  done
  echo "\$(date): \${WAIT_FOR_UNIT} is no longer active"
fi
echo "\$(date): preparing new electron MC+data samples"

mapfile -t H5_FILES < <(
  {
    find '${NEW_H5_DIR}' \
      -maxdepth 1 -type f -name '*.h5' \
      ! -name 'DAOD_PHYSLITE.37001630*' \
      -size +0c
    find '${NEW_H5_DIR}/realdata' \
      -maxdepth 1 -type f -name '*.h5' \
      -size +0c
  } | sort -u
)

if [ "\${#H5_FILES[@]}" -eq 0 ]; then
  echo "No non-empty H5 files found under ${NEW_H5_DIR}" >&2
  exit 1
fi

printf '%s\n' "\${H5_FILES[@]}" > '${FILELIST}'
echo "Using \${#H5_FILES[@]} new H5 files"
N_MC=\$(find '${NEW_H5_DIR}' \
  -maxdepth 1 -type f -name '*.h5' \
  ! -name 'DAOD_PHYSLITE.37001630*' \
  -size +0c | wc -l)
N_DATA=\$(find '${NEW_H5_DIR}/realdata' \
  -maxdepth 1 -type f -name '*.h5' \
  -size +0c | wc -l)
echo "  MC files: \${N_MC}"
echo "  real-data files: \${N_DATA}"
echo "File list: ${FILELIST}"

cd '${ROOT}'

if [ '${REFIT_PREPROCESSOR}' = true ] || [ ! -f '${PREPROCESSOR}' ]; then
  echo "\$(date): fitting fresh full-electron log-standard preprocessor"
  rm -f '${PREPROCESSOR}' '${PREPROCESS_METADATA}'

  /root/.pixi/bin/pixi run python scripts/get_atlas_object_preprocessing.py \
    --h5-files "\${H5_FILES[@]}" \
    --datamodule-config '${CONFIG}' \
    --object-type electrons \
    --mode log_standard \
    --log-features pt,ptvarcone30 \
    --max-objects '${MAX_PREPROCESS_OBJECTS}' \
    --output-dir '${PREPROCESS_DIR}' \
    --output-name electrons_log_standard \
    2>&1 | tee '${LOG_DIR}/preprocess_electrons_log_standard.log'
else
  echo "\$(date): reusing completed preprocessor ${PREPROCESSOR}"
fi

if [ ! -f '${PREPROCESSOR}' ]; then
  echo "Fresh preprocessor was not created: ${PREPROCESSOR}" >&2
  exit 1
fi

H5_LIST=\$(IFS=,; echo "[\${H5_FILES[*]}]")

echo "\$(date): starting ${RUN_NAME} on physical GPU ${GPU}"

CUDA_VISIBLE_DEVICES='${GPU}' \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_MODE=\$([ '${LOGGER_OFFLINE}' = true ] && echo offline || echo online) \
WANDB_INIT_TIMEOUT=300 \
WANDB_SILENT=true \
/root/.pixi/bin/pixi run python scripts/train.py \
  datamodule=atlas_event_object \
  model=vqvae \
  callbacks=event_tokenizer \
  output_dir='${RESULTS}' \
  project_name='${PROJECT}' \
  network_name='${RUN_NAME}' \
  logger.offline='${LOGGER_OFFLINE}' \
  trainer.max_epochs='${EPOCHS}' \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.limit_val_batches=1000 \
  datamodule.object_type=electrons \
  datamodule.batch_size='${BATCH_SIZE}' \
  model.codebook_dim=8 \
  'model.encoder.model.hidden_dims=[128,256,512]' \
  'model.decoder.model.hidden_dims=[128,256,512]' \
  model.codebook_size=4096 \
  model.num_quantizers=4 \
  'model.feature_loss_weights=null' \
  model.dead_code_reset=false \
  model.data_codebook_init=false \
  +datamodule.transforms.preprocess._target_=heptokens.data.collation.preprocess_objects_batch \
  +datamodule.transforms.preprocess._partial_=true \
  +datamodule.transforms.preprocess.cst_fn._target_=joblib.load \
  +datamodule.transforms.preprocess.cst_fn.filename='${PREPROCESSOR}' \
  "datamodule.data_paths=\${H5_LIST}" \
  2>&1 | tee '${RUN_LOG}'

echo "\$(date): electron new-sample preprocessing and training finished"
EOF

chmod +x "${WORKER}"

echo "Electron new-sample MC+data queue"
if [ -n "${WAIT_FOR_UNIT}" ]; then
  echo "  waits for: ${WAIT_FOR_UNIT}"
else
  echo "  starts: immediately"
fi
echo "  GPU: ${GPU}"
echo "  MC samples: ${NEW_H5_DIR}/*.h5 except DAOD_PHYSLITE.37001630*"
echo "  data samples: ${NEW_H5_DIR}/realdata/*.h5"
echo "  preprocessor: ${PREPROCESSOR}"
echo "  refit preprocessor: ${REFIT_PREPROCESSOR}"
echo "  W&B offline: ${LOGGER_OFFLINE}"
echo "  run: ${PROJECT}/${RUN_NAME}"
echo "  worker: ${WORKER}"

if [ "${DRY_RUN}" -eq 1 ]; then
  echo "DRY RUN: would submit ${UNIT}.service"
  exit 0
fi

systemd-run \
  --unit="${UNIT}" \
  --description="Fresh-preprocessed full electron VQ-VAE on new MC+data samples, GPU ${GPU}" \
  --collect \
  --property=WorkingDirectory="${ROOT}" \
  /bin/bash "${WORKER}"

echo
echo "Submitted ${UNIT}.service"
echo "  progress: journalctl -u ${UNIT}.service -f"
