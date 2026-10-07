#!/usr/bin/env bash
set -euo pipefail

# Train the q1/cb16384/dim8 full-tokenizer control for every object type.
# GPU 0 and GPU 3 receive ordered, sequential queues:
#   GPU 0: jets, muons, taus
#   GPU 3: electrons, photons, tracks

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}

export PROJECT=${PROJECT:-atlas_object_final_tokenizers_new_mcdata}
export GPUS=${GPUS:-0,3}
export EPOCHS=${EPOCHS:-20}
export BATCH_SIZE=${BATCH_SIZE:-1024}
export MATRIX_MODE=q1_full
export UNIT_PREFIX=${UNIT_PREFIX:-atlas-object-final-q1-cb16384-dim8-new-mcdata}

exec bash "${ROOT}/scripts/submit_object_final_tokenizers_new_mcdata_queues.sh"
