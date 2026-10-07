#!/usr/bin/env bash
set -euo pipefail

# Train the full jets q8/cb4096/dim8 tokenizer on the final new MC+data sample.
# This uses the same architecture and controls as the selected q8 production runs.

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
GPU=${GPU:-0}

export PROJECT=${PROJECT:-atlas_object_final_tokenizers_new_mcdata}
export GPUS=$GPU
export EPOCHS=${EPOCHS:-20}
export BATCH_SIZE=${BATCH_SIZE:-1024}
export MATRIX_MODE=jets_q8_cb4096
export UNIT_PREFIX=${UNIT_PREFIX:-atlas-object-jets-full-dim8-cb4096-q8-new-mcdata}

exec bash "${ROOT}/scripts/submit_object_final_tokenizers_new_mcdata_queues.sh"
