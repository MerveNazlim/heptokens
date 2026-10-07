#!/usr/bin/env bash
set -euo pipefail

# Submit only the missing final q4/dim16 taus and tracks tokenizers.

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}

export MATRIX_MODE=q4_taus_tracks
export GPUS=${GPUS:-0,3}
export PROJECT=${PROJECT:-atlas_object_final_tokenizers_new_mcdata}
export UNIT_PREFIX=${UNIT_PREFIX:-atlas-object-final-q4-taus-tracks-new-mcdata}

exec bash "${ROOT}/scripts/submit_object_final_tokenizers_new_mcdata_queues.sh"
