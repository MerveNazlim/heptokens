#!/usr/bin/env bash
set -euo pipefail

# Run MC and data diagnostics for every completed q1/cb16384/dim8 tokenizer.
# The underlying submitter creates one sequential queue per selected GPU.

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
RUN_BASE=${RUN_BASE:-${ROOT}/results/atlas_object_final_tokenizers_new_mcdata}

export RUN_BASE
export OUTPUT_BASE=${OUTPUT_BASE:-${ROOT}/results/final_q1_cb16384_dim8_full_triptychs}
export LOGDIR=${LOGDIR:-${ROOT}/results/logs/final_q1_cb16384_dim8_full_triptychs}
export TMP_DIR=${TMP_DIR:-${ROOT}/results/tmp/final_q1_cb16384_dim8_full_triptychs}
export OBJECTS=${OBJECTS:-jets,electrons,muons,photons,taus,tracks}
export SAMPLES=${SAMPLES:-mc,data}
export GPUS=${GPUS:-0,3}
export CHECKPOINT_NAME=${CHECKPOINT_NAME:-best.ckpt}
export EXPECTED_NUM_QUANTIZERS=1
export EXPECTED_CODEBOOK_SIZE=16384
export EXPECTED_CODEBOOK_DIM=8

export JETS_RUN=${JETS_RUN:-${RUN_BASE}/jets_full_dim8_cb16384_q1_e20_new_mcdata}
export ELECTRONS_RUN=${ELECTRONS_RUN:-${RUN_BASE}/electrons_full_dim8_cb16384_q1_e20_new_mcdata}
export MUONS_RUN=${MUONS_RUN:-${RUN_BASE}/muons_full_dim8_cb16384_q1_e20_new_mcdata}
export PHOTONS_RUN=${PHOTONS_RUN:-${RUN_BASE}/photons_full_dim8_cb16384_q1_e20_new_mcdata}
export TAUS_RUN=${TAUS_RUN:-${RUN_BASE}/taus_full_dim8_cb16384_q1_e20_new_mcdata}
export TRACKS_RUN=${TRACKS_RUN:-${RUN_BASE}/tracks_full_dim8_cb16384_q1_e20_new_mcdata}

exec bash "${ROOT}/scripts/submit_final_object_full_triptychs.sh"
