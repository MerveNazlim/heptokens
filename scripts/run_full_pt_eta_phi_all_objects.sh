#!/usr/bin/env bash
set -euo pipefail

RUN_SOURCE=full \
OUTPUT_BASE="${OUTPUT_BASE:-results/google_stage1_scan_plots}" \
bash "$(dirname "${BASH_SOURCE[0]}")/run_stage1_pt_eta_phi_all_objects.sh"
