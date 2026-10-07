#!/usr/bin/env bash
set -euo pipefail

# Run this as magaras on Zephyr, where the completed Q1/Q8 tokenizer runs and
# fitted preprocessing transformers are available.  The bundle contains only
# the checkpoints and transformers needed by the data-GRL conversion jobs.

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
RESULTS=${RESULTS:-${ROOT}/results}
RUN_BASE=${RUN_BASE:-${RESULTS}/atlas_object_final_tokenizers_new_mcdata}
PREPROCESS_BASE=${PREPROCESS_BASE:-${RESULTS}/preprocessing/atlas_object_final_tokenizers_new_mcdata}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT}/results/data_grl_tokenizer_artifacts}
ARCHIVE_NAME=${ARCHIVE_NAME:-data-grl-q1-q8-tokenizer-artifacts.tar.gz}
GCS_DESTINATION=${GCS_DESTINATION:-gs://fcc-teststorage/bucket_results/foundation_pretrain/tokenizer_artifacts/data_grl_q1_q8_v1}

objects=(jets electrons muons photons taus tracks)

declare -A q1_runs=(
  [jets]=jets_full_dim8_cb16384_q1_e20_new_mcdata
  [electrons]=electrons_full_dim8_cb16384_q1_e20_new_mcdata
  [muons]=muons_full_dim8_cb16384_q1_e20_new_mcdata
  [photons]=photons_full_dim8_cb16384_q1_e20_new_mcdata
  [taus]=taus_full_dim8_cb16384_q1_e20_new_mcdata
  [tracks]=tracks_full_dim8_cb16384_q1_e20_new_mcdata
)

declare -A q8_runs=(
  [jets]=jets_full_dim8_cb2048_q8_e20_new_mcdata
  [electrons]=electrons_full_dim8_cb2048_q8_e20_new_mcdata
  [muons]=muons_full_dim8_cb2048_q8_e20_new_mcdata
  [photons]=photons_full_dim8_cb2048_q8_e20_new_mcdata
  [taus]=taus_full_dim8_cb4096_q8_e20_new_mcdata
  [tracks]=tracks_full_dim8_cb4096_q8_e20_new_mcdata
)

declare -A preprocessors=(
  [jets]=jets_log_standard.joblib
  [electrons]=electrons_log_standard.joblib
  [muons]=muons_log_standard.joblib
  [photons]=photons_log_standard.joblib
  [taus]=taus_log_standard.joblib
  [tracks]=tracks_log_standard_no_ndoflog.joblib
)

pick_checkpoint() {
  local run_dir=$1
  if [[ -s "$run_dir/checkpoints/best.ckpt" ]]; then
    printf '%s\n' "$run_dir/checkpoints/best.ckpt"
  elif [[ -s "$run_dir/checkpoints/last.ckpt" ]]; then
    printf '%s\n' "$run_dir/checkpoints/last.ckpt"
  else
    echo "No best.ckpt or last.ckpt under $run_dir" >&2
    return 1
  fi
}

mkdir -p "$OUTPUT_DIR"
staging=$(mktemp -d "${TMPDIR:-/tmp}/data-grl-tokenizers.XXXXXX")
trap 'rm -rf "$staging"' EXIT
bundle="$staging/data_grl_tokenizer_artifacts"
mkdir -p "$bundle/q1" "$bundle/q8" "$bundle/preprocessing"

for object in "${objects[@]}"; do
  q1_source=$(pick_checkpoint "$RUN_BASE/${q1_runs[$object]}")
  q8_source=$(pick_checkpoint "$RUN_BASE/${q8_runs[$object]}")
  preprocess_source="$PREPROCESS_BASE/${preprocessors[$object]}"
  [[ -s "$preprocess_source" ]] || {
    echo "Missing preprocessor: $preprocess_source" >&2
    exit 2
  }

  cp "$q1_source" "$bundle/q1/${object}.ckpt"
  cp "$q8_source" "$bundle/q8/${object}.ckpt"
  cp "$preprocess_source" "$bundle/preprocessing/${object}.joblib"
done

(
  cd "$bundle"
  find q1 q8 preprocessing -type f -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum > SHA256SUMS
)

python3 - "$bundle" "$RUN_BASE" "$PREPROCESS_BASE" <<'PY'
import json
import sys
from pathlib import Path

bundle = Path(sys.argv[1])
manifest = {
    "format_version": 1,
    "purpose": "data-GRL aligned Q1/Q8/continuous export",
    "source_run_base": sys.argv[2],
    "source_preprocess_base": sys.argv[3],
    "q1": {
        "codebook_dim": 8,
        "codebook_size": 16384,
        "num_quantizers": 1,
        "objects": sorted(path.stem for path in (bundle / "q1").glob("*.ckpt")),
    },
    "q8": {
        "codebook_dim": 8,
        "num_quantizers": 8,
        "codebook_sizes": {
            "jets": 2048,
            "electrons": 2048,
            "muons": 2048,
            "photons": 2048,
            "taus": 4096,
            "tracks": 4096,
        },
        "objects": sorted(path.stem for path in (bundle / "q8").glob("*.ckpt")),
    },
    "preprocessing": {
        path.stem: path.name
        for path in sorted((bundle / "preprocessing").glob("*.joblib"))
    },
}
(bundle / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

archive="$OUTPUT_DIR/$ARCHIVE_NAME"
tar -C "$staging" -czf "$archive" data_grl_tokenizer_artifacts
(
  cd "$OUTPUT_DIR"
  sha256sum "$ARCHIVE_NAME" | tee "$ARCHIVE_NAME.sha256"
)

echo "Validated bundle contents:"
tar -tzf "$archive" | LC_ALL=C sort

if command -v gcloud >/dev/null 2>&1; then
  echo "Uploading tokenizer bundle to $GCS_DESTINATION/"
  gcloud storage cp "$archive" "$archive.sha256" "$GCS_DESTINATION/"
  gcloud storage ls --long "$GCS_DESTINATION/"
else
  echo "gcloud is unavailable; bundle retained locally at $archive"
fi
