#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: create_core_snapshot.sh --tag TAG [--root REPO] [--output FILE] [--run DIR ...]

Create a read-only comparison archive containing the scientific core of HepTokens,
Git metadata, resolved Hydra configuration for selected runs, and checkpoint hashes.
EOF
}

hash_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

ROOT=$PWD
TAG=
OUTPUT=
RUNS=()
while (($#)); do
  case "$1" in
    --tag) TAG=$2; shift 2 ;;
    --root) ROOT=$2; shift 2 ;;
    --output) OUTPUT=$2; shift 2 ;;
    --run) RUNS+=("$2"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$TAG" ]]; then
  echo "--tag is required" >&2
  exit 2
fi
ROOT=$(cd "$ROOT" && pwd)
STAMP=$(date +%Y%m%d-%H%M%S)
OUTPUT=${OUTPUT:-$PWD/heptokens_core_${TAG}_${STAMP}.tar.gz}
TMP=$(mktemp -d "${TMPDIR:-/tmp}/heptokens-core-${TAG}.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
SNAPSHOT="$TMP/snapshot"
mkdir -p "$SNAPSHOT/tree" "$SNAPSHOT/runs"

CORE_PATHS=(
  src/heptokens
  scripts/train.py
  scripts/foundation_model.py
  scripts/get_preprocessing.py
  scripts/get_atlas_object_preprocessing.py
  scripts/get_atlas_combined_object_preprocessing.py
  scripts/tokenize_objects_to_grouped_parquet.py
  scripts/prepare_token_parquet_pretrain_shards.py
  scripts/prepare_grouped_hzz_classification_shards.py
  configs/train.yaml
  configs/hydra/default.yaml
  configs/model/vqvae.yaml
  configs/model/foundation_grouped_pretrain.yaml
  configs/model/foundation_grouped_autoregressive_pretrain.yaml
  configs/model/foundation_grouped_cls_classifier.yaml
  configs/model/foundation_grouped_cls_mlp_classifier.yaml
  configs/datamodule/atlas_event_object.yaml
  configs/datamodule/atlas_event_object_iterable.yaml
  configs/datamodule/token_parquet_pretrain.yaml
  configs/datamodule/token_parquet_grouped_classification.yaml
  configs/callbacks/encode.yaml
  configs/callbacks/event_tokenizer.yaml
  configs/callbacks/pretrain.yaml
  configs/callbacks/grouped_classification.yaml
  pyproject.toml
  pixi.toml
  pixi.lock
)

PRESENT=()
{
  for path in "${CORE_PATHS[@]}"; do
    if [[ -e "$ROOT/$path" ]]; then
      PRESENT+=("$path")
      echo "present  $path"
    else
      echo "missing  $path"
    fi
  done
} > "$SNAPSHOT/core_paths.txt"

if ((${#PRESENT[@]})); then
  tar -C "$ROOT" \
    --exclude='*/__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='.DS_Store' \
    --exclude='*/Untitled*' \
    -cf - "${PRESENT[@]}" \
    | tar -C "$SNAPSHOT/tree" -xf -
fi

{
  echo "tag=$TAG"
  echo "captured_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname -f 2>/dev/null || hostname)"
  echo "root=$ROOT"
  echo "kernel=$(uname -srmo)"
  command -v python >/dev/null && echo "python=$(python --version 2>&1)"
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
} > "$SNAPSHOT/metadata.txt"

if git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  {
    echo "commit=$(git -C "$ROOT" rev-parse HEAD)"
    echo "branch=$(git -C "$ROOT" branch --show-current)"
    git -C "$ROOT" remote -v
  } > "$SNAPSHOT/git_identity.txt"
  git -C "$ROOT" status --short --untracked-files=all -- "${PRESENT[@]}" \
    > "$SNAPSHOT/git_status.txt"
  git -C "$ROOT" diff --binary HEAD -- "${PRESENT[@]}" \
    > "$SNAPSHOT/git_worktree.patch"
else
  echo "not a Git worktree" > "$SNAPSHOT/git_identity.txt"
  : > "$SNAPSHOT/git_status.txt"
  : > "$SNAPSHOT/git_worktree.patch"
fi

for index in "${!RUNS[@]}"; do
  run=${RUNS[$index]}
  [[ "$run" = /* ]] || run="$ROOT/$run"
  destination="$SNAPSHOT/runs/$(printf '%02d' "$index")"
  mkdir -p "$destination"
  echo "$run" > "$destination/source_path.txt"
  basename "$run" > "$destination/run_name.txt"
  if [[ ! -d "$run" ]]; then
    echo "missing run directory" > "$destination/MISSING.txt"
    continue
  fi
  if [[ -d "$run/.hydra" ]]; then
    cp -a "$run/.hydra" "$destination/"
  fi
  : > "$destination/checkpoint_sha256.txt"
  if [[ -d "$run/checkpoints" ]]; then
    while IFS= read -r checkpoint; do
      checksum=$(hash_file "$checkpoint")
      printf '%s  %s\n' "$checksum" "$(basename "$checkpoint")" \
        >> "$destination/checkpoint_sha256.txt"
    done < <(find "$run/checkpoints" -maxdepth 1 -type f -name '*.ckpt' -print | LC_ALL=C sort)
  fi
done

(
  cd "$SNAPSHOT"
  while IFS= read -r file; do
    printf '%s  %s\n' "$(hash_file "$file")" "$file"
  done < <(find tree -type f -print | LC_ALL=C sort)
) > "$SNAPSHOT/core_sha256.txt"

mkdir -p "$(dirname "$OUTPUT")"
tar -C "$TMP" -czf "$OUTPUT" snapshot
echo "Wrote $OUTPUT"
