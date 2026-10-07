#!/usr/bin/env bash
set -euo pipefail

if (($# != 3)); then
  echo "Usage: compare_core_snapshots.sh ZEPHYR.tar.gz GOOGLE.tar.gz OUTPUT_DIR" >&2
  exit 2
fi

LEFT_ARCHIVE=$1
RIGHT_ARCHIVE=$2
OUTPUT=$3
mkdir -p "$OUTPUT"
TMP=$(mktemp -d "${TMPDIR:-/tmp}/heptokens-compare.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/zephyr" "$TMP/google"
tar -xzf "$LEFT_ARCHIVE" -C "$TMP/zephyr"
tar -xzf "$RIGHT_ARCHIVE" -C "$TMP/google"

LEFT="$TMP/zephyr/snapshot"
RIGHT="$TMP/google/snapshot"

diff -qr "$LEFT/tree" "$RIGHT/tree" > "$OUTPUT/differing_core_files.txt" || true
diff -ru "$LEFT/tree" "$RIGHT/tree" > "$OUTPUT/core_code.diff" || true
diff -ru -x source_path.txt "$LEFT/runs" "$RIGHT/runs" \
  > "$OUTPUT/run_provenance.diff" || true
diff -u "$LEFT/core_paths.txt" "$RIGHT/core_paths.txt" \
  > "$OUTPUT/core_path_presence.diff" || true

cp "$LEFT/metadata.txt" "$OUTPUT/zephyr_metadata.txt"
cp "$RIGHT/metadata.txt" "$OUTPUT/google_metadata.txt"
cp "$LEFT/git_identity.txt" "$OUTPUT/zephyr_git_identity.txt"
cp "$RIGHT/git_identity.txt" "$OUTPUT/google_git_identity.txt"
cp "$LEFT/git_status.txt" "$OUTPUT/zephyr_git_status.txt"
cp "$RIGHT/git_status.txt" "$OUTPUT/google_git_status.txt"

{
  echo "Core file differences: $(wc -l < "$OUTPUT/differing_core_files.txt" | tr -d ' ')"
  echo "Core diff lines: $(wc -l < "$OUTPUT/core_code.diff" | tr -d ' ')"
  echo "Run provenance diff lines: $(wc -l < "$OUTPUT/run_provenance.diff" | tr -d ' ')"
  echo
  cat "$OUTPUT/differing_core_files.txt"
} > "$OUTPUT/summary.txt"

echo "Wrote comparison to $OUTPUT"
cat "$OUTPUT/summary.txt"
