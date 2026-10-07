#!/usr/bin/env bash
set -euo pipefail

# Build matched five-class Q1 and Q8 datasets after the additional H5 files
# have been tokenized. The class definitions and split seed are identical.

ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
RESULTS=${RESULTS:-${ROOT}/results}
DRY_RUN=${DRY_RUN:-1}
OVERWRITE=${OVERWRITE:-0}
PIXI=${PIXI:-/root/.pixi/bin/pixi}
PYARROW_PKG=${PYARROW_PKG:-/home/magaras/.cache/rattler/cache/pkgs/pyarrow-core-16.1.0-py311hf1d6e26_2_cpu}

find_one () {
  local directory=$1
  local sample=$2
  mapfile -t matches < <(find "$directory" -maxdepth 1 -type f -name "event_tokens_${sample}_*seq256.parquet" -print | sort)
  if [[ ${#matches[@]} -ne 1 ]]; then
    echo "Expected one ${sample} Parquet in ${directory}, found ${#matches[@]}" >&2
    printf '  %s\n' "${matches[@]}" >&2
    exit 1
  fi
  printf '%s\n' "${matches[0]}"
}

Q1_BASE=${Q1_BASE:-${RESULTS}/event_tokens_grouped_cls_q1_final_new_mcdata}
Q8_BASE=${Q8_BASE:-${RESULTS}/event_tokens_grouped_cls_final_new_mcdata}
Q1_SIGNAL_PARQUET=${Q1_SIGNAL_PARQUET:-$(find_one "$Q1_BASE" signal)}
Q1_BACKGROUND_PARQUET=${Q1_BACKGROUND_PARQUET:-$(find_one "$Q1_BASE" background)}
Q8_SIGNAL_PARQUET=${Q8_SIGNAL_PARQUET:-$(find_one "$Q8_BASE" signal)}
Q8_BACKGROUND_PARQUET=${Q8_BACKGROUND_PARQUET:-$(find_one "$Q8_BASE" background)}
Q1_NEW_PARQUET=${Q1_NEW_PARQUET:-${RESULTS}/event_tokens_hzz_production_modes/q1/event_tokens_signal_hzz_production_modes_q1_seq256.parquet}
Q8_NEW_PARQUET=${Q8_NEW_PARQUET:-${RESULTS}/event_tokens_hzz_production_modes/q8/event_tokens_signal_hzz_production_modes_q8_seq256.parquet}
OUTPUT_BASE=${OUTPUT_BASE:-${RESULTS}/grouped_hzz_production_multiclass_prepared}

for path in \
  "$Q1_SIGNAL_PARQUET" "$Q1_BACKGROUND_PARQUET" "$Q1_NEW_PARQUET" \
  "$Q8_SIGNAL_PARQUET" "$Q8_BACKGROUND_PARQUET" "$Q8_NEW_PARQUET"; do
  [[ -s "$path" ]] || { echo "Missing input Parquet: $path" >&2; exit 1; }
done

STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)}
UNIT=${UNIT:-atlas-hzz-production-multiclass-shards-${STAMP}}
WORKER=${RESULTS}/tmp/atlas_hzz_production_multiclass/${UNIT}.sh
LOG_FILE=${RESULTS}/logs/atlas_hzz_production_multiclass/shard_preparation.log
mkdir -p "$(dirname "$WORKER")" "$(dirname "$LOG_FILE")"

overwrite_flag=
if [[ "$OVERWRITE" -eq 1 ]]; then
  overwrite_flag=--overwrite
fi

cat > "$WORKER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '${ROOT}'
exec > >(tee -a '${LOG_FILE}') 2>&1

CACHE_LIBS=\$(find /home/magaras/.cache/rattler/cache/pkgs -type d -path '*/lib' -print | paste -sd: -)
export PYTHONPATH='${ROOT}/src:${PYARROW_PKG}/lib/python3.11/site-packages'\${PYTHONPATH:+:\${PYTHONPATH}}
export LD_LIBRARY_PATH="\${CACHE_LIBS}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"

prepare () {
  local output=\$1
  local signal=\$2
  local background=\$3
  local added=\$4
  '${PIXI}' run python scripts/prepare_grouped_hzz_multiclass_shards.py \
    --class-spec "ggf:0:\${signal}:345060" \
    --class-spec "vbf:1:\${background}:346228" \
    --class-spec "vh:2:\${added}:346645,345066,346646,346647" \
    --class-spec "tth:3:\${background}:346340,346341,346342" \
    --class-spec "tth:3:\${added}:346414,346511" \
    --class-spec "zz_continuum:4:\${background}:700600" \
    --output-dir "\${output}" \
    --train-frac 0.70 \
    --val-frac 0.15 \
    --seed 42 \
    --read-batch-size 4096 \
    --shard-rows 50000 \
    ${overwrite_flag}
}

prepare '${OUTPUT_BASE}/q1' '${Q1_SIGNAL_PARQUET}' '${Q1_BACKGROUND_PARQUET}' '${Q1_NEW_PARQUET}'
prepare '${OUTPUT_BASE}/q8' '${Q8_SIGNAL_PARQUET}' '${Q8_BACKGROUND_PARQUET}' '${Q8_NEW_PARQUET}'
EOF
chmod +x "$WORKER"

echo "Matched HZZ production-mode shard preparation"
echo "  Q1 output: ${OUTPUT_BASE}/q1"
echo "  Q8 output: ${OUTPUT_BASE}/q8"
echo "  classes:   ggf,vbf,vh,tth,zz_continuum"
echo "  worker:    ${WORKER}"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY RUN: would submit ${UNIT}.service"
  exit 0
fi

systemd-run \
  --unit="$UNIT" \
  --description="Prepare matched five-class Q1/Q8 HZZ production-mode shards" \
  --collect \
  --property=WorkingDirectory="$ROOT" \
  /bin/bash "$WORKER"

echo "Submitted ${UNIT}.service"
echo "Follow: journalctl -u ${UNIT}.service -f -o cat"
