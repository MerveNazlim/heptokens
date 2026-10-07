#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/magaras/heptok_fork/heptokens
cd "$ROOT"

MC_DIR=/home/zephyr/Data/viviana/bnl-treasure/data/h5
DATA_DIR=/home/zephyr/Data/viviana/bnl-treasure/data/h5/realdata

OUT_ROOT=results/fair_mc_vs_mcdata_eval
LOGDIR=results/logs/fair_mc_vs_mcdata_eval
mkdir -p "$OUT_ROOT" "$LOGDIR"

GPU=${GPU:-0}
MAX_VALID_OBJECTS=${MAX_VALID_OBJECTS:-1000000}
BATCH_SIZE=${BATCH_SIZE:-2048}
N_FILES=${N_FILES:-20}
NUM_EVENTS_PER_FILE=${NUM_EVENTS_PER_FILE:-200000}
FORCE=${FORCE:-0}

mapfile -t MC_FILES < <(
  find "$MC_DIR" -maxdepth 1 -type f -name "*.h5" -size +0c \
    | sort \
    | head -n "$N_FILES"
)

mapfile -t DATA_FILES < <(
  find "$DATA_DIR" -maxdepth 1 -type f -name "*.h5" -size +0c \
    | sort \
    | head -n "$N_FILES"
)

echo "MC files: ${#MC_FILES[@]}"
echo "Data files: ${#DATA_FILES[@]}"
echo "num_events_per_file: ${NUM_EVENTS_PER_FILE}"
echo "max_valid_objects: ${MAX_VALID_OBJECTS}"

if [ "${#MC_FILES[@]}" -eq 0 ] || [ "${#DATA_FILES[@]}" -eq 0 ]; then
  echo "Missing MC or real-data files"
  exit 1
fi

printf '%s\n' "${MC_FILES[@]}" > "${OUT_ROOT}/mc_files_used.txt"
printf '%s\n' "${DATA_FILES[@]}" > "${OUT_ROOT}/realdata_files_used.txt"

COMPARISONS=(
  "jets|results/atlas_event_tokenizers_1606_jets_logstd_capacity_scan/jets_logstd_dim8_cb4096_q4|results/atlas_event_tokenizers_0107_logstd_mc_realdata/jets_logstd_dim8_cb4096_q4"
  "electrons|results/atlas_event_tokenizers_1806_nonjet_logstd_capacity_scan/electrons_logstd_dim8_cb4096_q4|results/atlas_event_tokenizers_0107_logstd_mc_realdata/electrons_logstd_dim8_cb4096_q4"
  "muons|results/atlas_event_tokenizers_1806_nonjet_logstd_capacity_scan/muons_logstd_dim8_cb4096_q4|results/atlas_event_tokenizers_0107_logstd_mc_realdata/muons_logstd_dim8_cb4096_q4"
  "photons|results/atlas_event_tokenizers_1806_nonjet_logstd_capacity_scan/photons_logstd_dim8_cb4096_q4|results/atlas_event_tokenizers_0107_logstd_mc_realdata/photons_logstd_dim8_cb4096_q4"
  "taus|results/atlas_event_tokenizers_1806_nonjet_logstd_capacity_scan/taus_logstd_dim8_cb4096_q4|results/atlas_event_tokenizers_0107_logstd_mc_realdata/taus_logstd_dim8_cb4096_q4"
  "taus_8k|results/atlas_event_tokenizers_1806_nonjet_logstd_capacity_scan/taus_logstd_dim8_cb8192_q4|results/atlas_event_tokenizers_0107_logstd_mc_realdata/taus_logstd_dim8_cb8192_q4"
  "tracks|results/atlas_event_tokenizers_1906_tracks_logstd_tests/tracks_logstd_no_ndoflog_dim8_cb8192_q4|results/atlas_event_tokenizers_0107_logstd_mc_realdata_b1024_debug/tracks_logstd_dim8_cb8192_q4_b1024"
)

run_diag () {
  SAMPLE=$1
  TAG=$2
  RUN_DIR=$3
  shift 3
  FILES=("$@")

  OUT_DIR="${OUT_ROOT}/${SAMPLE}/${TAG}"

  if [ "$FORCE" -eq 1 ]; then
    rm -rf "$OUT_DIR"
  fi
  mkdir -p "$OUT_DIR"

  if [ "$FORCE" -ne 1 ] \
    && [ -f "$OUT_DIR/reconstruction_metrics.json" ] \
    && [ -f "$OUT_DIR/codebook_usage_summary.json" ]; then
    echo "Skipping existing ${SAMPLE}/${TAG}"
    return 0
  fi

  if [ ! -f "$RUN_DIR/full_config.yaml" ]; then
    echo "Missing run: $RUN_DIR"
    return 2
  fi

  if [ ! -f "$RUN_DIR/checkpoints/best.ckpt" ] && [ ! -f "$RUN_DIR/checkpoints/last.ckpt" ]; then
    echo "Missing checkpoint: $RUN_DIR"
    return 2
  fi

  {
    echo "sample: $SAMPLE"
    echo "tag: $TAG"
    echo "run_dir: $RUN_DIR"
    echo "num_events_per_file: $NUM_EVENTS_PER_FILE"
    echo "max_valid_objects: $MAX_VALID_OBJECTS"
    echo "files:"
    printf '%s\n' "${FILES[@]}"
  } > "${OUT_DIR}/diagnostic_manifest.txt"

  echo "Running ${SAMPLE}/${TAG}"

  CUDA_VISIBLE_DEVICES="$GPU" /root/.pixi/bin/pixi run python scripts/analyze_vqvae_tokenizer.py \
    --run-dir "$RUN_DIR" \
    --checkpoint "$RUN_DIR/checkpoints/last.ckpt" \
    --h5-files "${FILES[@]}" \
    --device cuda \
    --split val \
    --batch-size "$BATCH_SIZE" \
    --num-events-per-file "$NUM_EVENTS_PER_FILE" \
    --max-valid-objects "$MAX_VALID_OBJECTS" \
    --output-dir "$OUT_DIR" \
    > "${LOGDIR}/${SAMPLE}_${TAG}.log" 2>&1
}

STATUS=0

for ROW in "${COMPARISONS[@]}"; do
  IFS="|" read -r OBJ OLD_RUN NEW_RUN <<< "$ROW"

  run_diag "mc" "${OBJ}_mc_only" "$OLD_RUN" "${MC_FILES[@]}" || STATUS=1
  run_diag "mc" "${OBJ}_mcdata" "$NEW_RUN" "${MC_FILES[@]}" || STATUS=1

  run_diag "realdata" "${OBJ}_mc_only" "$OLD_RUN" "${DATA_FILES[@]}" || STATUS=1
  run_diag "realdata" "${OBJ}_mcdata" "$NEW_RUN" "${DATA_FILES[@]}" || STATUS=1
done

/root/.pixi/bin/pixi run python - <<'PY'
import json
import math
from pathlib import Path

root = Path("results/fair_mc_vs_mcdata_eval")
objects = ["jets", "electrons", "muons", "photons", "taus", "taus_8k", "tracks"]

feature_map = {
    "jets": ["pt", "eta", "mass", "n_trk", "GN2_pc", "GN2_pu"],
    "electrons": ["pt", "eta", "charge", "ptvarcone30", "topoetcone20", "LHMedium", "LHTight"],
    "muons": ["pt", "eta", "charge", "ptvarcone30", "topoetcone20"],
    "photons": ["pt", "eta", "isTight", "ptcone20", "topoetcone20", "topoetcone40"],
    "taus": ["pt", "eta", "charge", "NNDecayMode", "RNNJetScore", "RNNEleScore"],
    "taus_8k": ["pt", "eta", "charge", "NNDecayMode", "RNNJetScore", "RNNEleScore"],
    "tracks": ["pt", "eta", "phi", "d0", "z0", "nDoF"],
}

def load(sample, tag):
    d = root / sample / tag
    reco_path = d / "reconstruction_metrics.json"
    usage_path = d / "codebook_usage_summary.json"
    if not reco_path.exists() or not usage_path.exists():
        return None, None
    return json.loads(reco_path.read_text()), json.loads(usage_path.read_text())

def mean_used(usage):
    vals = [
        float(item["percent_used"])
        for key, item in usage.items()
        if key.startswith("quantizer_")
    ]
    return sum(vals) / len(vals) if vals else float("nan")

def fmt(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    if abs(x) >= 100:
        return f"{x:.2f}"
    if abs(x) >= 10:
        return f"{x:.3f}"
    if abs(x) >= 1:
        return f"{x:.4f}"
    return f"{x:.5f}"

def row_for(sample, obj, variant):
    reco, usage = load(sample, f"{obj}_{variant}")
    if reco is None:
        return None

    out = {
        "model": "MC only" if variant == "mc_only" else "MC+data",
        "used": mean_used(usage),
    }

    for feat in feature_map[obj]:
        if feat in reco:
            out[f"{feat}_rmse"] = reco[feat]["rmse"]
            out[f"{feat}_mae"] = reco[feat]["mae"]
    return out

report = []

for sample in ["mc", "realdata"]:
    report.append(f"\n\n## {sample.upper()} fixed-sample evaluation\n")

    for obj in objects:
        rows = [row_for(sample, obj, variant) for variant in ["mc_only", "mcdata"]]
        rows = [row for row in rows if row is not None]
        if len(rows) < 2:
            continue

        features = [f for f in feature_map[obj] if any(f"{f}_rmse" in r for r in rows)]

        report.append(f"\n### {obj}\n")
        header = ["model", "mean used %"] + [f"{f} RMSE" for f in features] + [f"{f} MAE" for f in features]
        report.append("| " + " | ".join(header) + " |")
        report.append("|" + "|".join(["---"] * len(header)) + "|")

        for r in rows:
            vals = [r["model"], fmt(r["used"])]
            vals += [fmt(r.get(f"{f}_rmse")) for f in features]
            vals += [fmt(r.get(f"{f}_mae")) for f in features]
            report.append("| " + " | ".join(vals) + " |")

        wins = {"MC only": 0, "MC+data": 0}
        old, new = rows
        for f in features:
            for metric in ["rmse", "mae"]:
                key = f"{f}_{metric}"
                if key in old and key in new:
                    winner = "MC only" if old[key] < new[key] else "MC+data"
                    wins[winner] += 1

        report.append(
            f"\nWinner count over listed RMSE/MAE metrics: "
            f"MC only={wins['MC only']}, MC+data={wins['MC+data']}\n"
        )

text = "\n".join(report)
print(text)
(root / "fair_comparison_report.md").write_text(text + "\n")
PY

exit "$STATUS"