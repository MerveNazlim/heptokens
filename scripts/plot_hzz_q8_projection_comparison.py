#!/usr/bin/env python3
"""Plot cached projected-Q8 and Google Q8-direct HZZ test predictions on CPU."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from evaluate_q1_hierarchical_flat_hzz import write_plots
from evaluate_q8_decoder_downstream_comparison import plot_roc


def validate_scores(labels: np.ndarray, scores: np.ndarray) -> None:
    if labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("Expected one label and signal score per event")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Expected binary labels background=0, signal=1")
    if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
        raise ValueError("Signal scores must be finite probabilities in [0,1]")


def load_inputs(direct_dir: Path, projected_dir: Path, prepared_dir: Path) -> tuple[list[dict], dict]:
    manifest_path = prepared_dir / "manifest.json"
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("token_shape") != [256, 8] or manifest.get("identity_overlap_detected") is not False:
        raise ValueError("Expected the audited grouped-CLS Q8 dataset")
    for name, dsid, label in (("signal", 345060, 1), ("background", 700600, 0)):
        if (manifest[name]["dsid"], manifest[name]["label"]) != (dsid, label):
            raise ValueError(f"Unexpected {name} DSID/label in the reference split")

    report = json.loads((direct_dir / "evaluation_summary.json").read_text())
    if report.get("manifest_sha256") != digest:
        raise ValueError("Google predictions were not exported from this exact manifest")
    if report.get("object_projection_mode") != "identity" or report.get("signal_label") != 1:
        raise ValueError("Google export is not a Q8 identity-backbone signal-score export")
    if report.get("score_definition") != "softmax(logits)[:,1]":
        raise ValueError("Unexpected Google signal-score definition")
    with np.load(direct_dir / "test_predictions.npz", allow_pickle=False) as archive:
        direct_labels = archive["label"].copy()
        direct_scores = archive["signal_score"].copy()
        sources, indices = archive["source_file"], archive["event_index"]
        if sources.shape != direct_labels.shape or indices.shape != direct_labels.shape:
            raise ValueError("Google event identities are not aligned with its labels")
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("Google event indices must be integers")
        identities = [(str(source), int(index)) for source, index in zip(sources, indices, strict=True)]
        if len(set(identities)) != len(identities) or np.any(indices < 0) or any(not str(s) for s in sources):
            raise ValueError("Google export contains duplicate or invalid event identities")

    rows = json.loads((projected_dir / "summary.json").read_text())
    matches = [row for row in rows if row.get("setup") == "Pretrained, fine-tuned" and row.get("representation") == "q8"]
    if len(matches) != 1:
        raise ValueError("Expected one pretrained, fine-tuned Q8 baseline in summary.json")
    baseline = matches[0]
    config_path = Path(baseline["checkpoint"]).parent.parent / "full_config.yaml"
    config = yaml.safe_load(config_path.read_text())
    model_config = config["model"]
    if model_config.get("object_projection_mode", "linear") != "linear" or model_config.get("freeze_backbone") is not False:
        raise ValueError("Baseline is not a fine-tuned, linear-projection classifier")
    baseline_prepared = Path(config["datamodule"]["prepared_dir"])
    if not baseline_prepared.is_absolute():
        baseline_prepared = Path(__file__).resolve().parents[1] / baseline_prepared
    baseline_digest = hashlib.sha256((baseline_prepared / "manifest.json").read_bytes()).hexdigest()
    if baseline_digest != digest:
        raise ValueError("Projected Q8 was fine-tuned with a different dataset manifest")
    with np.load(projected_dir / "roc_arrays.npz", allow_pickle=False) as archive:
        projected_labels = archive["pretrained_fine_tuned_labels"].copy()
        projected_scores = archive["pretrained_fine_tuned_scores"].copy()

    inputs = [
        {"name": "Q8 with projection", "key": "projected", "color": "#0072B2", "linestyle": "-",
         "labels": projected_labels, "scores": projected_scores, "checkpoint": baseline["checkpoint"]},
        {"name": "Q8 no projection", "key": "direct", "color": "#D55E00", "linestyle": "--",
         "labels": direct_labels, "scores": direct_scores, "checkpoint_sha256": report.get("checkpoint_sha256")},
    ]
    expected = manifest["split_counts"]["test"]
    for item in inputs:
        labels, scores = item["labels"], item["scores"]
        validate_scores(labels, scores)
        observed = {"total": int(labels.size), "signal": int(np.count_nonzero(labels == 1)),
                    "background": int(np.count_nonzero(labels == 0))}
        if any(observed[name] != expected[name] for name in observed):
            raise ValueError(f"Incomplete or different test class counts for {item['name']}: {observed}")
    if int(report["test_events"]) != direct_labels.size:
        raise ValueError("Google report count disagrees with the saved predictions")
    if not np.isclose(roc_auc_score(direct_labels, direct_scores), report["test_auc"], atol=1e-8, rtol=0):
        raise ValueError("Google scores do not reproduce the exported AUC")
    if not np.isclose(roc_auc_score(projected_labels, projected_scores), baseline["auc"], atol=1e-8, rtol=0):
        raise ValueError("Baseline scores do not reproduce the cached AUC")
    metadata = {
        "direct_predictions": str(direct_dir.resolve()), "projected_results": str(projected_dir.resolve()),
        "prepared_dir": str(prepared_dir.resolve()), "manifest_sha256": digest,
        "google_export": report, "projected_training_config": str(config_path),
        "projected_evaluation": baseline,
        "comparison_validation": "Matching dataset manifests and complete test class counts; Google identities are unique. The old projected cache has no event IDs, so cross-model identity pairing cannot be independently audited.",
        "ordering": "Each model uses its own paired labels/scores; exports need not share row order.",
        "background_rejection_floor": "1 / number of test background events",
    }
    return inputs, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-predictions", type=Path, required=True)
    parser.add_argument("--projected-results", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Choose a new output directory; previous plots are kept")
    inputs, metadata = load_inputs(args.direct_predictions, args.projected_results, args.prepared_dir)
    rows, curves, arrays = [], [], {}
    for item in inputs:
        labels, scores = item["labels"], item["scores"]
        fpr, tpr, thresholds = roc_curve(labels, scores)
        auc = float(roc_auc_score(labels, scores))
        n_background = int(np.count_nonzero(labels == 0))
        row = {"run": item["name"], "test_events": int(labels.size), "auc": auc,
               "accuracy_at_0p5": float(accuracy_score(labels, scores >= 0.5))}
        for efficiency in (50, 70, 80):
            row[f"rejection_at_{efficiency}pct"] = float(1 / max(np.interp(efficiency / 100, tpr, fpr), 1 / n_background))
        rows.append(row)
        curves.append({"name": item["name"], "color": item["color"], "linestyle": item["linestyle"],
                       "fpr": fpr, "tpr": tpr, "auc": auc, "background_events": n_background})
        for suffix, value in (("labels", labels), ("scores", scores), ("fpr", fpr), ("tpr", tpr), ("thresholds", thresholds)):
            arrays[f"{item['key']}_{suffix}"] = value
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_plots(args.output_dir, curves)
    plot_roc(curves, args.output_dir, "roc")
    np.savez_compressed(args.output_dir / "roc_arrays.npz", **arrays)
    (args.output_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output_dir / "comparison_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['run']:20s} AUC={row['auc']:.6f} accuracy={row['accuracy_at_0p5']:.4f} "
              f"R50={row['rejection_at_50pct']:.2f} R70={row['rejection_at_70pct']:.2f} R80={row['rejection_at_80pct']:.2f}")
    print("Used each export's own label/score order. No model loading or GPU inference.")
    print(f"Plots: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
