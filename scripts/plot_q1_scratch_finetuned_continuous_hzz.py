#!/usr/bin/env python3
"""Combine saved HZZ results without rerunning training or model inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from evaluate_q1_hierarchical_flat_hzz import write_plots


def read_saved_curve(
    source_dir: Path,
    setup: str,
    representation: str,
    name: str,
    color: str,
    linestyle: str = "-",
) -> tuple[dict, dict, np.ndarray]:
    rows = json.loads((source_dir / "summary.json").read_text())
    matches = [row for row in rows if row.get("setup") == setup]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {setup!r} result in {source_dir}")
    row = matches[0]
    if row["representation"] != representation:
        raise ValueError(f"Expected {representation} results under {source_dir}")
    prefix = re.sub(r"[^A-Za-z0-9]+", "_", setup).strip("_").lower()
    with np.load(source_dir / "roc_arrays.npz", allow_pickle=False) as arrays:
        fpr = arrays[f"{prefix}_fpr"]
        tpr = arrays[f"{prefix}_tpr"]
        labels = arrays[f"{prefix}_labels"]
    if labels.ndim != 1 or labels.size != row["events"]:
        raise ValueError(f"Saved event count does not match labels for {name}")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError(f"Expected both binary classes for {name}")
    if fpr.ndim != 1 or fpr.size == 0 or fpr.shape != tpr.shape:
        raise ValueError(f"Invalid saved ROC array shapes for {name}")
    for values in (fpr, tpr):
        if (
            not np.isfinite(values).all()
            or np.any(values < 0)
            or np.any(values > 1)
            or np.any(np.diff(values) < 0)
        ):
            raise ValueError(f"Invalid saved ROC values for {name}")
    if not np.isfinite(row["auc"]) or not 0 <= row["auc"] <= 1:
        raise ValueError(f"Invalid saved AUC for {name}")
    curve = {
        "name": name,
        "auc": row["auc"],
        "fpr": fpr,
        "tpr": tpr,
        "background_events": int(np.count_nonzero(labels == 0)),
        "color": color,
        "linestyle": linestyle,
    }
    summary = {
        "run": name,
        "representation": representation,
        "setup": setup,
        "events": row["events"],
        "auc": row["auc"],
        "rejection_at_50pct": row["rejection_at_50pct"],
        "rejection_at_70pct": row["rejection_at_70pct"],
        "rejection_at_80pct": row["rejection_at_80pct"],
        "checkpoint": row["checkpoint"],
        "source_results": str(source_dir.resolve()),
    }
    return curve, summary, labels


def validate_direct_replacement(
    directory: Path,
    curve: dict,
    summary: dict,
    labels: np.ndarray,
    reference_labels: np.ndarray,
    scratch_summary: dict,
    prepared_dir: Path | None,
) -> dict:
    from sklearn.metrics import roc_auc_score, roc_curve

    report = json.loads((directory / "evaluation_summary.json").read_text())
    expected = {
        "architecture": "q1_direct256",
        "representation": "q1",
        "setup": "Pretrained, fine-tuned",
        "quantizer_embedding_dim": 256,
        "object_projection_mode": "identity",
        "signal_label": 1,
        "score_definition": "softmax(logits)[:,1]",
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"Q1-direct export requires {key}={value!r}")
    for key in ("events", "auc", "checkpoint"):
        if report.get(key) != summary[key]:
            raise ValueError(f"Q1-direct summary and evaluation report disagree on {key}")

    with np.load(directory / "test_predictions.npz", allow_pickle=False) as predictions:
        scores = predictions["signal_score"]
        if not np.array_equal(predictions["label"], labels):
            raise ValueError("Q1-direct prediction labels differ from its saved ROC labels")
        if scores.shape != labels.shape or not np.isfinite(scores).all():
            raise ValueError("Invalid Q1-direct signal scores")
        if np.any(scores < 0) or np.any(scores > 1):
            raise ValueError("Q1-direct signal scores must be probabilities in [0,1]")
        sources, indices = predictions["source_file"], predictions["event_index"]
        if sources.shape != labels.shape or indices.shape != labels.shape:
            raise ValueError("Q1-direct event identities are not aligned with predictions")
        if not np.issubdtype(indices.dtype, np.integer) or np.any(indices < 0):
            raise ValueError("Invalid Q1-direct event indices")
        identities = [(str(source), int(index)) for source, index in zip(sources, indices)]
        if any(not source for source, _ in identities) or len(set(identities)) != len(labels):
            raise ValueError("Duplicate or missing Q1-direct event identities")
        with np.load(directory / "roc_arrays.npz", allow_pickle=False) as arrays:
            if not np.array_equal(arrays["pretrained_fine_tuned_scores"], scores):
                raise ValueError("Q1-direct prediction scores differ from its saved ROC scores")
        fpr, tpr, _ = roc_curve(labels, scores)
        for key, values in (("fpr", fpr), ("tpr", tpr)):
            if curve[key].shape != values.shape or not np.allclose(
                curve[key], values, atol=1e-12, rtol=0
            ):
                raise ValueError(f"Q1-direct predictions do not reproduce saved ROC {key}")
        if not np.isclose(roc_auc_score(labels, scores), curve["auc"], atol=1e-10, rtol=0):
            raise ValueError("Q1-direct predictions do not reproduce saved AUC")

    counts = {
        "signal": int(np.count_nonzero(labels == 1)),
        "background": int(np.count_nonzero(labels == 0)),
    }
    reference_counts = {
        "signal": int(np.count_nonzero(reference_labels == 1)),
        "background": int(np.count_nonzero(reference_labels == 0)),
    }
    if counts != reference_counts or report.get("class_counts") != counts:
        raise ValueError("Q1-direct test class counts differ from the original comparison")

    config_path = Path(scratch_summary["checkpoint"]).parent.parent / "full_config.yaml"
    manifest_source = "--q1-prepared-dir" if prepared_dir is not None else str(config_path)
    if prepared_dir is None:
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Missing scratch classifier config {config_path}; supply --q1-prepared-dir "
                "with the original Q1 comparison's classification shards"
            )
        import yaml

        config = yaml.safe_load(config_path.read_text())
        saved_prepared = config["datamodule"].get("prepared_dir")
        if not saved_prepared or "${" in str(saved_prepared):
            raise ValueError("Supply --q1-prepared-dir; saved prepared_dir is not a concrete path")
        prepared_dir = Path(saved_prepared)
        if not prepared_dir.is_absolute():
            prepared_dir = Path(__file__).resolve().parents[1] / prepared_dir
    manifest_path = prepared_dir / "manifest.json"
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if report.get("manifest_sha256") != digest:
        raise ValueError("Q1-direct evaluation used a different Q1 classification manifest")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("token_shape") != [256, 1]
        or manifest.get("identity_overlap_detected") is not False
    ):
        raise ValueError("Expected the original audited Q1 [256,1] classification manifest")
    for name, dsid, label in (("signal", 345060, 1), ("background", 700600, 0)):
        if (manifest[name]["dsid"], manifest[name]["label"]) != (dsid, label):
            raise ValueError(f"Unexpected {name} DSID/label in the Q1 manifest")
    expected_counts = manifest["split_counts"]["test"]
    if expected_counts.get("total") != labels.size or any(
        expected_counts.get(name) != count for name, count in counts.items()
    ):
        raise ValueError("Q1-direct results do not cover the complete original test split")

    return {
        "direct_evaluation": report,
        "q1_prepared_dir": str(prepared_dir.resolve()),
        "q1_manifest_sha256": digest,
        "manifest_source": manifest_source,
        "validation": (
            "Direct predictions reproduce the exported ROC/AUC; event identities are unique. "
            "Q1 manifest and complete test class counts match the original comparison. "
            "Scratch/continuous ordered labels match. Their older caches have no event IDs, "
            "so cross-model event-identity pairing cannot be independently audited. "
            "Each curve uses its own paired labels/scores; the direct export may have a different row order."
        ),
    }


def main() -> None:
    base = Path("results/atlas_hzz_200m_data_pretrain_comparison")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q1-results", type=Path, default=base / "four_setup_q1")
    parser.add_argument(
        "--q1-direct-results",
        type=Path,
        help="Replace only the fine-tuned tokenized curve with a Q1 direct-256 test export",
    )
    parser.add_argument(
        "--q1-prepared-dir",
        type=Path,
        help="Original Q1 comparison's classification shards; otherwise read its scratch training config",
    )
    parser.add_argument(
        "--continuous-results",
        type=Path,
        default=base / "four_setup_continuous",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base / "q1_scratch_vs_finetuned_vs_continuous",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    sources = [args.q1_results, args.continuous_results]
    if args.q1_direct_results is not None:
        sources.append(args.q1_direct_results)
    if output in {source.resolve() for source in sources}:
        parser.error("--output-dir must differ from the source results directories")
    if args.q1_prepared_dir is not None and args.q1_direct_results is None:
        parser.error("--q1-prepared-dir requires --q1-direct-results")

    selections = (
        (args.q1_results, "Random, from scratch", "q1", "Tokenized, from scratch", "#0072B2", "--"),
        (
            args.q1_direct_results or args.q1_results,
            "Pretrained, fine-tuned",
            "q1",
            "Tokenized, fine-tuned",
            "#D55E00",
            "-",
        ),
        (
            args.continuous_results,
            "Pretrained, fine-tuned",
            "continuous",
            "Continuous, fine-tuned",
            "#009E73",
            "-",
        ),
    )
    curves = []
    summaries = []
    saved_labels = []
    reference_labels = None
    for index, selection in enumerate(selections):
        curve, summary, labels = read_saved_curve(*selection)
        if reference_labels is None:
            reference_labels = labels
        elif not (index == 1 and args.q1_direct_results is not None) and not np.array_equal(
            reference_labels, labels
        ):
            raise ValueError("Saved test label sequences differ; use matched evaluations")
        curves.append(curve)
        summaries.append(summary)
        saved_labels.append(labels)

    metadata = {"source_results": [summary["source_results"] for summary in summaries]}
    if args.q1_direct_results is not None:
        metadata.update(
            validate_direct_replacement(
                args.q1_direct_results,
                curves[1],
                summaries[1],
                saved_labels[1],
                reference_labels,
                summaries[0],
                args.q1_prepared_dir,
            )
        )
        print("PASS: Q1-direct ROC/AUC, original Q1 manifest and full test class counts verified")
        print("Direct row order is independent; older scratch/continuous caches have no event IDs.")

    output.mkdir(parents=True, exist_ok=True)
    write_plots(output, curves)
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (output / "comparison_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    direct_caption = (
        "The fine-tuned tokenized curve is replaced by the Q1 direct-256 classifier "
        "(256-dimensional token embedding, identity object projection), using its exported test ROC/AUC. "
        "The scratch and continuous curves are unchanged; the tokenized scratch/fine-tuned pair "
        "is therefore not an architecture-matched pretraining ablation. "
        "The direct Q1 dataset manifest and complete "
        "test class counts are checked; the older baseline caches contain no event identities, so "
        "cross-model event-identity pairing is not independently verified. "
        if args.q1_direct_results is not None
        else "Ordered binary test labels are checked for agreement; this check alone does not prove "
        "event-identity matching. "
    )
    (output / "caption.md").write_text(
        "# HZZ classifier comparison\n\n"
        "Tokenized inputs use the single-codebook VQ-VAE representation (Q1). "
        "Legend labels describe event-classifier training: from scratch or pretrained then fine-tuned; "
        "they do not mean that the tokenizer was fine-tuned. Continuous inputs are shown for the "
        "pretrained, fine-tuned event classifier. AUC and ROC arrays are read from the existing "
        "saved evaluations without new inference or changes to their numerical values. "
        "Background rejection is 1/background efficiency, with the existing finite-sample floor. "
        + direct_caption
        + "Source result directories and checkpoints are recorded in summary.json; "
        "the direct-export audit, when supplied, is recorded in comparison_metadata.json.\n"
    )
    for summary in summaries:
        print(f"{summary['run']:<25} AUC={summary['auc']:.4f}")
    print(f"Plots: {output} (saved results only; no inference)")


if __name__ == "__main__":
    main()
