#!/usr/bin/env python3
"""Evaluate VQ, raw, hierarchical, and decoded-Q8 event representations."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from heptokens.data.benchmark_sequence import sequence_batch_to_device
from heptokens.data.benchmark_token_parquet import (
    GroupedTokenParquetClassificationModule,
)
from heptokens.models.foundation_continuous import LitContinuousCLSClassifier
from heptokens.models.foundation_grouped_cls_classifier import LitGroupedCLSClassifier


@dataclass(frozen=True)
class RunSpec:
    name: str
    representation: str
    checkpoint: Path


def parse_run(value: str) -> RunSpec:
    try:
        name, representation, checkpoint = value.split("=", 2)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Run must be NAME=REPRESENTATION=CHECKPOINT"
        ) from exc
    if representation not in {"vq", "flat", "hierarchical", "decoded_q8"}:
        raise argparse.ArgumentTypeError(f"Unknown representation {representation!r}")
    path = Path(checkpoint)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Checkpoint does not exist: {path}")
    return RunSpec(name=name, representation=representation, checkpoint=path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model(spec: RunSpec, device: torch.device):
    model_class = (
        LitGroupedCLSClassifier
        if spec.representation == "vq"
        else LitContinuousCLSClassifier
    )
    model = model_class.load_from_checkpoint(
        spec.checkpoint,
        map_location="cpu",
        backbone_ckpt_path=None,
    )
    model.eval().to(device)
    return model


@torch.inference_mode()
def predict(model, dataloader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    scores = []
    labels = []
    for batch in dataloader:
        batch = sequence_batch_to_device(batch, str(device))
        prediction = model.predict_step(batch)
        scores.append(prediction["score"].detach().cpu().numpy())
        labels.append(prediction["label"].detach().cpu().numpy())
    return np.concatenate(scores), np.concatenate(labels)


def rejection_at_efficiency(
    labels: np.ndarray, scores: np.ndarray, efficiency: float
) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    false_positive_rate = float(np.interp(efficiency, tpr, fpr))
    return float("inf") if false_positive_rate <= 0 else 1.0 / false_positive_rate


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    reference_labels = None
    rows = []
    curves = []

    for run in args.run:
        require_continuous = run.representation != "vq"
        feature_column = (
            "decoded_continuous_features"
            if run.representation == "decoded_q8"
            else "continuous_features"
        )
        datamodule = GroupedTokenParquetClassificationModule(
            prepared_dir=args.prepared_dir,
            n_classes=2,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_sequences=args.max_sequences,
            label_column="label",
            pin_memory=True,
            persistent_workers=False,
            require_continuous=require_continuous,
            continuous_feature_column=(feature_column if require_continuous else None),
        )
        model = load_model(run, device)
        scores, labels = predict(model, datamodule.test_dataloader(), device)
        if reference_labels is None:
            reference_labels = labels
        elif not np.array_equal(labels, reference_labels):
            raise RuntimeError(f"Test labels/order differ for run {run.name}")
        fpr, tpr, _ = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)
        row = {
            "run": run.name,
            "representation": run.representation,
            "events": len(labels),
            "auc": auc,
            "rejection_at_50pct": rejection_at_efficiency(labels, scores, 0.5),
            "rejection_at_70pct": rejection_at_efficiency(labels, scores, 0.7),
            "rejection_at_80pct": rejection_at_efficiency(labels, scores, 0.8),
            "checkpoint": str(run.checkpoint.resolve()),
        }
        rows.append(row)
        curves.append((run.name, auc, fpr, tpr))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axis = plt.subplots(figsize=(8, 7))
    for name, auc, fpr, tpr in curves:
        valid = fpr > 0
        axis.plot(tpr[valid], 1.0 / fpr[valid], label=f"{name} (AUC={auc:.3f})")
    axis.set_yscale("log")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(1.0, None)
    axis.set_xlabel("Signal efficiency")
    axis.set_ylabel("Background rejection")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(loc="best", fontsize=9)
    figure.tight_layout()
    figure.savefig(output_dir / "roc_comparison.png", dpi=200)
    plt.close(figure)

    for row in rows:
        print(
            f"{row['run']:<32} AUC={row['auc']:.4f} "
            f"R50={row['rejection_at_50pct']:.2f} "
            f"R70={row['rejection_at_70pct']:.2f} "
            f"R80={row['rejection_at_80pct']:.2f}"
        )
    print(f"Wrote evaluation to {output_dir}")


if __name__ == "__main__":
    main()
