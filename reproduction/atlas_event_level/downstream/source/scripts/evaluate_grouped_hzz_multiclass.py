#!/usr/bin/env python3
"""Evaluate the grouped-CLS HZZ multiclass classifier on the test split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from heptokens.data.sequence import sequence_batch_to_device
from heptokens.data.token_parquet import GroupedTokenParquetClassificationModule
from heptokens.models.foundation_grouped_cls_classifier import LitGroupedCLSClassifier


DISPLAY_NAMES = {
    "ggf": "ggF",
    "vbf": "VBF",
    "vh": "VH",
    "zh": "ZH",
    "ggzh": "ggZH",
    "wh": "WH",
    "tth": r"$t\bar{t}H + tH$",
    "th": "tH",
    "zz_continuum": r"$ZZ^{(*)}$ continuum",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def load_manifest(prepared_dir: Path) -> tuple[dict, list[str], list[str]]:
    manifest_path = prepared_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing multiclass manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    classes = sorted(manifest["classes"], key=lambda item: int(item["label"]))
    labels = [int(item["label"]) for item in classes]
    if labels != list(range(len(classes))):
        raise ValueError(f"Class labels must be contiguous from zero, found {labels}")
    class_names = [str(item["name"]) for item in classes]
    display_names = [DISPLAY_NAMES.get(name, name) for name in class_names]
    return manifest, class_names, display_names


@torch.inference_mode()
def predict(
    model: LitGroupedCLSClassifier,
    dataloader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = []
    labels = []
    for batch_index, batch in enumerate(dataloader, start=1):
        batch = sequence_batch_to_device(batch, str(device))
        prediction = model.predict_step(batch)
        probabilities.append(prediction["probabilities"].detach().cpu().numpy())
        labels.append(prediction["label"].detach().cpu().numpy())
        if batch_index % 100 == 0:
            print(f"evaluated {batch_index:,} test batches", flush=True)
    return np.concatenate(probabilities), np.concatenate(labels)


def plot_confusion_matrix(
    matrix: np.ndarray,
    display_names: list[str],
    output_dir: Path,
) -> None:
    side = max(8.0, 0.9 * len(display_names) + 2.0)
    figure, axis = plt.subplots(figsize=(side, side - 0.5))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Fraction of true class")
    ticks = np.arange(len(display_names))
    axis.set_xticks(ticks, labels=display_names, rotation=25, ha="right")
    axis.set_yticks(ticks, labels=display_names)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Normalized confusion matrix")

    threshold = 0.5 * (float(matrix.max()) + float(matrix.min()))
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{100.0 * value:.1f}%",
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=9 if len(display_names) > 5 else 11,
            )
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"normalized_confusion_matrix.{suffix}", dpi=240)
    plt.close(figure)


def plot_roc_curves(
    labels: np.ndarray,
    probabilities: np.ndarray,
    display_names: list[str],
    output_dir: Path,
) -> dict[str, float]:
    figure, axis = plt.subplots(figsize=(8.2, 7.0))
    colors = plt.get_cmap("tab10").colors
    per_class_auc = {}
    for class_index, display_name in enumerate(display_names):
        color = colors[class_index % len(colors)]
        binary_labels = (labels == class_index).astype(np.int8)
        false_positive_rate, true_positive_rate, _ = roc_curve(
            binary_labels, probabilities[:, class_index]
        )
        class_auc = float(auc(false_positive_rate, true_positive_rate))
        per_class_auc[display_name] = class_auc
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            color=color,
            linewidth=2.2,
            label=f"{display_name} vs rest (AUC = {class_auc:.3f})",
        )

    axis.plot([0, 1], [0, 1], linestyle="--", color="0.35", linewidth=1.5)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("One-vs-rest ROC curves")
    axis.grid(True, alpha=0.2)
    axis.legend(loc="lower right", frameon=True, fontsize=10)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"multiclass_roc.{suffix}", dpi=240)
    plt.close(figure)
    return per_class_auc


def main() -> None:
    args = parse_args()
    prepared_dir = Path(args.prepared_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing classifier checkpoint: {checkpoint}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest, class_names, display_names = load_manifest(prepared_dir)
    n_classes = len(class_names)
    device = torch.device(args.device)
    datamodule = GroupedTokenParquetClassificationModule(
        prepared_dir=str(prepared_dir),
        n_classes=n_classes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        stream_batch_size=4096,
        shuffle_buffer_size=8192,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        label_column="label",
    )
    model = LitGroupedCLSClassifier.load_from_checkpoint(
        checkpoint,
        map_location="cpu",
        backbone_ckpt_path=None,
    )
    if model.n_classes != n_classes:
        raise ValueError(
            f"Checkpoint has {model.n_classes} classes, manifest has {n_classes}"
        )
    model.eval().to(device)

    probabilities, labels = predict(model, datamodule.test_dataloader(), device)
    expected_test_events = int(manifest["split_counts"]["test"]["total"])
    if labels.size != expected_test_events:
        raise RuntimeError(
            f"Evaluated {labels.size} events, expected {expected_test_events}"
        )
    if probabilities.shape != (labels.size, n_classes):
        raise RuntimeError(f"Unexpected probability shape: {probabilities.shape}")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError("Softmax probabilities do not sum to one")

    predictions = probabilities.argmax(axis=1)
    raw_confusion = confusion_matrix(labels, predictions, labels=range(n_classes))
    normalized_confusion = confusion_matrix(
        labels, predictions, labels=range(n_classes), normalize="true"
    )
    plot_confusion_matrix(normalized_confusion, display_names, output_dir)
    per_class_auc = plot_roc_curves(
        labels, probabilities, display_names, output_dir
    )

    report = classification_report(
        labels,
        predictions,
        labels=range(n_classes),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    summary = {
        "checkpoint": str(checkpoint),
        "prepared_dir": str(prepared_dir),
        "test_events": int(labels.size),
        "class_names": class_names,
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_ovr_auc": float(
            roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")
        ),
        "weighted_ovr_auc": float(
            roc_auc_score(labels, probabilities, multi_class="ovr", average="weighted")
        ),
        "per_class_auc": {
            class_names[index]: per_class_auc[display_names[index]]
            for index in range(n_classes)
        },
        "raw_confusion_matrix": raw_confusion.tolist(),
        "normalized_confusion_matrix": normalized_confusion.tolist(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "classification_report.csv").open("w", newline="") as handle:
        fieldnames = ["class", "precision", "recall", "f1-score", "support", "auc"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for class_index, class_name in enumerate(class_names):
            writer.writerow(
                {
                    "class": class_name,
                    **report[class_name],
                    "auc": summary["per_class_auc"][class_name],
                }
            )
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        labels=labels,
        predictions=predictions,
        probabilities=probabilities,
        class_names=np.asarray(class_names),
    )

    print(json.dumps(summary, indent=2))
    print(f"Wrote multiclass evaluation to {output_dir}")


if __name__ == "__main__":
    main()
