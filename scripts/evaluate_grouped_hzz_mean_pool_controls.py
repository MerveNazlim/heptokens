#!/usr/bin/env python3
"""Evaluate grouped HZZ mean-pooling controls on the held-out test split."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from heptokens.data.sequence import LABELS_KEY
from heptokens.data.token_parquet import GroupedTokenParquetClassificationModule
from heptokens.models.foundation_grouped_mean_classifier import (
    LitGroupedMaskedMeanClassifier,
)


RUNS = {
    "pretrained_frozen_mean_pool": "Pretrained, frozen",
    "random_frozen_mean_pool": "Random, frozen",
    "pretrained_finetuned_mean_pool": "Pretrained, fine-tuned",
    "random_finetuned_mean_pool": "Random, from scratch",
}
TARGET_SIGNAL_EFFICIENCIES = (0.50, 0.70, 0.80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(
            "results/atlas_hzz_grouped_mean_pool_classification"
        ),
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=Path(
            "results/event_tokens_grouped_final_new_mcdata/"
            "hzz_ggf_vs_zz_classification_shards"
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--stream-batch-size", type=int, default=4096)
    parser.add_argument("--shuffle-buffer-size", type=int, default=8192)
    return parser.parse_args()


def evaluate_checkpoint(
    checkpoint: Path,
    dataloader,
    device: torch.device,
) -> tuple[dict, tuple[np.ndarray, np.ndarray]]:
    model = LitGroupedMaskedMeanClassifier.load_from_checkpoint(
        checkpoint,
        map_location="cpu",
    )
    model.to(device)
    model.eval()

    scores = []
    labels = []
    loss_sum = 0.0
    event_count = 0
    with torch.inference_mode():
        for batch in dataloader:
            batch = {
                key: value.to(device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }
            logits = model(batch)
            targets = batch[LABELS_KEY].long()
            loss_sum += F.cross_entropy(logits, targets, reduction="sum").item()
            event_count += targets.numel()
            scores.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
            labels.append(targets.cpu())

    score = torch.cat(scores).numpy()
    label = torch.cat(labels).numpy()
    fpr, tpr, thresholds = roc_curve(label, score)
    n_background = int(np.count_nonzero(label == 0))
    minimum_fpr = 1.0 / max(n_background, 1)

    result = {
        "checkpoint": str(checkpoint.resolve()),
        "events": int(event_count),
        "signal_events": int(np.count_nonzero(label == 1)),
        "background_events": n_background,
        "test_loss": loss_sum / event_count,
        "test_accuracy": accuracy_score(label, score >= 0.5),
        "test_auc": roc_auc_score(label, score),
    }
    for target in TARGET_SIGNAL_EFFICIENCIES:
        index = min(int(np.searchsorted(tpr, target, side="left")), len(tpr) - 1)
        suffix = int(target * 100)
        result[f"background_efficiency_at_signal_efficiency_{suffix}"] = float(
            fpr[index]
        )
        result[f"background_rejection_at_signal_efficiency_{suffix}"] = float(
            1.0 / max(fpr[index], minimum_fpr)
        )
        result[f"threshold_at_signal_efficiency_{suffix}"] = float(thresholds[index])
    return result, (tpr, fpr)


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = [
        "run",
        "label",
        "events",
        "signal_events",
        "background_events",
        "test_loss",
        "test_accuracy",
        "test_auc",
    ]
    for target in TARGET_SIGNAL_EFFICIENCIES:
        suffix = int(target * 100)
        columns.extend(
            [
                f"background_efficiency_at_signal_efficiency_{suffix}",
                f"background_rejection_at_signal_efficiency_{suffix}",
                f"threshold_at_signal_efficiency_{suffix}",
            ]
        )
    columns.append("checkpoint")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def plot_roc(
    path: Path,
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    rows: list[dict],
) -> None:
    labels = {row["run"]: row for row in rows}
    fig, ax = plt.subplots(figsize=(8, 6))
    for run_name, (tpr, fpr) in curves.items():
        row = labels[run_name]
        ax.plot(
            tpr,
            fpr,
            linewidth=2,
            label=f"{row['label']} (AUC={row['test_auc']:.3f})",
        )
    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Background efficiency")
    ax.set_yscale("log")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1e-5, 1.0)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def print_summary(rows: list[dict]) -> None:
    header = (
        f"{'run':32s} {'loss':>8s} {'acc':>8s} {'AUC':>8s} "
        f"{'R@50%':>10s} {'R@70%':>10s} {'R@80%':>10s}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['run']:32s} "
            f"{row['test_loss']:8.4f} "
            f"{row['test_accuracy']:8.4f} "
            f"{row['test_auc']:8.4f} "
            f"{row['background_rejection_at_signal_efficiency_50']:10.2f} "
            f"{row['background_rejection_at_signal_efficiency_70']:10.2f} "
            f"{row['background_rejection_at_signal_efficiency_80']:10.2f}"
        )


def resolve_checkpoint(run_dir: Path) -> Path:
    # Prefer validation-selected weights; older completed controls only saved epoch 10.
    best = run_dir / "checkpoints" / "best.ckpt"
    if best.is_file():
        return best
    last = run_dir / "checkpoints" / "last.ckpt"
    if last.is_file():
        print(f"WARNING: {best} is missing; evaluating final-epoch {last}", flush=True)
        return last
    raise FileNotFoundError(f"No best.ckpt or last.ckpt found under {run_dir}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Invalid batch size or worker count")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    output_dir = args.output_dir or args.project_dir / "test_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    datamodule = GroupedTokenParquetClassificationModule(
        prepared_dir=str(args.prepared_dir),
        n_classes=2,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        stream_batch_size=args.stream_batch_size,
        shuffle_buffer_size=args.shuffle_buffer_size,
        label_column="label",
    )

    rows = []
    curves = {}
    for run_name, display_label in RUNS.items():
        checkpoint = resolve_checkpoint(args.project_dir / run_name)
        print(f"Evaluating {run_name}: {checkpoint}", flush=True)
        result, curve = evaluate_checkpoint(
            checkpoint,
            datamodule.test_dataloader(),
            device,
        )
        result["run"] = run_name
        result["label"] = display_label
        rows.append(result)
        curves[run_name] = curve

    write_csv(output_dir / "test_metrics.csv", rows)
    (output_dir / "test_metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    plot_roc(output_dir / "test_roc.png", curves, rows)
    print_summary(rows)
    print(f"\nWrote test evaluation to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
