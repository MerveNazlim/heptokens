#!/usr/bin/env python3
"""Compare four downstream training setups for one event representation."""

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
from sklearn.metrics import roc_auc_score, roc_curve

import evaluate_q1_q8_continuous_hzz as common
from heptokens.data.benchmark_token_parquet import (
    GroupedTokenParquetClassificationModule,
)


SETUPS = (
    ("Pretrained, frozen", "pretrained_frozen_checkpoint", "#0072B2"),
    ("Random, frozen", "random_frozen_checkpoint", "#E69F00"),
    ("Pretrained, fine-tuned", "pretrained_finetuned_checkpoint", "#D55E00"),
    ("Random, from scratch", "random_finetuned_checkpoint", "#CC79A7"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", choices=("q1", "q8", "continuous"), required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--pretrained-frozen-checkpoint", type=Path, required=True)
    parser.add_argument("--random-frozen-checkpoint", type=Path, required=True)
    parser.add_argument("--pretrained-finetuned-checkpoint", type=Path, required=True)
    parser.add_argument("--random-finetuned-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--stream-batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def write_plots(output_dir: Path, title: str, curves: list[tuple]) -> None:
    for rejection_plot in (False, True):
        figure, axis = plt.subplots(figsize=(8, 6))
        for name, auc, fpr, tpr, background_events, color in curves:
            y = (
                1.0 / np.maximum(fpr, 1.0 / max(background_events, 1))
                if rejection_plot
                else fpr
            )
            axis.plot(tpr, y, linewidth=2.2, color=color, label=f"{name} (AUC={auc:.3f})")
        axis.set_yscale("log")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(1.0, None) if rejection_plot else axis.set_ylim(1e-5, 1.0)
        axis.set_xlabel("Signal efficiency")
        axis.set_ylabel("Background rejection" if rejection_plot else "Background efficiency")
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(loc="best")
        figure.tight_layout()
        stem = "background_rejection_comparison" if rejection_plot else "roc_comparison"
        figure.savefig(output_dir / f"{stem}.png", dpi=200)
        figure.savefig(output_dir / f"{stem}.pdf")
        plt.close(figure)


def main() -> None:
    args = parse_args()
    if not (args.prepared_dir / "manifest.json").is_file():
        raise FileNotFoundError(f"Missing manifest under {args.prepared_dir}")

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    curves = []
    arrays = {}
    reference_labels = None

    for name, argument_name, color in SETUPS:
        checkpoint = getattr(args, argument_name)
        spec = common.RunSpec(name, args.representation, checkpoint, args.prepared_dir)
        common.validate_paths(spec)
        require_continuous = args.representation == "continuous"
        datamodule = GroupedTokenParquetClassificationModule(
            prepared_dir=str(args.prepared_dir),
            n_classes=2,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            stream_batch_size=args.stream_batch_size,
            shuffle_buffer_size=8192,
            label_column="label",
            pin_memory=device.type == "cuda",
            persistent_workers=False,
            require_continuous=require_continuous,
            continuous_feature_column=("continuous_features" if require_continuous else None),
        )
        model = common.load_model(spec, device)
        scores, labels = common.predict(model, datamodule.test_dataloader(), device)
        if reference_labels is None:
            reference_labels = labels
        elif not np.array_equal(reference_labels, labels):
            raise RuntimeError(f"Test labels changed while evaluating {name}")

        fpr, tpr, thresholds = roc_curve(labels, scores)
        auc = float(roc_auc_score(labels, scores))
        background_events = int(np.count_nonzero(labels == 0))
        rows.append(
            {
                "setup": name,
                "representation": args.representation,
                "events": int(len(labels)),
                "auc": auc,
                "rejection_at_50pct": common.rejection_at_efficiency(labels, scores, 0.50),
                "rejection_at_70pct": common.rejection_at_efficiency(labels, scores, 0.70),
                "rejection_at_80pct": common.rejection_at_efficiency(labels, scores, 0.80),
                "checkpoint": str(checkpoint.resolve()),
            }
        )
        curves.append((name, auc, fpr, tpr, background_events, color))
        prefix = common.safe_name(name)
        arrays[f"{prefix}_fpr"] = fpr
        arrays[f"{prefix}_tpr"] = tpr
        arrays[f"{prefix}_thresholds"] = thresholds
        arrays[f"{prefix}_scores"] = scores
        arrays[f"{prefix}_labels"] = labels
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    np.savez_compressed(args.output_dir / "roc_arrays.npz", **arrays)
    write_plots(args.output_dir, args.title, curves)

    print("\nsetup                         AUC      R@50%    R@70%    R@80%")
    print("-" * 70)
    for row in rows:
        print(
            f"{row['setup']:<29} {row['auc']:.4f}   "
            f"{row['rejection_at_50pct']:8.2f} "
            f"{row['rejection_at_70pct']:8.2f} "
            f"{row['rejection_at_80pct']:8.2f}"
        )
    print(f"\nWrote comparison to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
