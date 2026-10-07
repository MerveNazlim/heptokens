#!/usr/bin/env python3
"""Compare epoch metrics from two offline W&B run files without uploading them."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


METRICS = (
    "train/total_loss",
    "train/recon_loss",
    "train/commit_loss",
    "val/total_loss",
    "val/recon_loss",
    "val/commit_loss",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True, help="Left run-*.wandb file")
    parser.add_argument("--right", type=Path, required=True, help="Right run-*.wandb file")
    parser.add_argument("--left-label", default="Google/Condor")
    parser.add_argument("--right-label", default="Zephyr")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def item_key(item) -> str:
    nested = list(getattr(item, "nested_key", []))
    return ".".join(nested) if nested else item.key


def history_rows(path: Path) -> tuple[list[dict], set[str]]:
    store = DataStore()
    store.open_for_scan(str(path))
    rows: list[dict] = []
    keys: set[str] = set()
    while True:
        data = store.scan_data()
        if data is None:
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if not record.HasField("history"):
            continue
        row = {}
        for item in record.history.item:
            key = item_key(item)
            if not key:
                continue
            try:
                value = json.loads(item.value_json)
            except (TypeError, json.JSONDecodeError):
                continue
            row[key] = value
            keys.add(key)
        if row:
            rows.append(row)
    return rows, keys


def epoch_summary(rows: list[dict]) -> dict[int, dict[str, float]]:
    values: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        epoch_value = row.get("epoch")
        if epoch_value is None:
            continue
        try:
            epoch = int(float(epoch_value))
        except (TypeError, ValueError):
            continue
        for metric in METRICS:
            value = row.get(metric)
            if isinstance(value, (int, float)) and np.isfinite(value):
                values[epoch][metric].append(float(value))

    summary = {}
    for epoch, epoch_values in sorted(values.items()):
        summary[epoch] = {}
        for metric, samples in epoch_values.items():
            # Training metrics are step-level values sampled by the logger, so
            # compare their within-epoch mean. Validation values are epoch-level;
            # use the last emitted value in case W&B stored it more than once.
            summary[epoch][metric] = (
                float(np.mean(samples)) if metric.startswith("train/") else samples[-1]
            )
            summary[epoch][f"{metric}__n"] = len(samples)
    return summary


def write_csv(path: Path, summaries: dict[str, dict[int, dict[str, float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["run", "epoch"] + list(METRICS) + [f"{m}__n" for m in METRICS]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for label, summary in summaries.items():
            for epoch, metrics in sorted(summary.items()):
                writer.writerow({"run": label, "epoch": epoch, **metrics})


def plot(path: Path, summaries: dict[str, dict[int, dict[str, float]]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, stem in zip(axes, ("total_loss", "recon_loss", "commit_loss")):
        for label, summary in summaries.items():
            for split, linestyle in (("train", "-"), ("val", "--")):
                metric = f"{split}/{stem}"
                points = [
                    (epoch + 1, values[metric])
                    for epoch, values in sorted(summary.items())
                    if metric in values
                ]
                if points:
                    x, y = zip(*points)
                    ax.plot(x, y, marker="o", linestyle=linestyle, label=f"{label} {split}")
        ax.set_title(stem.replace("_", " "))
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("logged loss")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for label, path in ((args.left_label, args.left), (args.right_label, args.right)):
        rows, keys = history_rows(path)
        summary = epoch_summary(rows)
        summaries[label] = summary
        print(f"{label}: {len(rows)} history rows; epochs={sorted(summary)}")
        present = sorted(key for key in keys if key in METRICS or key == "epoch")
        print(f"{label}: relevant keys={present}")
    write_csv(args.output_dir / "epoch_metrics.csv", summaries)
    plot(args.output_dir / "epoch_metrics.png", summaries)
    print(f"Wrote {args.output_dir / 'epoch_metrics.csv'}")
    print(f"Wrote {args.output_dir / 'epoch_metrics.png'}")


if __name__ == "__main__":
    main()
