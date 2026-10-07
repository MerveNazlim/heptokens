#!/usr/bin/env python3
"""Summarize and plot a grouped foundation-pretraining run from W&B history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml


METRICS = (
    "train/total_loss",
    "train/mask_acc",
    "valid/total_loss",
    "valid/mask_acc",
    "lr-AdamW",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch a synced W&B run, aggregate noisy training metrics by epoch, "
            "and inspect its final checkpoint."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--wandb-run-path",
        help="Explicit entity/project/run_id. Inferred when omitted.",
    )
    parser.add_argument("--entity", help="W&B entity used when inferring the run path")
    parser.add_argument("--project", help="Override the project stored in Hydra config")
    parser.add_argument("--run-id", help="Override the run ID inferred from wandb/")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rolling-window", type=int, default=200)
    return parser.parse_args()


def load_config(run_dir: Path) -> dict:
    candidates = (run_dir / ".hydra" / "config.yaml", run_dir / "full_config.yaml")
    for path in candidates:
        if path.is_file():
            with path.open() as handle:
                return yaml.safe_load(handle) or {}
    raise FileNotFoundError(f"No Hydra/full configuration found under {run_dir}")


def infer_run_id(run_dir: Path) -> str:
    candidates = sorted((run_dir / "wandb").glob("*run-*-*"))
    if not candidates:
        raise FileNotFoundError(f"No W&B run directory found under {run_dir / 'wandb'}")
    return candidates[-1].name.rsplit("-", 1)[-1]


def checkpoint_summary(run_dir: Path) -> dict:
    checkpoints = sorted((run_dir / "checkpoints").glob("*.ckpt"))
    if not checkpoints:
        return {"checkpoint": None}
    checkpoint_path = checkpoints[-1]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", {})
    return {
        "checkpoint": str(checkpoint_path),
        "epoch": int(checkpoint.get("epoch", -1)),
        "global_step": int(checkpoint.get("global_step", -1)),
        "state_tensors": len(state_dict),
        "parameters": int(
            sum(value.numel() for value in state_dict.values() if torch.is_tensor(value))
        ),
        "size_mib": checkpoint_path.stat().st_size / 2**20,
    }


def fetch_history(run_path: str) -> tuple[pd.DataFrame, dict]:
    import wandb

    run = wandb.Api(timeout=120).run(run_path)
    rows = []
    wanted = {"_step", "epoch", *METRICS}
    for row in run.scan_history(page_size=10_000):
        selected = {key: row.get(key) for key in wanted}
        if any(selected.get(metric) is not None for metric in METRICS):
            rows.append(selected)
    if not rows:
        raise RuntimeError(f"No requested metrics found in W&B run {run_path}")
    metadata = {
        "wandb_run_path": run_path,
        "name": run.name,
        "state": run.state,
        "url": run.url,
    }
    return pd.DataFrame(rows), metadata


def clean_history(history: pd.DataFrame) -> pd.DataFrame:
    for column in ("_step", "epoch", *METRICS):
        if column in history:
            history[column] = pd.to_numeric(history[column], errors="coerce")
    history = history.sort_values("_step").reset_index(drop=True)
    if "epoch" not in history or history["epoch"].isna().all():
        raise RuntimeError("W&B history does not contain usable epoch values")
    history["epoch_index"] = np.floor(history["epoch"]).astype("Int64")
    return history


def aggregate_epochs(history: pd.DataFrame) -> pd.DataFrame:
    records = []
    for epoch, frame in history.dropna(subset=["epoch_index"]).groupby("epoch_index"):
        record: dict[str, float | int] = {"epoch": int(epoch)}
        for metric in METRICS:
            values = frame[metric].dropna() if metric in frame else pd.Series(dtype=float)
            if values.empty:
                continue
            if metric.startswith("train/"):
                record[f"{metric}_mean"] = float(values.mean())
                record[f"{metric}_median"] = float(values.median())
                record[f"{metric}_last"] = float(values.iloc[-1])
                record[f"{metric}_points"] = int(len(values))
            else:
                record[metric] = float(values.iloc[-1])
        records.append(record)
    return pd.DataFrame(records).sort_values("epoch").reset_index(drop=True)


def plot_history(history: pd.DataFrame, epochs: pd.DataFrame, output_dir: Path, window: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    specs = (
        ("train/total_loss", "Training loss", axes[0, 0]),
        ("train/mask_acc", "Training masked-code accuracy", axes[0, 1]),
        ("valid/total_loss", "Validation loss", axes[1, 0]),
        ("valid/mask_acc", "Validation masked-code accuracy", axes[1, 1]),
    )
    for metric, title, axis in specs:
        values = history.dropna(subset=[metric]) if metric in history else pd.DataFrame()
        if not values.empty and metric.startswith("train/"):
            axis.plot(values["_step"], values[metric], alpha=0.16, linewidth=0.6, label="batch")
            smoothed = values[metric].rolling(window, min_periods=max(5, window // 10)).mean()
            axis.plot(values["_step"], smoothed, linewidth=1.8, label=f"rolling mean ({window})")
            axis.set_xlabel("W&B step")
            axis.legend(frameon=False)
        elif not epochs.empty and metric in epochs:
            axis.plot(epochs["epoch"] + 1, epochs[metric], marker="o", linewidth=1.8)
            axis.set_xlabel("Epoch")
        else:
            axis.text(0.5, 0.5, "metric not logged", ha="center", va="center")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    fig.suptitle("Grouped foundation pretraining diagnostics", fontsize=16)
    fig.savefig(output_dir / "training_diagnostics.png", dpi=180)
    fig.savefig(output_dir / "training_diagnostics.pdf")
    plt.close(fig)

    if "lr-AdamW" in history and history["lr-AdamW"].notna().any():
        values = history.dropna(subset=["lr-AdamW"])
        fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
        axis.plot(values["_step"], values["lr-AdamW"], linewidth=1.5)
        axis.set(xlabel="W&B step", ylabel="Learning rate", title="Learning-rate schedule")
        axis.grid(alpha=0.2)
        fig.savefig(output_dir / "learning_rate.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / "training_analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(run_dir)
    project = args.project or config.get("project_name")
    run_id = args.run_id or infer_run_id(run_dir)
    if args.wandb_run_path:
        run_path = args.wandb_run_path
    else:
        import wandb

        entity = args.entity or wandb.Api(timeout=120).default_entity
        if not entity or not project:
            raise ValueError("Could not infer W&B entity/project; provide --wandb-run-path")
        run_path = f"{entity}/{project}/{run_id}"

    history, wandb_metadata = fetch_history(run_path)
    history = clean_history(history)
    epochs = aggregate_epochs(history)
    checkpoint = checkpoint_summary(run_dir)

    history.to_csv(output_dir / "history.csv", index=False)
    epochs.to_csv(output_dir / "epoch_summary.csv", index=False)
    summary = {**wandb_metadata, **checkpoint, "epochs_in_history": len(epochs)}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_history(history, epochs, output_dir, args.rolling_window)

    print(json.dumps(summary, indent=2))
    print("\nEpoch summary:")
    print(epochs.to_string(index=False))
    print(f"\nWrote analysis to {output_dir}")


if __name__ == "__main__":
    main()
