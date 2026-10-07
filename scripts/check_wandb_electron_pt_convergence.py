#!/usr/bin/env python3
"""Compare late-epoch electron-pT convergence for MC-only and MC+data runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
import wandb

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_RUNS = {
    "MC-only": (
        "magaras-brookhaven-national-laboratory/"
        "atlas_event_tokenizers_1806_nonjet_logstd_capacity_scan/nyxefilg"
    ),
    "MC+data": (
        "magaras-brookhaven-national-laboratory/"
        "atlas_event_tokenizers_0107_logstd_mc_realdata/8fecb5wp"
    ),
}

METRICS = [
    "train/feature_mae/pt",
    "val/feature_mae/pt",
    "train/feature_rmse/pt",
    "val/feature_rmse/pt",
]

COLORS = {"MC-only": "#4C83F1", "MC+data": "#FF9F1C"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=ENTITY/PROJECT/RUN_ID",
        help="Override or add a W&B run; repeatable.",
    )
    parser.add_argument("--last-n", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        default="results/wandb_electron_pt_convergence",
    )
    return parser.parse_args()


def selected_runs(overrides: list[str]) -> dict[str, str]:
    runs = dict(DEFAULT_RUNS)
    for value in overrides:
        if "=" not in value:
            raise ValueError(f"Expected LABEL=ENTITY/PROJECT/RUN_ID, got {value!r}")
        label, path = value.split("=", 1)
        runs[label.strip()] = path.strip()
    return runs


def fetch_metric(run, metric: str) -> tuple[np.ndarray, np.ndarray]:
    points: dict[float, float] = {}
    rows = run.scan_history(keys=["_step", "epoch", metric], page_size=1000)
    fallback_index = 0
    for row in rows:
        value = row.get(metric)
        if value is None or not np.isfinite(value):
            continue
        epoch = row.get("epoch")
        if epoch is None or not np.isfinite(epoch):
            epoch = float(fallback_index)
        points[float(epoch)] = float(value)
        fallback_index += 1
    if not points:
        return np.asarray([]), np.asarray([])
    x = np.asarray(sorted(points), dtype=np.float64)
    y = np.asarray([points[value] for value in x], dtype=np.float64)
    return x, y


def trend_summary(x: np.ndarray, y: np.ndarray, last_n: int) -> dict:
    if len(y) == 0:
        return {
            "n_points": 0,
            "first": np.nan,
            "last": np.nan,
            "relative_change_percent": np.nan,
            "fitted_change_percent": np.nan,
            "assessment": "missing",
        }
    n = min(last_n, len(y))
    tail_x = x[-n:]
    tail_y = y[-n:]
    relative_change = (
        100.0 * (tail_y[-1] - tail_y[0]) / abs(tail_y[0])
        if tail_y[0] != 0
        else np.nan
    )
    if n >= 2 and np.ptp(tail_x) > 0:
        slope, intercept = np.polyfit(tail_x, tail_y, 1)
        fitted_first = slope * tail_x[0] + intercept
        fitted_last = slope * tail_x[-1] + intercept
        fitted_change = (
            100.0 * (fitted_last - fitted_first) / abs(fitted_first)
            if fitted_first != 0
            else np.nan
        )
    else:
        fitted_change = np.nan

    if not np.isfinite(fitted_change):
        assessment = "insufficient history"
    elif fitted_change < -1.0:
        assessment = "still decreasing"
    elif fitted_change > 1.0:
        assessment = "increasing"
    else:
        assessment = "approximately flat"
    return {
        "n_points": int(len(y)),
        "first": float(tail_y[0]),
        "last": float(tail_y[-1]),
        "relative_change_percent": float(relative_change),
        "fitted_change_percent": float(fitted_change),
        "assessment": assessment,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api(timeout=60)
    runs = selected_runs(args.run)
    histories: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    summary_rows = []

    for label, path in runs.items():
        run = api.run(path)
        print(f"\n## {label}\nW&B: {path}\nURL: {run.url}")
        for metric in METRICS:
            x, y = fetch_metric(run, metric)
            histories[(label, metric)] = (x, y)
            summary = trend_summary(x, y, args.last_n)
            summary_rows.append({"run": label, "metric": metric, **summary})
            print(f"\n{metric}")
            if len(y) == 0:
                print("  missing")
                continue
            for epoch, value in zip(x[-args.last_n :], y[-args.last_n :]):
                print(f"  epoch={epoch:g} value={value:.8g}")
            print(
                "  fitted last-window change="
                f"{summary['fitted_change_percent']:+.3f}% "
                f"({summary['assessment']})"
            )

    with (output_dir / "convergence_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    for ax, metric_name in zip(axes, ["mae", "rmse"]):
        for label in runs:
            for split, linestyle in [("train", "-"), ("val", "--")]:
                metric = f"{split}/feature_{metric_name}/pt"
                x, y = histories[(label, metric)]
                if len(y) == 0:
                    continue
                ax.plot(
                    x,
                    y,
                    color=COLORS.get(label),
                    linestyle=linestyle,
                    linewidth=2.0,
                    marker="o",
                    markersize=3.5,
                    label=f"{label} {split}",
                )
        ax.set_xlabel("epoch")
        ax.set_ylabel(f"standardized pT {metric_name.upper()}")
        ax.set_title(f"Electron pT {metric_name.upper()} convergence")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "electron_pt_train_val_convergence.png", dpi=180)
    plt.close(fig)

    print(f"\nWrote {output_dir / 'convergence_summary.csv'}")
    print(f"Wrote {output_dir / 'electron_pt_train_val_convergence.png'}")


if __name__ == "__main__":
    main()
