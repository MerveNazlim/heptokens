#!/usr/bin/env python3
"""Plot tokenizer capacity versus average feature resolution for selected runs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
import numpy as np
import torch
from omegaconf import OmegaConf

from plot_tokenizer_summary import (
    COLORS,
    binned_response_iqr_over_median,
    feature_index,
    feature_label,
    load_or_collect_arrays,
    safe_filename,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make a slide-friendly capacity/resolution tradeoff plot."
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        metavar="RUN_DIR=LABEL",
        help="Selected run directories and public-facing labels.",
    )
    parser.add_argument("--object", required=True, help="Object name, e.g. electrons or jets.")
    parser.add_argument("--feature", default="pt", help="Feature used for resolution, e.g. pt.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--h5-files", nargs="+")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--max-valid-objects", type=int, default=1_000_000)
    parser.add_argument("--num-events-per-file", type=int)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--capacity-metric",
        choices=["token_vocabulary", "codebook_parameters", "latent_width"],
        default="token_vocabulary",
        help=(
            "X-axis capacity proxy. token_vocabulary = codebook_size x quantizers; "
            "codebook_parameters = codebook_size x quantizers x codebook_dim; "
            "latent_width = quantizers x codebook_dim."
        ),
    )
    parser.add_argument(
        "--plot-style",
        choices=["points", "heatmap"],
        default="points",
        help="Use points for selected working points or heatmap for a scan grid.",
    )
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def parse_runs(values: list[str]) -> list[tuple[Path, str]]:
    parsed = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected RUN_DIR=LABEL, got {value}")
        run_dir, label = value.split("=", 1)
        parsed.append((Path(run_dir).expanduser().resolve(), label.strip()))
    return parsed


def run_capacity(run_dir: Path, metric: str) -> tuple[float, str]:
    cfg = OmegaConf.to_container(OmegaConf.load(run_dir / "full_config.yaml"), resolve=True)
    model_cfg = cfg.get("model") or {}
    codebook_size = int(model_cfg["codebook_size"])
    num_quantizers = int(model_cfg["num_quantizers"])
    codebook_dim = int(model_cfg["codebook_dim"])

    if metric == "token_vocabulary":
        return float(codebook_size * num_quantizers), "Vocabulary capacity per object"
    if metric == "codebook_parameters":
        return float(codebook_size * num_quantizers * codebook_dim), "Codebook capacity"
    if metric == "latent_width":
        return float(num_quantizers * codebook_dim), "Latent width per object"
    raise ValueError(metric)


def run_hparams(run_dir: Path) -> tuple[int, int, int]:
    cfg = OmegaConf.to_container(OmegaConf.load(run_dir / "full_config.yaml"), resolve=True)
    model_cfg = cfg.get("model") or {}
    return (
        int(model_cfg["codebook_dim"]),
        int(model_cfg["codebook_size"]),
        int(model_cfg["num_quantizers"]),
    )


def average_binned_resolution(
    run_dir: Path,
    *,
    feature_name: str,
    output_dir: Path,
    h5_files: list[str] | None,
    split: str,
    max_valid_objects: int,
    num_events_per_file: int | None,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[float, int]:
    original, reconstruction, feature_names = load_or_collect_arrays(
        run_dir,
        output_dir=output_dir,
        h5_files=h5_files,
        split=split,
        max_valid_objects=max_valid_objects,
        num_events_per_file=num_events_per_file,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    idx = feature_index(feature_names, feature_name)
    if idx is None:
        raise ValueError(f"Feature {feature_name!r} not found in {run_dir}")

    truth = original[:, idx]
    finite_truth = truth[np.isfinite(truth)]
    finite_truth = finite_truth[np.abs(finite_truth) > 1e-12]
    if len(finite_truth) == 0:
        raise ValueError(f"No finite nonzero truth values for {feature_name} in {run_dir}")
    lo, hi = np.percentile(finite_truth, [1.0, 99.0])
    bins = np.linspace(lo, hi, 15)
    _, values = binned_response_iqr_over_median(truth, truth, reconstruction[:, idx], bins)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        raise ValueError(f"No finite binned resolution values for {feature_name} in {run_dir}")
    return float(np.mean(finite)), int(len(finite))


def jitter_duplicate_x(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if len(arr) <= 1:
        return arr
    out = arr.copy()
    for value in np.unique(arr):
        idx = np.flatnonzero(arr == value)
        if len(idx) <= 1:
            continue
        scale = max(abs(value) * 0.025, 1.0)
        offsets = np.linspace(-scale, scale, len(idx))
        out[idx] = arr[idx] + offsets
    return out


def plot_heatmap(
    points: list[dict],
    *,
    object_name: str,
    feature_name: str,
    output_dir: Path,
) -> None:
    dims = sorted({int(point["codebook_dim"]) for point in points}, reverse=True)
    codebook_sizes = sorted({int(point["codebook_size"]) for point in points})
    num_quantizers = sorted({int(point["num_quantizers"]) for point in points})

    grid = np.full((len(dims), len(codebook_sizes)), np.nan, dtype=float)
    labels = [["" for _ in codebook_sizes] for _ in dims]
    for point in points:
        row = dims.index(int(point["codebook_dim"]))
        col = codebook_sizes.index(int(point["codebook_size"]))
        if np.isfinite(grid[row, col]):
            log.warning(
                "Multiple runs map to dim=%s, cb=%s; keeping lower resolution value",
                point["codebook_dim"],
                point["codebook_size"],
            )
            if point["resolution"] >= grid[row, col]:
                continue
        grid[row, col] = point["resolution"]
        labels[row][col] = point["label"]

    masked = np.ma.masked_invalid(grid)
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad(color="0.92")
    image = ax.imshow(masked, cmap=cmap, aspect="auto")

    for row in range(len(dims)):
        for col in range(len(codebook_sizes)):
            value = grid[row, col]
            if not np.isfinite(value):
                ax.text(col, row, "missing", ha="center", va="center", fontsize=11, color="0.45")
                continue
            color = "white" if value > np.nanmedian(grid) else "black"
            ax.text(col, row, f"{value:.4f}", ha="center", va="center", fontsize=13, color=color)

    ax.set_xticks(np.arange(len(codebook_sizes)))
    ax.set_xticklabels([str(value) for value in codebook_sizes], fontsize=15)
    ax.set_yticks(np.arange(len(dims)))
    ax.set_yticklabels([str(value) for value in dims], fontsize=15)
    ax.set_xlabel("Codebook size", fontsize=17)
    ax.set_ylabel("Codebook dim", fontsize=17)
    ax.set_title(
        f"{object_name.capitalize()}: capacity-resolution scan",
        fontsize=20,
        pad=18,
    )
    if len(num_quantizers) == 1:
        ax.text(
            0.5,
            1.02,
            f"nq = {num_quantizers[0]}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=15,
        )
    else:
        ax.text(
            0.5,
            1.02,
            "mixed number of quantizers",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=13,
        )

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(which="both", direction="in", top=True, right=True, length=6, width=1.2)
    ax.tick_params(which="minor", length=3)
    ax.minorticks_on()

    cbar = fig.colorbar(image, ax=ax, pad=0.035)
    cbar.set_label(f"Average {feature_label(feature_name)} resolution", fontsize=16)
    cbar.ax.tick_params(labelsize=13)
    ax.text(
        0.01,
        -0.16,
        "lower is better",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        color="0.25",
    )
    fig.subplots_adjust(left=0.14, right=0.90, bottom=0.18, top=0.86)

    stem = f"{safe_filename(object_name)}_{safe_filename(feature_name)}_capacity_resolution_heatmap"
    fig.savefig(output_dir / f"{stem}.png", dpi=260, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", output_dir / f"{stem}.png")


def plot_points(
    points: list[dict],
    *,
    object_name: str,
    feature_name: str,
    output_dir: Path,
    x_label: str | None,
) -> None:
    x = [point["capacity"] for point in points]
    x_plot = jitter_duplicate_x(x)
    y = [point["resolution"] for point in points]

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for idx, (point, x_value, y_value) in enumerate(zip(points, x_plot, y)):
        ax.scatter(
            x_value,
            y_value,
            s=130,
            color=COLORS[idx % len(COLORS)],
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
        ax.annotate(
            point["label"],
            (x_value, y_value),
            xytext=(8, 7),
            textcoords="offset points",
            fontsize=14,
            fontweight="bold",
        )

    order = np.argsort(x_plot)
    ax.plot(np.asarray(x_plot)[order], np.asarray(y)[order], color="0.35", linewidth=1.5, alpha=0.55)
    ax.set_xlabel(x_label or "Tokenizer capacity", fontsize=17)
    ax.set_ylabel(f"Average {feature_label(feature_name)} resolution", fontsize=17)
    ax.set_title(f"{object_name.capitalize()}: capacity-resolution tradeoff", fontsize=20, pad=12)
    ax.grid(alpha=0.22)
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.text(
        0.02,
        0.96,
        "lower is better",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        color="0.25",
    )
    ax.annotate(
        "more compact",
        xy=(0.02, -0.18),
        xytext=(0.02, -0.18),
        xycoords="axes fraction",
        textcoords="axes fraction",
        fontsize=12,
        ha="left",
        va="center",
        color="0.3",
    )
    ax.annotate(
        "larger vocabulary",
        xy=(0.98, -0.18),
        xytext=(0.98, -0.18),
        xycoords="axes fraction",
        textcoords="axes fraction",
        fontsize=12,
        ha="right",
        va="center",
        color="0.3",
    )
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.22, top=0.86)

    stem = f"{safe_filename(object_name)}_{safe_filename(feature_name)}_capacity_resolution_tradeoff"
    fig.savefig(output_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", output_dir / f"{stem}.png")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    points = []
    x_label = None
    for run_dir, label in parse_runs(args.runs):
        if not run_dir.exists():
            raise FileNotFoundError(run_dir)
        capacity, this_x_label = run_capacity(run_dir, args.capacity_metric)
        codebook_dim, codebook_size, num_quantizers = run_hparams(run_dir)
        x_label = this_x_label
        resolution, n_bins = average_binned_resolution(
            run_dir,
            feature_name=args.feature,
            output_dir=output_dir,
            h5_files=args.h5_files,
            split=args.split,
            max_valid_objects=args.max_valid_objects,
            num_events_per_file=args.num_events_per_file,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        points.append(
            {
                "run": run_dir.name,
                "label": label,
                "capacity": capacity,
                "codebook_dim": codebook_dim,
                "codebook_size": codebook_size,
                "num_quantizers": num_quantizers,
                "resolution": resolution,
                "n_bins": n_bins,
            }
        )

    if args.plot_style == "heatmap":
        plot_heatmap(
            points,
            object_name=args.object,
            feature_name=args.feature,
            output_dir=output_dir,
        )
    else:
        plot_points(
            points,
            object_name=args.object,
            feature_name=args.feature,
            output_dir=output_dir,
            x_label=x_label,
        )

    csv_stem = f"{safe_filename(args.object)}_{safe_filename(args.feature)}_capacity_resolution"
    csv_path = output_dir / f"{csv_stem}.csv"
    lines = ["label,run,capacity,average_resolution,n_bins"]
    for point in points:
        lines.append(
            f"{point['label']},{point['run']},{point['capacity']:.8g},"
            f"{point['resolution']:.8g},{point['n_bins']}"
        )
    csv_path.write_text("\n".join(lines) + "\n")
    log.info("Wrote %s", csv_path)


if __name__ == "__main__":
    main()
