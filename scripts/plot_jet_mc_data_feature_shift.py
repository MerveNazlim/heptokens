#!/usr/bin/env python3
"""Compare raw jet feature distributions and correlations in MC and real data.

This is the jet counterpart of plot_electron_mc_data_isolation_shift.py. It
reads physical values directly from HDF5 and does not run tokenizer
preprocessing or VQ-VAE reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


LOG = logging.getLogger(__name__)

FEATURE_PATHS = {
    "pt": "common/jets/pt",
    "eta": "common/jets/eta",
    "mass": "common/jets/mass",
    "n_trk": "common/jets/n_trk",
    "GN2_pc": "atlas/jets/GN2_pc",
    "GN2_pu": "atlas/jets/GN2_pu",
}
MASK_PATH = "common/jets/mask"
COLORS = {"MC": "#4C83F1", "real data": "#222222"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot raw MC/data jet feature and correlation shifts."
    )
    parser.add_argument(
        "--mc-dir", default="/home/zephyr/Data/viviana/bnl-treasure/data/h5"
    )
    parser.add_argument(
        "--data-dir",
        default="/home/zephyr/Data/viviana/bnl-treasure/data/h5/realdata",
    )
    parser.add_argument(
        "--output-dir", default="results/jet_mc_data_feature_shift"
    )
    parser.add_argument("--n-files", type=int, default=20)
    parser.add_argument("--max-events-per-file", type=int, default=200_000)
    parser.add_argument("--max-objects", type=int, default=500_000)
    parser.add_argument("--n-bins", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def fixed_files(directory: str, n_files: int) -> list[Path]:
    return sorted(
        path
        for path in Path(directory).glob("*.h5")
        if path.is_file() and path.stat().st_size > 0
    )[:n_files]


def read_domain(
    files: list[Path], max_events_per_file: int, max_objects: int, seed: int
) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for path in files:
        try:
            with h5py.File(path, "r") as handle:
                required = [MASK_PATH, *FEATURE_PATHS.values()]
                missing = [name for name in required if name not in handle]
                if missing:
                    LOG.warning("Skipping %s; missing %s", path, missing)
                    continue

                n_events = min(handle[MASK_PATH].shape[0], max_events_per_file)
                mask = np.asarray(handle[MASK_PATH][:n_events], dtype=bool)
                values = {}
                for name, h5_path in FEATURE_PATHS.items():
                    array = np.asarray(handle[h5_path][:n_events])
                    if array.ndim != 2 or array.shape != mask.shape:
                        raise ValueError(
                            f"{path}: {h5_path} shape {array.shape} does not match "
                            f"mask shape {mask.shape}"
                        )
                    values[name] = array[mask]
                chunks.append(pd.DataFrame(values))
        except OSError as exc:
            LOG.warning("Skipping unreadable file %s: %s", path, exc)

    if not chunks:
        raise RuntimeError("No valid jet objects were read")

    frame = pd.concat(chunks, ignore_index=True)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) > max_objects:
        frame = frame.sample(max_objects, random_state=seed).reset_index(drop=True)
    frame["abs_eta"] = frame["eta"].abs()
    return frame


def pooled_plot_range(mc: np.ndarray, data: np.ndarray) -> tuple[float, float]:
    pooled = np.concatenate([mc[np.isfinite(mc)], data[np.isfinite(data)]])
    low, high = np.percentile(pooled, [0.5, 99.5])
    if low == high:
        low, high = float(np.min(pooled)), float(np.max(pooled))
    if low == high:
        low, high = low - 0.5, high + 0.5
    return float(low), float(high)


def empirical_ks(left: np.ndarray, right: np.ndarray) -> float:
    left = np.sort(left[np.isfinite(left)])
    right = np.sort(right[np.isfinite(right)])
    points = np.sort(np.concatenate([left, right]))
    left_cdf = np.searchsorted(left, points, side="right") / len(left)
    right_cdf = np.searchsorted(right, points, side="right") / len(right)
    return float(np.max(np.abs(left_cdf - right_cdf)))


def robust_stats(values: np.ndarray) -> dict[str, float]:
    q25, median, q75 = np.percentile(values, [25, 50, 75])
    return {
        "median": float(median),
        "iqr": float(q75 - q25),
    }


def plot_distributions(mc: pd.DataFrame, data: pd.DataFrame, output_dir: Path) -> None:
    features = list(FEATURE_PATHS)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, feature in zip(axes.flat, features):
        low, high = pooled_plot_range(mc[feature].to_numpy(), data[feature].to_numpy())
        bins = np.linspace(low, high, 61)
        for label, frame in [("MC", mc), ("real data", data)]:
            ax.hist(
                frame[feature],
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2.2,
                color=COLORS[label],
                label=label,
            )
        ax.set_xlabel(feature)
        ax.set_ylabel("Normalized density")
        ax.set_yscale("log")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Jet input distributions: MC vs real data")
    fig.tight_layout()
    fig.savefig(output_dir / "jet_feature_distributions.png", dpi=200)
    plt.close(fig)


def binned_median_iqr(
    x: np.ndarray, y: np.ndarray, bins: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers = 0.5 * (bins[:-1] + bins[1:])
    median = np.full(len(centers), np.nan)
    q25 = np.full(len(centers), np.nan)
    q75 = np.full(len(centers), np.nan)
    for index in range(len(centers)):
        selected = (x >= bins[index]) & (x < bins[index + 1])
        if index == len(centers) - 1:
            selected = (x >= bins[index]) & (x <= bins[index + 1])
        if np.count_nonzero(selected) < 30:
            continue
        q25[index], median[index], q75[index] = np.percentile(y[selected], [25, 50, 75])
    return centers, median, q25, q75


def plot_conditional_features(
    mc: pd.DataFrame, data: pd.DataFrame, output_dir: Path, n_bins: int
) -> None:
    targets = ["mass", "n_trk", "GN2_pc", "GN2_pu"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    pooled_pt = np.concatenate([mc["pt"], data["pt"]])
    bins = np.unique(np.quantile(pooled_pt, np.linspace(0, 1, n_bins + 1)))

    for ax, target in zip(axes.flat, targets):
        for label, frame in [("MC", mc), ("real data", data)]:
            centers, median, q25, q75 = binned_median_iqr(
                frame["pt"].to_numpy(), frame[target].to_numpy(), bins
            )
            valid = np.isfinite(median)
            ax.plot(
                centers[valid], median[valid], marker="o", linewidth=2,
                color=COLORS[label], label=label,
            )
            ax.fill_between(
                centers[valid], q25[valid], q75[valid],
                color=COLORS[label], alpha=0.13,
            )
        ax.set_xlabel("jet pT [GeV]")
        ax.set_ylabel(f"{target} median and IQR")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Jet feature correlations with pT in MC and real data")
    fig.tight_layout()
    fig.savefig(output_dir / "jet_features_vs_pt.png", dpi=200)
    plt.close(fig)


def spearman_matrix(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return frame[features].rank(method="average").corr(method="pearson")


def plot_correlations(mc: pd.DataFrame, data: pd.DataFrame, output_dir: Path) -> None:
    features = ["pt", "abs_eta", "mass", "n_trk", "GN2_pc", "GN2_pu"]
    mc_corr = spearman_matrix(mc, features)
    data_corr = spearman_matrix(data, features)
    delta = data_corr - mc_corr

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for ax, (matrix, title) in zip(
        axes, [(mc_corr, "MC"), (data_corr, "real data"), (delta, "data - MC")]
    ):
        limit = 1.0 if title != "data - MC" else max(0.1, float(np.nanmax(np.abs(matrix))))
        image = ax.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit)
        ax.set_xticks(range(len(features)), features, rotation=45, ha="right")
        ax.set_yticks(range(len(features)), features)
        ax.set_title(title)
        for row in range(len(features)):
            for column in range(len(features)):
                ax.text(
                    column, row, f"{matrix.iloc[row, column]:.2f}",
                    ha="center", va="center", fontsize=8,
                )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Jet Spearman feature correlations")
    fig.tight_layout()
    fig.savefig(output_dir / "jet_mc_data_correlation_matrices.png", dpi=200)
    plt.close(fig)

    mc_corr.to_csv(output_dir / "jet_mc_spearman.csv")
    data_corr.to_csv(output_dir / "jet_data_spearman.csv")
    delta.to_csv(output_dir / "jet_data_minus_mc_spearman.csv")


def write_report(mc: pd.DataFrame, data: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    for feature in FEATURE_PATHS:
        mc_values = mc[feature].to_numpy()
        data_values = data[feature].to_numpy()
        mc_stats = robust_stats(mc_values)
        data_stats = robust_stats(data_values)
        rows.append(
            {
                "feature": feature,
                "mc_median": mc_stats["median"],
                "data_median": data_stats["median"],
                "mc_iqr": mc_stats["iqr"],
                "data_iqr": data_stats["iqr"],
                "data_over_mc_iqr": (
                    data_stats["iqr"] / mc_stats["iqr"] if mc_stats["iqr"] != 0 else np.nan
                ),
                "ks_distance": empirical_ks(mc_values, data_values),
            }
        )

    with (output_dir / "jet_mc_data_shift_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Jet MC/data input-shift summary",
        "",
        f"MC jets: {len(mc):,}",
        f"Real-data jets: {len(data):,}",
        "",
        "| feature | MC median | data median | MC IQR | data IQR | data/MC IQR | KS distance |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['feature']} | {row['mc_median']:.5g} | {row['data_median']:.5g} | "
            f"{row['mc_iqr']:.5g} | {row['data_iqr']:.5g} | "
            f"{row['data_over_mc_iqr']:.3f} | {row['ks_distance']:.3f} |"
        )
    (output_dir / "jet_mc_data_shift_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mc_files = fixed_files(args.mc_dir, args.n_files)
    data_files = fixed_files(args.data_dir, args.n_files)
    if not mc_files or not data_files:
        raise FileNotFoundError("No non-empty MC or real-data H5 files found")

    LOG.info("Reading %d MC and %d real-data files", len(mc_files), len(data_files))
    mc = read_domain(mc_files, args.max_events_per_file, args.max_objects, args.seed)
    data = read_domain(data_files, args.max_events_per_file, args.max_objects, args.seed)
    LOG.info("Using %d MC and %d real-data jets", len(mc), len(data))

    plot_distributions(mc, data, output_dir)
    plot_conditional_features(mc, data, output_dir, args.n_bins)
    plot_correlations(mc, data, output_dir)
    write_report(mc, data, output_dir)
    LOG.info("Wrote diagnostics to %s", output_dir)


if __name__ == "__main__":
    main()
