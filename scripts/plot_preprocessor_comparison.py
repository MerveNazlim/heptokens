#!/usr/bin/env python3
"""Compare two fitted object preprocessing transformers.

Useful for checking whether MC-only and MC+data preprocessing differ enough to
affect tokenizer diagnostics.  For log_standard transformers, the stored
StandardScaler parameters live in the space after log-scaling selected features.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from joblib import load as joblib_load

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot learned parameter differences between two preprocessing joblibs."
    )
    parser.add_argument("--mc-only", required=True, help="MC-only preprocessor joblib.")
    parser.add_argument("--mcdata", required=True, help="MC+data preprocessor joblib.")
    parser.add_argument(
        "--feature-names",
        help=(
            "Comma-separated feature names. If omitted, try metadata JSON next to "
            "the MC-only joblib."
        ),
    )
    parser.add_argument("--object", default="object", help="Object name for plot titles.")
    parser.add_argument(
        "--output-dir",
        default="results/fair_mc_vs_mcdata_eval/preprocessor_debug",
    )
    return parser.parse_args()


def metadata_feature_names(joblib_path: Path) -> list[str] | None:
    metadata_path = joblib_path.with_suffix(".json")
    if not metadata_path.exists():
        return None
    data = json.loads(metadata_path.read_text())
    names = data.get("feature_names")
    return list(names) if names else None


def feature_names(args: argparse.Namespace, n_features: int) -> list[str]:
    if args.feature_names:
        names = [name.strip() for name in args.feature_names.split(",") if name.strip()]
    else:
        names = metadata_feature_names(Path(args.mc_only))
        if names is None:
            names = [f"f{idx}" for idx in range(n_features)]
    if len(names) != n_features:
        raise ValueError(f"Got {len(names)} feature names but transformer has {n_features} features")
    return names


def final_transformer(transformer):
    return getattr(transformer, "final_transformer", transformer)


def log_indices(transformer) -> set[int]:
    log_transformer = getattr(transformer, "log_transformer", None)
    if log_transformer is None:
        return set()
    indices: set[int] = set()
    for config in getattr(log_transformer, "feature_configs", []):
        raw = config.get("indices", [])
        if isinstance(raw, int):
            indices.add(raw)
        else:
            indices.update(int(value) for value in raw)
    return indices


def scaler_params(transformer) -> tuple[np.ndarray, np.ndarray]:
    final = final_transformer(transformer)
    if not hasattr(final, "mean_") or not hasattr(final, "scale_"):
        raise TypeError(
            f"Expected final transformer with mean_/scale_, got {type(final).__name__}"
        )
    return np.asarray(final.mean_, dtype=float), np.asarray(final.scale_, dtype=float)


def inverse_at_feature_z(transformer, feature_idx: int, z_values: np.ndarray, n_features: int) -> np.ndarray:
    z = np.zeros((len(z_values), n_features), dtype=float)
    z[:, feature_idx] = z_values
    return transformer.inverse_transform(z)[:, feature_idx]


def write_parameter_csv(
    path: Path,
    names: list[str],
    old_mean: np.ndarray,
    old_scale: np.ndarray,
    new_mean: np.ndarray,
    new_scale: np.ndarray,
    old_log: set[int],
    new_log: set[int],
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "feature",
                "mc_only_mean",
                "mcdata_mean",
                "mean_delta_mcdata_minus_mc_only",
                "mc_only_scale",
                "mcdata_scale",
                "scale_ratio_mcdata_over_mc_only",
                "mc_only_log_feature",
                "mcdata_log_feature",
            ],
        )
        writer.writeheader()
        for idx, name in enumerate(names):
            writer.writerow(
                {
                    "feature": name,
                    "mc_only_mean": old_mean[idx],
                    "mcdata_mean": new_mean[idx],
                    "mean_delta_mcdata_minus_mc_only": new_mean[idx] - old_mean[idx],
                    "mc_only_scale": old_scale[idx],
                    "mcdata_scale": new_scale[idx],
                    "scale_ratio_mcdata_over_mc_only": new_scale[idx] / old_scale[idx]
                    if old_scale[idx]
                    else np.nan,
                    "mc_only_log_feature": idx in old_log,
                    "mcdata_log_feature": idx in new_log,
                }
            )


def write_inverse_grid_csv(
    path: Path,
    names: list[str],
    old_transformer,
    new_transformer,
    z_values: np.ndarray,
) -> None:
    n_features = len(names)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["feature", "z_value", "mc_only_physical", "mcdata_physical"],
        )
        writer.writeheader()
        for idx, name in enumerate(names):
            old_phys = inverse_at_feature_z(old_transformer, idx, z_values, n_features)
            new_phys = inverse_at_feature_z(new_transformer, idx, z_values, n_features)
            for z_value, old_value, new_value in zip(z_values, old_phys, new_phys):
                writer.writerow(
                    {
                        "feature": name,
                        "z_value": z_value,
                        "mc_only_physical": old_value,
                        "mcdata_physical": new_value,
                    }
                )


def plot_params(
    path: Path,
    object_name: str,
    names: list[str],
    old_mean: np.ndarray,
    old_scale: np.ndarray,
    new_mean: np.ndarray,
    new_scale: np.ndarray,
    old_log: set[int],
    new_log: set[int],
) -> None:
    x = np.arange(len(names))
    width = 0.38
    labels = [
        f"{name}{'*' if idx in old_log or idx in new_log else ''}"
        for idx, name in enumerate(names)
    ]

    fig, axes = plt.subplots(2, 1, figsize=(max(8, 0.8 * len(names)), 7.2), sharex=True)
    axes[0].bar(x - width / 2, old_mean, width, label="MC-only", color="#4C83F1")
    axes[0].bar(x + width / 2, new_mean, width, label="MC+data", color="#FF9F1C")
    axes[0].set_ylabel("learned mean")
    axes[0].set_title(f"{object_name}: preprocessing means")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x - width / 2, old_scale, width, label="MC-only", color="#4C83F1")
    axes[1].bar(x + width / 2, new_scale, width, label="MC+data", color="#FF9F1C")
    axes[1].set_ylabel("learned scale")
    axes[1].set_title("preprocessing scales")
    axes[1].grid(axis="y", alpha=0.25)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    fig.text(0.01, 0.01, "* log-transformed before standard scaling", fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_deltas(
    path: Path,
    object_name: str,
    names: list[str],
    old_mean: np.ndarray,
    old_scale: np.ndarray,
    new_mean: np.ndarray,
    new_scale: np.ndarray,
) -> None:
    x = np.arange(len(names))
    scale_ratio = np.divide(new_scale, old_scale, out=np.full_like(new_scale, np.nan), where=old_scale != 0)

    fig, axes = plt.subplots(2, 1, figsize=(max(8, 0.8 * len(names)), 7.0), sharex=True)
    axes[0].bar(x, new_mean - old_mean, color="#6C757D")
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_ylabel("MC+data - MC-only")
    axes[0].set_title(f"{object_name}: mean shift")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, scale_ratio, color="#6C757D")
    axes[1].axhline(1, color="black", linewidth=1)
    axes[1].set_ylabel("MC+data / MC-only")
    axes[1].set_title("scale ratio")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_inverse_grid(
    path: Path,
    object_name: str,
    names: list[str],
    old_transformer,
    new_transformer,
) -> None:
    z_values = np.linspace(-3, 3, 121)
    n_features = len(names)
    ncols = min(3, n_features)
    nrows = int(np.ceil(n_features / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.8 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)
    for idx, name in enumerate(names):
        ax = axes[idx]
        old_phys = inverse_at_feature_z(old_transformer, idx, z_values, n_features)
        new_phys = inverse_at_feature_z(new_transformer, idx, z_values, n_features)
        ax.plot(z_values, old_phys, label="MC-only", color="#4C83F1", linewidth=2)
        ax.plot(z_values, new_phys, label="MC+data", color="#FF9F1C", linewidth=2)
        ax.set_title(name)
        ax.set_xlabel("standardized value")
        ax.set_ylabel("physical value")
        ax.grid(alpha=0.25)
    for ax in axes[n_features:]:
        ax.set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle(f"{object_name}: inverse preprocessing map", fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    old_path = Path(args.mc_only)
    new_path = Path(args.mcdata)
    old_transformer = joblib_load(old_path)
    new_transformer = joblib_load(new_path)
    old_mean, old_scale = scaler_params(old_transformer)
    new_mean, new_scale = scaler_params(new_transformer)
    if len(old_mean) != len(new_mean):
        raise ValueError(f"Feature count mismatch: {len(old_mean)} vs {len(new_mean)}")

    names = feature_names(args, len(old_mean))
    old_log = log_indices(old_transformer)
    new_log = log_indices(new_transformer)

    output_dir = Path(args.output_dir) / args.object
    output_dir.mkdir(parents=True, exist_ok=True)

    write_parameter_csv(
        output_dir / "preprocessor_parameters.csv",
        names,
        old_mean,
        old_scale,
        new_mean,
        new_scale,
        old_log,
        new_log,
    )
    z_values = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=float)
    write_inverse_grid_csv(
        output_dir / "inverse_preprocessor_grid.csv",
        names,
        old_transformer,
        new_transformer,
        z_values,
    )
    plot_params(
        output_dir / "preprocessor_mean_scale.png",
        args.object,
        names,
        old_mean,
        old_scale,
        new_mean,
        new_scale,
        old_log,
        new_log,
    )
    plot_deltas(
        output_dir / "preprocessor_deltas.png",
        args.object,
        names,
        old_mean,
        old_scale,
        new_mean,
        new_scale,
    )
    plot_inverse_grid(
        output_dir / "inverse_preprocessor_map.png",
        args.object,
        names,
        old_transformer,
        new_transformer,
    )

    print(f"Wrote {output_dir}")
    print("Largest scale-ratio changes:")
    scale_ratio = np.divide(new_scale, old_scale, out=np.full_like(new_scale, np.nan), where=old_scale != 0)
    order = np.argsort(np.abs(np.log(scale_ratio)))[::-1]
    for idx in order[: min(5, len(order))]:
        print(
            f"  {names[idx]}: mean {old_mean[idx]:.5g} -> {new_mean[idx]:.5g}, "
            f"scale {old_scale[idx]:.5g} -> {new_scale[idx]:.5g} "
            f"(ratio {scale_ratio[idx]:.3g})"
        )


if __name__ == "__main__":
    main()
