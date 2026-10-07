#!/usr/bin/env python3
"""Make compact summary plots for tokenizer capacity scans."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import hydra
import matplotlib
import numpy as np
import torch
from omegaconf import OmegaConf

from analyze_vqvae_tokenizer import (
    analysis_datamodule_cfg,
    apply_hep_style,
    choose_device,
    collect_diagnostics_from_loader,
    dataloader_from_datamodule,
    feature_label,
    feature_names_from_cfg,
    find_checkpoint,
    safe_filename,
    transform_list_and_cst_fn_from_cfg,
)
from heptokens.models.vq_vae import LitVqVae

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


log = logging.getLogger(__name__)


DEFAULT_FEATURES = {
    "jets": ["pt", "GN2_pc", "n_trk"],
    "electrons": ["pt", "ptvarcone30", "topoetcone20"],
    "muons": ["pt", "ptvarcone30", "topoetcone20"],
    "photons": ["pt", "ptcone20", "topoetcone40"],
    "taus": ["pt", "NNDecayMode", "RNNJetScore"],
    "tracks": ["pt", "nDoF", "chiSquared"],
}

COLORS = [
    "#4C83F1",
    "#FF9F1C",
    "#E71D36",
    "#2EC4B6",
    "#7A5CFF",
    "#2A9D8F",
    "#B5179E",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create slide-friendly summary plots from tokenizer scan run directories."
    )
    parser.add_argument(
        "--run-roots",
        nargs="+",
        required=True,
        help="Directories containing tokenizer runs.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for summary plots and tables.",
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        help="Optional object names to include, e.g. jets electrons tracks.",
    )
    parser.add_argument(
        "--features",
        action="append",
        default=[],
        metavar="OBJECT=FEATURE1,FEATURE2",
        help=(
            "Features for response-vs-truth plots. May be repeated. "
            "Best for positive nonzero features."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of best runs per object to plot.",
    )
    parser.add_argument(
        "--prefer-runs",
        action="append",
        default=[],
        metavar="OBJECT=RUN_NAME1,RUN_NAME2",
        help="Force specific runs to be included first for an object.",
    )
    parser.add_argument(
        "--legend-labels",
        action="append",
        default=[],
        metavar="RUN_NAME=LABEL",
        help="Override plot legend labels for selected runs. May be repeated.",
    )
    parser.add_argument(
        "--h5-files",
        nargs="+",
        help="Optional H5 files for binned diagnostics. Defaults to files in full_config.yaml.",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--max-valid-objects", type=int, default=1_000_000)
    parser.add_argument("--num-events-per-file", type=int)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--skip-binned",
        action="store_true",
        help="Only make codebook utilization plots and tables.",
    )
    parser.add_argument(
        "--only-binned",
        action="store_true",
        help="Only make binned response-vs-truth plots.",
    )
    parser.add_argument(
        "--metric",
        choices=["overall_mae", "overall_rmse"],
        default="overall_mae",
        help="Metric used to rank top runs.",
    )
    return parser.parse_args()


def parse_feature_overrides(values: list[str]) -> dict[str, list[str]]:
    features = {key: list(value) for key, value in DEFAULT_FEATURES.items()}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected OBJECT=FEATURE1,FEATURE2, got {value}")
        object_name, names = value.split("=", 1)
        features[object_name] = [name.strip() for name in names.split(",") if name.strip()]
    return features


def parse_preferred_runs(values: list[str]) -> dict[str, list[str]]:
    preferred: dict[str, list[str]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected OBJECT=RUN1,RUN2, got {value}")
        object_name, names = value.split("=", 1)
        preferred[object_name] = [name.strip() for name in names.split(",") if name.strip()]
    return preferred


def parse_legend_labels(values: list[str]) -> dict[str, str]:
    labels = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected RUN_NAME=LABEL, got {value}")
        run_name, label = value.split("=", 1)
        labels[run_name.strip()] = label.strip()
    return labels


def discover_runs(run_roots: list[str]) -> list[Path]:
    runs = []
    for root in run_roots:
        root_path = Path(root).expanduser().resolve()
        if not root_path.exists():
            log.warning("Skipping missing run root: %s", root_path)
            continue
        for cfg_path in root_path.glob("*/full_config.yaml"):
            runs.append(cfg_path.parent)
    return sorted(set(runs))


def object_type_from_cfg(cfg) -> str:
    datamodule = OmegaConf.to_container(cfg.datamodule, resolve=True)
    object_type = datamodule.get("object_type")
    if object_type:
        return str(object_type)
    collections = datamodule.get("object_collections") or []
    if collections and collections[0].get("object_name"):
        return str(collections[0]["object_name"])
    return "object"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def feature_metric(reconstruction: dict[str, Any], feature_name: str, metric: str) -> float:
    lower = feature_name.lower()
    for actual_name, values in reconstruction.items():
        if actual_name.lower() == lower:
            return float(values.get(metric, np.nan))
    return float("nan")


def summarize_reconstruction(reconstruction: dict[str, Any]) -> tuple[float, float, float]:
    entries = list(reconstruction.values())
    total = sum(int(entry.get("n", 0)) for entry in entries)
    if total <= 0:
        return float("nan"), float("nan"), float("nan")
    overall_mae = sum(float(entry["mae"]) * int(entry["n"]) for entry in entries) / total
    overall_rmse = np.sqrt(
        sum(float(entry["rmse"]) ** 2 * int(entry["n"]) for entry in entries) / total
    )
    mean_abs_bias = sum(abs(float(entry["bias"])) * int(entry["n"]) for entry in entries) / total
    return float(overall_mae), float(overall_rmse), float(mean_abs_bias)


def usage_from_counts(counts: np.ndarray) -> dict[str, float]:
    used_fraction = np.count_nonzero(counts, axis=1) / counts.shape[1]
    return {
        "mean_used_fraction": float(np.mean(used_fraction)),
        "q0_used_fraction": float(used_fraction[0]),
        "last_used_fraction": float(used_fraction[-1]),
    }


def load_run_summary(run_dir: Path, selected_features: list[str]) -> dict[str, Any] | None:
    cfg_path = run_dir / "full_config.yaml"
    if not cfg_path.exists():
        return None
    cfg = OmegaConf.load(cfg_path)
    object_name = object_type_from_cfg(cfg)
    analysis_dir = run_dir / "figures" / "tokenizer_analysis"
    reconstruction = load_json(analysis_dir / "reconstruction_metrics.json")
    counts_path = analysis_dir / "codebook_counts.npy"
    if not reconstruction or not counts_path.exists():
        log.warning("Skipping %s: diagnostics missing", run_dir.name)
        return None
    counts = np.load(counts_path)
    overall_mae, overall_rmse, mean_abs_bias = summarize_reconstruction(reconstruction)
    usage = usage_from_counts(counts)

    row: dict[str, Any] = {
        "object": object_name,
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "overall_mae": overall_mae,
        "overall_rmse": overall_rmse,
        "mean_abs_bias": mean_abs_bias,
        "n_quantizers": int(counts.shape[0]),
        "codebook_size": int(counts.shape[1]),
        **usage,
    }
    for name in selected_features:
        row[f"rmse_{name}"] = feature_metric(reconstruction, name, "rmse")
        row[f"mae_{name}"] = feature_metric(reconstruction, name, "mae")
        row[f"bias_{name}"] = feature_metric(reconstruction, name, "bias")
    return row


def select_top_runs(
    rows: list[dict[str, Any]],
    *,
    top_k: int,
    metric: str,
    preferred: dict[str, list[str]],
) -> dict[str, list[dict[str, Any]]]:
    by_object: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_object.setdefault(row["object"], []).append(row)

    selected = {}
    for object_name, object_rows in by_object.items():
        run_by_name = {row["run_name"]: row for row in object_rows}
        ordered = []
        for run_name in preferred.get(object_name, []):
            if run_name in run_by_name:
                ordered.append(run_by_name[run_name])

        ranked = sorted(
            object_rows,
            key=lambda row: (
                not np.isfinite(float(row.get(metric, np.nan))),
                float(row.get(metric, np.inf)),
                row["codebook_size"] * row["n_quantizers"],
            ),
        )
        for row in ranked:
            if row not in ordered:
                ordered.append(row)
            if len(ordered) >= top_k:
                break
        selected[object_name] = ordered[:top_k]
    return selected


def write_summary_tables(
    rows_by_object: dict[str, list[dict[str, Any]]],
    features_by_object: dict[str, list[str]],
    output_dir: Path,
) -> None:
    csv_path = output_dir / "summary_top_runs.csv"
    fieldnames = [
        "object",
        "run_name",
        "overall_mae",
        "overall_rmse",
        "mean_used_fraction",
        "q0_used_fraction",
        "last_used_fraction",
    ]
    for features in features_by_object.values():
        for feature in features:
            key = f"rmse_{feature}"
            if key not in fieldnames:
                fieldnames.append(key)

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for object_name in sorted(rows_by_object):
            for row in rows_by_object[object_name]:
                writer.writerow(row)

    lines = ["# Tokenizer Summary Top Runs", ""]
    for object_name in sorted(rows_by_object):
        features = features_by_object.get(object_name, [])
        lines.append(f"## {object_name}")
        header = [
            "run",
            "MAE",
            "RMSE",
            "mean used",
            "q0 used",
            "last-q used",
            *[f"{feature} RMSE" for feature in features],
        ]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for row in rows_by_object[object_name]:
            values = [
                row["run_name"],
                f"{float(row['overall_mae']):.4g}",
                f"{float(row['overall_rmse']):.4g}",
                f"{100 * float(row['mean_used_fraction']):.1f}%",
                f"{100 * float(row['q0_used_fraction']):.1f}%",
                f"{100 * float(row['last_used_fraction']):.1f}%",
            ]
            for feature in features:
                value = row.get(f"rmse_{feature}", np.nan)
                values.append("" if not np.isfinite(float(value)) else f"{float(value):.4g}")
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    (output_dir / "summary_top_runs.md").write_text("\n".join(lines) + "\n")


def short_label(run_name: str, object_name: str) -> str:
    label = run_name
    for prefix in (f"{object_name}_", f"{object_name}s_"):
        if label.startswith(prefix):
            label = label[len(prefix) :]
    label = label.replace("logstd_", "").replace("no_ndoflog_", "no nDoF log ")
    return label


def plot_codebook_utilization(
    object_name: str,
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    max_quantizers = max(int(row["n_quantizers"]) for row in rows)
    x = np.arange(len(rows))
    width = min(0.8 / max_quantizers, 0.18)

    fig, ax = plt.subplots(figsize=(max(8, 2.5 * len(rows)), 5.0))
    for q_idx in range(max_quantizers):
        values = []
        for row in rows:
            counts = np.load(Path(row["run_dir"]) / "figures" / "tokenizer_analysis" / "codebook_counts.npy")
            if q_idx < counts.shape[0]:
                values.append(100 * np.count_nonzero(counts[q_idx]) / counts.shape[1])
            else:
                values.append(np.nan)
        offset = (q_idx - (max_quantizers - 1) / 2) * width
        ax.bar(x + offset, values, width=width, color=COLORS[q_idx % len(COLORS)], label=f"q{q_idx}")

    ax.set_xticks(x)
    ax.set_xticklabels([short_label(row["run_name"], object_name) for row in rows], rotation=20, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Codebook utilization [%]")
    ax.set_title(f"{object_name}: codebook utilization")
    ax.legend(ncols=max_quantizers, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    apply_hep_style(ax)
    fig.tight_layout()
    fig.savefig(output_dir / f"{safe_filename(object_name)}_codebook_utilization.png", dpi=180)
    plt.close(fig)


def load_or_collect_arrays(
    run_dir: Path,
    *,
    output_dir: Path,
    h5_files: list[str] | None,
    split: str,
    max_valid_objects: int,
    num_events_per_file: int | None,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cache_dir = output_dir / "cache_arrays"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{safe_filename(run_dir.name)}_{split}_{max_valid_objects}.npz"
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        return cached["original"], cached["reconstruction"], list(cached["feature_names"])

    cfg = OmegaConf.load(run_dir / "full_config.yaml")
    _, cst_inverse_transformer = transform_list_and_cst_fn_from_cfg(cfg)
    checkpoint = find_checkpoint(run_dir, None)
    model = LitVqVae.load_from_checkpoint(checkpoint, map_location=device)
    model.to(device)
    model.eval()

    loader_args = SimpleNamespace(
        h5_files=h5_files,
        num_events_per_file=num_events_per_file,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    datamodule_cfg = analysis_datamodule_cfg(cfg, loader_args)
    datamodule = hydra.utils.instantiate(datamodule_cfg)
    loader = dataloader_from_datamodule(datamodule, split)

    original, reconstruction, _, _ = collect_diagnostics_from_loader(
        model=model,
        cst_inverse_transformer=cst_inverse_transformer,
        loader=loader,
        device=device,
        max_valid_objects=max_valid_objects,
    )
    feature_names = feature_names_from_cfg(cfg, original.shape[1])
    np.savez_compressed(
        cache_path,
        original=original,
        reconstruction=reconstruction,
        feature_names=np.asarray(feature_names),
    )
    return original, reconstruction, feature_names


def feature_index(feature_names: list[str], feature_name: str) -> int | None:
    lower = feature_name.lower()
    for idx, actual_name in enumerate(feature_names):
        if actual_name.lower() == lower:
            return idx
    return None


def binned_response_iqr_over_median(
    truth: np.ndarray,
    original: np.ndarray,
    reconstruction: np.ndarray,
    bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    centers = 0.5 * (bins[:-1] + bins[1:])
    values = np.full_like(centers, np.nan, dtype=np.float64)
    for idx in range(len(centers)):
        mask = (truth >= bins[idx]) & (truth < bins[idx + 1])
        if np.count_nonzero(mask) < 20:
            continue
        orig = original[mask]
        reco = reconstruction[mask]
        finite = np.isfinite(orig) & np.isfinite(reco)
        orig = orig[finite]
        reco = reco[finite]
        if len(orig) < 20:
            continue
        nonzero = np.abs(orig) > 1e-12
        response = reco[nonzero] / orig[nonzero]
        response = response[np.isfinite(response)]
        if len(response) < 20:
            continue
        q25, q50, q75 = np.percentile(response, [25, 50, 75])
        values[idx] = (q75 - q25) / abs(q50) if abs(q50) > 1e-12 else np.nan
    return centers, values


def plot_binned_feature_response(
    object_name: str,
    feature_name: str,
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    h5_files: list[str] | None,
    split: str,
    max_valid_objects: int,
    num_events_per_file: int | None,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    legend_labels: dict[str, str] | None = None,
) -> None:
    curves = []
    for row in rows:
        run_dir = Path(row["run_dir"])
        log.info("Collecting binned response diagnostics for %s / %s", run_dir.name, feature_name)
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
        feat_idx = feature_index(feature_names, feature_name)
        if feat_idx is None:
            log.warning("Skipping %s for %s: missing feature", feature_name, run_dir.name)
            continue
        truth_feature = original[:, feat_idx]
        finite_truth = truth_feature[np.isfinite(truth_feature)]
        finite_truth = finite_truth[np.abs(finite_truth) > 1e-12]
        if len(finite_truth) == 0:
            continue
        lo, hi = np.percentile(finite_truth, [1.0, 99.0])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = float(np.min(finite_truth)), float(np.max(finite_truth))
        if lo == hi:
            log.warning("Skipping %s for %s: no useful truth range", feature_name, run_dir.name)
            continue
        bins = np.linspace(lo, hi, 15)
        centers, values = binned_response_iqr_over_median(
            original[:, feat_idx],
            original[:, feat_idx],
            reconstruction[:, feat_idx],
            bins=bins,
        )
        curves.append((row, centers, values))

    if not curves:
        return

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    for idx, (row, centers, values) in enumerate(curves):
        valid = np.isfinite(values)
        ax.plot(
            centers[valid],
            values[valid],
            marker="o",
            markersize=7.0,
            linewidth=2.4,
            color=COLORS[idx % len(COLORS)],
            label=(legend_labels or {}).get(
                row["run_name"],
                short_label(row["run_name"], object_name),
            ),
        )
    label = feature_label(feature_name)
    apply_hep_style(ax)
    ax.set_xlabel(f"Truth {label}", fontsize=18)
    ax.set_ylabel("Relative resolution", fontsize=18)
    ax.set_title(f"{object_name.capitalize()}: {label} response vs truth {label}", fontsize=20, pad=12)
    ax.legend(frameon=False, fontsize=13, loc="best")
    ax.grid(alpha=0.22)
    ax.tick_params(axis="both", which="major", labelsize=14)
    # ax.text(
    #     0.98,
    #     0.06,
    #     "lower is better",
    #     transform=ax.transAxes,
    #     ha="right",
    #     va="bottom",
    #     fontsize=13,
    #     color="0.25",
    # )
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.16, top=0.86)
    out_dir = output_dir / "binned_response"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_dir / f"{safe_filename(object_name)}_{safe_filename(feature_name)}_response_vs_truth.png",
        dpi=240,
        bbox_inches="tight",
    )
    fig.savefig(
        out_dir / f"{safe_filename(object_name)}_{safe_filename(feature_name)}_response_vs_truth.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_combined_codebook_usage(rows_by_object: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    objects = sorted(rows_by_object)
    fig, axes = plt.subplots(len(objects), 1, figsize=(11, max(3.0 * len(objects), 4.0)), squeeze=False)
    for ax, object_name in zip(axes[:, 0], objects):
        rows = rows_by_object[object_name]
        x = np.arange(len(rows))
        width = 0.18
        for q_idx in range(max(int(row["n_quantizers"]) for row in rows)):
            values = []
            for row in rows:
                counts = np.load(Path(row["run_dir"]) / "figures" / "tokenizer_analysis" / "codebook_counts.npy")
                if q_idx < counts.shape[0]:
                    values.append(100 * np.count_nonzero(counts[q_idx]) / counts.shape[1])
                else:
                    values.append(np.nan)
            ax.bar(x + (q_idx - 1.5) * width, values, width=width, color=COLORS[q_idx], label=f"q{q_idx}")
        ax.set_xticks(x)
        ax.set_xticklabels([short_label(row["run_name"], object_name) for row in rows], rotation=15, ha="right")
        ax.set_ylabel(f"{object_name}\nutil. [%]")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.2)
        apply_hep_style(ax)
    axes[0, 0].legend(ncols=4, frameon=False, loc="upper right")
    fig.suptitle("Codebook utilization for top tokenizer scans", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_dir / "codebook_utilization_all_objects.png", dpi=180)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    features_by_object = parse_feature_overrides(args.features)
    preferred = parse_preferred_runs(args.prefer_runs)
    legend_labels = parse_legend_labels(args.legend_labels)
    selected_objects = set(args.objects or [])

    run_dirs = discover_runs(args.run_roots)
    rows = []
    for run_dir in run_dirs:
        cfg = OmegaConf.load(run_dir / "full_config.yaml")
        object_name = object_type_from_cfg(cfg)
        if selected_objects and object_name not in selected_objects:
            continue
        row = load_run_summary(run_dir, features_by_object.get(object_name, []))
        if row is not None:
            rows.append(row)

    if not rows:
        raise RuntimeError("No runs with diagnostics found. Run analyze_vqvae_tokenizer.py first.")

    rows_by_object = select_top_runs(
        rows,
        top_k=args.top_k,
        metric=args.metric,
        preferred=preferred,
    )
    if not args.only_binned:
        write_summary_tables(rows_by_object, features_by_object, output_dir)

        for object_name, object_rows in rows_by_object.items():
            plot_codebook_utilization(object_name, object_rows, output_dir)
        plot_combined_codebook_usage(rows_by_object, output_dir)

    if not args.skip_binned:
        device = choose_device(args.device)
        for object_name, object_rows in rows_by_object.items():
            for feature_name in features_by_object.get(object_name, []):
                plot_binned_feature_response(
                    object_name,
                    feature_name,
                    object_rows,
                    output_dir=output_dir,
                    h5_files=args.h5_files,
                    split=args.split,
                    max_valid_objects=args.max_valid_objects,
                    num_events_per_file=args.num_events_per_file,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    device=device,
                    legend_labels=legend_labels,
                )

    log.info("Wrote tokenizer summary plots to %s", output_dir)


if __name__ == "__main__":
    main()
