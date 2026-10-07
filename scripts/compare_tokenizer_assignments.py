#!/usr/bin/env python3
"""Compare two object tokenizers on the same objects.

This is intended for debugging MC-only vs MC+data VQ-VAE tokenizers.  Code IDs
from independently trained codebooks are not semantically aligned, so this
script compares assignment concentration and reconstruction error object by
object, rather than treating equal code IDs as equal physics meanings.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_vqvae_tokenizer import (  # noqa: E402
    analysis_datamodule_cfg,
    choose_device,
    codebook_counts,
    codebook_summary,
    dataloader_from_datamodule,
    dataset_kwargs_from_cfg,
    feature_names_from_cfg,
    find_checkpoint,
    safe_filename,
    to_device,
    transform_list_and_cst_fn_from_cfg,
)
from heptokens.data.atlas_event_mappable import AtlasEventMapDataset  # noqa: E402
from heptokens.models.vq_vae import LitVqVae  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


log = logging.getLogger(__name__)


DEFAULT_RUNS = {
    "jets": {
        "mc_only": "results/atlas_event_tokenizers_1606_jets_logstd_capacity_scan/jets_logstd_dim8_cb4096_q4",
        "mcdata": "results/atlas_event_tokenizers_0107_logstd_mc_realdata/jets_logstd_dim8_cb4096_q4",
    },
    "electrons": {
        "mc_only": "results/atlas_event_tokenizers_1806_nonjet_logstd_capacity_scan/electrons_logstd_dim8_cb4096_q4",
        "mcdata": "results/atlas_event_tokenizers_0107_logstd_mc_realdata/electrons_logstd_dim8_cb4096_q4",
    },
    "muons": {
        "mc_only": "results/atlas_event_tokenizers_1806_nonjet_logstd_capacity_scan/muons_logstd_dim8_cb4096_q4",
        "mcdata": "results/atlas_event_tokenizers_0107_logstd_mc_realdata/muons_logstd_dim8_cb4096_q4",
    },
    "photons": {
        "mc_only": "results/atlas_event_tokenizers_1806_nonjet_logstd_capacity_scan/photons_logstd_dim8_cb4096_q4",
        "mcdata": "results/atlas_event_tokenizers_0107_logstd_mc_realdata/photons_logstd_dim8_cb4096_q4",
    },
    "taus": {
        "mc_only": "results/atlas_event_tokenizers_1806_nonjet_logstd_capacity_scan/taus_logstd_dim8_cb4096_q4",
        "mcdata": "results/atlas_event_tokenizers_0107_logstd_mc_realdata/taus_logstd_dim8_cb4096_q4",
    },
    "tracks": {
        "mc_only": "results/atlas_event_tokenizers_1906_tracks_logstd_tests/tracks_logstd_no_ndoflog_dim8_cb8192_q4",
        "mcdata": "results/atlas_event_tokenizers_0107_logstd_mc_realdata/tracks_logstd_dim8_cb8192_q4",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare codebook assignments and reconstruction errors for two tokenizers."
    )
    parser.add_argument("--object", default="electrons", choices=sorted(DEFAULT_RUNS))
    parser.add_argument("--mc-only-run-dir", help="Override MC-only run directory.")
    parser.add_argument("--mcdata-run-dir", help="Override MC+data run directory.")
    parser.add_argument("--h5-files", nargs="+", help="Fixed H5 files to evaluate.")
    parser.add_argument("--mc-dir", default="/home/zephyr/Data/viviana/bnl-treasure/data/h5")
    parser.add_argument("--n-files", type=int, default=20)
    parser.add_argument(
        "--split",
        choices=["sequential", "train", "val", "test"],
        default="val",
        help=(
            "Use val/train/test to match the standard diagnostic plots. Sequential "
            "keeps the older manual full-file path and is mainly for debugging."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-events-per-file", type=int)
    parser.add_argument("--max-valid-objects", type=int, default=500_000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--features",
        default="pt,eta,ptvarcone30,topoetcone20,LHMedium,LHTight",
        help="Comma-separated feature names for per-object error comparison.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/fair_mc_vs_mcdata_eval/assignment_debug",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=14,
        help="Number of truth-feature bins for the plot-style IQR/median resolution metric.",
    )
    parser.add_argument(
        "--min-bin-count",
        type=int,
        default=50,
        help="Minimum objects required in a bin for the plot-style resolution metric.",
    )
    parser.add_argument(
        "--min-denominator",
        type=float,
        default=1e-8,
        help="Minimum |median(original)| denominator for the plot-style resolution metric.",
    )
    return parser.parse_args()


def fixed_files(directory: str, n_files: int) -> list[str]:
    files = sorted(
        str(path)
        for path in Path(directory).glob("*.h5")
        if path.is_file() and path.stat().st_size > 0
    )
    return files[:n_files]


def load_arrays(
    *,
    label: str,
    run_dir: Path,
    h5_files: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    cfg = OmegaConf.load(run_dir / "full_config.yaml")
    transforms, cst_inverse_transformer = transform_list_and_cst_fn_from_cfg(cfg)
    checkpoint = find_checkpoint(run_dir, None)
    log.info("Loading %s checkpoint: %s", label, checkpoint)
    model = LitVqVae.load_from_checkpoint(checkpoint, map_location=device)
    model.to(device)
    model.eval()

    loader_args = SimpleNamespace(
        h5_files=h5_files,
        num_events_per_file=args.num_events_per_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if args.split != "sequential":
        datamodule_cfg = analysis_datamodule_cfg(cfg, loader_args)
        datamodule = hydra.utils.instantiate(datamodule_cfg)
        loader = dataloader_from_datamodule(datamodule, args.split)
        original_std, reconstruction_std, code_indices, n_seen = collect_arrays_from_loader(
            model=model,
            loader=loader,
            cst_inverse_transformer=None,
            device=device,
            max_valid_objects=args.max_valid_objects,
        )
        if cst_inverse_transformer is not None:
            with np.errstate(over="ignore", invalid="ignore"):
                original = cst_inverse_transformer.inverse_transform(original_std)
                reconstruction = cst_inverse_transformer.inverse_transform(reconstruction_std)
        else:
            original = original_std.copy()
            reconstruction = reconstruction_std.copy()
        feature_names = feature_names_from_cfg(cfg, original_std.shape[1])
        return {
            "original": original,
            "reconstruction": reconstruction,
            "original_std": original_std,
            "reconstruction_std": reconstruction_std,
            "indices": code_indices,
            "feature_names": feature_names,
            "n_seen": n_seen,
        }

    data_paths, dataset_kwargs = dataset_kwargs_from_cfg(cfg, loader_args, run_dir)

    originals_std: list[np.ndarray] = []
    recons_std: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []
    n_seen = 0

    with torch.no_grad():
        for data_path in data_paths:
            if n_seen >= args.max_valid_objects:
                break
            log.info("Analyzing %s", data_path)
            dataset = AtlasEventMapDataset(data_path, **dataset_kwargs)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
            )
            for batch in loader:
                if n_seen >= args.max_valid_objects:
                    break
                for transform in transforms:
                    batch = transform(batch)
                batch = to_device(batch, device)

                z_q, indices, _ = model.encode(batch)
                reconstruction = model.decode(z_q, batch)

                mask = batch["mask"].bool()
                original_valid = batch["csts"][mask].detach().cpu().float().numpy()
                reconstruction_valid = reconstruction[mask].detach().cpu().float().numpy()
                indices_valid = indices[mask].detach().cpu().long().numpy()
                if len(original_valid) == 0:
                    continue

                remaining = args.max_valid_objects - n_seen
                if len(original_valid) > remaining:
                    original_valid = original_valid[:remaining]
                    reconstruction_valid = reconstruction_valid[:remaining]
                    indices_valid = indices_valid[:remaining]

                originals_std.append(original_valid)
                recons_std.append(reconstruction_valid)
                all_indices.append(indices_valid)
                n_seen += len(original_valid)

    if not originals_std:
        raise RuntimeError(f"{label}: no valid objects found")

    # Keep the inverse transform and residual arithmetic in float64.  Some
    # features are decoded in log space and then exponentiated; float32 can
    # overflow much earlier and make the diagnostic report artificial infs.
    original_std = np.concatenate(originals_std, axis=0).astype(np.float64, copy=False)
    reconstruction_std = np.concatenate(recons_std, axis=0).astype(np.float64, copy=False)
    code_indices = np.concatenate(all_indices, axis=0)

    if cst_inverse_transformer is not None:
        with np.errstate(over="ignore", invalid="ignore"):
            original = cst_inverse_transformer.inverse_transform(original_std)
            reconstruction = cst_inverse_transformer.inverse_transform(reconstruction_std)
    else:
        original = original_std.copy()
        reconstruction = reconstruction_std.copy()

    feature_names = feature_names_from_cfg(cfg, original_std.shape[1])
    return {
        "original": original,
        "reconstruction": reconstruction,
        "original_std": original_std,
        "reconstruction_std": reconstruction_std,
        "indices": code_indices,
        "feature_names": feature_names,
        "n_seen": n_seen,
    }


def collect_arrays_from_loader(
    *,
    model: LitVqVae,
    loader: DataLoader,
    cst_inverse_transformer,
    device: torch.device,
    max_valid_objects: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    originals = []
    recons = []
    all_indices = []
    n_seen = 0

    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            z_q, indices, _ = model.encode(batch)
            reconstruction = model.decode(z_q, batch)

            mask = batch["mask"].bool()
            original_valid = batch["csts"][mask].detach().cpu().double().numpy()
            reconstruction_valid = reconstruction[mask].detach().cpu().double().numpy()
            indices_valid = indices[mask].detach().cpu().long().numpy()

            if cst_inverse_transformer is not None:
                with np.errstate(over="ignore", invalid="ignore"):
                    original_valid = cst_inverse_transformer.inverse_transform(original_valid)
                    reconstruction_valid = cst_inverse_transformer.inverse_transform(
                        reconstruction_valid
                    )

            originals.append(original_valid)
            recons.append(reconstruction_valid)
            all_indices.append(indices_valid)
            n_seen += len(original_valid)
            if n_seen >= max_valid_objects:
                return (
                    np.concatenate(originals, axis=0)[:max_valid_objects],
                    np.concatenate(recons, axis=0)[:max_valid_objects],
                    np.concatenate(all_indices, axis=0)[:max_valid_objects],
                    n_seen,
                )

    if not originals:
        raise RuntimeError("No valid objects found. Check the object mask and input paths.")
    return (
        np.concatenate(originals, axis=0),
        np.concatenate(recons, axis=0),
        np.concatenate(all_indices, axis=0),
        n_seen,
    )


def entropy_perplexity(counts: np.ndarray) -> tuple[float, float]:
    total = counts.sum()
    if total <= 0:
        return 0.0, 0.0
    p = counts[counts > 0] / total
    entropy = float(-np.sum(p * np.log(p)))
    return entropy, float(np.exp(entropy))


def top_fraction(counts: np.ndarray, top_k: int) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    return float(np.sort(counts)[-top_k:].sum() / total)


def tuple_summary(indices: np.ndarray) -> dict:
    if len(indices) == 0:
        return {
            "assignments": 0,
            "unique_tuples": 0,
            "percent_unique_tuples": 0.0,
            "top1_tuple_fraction": 0.0,
            "top10_tuple_fraction": 0.0,
        }
    tuples, counts = np.unique(indices, axis=0, return_counts=True)
    order = np.argsort(counts)[::-1]
    sorted_counts = counts[order]
    total = int(counts.sum())
    return {
        "assignments": total,
        "unique_tuples": int(len(tuples)),
        "percent_unique_tuples": float(100.0 * len(tuples) / total),
        "top1_tuple_fraction": float(sorted_counts[0] / total),
        "top10_tuple_fraction": float(sorted_counts[:10].sum() / total),
        "top_tuples": [
            {
                "rank": rank + 1,
                "tuple": [int(value) for value in tuples[order[rank]].tolist()],
                "count": int(sorted_counts[rank]),
                "fraction": float(sorted_counts[rank] / total),
            }
            for rank in range(min(10, len(sorted_counts)))
        ],
    }


def write_usage_csv(path: Path, label: str, counts: np.ndarray) -> list[dict]:
    rows = []
    for q_idx, q_counts in enumerate(counts):
        entropy, perplexity = entropy_perplexity(q_counts)
        used = int(np.count_nonzero(q_counts))
        row = {
            "model": label,
            "quantizer": q_idx,
            "assignments": int(q_counts.sum()),
            "used_codes": used,
            "percent_used": 100.0 * used / len(q_counts),
            "perplexity": perplexity,
            "top1_fraction": top_fraction(q_counts, 1),
            "top10_fraction": top_fraction(q_counts, 10),
            "max_frequency": int(q_counts.max()),
        }
        rows.append(row)

    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
    return rows


def write_top_codes_csv(path: Path, label: str, counts: np.ndarray, top_k: int = 20) -> None:
    rows = []
    for q_idx, q_counts in enumerate(counts):
        order = np.argsort(q_counts)[::-1][:top_k]
        total = q_counts.sum()
        for rank, code in enumerate(order, start=1):
            rows.append(
                {
                    "model": label,
                    "quantizer": q_idx,
                    "rank": rank,
                    "code": int(code),
                    "count": int(q_counts[code]),
                    "fraction": float(q_counts[code] / total) if total else 0.0,
                }
            )

    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def write_feature_metrics_csv(
    path: Path,
    label: str,
    original: np.ndarray,
    reconstruction: np.ndarray,
    feature_names: list[str],
) -> dict:
    metrics = robust_metrics(original, reconstruction, feature_names)
    rows = []
    for name, values in metrics.items():
        rows.append(
            {
                "model": label,
                "feature": name,
                "n": values["n"],
                "n_nonfinite": values["n_nonfinite"],
                "mean_original": values["mean_original"],
                "mean_reconstructed": values["mean_reconstructed"],
                "mae": values["mae"],
                "rmse": values["rmse"],
                "bias": values["bias"],
                "std": values["std"],
                "median_abs": values["median_abs"],
                "p99_abs": values["p99_abs"],
            }
        )

    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
    return metrics


def feature_index(feature_names: list[str], feature: str) -> int | None:
    target = feature.lower()
    for idx, name in enumerate(feature_names):
        if name.lower() == target:
            return idx
    return None


def write_error_comparison_csv(
    path: Path,
    features: list[str],
    comparisons: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    feature_names: list[str],
) -> list[dict]:
    rows = []
    for space, original_old, reco_old, original_new, reco_new in comparisons:
        for feature in features:
            idx = feature_index(feature_names, feature)
            if idx is None:
                continue
            old_residual = reco_old[:, idx].astype(np.float64) - original_old[:, idx].astype(np.float64)
            new_residual = reco_new[:, idx].astype(np.float64) - original_new[:, idx].astype(np.float64)
            old_abs_all = np.abs(old_residual)
            new_abs_all = np.abs(new_residual)
            finite = np.isfinite(old_abs_all) & np.isfinite(new_abs_all)
            old_abs = old_abs_all[finite]
            new_abs = new_abs_all[finite]
            delta = new_abs - old_abs
            row = {
                "space": space,
                "feature": feature,
                "n_total": int(len(old_abs_all)),
                "n_finite_pair": int(np.count_nonzero(finite)),
                "mc_only_nonfinite_abs_error": int(np.count_nonzero(~np.isfinite(old_abs_all))),
                "mcdata_nonfinite_abs_error": int(np.count_nonzero(~np.isfinite(new_abs_all))),
            }
            if len(delta) == 0:
                row.update(
                    {
                        "mc_only_mae": float("nan"),
                        "mcdata_mae": float("nan"),
                        "mcdata_minus_mc_only_mae": float("nan"),
                        "mc_only_median_abs_error": float("nan"),
                        "mcdata_median_abs_error": float("nan"),
                        "mc_only_p99_abs_error": float("nan"),
                        "mcdata_p99_abs_error": float("nan"),
                        "fraction_mcdata_better": float("nan"),
                        "fraction_mc_only_better": float("nan"),
                        "median_delta_abs_error": float("nan"),
                        "p90_delta_abs_error": float("nan"),
                    }
                )
                rows.append(row)
                continue
            row.update(
                {
                    "mc_only_mae": float(np.mean(old_abs)),
                    "mcdata_mae": float(np.mean(new_abs)),
                    "mcdata_minus_mc_only_mae": float(np.mean(delta)),
                    "mc_only_median_abs_error": float(np.median(old_abs)),
                    "mcdata_median_abs_error": float(np.median(new_abs)),
                    "mc_only_p99_abs_error": float(np.percentile(old_abs, 99)),
                    "mcdata_p99_abs_error": float(np.percentile(new_abs, 99)),
                    "fraction_mcdata_better": float(np.mean(new_abs < old_abs)),
                    "fraction_mc_only_better": float(np.mean(old_abs < new_abs)),
                    "median_delta_abs_error": float(np.median(delta)),
                    "p90_delta_abs_error": float(np.percentile(delta, 90)),
                }
            )
            rows.append(row)

    if not rows:
        with path.open("w", newline="") as handle:
            handle.write("")
        return rows

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_nonfinite_summary_csv(
    path: Path,
    arrays: list[tuple[str, str, np.ndarray]],
    feature_names: list[str],
) -> list[dict]:
    rows = []
    for model, array_name, values in arrays:
        for idx, feature in enumerate(feature_names):
            feature_values = values[:, idx].astype(np.float64, copy=False)
            finite = np.isfinite(feature_values)
            finite_values = feature_values[finite]
            rows.append(
                {
                    "model": model,
                    "array": array_name,
                    "feature": feature,
                    "n": int(len(feature_values)),
                    "n_nonfinite": int(np.count_nonzero(~finite)),
                    "fraction_nonfinite": float(np.mean(~finite)) if len(feature_values) else 0.0,
                    "min_finite": float(np.min(finite_values)) if len(finite_values) else float("nan"),
                    "max_finite": float(np.max(finite_values)) if len(finite_values) else float("nan"),
                    "p99_abs_finite": float(np.percentile(np.abs(finite_values), 99))
                    if len(finite_values)
                    else float("nan"),
                }
            )

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def make_bins(values: np.ndarray, n_bins: int) -> np.ndarray | None:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return None
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if lo == hi:
        return None
    return np.linspace(lo, hi, n_bins + 1)


def binned_iqr_resolution_rows(
    *,
    model: str,
    feature: str,
    original: np.ndarray,
    reconstruction: np.ndarray,
    bins: np.ndarray,
    min_bin_count: int,
    min_denominator: float,
) -> list[dict]:
    rows = []
    for bin_idx in range(len(bins) - 1):
        in_bin = (original >= bins[bin_idx]) & (original < bins[bin_idx + 1])
        finite = in_bin & np.isfinite(original) & np.isfinite(reconstruction)
        count = int(np.count_nonzero(finite))
        row = {
            "model": model,
            "feature": feature,
            "bin": bin_idx,
            "bin_low": float(bins[bin_idx]),
            "bin_high": float(bins[bin_idx + 1]),
            "bin_center": float(0.5 * (bins[bin_idx] + bins[bin_idx + 1])),
            "n": count,
            "median_original": float("nan"),
            "iqr_residual": float("nan"),
            "iqr_over_abs_median_original": float("nan"),
        }
        if count >= min_bin_count:
            truth = original[finite].astype(np.float64)
            reco = reconstruction[finite].astype(np.float64)
            residual = reco - truth
            median_original = float(np.median(truth))
            denom = abs(median_original)
            if denom >= min_denominator:
                q25, q75 = np.percentile(residual, [25, 75])
                iqr = float(q75 - q25)
                row.update(
                    {
                        "median_original": median_original,
                        "iqr_residual": iqr,
                        "iqr_over_abs_median_original": float(iqr / denom),
                    }
                )
        rows.append(row)
    return rows


def write_binned_resolution_csvs(
    *,
    output_dir: Path,
    features: list[str],
    old_original: np.ndarray,
    old_reco: np.ndarray,
    new_original: np.ndarray,
    new_reco: np.ndarray,
    feature_names: list[str],
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict]]:
    rows = []
    summary_rows = []

    for feature in features:
        idx = feature_index(feature_names, feature)
        if idx is None:
            continue

        # Same convention as the resolution plots: define truth bins from the
        # fixed sample's original feature values, then evaluate each tokenizer
        # on those same bins.
        bins = make_bins(old_original[:, idx].astype(np.float64), args.n_bins)
        if bins is None:
            continue

        model_rows = {}
        for model, original, reco in [
            ("MC-only", old_original[:, idx], old_reco[:, idx]),
            ("MC+data", new_original[:, idx], new_reco[:, idx]),
        ]:
            current_rows = binned_iqr_resolution_rows(
                model=model,
                feature=feature,
                original=original.astype(np.float64),
                reconstruction=reco.astype(np.float64),
                bins=bins,
                min_bin_count=args.min_bin_count,
                min_denominator=args.min_denominator,
            )
            rows.extend(current_rows)
            model_rows[model] = current_rows

            finite_rows = [
                row
                for row in current_rows
                if np.isfinite(row["iqr_over_abs_median_original"])
            ]
            values = np.asarray(
                [row["iqr_over_abs_median_original"] for row in finite_rows],
                dtype=np.float64,
            )
            counts = np.asarray([row["n"] for row in finite_rows], dtype=np.float64)
            summary_rows.append(
                {
                    "feature": feature,
                    "model": model,
                    "n_bins_used": int(len(values)),
                    "mean_binned_resolution": float(np.mean(values)) if len(values) else float("nan"),
                    "weighted_mean_binned_resolution": float(np.average(values, weights=counts))
                    if len(values) and np.sum(counts) > 0
                    else float("nan"),
                    "median_binned_resolution": float(np.median(values)) if len(values) else float("nan"),
                }
            )

    if rows:
        with (output_dir / "binned_iqr_resolution.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    if summary_rows:
        with (output_dir / "binned_iqr_resolution_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)

    return rows, summary_rows


def scaled_rmse(residual: np.ndarray) -> float:
    finite = residual[np.isfinite(residual)]
    if len(finite) == 0:
        return float("nan")
    max_abs = float(np.max(np.abs(finite)))
    if max_abs == 0.0:
        return 0.0
    scaled = finite / max_abs
    return float(max_abs * np.sqrt(np.mean(scaled**2)))


def finite_delta_summary(old: np.ndarray, new: np.ndarray) -> dict:
    finite = np.isfinite(old) & np.isfinite(new)
    if not np.any(finite):
        return {
            "n_total": int(old.size),
            "n_finite_pair": 0,
            "n_nonfinite_pair": int(old.size),
            "max_abs": float("nan"),
            "mean_abs": float("nan"),
        }
    delta = np.abs(old[finite].astype(np.float64) - new[finite].astype(np.float64))
    return {
        "n_total": int(old.size),
        "n_finite_pair": int(np.count_nonzero(finite)),
        "n_nonfinite_pair": int(old.size - np.count_nonzero(finite)),
        "max_abs": float(np.max(delta)),
        "mean_abs": float(np.mean(delta)),
    }


def robust_metrics(original: np.ndarray, reconstruction: np.ndarray, feature_names: list[str]) -> dict:
    out = {}
    for idx, name in enumerate(feature_names):
        orig = original[:, idx].astype(np.float64)
        reco = reconstruction[:, idx].astype(np.float64)
        finite = np.isfinite(orig) & np.isfinite(reco)
        residual = (reco[finite] - orig[finite]).astype(np.float64)
        if len(residual) == 0:
            out[name] = {
                "n": 0,
                "n_nonfinite": int(len(orig)),
                "mean_original": float("nan"),
                "mean_reconstructed": float("nan"),
                "mae": float("nan"),
                "rmse": float("nan"),
                "bias": float("nan"),
                "std": float("nan"),
                "median_abs": float("nan"),
                "p99_abs": float("nan"),
            }
            continue
        abs_residual = np.abs(residual)
        max_abs = float(np.max(abs_residual))
        scaled_residual = residual / max_abs if max_abs > 0 else residual
        out[name] = {
            "n": int(len(residual)),
            "n_nonfinite": int(np.count_nonzero(~finite)),
            "mean_original": float(np.mean(orig[finite])),
            "mean_reconstructed": float(np.mean(reco[finite])),
            "mae": float(np.mean(abs_residual)),
            "rmse": scaled_rmse(residual),
            "bias": float(np.mean(residual)),
            "std": float(max_abs * np.std(scaled_residual)) if max_abs > 0 else 0.0,
            "median_abs": float(np.median(abs_residual)),
            "p99_abs": float(np.percentile(abs_residual, 99)),
        }
    return out


def plot_cumulative_usage(output_path: Path, old_counts: np.ndarray, new_counts: np.ndarray) -> None:
    n_quantizers = old_counts.shape[0]
    fig, axes = plt.subplots(1, n_quantizers, figsize=(4.0 * n_quantizers, 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for q_idx, ax in enumerate(axes):
        for label, counts, color in [
            ("MC-only", old_counts[q_idx], "#4C83F1"),
            ("MC+data", new_counts[q_idx], "#FF9F1C"),
        ]:
            sorted_counts = np.sort(counts)[::-1]
            total = sorted_counts.sum()
            if total <= 0:
                continue
            cumulative = np.cumsum(sorted_counts) / total
            ax.plot(
                np.arange(1, len(cumulative) + 1),
                cumulative,
                label=label,
                color=color,
                linewidth=2,
            )
        ax.set_title(f"q{q_idx}")
        ax.set_xlabel("top N codes")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("fraction of assignments")
    axes[-1].legend(frameon=False)
    fig.suptitle("Code concentration")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_error_scatter(
    output_dir: Path,
    features: list[str],
    original_old: np.ndarray,
    reco_old: np.ndarray,
    original_new: np.ndarray,
    reco_new: np.ndarray,
    feature_names: list[str],
) -> None:
    for feature in features:
        idx = feature_index(feature_names, feature)
        if idx is None:
            continue
        old_abs = np.abs(
            reco_old[:, idx].astype(np.float64) - original_old[:, idx].astype(np.float64)
        )
        new_abs = np.abs(
            reco_new[:, idx].astype(np.float64) - original_new[:, idx].astype(np.float64)
        )
        finite = np.isfinite(old_abs) & np.isfinite(new_abs)
        old_abs = old_abs[finite]
        new_abs = new_abs[finite]
        if len(old_abs) == 0:
            continue

        sample = np.linspace(0, len(old_abs) - 1, min(len(old_abs), 50_000)).astype(int)
        hi = np.percentile(np.concatenate([old_abs[sample], new_abs[sample]]), 99.0)
        hi = max(float(hi), 1e-12)

        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        ax.scatter(old_abs[sample], new_abs[sample], s=2, alpha=0.15)
        ax.plot([0, hi], [0, hi], color="black", linewidth=1.2)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_xlabel("MC-only abs error")
        ax.set_ylabel("MC+data abs error")
        ax.set_title(f"{feature}: same-object error comparison")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / f"{safe_filename(feature)}_same_object_error_scatter.png", dpi=180)
        plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    device = choose_device(args.device)

    old_run = Path(args.mc_only_run_dir or DEFAULT_RUNS[args.object]["mc_only"]).resolve()
    new_run = Path(args.mcdata_run_dir or DEFAULT_RUNS[args.object]["mcdata"]).resolve()
    h5_files = list(args.h5_files) if args.h5_files else fixed_files(args.mc_dir, args.n_files)
    if not h5_files:
        raise FileNotFoundError("No H5 files selected")

    output_dir = Path(args.output_dir) / args.object
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "files_used.txt").write_text("\n".join(h5_files) + "\n")

    old_arrays = load_arrays(
        label="MC-only",
        run_dir=old_run,
        h5_files=h5_files,
        args=args,
        device=device,
    )
    new_arrays = load_arrays(
        label="MC+data",
        run_dir=new_run,
        h5_files=h5_files,
        args=args,
        device=device,
    )

    n = min(
        len(old_arrays["original"]),
        len(new_arrays["original"]),
        len(old_arrays["indices"]),
        len(new_arrays["indices"]),
    )
    for arrays in [old_arrays, new_arrays]:
        for key in ["original", "reconstruction", "original_std", "reconstruction_std", "indices"]:
            arrays[key] = arrays[key][:n]

    old_features = old_arrays["feature_names"]
    new_features = new_arrays["feature_names"]
    if [name.lower() for name in old_features] != [name.lower() for name in new_features]:
        log.warning("Feature names differ: MC-only=%s MC+data=%s", old_features, new_features)

    physical_alignment = finite_delta_summary(old_arrays["original"], new_arrays["original"])
    standardized_alignment = finite_delta_summary(old_arrays["original_std"], new_arrays["original_std"])
    original_match = {
        "physical": physical_alignment,
        "standardized": standardized_alignment,
    }

    codebook_size = int(OmegaConf.load(old_run / "full_config.yaml").model.codebook_size)
    old_counts = codebook_counts(old_arrays["indices"], codebook_size)
    new_counts = codebook_counts(new_arrays["indices"], codebook_size)

    for filename in [
        "usage_summary.csv",
        "top_codes.csv",
        "feature_metrics.csv",
        "same_object_error_comparison.csv",
        "nonfinite_summary.csv",
        "binned_iqr_resolution.csv",
        "binned_iqr_resolution_summary.csv",
    ]:
        path = output_dir / filename
        if path.exists():
            path.unlink()

    old_usage_rows = write_usage_csv(output_dir / "usage_summary.csv", "MC-only", old_counts)
    new_usage_rows = write_usage_csv(output_dir / "usage_summary.csv", "MC+data", new_counts)
    write_top_codes_csv(output_dir / "top_codes.csv", "MC-only", old_counts)
    write_top_codes_csv(output_dir / "top_codes.csv", "MC+data", new_counts)
    old_metrics = write_feature_metrics_csv(
        output_dir / "feature_metrics.csv",
        "MC-only",
        old_arrays["original"],
        old_arrays["reconstruction"],
        old_features,
    )
    new_metrics = write_feature_metrics_csv(
        output_dir / "feature_metrics.csv",
        "MC+data",
        new_arrays["original"],
        new_arrays["reconstruction"],
        old_features,
    )
    old_metrics_std = robust_metrics(
        old_arrays["original_std"],
        old_arrays["reconstruction_std"],
        old_features,
    )
    new_metrics_std = robust_metrics(
        new_arrays["original_std"],
        new_arrays["reconstruction_std"],
        old_features,
    )
    nonfinite_rows = write_nonfinite_summary_csv(
        output_dir / "nonfinite_summary.csv",
        [
            ("MC-only", "original_physical", old_arrays["original"]),
            ("MC-only", "reconstruction_physical", old_arrays["reconstruction"]),
            ("MC-only", "original_standardized", old_arrays["original_std"]),
            ("MC-only", "reconstruction_standardized", old_arrays["reconstruction_std"]),
            ("MC+data", "original_physical", new_arrays["original"]),
            ("MC+data", "reconstruction_physical", new_arrays["reconstruction"]),
            ("MC+data", "original_standardized", new_arrays["original_std"]),
            ("MC+data", "reconstruction_standardized", new_arrays["reconstruction_std"]),
        ],
        old_features,
    )

    features = [feature.strip() for feature in args.features.split(",") if feature.strip()]
    error_rows = write_error_comparison_csv(
        output_dir / "same_object_error_comparison.csv",
        features,
        [
            (
                "physical",
                old_arrays["original"],
                old_arrays["reconstruction"],
                new_arrays["original"],
                new_arrays["reconstruction"],
            ),
            (
                "standardized",
                old_arrays["original_std"],
                old_arrays["reconstruction_std"],
                new_arrays["original_std"],
                new_arrays["reconstruction_std"],
            ),
        ],
        old_features,
    )
    binned_rows, binned_summary_rows = write_binned_resolution_csvs(
        output_dir=output_dir,
        features=features,
        old_original=old_arrays["original"],
        old_reco=old_arrays["reconstruction"],
        new_original=new_arrays["original"],
        new_reco=new_arrays["reconstruction"],
        feature_names=old_features,
        args=args,
    )

    plot_cumulative_usage(output_dir / "cumulative_code_usage.png", old_counts, new_counts)
    plot_error_scatter(
        output_dir,
        features,
        old_arrays["original"],
        old_arrays["reconstruction"],
        new_arrays["original"],
        new_arrays["reconstruction"],
        old_features,
    )

    summary = {
        "object": args.object,
        "n_compared_objects": int(n),
        "feature_names": old_features,
        "mc_only_seen_before_cap": int(old_arrays["n_seen"]),
        "mcdata_seen_before_cap": int(new_arrays["n_seen"]),
        "mc_only_run": str(old_run),
        "mcdata_run": str(new_run),
        "files_used": h5_files,
        "original_alignment": original_match,
        "mc_only_codebook": codebook_summary(old_counts),
        "mcdata_codebook": codebook_summary(new_counts),
        "mc_only_tuple_summary": tuple_summary(old_arrays["indices"]),
        "mcdata_tuple_summary": tuple_summary(new_arrays["indices"]),
        "same_object_error_comparison": error_rows,
        "mc_only_metrics": old_metrics,
        "mcdata_metrics": new_metrics,
        "mc_only_standardized_metrics": old_metrics_std,
        "mcdata_standardized_metrics": new_metrics_std,
        "nonfinite_summary": nonfinite_rows,
        "binned_iqr_resolution_summary": binned_summary_rows,
        "usage_rows": old_usage_rows + new_usage_rows,
    }
    (output_dir / "assignment_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )

    print(f"Wrote {output_dir}")
    print(json.dumps({"n_compared_objects": int(n), **original_match}, indent=2))
    print("\nFeature order:")
    print("  " + ", ".join(old_features))
    print("\nSame-object physical-space error comparison:")
    for row in [row for row in error_rows if row["space"] == "physical"]:
        print(
            f"  {row['feature']}: finite={row['n_finite_pair']}/{row['n_total']}, "
            f"MC-only MAE={row['mc_only_mae']:.5g}, MC+data MAE={row['mcdata_mae']:.5g}, "
            f"MC+data better={row['fraction_mcdata_better']:.3f}"
        )
    print("\nSame-object standardized-space error comparison:")
    for row in [row for row in error_rows if row["space"] == "standardized"]:
        print(
            f"  {row['feature']}: finite={row['n_finite_pair']}/{row['n_total']}, "
            f"MC-only MAE={row['mc_only_mae']:.5g}, MC+data MAE={row['mcdata_mae']:.5g}, "
            f"MC+data better={row['fraction_mcdata_better']:.3f}"
        )
    print("\nPlot-style binned IQR/median resolution:")
    by_feature = {}
    for row in binned_summary_rows:
        by_feature.setdefault(row["feature"], {})[row["model"]] = row
    for feature, items in by_feature.items():
        old_row = items.get("MC-only")
        new_row = items.get("MC+data")
        if old_row is None or new_row is None:
            continue
        old_value = old_row["mean_binned_resolution"]
        new_value = new_row["mean_binned_resolution"]
        if np.isfinite(old_value) and np.isfinite(new_value):
            better = "MC-only" if old_value < new_value else "MC+data"
        else:
            better = "-"
        print(
            f"  {feature}: MC-only={old_value:.5g}, MC+data={new_value:.5g}, better={better}"
        )


if __name__ == "__main__":
    main()
