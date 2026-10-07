#!/usr/bin/env python3
"""Compare tokenizer code usage in bins of original object pT.

This diagnostic is meant for independent MC-only vs MC+data VQ-VAE codebooks.
Code IDs are not aligned across separately trained models, so the useful
question is not whether the same code number is used, but how many distinct
codes / code tuples each tokenizer uses in the same pT region.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from omegaconf import OmegaConf

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_vqvae_tokenizer import (  # noqa: E402
    apply_hep_style,
    choose_device,
    collect_diagnostics_for_h5_files,
    feature_names_from_cfg,
    find_checkpoint,
    safe_filename,
    transform_list_and_cst_fn_from_cfg,
)
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


COLORS = {
    "MC-only": "#4C83F1",
    "MC+data": "#FF9F1C",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MC-only vs MC+data code usage in original pT bins."
    )
    parser.add_argument("--object", default="electrons", choices=sorted(DEFAULT_RUNS))
    parser.add_argument("--mc-only-run-dir", help="Override MC-only run directory.")
    parser.add_argument("--mcdata-run-dir", help="Override MC+data run directory.")
    parser.add_argument("--mc-dir", default="/home/zephyr/Data/viviana/bnl-treasure/data/h5")
    parser.add_argument(
        "--data-dir",
        default="/home/zephyr/Data/viviana/bnl-treasure/data/h5/realdata",
    )
    parser.add_argument(
        "--sample",
        choices=["mc", "data", "mixed"],
        default="mc",
        help="Which fixed H5 sample to evaluate. Default is MC.",
    )
    parser.add_argument("--h5-files", nargs="+", help="Explicit fixed H5 files to use.")
    parser.add_argument("--n-files", type=int, default=20)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-events-per-file", type=int)
    parser.add_argument("--max-valid-objects", type=int, default=500_000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--n-bins", type=int, default=12)
    parser.add_argument("--min-bin-count", type=int, default=50)
    parser.add_argument(
        "--pt-percentile-range",
        default="1,99",
        help="Percentile range used for pT bin edges, e.g. 1,99.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/fair_mc_vs_mcdata_eval/code_usage_by_pt",
    )
    return parser.parse_args()


def fixed_files(directory: str, n_files: int) -> list[str]:
    return sorted(
        str(path)
        for path in Path(directory).glob("*.h5")
        if path.is_file() and path.stat().st_size > 0
    )[:n_files]


def select_h5_files(args: argparse.Namespace) -> list[str]:
    if args.h5_files:
        return list(args.h5_files)
    if args.sample == "mc":
        return fixed_files(args.mc_dir, args.n_files)
    if args.sample == "data":
        return fixed_files(args.data_dir, args.n_files)
    mc_files = fixed_files(args.mc_dir, args.n_files)
    data_files = fixed_files(args.data_dir, args.n_files)
    return sorted(mc_files + data_files)


def feature_index(feature_names: list[str], feature_name: str) -> int:
    target = feature_name.lower()
    for idx, name in enumerate(feature_names):
        if name.lower() == target:
            return idx
    raise ValueError(f"Feature {feature_name!r} not found in {feature_names}")


def collect_original_and_indices(
    *,
    run_dir: Path,
    h5_files: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    cfg = OmegaConf.load(run_dir / "full_config.yaml")
    _, cst_inverse_transformer = transform_list_and_cst_fn_from_cfg(cfg)
    checkpoint = find_checkpoint(run_dir, None)
    log.info("Loading %s", checkpoint)

    model = LitVqVae.load_from_checkpoint(checkpoint, map_location=device)
    model.to(device)
    model.eval()

    original_std, _, code_indices, _ = collect_diagnostics_for_h5_files(
        cfg=cfg,
        model=model,
        h5_files=h5_files,
        split=args.split,
        num_events_per_file=args.num_events_per_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cst_inverse_transformer=None,
        device=device,
        max_valid_objects=args.max_valid_objects,
    )

    if cst_inverse_transformer is not None:
        original = cst_inverse_transformer.inverse_transform(original_std)
    else:
        original = original_std.copy()

    return {
        "original": original,
        "original_std": original_std,
        "indices": code_indices,
        "feature_names": feature_names_from_cfg(cfg, original_std.shape[1]),
        "codebook_size": int(cfg.model.codebook_size),
        "num_quantizers": int(cfg.model.num_quantizers),
    }


def entropy_perplexity(counts: np.ndarray) -> tuple[float, float]:
    total = int(counts.sum())
    if total <= 0:
        return 0.0, 0.0
    probabilities = counts[counts > 0].astype(np.float64) / total
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    return entropy, float(np.exp(entropy))


def tuple_counts(indices: np.ndarray) -> np.ndarray:
    if len(indices) == 0:
        return np.asarray([], dtype=np.int64)
    _, counts = np.unique(indices, axis=0, return_counts=True)
    return counts


def top_fraction(counts: np.ndarray, top_k: int) -> float:
    total = int(counts.sum())
    if total <= 0:
        return 0.0
    return float(np.sort(counts)[-top_k:].sum() / total)


def summarize_bin(
    *,
    label: str,
    indices: np.ndarray,
    codebook_size: int,
    bin_idx: int,
    bin_low: float,
    bin_high: float,
    bin_center: float,
) -> dict:
    n_objects = int(len(indices))
    row = {
        "bin": bin_idx,
        "pt_low": bin_low,
        "pt_high": bin_high,
        "pt_center": bin_center,
        "model": label,
        "n_objects": n_objects,
    }

    t_counts = tuple_counts(indices)
    _, tuple_perplexity = entropy_perplexity(t_counts)
    row.update(
        {
            "unique_code_tuples": int(len(t_counts)),
            "unique_code_tuples_per_1000_objects": (
                float(1000.0 * len(t_counts) / n_objects) if n_objects else 0.0
            ),
            "tuple_perplexity": tuple_perplexity,
            "tuple_perplexity_per_1000_objects": (
                float(1000.0 * tuple_perplexity / n_objects) if n_objects else 0.0
            ),
            "top1_tuple_fraction": top_fraction(t_counts, 1),
            "top10_tuple_fraction": top_fraction(t_counts, 10),
        }
    )

    for q_idx in range(indices.shape[1]):
        q_counts = np.bincount(indices[:, q_idx], minlength=codebook_size)[:codebook_size]
        _, q_perplexity = entropy_perplexity(q_counts)
        q_used = int(np.count_nonzero(q_counts))
        row[f"q{q_idx}_used_codes"] = q_used
        row[f"q{q_idx}_percent_used"] = float(100.0 * q_used / codebook_size)
        row[f"q{q_idx}_perplexity"] = q_perplexity
        row[f"q{q_idx}_top1_fraction"] = top_fraction(q_counts, 1)
        row[f"q{q_idx}_top10_fraction"] = top_fraction(q_counts, 10)
    return row


def make_pt_bins(pt: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    finite_pt = pt[np.isfinite(pt)]
    if len(finite_pt) == 0:
        raise RuntimeError("No finite pT values found")
    lo_pct, hi_pct = [float(value) for value in args.pt_percentile_range.split(",", 1)]
    lo, hi = np.percentile(finite_pt, [lo_pct, hi_pct])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.min(finite_pt)), float(np.max(finite_pt))
    if lo == hi:
        raise RuntimeError("Could not build pT bins because pT has no range")
    return np.linspace(lo, hi, args.n_bins + 1)


def plot_metric(
    rows: list[dict],
    metric: str,
    ylabel: str,
    output_path: Path,
    *,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for label in ["MC-only", "MC+data"]:
        selected = [row for row in rows if row["model"] == label]
        x = np.asarray([row["pt_center"] for row in selected], dtype=np.float64)
        y = np.asarray([row[metric] for row in selected], dtype=np.float64)
        ax.plot(x, y, marker="o", linewidth=2.0, color=COLORS[label], label=label)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Original object pT", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=12)
    apply_hep_style(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_quantizer_grid(
    rows: list[dict],
    *,
    num_quantizers: int,
    metric_suffix: str,
    ylabel: str,
    output_path: Path,
    title: str,
) -> None:
    ncols = min(2, num_quantizers)
    nrows = (num_quantizers + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.0 * ncols, 4.5 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()

    for q_idx in range(num_quantizers):
        ax = flat_axes[q_idx]
        metric = f"q{q_idx}_{metric_suffix}"
        for label in ["MC-only", "MC+data"]:
            selected = [row for row in rows if row["model"] == label]
            x = np.asarray([row["pt_center"] for row in selected], dtype=np.float64)
            y = np.asarray([row[metric] for row in selected], dtype=np.float64)
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.0,
                color=COLORS[label],
                label=label,
            )
        ax.set_title(f"Quantizer {q_idx}", fontsize=14)
        ax.set_xlabel("Original object pT", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(alpha=0.25)
        apply_hep_style(ax)

    for ax in flat_axes[num_quantizers:]:
        ax.set_visible(False)

    flat_axes[0].legend(frameon=False, fontsize=11)
    fig.suptitle(title, fontsize=17)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    output_dir = Path(args.output_dir).resolve() / safe_filename(args.object) / args.sample
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_files = select_h5_files(args)
    if not h5_files:
        raise FileNotFoundError("No H5 files selected")
    (output_dir / "h5_files_used.txt").write_text("\n".join(h5_files) + "\n")
    log.info("Using %d H5 files", len(h5_files))

    run_dirs = {
        "MC-only": Path(args.mc_only_run_dir or DEFAULT_RUNS[args.object]["mc_only"]).resolve(),
        "MC+data": Path(args.mcdata_run_dir or DEFAULT_RUNS[args.object]["mcdata"]).resolve(),
    }
    for label, run_dir in run_dirs.items():
        if not (run_dir / "full_config.yaml").exists():
            raise FileNotFoundError(f"{label} run is missing full_config.yaml: {run_dir}")

    device = choose_device(args.device)
    arrays = {
        label: collect_original_and_indices(
            run_dir=run_dir,
            h5_files=h5_files,
            args=args,
            device=device,
        )
        for label, run_dir in run_dirs.items()
    }

    n_common = min(len(arrays["MC-only"]["indices"]), len(arrays["MC+data"]["indices"]))
    for label in arrays:
        for key in ["original", "original_std", "indices"]:
            arrays[label][key] = arrays[label][key][:n_common]

    feature_names = arrays["MC-only"]["feature_names"]
    pt_idx = feature_index(feature_names, "pt")
    pt_idx_other = feature_index(arrays["MC+data"]["feature_names"], "pt")
    pt = arrays["MC-only"]["original"][:, pt_idx].astype(np.float64)
    other_pt = arrays["MC+data"]["original"][:, pt_idx_other].astype(np.float64)
    finite_pair = np.isfinite(pt) & np.isfinite(other_pt)
    max_abs_pt_delta = (
        float(np.max(np.abs(pt[finite_pair] - other_pt[finite_pair])))
        if np.any(finite_pair)
        else float("nan")
    )
    log.info("Max absolute pT difference between tokenizer inputs: %.6g", max_abs_pt_delta)

    bins = make_pt_bins(pt, args)
    rows = []
    for bin_idx in range(len(bins) - 1):
        in_bin = (pt >= bins[bin_idx]) & (pt < bins[bin_idx + 1])
        if bin_idx == len(bins) - 2:
            in_bin = (pt >= bins[bin_idx]) & (pt <= bins[bin_idx + 1])
        if int(np.count_nonzero(in_bin)) < args.min_bin_count:
            continue
        for label in ["MC-only", "MC+data"]:
            rows.append(
                summarize_bin(
                    label=label,
                    indices=arrays[label]["indices"][in_bin],
                    codebook_size=arrays[label]["codebook_size"],
                    bin_idx=bin_idx,
                    bin_low=float(bins[bin_idx]),
                    bin_high=float(bins[bin_idx + 1]),
                    bin_center=float(0.5 * (bins[bin_idx] + bins[bin_idx + 1])),
                )
            )

    if not rows:
        raise RuntimeError("No pT bins passed the minimum count requirement")

    csv_path = output_dir / "code_usage_by_pt_bin.csv"
    with csv_path.open("w", newline="") as handle:
        fieldnames = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "object": args.object,
        "sample": args.sample,
        "n_h5_files": len(h5_files),
        "n_common_objects": int(n_common),
        "feature_names": feature_names,
        "mcdata_feature_names": arrays["MC+data"]["feature_names"],
        "pt_feature_index": pt_idx,
        "mcdata_pt_feature_index": pt_idx_other,
        "max_abs_pt_delta_between_runs": max_abs_pt_delta,
        "codebook_size": {
            label: arrays[label]["codebook_size"] for label in arrays
        },
        "num_quantizers": {
            label: arrays[label]["num_quantizers"] for label in arrays
        },
        "run_dirs": {label: str(path) for label, path in run_dirs.items()},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    num_quantizers = min(
        arrays["MC-only"]["num_quantizers"],
        arrays["MC+data"]["num_quantizers"],
    )
    if arrays["MC-only"]["num_quantizers"] != arrays["MC+data"]["num_quantizers"]:
        log.warning(
            "Tokenizer quantizer counts differ (MC-only=%d, MC+data=%d); "
            "plotting the first %d shared stages",
            arrays["MC-only"]["num_quantizers"],
            arrays["MC+data"]["num_quantizers"],
            num_quantizers,
        )

    plot_metric(
        rows,
        "unique_code_tuples",
        "unique code tuples",
        output_dir / "unique_code_tuples_by_pt.png",
        title=f"{args.object}: unique code tuples by pT bin",
    )
    plot_metric(
        rows,
        "tuple_perplexity",
        "effective code tuples (perplexity)",
        output_dir / "tuple_perplexity_by_pt.png",
        title=f"{args.object}: effective code tuples by pT bin",
    )
    plot_metric(
        rows,
        "top10_tuple_fraction",
        "fraction in top 10 tuples",
        output_dir / "top10_tuple_fraction_by_pt.png",
        title=f"{args.object}: assignment concentration by pT bin",
    )
    for q_idx in range(num_quantizers):
        plot_metric(
            rows,
            f"q{q_idx}_used_codes",
            f"q{q_idx} used codes",
            output_dir / f"q{q_idx}_used_codes_by_pt.png",
            title=f"{args.object}: quantizer {q_idx} usage by pT bin",
        )
        plot_metric(
            rows,
            f"q{q_idx}_perplexity",
            f"q{q_idx} effective codes (perplexity)",
            output_dir / f"q{q_idx}_perplexity_by_pt.png",
            title=f"{args.object}: quantizer {q_idx} effective codes by pT bin",
        )

    plot_quantizer_grid(
        rows,
        num_quantizers=num_quantizers,
        metric_suffix="used_codes",
        ylabel="used codes",
        output_path=output_dir / "all_quantizers_used_codes_by_pt.png",
        title=f"{args.object}: code usage at every quantizer stage",
    )
    plot_quantizer_grid(
        rows,
        num_quantizers=num_quantizers,
        metric_suffix="perplexity",
        ylabel="effective codes (perplexity)",
        output_path=output_dir / "all_quantizers_perplexity_by_pt.png",
        title=f"{args.object}: effective code usage at every quantizer stage",
    )

    log.info("Wrote %s", csv_path)
    log.info("Wrote plots to %s", output_dir)


if __name__ == "__main__":
    main()
