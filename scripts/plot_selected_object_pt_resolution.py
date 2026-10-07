#!/usr/bin/env python3
"""Plot pT reconstruction resolution for selected object-tokenizer working points."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
import numpy as np
import torch

from plot_tokenizer_summary import (
    COLORS,
    binned_response_iqr_over_median,
    feature_index,
    load_or_collect_arrays,
    safe_filename,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


log = logging.getLogger(__name__)


DEFAULT_LABELS = {
    "jets": "jets",
    "electrons": "electrons",
    "muons": "muons",
    "photons": "photons",
    "taus": "taus",
    "tracks": "tracks",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Make a slide-friendly summary plot of pT resolution for the chosen "
            "working point of each object tokenizer."
        )
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        metavar="OBJECT=RUN_DIR",
        help="Selected run directories, e.g. jets=results/.../jets_logstd_dim8_cb4096_q4.",
    )
    parser.add_argument(
        "--labels",
        action="append",
        default=[],
        metavar="OBJECT=LABEL",
        help="Optional public-facing object label override. May be repeated.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--h5-files", nargs="+")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--max-valid-objects", type=int, default=1_000_000)
    parser.add_argument("--num-events-per-file", type=int)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--n-bins", type=int, default=12)
    parser.add_argument(
        "--min-objects-per-bin",
        type=int,
        default=20,
        help="Minimum valid objects required for drawing a bin.",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        help="Optional maximum x-axis value in GeV, useful for slide formatting.",
    )
    parser.add_argument(
        "--title",
        default="Final object tokenizers: pT reconstruction",
        help="Plot title.",
    )
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def parse_object_paths(values: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected OBJECT=RUN_DIR, got {value}")
        object_name, run_dir = value.split("=", 1)
        parsed.append((object_name.strip(), Path(run_dir).expanduser().resolve()))
    return parsed


def parse_labels(values: list[str]) -> dict[str, str]:
    labels = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected OBJECT=LABEL, got {value}")
        object_name, label = value.split("=", 1)
        labels[object_name.strip()] = label.strip()
    return labels


def binned_resolution_for_run(
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
    n_bins: int,
    min_objects_per_bin: int,
) -> tuple[np.ndarray, np.ndarray]:
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
    idx = feature_index(feature_names, "pt")
    if idx is None:
        raise ValueError(f"Feature 'pt' not found in {run_dir}")

    truth = original[:, idx]
    finite_truth = truth[np.isfinite(truth)]
    finite_truth = finite_truth[np.abs(finite_truth) > 1e-12]
    if len(finite_truth) < min_objects_per_bin:
        raise ValueError(f"Not enough finite pT values in {run_dir}")

    lo, hi = np.percentile(finite_truth, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.min(finite_truth)), float(np.max(finite_truth))
    bins = np.linspace(lo, hi, n_bins + 1)
    centers, values = binned_response_iqr_over_median(
        truth,
        truth,
        reconstruction[:, idx],
        bins=bins,
    )

    # Keep only bins with enough entries. The imported helper uses 20 by default;
    # this extra mask allows stricter slide-quality plots.
    for bin_idx in range(len(centers)):
        in_bin = (truth >= bins[bin_idx]) & (truth < bins[bin_idx + 1])
        if np.count_nonzero(in_bin) < min_objects_per_bin:
            values[bin_idx] = np.nan

    return centers, values


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    label_overrides = parse_labels(args.labels)

    curves = []
    for object_name, run_dir in parse_object_paths(args.runs):
        if not run_dir.exists():
            raise FileNotFoundError(run_dir)
        log.info("Collecting pT resolution for %s from %s", object_name, run_dir)
        centers, values = binned_resolution_for_run(
            run_dir,
            output_dir=output_dir,
            h5_files=args.h5_files,
            split=args.split,
            max_valid_objects=args.max_valid_objects,
            num_events_per_file=args.num_events_per_file,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            n_bins=args.n_bins,
            min_objects_per_bin=args.min_objects_per_bin,
        )
        label = label_overrides.get(object_name, DEFAULT_LABELS.get(object_name, object_name))
        curves.append((object_name, label, centers, values))

    fig, ax = plt.subplots(figsize=(9.0, 5.3))
    for idx, (_, label, centers, values) in enumerate(curves):
        valid = np.isfinite(values)
        if not np.any(valid):
            continue
        ax.plot(
            centers[valid],
            values[valid],
            marker="o",
            markersize=6.5,
            linewidth=2.3,
            color=COLORS[idx % len(COLORS)],
            label=label,
        )

    ax.set_title(args.title, fontsize=19, pad=12)
    ax.set_xlabel("Truth object pT [GeV]", fontsize=17)
    ax.set_ylabel("Relative pT resolution", fontsize=17)
    # ax.text(
    #     0.02,
    #     0.96,
    #     "lower is better",
    #     transform=ax.transAxes,
    #     ha="left",
    #     va="top",
    #     fontsize=13,
    #     color="0.25",
    # )
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=12, loc="best", ncol=2)
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.tick_params(which="both", direction="in", top=True, right=True, length=5, width=1.1)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    if args.x_max is not None:
        ax.set_xlim(left=0, right=args.x_max)
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.17, top=0.88)

    object_stem = "_".join(safe_filename(item[0]) for item in curves)
    stem = f"selected_{object_stem}_pt_resolution"
    fig.savefig(output_dir / f"{stem}.png", dpi=260, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    log.info("Wrote %s", output_dir / f"{stem}.png")


if __name__ == "__main__":
    main()
