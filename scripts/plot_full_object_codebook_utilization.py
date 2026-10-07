#!/usr/bin/env python3
"""Plot codebook utilization for the chosen full object tokenizers."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import matplotlib
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_vqvae_tokenizer import apply_hep_style, safe_filename  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


log = logging.getLogger(__name__)

DEFAULT_OBJECTS = ("jets", "electrons", "muons", "photons", "taus")
OBJECT_LABELS = {
    "jets": "Jets",
    "electrons": "Electrons",
    "muons": "Muons",
    "photons": "Photons",
    "taus": "Taus",
    "tracks": "Tracks",
}
Q_COLORS = (
    "#4C78A8",
    "#F58518",
    "#E45756",
    "#72B7B2",
    "#54A24B",
    "#EECA3B",
    "#B279A2",
    "#FF9DA6",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Make a single grouped-bar codebook-utilization plot for the chosen "
            "full object tokenizers, using existing triptych diagnostics."
        )
    )
    parser.add_argument(
        "--input-base",
        default="results/final_object_full_triptychs",
        help="Directory containing <object>/<sample>/codebook_counts.npy.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/final_object_full_triptychs/summary",
    )
    parser.add_argument("--objects", nargs="+", default=list(DEFAULT_OBJECTS))
    parser.add_argument(
        "--samples",
        nargs="+",
        choices=["mc", "data"],
        default=["mc", "data"],
    )
    parser.add_argument(
        "--skip-missing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip missing codebook_counts.npy files instead of failing.",
    )
    return parser.parse_args()


def counts_path(input_base: Path, object_name: str, sample: str) -> Path:
    return input_base / object_name / sample / "codebook_counts.npy"


def utilization_by_quantizer(counts: np.ndarray) -> list[float]:
    if counts.ndim != 2:
        raise ValueError(f"Expected counts with shape (n_quantizers, codebook_size), got {counts.shape}")
    return [100.0 * np.count_nonzero(row) / row.shape[0] for row in counts]


def load_rows(args: argparse.Namespace) -> list[dict]:
    input_base = Path(args.input_base)
    rows = []
    for sample in args.samples:
        for object_name in args.objects:
            path = counts_path(input_base, object_name, sample)
            if not path.exists():
                message = f"Missing {path}"
                if args.skip_missing:
                    log.warning(message)
                    continue
                raise FileNotFoundError(message)
            counts = np.load(path)
            for quantizer_idx, used_percent in enumerate(utilization_by_quantizer(counts)):
                rows.append(
                    {
                        "sample": sample,
                        "object": object_name,
                        "quantizer": quantizer_idx,
                        "used_percent": float(used_percent),
                        "used_codes": int(np.count_nonzero(counts[quantizer_idx])),
                        "total_codes": int(counts.shape[1]),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sample_title(sample: str) -> str:
    return "MC sample" if sample == "mc" else "data"


def plot_utilization(rows: list[dict], args: argparse.Namespace, output_dir: Path) -> None:
    objects = [object_name for object_name in args.objects if any(row["object"] == object_name for row in rows)]
    samples = [sample for sample in args.samples if any(row["sample"] == sample for row in rows)]
    max_q = max(int(row["quantizer"]) for row in rows) + 1

    fig, axes = plt.subplots(
        len(samples),
        1,
        figsize=(max(9.0, 1.45 * len(objects) + 4.0), 4.2 * len(samples)),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]
    x = np.arange(len(objects))
    width = min(0.78 / max_q, 0.11)

    for ax, sample in zip(axes, samples):
        sample_rows = [row for row in rows if row["sample"] == sample]
        for q_idx in range(max_q):
            values = []
            for object_name in objects:
                match = [
                    row
                    for row in sample_rows
                    if row["object"] == object_name and int(row["quantizer"]) == q_idx
                ]
                values.append(match[0]["used_percent"] if match else np.nan)
            offset = (q_idx - (max_q - 1) / 2.0) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                color=Q_COLORS[q_idx % len(Q_COLORS)],
                label=f"q{q_idx}",
            )

        ax.set_title(sample_title(sample), fontsize=18)
        ax.set_ylabel("Utilization [%]", fontsize=15)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.22)
        apply_hep_style(ax)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([OBJECT_LABELS.get(obj, obj) for obj in objects], rotation=20, ha="right")
    axes[-1].set_xlabel("Object tokenizer", fontsize=15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=min(max_q, 8),
        frameon=False,
        fontsize=14,
    )
    fig.suptitle("Codebook utilization for chosen full object tokenizers", fontsize=20, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    sample_tag = "_".join(samples)
    output_png = output_dir / f"chosen_full_object_codebook_utilization_{sample_tag}.png"
    output_pdf = output_dir / f"chosen_full_object_codebook_utilization_{sample_tag}.pdf"
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", output_png)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args)
    if not rows:
        raise RuntimeError("No codebook-count rows loaded")

    sample_tag = "_".join(args.samples)
    write_csv(output_dir / f"chosen_full_object_codebook_utilization_{safe_filename(sample_tag)}.csv", rows)
    plot_utilization(rows, args, output_dir)


if __name__ == "__main__":
    main()
