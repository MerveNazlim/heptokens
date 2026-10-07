#!/usr/bin/env python3
"""Make cached pT/eta/phi summary plots for chosen object tokenizers."""

from __future__ import annotations

import argparse
import logging
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib
import numpy as np
import torch
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import compare_electron_domain_controls as domain  # noqa: E402
import compare_object_google_stage1_scan as scan  # noqa: E402
from analyze_vqvae_tokenizer import apply_hep_style, feature_label, safe_filename  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


log = logging.getLogger(__name__)

DEFAULT_OBJECTS = ("jets", "electrons", "muons", "photons", "taus")
SUMMARY_FEATURES = ("pt", "eta", "phi")
OBJECT_DISPLAY_NAMES = {
    "jets": "Jets",
    "electrons": "Electrons",
    "muons": "Muons",
    "photons": "Photons",
    "taus": "Taus",
    "tracks": "Tracks",
}
OBJECT_STYLES = {
    "jets": {"color": "#4C78A8", "linestyle": "-"},
    "electrons": {"color": "#F58518", "linestyle": "-"},
    "muons": {"color": "#54A24B", "linestyle": "-"},
    "photons": {"color": "#CC79A7", "linestyle": "-"},
    "taus": {"color": "#7A5195", "linestyle": "-"},
    "tracks": {"color": "#00876C", "linestyle": "-"},
}
CHOSEN_STAGE1_LABELS = {
    "jets": "stage1 q8 cb4096 dim8",
    "electrons": "stage1 q8 cb2048 dim8",
    "muons": "stage1 q8 cb2048 dim8",
    "photons": "stage1 q8 cb2048 dim8",
    "taus": "stage1 q8 cb4096 dim8",
    "tracks": "stage1 q8 cb4096 dim8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot pT/eta/phi binned resolution summaries across all object types "
            "for chosen q8 tokenizers, using cached arrays."
        )
    )
    parser.add_argument("--objects", nargs="+", default=list(DEFAULT_OBJECTS))
    parser.add_argument(
        "--run-source",
        choices=["stage1", "full"],
        default="stage1",
        help="Use chosen Google stage-1 scan runs or chosen full final runs.",
    )
    parser.add_argument("--output-base", default="results/google_stage1_scan_plots")
    parser.add_argument(
        "--cache-base",
        help=(
            "Base directory containing per-object cache_arrays. Defaults to "
            "--output-base. Use this to redraw into a fresh output directory "
            "while reusing caches from an existing plot directory."
        ),
    )
    parser.add_argument(
        "--mc-dir",
        default="/home/zephyr/Data/viviana/bnl-treasure/data_new/h5",
    )
    parser.add_argument(
        "--data-dir",
        default="/home/zephyr/Data/viviana/bnl-treasure/data_new/h5/realdata",
    )
    parser.add_argument("--exclude-mc-pattern", default="DAOD_PHYSLITE.370016*")
    parser.add_argument("--n-files", type=int, default=20)
    parser.add_argument("--checkpoint-name", default="last.ckpt")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-events-per-file", type=int)
    parser.add_argument("--max-valid-objects", type=int, default=1_000_000)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--n-bins", type=int, default=14)
    parser.add_argument("--min-bin-count", type=int, default=50)
    parser.add_argument("--min-denominator", type=float, default=1e-8)
    parser.add_argument("--log-y", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--samples",
        nargs="+",
        choices=["mc", "data"],
        default=["mc", "data"],
    )
    parser.add_argument(
        "--skip-missing-runs",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--force", action="store_true", help="Ignore cache and recompute.")
    return parser.parse_args()


def scan_args_for_object(args: argparse.Namespace, object_name: str) -> argparse.Namespace:
    return argparse.Namespace(
        object=object_name,
        google_base=str(Path("google_results") / object_name),
        full_reference_run="none",
        full_reference_label="none",
        mc_dir=args.mc_dir,
        data_dir=args.data_dir,
        exclude_mc_pattern=args.exclude_mc_pattern,
        output_dir=str(Path(args.cache_base or args.output_base) / object_name),
        n_files=args.n_files,
        num_events_per_file=args.num_events_per_file,
        max_valid_objects=args.max_valid_objects,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split=args.split,
        device="cpu",
        checkpoint_name=args.checkpoint_name,
        n_bins=args.n_bins,
        min_bin_count=args.min_bin_count,
        min_denominator=args.min_denominator,
        features="all",
        line_features="all",
        google_preprocessor="",
        skip_missing_runs=args.skip_missing_runs,
        cache_only=True,
        force=args.force,
        log_y=args.log_y,
    )


def chosen_run_for_object(args: argparse.Namespace, object_name: str) -> tuple[str, Path]:
    """Return the cache label and run path for the chosen tokenizer."""
    if args.run_source == "full":
        return (
            scan.DEFAULT_FULL_REFERENCE_LABELS[object_name],
            Path(scan.DEFAULT_FULL_REFERENCE_RUNS[object_name]),
        )

    target_label = CHOSEN_STAGE1_LABELS[object_name]
    object_args = scan_args_for_object(args, object_name)
    run_dirs = scan.stage1_run_dirs(object_args)
    if target_label not in run_dirs:
        available = ", ".join(run_dirs) if run_dirs else "none"
        raise FileNotFoundError(
            f"Missing chosen stage-1 run for {object_name}: {target_label}. "
            f"Available labels: {available}"
        )
    return target_label, run_dirs[target_label]


def files_by_sample(args: argparse.Namespace) -> OrderedDict[str, list[str]]:
    samples: OrderedDict[str, list[str]] = OrderedDict()
    if "mc" in args.samples:
        samples["MC sample"] = scan.fixed_files(
            args.mc_dir,
            args.n_files,
            exclude_pattern=args.exclude_mc_pattern,
        )
    if "data" in args.samples:
        samples["data"] = scan.fixed_files(args.data_dir, args.n_files)
    for label, files in samples.items():
        if not files:
            raise FileNotFoundError(f"No H5 files selected for {label}")
    return samples


def feature_index(feature_names: list[str], feature_name: str) -> int | None:
    lookup = {name.lower(): idx for idx, name in enumerate(feature_names)}
    return lookup.get(feature_name.lower())


def collect_cached_outputs(
    args: argparse.Namespace,
) -> dict[str, dict[str, OrderedDict[str, dict]]]:
    device = torch.device("cpu")
    sample_files = files_by_sample(args)
    outputs: dict[str, dict[str, OrderedDict[str, dict]]] = {}

    for object_name in args.objects:
        object_args = scan_args_for_object(args, object_name)
        output_dir = Path(object_args.output_dir)
        model_label, run_dir = chosen_run_for_object(args, object_name)
        run_dirs = OrderedDict([(model_label, run_dir)])
        scan.apply_styles(model_label)
        scan.apply_display_styles(list(run_dirs))
        display_model_label = scan.display_label(model_label)
        shown_model_label = display_model_label.removeprefix("full ")
        domain.COLORS[shown_model_label] = domain.COLORS.get(
            display_model_label, domain.COLORS.get(model_label, "#111111")
        )
        domain.LINESTYLES[shown_model_label] = domain.LINESTYLES.get(
            display_model_label, domain.LINESTYLES.get(model_label, "-")
        )

        outputs[object_name] = {}
        for sample_label, h5_files in sample_files.items():
            outputs[object_name][sample_label] = OrderedDict()
            for model_label, run_dir in run_dirs.items():
                item = scan.collect_one(
                    sample_label=sample_label,
                    model_label=model_label,
                    run_dir=run_dir,
                    files=h5_files,
                    output_dir=output_dir,
                    args=object_args,
                    device=device,
                )
                outputs[object_name][sample_label][shown_model_label] = item
    return outputs


def make_feature_curves(
    *,
    models: OrderedDict[str, dict],
    feature_name: str,
    args: argparse.Namespace,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    if not models:
        return []

    reference = next(iter(models.values()))
    idx = feature_index(reference["feature_names"], feature_name)
    if idx is None:
        return []

    bins = domain.make_bins(reference["original"][:, idx], args.n_bins)
    curves = []
    metric_kind = scan.binned_metric_kind(feature_name)
    for model_label, item in models.items():
        idx = feature_index(item["feature_names"], feature_name)
        if idx is None:
            continue
        centers, values, *_ = scan.binned_metric_values(
            item["original"][:, idx],
            item["reconstruction"][:, idx],
            bins,
            metric_kind=metric_kind,
            min_bin_count=args.min_bin_count,
            min_denominator=args.min_denominator,
        )
        curves.append((model_label, centers, values))
    return curves


def resolution_ylabel(feature_name: str) -> str:
    key = feature_name.lower()
    if key == "pt":
        return r"IQR / median of $p_\mathrm{T}^\mathrm{reco}/p_\mathrm{T}^\mathrm{truth}$"
    if key == "eta":
        return r"IQR($\eta^\mathrm{reco} - \eta^\mathrm{truth}$)"
    if key == "phi":
        return r"IQR($\phi^\mathrm{reco} - \phi^\mathrm{truth}$)"
    return scan.binned_metric_label(scan.binned_metric_kind(feature_name), feature_name)


def resolution_xlabel(feature_name: str) -> str:
    key = feature_name.lower()
    if key == "pt":
        return "Truth object pT [GeV]"
    if key == "eta":
        return r"Truth object $\eta$"
    if key == "phi":
        return r"Truth object $\phi$"
    return scan.truth_axis_label(feature_name)


def resolution_title(sample_label: str, feature_name: str) -> str:
    sample = "MC sample" if sample_label == "MC sample" else "Data"
    label = feature_label(feature_name)
    if feature_name.lower() == "pt":
        label = "pT"
    return f"{sample}: object tokenizers {label} reconstruction"


def nice_axis_upper(ymax: float) -> float:
    if not np.isfinite(ymax) or ymax <= 0.0:
        return 1.0
    target = 1.15 * ymax
    magnitude = 10.0 ** np.floor(np.log10(target))
    for multiplier in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0):
        upper = multiplier * magnitude
        if upper >= target:
            return upper
    return 10.0 * magnitude


def plot_sample_feature_summary(
    *,
    sample_label: str,
    feature_name: str,
    outputs: dict[str, dict[str, OrderedDict[str, dict]]],
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    object_names = [name for name in args.objects if sample_label in outputs.get(name, {})]
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    legend_handles = []
    legend_labels = []
    plotted_values = []

    for object_name in object_names:
        models = outputs[object_name][sample_label]
        curves = make_feature_curves(models=models, feature_name=feature_name, args=args)
        style = OBJECT_STYLES.get(object_name, {"color": None, "linestyle": "-"})

        for _, centers, values in curves:
            valid = np.isfinite(values)
            if args.log_y:
                valid &= values > 0.0
            if not np.any(valid):
                continue
            ax.plot(
                centers[valid],
                values[valid],
                marker="o",
                linewidth=1.9,
                markersize=5.2,
                label=OBJECT_DISPLAY_NAMES.get(object_name, object_name),
                color=style["color"],
                linestyle=style["linestyle"],
            )
            plotted_values.extend(values[valid].tolist())

    for object_name in object_names:
        style = OBJECT_STYLES.get(object_name, {"color": "0.2", "linestyle": "-"})
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linewidth=1.9,
                markersize=5.2,
                color=style["color"],
                linestyle=style["linestyle"],
            )
        )
        legend_labels.append(OBJECT_DISPLAY_NAMES.get(object_name, object_name))

    if args.log_y:
        ax.set_yscale("log")
    elif plotted_values:
        ax.set_ylim(0.0, nice_axis_upper(max(plotted_values)))

    ax.grid(alpha=0.22)
    apply_hep_style(ax)
    ax.set_title(resolution_title(sample_label, feature_name), fontsize=15)
    ax.set_xlabel(resolution_xlabel(feature_name), fontsize=13)
    ax.set_ylabel(resolution_ylabel(feature_name), fontsize=13)

    if not plotted_values:
        ax.text(
            0.5,
            0.5,
            "missing feature",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="0.35",
        )

    if legend_handles and legend_labels:
        ax.legend(
            legend_handles,
            legend_labels,
            loc="upper left",
            frameon=False,
            fontsize=10,
            ncol=2,
            handlelength=2.2,
            columnspacing=1.2,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    outputs = collect_cached_outputs(args)

    output_dir = Path(args.output_base) / "summary_pt_eta_phi"
    run_tag = "full" if args.run_source == "full" else "stage1"
    sample_to_stem = {
        "MC sample": "mc",
        "data": "data",
    }
    for sample_label in files_by_sample(args):
        for feature_name in SUMMARY_FEATURES:
            plot_sample_feature_summary(
                sample_label=sample_label,
                feature_name=feature_name,
                outputs=outputs,
                output_path=output_dir
                / (
                    f"chosen_{run_tag}_q8_{sample_to_stem[sample_label]}_"
                    f"{safe_filename(feature_name)}_resolution.png"
                ),
                args=args,
            )
    log.info("Wrote pT/eta/phi summaries to %s", output_dir)


if __name__ == "__main__":
    main()
