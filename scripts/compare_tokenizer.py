
"""Compare reconstruction performance of two trained VQ-VAE / PQ-VAE tokenizers.
 
Produces per-feature overlay plots (truth + model A + model B) and a
side-by-side residual-metric summary. Codebook diagnostics are intentionally
omitted — reconstruction quality only.
 
Example
-------
python compare_tokenizers.py \
    --run-dir-a  /path/to/run_A \
    --run-dir-b  /path/to/run_B \
    --label-a    "VQ-VAE v1" \
    --label-b    "VQ-VAE v2" \
    --output-dir /path/to/comparison_figures
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
import numpy as np
from omegaconf import OmegaConf

from analyze_vqvae_tokenizer import (
    apply_hep_style,
    choose_device,
    collect_diagnostics_for_h5_files,
    display_values,
    feature_label,
    feature_names_from_cfg,
    finite_pair,
    find_checkpoint,
    histogram_bins,
    load_analysis_model,
    model_cls_from_type,
    object_name_from_cfg,
    reconstruction_metrics,
    safe_filename,
    transform_list_and_cst_fn_from_cfg,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

_COLORS = ("#00A000", "#E07000")  # green for A, orange for B

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare reconstruction performance of two tokenizer runs."
    )
    parser.add_argument("--model-type-a", choices=["vqvae", "pqvae"], default="vqvae", )
    parser.add_argument("--model-type-b", choices=["vqvae", "pqvae"], default="pqvae", )
    parser.add_argument("--run-dir-a", required=True, help="Run directory for model A.")
    parser.add_argument("--run-dir-b", required=True, help="Run directory for model B.")
    parser.add_argument("--label-a", default="Model A", help="Legend label for model A.")
    parser.add_argument("--label-b", default="Model B", help="Legend label for model B.")
    parser.add_argument("--checkpoint-a", default=None, help="Checkpoint for model A (default: best/last).")
    parser.add_argument("--checkpoint-b", default=None, help="Checkpoint for model B (default: best/last).")
    parser.add_argument("--output-dir", required=True, help="Directory for comparison figures.")
    parser.add_argument(
        "--model-type",
        choices=["vqvae", "pqvae"],
        default="vqvae",
        help="Architecture stored in both checkpoints.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="val",
    )
    parser.add_argument("--h5-files", nargs="+", help="Override H5 files (applied to both models).")
    parser.add_argument("--num-events-per-file", type=int, default=None)
    parser.add_argument("--max-valid-objects", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def plot_comparison_overlay(
    original: np.ndarray,
    reco_a: np.ndarray,
    reco_b: np.ndarray,
    feature_names: list[str],
    object_name: str,
    label_a: str,
    label_b: str,
    output_dir: Path,
) -> None:
    """One overlay plot per feature: truth (black) + model A + model B."""
    color_a, color_b = _COLORS
    for feat_idx, feat_name in enumerate(feature_names):
        if feat_idx >= original.shape[1]:
            continue

        orig, ra, axis_label = display_values(
            original[:, feat_idx], reco_a[:, feat_idx], feat_name
        )
        _, rb, _ = display_values(
            original[:, feat_idx], reco_b[:, feat_idx], feat_name
        )

        finite = np.isfinite(orig) & np.isfinite(ra) & np.isfinite(rb)
        orig, ra, rb = orig[finite], ra[finite], rb[finite]
        if len(orig) == 0:
            log.warning("Skipping %s: no finite values", feat_name)
            continue

        bins = histogram_bins(np.concatenate([orig, ra, rb]), n_bins=75)

        fig, ax = plt.subplots(figsize=(8.0, 6.0))
        ax.hist(orig, bins=bins, histtype="step", density=True,
                linewidth=2.2, color="black", label="Truth")
        ax.hist(ra,   bins=bins, histtype="step", density=True,
                linewidth=2.0, color=color_a, label=label_a)
        ax.hist(rb,   bins=bins, histtype="step", density=True,
                linewidth=2.0, color=color_b, linestyle="--", label=label_b)

        ax.set_title(f"{object_name.capitalize()} {feature_label(feat_name)}", fontsize=17)
        ax.set_xlabel(axis_label, fontsize=14)
        ax.set_ylabel("Normalized", fontsize=14)
        ax.legend(frameon=False, fontsize=12)
        apply_hep_style(ax)
        fig.tight_layout()
        fig.savefig(output_dir / f"{safe_filename(feat_name)}_comparison_overlay.png", dpi=180)
        plt.close(fig)


def plot_comparison_residuals(
    original: np.ndarray,
    reco_a: np.ndarray,
    reco_b: np.ndarray,
    feature_names: list[str],
    object_name: str,
    label_a: str,
    label_b: str,
    output_dir: Path,
) -> None:
    """Side-by-side residual histogram for each feature."""
    color_a, color_b = _COLORS
    for feat_idx, feat_name in enumerate(feature_names):
        if feat_idx >= original.shape[1]:
            continue

        orig_raw = original[:, feat_idx]
        ra_raw   = reco_a[:, feat_idx]
        rb_raw   = reco_b[:, feat_idx]

        finite = np.isfinite(orig_raw) & np.isfinite(ra_raw) & np.isfinite(rb_raw)
        orig, ra, rb = orig_raw[finite], ra_raw[finite], rb_raw[finite]
        if len(orig) == 0:
            continue

        res_a = ra - orig
        res_b = rb - orig
        bins = histogram_bins(np.concatenate([res_a, res_b]), n_bins=70)

        fig, ax = plt.subplots(figsize=(8.0, 5.5))
        ax.hist(res_a, bins=bins, histtype="step", density=True,
                linewidth=2.0, color=color_a, label=label_a)
        ax.hist(res_b, bins=bins, histtype="step", density=True,
                linewidth=2.0, color=color_b, linestyle="--", label=label_b)
        ax.axvline(0, color="black", linewidth=1.2)

        ax.set_title(f"{object_name.capitalize()} {feature_label(feat_name)} — residuals", fontsize=16)
        ax.set_xlabel("reco − truth", fontsize=13)
        ax.set_ylabel("Normalized", fontsize=13)
        ax.legend(frameon=False, fontsize=12)
        apply_hep_style(ax)
        fig.tight_layout()
        fig.savefig(output_dir / f"{safe_filename(feat_name)}_comparison_residual.png", dpi=180)
        plt.close(fig)


def plot_comparison_metric_bar(
    metrics_a: dict,
    metrics_b: dict,
    label_a: str,
    label_b: str,
    output_path: Path,
    metric_key: str = "rmse",
    y_label: str = "RMSE (reco − truth)",
) -> None:
    """Bar chart comparing a single scalar metric across all shared features."""
    shared_features = [f for f in metrics_a if f in metrics_b]
    if not shared_features:
        log.warning("No shared features between the two models — skipping metric bar chart.")
        return

    vals_a = np.array([metrics_a[f][metric_key] for f in shared_features])
    vals_b = np.array([metrics_b[f][metric_key] for f in shared_features])
    labels = [feature_label(f) for f in shared_features]

    x = np.arange(len(labels))
    width = 0.38
    color_a, color_b = _COLORS

    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 0.9), 5.0))
    ax.bar(x - width / 2, vals_a, width, label=label_a, color=color_a, alpha=0.85)
    ax.bar(x + width / 2, vals_b, width, label=label_b, color=color_b, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=11)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(f"Reconstruction comparison — {y_label}", fontsize=15)
    ax.legend(frameon=False, fontsize=12)
    ax.grid(axis="y", alpha=0.25)
    apply_hep_style(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device     = choose_device(args.device)
    model_cls_a = model_cls_from_type(args.model_type_a)
    model_cls_b = model_cls_from_type(args.model_type_b)
    h5_files   = list(args.h5_files) if args.h5_files else None

    # ── Load both models ─────────────────────────────────────────────────────
    run_a = Path(args.run_dir_a).resolve()
    run_b = Path(args.run_dir_b).resolve()

    cfg_a = OmegaConf.load(run_a / "full_config.yaml")
    cfg_b = OmegaConf.load(run_b / "full_config.yaml")

    _, inv_a = transform_list_and_cst_fn_from_cfg(cfg_a)
    _, inv_b = transform_list_and_cst_fn_from_cfg(cfg_b)

    model_a = load_analysis_model(run_a, args.checkpoint_a, device, model_cls=model_cls_a)
    model_b = load_analysis_model(run_b, args.checkpoint_b, device, model_cls=model_cls_b)

    # ── Collect reconstruction arrays ────────────────────────────────────────
    log.info("Collecting diagnostics for model A (%s)", args.label_a)
    original_a, reco_a, _, _ = collect_diagnostics_for_h5_files(
        cfg=cfg_a, model=model_a, h5_files=h5_files,
        split=args.split, num_events_per_file=args.num_events_per_file,
        batch_size=args.batch_size, num_workers=args.num_workers,
        cst_inverse_transformer=inv_a, device=device,
        max_valid_objects=args.max_valid_objects,
    )

    log.info("Collecting diagnostics for model B (%s)", args.label_b)
    original_b, reco_b, _, _ = collect_diagnostics_for_h5_files(
        cfg=cfg_b, model=model_b, h5_files=h5_files,
        split=args.split, num_events_per_file=args.num_events_per_file,
        batch_size=args.batch_size, num_workers=args.num_workers,
        cst_inverse_transformer=inv_b, device=device,
        max_valid_objects=args.max_valid_objects,
    )

    feat_names_a = feature_names_from_cfg(cfg_a, original_a.shape[1])
    feat_names_b = feature_names_from_cfg(cfg_b, original_b.shape[1])
    object_name  = object_name_from_cfg(cfg_a)

    # ── Reconstruction metrics ────────────────────────────────────────────────
    metrics_a = reconstruction_metrics(original_a, reco_a, feat_names_a)
    metrics_b = reconstruction_metrics(original_b, reco_b, feat_names_b)

    (output_dir / "reconstruction_metrics_a.json").write_text(
        json.dumps(metrics_a, indent=2) + "\n"
    )
    (output_dir / "reconstruction_metrics_b.json").write_text(
        json.dumps(metrics_b, indent=2) + "\n"
    )

    # ── Comparison plots ──────────────────────────────────────────────────────
    # For the per-feature plots both models must share the same feature set and
    # the same truth array. If the configs differ, we fall back to model-A's
    # features and only plot features present in both.
    shared_features = [f for f in feat_names_a if f in feat_names_b]
    if feat_names_a != feat_names_b:
        log.warning(
            "Feature lists differ between models. Comparison plots will use the "
            "%d shared features only.", len(shared_features)
        )

    # Align arrays to shared features.
    idx_a = [feat_names_a.index(f) for f in shared_features]
    idx_b = [feat_names_b.index(f) for f in shared_features]

    # Use model A's truth as the reference (both should be identical if the
    # same H5 files and preprocessing are used; warn if sizes differ).
    n = min(len(original_a), len(original_b))
    if len(original_a) != len(original_b):
        log.warning(
            "Object counts differ (A=%d, B=%d). Truncating to %d for joint plots.",
            len(original_a), len(original_b), n,
        )

    orig_shared = original_a[:n][:, idx_a]
    ra_shared   = reco_a[:n][:, idx_a]
    rb_shared   = reco_b[:n][:, idx_b]

    overlay_dir  = output_dir / "overlay"
    residual_dir = output_dir / "residuals"
    overlay_dir.mkdir(exist_ok=True)
    residual_dir.mkdir(exist_ok=True)

    log.info("Plotting comparison overlays …")
    plot_comparison_overlay(
        orig_shared, ra_shared, rb_shared,
        shared_features, object_name,
        args.label_a, args.label_b, overlay_dir,
    )

    log.info("Plotting comparison residuals …")
    plot_comparison_residuals(
        orig_shared, ra_shared, rb_shared,
        shared_features, object_name,
        args.label_a, args.label_b, residual_dir,
    )

    log.info("Plotting metric summary bars …")
    for metric_key, ylabel in [
        ("rmse", "RMSE (reco − truth)"),
        ("mae",  "MAE  (reco − truth)"),
    ]:
        plot_comparison_metric_bar(
            metrics_a, metrics_b,
            args.label_a, args.label_b,
            output_dir / f"comparison_{metric_key}.png",
            metric_key=metric_key,
            y_label=ylabel,
        )

    log.info("Done. Figures written to %s", output_dir)


if __name__ == "__main__":
    main()

