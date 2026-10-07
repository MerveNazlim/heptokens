#!/usr/bin/env python3
"""Compare two jet tokenizers on one shared right-run validation sample."""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
import sys

import matplotlib
import numpy as np
import torch
from omegaconf import OmegaConf

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_vqvae_tokenizer import (  # noqa: E402
    apply_hep_style,
    binned_residual_iqr_over_median_truth,
    choose_device,
    collect_diagnostics_for_h5_files,
    feature_label,
    feature_names_from_cfg,
    find_checkpoint,
    safe_filename,
    transform_list_and_cst_fn_from_cfg,
)
from heptokens.models.vq_vae import LitVqVae  # noqa: E402
from plot_mc_vs_mcdata_resolution import (  # noqa: E402
    feature_index,
    fixed_files,
    make_bins,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


log = logging.getLogger(__name__)

DEFAULT_LEFT_RUN = (
    "google_results/jets/q4/cluster_26_proc_0/results/"
    "atlas_object_final_tokenizers_new_mcdata/jets_full_dim16_cb16384_q4_e20_new_mcdata"
)
DEFAULT_RIGHT_RUN = (
    "results/atlas_object_final_tokenizers_new_mcdata/"
    "jets_full_dim16_cb16384_q4_e20_new_mcdata"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two jet tokenizer runs with one MC/data two-panel resolution "
            "plot per feature."
        )
    )
    parser.add_argument("--left-run", default=DEFAULT_LEFT_RUN)
    parser.add_argument("--right-run", default=DEFAULT_RIGHT_RUN)
    parser.add_argument("--left-label", default="Google/Condor q4")
    parser.add_argument("--right-label", default="Zephyr q4")
    parser.add_argument(
        "--mc-dir",
        default="/home/zephyr/Data/viviana/bnl-treasure/data_new/h5",
        help="Directory containing MC H5 files.",
    )
    parser.add_argument(
        "--data-dir",
        default="/home/zephyr/Data/viviana/bnl-treasure/data_new/h5/realdata",
        help="Directory containing real-data H5 files.",
    )
    parser.add_argument(
        "--exclude-mc-pattern",
        default="DAOD_PHYSLITE.370016*",
        help="MC filename glob to exclude. Set to empty string to disable.",
    )
    parser.add_argument("--output-dir", default="results/jet_q4_google_vs_zephyr")
    parser.add_argument(
        "--features",
        default="",
        help="Comma-separated jet features. Default: all features from left config.",
    )
    parser.add_argument("--n-files", type=int, default=20)
    parser.add_argument(
        "--max-valid-objects",
        type=int,
        default=1_000_000,
        help="Maximum valid objects to retain; use 0 to retain all objects.",
    )
    parser.add_argument("--num-events-per-file", type=int)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument(
        "--all-events",
        action="store_true",
        help=(
            "Evaluate every event from the selected files through one shared "
            "right-run mappable datamodule, ignoring saved train/val/test membership."
        ),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--checkpoint", choices=["best.ckpt", "last.ckpt"], default="last.ckpt")
    parser.add_argument(
        "--preprocessor",
        default="",
        help=(
            "Use one common preprocessor for both runs when a saved path is missing. "
            "If omitted, common local jet preprocessor locations are tried."
        ),
    )
    parser.add_argument(
        "--left-preprocessor",
        default="",
        help="Preprocessor used for the left checkpoint (overrides its saved path).",
    )
    parser.add_argument(
        "--right-preprocessor",
        default="",
        help="Preprocessor used for the right checkpoint (overrides its saved path).",
    )
    parser.add_argument("--n-bins", type=int, default=14)
    parser.add_argument("--min-bin-count", type=int, default=50)
    parser.add_argument("--min-denominator", type=float, default=1e-8)
    parser.add_argument("--force", action="store_true", help="Delete this script's cache first.")
    return parser.parse_args()


def selected_files(directory: str, n_files: int, exclude_pattern: str = "") -> list[str]:
    files = fixed_files(directory, 10**9)
    if exclude_pattern:
        files = [
            path
            for path in files
            if not Path(path).match(f"*/{exclude_pattern}")
            and not Path(path).match(exclude_pattern)
        ]
    return files[:n_files]


def default_features(run_dir: Path) -> list[str]:
    cfg = OmegaConf.load(run_dir / "full_config.yaml")
    datamodule = OmegaConf.to_container(cfg.datamodule, resolve=True)
    object_type = datamodule.get("object_type", "jets")
    names = []
    for collection in datamodule.get("object_collections") or []:
        if collection.get("object_name") == object_type:
            names = [Path(path).name for path in collection.get("inputs") or []]
            break
    if not names:
        raise RuntimeError(f"Could not infer features from {run_dir / 'full_config.yaml'}")
    return names


def existing_preprocessor_from_cfg(cfg) -> Path | None:
    filename = OmegaConf.select(cfg, "datamodule.transforms.preprocess.cst_fn.filename")
    if not filename:
        return None
    path = Path(str(filename))
    return path if path.exists() else None


def patch_missing_preprocessor(cfg, args: argparse.Namespace, run_dir: Path) -> Path | None:
    resolved_run = run_dir.resolve()
    if resolved_run == Path(args.left_run).resolve():
        explicit = args.left_preprocessor
    elif resolved_run == Path(args.right_run).resolve():
        explicit = args.right_preprocessor
    else:
        explicit = ""

    if explicit:
        path = Path(explicit).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Explicit preprocessor does not exist: {path}")
        OmegaConf.update(
            cfg,
            "datamodule.transforms.preprocess.cst_fn.filename",
            str(path),
            force_add=True,
        )
        log.info("Using explicit preprocessor %s for %s", path, run_dir)
        return path

    existing = existing_preprocessor_from_cfg(cfg)
    if existing is not None:
        return existing

    saved = OmegaConf.select(cfg, "datamodule.transforms.preprocess.cst_fn.filename")
    candidates = []
    if args.preprocessor:
        candidates.append(Path(args.preprocessor))
    candidates.extend(
        [
            Path("results/preprocessing/atlas_object_final_tokenizers_new_mcdata/jets_log_standard.joblib"),
            Path("results/preprocessing/jets_log_standard.joblib"),
            run_dir.parent.parent.parent.parent.parent
            / "preprocessing"
            / "atlas_object_final_tokenizers_new_mcdata"
            / "jets_log_standard.joblib",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            OmegaConf.update(
                cfg,
                "datamodule.transforms.preprocess.cst_fn.filename",
                str(candidate.resolve()),
                force_add=True,
            )
            log.info(
                "Using local preprocessor %s for %s; saved path was %s",
                candidate,
                run_dir,
                saved,
            )
            return candidate.resolve()

    if saved:
        raise FileNotFoundError(
            "Saved preprocessor path is missing and no local replacement was found. "
            f"Saved path: {saved}. Pass --preprocessor /path/to/jets_log_standard.joblib"
        )
    return None


def load_or_collect_arrays(
    *,
    run_dir: Path,
    sample_name: str,
    model_name: str,
    h5_files: list[str],
    output_dir: Path,
    split: str,
    max_valid_objects: int,
    num_events_per_file: int | None,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    checkpoint_name: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    checkpoint = find_checkpoint(run_dir, str(run_dir / "checkpoints" / checkpoint_name))
    checkpoint_stat = checkpoint.stat()

    cfg = OmegaConf.load(run_dir / "full_config.yaml")
    reference_cfg_path = Path(args.right_run).resolve() / "full_config.yaml"
    reference_cfg = OmegaConf.load(reference_cfg_path)
    cfg.datamodule = OmegaConf.create(
        OmegaConf.to_container(reference_cfg.datamodule, resolve=False)
    )
    if args.all_events:
        OmegaConf.set_struct(cfg.datamodule, False)
        cfg.datamodule._target_ = "heptokens.data.atlas_event_mappable.AtlasEventMapModule"
        cfg.datamodule.train_frac = 0.0
        cfg.datamodule.val_frac = 1.0
        cfg.datamodule.test_frac = 0.0
    preprocessor = patch_missing_preprocessor(cfg, args, run_dir)
    _, cst_inverse_transformer = transform_list_and_cst_fn_from_cfg(cfg)

    cache_dir = output_dir / "cache_arrays"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_input = "\n".join(
        h5_files
        + [
            f"num_events_per_file={num_events_per_file}",
            f"checkpoint={checkpoint.resolve()}",
            f"checkpoint_size={checkpoint_stat.st_size}",
            f"checkpoint_mtime_ns={checkpoint_stat.st_mtime_ns}",
            f"preprocessor={preprocessor}",
            f"shared_evaluation_datamodule={reference_cfg_path}",
            f"shared_evaluation_datamodule_mtime_ns={reference_cfg_path.stat().st_mtime_ns}",
            f"all_events={args.all_events}",
            "analysis_path=two_jet_tokenizer_compare_shared_right_datamodule_v5",
        ]
    )
    file_digest = hashlib.sha1(cache_input.encode("utf-8")).hexdigest()[:12]
    cache_name = (
        f"{safe_filename(sample_name)}_{safe_filename(model_name)}_"
        f"{safe_filename(run_dir.name)}_{split}_{len(h5_files)}files_"
        f"{file_digest}_{max_valid_objects}_twojet_v2.npz"
    )
    cache_path = cache_dir / cache_name
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        return cached["original"], cached["reconstruction"], list(cached["feature_names"])

    log.info("Loading %s", checkpoint)
    model = LitVqVae.load_from_checkpoint(checkpoint, map_location=device)
    model.to(device)
    model.eval()

    object_limit = max_valid_objects if max_valid_objects > 0 else sys.maxsize
    original, reconstruction, _, _ = collect_diagnostics_for_h5_files(
        cfg=cfg,
        model=model,
        h5_files=h5_files,
        split=split,
        num_events_per_file=num_events_per_file,
        batch_size=batch_size,
        num_workers=num_workers,
        cst_inverse_transformer=cst_inverse_transformer,
        device=device,
        max_valid_objects=object_limit,
    )
    feature_names = feature_names_from_cfg(cfg, original.shape[1])
    np.savez_compressed(
        cache_path,
        original=original,
        reconstruction=reconstruction,
        feature_names=np.asarray(feature_names),
    )
    return original, reconstruction, feature_names


def collect_feature_curve(
    *,
    run_dir: Path,
    model_label: str,
    sample_name: str,
    h5_files: list[str],
    feature_name: str,
    bins: np.ndarray,
    output_dir: Path,
    args: argparse.Namespace,
):
    original, reconstruction, feature_names = load_or_collect_arrays(
        run_dir=run_dir,
        sample_name=sample_name,
        model_name=model_label,
        h5_files=h5_files,
        output_dir=output_dir,
        split=args.split,
        max_valid_objects=args.max_valid_objects,
        num_events_per_file=args.num_events_per_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=choose_device(args.device),
        checkpoint_name=args.checkpoint,
        args=args,
    )
    idx = feature_index(feature_names, feature_name)
    if idx is None:
        log.warning("Missing feature %s in %s. Available: %s", feature_name, run_dir, feature_names)
        return None
    centers, values, counts, medians, residual_iqr = binned_residual_iqr_over_median_truth(
        original[:, idx],
        reconstruction[:, idx],
        bins,
        min_bin_count=args.min_bin_count,
        min_denominator=args.min_denominator,
    )
    return centers, values, counts, residual_iqr, medians


def write_curve_csv(
    *,
    csv_path: Path,
    rows: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w") as handle:
        handle.write(
            "sample,model,bin_center,relative_resolution,objects,"
            "residual_iqr,median_original\n"
        )
        for sample, model, centers, values, counts, residual_iqr, medians in rows:
            for center, value, count, iqr, median in zip(
                centers,
                values,
                counts,
                residual_iqr,
                medians,
            ):
                handle.write(
                    f"{sample},{model},{center},{value},{int(count)},{iqr},{median}\n"
                )


def plot_feature(
    *,
    feature_name: str,
    runs: dict[str, Path],
    samples: dict[str, list[str]],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    colors = {
        args.left_label: "#4C83F1",
        args.right_label: "#FF9F1C",
    }
    linestyles = {
        args.left_label: "-",
        args.right_label: "--",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    csv_rows = []

    for ax, (sample_name, h5_files) in zip(axes, samples.items()):
        reference_original, _, reference_features = load_or_collect_arrays(
            run_dir=next(iter(runs.values())),
            sample_name=sample_name,
            model_name=next(iter(runs.keys())),
            h5_files=h5_files,
            output_dir=output_dir,
            split=args.split,
            max_valid_objects=args.max_valid_objects,
            num_events_per_file=args.num_events_per_file,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=choose_device(args.device),
            checkpoint_name=args.checkpoint,
            args=args,
        )
        feature_idx = feature_index(reference_features, feature_name)
        if feature_idx is None:
            log.warning("Skipping %s: missing in reference run", feature_name)
            plt.close(fig)
            return
        bins = make_bins(reference_original[:, feature_idx], args.n_bins)
        if bins is None:
            log.warning("Skipping %s: no useful bin range", feature_name)
            plt.close(fig)
            return

        curves = []
        for label, run_dir in runs.items():
            curve = collect_feature_curve(
                run_dir=run_dir,
                model_label=label,
                sample_name=sample_name,
                h5_files=h5_files,
                feature_name=feature_name,
                bins=bins,
                output_dir=output_dir,
                args=args,
            )
            if curve is None:
                continue
            centers, values, counts, residual_iqr, medians = curve
            curves.append((label, centers, values))
            csv_rows.append((sample_name, label, centers, values, counts, residual_iqr, medians))

        if curves:
            common_valid = np.ones_like(curves[0][2], dtype=bool)
            for _, _, values in curves:
                common_valid &= np.isfinite(values)
        else:
            common_valid = np.asarray([], dtype=bool)

        for label, centers, values in curves:
            ax.plot(
                centers[common_valid],
                values[common_valid],
                marker="o",
                linewidth=2.0,
                color=colors.get(label),
                linestyle=linestyles.get(label, "-"),
                label=label,
            )
        ax.set_title(sample_name)
        ax.set_xlabel(f"Original {feature_label(feature_name)}")
        ax.grid(alpha=0.25)
        apply_hep_style(ax)

    axes[0].set_ylabel(r"IQR(reco - original) / |median(original)|")
    axes[-1].legend(frameon=False, fontsize=10)
    fig.suptitle(f"jets: {feature_label(feature_name)} resolution", fontsize=16)
    fig.tight_layout()

    plot_dir = output_dir / "resolution_curves"
    plot_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_filename(feature_name)
    fig.savefig(plot_dir / f"jets_{stem}_google_vs_zephyr_resolution.png", dpi=180)
    plt.close(fig)

    write_curve_csv(csv_path=output_dir / "tables" / f"jets_{stem}_curves.csv", rows=csv_rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    if args.all_events and args.split != "val":
        raise ValueError("--all-events uses the shared val slot; do not override --split")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.force:
        cache_dir = output_dir / "cache_arrays"
        if cache_dir.exists():
            for path in cache_dir.glob("*.npz"):
                path.unlink()

    runs = {
        args.left_label: Path(args.left_run).resolve(),
        args.right_label: Path(args.right_run).resolve(),
    }
    for label, run_dir in runs.items():
        if not (run_dir / "full_config.yaml").exists():
            raise FileNotFoundError(f"{label} is missing full_config.yaml: {run_dir}")
    log.info(
        "Using the right run's saved datamodule for both models so evaluation "
        "events and traversal order are identical: %s",
        runs[args.right_label] / "full_config.yaml",
    )
    if args.all_events:
        log.info(
            "All-event mode enabled: train/val/test membership is ignored and "
            "every selected event is evaluated"
        )

    mc_files = selected_files(args.mc_dir, args.n_files, args.exclude_mc_pattern)
    data_files = selected_files(args.data_dir, args.n_files)
    if not mc_files:
        raise FileNotFoundError(f"No MC files found under {args.mc_dir}")
    if not data_files:
        raise FileNotFoundError(f"No real-data files found under {args.data_dir}")
    samples = {
        "MC sample": mc_files,
        "real-data sample": data_files,
    }
    log.info("MC files: %d", len(mc_files))
    log.info("real-data files: %d", len(data_files))

    features = (
        [name.strip() for name in args.features.split(",") if name.strip()]
        if args.features
        else default_features(next(iter(runs.values())))
    )
    log.info("Features: %s", features)

    for feature_name in features:
        log.info("Plotting %s", feature_name)
        plot_feature(
            feature_name=feature_name,
            runs=runs,
            samples=samples,
            output_dir=output_dir,
            args=args,
        )

    log.info("Wrote plots to %s", output_dir / "resolution_curves")
    log.info("Wrote tables to %s", output_dir / "tables")


if __name__ == "__main__":
    main()
