#!/usr/bin/env python3
"""Plot fair MC-only vs MC+data tokenizer resolution curves.

The y-axis is IQR(reconstructed - original) / |median(original)| in bins of
the original feature value. Each panel compares the MC-only tokenizer to the
MC+data tokenizer on a fixed evaluation sample.
"""

from __future__ import annotations

import argparse
import hashlib
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

from analyze_vqvae_tokenizer import (
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
from heptokens.models.vq_vae import LitVqVae

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
        "mc_only": "results/atlas_event_tokenizers_1806_nonjet_logstd_capacity_scan/taus_logstd_dim8_cb8192_q4",
        "mcdata": "results/atlas_event_tokenizers_0107_logstd_mc_realdata/taus_logstd_dim8_cb8192_q4",
    },
    "tracks": {
        "mc_only": "results/atlas_event_tokenizers_1906_tracks_logstd_tests/tracks_logstd_no_ndoflog_dim8_cb8192_q4",
        "mcdata": "results/atlas_event_tokenizers_0107_logstd_mc_realdata/tracks_logstd_dim8_cb8192_q4",
    },
}

DEFAULT_WEEKEND_RUNS = {
    "jets": {
        "MC+data q6": (
            "results/atlas_object_quantizer_weekend_controls/"
            "jets_full_dim8_cb4096_q6_e20_mcdata"
        ),
        "MC+data q8": (
            "results/atlas_object_quantizer_weekend_controls/"
            "jets_full_dim8_cb4096_q8_e20_mcdata"
        ),
        "MC+data q8, cb2048": (
            "results/atlas_object_quantizer_weekend_controls/"
            "jets_full_dim8_cb2048_q8_e20_mcdata"
        ),
    },
    "muons": {
        "MC+data q6": (
            "results/atlas_object_quantizer_weekend_controls/"
            "muons_full_dim8_cb4096_q6_e20_mcdata"
        ),
        "MC+data q8": (
            "results/atlas_object_quantizer_weekend_controls/"
            "muons_full_dim8_cb4096_q8_e20_mcdata"
        ),
        "MC+data q8, cb2048": (
            "results/atlas_object_quantizer_weekend_controls/"
            "muons_full_dim8_cb2048_q8_e20_mcdata"
        ),
    },
    "photons": {
        "MC+data q6": (
            "results/atlas_object_quantizer_weekend_controls/"
            "photons_full_dim8_cb4096_q6_e20_mcdata"
        ),
        "MC+data q8, cb2048": (
            "results/atlas_object_quantizer_weekend_controls/"
            "photons_full_dim8_cb2048_q8_e20_mcdata"
        ),
        # photons_full_dim8_cb4096_q8_e20_mcdata is intentionally omitted:
        # it did not finish.
    },
    # taus_full_dim8_cb8192_q6_e20_mcdata is intentionally omitted:
    # it did not finish. Add completed tau/track controls here when available.
}

DEFAULT_FEATURES = {
    "jets": ["pt", "GN2_pc"],
    "electrons": ["pt", "ptvarcone30", "topoetcone20"],
    "muons": ["pt", "ptvarcone30", "topoetcone20"],
    "photons": ["pt", "ptcone20", "topoetcone20"],
    "taus": ["pt", "RNNJetScore"],
    "tracks": ["pt", "d0", "z0", "nDoF", "qOverP"],
}

COLORS = {
    "MC-only tokenizer": "#4C83F1",
    "MC+data tokenizer": "#FF9F1C",
    "MC+data q6": "#B45F06",
    "MC+data q8": "#CC79A7",
    "MC+data q8, cb2048": "#009E73",
}

LINESTYLES = {
    "MC-only tokenizer": "-",
    "MC+data tokenizer": "-",
    "MC+data q6": "--",
    "MC+data q8": ":",
    "MC+data q8, cb2048": "-.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot MC-only vs MC+data tokenizer binned resolution curves."
    )
    parser.add_argument("--mc-dir", default="/home/zephyr/Data/viviana/bnl-treasure/data/h5")
    parser.add_argument(
        "--data-dir",
        default="/home/zephyr/Data/viviana/bnl-treasure/data/h5/realdata",
    )
    parser.add_argument(
        "--output-dir",
        default="results/fair_mc_vs_mcdata_eval/resolution_plots",
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        default=["jets", "electrons", "muons", "photons", "taus", "tracks"],
        help="Objects to plot for now. Taus/tracks can be added later.",
    )
    parser.add_argument(
        "--features",
        action="append",
        default=[],
        metavar="OBJECT=FEATURE1,FEATURE2",
        help="Override features for one object. May be repeated.",
    )
    parser.add_argument(
        "--include-weekend-controls",
        action="store_true",
        help=(
            "Overlay completed atlas_object_quantizer_weekend_controls runs "
            "for each selected object."
        ),
    )
    parser.add_argument(
        "--no-default-runs",
        action="store_true",
        help=(
            "Do not include the built-in MC-only/MC+data baseline runs. Use this "
            "with --run when comparing a custom set of tokenizers."
        ),
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="OBJECT:LABEL=RUN_DIR",
        help=(
            "Add one extra run to an object. May be repeated, e.g. "
            "--run 'jets:MC+data q6=results/.../jets_full_dim8_cb4096_q6_e20_mcdata'."
        ),
    )
    parser.add_argument(
        "--run-preprocessor",
        action="append",
        default=[],
        metavar="OBJECT:LABEL=JOBLIB",
        help=(
            "Override the saved preprocessor for one --run. The OBJECT and LABEL "
            "must exactly match the corresponding --run entry. May be repeated."
        ),
    )
    parser.add_argument(
        "--mcdata-run",
        action="append",
        default=[],
        metavar="OBJECT=RUN_DIR",
        help="Override the baseline MC+data run directory for one object.",
    )
    parser.add_argument(
        "--mconly-run",
        action="append",
        default=[],
        metavar="OBJECT=RUN_DIR",
        help="Override the baseline MC-only run directory for one object.",
    )
    parser.add_argument("--n-files", type=int, default=20)
    parser.add_argument(
        "--exclude-mc-pattern",
        default="",
        help="MC filename glob to exclude, e.g. DAOD_PHYSLITE.370016*.",
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        choices=["mc", "realdata", "mixed"],
        default=["mc", "realdata"],
        help=(
            "Evaluation samples to plot. mixed uses the same selected MC files "
            "plus the same selected real-data files."
        ),
    )
    parser.add_argument("--max-valid-objects", type=int, default=1_000_000)
    parser.add_argument("--num-events-per-file", type=int)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--checkpoint",
        choices=["best.ckpt", "last.ckpt"],
        default="last.ckpt",
        help="Checkpoint loaded from every tokenizer run (default: last.ckpt).",
    )
    parser.add_argument("--n-bins", type=int, default=14)
    parser.add_argument("--min-bin-count", type=int, default=50)
    parser.add_argument("--min-denominator", type=float, default=1e-8)
    return parser.parse_args()


def parse_feature_overrides(overrides: list[str]) -> dict[str, list[str]]:
    features = {key: list(value) for key, value in DEFAULT_FEATURES.items()}
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Expected OBJECT=FEATURE1,FEATURE2, got {override}")
        object_name, names = override.split("=", 1)
        features[object_name] = [name.strip() for name in names.split(",") if name.strip()]
    return features


def parse_extra_run_overrides(overrides: list[str]) -> dict[str, dict[str, str]]:
    parsed: dict[str, dict[str, str]] = {}
    for override in overrides:
        if ":" not in override or "=" not in override:
            raise ValueError(f"Expected OBJECT:LABEL=RUN_DIR, got {override}")
        object_name, rest = override.split(":", 1)
        label, run_dir = rest.split("=", 1)
        object_name = object_name.strip()
        label = label.strip()
        run_dir = run_dir.strip()
        if not object_name or not label or not run_dir:
            raise ValueError(f"Expected OBJECT:LABEL=RUN_DIR, got {override}")
        parsed.setdefault(object_name, {})[label] = run_dir
    return parsed


def parse_object_run_overrides(overrides: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Expected OBJECT=RUN_DIR, got {override}")
        object_name, run_dir = override.split("=", 1)
        object_name = object_name.strip()
        run_dir = run_dir.strip()
        if not object_name or not run_dir:
            raise ValueError(f"Expected OBJECT=RUN_DIR, got {override}")
        parsed[object_name] = run_dir
    return parsed


def fixed_files(directory: str, n_files: int, exclude_pattern: str = "") -> list[str]:
    files = sorted(
        path
        for path in Path(directory).glob("*.h5")
        if path.is_file() and path.stat().st_size > 0
    )
    if exclude_pattern:
        files = [
            path
            for path in files
            if not path.match(f"*/{exclude_pattern}") and not path.match(exclude_pattern)
        ]
    return [str(path) for path in files[:n_files]]


def feature_index(feature_names: list[str], feature_name: str) -> int | None:
    lower = feature_name.lower()
    for idx, actual_name in enumerate(feature_names):
        if actual_name.lower() == lower:
            return idx
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
    preprocessor_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    checkpoint = find_checkpoint(
        run_dir,
        str(run_dir / "checkpoints" / checkpoint_name),
    )
    checkpoint_stat = checkpoint.stat()

    resolved_preprocessor = None
    preprocessor_sha256 = None
    if preprocessor_path:
        resolved_preprocessor = Path(preprocessor_path).resolve()
        if not resolved_preprocessor.is_file():
            raise FileNotFoundError(
                f"Explicit preprocessor does not exist: {resolved_preprocessor}"
            )
        preprocessor_sha256 = hashlib.sha256(
            resolved_preprocessor.read_bytes()
        ).hexdigest()

    cache_dir = output_dir / "cache_arrays"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_input = "\n".join(
        h5_files
        + [
            f"num_events_per_file={num_events_per_file}",
            f"checkpoint={checkpoint.resolve()}",
            f"checkpoint_size={checkpoint_stat.st_size}",
            f"checkpoint_mtime_ns={checkpoint_stat.st_mtime_ns}",
            f"preprocessor={resolved_preprocessor}",
            f"preprocessor_sha256={preprocessor_sha256}",
            "analysis_path=canonical_saved_datamodule_per_run_preprocessor_v2",
        ]
    )
    file_digest = hashlib.sha1(cache_input.encode("utf-8")).hexdigest()[:12]
    cache_name = (
        f"{safe_filename(sample_name)}_{safe_filename(model_name)}_"
        f"{safe_filename(run_dir.name)}_{split}_{len(h5_files)}files_"
        f"{file_digest}_{max_valid_objects}_canonical_v1.npz"
    )
    cache_path = cache_dir / cache_name
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        return cached["original"], cached["reconstruction"], list(cached["feature_names"])

    cfg = OmegaConf.load(run_dir / "full_config.yaml")
    if resolved_preprocessor is not None:
        OmegaConf.update(
            cfg,
            "datamodule.transforms.preprocess.cst_fn.filename",
            str(resolved_preprocessor),
            force_add=True,
        )
        log.info("Using explicit preprocessor %s for %s", resolved_preprocessor, model_name)
    _, cst_inverse_transformer = transform_list_and_cst_fn_from_cfg(cfg)
    log.info("Loading %s", checkpoint)
    model = LitVqVae.load_from_checkpoint(checkpoint, map_location=device)
    model.to(device)
    model.eval()

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


def make_bins(values: np.ndarray, n_bins: int) -> np.ndarray | None:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return None
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.min(finite)), float(np.max(finite))
    if lo == hi:
        return None
    return np.linspace(lo, hi, n_bins + 1)


def collect_curve(
    *,
    object_name: str,
    feature_name: str,
    sample_name: str,
    model_label: str,
    run_dir: Path,
    h5_files: list[str],
    output_dir: Path,
    split: str,
    max_valid_objects: int,
    num_events_per_file: int | None,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    checkpoint_name: str,
    bins: np.ndarray,
    min_bin_count: int,
    min_denominator: float,
    preprocessor_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    original, reconstruction, feature_names = load_or_collect_arrays(
        run_dir=run_dir,
        sample_name=sample_name,
        model_name=model_label,
        h5_files=h5_files,
        output_dir=output_dir,
        split=split,
        max_valid_objects=max_valid_objects,
        num_events_per_file=num_events_per_file,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        checkpoint_name=checkpoint_name,
        preprocessor_path=preprocessor_path,
    )
    idx = feature_index(feature_names, feature_name)
    if idx is None:
        log.warning("Skipping %s/%s: missing feature %s", object_name, model_label, feature_name)
        return None

    centers, values, _, _, _ = binned_residual_iqr_over_median_truth(
        original[:, idx],
        reconstruction[:, idx],
        bins,
        min_bin_count=min_bin_count,
        min_denominator=min_denominator,
    )
    return centers, values


def plot_object_feature(
    *,
    object_name: str,
    feature_name: str,
    run_dirs: dict[str, Path],
    samples: dict[str, list[str]],
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    run_preprocessors: dict[str, str],
) -> None:
    # Use the first configured run arrays for each sample to define common bin
    # edges. This is the MC-only baseline for the default comparison, or the
    # first --run when --no-default-runs is used.
    reference_label, reference_run_dir = next(iter(run_dirs.items()))
    sample_bins: dict[str, np.ndarray] = {}
    for sample_name, h5_files in samples.items():
        original, _, feature_names = load_or_collect_arrays(
            run_dir=reference_run_dir,
            sample_name=sample_name,
            model_name=reference_label,
            h5_files=h5_files,
            output_dir=output_dir,
            split=args.split,
            max_valid_objects=args.max_valid_objects,
            num_events_per_file=args.num_events_per_file,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            checkpoint_name=args.checkpoint,
            preprocessor_path=run_preprocessors.get(reference_label),
        )
        idx = feature_index(feature_names, feature_name)
        if idx is None:
            log.warning("Skipping %s/%s: feature missing", object_name, feature_name)
            return
        bins = make_bins(original[:, idx], args.n_bins)
        if bins is None:
            log.warning("Skipping %s/%s: no useful range", object_name, feature_name)
            return
        sample_bins[sample_name] = bins

    n_samples = len(samples)
    fig_width = 6.5 * n_samples
    fig, axes = plt.subplots(1, n_samples, figsize=(fig_width, 5.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (sample_name, h5_files) in zip(axes, samples.items()):
        curves = []
        for model_label, run_dir in run_dirs.items():
            curve = collect_curve(
                object_name=object_name,
                feature_name=feature_name,
                sample_name=sample_name,
                model_label=model_label,
                run_dir=run_dir,
                h5_files=h5_files,
                output_dir=output_dir,
                split=args.split,
                max_valid_objects=args.max_valid_objects,
                num_events_per_file=args.num_events_per_file,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                checkpoint_name=args.checkpoint,
                bins=sample_bins[sample_name],
                min_bin_count=args.min_bin_count,
                min_denominator=args.min_denominator,
                preprocessor_path=run_preprocessors.get(model_label),
            )
            if curve is None:
                continue
            curves.append((model_label, curve[0], curve[1]))

        if curves:
            # Keep the x-range identical for all curves in this panel. Individual
            # models can have NaNs in some bins, but showing different x ranges
            # makes the comparison look like different samples were used.
            common_valid = np.ones_like(curves[0][2], dtype=bool)
            for _, _, values in curves:
                common_valid &= np.isfinite(values)

        for model_label, centers, values in curves:
            ax.plot(
                centers[common_valid],
                values[common_valid],
                marker="o",
                linewidth=2.0,
                color=COLORS.get(model_label),
                label=model_label,
                linestyle=LINESTYLES.get(model_label, "-"),
            )

        label = feature_label(feature_name)
        ax.set_title(sample_name)
        ax.set_xlabel(f"Original {label}")
        ax.grid(alpha=0.25)
        apply_hep_style(ax)

    axes[0].set_ylabel(r"IQR(reco - original) / |median(original)|")
    axes[-1].legend(frameon=False, fontsize=10)
    fig.suptitle(f"{object_name}: {feature_label(feature_name)} resolution", fontsize=16)
    fig.tight_layout()

    plot_dir = output_dir / "resolution_curves"
    plot_dir.mkdir(parents=True, exist_ok=True)
    sample_stem = "_".join(safe_filename(name.lower().replace(" sample", "")) for name in samples)
    fig.savefig(
        plot_dir
        / (
            f"{safe_filename(object_name)}_{safe_filename(feature_name)}_"
            f"{sample_stem}_resolution.png"
        ),
        dpi=180,
    )
    plt.close(fig)


def selected_samples(args: argparse.Namespace) -> dict[str, list[str]]:
    mc_files = fixed_files(args.mc_dir, args.n_files, args.exclude_mc_pattern)
    data_files = fixed_files(args.data_dir, args.n_files)

    samples: dict[str, list[str]] = {}
    for sample in args.samples:
        if sample == "mc":
            samples["MC sample"] = mc_files
        elif sample == "realdata":
            samples["real-data sample"] = data_files
        elif sample == "mixed":
            samples["mixed MC+data sample"] = mc_files + data_files
        else:
            raise ValueError(sample)
    return samples


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    features = parse_feature_overrides(args.features)
    extra_run_overrides = parse_extra_run_overrides(args.run)
    run_preprocessor_overrides = parse_extra_run_overrides(args.run_preprocessor)
    mcdata_run_overrides = parse_object_run_overrides(args.mcdata_run)
    mconly_run_overrides = parse_object_run_overrides(args.mconly_run)
    device = choose_device(args.device)
    log.info("Using %s for every tokenizer run", args.checkpoint)

    samples = selected_samples(args)
    for sample_name, files in samples.items():
        if not files:
            raise FileNotFoundError(f"No H5 files found for {sample_name}")
        log.info("%s: using %d files", sample_name, len(files))

    for object_name in args.objects:
        if object_name not in DEFAULT_RUNS:
            log.warning("Skipping unknown object %s", object_name)
            continue

        run_dirs = {}
        if not args.no_default_runs:
            run_dirs = {
                "MC-only tokenizer": Path(
                    mconly_run_overrides.get(object_name, DEFAULT_RUNS[object_name]["mc_only"])
                ).resolve(),
                "MC+data tokenizer": Path(
                    mcdata_run_overrides.get(object_name, DEFAULT_RUNS[object_name]["mcdata"])
                ).resolve(),
            }
        if args.include_weekend_controls:
            for label, run_dir in DEFAULT_WEEKEND_RUNS.get(object_name, {}).items():
                run_dirs[label] = Path(run_dir).resolve()
        for label, run_dir in extra_run_overrides.get(object_name, {}).items():
            run_dirs[label] = Path(run_dir).resolve()
        if not run_dirs:
            log.warning("Skipping %s because no run directories were configured", object_name)
            continue

        required_missing = [
            str(path)
            for label, path in run_dirs.items()
            if label in {"MC-only tokenizer", "MC+data tokenizer"}
            and not (path / "full_config.yaml").exists()
        ]
        if required_missing:
            log.warning(
                "Skipping %s because baseline run directories are missing: %s",
                object_name,
                required_missing,
            )
            continue
        for label, path in list(run_dirs.items()):
            if not (path / "full_config.yaml").exists():
                log.warning("Skipping %s / %s because run is missing: %s", object_name, label, path)
                run_dirs.pop(label)

        for feature_name in features.get(object_name, []):
            log.info("Plotting %s / %s", object_name, feature_name)
            plot_object_feature(
                object_name=object_name,
                feature_name=feature_name,
                run_dirs=run_dirs,
                samples=samples,
                output_dir=output_dir,
                args=args,
                device=device,
                run_preprocessors=run_preprocessor_overrides.get(object_name, {}),
            )

    log.info("Wrote plots to %s", output_dir / "resolution_curves")


if __name__ == "__main__":
    main()
