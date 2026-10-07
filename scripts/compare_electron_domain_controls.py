#!/usr/bin/env python3
"""Compare the completed electron domain-control tokenizer runs fairly.

The two controls are evaluated on identical ordered objects from fixed MC and
real-data files:

1. Feature competition: natural MC+data full features vs kinematics only.
2. Domain conflict: MC-only, natural MC+data, and data-only full features.
3. Feature-group controls: kinematics only vs kinematics+ID vs
   kinematics+isolation.

Each model keeps the preprocessing transformer used during its own training.
Reconstruction always follows the canonical ``model.encode`` and full decoder
path in ``analyze_vqvae_tokenizer.py``.  The script refuses to report results
unless the shared physical inputs agree object by object.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from collections import OrderedDict
from pathlib import Path

import hydra
import matplotlib
import numpy as np
import torch
from omegaconf import OmegaConf

from analyze_vqvae_tokenizer import (
    analysis_datamodule_cfg,
    apply_hep_style,
    binned_residual_iqr_over_median_truth,
    choose_device,
    codebook_counts,
    codebook_summary,
    collect_diagnostics_from_loader,
    dataloader_from_datamodule,
    feature_names_from_cfg,
    find_checkpoint,
    reconstruction_metrics,
    safe_filename,
    transform_list_and_cst_fn_from_cfg,
)
from heptokens.models.vq_vae import LitVqVae

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


log = logging.getLogger(__name__)


DEFAULT_NATURAL_RUN = (
    "results/atlas_event_tokenizers_0107_logstd_mc_realdata/"
    "electrons_logstd_dim8_cb4096_q4"
)
DEFAULT_MC_ONLY_RUN = (
    "results/atlas_event_tokenizers_1806_nonjet_logstd_capacity_scan/"
    "electrons_logstd_dim8_cb4096_q4"
)
DEFAULT_KINEMATICS_RUN = (
    "results/atlas_electron_domain_controls/"
    "electrons_kinematics_dim8_cb4096_q4_mcdata_offline"
)
DEFAULT_KINEMATICS_ID_RUN = (
    "results/atlas_electron_feature_group_controls/"
    "electrons_kinematics_id_dim8_cb4096_q4_mcdata"
)
DEFAULT_KINEMATICS_ISOLATION_RUN = (
    "results/atlas_electron_feature_group_controls/"
    "electrons_kinematics_isolation_dim8_cb4096_q4_mcdata"
)
DEFAULT_MC_ONLY_KINEMATICS_RUN = (
    "results/atlas_electron_feature_group_controls_mconly/"
    "electrons_kinematics_dim8_cb4096_q4_mconly"
)
DEFAULT_MC_ONLY_KINEMATICS_ID_RUN = (
    "results/atlas_electron_feature_group_controls_mconly/"
    "electrons_kinematics_id_dim8_cb4096_q4_mconly"
)
DEFAULT_MC_ONLY_KINEMATICS_ISOLATION_RUN = (
    "results/atlas_electron_feature_group_controls_mconly/"
    "electrons_kinematics_isolation_dim8_cb4096_q4_mconly"
)
DEFAULT_DATA_ONLY_RUN = (
    "results/atlas_electron_domain_controls/"
    "electrons_full_dim8_cb4096_q4_dataonly_offline"
)
DEFAULT_KINW2_RUN = (
    "results/atlas_electron_kinematic_weight_study/"
    "electrons_full_dim8_cb4096_q4_kinw2_mcdata"
)
DEFAULT_KINW4_RUN = (
    "results/atlas_electron_kinematic_weight_study/"
    "electrons_full_dim8_cb4096_q4_kinw4_mcdata"
)
DEFAULT_LARGER_CAPACITY_RUN = (
    "results/atlas_electron_kinematic_weight_study/"
    "electrons_full_dim16_cb4096_q4_larger_capacity_mcdata"
)
DEFAULT_FULL_LONGER_RUN = (
    "results/atlas_electron_full_training_controls/"
    "electrons_full_dim8_cb4096_q4_e60_mcdata"
)
DEFAULT_FULL_Q8_RUN = (
    "results/atlas_electron_full_training_controls/"
    "electrons_full_dim8_cb4096_q8_e20_mcdata"
)

COLORS = {
    "MC-only (full)": "#4C83F1",
    "MC+data (full)": "#F28E2B",
    "MC+data (full, e60)": "#B45F06",
    "MC+data (full, q8)": "#CC79A7",
    "MC+data (kinematics only)": "#8B5CF6",
    "MC+data (kinematics+ID)": "#D62728",
    "MC+data (kinematics+isolation)": "#7F7F7F",
    "MC-only (kinematics only)": "#6D45D9",
    "MC-only (kinematics+ID)": "#A51E22",
    "MC-only (kinematics+isolation)": "#555555",
    "Data-only (full)": "#2CA58D",
    "MC+data (2x kin loss)": "#D62728",
    "MC+data (4x kin loss)": "#7F7F7F",
    "MC+data (larger capacity)": "#8C564B",
}

LINESTYLES = {
    "MC+data (full, e60)": "--",
    "MC+data (full, q8)": ":",
    "MC-only (kinematics only)": "--",
    "MC-only (kinematics+ID)": "--",
    "MC-only (kinematics+isolation)": "--",
    "MC+data (2x kin loss)": "--",
    "MC+data (4x kin loss)": "--",
    "MC+data (larger capacity)": ":",
}

LEGACY_MODEL_LABELS = {
    "MC+data (full)": "Natural MC+data (full)",
    "MC+data (kinematics only)": "Natural MC+data (kinematics only)",
}

BINNED_RESOLUTION_FEATURES = (
    "pt",
    "LHMedium",
    "LHTight",
    "ptvarcone30",
    "topoetcone20",
)

PAIRWISE_DOMAIN_COMPARISONS = (
    ("Full", "MC-only (full)", "MC+data (full)"),
    (
        "Kinematics only",
        "MC-only (kinematics only)",
        "MC+data (kinematics only)",
    ),
    ("Kinematics+ID", "MC-only (kinematics+ID)", "MC+data (kinematics+ID)"),
    (
        "Kinematics+isolation",
        "MC-only (kinematics+isolation)",
        "MC+data (kinematics+isolation)",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare electron feature-competition and domain-conflict controls."
    )
    parser.add_argument("--natural-run", default=DEFAULT_NATURAL_RUN)
    parser.add_argument("--mc-only-run", default=DEFAULT_MC_ONLY_RUN)
    parser.add_argument("--kinematics-run", default=DEFAULT_KINEMATICS_RUN)
    parser.add_argument("--kinematics-id-run", default=DEFAULT_KINEMATICS_ID_RUN)
    parser.add_argument(
        "--kinematics-isolation-run",
        default=DEFAULT_KINEMATICS_ISOLATION_RUN,
    )
    parser.add_argument(
        "--mc-only-kinematics-run",
        default=DEFAULT_MC_ONLY_KINEMATICS_RUN,
    )
    parser.add_argument(
        "--mc-only-kinematics-id-run",
        default=DEFAULT_MC_ONLY_KINEMATICS_ID_RUN,
    )
    parser.add_argument(
        "--mc-only-kinematics-isolation-run",
        default=DEFAULT_MC_ONLY_KINEMATICS_ISOLATION_RUN,
    )
    parser.add_argument("--data-only-run", default=DEFAULT_DATA_ONLY_RUN)
    parser.add_argument("--kinw2-run", default=DEFAULT_KINW2_RUN)
    parser.add_argument("--kinw4-run", default=DEFAULT_KINW4_RUN)
    parser.add_argument("--larger-capacity-run", default=DEFAULT_LARGER_CAPACITY_RUN)
    parser.add_argument("--full-longer-run", default=DEFAULT_FULL_LONGER_RUN)
    parser.add_argument("--full-q8-run", default=DEFAULT_FULL_Q8_RUN)
    parser.add_argument(
        "--kinematic-study-runs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include the completed kinematic-weight/capacity study runs.",
    )
    parser.add_argument(
        "--mc-dir", default="/home/zephyr/Data/viviana/bnl-treasure/data/h5"
    )
    parser.add_argument(
        "--data-dir",
        default="/home/zephyr/Data/viviana/bnl-treasure/data/h5/realdata",
    )
    parser.add_argument(
        "--output-dir",
        default="results/atlas_electron_domain_controls/comparison",
    )
    parser.add_argument("--n-files", type=int, default=20)
    parser.add_argument("--num-events-per-file", type=int)
    parser.add_argument("--max-valid-objects", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--checkpoint-name", default="last.ckpt")
    parser.add_argument("--n-bins", type=int, default=14)
    parser.add_argument("--min-bin-count", type=int, default=50)
    parser.add_argument("--min-denominator", type=float, default=1e-8)
    parser.add_argument("--force", action="store_true", help="Ignore cached arrays.")
    return parser.parse_args()


def fixed_files(directory: str, n_files: int) -> list[str]:
    return [
        str(path)
        for path in sorted(Path(directory).glob("*.h5"))
        if path.is_file() and path.stat().st_size > 0
    ][:n_files]


def sample_files(args: argparse.Namespace) -> OrderedDict[str, list[str]]:
    samples: OrderedDict[str, list[str]] = OrderedDict(
        [
            ("MC sample", fixed_files(args.mc_dir, args.n_files)),
            ("real-data sample", fixed_files(args.data_dir, args.n_files)),
        ]
    )
    for label, files in samples.items():
        if not files:
            raise FileNotFoundError(f"No non-empty H5 files found for {label}")
    return samples


def explicit_checkpoint(run_dir: Path, checkpoint_name: str) -> Path:
    return find_checkpoint(
        run_dir,
        str((run_dir / "checkpoints" / checkpoint_name).resolve()),
    )


def cache_path(
    *,
    output_dir: Path,
    sample_label: str,
    model_label: str,
    run_dir: Path,
    checkpoint: Path,
    files: list[str],
    args: argparse.Namespace,
) -> Path:
    payload = "\n".join(
        [
            str(run_dir),
            str(checkpoint),
            str(checkpoint.stat().st_mtime_ns),
            f"split={args.split}",
            f"events={args.num_events_per_file}",
            f"objects={args.max_valid_objects}",
            "loader=shared_selection_run_specific_features_v1",
            *files,
        ]
    )
    digest = hashlib.sha1(payload.encode()).hexdigest()[:14]
    return (
        output_dir
        / "cache"
        / f"{safe_filename(sample_label)}_{safe_filename(model_label)}_{digest}.npz"
    )


def copy_model_input_definition(evaluation_cfg, model_cfg) -> None:
    """Keep shared event selection while loading the model's own feature set."""
    for key in ("object_collections", "object_type", "output_mode", "mask_input"):
        if key in model_cfg.datamodule:
            value = model_cfg.datamodule[key]
            evaluation_cfg.datamodule[key] = (
                OmegaConf.to_container(value, resolve=False)
                if OmegaConf.is_config(value)
                else value
            )
    transforms = model_cfg.datamodule.get("transforms", {})
    evaluation_cfg.datamodule.transforms = (
        OmegaConf.to_container(transforms, resolve=False)
        if OmegaConf.is_config(transforms)
        else transforms
    )


def collect_one(
    *,
    sample_label: str,
    model_label: str,
    run_dir: Path,
    reference_datamodule_cfg,
    files: list[str],
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    cfg_path = run_dir / "full_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    cfg = OmegaConf.load(cfg_path)
    checkpoint = explicit_checkpoint(run_dir, args.checkpoint_name)
    cached = cache_path(
        output_dir=output_dir,
        sample_label=sample_label,
        model_label=model_label,
        run_dir=run_dir,
        checkpoint=checkpoint,
        files=files,
        args=args,
    )
    legacy_label = LEGACY_MODEL_LABELS.get(model_label)
    if not cached.exists() and legacy_label:
        legacy_cached = cache_path(
            output_dir=output_dir,
            sample_label=sample_label,
            model_label=legacy_label,
            run_dir=run_dir,
            checkpoint=checkpoint,
            files=files,
            args=args,
        )
        if legacy_cached.exists():
            cached = legacy_cached
    if cached.exists() and not args.force:
        log.info("Reading cache %s", cached)
        item = np.load(cached, allow_pickle=True)
        return {
            "original": item["original"],
            "reconstruction": item["reconstruction"],
            "indices": item["indices"],
            "feature_names": [str(value) for value in item["feature_names"]],
            "codebook_size": int(item["codebook_size"]),
            "checkpoint": str(item["checkpoint"]),
        }

    _, inverse_transformer = transform_list_and_cst_fn_from_cfg(cfg)
    log.info("Loading %s for %s / %s", checkpoint, sample_label, model_label)
    model = LitVqVae.load_from_checkpoint(checkpoint, map_location=device)
    model.to(device)
    model.eval()

    evaluation_cfg = OmegaConf.create(
        {
            "datamodule": OmegaConf.to_container(
                reference_datamodule_cfg, resolve=False
            )
        }
    )
    copy_model_input_definition(evaluation_cfg, cfg)
    loader_args = argparse.Namespace(
        h5_files=list(files),
        num_events_per_file=args.num_events_per_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    datamodule_cfg = analysis_datamodule_cfg(evaluation_cfg, loader_args)
    datamodule_cfg.num_events = args.num_events_per_file
    datamodule = hydra.utils.instantiate(datamodule_cfg)
    loader = dataloader_from_datamodule(datamodule, args.split)

    original, reconstruction, indices, _ = collect_diagnostics_from_loader(
        model=model,
        loader=loader,
        cst_inverse_transformer=inverse_transformer,
        device=device,
        max_valid_objects=args.max_valid_objects,
    )
    feature_names = feature_names_from_cfg(cfg, original.shape[1])
    codebook_size = int(cfg.model.codebook_size)

    cached.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cached,
        original=original,
        reconstruction=reconstruction,
        indices=indices,
        feature_names=np.asarray(feature_names),
        codebook_size=np.asarray(codebook_size),
        checkpoint=np.asarray(str(checkpoint)),
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "original": original,
        "reconstruction": reconstruction,
        "indices": indices,
        "feature_names": feature_names,
        "codebook_size": codebook_size,
        "checkpoint": str(checkpoint),
    }


def feature_indices(item: dict) -> dict[str, int]:
    return {name.lower(): idx for idx, name in enumerate(item["feature_names"])}


def common_feature_names(*models: dict) -> list[str]:
    if not models:
        return []

    common = set(feature_indices(models[0]))
    for model in models[1:]:
        common &= set(feature_indices(model))

    return [
        name for name in models[0]["feature_names"] if name.lower() in common
    ]


def aligned_arrays(item: dict, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    indices = feature_indices(item)
    columns = [indices[name.lower()] for name in names]
    return item["original"][:, columns], item["reconstruction"][:, columns]


def verify_shared_inputs(
    sample_label: str,
    test_label: str,
    models: OrderedDict[str, dict],
    feature_names: list[str],
) -> None:
    reference_label, reference = next(iter(models.items()))
    reference_original, _ = aligned_arrays(reference, feature_names)
    for model_label, item in list(models.items())[1:]:
        original, _ = aligned_arrays(item, feature_names)
        if original.shape != reference_original.shape:
            raise RuntimeError(
                f"{test_label} / {sample_label}: ordered object shapes differ for "
                f"{reference_label} and {model_label}: "
                f"{reference_original.shape} vs {original.shape}"
            )
        finite_left = np.isfinite(reference_original)
        finite_right = np.isfinite(original)
        if not np.array_equal(finite_left, finite_right):
            raise RuntimeError(
                f"{test_label} / {sample_label}: physical finite masks differ for "
                f"{reference_label} and {model_label}"
            )
        finite = finite_left & finite_right
        left_finite = reference_original[finite]
        right_finite = original[finite]
        max_delta = (
            float(np.max(np.abs(left_finite - right_finite)))
            if finite.any()
            else 0.0
        )
        # Each run applies and then reverses its own fitted float32
        # standardization. The same physical object can therefore differ by a
        # few float32 ULPs after inverse transformation, especially for pT.
        # A relative tolerance keeps this identity check strict without
        # mistaking sub-MeV round-off for a different object ordering.
        same_objects = np.allclose(
            left_finite,
            right_finite,
            rtol=5e-6,
            atol=5e-6,
        )
        if not same_objects:
            raise RuntimeError(
                f"{test_label} / {sample_label}: models did not receive identical "
                f"ordered physical objects; max delta={max_delta:.6g}"
            )
    log.info(
        "%s / %s: verified %d identical ordered objects over %s",
        test_label,
        sample_label,
        len(reference_original),
        ", ".join(feature_names),
    )


def make_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        raise ValueError("Cannot define bins without finite values")
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(finite.min()), float(finite.max())
    if lo == hi:
        raise ValueError("Cannot define bins for a constant feature")
    return np.linspace(lo, hi, n_bins + 1)


def codebook_usage(item: dict) -> tuple[float, dict]:
    usage = codebook_summary(
        codebook_counts(item["indices"], int(item["codebook_size"]))
    )
    percentages = [
        float(value["percent_used"])
        for key, value in usage.items()
        if key.startswith("quantizer_")
    ]
    return float(np.mean(percentages)), usage


def evaluate_test(
    *,
    test_key: str,
    test_title: str,
    samples: OrderedDict[str, OrderedDict[str, dict]],
    feature_names: list[str],
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict]]:
    metrics_rows: list[dict] = []
    resolution_rows: list[dict] = []
    for sample_label, models in samples.items():
        verify_shared_inputs(sample_label, test_title, models, feature_names)
        reference_original, _ = aligned_arrays(next(iter(models.values())), feature_names)
        feature_lookup = {
            name.lower(): idx for idx, name in enumerate(feature_names)
        }
        binned_features = [
            name for name in BINNED_RESOLUTION_FEATURES if name in feature_lookup
        ]
        bins_by_feature = {}
        for name in binned_features:
            try:
                bins_by_feature[name] = make_bins(
                    reference_original[:, feature_lookup[name]],
                    args.n_bins,
                )
            except ValueError as exc:
                log.warning(
                    "%s / %s: skipping binned plot for %s: %s",
                    test_title,
                    sample_label,
                    name,
                    exc,
                )
        binned_features = list(bins_by_feature)

        for model_label, item in models.items():
            original, reconstruction = aligned_arrays(item, feature_names)
            metrics = reconstruction_metrics(original, reconstruction, feature_names)
            mean_used, usage = codebook_usage(item)
            for feature_name, values in metrics.items():
                row = {
                    "test": test_key,
                    "sample": sample_label,
                    "model": model_label,
                    "feature": feature_name,
                    "n": int(values["n"]),
                    "mae": float(values["mae"]),
                    "rmse": float(values["rmse"]),
                    "bias": float(values["bias"]),
                    "mean_codebook_used_percent": mean_used,
                }
                for quantizer_idx in range(item["indices"].shape[1]):
                    row[f"q{quantizer_idx}_used_percent"] = float(
                        usage[f"quantizer_{quantizer_idx}"]["percent_used"]
                    )
                metrics_rows.append(row)

            for feature_name in binned_features:
                feature_idx = feature_lookup[feature_name]
                centers, values, counts, medians, residual_iqrs = (
                    binned_residual_iqr_over_median_truth(
                        original[:, feature_idx],
                        reconstruction[:, feature_idx],
                        bins_by_feature[feature_name],
                        min_bin_count=args.min_bin_count,
                        min_denominator=args.min_denominator,
                    )
                )
                for center, value, count, median, residual_iqr in zip(
                    centers, values, counts, medians, residual_iqrs
                ):
                    resolution_rows.append(
                        {
                            "test": test_key,
                            "sample": sample_label,
                            "model": model_label,
                            "feature": feature_name,
                            "bin_center": float(center),
                            "relative_resolution": float(value),
                            "objects": int(count),
                            "median_original": float(median),
                            "residual_iqr": float(residual_iqr),
                        }
                    )
    return metrics_rows, resolution_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_feature_resolution(
    *,
    test_title: str,
    feature_name: str,
    samples: OrderedDict[str, OrderedDict[str, dict]],
    resolution_rows: list[dict],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(samples), figsize=(6.7 * len(samples), 5.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, sample_label in zip(axes, samples):
        for model_label in samples[sample_label]:
            selected = [
                row
                for row in resolution_rows
                if row["sample"] == sample_label
                and row["model"] == model_label
                and row["feature"] == feature_name
            ]
            x = np.asarray([row["bin_center"] for row in selected])
            y = np.asarray([row["relative_resolution"] for row in selected])
            valid = np.isfinite(y)
            ax.plot(
                x[valid],
                y[valid],
                marker="o",
                linewidth=2.0,
                label=model_label,
                color=COLORS.get(model_label),
                linestyle=LINESTYLES.get(model_label, "-"),
            )
        ax.set_title(sample_label, fontsize=18)
        x_label = r"Original electron $p_T$" if feature_name == "pt" else f"Original {feature_name}"
        ax.set_xlabel(x_label, fontsize=14)
        ax.grid(alpha=0.22)
        apply_hep_style(ax)
    axes[0].set_ylabel(r"IQR(reco - original) / |median(original)|", fontsize=13)
    legend_items = len(next(iter(samples.values()))) if samples else 0
    axes[-1].legend(frameon=False, fontsize=8, ncol=3 if legend_items > 6 else 2)
    fig.suptitle(test_title, fontsize=18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def pairwise_pt_delta_rows(
    *,
    resolution_rows: list[dict],
    samples: OrderedDict[str, OrderedDict[str, dict]],
) -> list[dict]:
    rows = []
    available_models = {
        sample_label: set(models)
        for sample_label, models in samples.items()
    }
    for sample_label, models in available_models.items():
        for comparison_label, mc_only_label, mcdata_label in PAIRWISE_DOMAIN_COMPARISONS:
            if mc_only_label not in models or mcdata_label not in models:
                continue
            mc_only_rows = {
                row["bin_center"]: row
                for row in resolution_rows
                if row["sample"] == sample_label
                and row["model"] == mc_only_label
                and row["feature"] == "pt"
            }
            mcdata_rows = {
                row["bin_center"]: row
                for row in resolution_rows
                if row["sample"] == sample_label
                and row["model"] == mcdata_label
                and row["feature"] == "pt"
            }
            for bin_center in sorted(set(mc_only_rows) & set(mcdata_rows)):
                mc_only_value = float(mc_only_rows[bin_center]["relative_resolution"])
                mcdata_value = float(mcdata_rows[bin_center]["relative_resolution"])
                if not (np.isfinite(mc_only_value) and np.isfinite(mcdata_value)):
                    continue
                delta = mcdata_value - mc_only_value
                rows.append(
                    {
                        "sample": sample_label,
                        "comparison": comparison_label,
                        "mc_only_model": mc_only_label,
                        "mcdata_model": mcdata_label,
                        "bin_center": float(bin_center),
                        "mc_only_relative_resolution": mc_only_value,
                        "mcdata_relative_resolution": mcdata_value,
                        "delta_mcdata_minus_mconly": delta,
                        "percent_delta_vs_mconly": (
                            100.0 * delta / mc_only_value
                            if mc_only_value != 0
                            else float("nan")
                        ),
                    }
                )
    return rows


def plot_pairwise_pt_deltas(
    *,
    samples: OrderedDict[str, OrderedDict[str, dict]],
    delta_rows: list[dict],
    output_path: Path,
) -> None:
    if not delta_rows:
        return
    fig, axes = plt.subplots(1, len(samples), figsize=(6.7 * len(samples), 5.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, sample_label in zip(axes, samples):
        for comparison_label, _, _ in PAIRWISE_DOMAIN_COMPARISONS:
            selected = [
                row
                for row in delta_rows
                if row["sample"] == sample_label
                and row["comparison"] == comparison_label
            ]
            if not selected:
                continue
            x = np.asarray([row["bin_center"] for row in selected])
            y = np.asarray([row["delta_mcdata_minus_mconly"] for row in selected])
            valid = np.isfinite(y)
            ax.plot(
                x[valid],
                y[valid],
                marker="o",
                linewidth=2.0,
                label=comparison_label,
            )
        ax.axhline(0.0, color="0.35", linewidth=1.1, linestyle="--")
        ax.set_title(sample_label, fontsize=18)
        ax.set_xlabel(r"Original electron $p_T$", fontsize=14)
        ax.grid(alpha=0.22)
        apply_hep_style(ax)
    axes[0].set_ylabel(
        r"$\Delta$ pT resolution (MC+data - MC-only)",
        fontsize=13,
    )
    axes[-1].legend(frameon=False, fontsize=9, ncol=1)
    fig.suptitle("MC+data minus MC-only pT binned resolution", fontsize=18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def finite_mean_binned(
    resolution_rows: list[dict],
    sample_label: str,
    model_label: str,
    feature_name: str,
) -> float:
    values = np.asarray(
        [
            row["relative_resolution"]
            for row in resolution_rows
            if row["sample"] == sample_label
            and row["model"] == model_label
            and row["feature"] == feature_name
        ],
        dtype=np.float64,
    )
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if len(finite) else float("nan")


def percent_lower(left: float, right: float) -> tuple[str, float]:
    if left <= right:
        return "left", 100.0 * (right - left) / right if right != 0 else 0.0
    return "right", 100.0 * (left - right) / left if left != 0 else 0.0


def best_model_with_improvement(
    models: list[str], values: list[float]
) -> tuple[str, float]:
    """Return the lowest finite result and its gain over the runner-up."""
    finite = [
        (model, float(value))
        for model, value in zip(models, values)
        if np.isfinite(value)
    ]
    if not finite:
        return "-", float("nan")
    finite.sort(key=lambda item: item[1])
    winner, best = finite[0]
    if len(finite) == 1:
        return winner, 0.0
    runner_up = finite[1][1]
    improvement = 100.0 * (runner_up - best) / runner_up if runner_up != 0 else 0.0
    return winner, improvement


def report_section(
    *,
    title: str,
    models: list[str],
    metrics_rows: list[dict],
    resolution_rows: list[dict],
) -> list[str]:
    lines = [f"# {title}", ""]
    samples = list(OrderedDict.fromkeys(row["sample"] for row in metrics_rows))
    for sample_label in samples:
        lines.extend([f"## {sample_label}", ""])
        lines.append(
            "| feature | metric | "
            + " | ".join(models)
            + " | better |"
        )
        lines.append("|---|---|" + "---:|" * len(models) + "---|")
        features = list(
            OrderedDict.fromkeys(
                row["feature"]
                for row in metrics_rows
                if row["sample"] == sample_label
            )
        )
        for feature_name in features:
            selected = {
                row["model"]: row
                for row in metrics_rows
                if row["sample"] == sample_label and row["feature"] == feature_name
            }
            for metric in ("mae", "rmse"):
                values = [float(selected[model][metric]) for model in models]
                winner, improvement = best_model_with_improvement(models, values)
                lines.append(
                    f"| {feature_name} | {metric.upper()} | "
                    + " | ".join(f"{value:.6g}" for value in values)
                    + f" | {winner} ({improvement:.1f}% lower) |"
                )

        available_binned_features = list(
            OrderedDict.fromkeys(
                row["feature"]
                for row in resolution_rows
                if row["sample"] == sample_label
            )
        )
        for feature_name in available_binned_features:
            binned = [
                finite_mean_binned(
                    resolution_rows, sample_label, model, feature_name
                )
                for model in models
            ]
            winner, improvement = best_model_with_improvement(models, binned)
            display_name = "pT" if feature_name == "pt" else feature_name
            lines.append(
                f"| {display_name} | mean binned resolution | "
                + " | ".join(f"{value:.6g}" for value in binned)
                + f" | {winner} ({improvement:.1f}% lower) |"
            )
        lines.append("")
    return lines


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    files_by_sample = sample_files(args)

    run_dirs = OrderedDict(
        [
            ("MC-only (full)", Path(args.mc_only_run).resolve()),
            ("MC+data (full)", Path(args.natural_run).resolve()),
            (
                "MC+data (kinematics only)",
                Path(args.kinematics_run).resolve(),
            ),
            (
                "MC+data (kinematics+ID)",
                Path(args.kinematics_id_run).resolve(),
            ),
            (
                "MC+data (kinematics+isolation)",
                Path(args.kinematics_isolation_run).resolve(),
            ),
            (
                "MC+data (full, e60)",
                Path(args.full_longer_run).resolve(),
            ),
            (
                "MC+data (full, q8)",
                Path(args.full_q8_run).resolve(),
            ),
            (
                "MC-only (kinematics only)",
                Path(args.mc_only_kinematics_run).resolve(),
            ),
            (
                "MC-only (kinematics+ID)",
                Path(args.mc_only_kinematics_id_run).resolve(),
            ),
            (
                "MC-only (kinematics+isolation)",
                Path(args.mc_only_kinematics_isolation_run).resolve(),
            ),
            ("Data-only (full)", Path(args.data_only_run).resolve()),
        ]
    )
    if args.kinematic_study_runs:
        run_dirs.update(
            [
                ("MC+data (2x kin loss)", Path(args.kinw2_run).resolve()),
                ("MC+data (4x kin loss)", Path(args.kinw4_run).resolve()),
                (
                    "MC+data (larger capacity)",
                    Path(args.larger_capacity_run).resolve(),
                ),
            ]
        )
    checkpoints = {}
    for label, run_dir in run_dirs.items():
        if not (run_dir / "full_config.yaml").exists():
            raise FileNotFoundError(f"Missing completed run for {label}: {run_dir}")
        checkpoints[label] = str(explicit_checkpoint(run_dir, args.checkpoint_name))

    reference_cfg = OmegaConf.load(run_dirs["MC+data (full)"] / "full_config.yaml")
    reference_datamodule_cfg = reference_cfg.datamodule

    all_outputs: OrderedDict[str, OrderedDict[str, dict]] = OrderedDict()
    for sample_label, files in files_by_sample.items():
        all_outputs[sample_label] = OrderedDict()
        for model_label, run_dir in run_dirs.items():
            all_outputs[sample_label][model_label] = collect_one(
                sample_label=sample_label,
                model_label=model_label,
                run_dir=run_dir,
                reference_datamodule_cfg=reference_datamodule_cfg,
                files=files,
                output_dir=output_dir,
                args=args,
                device=device,
            )

    mc_only_label = "MC-only (full)"
    natural_label = "MC+data (full)"
    full_longer_label = "MC+data (full, e60)"
    full_q8_label = "MC+data (full, q8)"
    kinematics_label = "MC+data (kinematics only)"
    kinematics_id_label = "MC+data (kinematics+ID)"
    kinematics_isolation_label = "MC+data (kinematics+isolation)"
    mc_only_kinematics_label = "MC-only (kinematics only)"
    mc_only_kinematics_id_label = "MC-only (kinematics+ID)"
    mc_only_kinematics_isolation_label = "MC-only (kinematics+isolation)"
    data_only_label = "Data-only (full)"
    kinematic_study_labels = [
        "MC+data (2x kin loss)",
        "MC+data (4x kin loss)",
        "MC+data (larger capacity)",
    ]
    tests = OrderedDict(
        [
            (
                "full_training_controls",
                {
                    "title": "Electron full-feature training controls",
                    "models": [
                        mc_only_label,
                        natural_label,
                        full_longer_label,
                        full_q8_label,
                        data_only_label,
                    ],
                },
            ),
            (
                "feature_competition",
                {
                    "title": "Electron feature-group controls",
                    "models": [
                        natural_label,
                        kinematics_label,
                        kinematics_id_label,
                        kinematics_isolation_label,
                        mc_only_kinematics_label,
                        mc_only_kinematics_id_label,
                        mc_only_kinematics_isolation_label,
                        *(
                            label
                            for label in kinematic_study_labels
                            if label in run_dirs
                        ),
                    ],
                },
            ),
            (
                "domain_conflict",
                {
                    "title": "Electron domain and feature-group controls",
                    "models": [
                        mc_only_label,
                        natural_label,
                        kinematics_label,
                        kinematics_id_label,
                        kinematics_isolation_label,
                        mc_only_kinematics_label,
                        mc_only_kinematics_id_label,
                        mc_only_kinematics_isolation_label,
                        data_only_label,
                        *(
                            label
                            for label in kinematic_study_labels
                            if label in run_dirs
                        ),
                    ],
                },
            ),
            (
                "id_feature_group",
                {
                    "title": "Electron ID feature-group controls",
                    "models": [
                        mc_only_label,
                        natural_label,
                        kinematics_id_label,
                        mc_only_kinematics_id_label,
                        data_only_label,
                    ],
                },
            ),
            (
                "isolation_feature_group",
                {
                    "title": "Electron isolation feature-group controls",
                    "models": [
                        mc_only_label,
                        natural_label,
                        kinematics_isolation_label,
                        mc_only_kinematics_isolation_label,
                        data_only_label,
                    ],
                },
            ),
        ]
    )
    if args.kinematic_study_runs:
        tests["kinematic_loss_weighting"] = {
            "title": "Electron full-feature kinematic-loss and capacity controls",
            "models": [
                mc_only_label,
                natural_label,
                data_only_label,
                "MC+data (2x kin loss)",
                "MC+data (4x kin loss)",
                "MC+data (larger capacity)",
            ],
        }

    manifest = {
        "runs": {label: str(path) for label, path in run_dirs.items()},
        "checkpoints": checkpoints,
        "samples": files_by_sample,
        "split": args.split,
        "max_valid_objects": args.max_valid_objects,
        "num_events_per_file": args.num_events_per_file,
        "binned_resolution_features": list(BINNED_RESOLUTION_FEATURES),
        "metric": "IQR(reco-original) / abs(median(original)) in fixed original-value bins",
        "encode_path": "canonical model.encode via collect_diagnostics_from_loader",
    }
    (output_dir / "comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    report_lines: list[str] = []
    for test_key, definition in tests.items():
        selected_samples: OrderedDict[str, OrderedDict[str, dict]] = OrderedDict()
        for sample_label, outputs in all_outputs.items():
            selected_samples[sample_label] = OrderedDict(
                (label, outputs[label]) for label in definition["models"]
            )
        first_models = next(iter(selected_samples.values()))
        shared_features = common_feature_names(*first_models.values())
        shared_feature_keys = [name.lower() for name in shared_features]
        if "pt" not in shared_feature_keys:
            raise RuntimeError(f"{test_key}: shared feature list has no pT")

        metrics_rows, resolution_rows = evaluate_test(
            test_key=test_key,
            test_title=definition["title"],
            samples=selected_samples,
            feature_names=shared_features,
            args=args,
        )
        write_csv(output_dir / f"{test_key}_metrics.csv", metrics_rows)
        write_csv(output_dir / f"{test_key}_binned_resolution.csv", resolution_rows)
        for feature_name in BINNED_RESOLUTION_FEATURES:
            if feature_name not in shared_feature_keys:
                continue
            feature_rows = [
                row for row in resolution_rows if row["feature"] == feature_name
            ]
            write_csv(
                output_dir / f"{test_key}_{feature_name}_binned_resolution.csv",
                feature_rows,
            )
            plot_feature_resolution(
                test_title=definition["title"],
                feature_name=feature_name,
                samples=selected_samples,
                resolution_rows=feature_rows,
                output_path=output_dir / f"{test_key}_{feature_name}_resolution.png",
            )
        if test_key == "domain_conflict":
            delta_rows = pairwise_pt_delta_rows(
                resolution_rows=resolution_rows,
                samples=selected_samples,
            )
            write_csv(output_dir / "domain_conflict_pt_mconly_vs_mcdata_delta.csv", delta_rows)
            plot_pairwise_pt_deltas(
                samples=selected_samples,
                delta_rows=delta_rows,
                output_path=output_dir / "domain_conflict_pt_mconly_vs_mcdata_delta.png",
            )
        report_lines.extend(
            report_section(
                title=definition["title"],
                models=definition["models"],
                metrics_rows=metrics_rows,
                resolution_rows=resolution_rows,
            )
        )
        report_lines.append("")

    (output_dir / "comparison_report.md").write_text(
        "\n".join(report_lines).rstrip() + "\n"
    )
    log.info("Wrote both domain-control comparisons to %s", output_dir)


if __name__ == "__main__":
    main()
