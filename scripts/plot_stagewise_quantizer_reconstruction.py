#!/usr/bin/env python3
"""Compare cumulative residual-VQ reconstruction after each quantizer stage.

For a q4 tokenizer, the script decodes four cumulative latent representations:
q0, q0+q1, q0+q1+q2, and q0+q1+q2+q3. It compares MC-only and
MC+data tokenizers on the same fixed H5 sample and shared feature bins.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import hydra
import matplotlib
import numpy as np
import torch
from omegaconf import OmegaConf

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_vqvae_tokenizer import (  # noqa: E402
    analysis_datamodule_cfg,
    apply_hep_style,
    binned_residual_iqr_over_median_truth,
    choose_device,
    dataloader_from_datamodule,
    feature_label,
    feature_names_from_cfg,
    find_checkpoint,
    safe_filename,
    to_device,
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
        "mc_only": "results/atlas_event_tokenizers_1806_nonjet_logstd_capacity_scan/taus_logstd_dim8_cb8192_q4",
        "mcdata": "results/atlas_event_tokenizers_0107_logstd_mc_realdata/taus_logstd_dim8_cb8192_q4",
    },
    "tracks": {
        "mc_only": "results/atlas_event_tokenizers_1906_tracks_logstd_tests/tracks_logstd_no_ndoflog_dim8_cb8192_q4",
        "mcdata": "results/atlas_event_tokenizers_0107_logstd_mc_realdata/tracks_logstd_dim8_cb8192_q4",
    },
}

MODEL_COLORS = {
    "MC-only": "#4C83F1",
    "MC+data": "#FF9F1C",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot cumulative reconstruction quality after each residual-VQ stage."
    )
    parser.add_argument("--object", default="electrons", choices=sorted(DEFAULT_RUNS))
    parser.add_argument("--feature", default="pt")
    parser.add_argument("--mc-only-run-dir")
    parser.add_argument("--mcdata-run-dir")
    parser.add_argument("--mc-dir", default="/home/zephyr/Data/viviana/bnl-treasure/data/h5")
    parser.add_argument(
        "--data-dir",
        default="/home/zephyr/Data/viviana/bnl-treasure/data/h5/realdata",
    )
    parser.add_argument("--sample", choices=["mc", "data", "mixed"], default="mc")
    parser.add_argument("--h5-files", nargs="+")
    parser.add_argument("--n-files", type=int, default=20)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-events-per-file", type=int)
    parser.add_argument("--max-valid-objects", type=int, default=500_000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--n-bins", type=int, default=14)
    parser.add_argument(
        "--min-bin-count",
        type=int,
        default=50,
        help="Nominal minimum count; lower-statistics displayed points are marked hollow.",
    )
    parser.add_argument(
        "--tail-min-bin-count",
        type=int,
        default=5,
        help="Absolute minimum count needed to display a tail-bin resolution.",
    )
    parser.add_argument("--min-denominator", type=float, default=1e-8)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=200,
        help="Bootstrap replicas used for 68%% resolution intervals; 0 disables them.",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=12345)
    parser.add_argument(
        "--feature-percentile-range",
        default="1,99",
        help="Percentile range used for shared feature bins.",
    )
    parser.add_argument(
        "--space",
        choices=["physical", "standardized"],
        default="physical",
        help="Evaluate after inverse preprocessing or in standardized training space.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/fair_mc_vs_mcdata_eval/stagewise_reconstruction",
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
    return sorted(
        fixed_files(args.mc_dir, args.n_files)
        + fixed_files(args.data_dir, args.n_files)
    )


def feature_index(feature_names: list[str], feature_name: str) -> int:
    target = feature_name.lower()
    for idx, name in enumerate(feature_names):
        if name.lower() == target:
            return idx
    raise ValueError(f"Feature {feature_name!r} not found in {feature_names}")


def stage_label(stage_idx: int, num_quantizers: int) -> str:
    label = "+".join(f"q{idx}" for idx in range(stage_idx + 1))
    if stage_idx == num_quantizers - 1:
        label += " (all)"
    return label


def cumulative_quantized_embeddings(
    model: LitVqVae,
    batch: dict[str, torch.Tensor],
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Capture every layer output during the model's official encode path."""
    layers = getattr(model.vector_quantization, "layers", None)
    if layers is None or len(layers) == 0:
        raise RuntimeError("The loaded tokenizer does not expose ResidualVQ layers")

    layer_quantized: list[torch.Tensor] = []
    encoder_outputs: list[torch.Tensor] = []

    def capture_quantized(_module, _inputs, output) -> None:
        if not isinstance(output, tuple) or len(output) < 1:
            raise RuntimeError("Unexpected residual quantizer layer output")
        layer_quantized.append(output[0])

    def capture_encoder_output(_module, _inputs, output) -> None:
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("Unexpected encoder output type")
        encoder_outputs.append(output)

    handles = [layer.register_forward_hook(capture_quantized) for layer in layers]
    handles.append(model.encoder.register_forward_hook(capture_encoder_output))
    try:
        full_z_q, indices, _ = model.encode(batch)
    finally:
        for handle in handles:
            handle.remove()

    if len(layer_quantized) != len(layers):
        raise RuntimeError(
            f"Captured {len(layer_quantized)} quantizer outputs, expected {len(layers)}"
        )
    if len(encoder_outputs) != 1:
        raise RuntimeError(
            f"Captured {len(encoder_outputs)} encoder outputs, expected exactly one"
        )

    cumulative = torch.zeros_like(full_z_q)
    stage_embeddings = []
    for quantized in layer_quantized:
        cumulative = cumulative + quantized
        stage_embeddings.append(cumulative.clone())

    embedding_delta = float(
        torch.max(torch.abs(stage_embeddings[-1] - full_z_q)).detach().cpu()
    )
    if embedding_delta > 1e-5:
        raise RuntimeError(
            "Captured cumulative embedding does not match model.encode(): "
            f"max |delta|={embedding_delta:.6g}"
        )
    return stage_embeddings, full_z_q, indices, encoder_outputs[0]


def convert_feature_space(
    values: np.ndarray,
    *,
    feature_idx: int,
    inverse_transformer,
    space: str,
) -> np.ndarray:
    values64 = np.asarray(values, dtype=np.float64)
    if space == "physical" and inverse_transformer is not None:
        with np.errstate(over="ignore", invalid="ignore"):
            values64 = inverse_transformer.inverse_transform(values64)
    return np.asarray(values64[:, feature_idx], dtype=np.float64)


def load_model_bundle(
    *,
    label: str,
    run_dir: Path,
    device: torch.device,
) -> dict:
    cfg = OmegaConf.load(run_dir / "full_config.yaml")
    _, inverse_transformer = transform_list_and_cst_fn_from_cfg(cfg)
    checkpoint = find_checkpoint(run_dir, None)
    log.info("Loading %s checkpoint: %s", label, checkpoint)
    model = LitVqVae.load_from_checkpoint(checkpoint, map_location=device)
    model.to(device)
    model.eval()
    return {
        "cfg": cfg,
        "inverse_transformer": inverse_transformer,
        "checkpoint": str(checkpoint),
        "model": model,
    }


def _physical_from_standardized(values: np.ndarray, transformer) -> np.ndarray:
    values64 = np.asarray(values, dtype=np.float64)
    if transformer is None:
        return values64
    with np.errstate(over="ignore", invalid="ignore"):
        return np.asarray(transformer.inverse_transform(values64), dtype=np.float64)


def _standardized_from_physical(values: np.ndarray, transformer) -> np.ndarray:
    values64 = np.asarray(values, dtype=np.float64)
    if transformer is None:
        return values64
    return np.asarray(transformer.transform(values64), dtype=np.float64)


def collect_shared_stage_reconstructions(
    *,
    bundles: dict[str, dict],
    h5_files: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, dict]:
    """Feed the exact same ordered physical objects to both tokenizers."""
    reference_bundle = bundles["MC-only"]
    cfg = reference_bundle["cfg"]
    loader_args = SimpleNamespace(
        h5_files=h5_files,
        num_events_per_file=args.num_events_per_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    datamodule_cfg = analysis_datamodule_cfg(cfg, loader_args)
    datamodule = hydra.utils.instantiate(datamodule_cfg)
    loader = dataloader_from_datamodule(datamodule, args.split)

    results = {
        label: {
            "originals": [],
            "indices": [],
            "continuous_reconstructions": [],
            "stage_reconstructions": None,
            "final_embedding_max_delta": 0.0,
            "full_decode_max_delta": 0.0,
        }
        for label in bundles
    }
    reference_feature_names: list[str] | None = None
    feature_idx: int | None = None
    n_seen = 0

    with torch.no_grad():
        for reference_batch in loader:
            reference_batch = to_device(reference_batch, device)
            mask = reference_batch["mask"].bool()
            reference_valid = (
                reference_batch["csts"][mask].detach().cpu().double().numpy()
            )

            if reference_feature_names is None:
                n_features = reference_batch["csts"].shape[-1]
                reference_feature_names = feature_names_from_cfg(cfg, n_features)
                feature_idx = feature_index(reference_feature_names, args.feature)
                for label, bundle in bundles.items():
                    names = feature_names_from_cfg(bundle["cfg"], n_features)
                    if [name.lower() for name in names] != [
                        name.lower() for name in reference_feature_names
                    ]:
                        raise RuntimeError(
                            f"{label} feature order differs from MC-only: "
                            f"{names} != {reference_feature_names}"
                        )

            physical_valid = _physical_from_standardized(
                reference_valid,
                reference_bundle["inverse_transformer"],
            )

            for label, bundle in bundles.items():
                model = bundle["model"]
                transformer = bundle["inverse_transformer"]
                model_valid = _standardized_from_physical(
                    physical_valid,
                    transformer,
                )
                model_batch = dict(reference_batch)
                model_batch["csts"] = reference_batch["csts"].clone()
                model_batch["csts"][mask] = torch.as_tensor(
                    model_valid,
                    dtype=model_batch["csts"].dtype,
                    device=device,
                )

                (
                    stage_embeddings,
                    full_z_q,
                    indices,
                    z_e,
                ) = cumulative_quantized_embeddings(model, model_batch)
                results[label]["indices"].append(
                    indices[mask].detach().cpu().long().numpy()
                )
                if results[label]["stage_reconstructions"] is None:
                    results[label]["stage_reconstructions"] = [
                        [] for _ in stage_embeddings
                    ]

                embedding_delta = float(
                    torch.max(torch.abs(stage_embeddings[-1] - full_z_q))
                    .detach()
                    .cpu()
                )
                results[label]["final_embedding_max_delta"] = max(
                    results[label]["final_embedding_max_delta"],
                    embedding_delta,
                )

                full_reconstruction = model.decode(full_z_q, model_batch)
                final_stage_reconstruction = model.decode(
                    stage_embeddings[-1],
                    model_batch,
                )
                decode_delta = float(
                    torch.max(
                        torch.abs(final_stage_reconstruction - full_reconstruction)
                    )
                    .detach()
                    .cpu()
                )
                results[label]["full_decode_max_delta"] = max(
                    results[label]["full_decode_max_delta"],
                    decode_delta,
                )

                original_feature = (
                    physical_valid[:, feature_idx]
                    if args.space == "physical"
                    else model_valid[:, feature_idx]
                )
                results[label]["originals"].append(
                    np.asarray(original_feature, dtype=np.float64)
                )

                continuous_reconstruction = model.decode(z_e, model_batch)
                continuous_valid = (
                    continuous_reconstruction[mask]
                    .detach()
                    .cpu()
                    .double()
                    .numpy()
                )
                results[label]["continuous_reconstructions"].append(
                    convert_feature_space(
                        continuous_valid,
                        feature_idx=feature_idx,
                        inverse_transformer=transformer,
                        space=args.space,
                    )
                )

                for stage_idx, z_q in enumerate(stage_embeddings):
                    reconstruction = model.decode(z_q, model_batch)
                    reconstruction_valid = (
                        reconstruction[mask].detach().cpu().double().numpy()
                    )
                    reconstructed_feature = convert_feature_space(
                        reconstruction_valid,
                        feature_idx=feature_idx,
                        inverse_transformer=transformer,
                        space=args.space,
                    )
                    results[label]["stage_reconstructions"][stage_idx].append(
                        reconstructed_feature
                    )

            n_seen += len(reference_valid)
            if n_seen >= args.max_valid_objects:
                break

    if reference_feature_names is None:
        raise RuntimeError("No valid objects found")

    finalized = {}
    for label, collected in results.items():
        if not collected["originals"] or collected["stage_reconstructions"] is None:
            raise RuntimeError(f"{label}: no valid objects found")
        finalized[label] = {
            "original": np.concatenate(collected["originals"])[
                : args.max_valid_objects
            ],
            "reconstructions": [
                np.concatenate(stage_values)[: args.max_valid_objects]
                for stage_values in collected["stage_reconstructions"]
            ],
            "indices": np.concatenate(collected["indices"], axis=0)[
                : args.max_valid_objects
            ],
            "continuous_reconstruction": np.concatenate(
                collected["continuous_reconstructions"]
            )[: args.max_valid_objects],
            "feature_names": reference_feature_names,
            "num_quantizers": len(collected["stage_reconstructions"]),
            "final_embedding_max_delta": collected[
                "final_embedding_max_delta"
            ],
            "full_decode_max_delta": collected["full_decode_max_delta"],
            "checkpoint": bundles[label]["checkpoint"],
        }
    return finalized


def make_bins(values: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        raise RuntimeError("No finite original feature values found")
    lo_pct, hi_pct = [
        float(value) for value in args.feature_percentile_range.split(",", 1)
    ]
    lo, hi = np.percentile(finite, [lo_pct, hi_pct])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.min(finite)), float(np.max(finite))
    if lo == hi:
        raise RuntimeError("Original feature has no usable range")
    return np.linspace(lo, hi, args.n_bins + 1)


def global_metrics(original: np.ndarray, reconstruction: np.ndarray) -> dict:
    finite = np.isfinite(original) & np.isfinite(reconstruction)
    residual = reconstruction[finite] - original[finite]
    if len(residual) == 0:
        return {
            "n_finite": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "median_absolute_error": float("nan"),
            "residual_iqr": float("nan"),
        }
    q25, q75 = np.percentile(residual, [25, 75])
    return {
        "n_finite": int(len(residual)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "median_absolute_error": float(np.median(np.abs(residual))),
        "residual_iqr": float(q75 - q25),
    }


def input_counts_in_bins(original: np.ndarray, bins: np.ndarray) -> np.ndarray:
    original = np.asarray(original, dtype=np.float64)
    counts = np.zeros(len(bins) - 1, dtype=np.int64)
    finite = np.isfinite(original)
    for bin_idx in range(len(counts)):
        in_bin = (original >= bins[bin_idx]) & (original < bins[bin_idx + 1])
        counts[bin_idx] = int(np.count_nonzero(in_bin & finite))
    return counts


def bootstrap_resolution_intervals(
    original: np.ndarray,
    reconstruction: np.ndarray,
    bins: np.ndarray,
    *,
    min_bin_count: int,
    min_denominator: float,
    n_bootstrap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(len(bins) - 1, np.nan, dtype=np.float64)
    upper = np.full(len(bins) - 1, np.nan, dtype=np.float64)
    if n_bootstrap <= 0:
        return lower, upper

    original = np.asarray(original, dtype=np.float64)
    reconstruction = np.asarray(reconstruction, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for bin_idx in range(len(bins) - 1):
        in_bin = (original >= bins[bin_idx]) & (original < bins[bin_idx + 1])
        finite = in_bin & np.isfinite(original) & np.isfinite(reconstruction)
        truth = original[finite]
        residual = reconstruction[finite] - truth
        if len(truth) < min_bin_count:
            continue

        replicas = []
        for _ in range(n_bootstrap):
            sample_idx = rng.integers(0, len(truth), size=len(truth))
            sampled_truth = truth[sample_idx]
            denominator = abs(float(np.median(sampled_truth)))
            if denominator < min_denominator:
                continue
            q25, q75 = np.percentile(residual[sample_idx], [25, 75])
            replicas.append(float(q75 - q25) / denominator)
        if replicas:
            lower[bin_idx], upper[bin_idx] = np.percentile(replicas, [16, 84])
    return lower, upper


def code_entropy_and_perplexity(indices: np.ndarray) -> tuple[int, float]:
    if len(indices) == 0:
        return 0, 0.0
    _, counts = np.unique(indices, return_counts=True)
    probabilities = counts.astype(np.float64) / counts.sum()
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    return int(len(counts)), float(np.exp(entropy))


def plot_code_usage_grid(
    rows: list[dict],
    *,
    num_quantizers: int,
    metric: str,
    ylabel: str,
    title: str,
    feature_name: str,
    nominal_min_count: int,
    output_path: Path,
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
        for model_label in ["MC-only", "MC+data"]:
            selected = [
                row
                for row in rows
                if row["model"] == model_label and row["quantizer"] == q_idx
            ]
            centers = np.asarray(
                [row["feature_center"] for row in selected], dtype=np.float64
            )
            values = np.asarray([row[metric] for row in selected], dtype=np.float64)
            counts = np.asarray([row["n_input"] for row in selected], dtype=np.int64)
            ax.plot(
                centers,
                values,
                marker="o",
                linewidth=2.0,
                color=MODEL_COLORS[model_label],
                label=model_label,
            )
            low_statistics = counts < nominal_min_count
            if np.any(low_statistics):
                ax.scatter(
                    centers[low_statistics],
                    values[low_statistics],
                    marker="o",
                    s=45,
                    facecolors="white",
                    edgecolors=MODEL_COLORS[model_label],
                    linewidths=1.7,
                    zorder=4,
                )
        ax.set_title(f"Quantizer {q_idx}", fontsize=14)
        ax.set_xlabel(f"Original {feature_label(feature_name)}", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(alpha=0.25)
        apply_hep_style(ax)
    for ax in flat_axes[num_quantizers:]:
        ax.set_visible(False)
    flat_axes[0].legend(frameon=False, fontsize=10)
    fig.suptitle(title, fontsize=17)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def calculate_resolution_curve(
    original: np.ndarray,
    reconstruction: np.ndarray,
    bins: np.ndarray,
    *,
    args: argparse.Namespace,
    bootstrap_seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    centers, resolution, counts, _, _ = binned_residual_iqr_over_median_truth(
        original,
        reconstruction,
        bins,
        min_bin_count=args.tail_min_bin_count,
        min_denominator=args.min_denominator,
    )
    input_counts = input_counts_in_bins(original, bins)
    finite_fraction = np.divide(
        counts,
        input_counts,
        out=np.full(len(counts), np.nan, dtype=np.float64),
        where=input_counts > 0,
    )
    ci_lower, ci_upper = bootstrap_resolution_intervals(
        original,
        reconstruction,
        bins,
        min_bin_count=args.tail_min_bin_count,
        min_denominator=args.min_denominator,
        n_bootstrap=args.bootstrap_samples,
        seed=bootstrap_seed,
    )
    return (
        centers,
        resolution,
        counts,
        input_counts,
        finite_fraction,
        ci_lower,
        ci_upper,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    output_dir = (
        Path(args.output_dir).resolve()
        / safe_filename(args.object)
        / args.sample
        / safe_filename(args.feature)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_files = select_h5_files(args)
    if not h5_files:
        raise FileNotFoundError("No H5 files selected")
    (output_dir / "h5_files_used.txt").write_text("\n".join(h5_files) + "\n")

    run_dirs = {
        "MC-only": Path(
            args.mc_only_run_dir or DEFAULT_RUNS[args.object]["mc_only"]
        ).resolve(),
        "MC+data": Path(
            args.mcdata_run_dir or DEFAULT_RUNS[args.object]["mcdata"]
        ).resolve(),
    }
    for label, run_dir in run_dirs.items():
        if not (run_dir / "full_config.yaml").exists():
            raise FileNotFoundError(f"{label} run is missing: {run_dir}")

    device = choose_device(args.device)
    bundles = {
        label: load_model_bundle(
            label=label,
            run_dir=run_dir,
            device=device,
        )
        for label, run_dir in run_dirs.items()
    }
    results = collect_shared_stage_reconstructions(
        bundles=bundles,
        h5_files=h5_files,
        args=args,
        device=device,
    )

    num_quantizers = min(result["num_quantizers"] for result in results.values())
    for result in results.values():
        result["reconstructions"] = [
            values for values in result["reconstructions"][:num_quantizers]
        ]

    original_a = results["MC-only"]["original"]
    bins = make_bins(original_a, args)
    curves: dict[
        str,
        list[
            tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
            ]
        ],
    ] = {}
    continuous_curves = {}
    metric_rows = []
    binned_rows = []
    continuous_binned_rows = []
    for model_label, result in results.items():
        curves[model_label] = []
        model_seed_offset = 0 if model_label == "MC-only" else 100_000
        continuous_curves[model_label] = calculate_resolution_curve(
            result["original"],
            result["continuous_reconstruction"],
            bins,
            args=args,
            bootstrap_seed=args.bootstrap_seed + model_seed_offset + 10_000,
        )
        for stage_idx, reconstruction in enumerate(result["reconstructions"]):
            curve = calculate_resolution_curve(
                result["original"],
                reconstruction,
                bins,
                args=args,
                bootstrap_seed=args.bootstrap_seed + model_seed_offset + stage_idx,
            )
            curves[model_label].append(curve)
            (
                centers,
                resolution,
                counts,
                input_counts,
                finite_fraction,
                ci_lower,
                ci_upper,
            ) = curve
            for bin_idx, (
                center,
                value,
                count,
                input_count,
                fraction,
                interval_lower,
                interval_upper,
            ) in enumerate(
                zip(
                    centers,
                    resolution,
                    counts,
                    input_counts,
                    finite_fraction,
                    ci_lower,
                    ci_upper,
                )
            ):
                binned_rows.append(
                    {
                        "model": model_label,
                        "stage": stage_label(stage_idx, num_quantizers),
                        "quantizers_used": stage_idx + 1,
                        "bin": bin_idx,
                        "feature_low": float(bins[bin_idx]),
                        "feature_high": float(bins[bin_idx + 1]),
                        "feature_center": float(center),
                        "n_input": int(input_count),
                        "n_finite": int(count),
                        "finite_fraction": float(fraction),
                        "metric_available": bool(np.isfinite(value)),
                        "nominal_count_pass": bool(input_count >= args.min_bin_count),
                        "relative_resolution": float(value),
                        "resolution_ci16": float(interval_lower),
                        "resolution_ci84": float(interval_upper),
                    }
                )
            metrics = global_metrics(result["original"], reconstruction)
            metrics.update(
                {
                    "model": model_label,
                    "stage": stage_label(stage_idx, num_quantizers),
                    "quantizers_used": stage_idx + 1,
                    "mean_binned_resolution": float(np.nanmean(resolution)),
                }
            )
            metric_rows.append(metrics)

        (
            centers,
            resolution,
            counts,
            input_counts,
            finite_fraction,
            ci_lower,
            ci_upper,
        ) = continuous_curves[model_label]
        for bin_idx, (
            center,
            value,
            count,
            input_count,
            fraction,
            interval_lower,
            interval_upper,
        ) in enumerate(
            zip(
                centers,
                resolution,
                counts,
                input_counts,
                finite_fraction,
                ci_lower,
                ci_upper,
            )
        ):
            continuous_binned_rows.append(
                {
                    "model": model_label,
                    "stage": "continuous z_e (no quantization)",
                    "bin": bin_idx,
                    "feature_low": float(bins[bin_idx]),
                    "feature_high": float(bins[bin_idx + 1]),
                    "feature_center": float(center),
                    "n_input": int(input_count),
                    "n_finite": int(count),
                    "finite_fraction": float(fraction),
                    "metric_available": bool(np.isfinite(value)),
                    "nominal_count_pass": bool(input_count >= args.min_bin_count),
                    "relative_resolution": float(value),
                    "resolution_ci16": float(interval_lower),
                    "resolution_ci84": float(interval_upper),
                }
            )

    code_usage_rows = []
    for model_label, result in results.items():
        original = np.asarray(result["original"], dtype=np.float64)
        indices = np.asarray(result["indices"], dtype=np.int64)
        for bin_idx in range(len(bins) - 1):
            in_bin = (
                (original >= bins[bin_idx])
                & (original < bins[bin_idx + 1])
                & np.isfinite(original)
            )
            n_input = int(np.count_nonzero(in_bin))
            if n_input < args.tail_min_bin_count:
                continue
            for q_idx in range(min(num_quantizers, indices.shape[1])):
                used_codes, perplexity = code_entropy_and_perplexity(
                    indices[in_bin, q_idx]
                )
                code_usage_rows.append(
                    {
                        "model": model_label,
                        "quantizer": q_idx,
                        "bin": bin_idx,
                        "feature_low": float(bins[bin_idx]),
                        "feature_high": float(bins[bin_idx + 1]),
                        "feature_center": float(
                            0.5 * (bins[bin_idx] + bins[bin_idx + 1])
                        ),
                        "n_input": n_input,
                        "nominal_count_pass": bool(
                            n_input >= args.min_bin_count
                        ),
                        "used_codes": used_codes,
                        "perplexity": perplexity,
                    }
                )

    stage_colors = plt.get_cmap("viridis")(
        np.linspace(0.12, 0.88, num_quantizers)
    )
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.4), sharex=True, sharey=True)
    for ax, model_label in zip(axes, ["MC-only", "MC+data"]):
        for stage_idx, (centers, resolution, _, _, _, _, _) in enumerate(
            curves[model_label]
        ):
            valid = np.isfinite(resolution)
            ax.plot(
                centers[valid],
                resolution[valid],
                marker="o",
                linewidth=2.0,
                color=stage_colors[stage_idx],
                label=stage_label(stage_idx, num_quantizers),
            )
        ax.set_title(model_label, fontsize=16)
        ax.set_xlabel(f"Original {feature_label(args.feature)}", fontsize=13)
        ax.grid(alpha=0.25)
        apply_hep_style(ax)
    axes[0].set_ylabel(
        r"IQR(reco - original) / |median(original)|",
        fontsize=13,
    )
    axes[1].legend(frameon=False, fontsize=10)
    fig.suptitle(
        f"{args.object}: cumulative quantizer reconstruction of {feature_label(args.feature)}",
        fontsize=17,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "stagewise_resolution_by_model.png", dpi=180)
    plt.close(fig)

    ncols = min(2, num_quantizers)
    nrows = (num_quantizers + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.0 * ncols, 4.5 * nrows),
        sharex=True,
        sharey=False,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for stage_idx in range(num_quantizers):
        ax = flat_axes[stage_idx]
        finite_values = np.concatenate(
            [
                curves[model_label][stage_idx][1][
                    np.isfinite(curves[model_label][stage_idx][1])
                ]
                for model_label in ["MC-only", "MC+data"]
            ]
        )
        finite_max = float(np.max(finite_values)) if len(finite_values) else 1.0
        marker_y = finite_max * 1.08 if finite_max > 0 else 1.0
        for model_label in ["MC-only", "MC+data"]:
            (
                centers,
                resolution,
                _,
                input_counts,
                _,
                ci_lower,
                ci_upper,
            ) = curves[model_label][stage_idx]
            valid = np.isfinite(resolution)
            ax.plot(
                centers[valid],
                resolution[valid],
                marker="o",
                linewidth=2.0,
                color=MODEL_COLORS[model_label],
                label=model_label,
            )
            low_statistics = valid & (input_counts < args.min_bin_count)
            if np.any(low_statistics):
                ax.scatter(
                    centers[low_statistics],
                    resolution[low_statistics],
                    marker="o",
                    s=45,
                    facecolors="white",
                    edgecolors=MODEL_COLORS[model_label],
                    linewidths=1.7,
                    zorder=4,
                )
            interval_valid = valid & np.isfinite(ci_lower) & np.isfinite(ci_upper)
            if np.any(interval_valid):
                ax.errorbar(
                    centers[interval_valid],
                    resolution[interval_valid],
                    yerr=np.vstack(
                        [
                            resolution[interval_valid] - ci_lower[interval_valid],
                            ci_upper[interval_valid] - resolution[interval_valid],
                        ]
                    ),
                    fmt="none",
                    ecolor=MODEL_COLORS[model_label],
                    elinewidth=1.1,
                    capsize=2.5,
                    alpha=0.55,
                    zorder=1,
                )
            unavailable = (~valid) & (input_counts > 0)
            if np.any(unavailable):
                ax.scatter(
                    centers[unavailable],
                    np.full(np.count_nonzero(unavailable), marker_y),
                    marker="x",
                    s=55,
                    linewidths=2.0,
                    color=MODEL_COLORS[model_label],
                    zorder=5,
                )
        ax.set_title(stage_label(stage_idx, num_quantizers), fontsize=14)
        ax.set_xlabel(f"Original {feature_label(args.feature)}", fontsize=12)
        ax.set_ylabel("relative resolution", fontsize=12)
        ax.grid(alpha=0.25)
        ax.set_ylim(bottom=0.0, top=marker_y * 1.08)
        if any(
            np.any(
                (~np.isfinite(curves[model_label][stage_idx][1]))
                & (curves[model_label][stage_idx][3] > 0)
            )
            for model_label in ["MC-only", "MC+data"]
        ):
            ax.text(
                0.98,
                0.96,
                r"$\times$ metric unavailable",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                color="0.35",
            )
        if any(
            np.any(
                np.isfinite(curves[model_label][stage_idx][1])
                & (curves[model_label][stage_idx][3] < args.min_bin_count)
            )
            for model_label in ["MC-only", "MC+data"]
        ):
            ax.text(
                0.98,
                0.88,
                f"hollow: <{args.min_bin_count} objects",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                color="0.35",
            )
        apply_hep_style(ax)
    for ax in flat_axes[num_quantizers:]:
        ax.set_visible(False)
    flat_axes[0].legend(frameon=False, fontsize=10)
    fig.suptitle(
        f"{args.object}: MC-only vs MC+data at each cumulative quantizer stage",
        fontsize=17,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison_at_each_stage.png", dpi=180)
    plt.close(fig)

    comparison_panels = [
        ("continuous $z_e$ (no quantization)", continuous_curves),
        (
            "q0+q1+q2+q3 (all quantizers)",
            {
                model_label: curves[model_label][-1]
                for model_label in ["MC-only", "MC+data"]
            },
        ),
    ]
    all_panel_values = np.concatenate(
        [
            panel_curves[model_label][1][
                np.isfinite(panel_curves[model_label][1])
            ]
            for _, panel_curves in comparison_panels
            for model_label in ["MC-only", "MC+data"]
        ]
    )
    panel_max = float(np.max(all_panel_values)) if len(all_panel_values) else 1.0
    marker_y = panel_max * 1.08 if panel_max > 0 else 1.0
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2), sharex=True, sharey=True)
    for ax, (panel_title, panel_curves) in zip(axes, comparison_panels):
        for model_label in ["MC-only", "MC+data"]:
            (
                centers,
                resolution,
                _,
                input_counts,
                _,
                ci_lower,
                ci_upper,
            ) = panel_curves[model_label]
            valid = np.isfinite(resolution)
            ax.plot(
                centers[valid],
                resolution[valid],
                marker="o",
                linewidth=2.0,
                color=MODEL_COLORS[model_label],
                label=model_label,
            )
            interval_valid = valid & np.isfinite(ci_lower) & np.isfinite(ci_upper)
            if np.any(interval_valid):
                ax.errorbar(
                    centers[interval_valid],
                    resolution[interval_valid],
                    yerr=np.vstack(
                        [
                            resolution[interval_valid] - ci_lower[interval_valid],
                            ci_upper[interval_valid] - resolution[interval_valid],
                        ]
                    ),
                    fmt="none",
                    ecolor=MODEL_COLORS[model_label],
                    elinewidth=1.1,
                    capsize=2.5,
                    alpha=0.55,
                )
            low_statistics = valid & (input_counts < args.min_bin_count)
            if np.any(low_statistics):
                ax.scatter(
                    centers[low_statistics],
                    resolution[low_statistics],
                    marker="o",
                    s=45,
                    facecolors="white",
                    edgecolors=MODEL_COLORS[model_label],
                    linewidths=1.7,
                    zorder=4,
                )
            unavailable = (~valid) & (input_counts > 0)
            if np.any(unavailable):
                ax.scatter(
                    centers[unavailable],
                    np.full(np.count_nonzero(unavailable), marker_y),
                    marker="x",
                    s=55,
                    linewidths=2.0,
                    color=MODEL_COLORS[model_label],
                )
        ax.set_title(panel_title, fontsize=14)
        ax.set_xlabel(f"Original {feature_label(args.feature)}", fontsize=12)
        ax.grid(alpha=0.25)
        apply_hep_style(ax)
    axes[0].set_ylabel("relative resolution", fontsize=12)
    axes[1].legend(frameon=False, fontsize=10)
    axes[0].set_ylim(bottom=0.0, top=marker_y * 1.08)
    fig.suptitle(
        f"{args.object}: continuous encoder-decoder vs quantized tokenizer",
        fontsize=17,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "continuous_vs_all_quantizers.png", dpi=180)
    plt.close(fig)

    if code_usage_rows:
        plot_code_usage_grid(
            code_usage_rows,
            num_quantizers=num_quantizers,
            metric="used_codes",
            ylabel="used codes",
            title=f"{args.object}: code usage at every quantizer stage",
            feature_name=args.feature,
            nominal_min_count=args.min_bin_count,
            output_path=output_dir / "all_quantizers_used_codes_by_feature.png",
        )
        plot_code_usage_grid(
            code_usage_rows,
            num_quantizers=num_quantizers,
            metric="perplexity",
            ylabel="effective codes (perplexity)",
            title=f"{args.object}: effective code usage at every quantizer stage",
            feature_name=args.feature,
            nominal_min_count=args.min_bin_count,
            output_path=output_dir / "all_quantizers_perplexity_by_feature.png",
        )

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    centers, _, _, input_counts, _, _, _ = curves["MC-only"][0]
    valid = input_counts > 0
    ax.plot(
        centers[valid],
        input_counts[valid],
        marker="o",
        linewidth=2.0,
        color="#333333",
        label="shared input sample",
    )
    ax.axhline(
        args.min_bin_count,
        color="#D62728",
        linestyle="--",
        linewidth=1.4,
        label=f"nominal minimum = {args.min_bin_count}",
    )
    ax.set_xlabel(f"Original {feature_label(args.feature)}", fontsize=13)
    ax.set_ylabel("input objects in bin", fontsize=12)
    ax.set_yscale("log")
    ax.set_title(f"{args.object}: shared evaluation-sample statistics", fontsize=15)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=10)
    apply_hep_style(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "input_objects_per_bin.png", dpi=180)
    plt.close(fig)
    stale_fraction_plot = output_dir / "finite_reconstruction_fraction_by_stage.png"
    if stale_fraction_plot.exists():
        stale_fraction_plot.unlink()

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for model_label in ["MC-only", "MC+data"]:
        selected = sorted(
            (row for row in metric_rows if row["model"] == model_label),
            key=lambda row: row["quantizers_used"],
        )
        ax.plot(
            [row["quantizers_used"] for row in selected],
            [row["mean_binned_resolution"] for row in selected],
            marker="o",
            linewidth=2.2,
            color=MODEL_COLORS[model_label],
            label=model_label,
        )
    ax.set_xticks(range(1, num_quantizers + 1))
    ax.set_xlabel("cumulative quantizers used", fontsize=13)
    ax.set_ylabel("mean binned relative resolution", fontsize=13)
    ax.set_title(f"{args.object}: reconstruction improvement by quantizer stage", fontsize=15)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=11)
    apply_hep_style(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "mean_resolution_vs_quantizer_stage.png", dpi=180)
    plt.close(fig)

    csv_path = output_dir / "stagewise_metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    binned_csv_path = output_dir / "stagewise_binned_resolution.csv"
    with binned_csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(binned_rows[0]))
        writer.writeheader()
        writer.writerows(binned_rows)

    continuous_csv_path = output_dir / "continuous_binned_resolution.csv"
    with continuous_csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(continuous_binned_rows[0]),
        )
        writer.writeheader()
        writer.writerows(continuous_binned_rows)

    if code_usage_rows:
        code_usage_csv_path = output_dir / "code_usage_by_feature_bin.csv"
        with code_usage_csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(code_usage_rows[0]),
            )
            writer.writeheader()
            writer.writerows(code_usage_rows)

    summary = {
        "object": args.object,
        "feature": args.feature,
        "space": args.space,
        "sample": args.sample,
        "n_h5_files": len(h5_files),
        "n_objects": {
            label: int(len(result["original"]))
            for label, result in results.items()
        },
        "num_quantizers_compared": num_quantizers,
        "continuous_latent_decoded": True,
        "shared_ordered_object_loader": True,
        "max_abs_original_feature_delta": float(
            np.nanmax(
                np.abs(
                    results["MC-only"]["original"]
                    - results["MC+data"]["original"]
                )
            )
        ),
        "final_embedding_max_delta": {
            label: result["final_embedding_max_delta"]
            for label, result in results.items()
        },
        "full_decode_max_delta": {
            label: result["full_decode_max_delta"]
            for label, result in results.items()
        },
        "run_dirs": {label: str(path) for label, path in run_dirs.items()},
        "checkpoints": {
            label: result["checkpoint"] for label, result in results.items()
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    log.info("Wrote plots and metrics to %s", output_dir)


if __name__ == "__main__":
    main()
