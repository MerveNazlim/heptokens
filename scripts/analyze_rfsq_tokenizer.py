"""Make post-training diagnostics for an event/object residual FSQ tokenizer.

Data loading, preprocessing, reconstruction plots and codebook plots are shared with
``analyze_vqvae_tokenizer.py`` and the per-dim level summary with ``analyze_fsq_tokenizer.py``.
The residual-specific diagnostics live here:

- reconstruction quality vs the number of stages used (decoding the first k stage codes only),
- the latent residual left after each stage (expected to shrink about L-fold per stage),
- the per-stage quantizer inputs ``residual / scale`` (should fill [-c, c], c = 1 + 1/(L-1);
  inputs beyond +-1 fall in the end levels' bins, about 1/L of them for a uniform residual),
- the per-stage, per-dim level usage,
- the "stage infinity" reconstruction: the soft-clamped latent decoded without quantization,
  i.e. the encoder-decoder limit that quantization can at best reach.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import analyze_fsq_tokenizer as fsq_base
import analyze_vqvae_tokenizer as base
from heptokens.models.rfsq import LitResidualFSQ, ResidualFSQ

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create reconstruction, per-stage and codebook-usage plots from a trained residual FSQ tokenizer run."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Training run directory containing full_config.yaml and checkpoints/.",
    )
    parser.add_argument(
        "--checkpoint",
        help="Checkpoint to analyze. Defaults to best.ckpt, then last.ckpt.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for plots. Defaults to <run-dir>/figures/tokenizer_analysis.",
    )
    parser.add_argument(
        "--num-events-per-file",
        type=int,
        default=None,
        help="Optional cap on events loaded per H5 file. By default, use the run config unchanged.",
    )
    parser.add_argument(
        "--max-valid-objects",
        type=int,
        default=200_000,
        help="Stop after this many valid reconstructed objects.",
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="val",
        help="Datamodule split to use for diagnostics. Defaults to validation.",
    )
    parser.add_argument(
        "--h5-files",
        nargs="+",
        help="Optional H5 files to analyze instead of files stored in full_config.yaml.",
    )
    return parser.parse_args()


def rfsq_stage_outputs(vq: ResidualFSQ, z: torch.Tensor) -> dict[str, torch.Tensor]:
    """Replay ``ResidualFSQ.n_quantize`` keeping every intermediate.

    Returns ``target`` [..., D] (the soft-clamped input that the stage sum approximates),
    ``stage_inputs`` [Q, ..., D] (``residual / scale`` before the hard clamp), ``stage_codes``
    [Q, ..., D] (scaled codes; their sum is ``z_hat``), ``digits`` [Q, ..., D] (level 0..L-1)
    and ``indices`` [..., Q].
    """
    c = vq.soft_clamp_value
    target = torch.tanh(z.float() / c) * c
    residual = target
    stage_inputs, stage_codes, digits, indices = [], [], [], []
    for layer, scale in zip(vq.layers, vq.scales):
        stage_input = residual / scale
        code = layer.quantize(stage_input)
        stage_inputs.append(stage_input)
        digits.append(layer._scale_and_shift(code).round().long())
        indices.append(layer.codes_to_indexes(code))
        code = code * scale
        stage_codes.append(code)
        residual = residual - code
    return {
        "target": target,
        "stage_inputs": torch.stack(stage_inputs),
        "stage_codes": torch.stack(stage_codes),
        "digits": torch.stack(digits),
        "indices": torch.stack(indices, dim=-1),
    }


def collect_rfsq_diagnostics(
    *,
    model: LitResidualFSQ,
    loader: DataLoader,
    cst_inverse_transformer,
    device: torch.device,
    max_valid_objects: int,
) -> dict[str, np.ndarray | int]:
    """Run the residual FSQ tokenizer over ``loader`` and keep valid (unmasked) objects only.

    Returns a dict with ``original`` [n, F] and ``reconstruction_by_depth`` [Q, n, F] (decoded
    from the first k stages, inverse-transformed when a transformer is given), ``indices``
    [n, Q], ``digits`` and ``stage_inputs`` [Q, n, D], ``latent_residual`` [Q, n, D]
    (target minus the first k stage codes), ``reconstruction_continuous`` [n, F] (the unquantized
    soft-clamped latent decoded), ``latent`` [n, D] (that latent) and ``n_seen``.
    """
    vq = model.vector_quantization
    n_stages = vq.num_quantizers
    chunks: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "original", "reconstruction_by_depth", "indices", "digits", "stage_inputs", "latent_residual",
            "reconstruction_continuous", "latent",
        )
    }
    n_seen = 0
    checked_model_path = False

    with torch.no_grad():
        for batch in loader:
            batch = base.to_device(batch, device)
            mask = batch["mask"].bool()
            if not bool(mask.any()):
                continue

            outputs = rfsq_stage_outputs(vq, model.latent(batch))
            partial = torch.cumsum(outputs["stage_codes"], dim=0)  # [Q, B, N, D]
            if not checked_model_path:
                # The replay must match the model's own encode path exactly.
                z_hat, indices = model.encode(batch)
                if not torch.allclose(partial[-1], z_hat, atol=1e-5) or not torch.equal(
                    outputs["indices"][mask], indices[mask]
                ):
                    raise RuntimeError("Per-stage replay disagrees with LitResidualFSQ.encode")
                checked_model_path = True

            original_valid = batch["csts"][mask].detach().cpu().float().numpy()
            recon_by_depth = []
            for depth in range(n_stages):
                recon = model.decode(partial[depth], batch)[mask].detach().cpu().float().numpy()
                if cst_inverse_transformer is not None:
                    recon = cst_inverse_transformer.inverse_transform(recon)
                recon_by_depth.append(recon)
            # "Stage infinity": decode the soft-clamped latent itself, without quantization.
            recon_continuous = model.decode(outputs["target"], batch)[mask].detach().cpu().float().numpy()
            if cst_inverse_transformer is not None:
                original_valid = cst_inverse_transformer.inverse_transform(original_valid)
                recon_continuous = cst_inverse_transformer.inverse_transform(recon_continuous)

            chunks["original"].append(original_valid)
            chunks["reconstruction_by_depth"].append(np.stack(recon_by_depth))
            chunks["indices"].append(outputs["indices"][mask].cpu().long().numpy())
            chunks["digits"].append(outputs["digits"][:, mask].cpu().long().numpy())
            chunks["stage_inputs"].append(outputs["stage_inputs"][:, mask].cpu().float().numpy())
            chunks["latent_residual"].append(
                (outputs["target"].unsqueeze(0) - partial)[:, mask].cpu().float().numpy()
            )
            chunks["reconstruction_continuous"].append(recon_continuous)
            chunks["latent"].append(outputs["target"][mask].cpu().float().numpy())
            n_seen += len(original_valid)
            if n_seen >= max_valid_objects:
                break

    if not chunks["original"]:
        raise RuntimeError("No valid objects found. Check the object mask and input paths.")
    # every other array is stage-major [Q, n, ...]
    object_axis = {"original": 0, "indices": 0, "reconstruction_continuous": 0, "latent": 0}
    result: dict[str, np.ndarray | int] = {}
    for key, values in chunks.items():
        axis = object_axis.get(key, 1)
        result[key] = np.take(np.concatenate(values, axis=axis), np.arange(min(n_seen, max_valid_objects)), axis=axis)
    result["n_seen"] = n_seen
    return result


def bits_per_object(levels: list[int], n_stages: int) -> float:
    return float(n_stages * np.log2(np.prod(levels)))


def residual_stats(original: np.ndarray, reconstruction: np.ndarray) -> dict:
    """RMSE / MAE / bias and the robust width sigma68 = (p84 - p16) / 2 of reco - truth."""
    orig, reco = base.finite_pair(original, reconstruction)
    residual = reco - orig
    p16, p84 = np.percentile(residual, [16, 84])
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "sigma68": float((p84 - p16) / 2),
    }


def depth_metrics(
    original: np.ndarray,
    reconstruction_by_depth: np.ndarray,
    latent_residual: np.ndarray,
    feature_names: list[str],
    levels: list[int],
) -> list[dict]:
    """Per number of stages k = 1..Q: bits, per-feature residual stats, latent residual RMS per dim."""
    result = []
    for depth, recon in enumerate(reconstruction_by_depth, start=1):
        result.append(
            {
                "stages": depth,
                "bits_per_object": bits_per_object(levels, depth),
                "features": {
                    name: residual_stats(original[:, idx], recon[:, idx])
                    for idx, name in enumerate(feature_names)
                },
                "latent_residual_rms": np.sqrt(np.mean(latent_residual[depth - 1] ** 2, axis=0)).tolist(),
            }
        )
    return result


def continuous_metrics(
    original: np.ndarray,
    reconstruction_continuous: np.ndarray,
    latent: np.ndarray,
    feature_names: list[str],
) -> dict:
    """The encoder-decoder limit: residual stats of the unquantized decode, and per-dim latent spread.

    A latent dim with std ~0 carries no information (e.g. stuck at the soft-clamp edge).
    """
    return {
        "features": {
            name: residual_stats(original[:, idx], reconstruction_continuous[:, idx])
            for idx, name in enumerate(feature_names)
        },
        "latent_std": np.std(latent, axis=0).tolist(),
        "latent_mean": np.mean(latent, axis=0).tolist(),
    }


def stage_level_summary(digits: np.ndarray, stage_inputs: np.ndarray, levels: list[int]) -> dict:
    """Per stage: the FSQ per-dim level summary plus the fraction of inputs beyond +-1.

    Inputs in (1, c] map to the end levels by design (their bin is as wide as the others), so
    ~1/L is expected for a uniform residual; much more means the residual exceeds the stage range.
    """
    summary = {}
    for stage in range(digits.shape[0]):
        dims = fsq_base.level_summary(fsq_base.level_counts(digits[stage], levels))
        for dim in range(len(levels)):
            dims[f"dim_{dim}"]["beyond_unit_fraction"] = float(np.mean(np.abs(stage_inputs[stage][:, dim]) > 1))
        summary[f"stage_{stage}"] = dims
    return summary


def plot_depth_curves(
    depths: list[dict], levels: list[int], output_path: Path, continuous: dict | None = None
) -> None:
    """Left: per-feature RMSE vs stages (relative to 1 stage); dotted: the unquantized decode.
    Right: latent residual RMS per dim."""
    stages = [entry["stages"] for entry in depths]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for name in depths[0]["features"]:
        first = depths[0]["features"][name]["rmse"]
        values = [entry["features"][name]["rmse"] / first if first else np.nan for entry in depths]
        line = axes[0].plot(stages, values, marker="o", label=base.feature_label(name))
        if continuous is not None and first:
            axes[0].axhline(
                continuous["features"][name]["rmse"] / first, color=line[0].get_color(), linestyle=":", alpha=0.7
            )
    if continuous is not None:
        axes[0].set_title("dotted: no quantization (stage infinity)", fontsize=10)
    axes[0].set_xlabel("stages used", fontsize=13)
    axes[0].set_ylabel("feature RMSE / RMSE(1 stage)", fontsize=13)
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=9, frameon=False)

    rms = np.array([entry["latent_residual_rms"] for entry in depths])  # [Q, D]
    for dim, n_levels in enumerate(levels):
        line = axes[1].plot(stages, rms[:, dim], marker="o", label=f"dim {dim} (L={n_levels})")
        # Uniform-error expectation: half-step / sqrt(3) at stage 1, shrinking L-fold per stage
        expected = [1 / ((n_levels - 1) * np.sqrt(3)) * float(n_levels) ** -(k - 1) for k in stages]
        axes[1].plot(stages, expected, linestyle="--", color=line[0].get_color(), alpha=0.5)
    axes[1].set_xlabel("stages used", fontsize=13)
    axes[1].set_ylabel("latent residual RMS", fontsize=13)
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=9, frameon=False, title="dashed: uniform-error expectation", title_fontsize=9)
    for ax in axes:
        ax.set_xticks(stages)
        ax.grid(alpha=0.25)
        base.apply_hep_style(ax)
    fig.suptitle("Residual FSQ: reconstruction vs number of stages", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_stage_grid(
    values_fn,
    n_stages: int,
    levels: list[int],
    title: str,
    xlabel: str,
    output_path: Path,
) -> None:
    """One subplot per (stage, dim); ``values_fn(ax, stage, dim, n_levels)`` draws it."""
    n_dims = len(levels)
    fig, axes = plt.subplots(
        n_stages, n_dims, figsize=(2.8 * n_dims, 2.2 * n_stages + 0.8), squeeze=False
    )
    for stage in range(n_stages):
        for dim, n_levels in enumerate(levels):
            ax = axes[stage][dim]
            values_fn(ax, stage, dim, n_levels)
            if stage == 0:
                ax.set_title(f"dim {dim} (L={n_levels})", fontsize=11)
            if dim == 0:
                ax.set_ylabel(f"stage {stage}", fontsize=11)
            if stage == n_stages - 1:
                ax.set_xlabel(xlabel, fontsize=10)
            ax.tick_params(labelsize=8)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_stage_level_usage(digits: np.ndarray, levels: list[int], output_path: Path) -> None:
    def draw(ax, stage, dim, n_levels):
        counts = np.bincount(digits[stage][:, dim], minlength=n_levels)[:n_levels]
        ax.bar(np.arange(n_levels), counts / max(int(counts.sum()), 1), width=0.85, color="#4C78A8")
        ax.set_xticks(np.arange(n_levels))

    plot_stage_grid(draw, digits.shape[0], levels, "Residual FSQ level usage per stage", "level", output_path)


def plot_stage_inputs(stage_inputs: np.ndarray, levels: list[int], output_path: Path) -> None:
    """Histogram of residual / scale per stage; red lines at +-1, beyond them the end levels' bins."""

    def draw(ax, stage, dim, n_levels):
        values = stage_inputs[stage][:, dim]
        ax.hist(np.clip(values, -1.5, 1.5), bins=np.linspace(-1.5, 1.5, 91), color="#4C78A8", alpha=0.85)
        for edge in (-1, 1):
            ax.axvline(edge, color="#B84A4A", linewidth=1.0)
        ax.text(
            0.03, 0.92, f"|x|>1: {100 * np.mean(np.abs(values) > 1):.1f}% (1/L={100 / n_levels:.0f}%)",
            transform=ax.transAxes, fontsize=8, va="top",
        )

    plot_stage_grid(
        draw,
        stage_inputs.shape[0],
        levels,
        "Residual FSQ stage inputs (residual / scale), clipped to [-1.5, 1.5] for display",
        "residual / scale",
        output_path,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    run_dir = Path(args.run_dir).resolve()
    cfg_path = run_dir / "full_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    cfg = OmegaConf.load(cfg_path)
    _, cst_inverse_transformer = base.transform_list_and_cst_fn_from_cfg(cfg)
    checkpoint = base.find_checkpoint(run_dir, args.checkpoint)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else run_dir / "figures" / "tokenizer_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = base.choose_device(args.device)
    log.info("Loading %s", checkpoint)
    model = LitResidualFSQ.load_from_checkpoint(checkpoint, map_location=device)
    model.to(device)
    model.eval()

    log.info("Collecting diagnostics through the canonical saved datamodule path")
    loader = base.canonical_dataloader_for_h5_files(
        cfg=cfg,
        h5_files=list(args.h5_files) if args.h5_files else None,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_events_per_file=args.num_events_per_file,
    )
    diagnostics = collect_rfsq_diagnostics(
        model=model,
        loader=loader,
        cst_inverse_transformer=cst_inverse_transformer,
        device=device,
        max_valid_objects=args.max_valid_objects,
    )
    original = diagnostics["original"]
    reconstruction = diagnostics["reconstruction_by_depth"][-1]  # all stages

    feature_names = base.feature_names_from_cfg(cfg, original.shape[1])
    object_name = base.object_name_from_cfg(cfg)
    vq = model.vector_quantization
    levels = list(vq.levels)

    counts = base.codebook_counts(diagnostics["indices"], vq.codebook_size)  # [Q, prod(levels)]
    reco_metrics = base.reconstruction_metrics(original, reconstruction, feature_names)
    usage_summary = base.codebook_summary(counts)
    depths = depth_metrics(
        original,
        diagnostics["reconstruction_by_depth"],
        diagnostics["latent_residual"],
        feature_names,
        levels,
    )
    stage_summary = stage_level_summary(diagnostics["digits"], diagnostics["stage_inputs"], levels)
    continuous = continuous_metrics(
        original, diagnostics["reconstruction_continuous"], diagnostics["latent"], feature_names
    )

    np.save(output_dir / "codebook_counts.npy", counts)
    (output_dir / "reconstruction_metrics.json").write_text(json.dumps(reco_metrics, indent=2) + "\n")
    (output_dir / "codebook_usage_summary.json").write_text(json.dumps(usage_summary, indent=2) + "\n")
    (output_dir / "rfsq_depth_metrics.json").write_text(
        json.dumps(
            {"levels": levels, "num_quantizers": vq.num_quantizers, "depths": depths, "continuous": continuous},
            indent=2,
        )
        + "\n"
    )
    (output_dir / "rfsq_level_summary.json").write_text(json.dumps(stage_summary, indent=2) + "\n")

    base.plot_feature_triptychs(original, reconstruction, feature_names, object_name, output_dir)
    base.plot_feature_overlay(
        original, reconstruction, feature_names, object_name, output_dir, model_label="Residual FSQ"
    )
    base.plot_residual_summary(reco_metrics, output_dir / "residual_summary.png")
    base.plot_codebook_usage(counts, output_dir / "codebook_frequency.png")
    base.plot_codebook_frequency_hist(counts, object_name, output_dir / "codebook_token_frequency_hist.png")
    base.plot_dead_tokens(counts, output_dir / "dead_tokens.png")
    plot_depth_curves(depths, levels, output_dir / "rfsq_depth_curves.png", continuous)
    plot_stage_level_usage(diagnostics["digits"], levels, output_dir / "rfsq_level_usage.png")
    plot_stage_inputs(diagnostics["stage_inputs"], levels, output_dir / "rfsq_stage_inputs.png")
    base.write_summary(
        output_dir / "summary.txt",
        counts=counts,
        original=original,
        reconstruction=reconstruction,
        feature_names=feature_names,
        n_seen=diagnostics["n_seen"],
    )
    log.info("Wrote residual FSQ tokenizer diagnostics to %s", output_dir)


if __name__ == "__main__":
    main()
