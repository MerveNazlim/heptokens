"""Make post-training diagnostics for an event/object VQ-VAE tokenizer."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from heptokens.data.event_mappable import EventMapDataset
from heptokens.models.vq_vae import LitVqVae

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger(__name__)


DATAMODULE_KEYS = {
    "_target_",
    "data_path",
    "data_paths",
    "train_frac",
    "val_frac",
    "test_frac",
    "seed",
    "n_classes",
    "num_workers",
    "batch_size",
    "pin_memory",
    "persistent_workers",
    "multiprocessing_context",
    "transforms",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create reconstruction and codebook-usage plots from a trained "
            "heptokens VQ-VAE tokenizer run."
        )
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
        default=20_000,
        help="Cap events loaded per H5 file for diagnostics.",
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
        "--h5-files",
        nargs="+",
        help="Optional H5 files to analyze instead of files stored in full_config.yaml.",
    )
    return parser.parse_args()


def choose_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def find_checkpoint(run_dir: Path, checkpoint: str | None) -> Path:
    if checkpoint is not None:
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    for name in ("best.ckpt", "last.ckpt"):
        path = run_dir / "checkpoints" / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No best.ckpt or last.ckpt found in {run_dir / 'checkpoints'}")


def dataset_kwargs_from_cfg(
    cfg,
    args: argparse.Namespace,
    run_dir: Path,
) -> tuple[list[str], dict]:
    datamodule = OmegaConf.to_container(cfg.datamodule, resolve=True)
    if args.h5_files:
        data_paths = list(args.h5_files)
    elif datamodule.get("data_paths"):
        data_paths = list(datamodule["data_paths"])
    else:
        data_paths = [datamodule["data_path"]]

    data_paths = [
        str((run_dir / path).resolve()) if not Path(path).is_absolute() else str(path)
        for path in data_paths
    ]

    dataset_kwargs = {
        key: value for key, value in datamodule.items() if key not in DATAMODULE_KEYS
    }
    dataset_kwargs["num_events"] = args.num_events_per_file
    return data_paths, dataset_kwargs


def feature_names_from_cfg(cfg, n_features: int) -> list[str]:
    datamodule = OmegaConf.to_container(cfg.datamodule, resolve=True)
    output_mode = datamodule.get("output_mode")
    collections = datamodule.get("object_collections") or []

    inputs = []
    if output_mode == "object":
        object_type = datamodule.get("object_type")
        for collection in collections:
            if collection.get("object_name") == object_type:
                inputs = collection.get("inputs") or []
                break
    elif collections:
        inputs = collections[0].get("inputs") or []

    names = [Path(path).name for path in inputs]
    if len(names) != n_features:
        names = [f"feature_{idx}" for idx in range(n_features)]
    return names


def object_name_from_cfg(cfg) -> str:
    datamodule = OmegaConf.to_container(cfg.datamodule, resolve=True)
    if datamodule.get("output_mode") == "object" and datamodule.get("object_type"):
        return str(datamodule["object_type"]).rstrip("s")
    return "object"


def to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def collect_diagnostics(
    *,
    model: LitVqVae,
    data_paths: list[str],
    dataset_kwargs: dict,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    max_valid_objects: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    originals = []
    recons = []
    all_indices = []
    n_seen = 0

    with torch.no_grad():
        for data_path in data_paths:
            log.info("Analyzing %s", data_path)
            dataset = EventMapDataset(data_path, **dataset_kwargs)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
            )
            for batch in loader:
                batch = to_device(batch, device)
                z_q, indices, _ = model.encode(batch)
                recon = model.decode(z_q, batch)

                mask = batch["mask"].bool()
                original_valid = batch["csts"][mask].detach().cpu().float().numpy()
                recon_valid = recon[mask].detach().cpu().float().numpy()
                indices_valid = indices[mask].detach().cpu().long().numpy()

                originals.append(original_valid)
                recons.append(recon_valid)
                all_indices.append(indices_valid)
                n_seen += len(original_valid)
                if n_seen >= max_valid_objects:
                    original = np.concatenate(originals, axis=0)[:max_valid_objects]
                    reconstruction = np.concatenate(recons, axis=0)[:max_valid_objects]
                    code_indices = np.concatenate(all_indices, axis=0)[:max_valid_objects]
                    return original, reconstruction, code_indices, n_seen

    if not originals:
        raise RuntimeError("No valid objects found. Check the object mask and input paths.")
    return (
        np.concatenate(originals, axis=0),
        np.concatenate(recons, axis=0),
        np.concatenate(all_indices, axis=0),
        n_seen,
    )


def apply_hep_style(ax) -> None:
    ax.tick_params(direction="in", which="both", top=True, right=True, width=1.2, labelsize=13)
    ax.tick_params(which="major", length=7)
    ax.tick_params(which="minor", length=3.5)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def feature_label(name: str) -> str:
    lower = name.lower()
    if lower in {"pt", "met"} or lower.endswith("/pt"):
        return r"$p_T$"
    if lower == "eta" or lower.endswith("/eta"):
        return r"$\eta$"
    if lower == "phi" or lower.endswith("/phi"):
        return r"$\phi$"
    return name.replace("_", " ")


def safe_filename(text: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text)


def display_values(
    original: np.ndarray,
    reconstruction: np.ndarray,
    feature_name: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    label = feature_label(feature_name)
    scale = 1.0
    unit = ""
    if feature_name.lower() in {"pt", "met", "sumet"} and np.nanpercentile(original, 95) > 1000:
        scale = 1000.0
        unit = " [GeV]"
    return original / scale, reconstruction / scale, f"{label}{unit}"


def finite_pair(original: np.ndarray, reconstruction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(original) & np.isfinite(reconstruction)
    return original[finite], reconstruction[finite]


def plot_feature_triptychs(
    original: np.ndarray,
    reconstruction: np.ndarray,
    feature_names: list[str],
    object_name: str,
    output_dir: Path,
) -> None:
    rng = np.random.default_rng(42)
    for feature_idx, feature_name in enumerate(feature_names):
        orig, reco, axis_label = display_values(
            original[:, feature_idx],
            reconstruction[:, feature_idx],
            feature_name,
        )
        orig, reco = finite_pair(orig, reco)
        if len(orig) == 0:
            continue

        combined = np.concatenate([orig, reco])
        lo, hi = np.percentile(combined, [0.5, 99.5])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = float(np.nanmin(combined)), float(np.nanmax(combined))
        bins = np.linspace(lo, hi, 70)
        residual = reco - orig
        res_lo, res_hi = np.percentile(residual, [0.5, 99.5])
        res_bins = np.linspace(res_lo, res_hi, 70)

        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
        ax = axes[0]
        ax.hist(
            orig,
            bins=bins,
            histtype="step",
            density=False,
            linewidth=1.9,
            color="#1f77b4",
            label="original",
        )
        ax.hist(
            reco,
            bins=bins,
            histtype="step",
            density=False,
            linewidth=1.9,
            color="#ff7f0e",
            label="reconstructed",
        )
        ax.set_title(f"{object_name} {feature_label(feature_name)}", fontsize=16)
        ax.set_xlabel(axis_label, fontsize=13)
        ax.set_ylabel("objects", fontsize=13)
        ax.legend(frameon=True, fontsize=11)
        apply_hep_style(ax)

        ax = axes[1]
        ax.hist(residual, bins=res_bins, color="#1f77b4", alpha=0.75)
        ax.axvline(0, color="black", linewidth=1.2)
        ax.set_title("residual", fontsize=16)
        ax.set_xlabel("reco - original", fontsize=13)
        ax.set_ylabel("objects", fontsize=13)
        apply_hep_style(ax)

        ax = axes[2]
        n_plot = min(len(orig), 50_000)
        idx = rng.choice(len(orig), size=n_plot, replace=False)
        ax.scatter(orig[idx], reco[idx], s=3, alpha=0.22, rasterized=True)
        line_lo, line_hi = np.percentile(np.concatenate([orig[idx], reco[idx]]), [0.5, 99.5])
        ax.plot([line_lo, line_hi], [line_lo, line_hi], color="black", linewidth=1.2)
        ax.set_xlim(line_lo, line_hi)
        ax.set_ylim(line_lo, line_hi)
        ax.set_title("correlation", fontsize=16)
        ax.set_xlabel("original", fontsize=13)
        ax.set_ylabel("reconstructed", fontsize=13)
        apply_hep_style(ax)

        fig.tight_layout()
        fig.savefig(
            output_dir / f"{safe_filename(feature_name)}_reconstruction_triptych.png",
            dpi=180,
        )
        plt.close(fig)


def plot_feature_overlay(
    original: np.ndarray,
    reconstruction: np.ndarray,
    feature_names: list[str],
    object_name: str,
    output_dir: Path,
) -> None:
    for feature_idx, feature_name in enumerate(feature_names):
        orig, reco, axis_label = display_values(
            original[:, feature_idx],
            reconstruction[:, feature_idx],
            feature_name,
        )
        orig, reco = finite_pair(orig, reco)
        combined = np.concatenate([orig, reco])
        lo, hi = np.percentile(combined, [0.5, 99.5])
        bins = np.linspace(lo, hi, 75)

        fig, ax = plt.subplots(figsize=(8.0, 6.0))
        ax.hist(
            orig,
            bins=bins,
            histtype="step",
            density=True,
            linewidth=2.0,
            color="black",
            label="Original",
        )
        ax.hist(
            reco,
            bins=bins,
            histtype="step",
            density=True,
            linewidth=2.0,
            color="#00A000",
            label="VQVAE",
        )
        ax.set_title(f"{object_name.capitalize()} {feature_label(feature_name)}", fontsize=17)
        ax.set_xlabel(axis_label, fontsize=14)
        ax.set_ylabel("Normalized", fontsize=14)
        ax.legend(frameon=False, fontsize=12)
        apply_hep_style(ax)
        fig.tight_layout()
        fig.savefig(output_dir / f"{safe_filename(feature_name)}_overlay.png", dpi=180)
        plt.close(fig)


def codebook_counts(indices: np.ndarray, codebook_size: int) -> np.ndarray:
    n_quantizers = indices.shape[1]
    counts = np.zeros((n_quantizers, codebook_size), dtype=np.int64)
    for quantizer_idx in range(n_quantizers):
        values = indices[:, quantizer_idx]
        values = values[values >= 0]
        counts[quantizer_idx] = np.bincount(values, minlength=codebook_size)[:codebook_size]
    return counts


def plot_codebook_usage(counts: np.ndarray, output_path: Path) -> None:
    n_quantizers = counts.shape[0]
    fig, axes = plt.subplots(n_quantizers, 1, figsize=(10, 2.6 * n_quantizers), sharex=True)
    axes = np.atleast_1d(axes)
    for quantizer_idx, ax in enumerate(axes):
        ax.bar(np.arange(counts.shape[1]), counts[quantizer_idx], width=1.0)
        used = int(np.count_nonzero(counts[quantizer_idx]))
        dead = int(counts.shape[1] - used)
        ax.set_ylabel(f"q{quantizer_idx}")
        ax.set_title(f"quantizer {quantizer_idx}: used={used}, dead={dead}")
        ax.grid(axis="y", alpha=0.2)
    axes[-1].set_xlabel("code index")
    fig.suptitle("Codebook usage frequency", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_codebook_frequency_hist(
    counts: np.ndarray,
    object_name: str,
    output_path: Path,
) -> None:
    flat_counts = counts.reshape(-1)
    used_fraction = np.count_nonzero(flat_counts) / len(flat_counts)
    hi = np.percentile(flat_counts, 99.5)
    if hi <= 0:
        hi = max(1, flat_counts.max())
    bins = np.linspace(0, hi, 80)

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.hist(flat_counts, bins=bins, color="#1f77b4", alpha=0.78)
    ax.set_title(f"{object_name} codebook: {100 * used_fraction:.1f}% used", fontsize=20)
    ax.set_xlabel("token frequency", fontsize=16)
    ax.set_ylabel("tokens", fontsize=16)
    apply_hep_style(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_dead_tokens(counts: np.ndarray, output_path: Path) -> None:
    used = np.count_nonzero(counts, axis=1)
    dead = counts.shape[1] - used
    labels = [f"q{idx}" for idx in range(counts.shape[0])]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, used, label="used", color="#315C9E")
    ax.bar(labels, dead, bottom=used, label="dead", color="#B84A4A")
    ax.set_ylabel("codes")
    ax.set_title("Used vs dead codes")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(
    output_path: Path,
    *,
    counts: np.ndarray,
    original: np.ndarray,
    reconstruction: np.ndarray,
    feature_names: list[str],
    n_seen: int,
) -> None:
    lines = []
    lines.append(f"valid_objects_analyzed: {len(original)}")
    lines.append(f"valid_objects_seen_before_stop: {n_seen}")
    lines.append("")
    lines.append("reconstruction:")
    for idx, name in enumerate(feature_names):
        diff = reconstruction[:, idx] - original[:, idx]
        lines.append(f"  {name}:")
        lines.append(f"    mae: {np.mean(np.abs(diff)):.8g}")
        lines.append(f"    rmse: {np.sqrt(np.mean(diff**2)):.8g}")
        lines.append(f"    bias: {np.mean(diff):.8g}")
    lines.append("")
    lines.append("codebook:")
    for quantizer_idx, quantizer_counts in enumerate(counts):
        used = int(np.count_nonzero(quantizer_counts))
        dead = int(len(quantizer_counts) - used)
        total = int(quantizer_counts.sum())
        lines.append(f"  quantizer_{quantizer_idx}:")
        lines.append(f"    total_assignments: {total}")
        lines.append(f"    used_codes: {used}")
        lines.append(f"    dead_codes: {dead}")
        lines.append(f"    used_fraction: {used / len(quantizer_counts):.6f}")
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    run_dir = Path(args.run_dir).resolve()
    cfg_path = run_dir / "full_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    cfg = OmegaConf.load(cfg_path)
    data_paths, dataset_kwargs = dataset_kwargs_from_cfg(cfg, args, run_dir)
    checkpoint = find_checkpoint(run_dir, args.checkpoint)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else run_dir / "figures" / "tokenizer_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    log.info("Loading %s", checkpoint)
    model = LitVqVae.load_from_checkpoint(checkpoint, map_location=device)
    model.to(device)
    model.eval()

    original, reconstruction, indices, n_seen = collect_diagnostics(
        model=model,
        data_paths=data_paths,
        dataset_kwargs=dataset_kwargs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        max_valid_objects=args.max_valid_objects,
    )

    feature_names = feature_names_from_cfg(cfg, original.shape[1])
    object_name = object_name_from_cfg(cfg)
    codebook_size = int(getattr(model.hparams, "codebook_size", cfg.model.codebook_size))
    counts = codebook_counts(indices, codebook_size)

    np.save(output_dir / "codebook_counts.npy", counts)
    plot_feature_triptychs(
        original,
        reconstruction,
        feature_names,
        object_name,
        output_dir,
    )
    plot_feature_overlay(
        original,
        reconstruction,
        feature_names,
        object_name,
        output_dir,
    )
    plot_codebook_usage(counts, output_dir / "codebook_frequency.png")
    plot_codebook_frequency_hist(
        counts,
        object_name,
        output_dir / "codebook_token_frequency_hist.png",
    )
    plot_dead_tokens(counts, output_dir / "dead_tokens.png")
    write_summary(
        output_dir / "summary.txt",
        counts=counts,
        original=original,
        reconstruction=reconstruction,
        feature_names=feature_names,
        n_seen=n_seen,
    )
    log.info("Wrote tokenizer diagnostics to %s", output_dir)


if __name__ == "__main__":
    main()
