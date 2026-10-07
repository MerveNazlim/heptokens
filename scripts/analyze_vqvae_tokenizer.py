"""Make post-training diagnostics for an event/object VQ-VAE tokenizer."""

from __future__ import annotations

import argparse
import functools
import json
import logging
from pathlib import Path

import hydra
import matplotlib
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import DataLoader

from heptokens.data.atlas_event_mappable import AtlasEventMapDataset
from heptokens.models.vq_vae import LitVqVae

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger(__name__)


DATAMODULE_KEYS = {
    "_target_",
    "data_path",
    "data_paths",
    "data_domains",
    "sampling_domain_fractions",
    "sampling_balance_by",
    "sampling_num_samples",
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

DEFAULT_OBJECT_FEATURES = {
    "jets": [
        "pt",
        "eta",
        "phi",
        "mass",
        "n_trk",
        "QG_nTracks",
        "QG_tracksWidth",
        "QG_tracksC1",
        "DL1d_pb",
        "DL1d_pc",
        "DL1d_pu",
        "GN2_pb",
        "GN2_pc",
        "GN2_pu",
    ],
    "electrons": [
        "pt",
        "eta",
        "phi",
        "charge",
        "trk_iso03",
        "LHLoose",
        "LHMedium",
        "LHTight",
    ],
    "muons": ["pt", "eta", "phi", "charge", "trk_iso03", "quality", "muonType"],
    "taus": [
        "pt",
        "eta",
        "phi",
        "charge",
        "is_1prong",
        "NNDecayMode",
        "RNNJetScore",
        "RNNEleScore",
    ],
    "photons": ["pt", "eta", "phi", "trk_iso03", "isLoose", "isTight"],
}

FEATURE_LABELS = {
    "pt": r"$p_T$",
    "eta": r"$\eta$",
    "phi": r"$\phi$",
    "mass": "mass",
    "met": "MET",
    "sumet": r"$\Sigma E_T$",
    "n_trk": "n tracks",
    "qg_ntracks": "QG nTracks",
    "qg_trackswidth": "QG tracksWidth",
    "qg_tracksc1": "QG tracksC1",
    "dl1d_pb": r"DL1d $p_b$",
    "dl1d_pc": r"DL1d $p_c$",
    "dl1d_pu": r"DL1d $p_u$",
    "gn2_pb": r"GN2 $p_b$",
    "gn2_pc": r"GN2 $p_c$",
    "gn2_pu": r"GN2 $p_u$",
    "trk_iso03": "track isolation",
    "lhloose": "LHLoose",
    "lhmedium": "LHMedium",
    "lhtight": "LHTight",
    "quality": "quality",
    "muontype": "muonType",
    "is_1prong": "is_1prong",
    "nndecaymode": "NNDecayMode",
    "rnnjetscore": "RNNJetScore",
    "rnnelescore": "RNNEleScore",
    "isloose": "isLoose",
    "istight": "isTight",
    "charge": "charge",
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
        default=None,
        help=(
            "Optional cap on events loaded per H5 file for diagnostics. "
            "By default, use the run config unchanged."
        ),
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
    parser.add_argument(
        "--derived-electron-run-dir",
        help=(
            "Optional electron tokenizer run directory. If this and "
            "--derived-muon-run-dir are set, also plot leading-lepton dR_ll and m_ll."
        ),
    )
    parser.add_argument(
        "--derived-muon-run-dir",
        help=(
            "Optional muon tokenizer run directory. If this and "
            "--derived-electron-run-dir are set, also plot leading-lepton dR_ll and m_ll."
        ),
    )
    parser.add_argument(
        "--max-derived-events",
        type=int,
        default=200_000,
        help="Maximum number of events with at least two leptons for derived lepton plots.",
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


def transform_list_and_cst_fn_from_cfg(cfg) -> tuple[list, object | None]:
    datamodule = cfg.datamodule
    raw_transforms = datamodule.get("transforms")
    if raw_transforms is None:
        return [], None

    transforms = hydra.utils.instantiate(raw_transforms)
    if isinstance(transforms, (dict, DictConfig)):
        transform_list = [value for value in transforms.values() if callable(value)]
    elif isinstance(transforms, (list, tuple, ListConfig)):
        transform_list = [value for value in transforms if callable(value)]
    elif callable(transforms):
        transform_list = [transforms]
    else:
        transform_list = []

    if not transform_list:
        raise RuntimeError(
            "The datamodule config defines transforms, but none were instantiated "
            f"as callables (container type: {type(transforms).__name__}). Refusing "
            "to analyze raw H5 features as if they were preprocessed model inputs."
        )

    cst_fn = None
    for transform in transform_list:
        if isinstance(transform, functools.partial):
            cst_fn = (transform.keywords or {}).get("cst_fn", cst_fn)

    if cst_fn is None:
        raw_container = OmegaConf.to_container(raw_transforms, resolve=True)
        if isinstance(raw_container, dict):
            for transform_cfg in raw_container.values():
                if not isinstance(transform_cfg, dict):
                    continue
                cst_cfg = transform_cfg.get("cst_fn")
                if isinstance(cst_cfg, dict) and cst_cfg.get("filename"):
                    import joblib

                    cst_fn = joblib.load(cst_cfg["filename"])
                    break
    return transform_list, cst_fn


def analysis_datamodule_cfg(cfg, args: argparse.Namespace):
    datamodule_cfg = OmegaConf.create(OmegaConf.to_container(cfg.datamodule, resolve=True))
    OmegaConf.set_struct(datamodule_cfg, False)

    if args.h5_files:
        datamodule_cfg.data_paths = list(args.h5_files)
        if "data_path" in datamodule_cfg:
            del datamodule_cfg["data_path"]

        # Analysis must follow the fixed evaluation files exactly. Domain-aware
        # sampling is a training concern and saved domain labels no longer align
        # when --h5-files replaces the run's original input list.
        datamodule_cfg.data_domains = None
        datamodule_cfg.sampling_domain_fractions = None
        datamodule_cfg.sampling_num_samples = None

    if args.num_events_per_file is not None:
        datamodule_cfg.num_events = args.num_events_per_file

    datamodule_cfg.batch_size = args.batch_size
    datamodule_cfg.num_workers = args.num_workers
    if args.num_workers == 0:
        datamodule_cfg.persistent_workers = False
        datamodule_cfg.multiprocessing_context = None

    return datamodule_cfg


def dataloader_from_datamodule(datamodule, split: str) -> DataLoader:
    datamodule.setup(split)
    if split == "train":
        return datamodule.train_dataloader()
    if split == "test":
        return datamodule.test_dataloader()
    return datamodule.val_dataloader()


def feature_names_from_cfg(cfg, n_features: int) -> list[str]:
    datamodule = OmegaConf.to_container(cfg.datamodule, resolve=True)
    output_mode = datamodule.get("output_mode")
    collections = datamodule.get("object_collections") or []
    object_type = datamodule.get("object_type")

    inputs = []
    if output_mode == "object":
        for collection in collections:
            if collection.get("object_name") == object_type:
                inputs = collection.get("inputs") or []
                break
    elif collections:
        inputs = collections[0].get("inputs") or []

    names = [Path(path).name for path in inputs]
    if not names and object_type in DEFAULT_OBJECT_FEATURES:
        names = DEFAULT_OBJECT_FEATURES[object_type]
    if len(names) != n_features:
        if len(names) > n_features:
            names = names[:n_features]
        else:
            names = names + [f"feature_{idx}" for idx in range(len(names), n_features)]
    return names


def object_name_from_cfg(cfg) -> str:
    datamodule = OmegaConf.to_container(cfg.datamodule, resolve=True)
    if datamodule.get("output_mode") == "object" and datamodule.get("object_type"):
        return str(datamodule["object_type"]).rstrip("s")
    return "object"


def feature_count_from_cfg(cfg) -> int:
    datamodule = OmegaConf.to_container(cfg.datamodule, resolve=True)
    collections = datamodule.get("object_collections") or []
    object_type = datamodule.get("object_type")
    if datamodule.get("output_mode") == "object":
        for collection in collections:
            if collection.get("object_name") == object_type:
                return len(collection.get("inputs") or [])
    if collections:
        return len(collections[0].get("inputs") or [])
    raise ValueError("Could not infer feature count from datamodule config")


def to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def collect_diagnostics_from_loader(
    *,
    model: LitVqVae,
    loader: DataLoader,
    cst_inverse_transformer,
    device: torch.device,
    max_valid_objects: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    originals = []
    recons = []
    all_indices = []
    n_seen = 0
    logged_first_batch = False

    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            z_q, indices, _ = model.encode(batch)
            recon = model.decode(z_q, batch)

            mask = batch["mask"].bool()
            original_valid = batch["csts"][mask].detach().cpu().float().numpy()
            recon_valid = recon[mask].detach().cpu().float().numpy()
            indices_valid = indices[mask].detach().cpu().long().numpy()

            if not logged_first_batch:
                log.info("First diagnostic batch csts shape: %s", tuple(batch["csts"].shape))
                log.info("First diagnostic batch mask shape: %s", tuple(batch["mask"].shape))
                log.info("First diagnostic batch indices shape: %s", tuple(indices.shape))
                if len(original_valid) > 0:
                    log.info(
                        "First diagnostic batch transformed csts mean/std: %.5g %.5g",
                        float(np.mean(original_valid)),
                        float(np.std(original_valid)),
                    )
                    for quantizer_idx in range(indices_valid.shape[1]):
                        values = indices_valid[:, quantizer_idx]
                        log.info(
                            "First diagnostic batch q%d used=%d min=%d max=%d n=%d",
                            quantizer_idx,
                            int(len(np.unique(values))),
                            int(values.min()),
                            int(values.max()),
                            int(len(values)),
                        )
                logged_first_batch = True

            if cst_inverse_transformer is not None:
                original_valid = cst_inverse_transformer.inverse_transform(original_valid)
                recon_valid = cst_inverse_transformer.inverse_transform(recon_valid)

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


def canonical_dataloader_for_h5_files(
    *,
    cfg,
    h5_files: list[str] | None,
    split: str,
    batch_size: int,
    num_workers: int,
    num_events_per_file: int | None,
):
    """Build the saved-run dataloader used by every tokenizer diagnostic."""
    loader_args = argparse.Namespace(
        h5_files=list(h5_files) if h5_files else None,
        num_events_per_file=num_events_per_file,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    datamodule_cfg = analysis_datamodule_cfg(cfg, loader_args)
    datamodule = hydra.utils.instantiate(datamodule_cfg)
    return dataloader_from_datamodule(datamodule, split)


def collect_diagnostics_for_h5_files(
    *,
    cfg,
    model: LitVqVae,
    h5_files: list[str] | None,
    split: str,
    batch_size: int,
    num_workers: int,
    num_events_per_file: int | None,
    cst_inverse_transformer,
    device: torch.device,
    max_valid_objects: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Collect diagnostics through the canonical saved datamodule path.

    Plotting and comparison scripts must use this function so object loading,
    masking, preprocessing, ``model.encode()``, decoding, and optional inverse
    preprocessing are identical everywhere.
    """
    loader = canonical_dataloader_for_h5_files(
        cfg=cfg,
        h5_files=h5_files,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        num_events_per_file=num_events_per_file,
    )
    return collect_diagnostics_from_loader(
        model=model,
        loader=loader,
        cst_inverse_transformer=cst_inverse_transformer,
        device=device,
        max_valid_objects=max_valid_objects,
    )


def apply_hep_style(ax) -> None:
    ax.tick_params(direction="in", which="both", top=True, right=True, width=1.2, labelsize=13)
    ax.tick_params(which="major", length=7)
    ax.tick_params(which="minor", length=3.5)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def feature_label(name: str) -> str:
    lower = Path(name).name.lower()
    if lower in FEATURE_LABELS:
        return FEATURE_LABELS[lower]
    return name.replace("_", " ")


def binned_residual_iqr_over_median_truth(
    original: np.ndarray,
    reconstruction: np.ndarray,
    bins: np.ndarray,
    *,
    min_bin_count: int = 20,
    min_denominator: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Canonical relative-resolution metric used by tokenizer diagnostics.

    In each original-feature bin, compute
    IQR(reconstruction - original) / abs(median(original)).
    """
    original = np.asarray(original, dtype=np.float64)
    reconstruction = np.asarray(reconstruction, dtype=np.float64)
    bins = np.asarray(bins, dtype=np.float64)
    centers = 0.5 * (bins[:-1] + bins[1:])
    values = np.full(len(centers), np.nan, dtype=np.float64)
    counts = np.zeros(len(centers), dtype=np.int64)
    medians = np.full(len(centers), np.nan, dtype=np.float64)
    residual_iqrs = np.full(len(centers), np.nan, dtype=np.float64)

    for bin_idx in range(len(centers)):
        in_bin = (original >= bins[bin_idx]) & (original < bins[bin_idx + 1])
        if bin_idx == len(centers) - 1:
            in_bin = (original >= bins[bin_idx]) & (original <= bins[bin_idx + 1])
        finite = in_bin & np.isfinite(original) & np.isfinite(reconstruction)
        counts[bin_idx] = int(np.count_nonzero(finite))
        if counts[bin_idx] < min_bin_count:
            continue

        truth = original[finite]
        residual = reconstruction[finite] - truth
        median_original = float(np.median(truth))
        denominator = abs(median_original)
        if denominator < min_denominator:
            continue
        q25, q75 = np.percentile(residual, [25, 75])
        residual_iqr = float(q75 - q25)
        medians[bin_idx] = median_original
        residual_iqrs[bin_idx] = residual_iqr
        values[bin_idx] = residual_iqr / denominator

    return centers, values, counts, medians, residual_iqrs


def safe_filename(text: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text)


def display_values(
    original: np.ndarray,
    reconstruction: np.ndarray,
    feature_name: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    label = feature_label(feature_name)
    lower = Path(feature_name).name.lower()
    scale = 1.0
    unit = ""
    if lower in {"pt", "met", "sumet", "mass"} and np.nanpercentile(original, 95) > 1000:
        scale = 1000.0
        unit = " [GeV]"
    return original / scale, reconstruction / scale, f"{label}{unit}"


def finite_pair(original: np.ndarray, reconstruction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(original) & np.isfinite(reconstruction)
    return original[finite], reconstruction[finite]


def histogram_bins(values: np.ndarray, n_bins: int = 75) -> np.ndarray:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.linspace(0.0, 1.0, n_bins)

    unique = np.unique(values)
    integer_like = np.allclose(unique, np.round(unique), atol=1e-6)
    if len(unique) <= 20 and integer_like:
        lo = int(np.floor(unique.min()))
        hi = int(np.ceil(unique.max()))
        return np.arange(lo - 0.5, hi + 1.5, 1.0)

    lo, hi = np.percentile(values, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(values.min()), float(values.max())
    if lo == hi:
        width = abs(lo) * 0.05 if lo != 0 else 1.0
        lo, hi = lo - width, hi + width
    return np.linspace(lo, hi, n_bins)


def plot_feature_triptychs(
    original: np.ndarray,
    reconstruction: np.ndarray,
    feature_names: list[str],
    object_name: str,
    output_dir: Path,
) -> None:
    rng = np.random.default_rng(42)
    for feature_idx, feature_name in enumerate(feature_names):
        if feature_idx >= original.shape[1] or feature_idx >= reconstruction.shape[1]:
            log.warning("Skipping %s: feature is not in the reconstructed arrays", feature_name)
            continue
        orig, reco, axis_label = display_values(
            original[:, feature_idx],
            reconstruction[:, feature_idx],
            feature_name,
        )
        orig, reco = finite_pair(orig, reco)
        if len(orig) == 0:
            log.warning("Skipping %s: no finite values", feature_name)
            continue

        combined = np.concatenate([orig, reco])
        bins = histogram_bins(combined, n_bins=70)
        residual = reco - orig
        res_bins = histogram_bins(residual, n_bins=70)

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
        line_lo, line_hi = histogram_bins(np.concatenate([orig[idx], reco[idx]])).take([0, -1])
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
        if feature_idx >= original.shape[1] or feature_idx >= reconstruction.shape[1]:
            log.warning("Skipping %s: feature is not in the reconstructed arrays", feature_name)
            continue
        orig, reco, axis_label = display_values(
            original[:, feature_idx],
            reconstruction[:, feature_idx],
            feature_name,
        )
        orig, reco = finite_pair(orig, reco)
        if len(orig) == 0:
            log.warning("Skipping %s: no finite values", feature_name)
            continue
        combined = np.concatenate([orig, reco])
        bins = histogram_bins(combined, n_bins=75)

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


def reconstruction_metrics(
    original: np.ndarray,
    reconstruction: np.ndarray,
    feature_names: list[str],
) -> dict:
    metrics = {}
    for idx, name in enumerate(feature_names):
        orig, reco = finite_pair(original[:, idx], reconstruction[:, idx])
        if len(orig) == 0:
            continue
        residual = reco - orig
        metrics[name] = {
            "n": int(len(orig)),
            "mean_original": float(np.mean(orig)),
            "mean_reconstructed": float(np.mean(reco)),
            "bias": float(np.mean(residual)),
            "std": float(np.std(residual)),
            "mae": float(np.mean(np.abs(residual))),
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "p16": float(np.percentile(residual, 16)),
            "p50": float(np.percentile(residual, 50)),
            "p84": float(np.percentile(residual, 84)),
        }
    return metrics


def codebook_summary(counts: np.ndarray) -> dict:
    summary = {}
    for quantizer_idx, quantizer_counts in enumerate(counts):
        used = int(np.count_nonzero(quantizer_counts))
        dead = int(len(quantizer_counts) - used)
        assignments = int(quantizer_counts.sum())
        used_counts = quantizer_counts[quantizer_counts > 0]
        summary[f"quantizer_{quantizer_idx}"] = {
            "assignments": assignments,
            "used_codes": used,
            "dead_codes": dead,
            "total_codes": int(len(quantizer_counts)),
            "used_fraction": float(used / len(quantizer_counts)),
            "percent_used": float(100 * used / len(quantizer_counts)),
            "max_frequency": int(quantizer_counts.max()) if len(quantizer_counts) else 0,
            "mean_frequency_used": float(used_counts.mean()) if len(used_counts) else 0.0,
        }
    return summary


def plot_residual_summary(metrics: dict, output_path: Path) -> None:
    if not metrics:
        return
    names = list(metrics)
    rmse = [metrics[name]["rmse"] for name in names]
    mae = [metrics[name]["mae"] for name in names]
    bias = [metrics[name]["bias"] for name in names]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, values, title in [
        (axes[0], rmse, "RMSE"),
        (axes[1], mae, "MAE"),
        (axes[2], bias, "bias"),
    ]:
        ax.bar(names, values, color="#4C78A8")
        ax.axhline(0, color="black", linewidth=1.1)
        ax.set_title(title, fontsize=15)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=35)
        apply_hep_style(ax)
    fig.suptitle("Reconstruction Residual Summary", fontsize=16)
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


def delta_phi(phi_a: float, phi_b: float) -> float:
    return float((phi_a - phi_b + np.pi) % (2 * np.pi) - np.pi)


def delta_r(lep_a: dict, lep_b: dict) -> float:
    deta = lep_a["eta"] - lep_b["eta"]
    dphi = delta_phi(lep_a["phi"], lep_b["phi"])
    return float(np.sqrt(deta * deta + dphi * dphi))


def four_vector(pt: float, eta: float, phi: float, mass: float) -> np.ndarray:
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt(np.maximum(px * px + py * py + pz * pz + mass * mass, 0.0))
    return np.array([energy, px, py, pz], dtype=np.float64)


def invariant_mass(lep_a: dict, lep_b: dict) -> float:
    total = four_vector(**lep_a) + four_vector(**lep_b)
    m2 = total[0] ** 2 - total[1] ** 2 - total[2] ** 2 - total[3] ** 2
    return float(np.sqrt(max(m2, 0.0)))


def load_analysis_model(run_dir: Path, checkpoint: str | None, device: torch.device) -> LitVqVae:
    ckpt = find_checkpoint(run_dir, checkpoint)
    log.info("Loading %s", ckpt)
    model = LitVqVae.load_from_checkpoint(ckpt, map_location=device)
    model.to(device)
    model.eval()
    return model


def decode_event_batch(
    model: LitVqVae,
    batch: dict,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch = to_device(batch, device)
    z_q, _, _ = model.encode(batch)
    recon = model.decode(z_q, batch)
    return (
        batch["csts"].detach().cpu().float().numpy(),
        recon.detach().cpu().float().numpy(),
        batch["mask"].detach().cpu().bool().numpy(),
    )


def feature_indices(feature_names: list[str], required: tuple[str, ...]) -> dict[str, int]:
    lower_to_idx = {Path(name).name.lower(): idx for idx, name in enumerate(feature_names)}
    missing = [name for name in required if name not in lower_to_idx]
    if missing:
        raise ValueError(f"Missing required features for derived plots: {missing}")
    return {name: lower_to_idx[name] for name in required}


def append_leptons(
    leptons: list[dict],
    original: np.ndarray,
    reconstruction: np.ndarray,
    mask: np.ndarray,
    event_idx: int,
    indices: dict[str, int],
    mass: float,
) -> None:
    valid_objects = np.flatnonzero(mask[event_idx])
    for obj_idx in valid_objects:
        orig = original[event_idx, obj_idx]
        reco = reconstruction[event_idx, obj_idx]
        values = [
            orig[indices["pt"]],
            orig[indices["eta"]],
            orig[indices["phi"]],
            reco[indices["pt"]],
            reco[indices["eta"]],
            reco[indices["phi"]],
        ]
        if not np.all(np.isfinite(values)) or orig[indices["pt"]] <= 0:
            continue
        leptons.append(
            {
                "sort_pt": float(orig[indices["pt"]]),
                "original": {
                    "pt": float(orig[indices["pt"]]),
                    "eta": float(orig[indices["eta"]]),
                    "phi": float(orig[indices["phi"]]),
                    "mass": mass,
                },
                "reconstructed": {
                    "pt": float(reco[indices["pt"]]),
                    "eta": float(reco[indices["eta"]]),
                    "phi": float(reco[indices["phi"]]),
                    "mass": mass,
                },
            }
        )


def collect_lepton_pair_diagnostics(
    *,
    electron_model: LitVqVae,
    muon_model: LitVqVae,
    electron_data_paths: list[str],
    electron_dataset_kwargs: dict,
    muon_dataset_kwargs: dict,
    electron_feature_names: list[str],
    muon_feature_names: list[str],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    max_events: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    electron_idx = feature_indices(electron_feature_names, ("pt", "eta", "phi"))
    muon_idx = feature_indices(muon_feature_names, ("pt", "eta", "phi"))
    values = {
        "dR_ll": {"original": [], "reconstructed": []},
        "m_ll": {"original": [], "reconstructed": []},
    }

    for data_path in electron_data_paths:
        log.info("Analyzing derived leptons in %s", data_path)
        electron_dataset = AtlasEventMapDataset(data_path, **electron_dataset_kwargs)
        muon_dataset = AtlasEventMapDataset(data_path, **muon_dataset_kwargs)
        if len(electron_dataset) != len(muon_dataset):
            raise ValueError(f"Electron and muon datasets have different lengths in {data_path}")

        electron_loader = DataLoader(
            electron_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
        muon_loader = DataLoader(
            muon_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )

        with torch.no_grad():
            for electron_batch, muon_batch in zip(electron_loader, muon_loader):
                e_orig, e_reco, e_mask = decode_event_batch(electron_model, electron_batch, device)
                m_orig, m_reco, m_mask = decode_event_batch(muon_model, muon_batch, device)
                n_events = e_orig.shape[0]

                for event_idx in range(n_events):
                    leptons = []
                    append_leptons(
                        leptons,
                        e_orig,
                        e_reco,
                        e_mask,
                        event_idx,
                        electron_idx,
                        0.000511,
                    )
                    append_leptons(
                        leptons,
                        m_orig,
                        m_reco,
                        m_mask,
                        event_idx,
                        muon_idx,
                        0.10566,
                    )
                    if len(leptons) < 2:
                        continue

                    lep_a, lep_b = sorted(leptons, key=lambda lep: lep["sort_pt"], reverse=True)[:2]
                    orig_a, orig_b = lep_a["original"], lep_b["original"]
                    reco_a, reco_b = lep_a["reconstructed"], lep_b["reconstructed"]

                    values["dR_ll"]["original"].append(delta_r(orig_a, orig_b))
                    values["dR_ll"]["reconstructed"].append(delta_r(reco_a, reco_b))
                    values["m_ll"]["original"].append(invariant_mass(orig_a, orig_b))
                    values["m_ll"]["reconstructed"].append(invariant_mass(reco_a, reco_b))

                    if len(values["dR_ll"]["original"]) >= max_events:
                        return {
                            name: (
                                np.asarray(items["original"], dtype=np.float32),
                                np.asarray(items["reconstructed"], dtype=np.float32),
                            )
                            for name, items in values.items()
                        }

    return {
        name: (
            np.asarray(items["original"], dtype=np.float32),
            np.asarray(items["reconstructed"], dtype=np.float32),
        )
        for name, items in values.items()
    }


def plot_derived_triptych(
    original: np.ndarray,
    reconstruction: np.ndarray,
    *,
    name: str,
    label: str,
    output_path: Path,
) -> dict:
    original, reconstruction = finite_pair(original, reconstruction)
    if len(original) == 0:
        return {}

    bins = histogram_bins(np.concatenate([original, reconstruction]), n_bins=70)
    residual = reconstruction - original
    residual_bins = histogram_bins(residual, n_bins=70)
    rng = np.random.default_rng(42)
    n_plot = min(len(original), 50_000)
    sample_idx = rng.choice(len(original), size=n_plot, replace=False)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    axes[0].hist(original, bins=bins, histtype="step", linewidth=1.9, label="original")
    axes[0].hist(reconstruction, bins=bins, histtype="step", linewidth=1.9, label="reconstructed")
    axes[0].set_title(label, fontsize=16)
    axes[0].set_xlabel(label, fontsize=13)
    axes[0].set_ylabel("events", fontsize=13)
    axes[0].legend(frameon=True, fontsize=11)
    apply_hep_style(axes[0])

    axes[1].hist(residual, bins=residual_bins, color="#1f77b4", alpha=0.75)
    axes[1].axvline(0, color="black", linewidth=1.2)
    axes[1].set_title("residual", fontsize=16)
    axes[1].set_xlabel("reco - original", fontsize=13)
    axes[1].set_ylabel("events", fontsize=13)
    apply_hep_style(axes[1])

    line_lo, line_hi = histogram_bins(
        np.concatenate([original[sample_idx], reconstruction[sample_idx]])
    ).take([0, -1])
    axes[2].scatter(
        original[sample_idx],
        reconstruction[sample_idx],
        s=3,
        alpha=0.22,
        rasterized=True,
    )
    axes[2].plot([line_lo, line_hi], [line_lo, line_hi], color="black", linewidth=1.2)
    axes[2].set_xlim(line_lo, line_hi)
    axes[2].set_ylim(line_lo, line_hi)
    axes[2].set_title("correlation", fontsize=16)
    axes[2].set_xlabel("original", fontsize=13)
    axes[2].set_ylabel("reconstructed", fontsize=13)
    apply_hep_style(axes[2])

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return {
        "n": int(len(original)),
        "bias": float(np.mean(residual)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "p16": float(np.percentile(residual, 16)),
        "p50": float(np.percentile(residual, 50)),
        "p84": float(np.percentile(residual, 84)),
    }


def run_derived_lepton_diagnostics(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
) -> None:
    electron_run_dir = Path(args.derived_electron_run_dir).resolve()
    muon_run_dir = Path(args.derived_muon_run_dir).resolve()
    electron_cfg = OmegaConf.load(electron_run_dir / "full_config.yaml")
    muon_cfg = OmegaConf.load(muon_run_dir / "full_config.yaml")

    electron_data_paths, electron_dataset_kwargs = dataset_kwargs_from_cfg(
        electron_cfg,
        args,
        electron_run_dir,
    )
    _, muon_dataset_kwargs = dataset_kwargs_from_cfg(muon_cfg, args, muon_run_dir)
    electron_model = load_analysis_model(electron_run_dir, None, device)
    muon_model = load_analysis_model(muon_run_dir, None, device)

    electron_feature_names = feature_names_from_cfg(electron_cfg, feature_count_from_cfg(electron_cfg))
    muon_feature_names = feature_names_from_cfg(muon_cfg, feature_count_from_cfg(muon_cfg))

    derived = collect_lepton_pair_diagnostics(
        electron_model=electron_model,
        muon_model=muon_model,
        electron_data_paths=electron_data_paths,
        electron_dataset_kwargs=electron_dataset_kwargs,
        muon_dataset_kwargs=muon_dataset_kwargs,
        electron_feature_names=electron_feature_names,
        muon_feature_names=muon_feature_names,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        max_events=args.max_derived_events,
    )

    derived_dir = output_dir / "derived_leptons"
    derived_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    labels = {
        "dR_ll": r"leading lepton $\Delta R_{\ell\ell}$",
        "m_ll": r"leading lepton $m_{\ell\ell}$ [GeV]",
    }
    for name, (original, reconstruction) in derived.items():
        summary[name] = plot_derived_triptych(
            original,
            reconstruction,
            name=name,
            label=labels[name],
            output_path=derived_dir / f"{name}_reconstruction_triptych.png",
        )
    (derived_dir / "derived_lepton_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    log.info("Wrote derived lepton diagnostics to %s", derived_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    run_dir = Path(args.run_dir).resolve()
    cfg_path = run_dir / "full_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    cfg = OmegaConf.load(cfg_path)
    _, cst_inverse_transformer = transform_list_and_cst_fn_from_cfg(cfg)
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

    log.info("Collecting diagnostics through the canonical saved datamodule path")
    original, reconstruction, indices, n_seen = collect_diagnostics_for_h5_files(
        cfg=cfg,
        model=model,
        h5_files=list(args.h5_files) if args.h5_files else None,
        split=args.split,
        num_events_per_file=args.num_events_per_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cst_inverse_transformer=cst_inverse_transformer,
        device=device,
        max_valid_objects=args.max_valid_objects,
    )

    feature_names = feature_names_from_cfg(cfg, original.shape[1])
    object_name = object_name_from_cfg(cfg)
    codebook_size = int(getattr(model.hparams, "codebook_size", cfg.model.codebook_size))
    counts = codebook_counts(indices, codebook_size)
    reco_metrics = reconstruction_metrics(original, reconstruction, feature_names)
    usage_summary = codebook_summary(counts)

    np.save(output_dir / "codebook_counts.npy", counts)
    (output_dir / "reconstruction_metrics.json").write_text(
        json.dumps(reco_metrics, indent=2) + "\n"
    )
    (output_dir / "codebook_usage_summary.json").write_text(
        json.dumps(usage_summary, indent=2) + "\n"
    )
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
    plot_residual_summary(reco_metrics, output_dir / "residual_summary.png")
    write_summary(
        output_dir / "summary.txt",
        counts=counts,
        original=original,
        reconstruction=reconstruction,
        feature_names=feature_names,
        n_seen=n_seen,
    )
    log.info("Wrote tokenizer diagnostics to %s", output_dir)

    if args.derived_electron_run_dir and args.derived_muon_run_dir:
        run_derived_lepton_diagnostics(args=args, output_dir=output_dir, device=device)


if __name__ == "__main__":
    main()
