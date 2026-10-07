"""Physics diagnostics from event-token parquet reconstruction.

This decodes object tokens from an event-level token parquet back through the
per-object VQ-VAE decoders, then compares event observables against the
original MC H5 files.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pyarrow.parquet as pq
import torch
from joblib import load as joblib_load
from omegaconf import OmegaConf

from heptokens.models.vq_vae import LitVqVae

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger(__name__)

ELECTRON_MASS_GEV = 0.000511
MUON_MASS_GEV = 0.10566
Z_MASS_GEV = 91.1876


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode event-token parquet rows and plot HZZ-style physics observables."
    )
    parser.add_argument("--parquet", required=True, help="Event token parquet.")
    parser.add_argument(
        "--h5-dir",
        default="/home/zephyr/Data/viviana/bnl-treasure/data/h5",
        help="Directory with MC H5 files. realdata/ is intentionally not searched.",
    )
    parser.add_argument("--h5-files", nargs="*", default=[], help="Optional explicit MC H5 files.")
    parser.add_argument(
        "--datamodule-config",
        default="configs/datamodule/atlas_event_object.yaml",
        help="Object feature config used for tokenizer training.",
    )
    parser.add_argument(
        "--tokenizer-checkpoints",
        nargs="+",
        required=True,
        help="Object checkpoint map, e.g. electrons=/path/best.ckpt muons=/path/best.ckpt.",
    )
    parser.add_argument(
        "--preprocess-transformers",
        nargs="+",
        required=True,
        help="Object preprocessing map, e.g. electrons=/path/electrons_log_standard.joblib.",
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        default=["electrons", "muons", "jets"],
        help="Objects to decode. HZZ observables need electrons and muons; n_jets/HT need jets.",
    )
    parser.add_argument("--output-dir", help="Defaults to <parquet stem>_physics_reco.")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-events", type=int, default=50_000)
    parser.add_argument("--max-objects", type=int, default=200)
    parser.add_argument("--jet-pt-threshold", type=float, default=20.0)
    parser.add_argument(
        "--event-index-mode",
        choices=["source_order", "parquet_column"],
        default="source_order",
        help=(
            "source_order uses row order within each source_file as the local H5 event index. "
            "Use parquet_column only if event_index is known to be local within each H5 file."
        ),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def parse_path_map(items: list[str], what: str) -> dict[str, str]:
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Each {what} must look like object=/path/file")
        key, value = item.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_config(path: str) -> dict:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def collection_map(config: dict) -> dict[str, dict]:
    return {
        collection["object_name"]: collection
        for collection in config.get("object_collections") or []
    }


def feature_names(collection: dict) -> list[str]:
    return [Path(path).name for path in collection.get("inputs") or []]


def feature_index(names: list[str], candidates: tuple[str, ...]) -> int | None:
    lookup = {name.lower(): idx for idx, name in enumerate(names)}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def collect_h5_files(args: argparse.Namespace) -> dict[str, Path]:
    paths = [Path(path) for path in args.h5_files]
    if args.h5_dir:
        paths.extend(sorted(Path(args.h5_dir).glob("*.h5")))
    files = {}
    for path in paths:
        if path.is_file() and path.stat().st_size > 0:
            files[path.name] = path
    if not files:
        raise ValueError("No MC H5 files found")
    return files


def load_models(checkpoints: dict[str, str], objects: list[str], device: torch.device) -> dict:
    models = {}
    for object_name in objects:
        if object_name not in checkpoints:
            continue
        checkpoint = checkpoints[object_name]
        log.info("Loading %s tokenizer from %s", object_name, checkpoint)
        model = LitVqVae.load_from_checkpoint(checkpoint, map_location=device)
        model.to(device)
        model.eval()
        models[object_name] = model
    return models


def load_preprocessors(paths: dict[str, str], objects: list[str]) -> dict:
    preprocessors = {}
    for object_name in objects:
        if object_name not in paths:
            continue
        path = Path(paths[object_name])
        if not path.exists():
            raise FileNotFoundError(path)
        log.info("Loading %s preprocessing from %s", object_name, path)
        preprocessors[object_name] = joblib_load(path)
    return preprocessors


def vocabulary_from_parquet(path: str) -> dict:
    metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
    key = b"heptokens_token_vocabulary"
    if key not in metadata:
        raise ValueError(f"{path} has no heptokens_token_vocabulary schema metadata")
    return json.loads(metadata[key])


def read_h5_path(handle: h5py.File, h5_path: str, event_indices: np.ndarray) -> np.ndarray:
    array = handle[h5_path][event_indices]
    return np.asarray(array)


def read_original_objects(
    *,
    handle: h5py.File,
    collection: dict,
    event_indices: np.ndarray,
    max_objects: int,
) -> tuple[np.ndarray, np.ndarray]:
    arrays = []
    n_objects = None
    for path in collection["inputs"]:
        array = read_h5_path(handle, path, event_indices).astype(np.float32)
        if array.ndim == 1:
            array = array[:, None]
        if n_objects is None:
            n_objects = min(array.shape[1], max_objects)
        arrays.append(array[:, :n_objects])
    csts = np.stack(arrays, axis=-1)
    csts = np.nan_to_num(csts, nan=0.0, posinf=0.0, neginf=0.0)
    csts = np.clip(csts, -1e4, 1e4)
    mask = read_h5_path(handle, collection["mask_input"], event_indices).astype(bool)
    if mask.ndim == 1:
        mask = mask[:, None]
    mask = mask[:, :n_objects]
    return csts, mask


def indices_from_event_tokens(
    tokens: np.ndarray,
    mask: np.ndarray,
    type_ids: np.ndarray,
    object_name: str,
    object_vocab: dict,
    max_objects: int,
) -> tuple[np.ndarray, np.ndarray]:
    type_id = int(object_vocab["type_id"])
    quantizers = object_vocab["quantizers"]
    n_quantizers = int(object_vocab["num_quantizers"])
    object_tokens = tokens[(mask.astype(bool)) & (type_ids == type_id)]
    n_complete = min(len(object_tokens) // n_quantizers, max_objects)

    indices = np.full((max_objects, n_quantizers), -1, dtype=np.int64)
    object_mask = np.zeros(max_objects, dtype=bool)
    for object_idx in range(n_complete):
        group = object_tokens[object_idx * n_quantizers : (object_idx + 1) * n_quantizers]
        ok = True
        codes = []
        for quantizer_idx, token in enumerate(group):
            base = int(quantizers[quantizer_idx]["base"])
            size = int(quantizers[quantizer_idx]["size"])
            code = int(token) - base
            if code < 0 or code >= size:
                ok = False
                break
            codes.append(code)
        if ok:
            indices[object_idx] = codes
            object_mask[object_idx] = True
    return indices, object_mask


def _codebook_tensor(layer, codebook_size: int, codebook_dim: int) -> torch.Tensor | None:
    codebook = getattr(layer, "_codebook", layer)
    for attr in ("embed", "codebook", "weight"):
        tensor = getattr(codebook, attr, None)
        if not torch.is_tensor(tensor):
            continue
        tensor = tensor.detach()
        if tensor.ndim == 3 and tensor.shape[-2:] == (codebook_size, codebook_dim):
            return tensor.reshape(-1, codebook_size, codebook_dim)[0]
        if tensor.ndim == 2 and tensor.shape == (codebook_size, codebook_dim):
            return tensor
    embedding = getattr(codebook, "embedding", None)
    if embedding is not None and hasattr(embedding, "weight"):
        tensor = embedding.weight.detach()
        if tensor.shape == (codebook_size, codebook_dim):
            return tensor
    return None


def quantized_from_indices(model: LitVqVae, indices: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    clean_indices = indices.clamp_min(0)
    candidates = [clean_indices, clean_indices.permute(2, 0, 1).contiguous()]
    for candidate in candidates:
        method = getattr(model.vector_quantization, "get_output_from_indices", None)
        if method is None:
            break
        try:
            output = method(candidate.to(next(model.parameters()).device))
            if isinstance(output, tuple):
                output = output[0]
            if output.ndim == 4 and output.shape[:3] == (*indices.shape,):
                output = output.sum(dim=2)
            elif output.ndim == 4 and output.shape[1:3] == indices.shape[:2]:
                output = output.sum(dim=0)
            if output.ndim == 3 and output.shape[:2] == indices.shape[:2]:
                return output.masked_fill(~mask.to(output.device).unsqueeze(-1), 0.0)
        except Exception:
            continue

    layers = getattr(model.vector_quantization, "layers", None)
    if layers is None:
        raise RuntimeError("Could not find ResidualVQ layers or get_output_from_indices")

    device = next(model.parameters()).device
    codebook_size = int(model.hparams.codebook_size)
    codebook_dim = int(model.hparams.codebook_dim)
    z_q = torch.zeros((*indices.shape[:2], codebook_dim), dtype=torch.float32, device=device)
    clean_indices = clean_indices.to(device)
    for quantizer_idx, layer in enumerate(layers):
        if quantizer_idx >= indices.shape[-1]:
            break
        codebook = _codebook_tensor(layer, codebook_size, codebook_dim)
        if codebook is None:
            raise RuntimeError(f"Could not read codebook tensor for quantizer {quantizer_idx}")
        z_q += codebook.to(device)[clean_indices[..., quantizer_idx]]
    return z_q.masked_fill(~mask.to(device).unsqueeze(-1), 0.0)


def inverse_transform_valid(values: np.ndarray, mask: np.ndarray, transformer) -> np.ndarray:
    if transformer is None:
        return values
    values = values.copy()
    if mask.any():
        values[mask] = transformer.inverse_transform(values[mask])
    return values


def decode_object_tokens(
    *,
    model: LitVqVae,
    indices: np.ndarray,
    mask: np.ndarray,
    transformer,
    device: torch.device,
    n_features: int,
) -> np.ndarray:
    indices_t = torch.from_numpy(indices).long().to(device)
    mask_t = torch.from_numpy(mask).bool().to(device)
    z_q = quantized_from_indices(model, indices_t, mask_t)
    batch = {
        "csts": torch.zeros((*mask.shape, n_features), dtype=torch.float32, device=device),
        "mask": mask_t,
    }
    with torch.no_grad():
        reco = model.decode(z_q, batch).detach().cpu().float().numpy()
    return inverse_transform_valid(reco, mask, transformer)


def four_vector(pt: float, eta: float, phi: float, mass: float) -> np.ndarray:
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt(max(px * px + py * py + pz * pz + mass * mass, 0.0))
    return np.array([energy, px, py, pz], dtype=np.float64)


def invariant_mass(vectors: list[np.ndarray]) -> float:
    if not vectors:
        return math.nan
    total = np.sum(vectors, axis=0)
    m2 = total[0] ** 2 - total[1] ** 2 - total[2] ** 2 - total[3] ** 2
    return float(np.sqrt(max(m2, 0.0)))


def leptons_from_objects(
    values: np.ndarray,
    mask: np.ndarray,
    names: list[str],
    *,
    flavor: str,
    mass: float,
) -> list[dict]:
    pt_idx = feature_index(names, ("pt",))
    eta_idx = feature_index(names, ("eta",))
    phi_idx = feature_index(names, ("phi",))
    charge_idx = feature_index(names, ("charge",))
    if pt_idx is None or eta_idx is None or phi_idx is None:
        return []

    leptons = []
    for idx in np.flatnonzero(mask):
        row = values[idx]
        pt, eta, phi = float(row[pt_idx]), float(row[eta_idx]), float(row[phi_idx])
        if not np.all(np.isfinite([pt, eta, phi])) or pt <= 0:
            continue
        charge = float(row[charge_idx]) if charge_idx is not None else math.nan
        leptons.append(
            {
                "idx": (flavor, int(idx)),
                "flavor": flavor,
                "pt": pt,
                "eta": eta,
                "phi": phi,
                "charge": charge,
                "mass": mass,
            }
        )
    return leptons


def lepton_vector(lepton: dict) -> np.ndarray:
    return four_vector(lepton["pt"], lepton["eta"], lepton["phi"], lepton["mass"])


def choose_z_pairs(leptons: list[dict]) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    candidates = []
    for i in range(len(leptons)):
        for j in range(i + 1, len(leptons)):
            a, b = leptons[i], leptons[j]
            if a["flavor"] != b["flavor"]:
                continue
            if np.isfinite(a["charge"]) and np.isfinite(b["charge"]) and a["charge"] * b["charge"] >= 0:
                continue
            mass = invariant_mass([lepton_vector(a), lepton_vector(b)])
            candidates.append((abs(mass - Z_MASS_GEV), i, j, mass))
    if not candidates:
        return None, None
    candidates.sort()
    _, i, j, _ = candidates[0]
    z1 = (i, j)
    remaining = [idx for idx in range(len(leptons)) if idx not in z1]
    z2_candidates = []
    for a_pos in range(len(remaining)):
        for b_pos in range(a_pos + 1, len(remaining)):
            i2, j2 = remaining[a_pos], remaining[b_pos]
            a, b = leptons[i2], leptons[j2]
            if a["flavor"] != b["flavor"]:
                continue
            if np.isfinite(a["charge"]) and np.isfinite(b["charge"]) and a["charge"] * b["charge"] >= 0:
                continue
            mass = invariant_mass([lepton_vector(a), lepton_vector(b)])
            z2_candidates.append((abs(mass - Z_MASS_GEV), i2, j2, mass))
    if not z2_candidates:
        return z1, None
    z2_candidates.sort()
    _, i2, j2, _ = z2_candidates[0]
    return z1, (i2, j2)


def event_observables(
    original: dict[str, tuple[np.ndarray, np.ndarray]],
    reconstructed: dict[str, tuple[np.ndarray, np.ndarray]],
    names_by_object: dict[str, list[str]],
    jet_pt_threshold: float,
) -> dict[str, tuple[float, float]]:
    orig_leptons = []
    reco_leptons_by_idx = {}

    for object_name, mass in [("electrons", ELECTRON_MASS_GEV), ("muons", MUON_MASS_GEV)]:
        if object_name not in original or object_name not in reconstructed:
            continue
        orig_values, orig_mask = original[object_name]
        reco_values, reco_mask = reconstructed[object_name]
        names = names_by_object[object_name]

        orig = leptons_from_objects(
            orig_values,
            orig_mask,
            names,
            flavor=object_name,
            mass=mass,
        )
        reco = leptons_from_objects(
            reco_values,
            reco_mask,
            names,
            flavor=object_name,
            mass=mass,
        )
        orig_leptons.extend(orig)
        reco_leptons_by_idx.update({lep["idx"]: lep for lep in reco})

    orig_leptons = sorted(orig_leptons, key=lambda lep: lep["pt"], reverse=True)
    reco_leptons = [reco_leptons_by_idx.get(lep["idx"]) for lep in orig_leptons]
    reco_leptons = [lep for lep in reco_leptons if lep is not None]

    out = {}
    if len(orig_leptons) >= 1 and len(reco_leptons) >= 1:
        out["leading_lepton_pt"] = (orig_leptons[0]["pt"], reco_leptons[0]["pt"])
    if len(orig_leptons) >= 2 and len(reco_leptons) >= 2:
        out["subleading_lepton_pt"] = (orig_leptons[1]["pt"], reco_leptons[1]["pt"])
    if len(orig_leptons) >= 4 and len(reco_leptons) >= 4:
        selected_orig = orig_leptons[:4]
        selected_reco = [reco_leptons_by_idx.get(lep["idx"]) for lep in selected_orig]
        if all(lep is not None for lep in selected_reco):
            out["m4l"] = (
                invariant_mass([lepton_vector(lep) for lep in selected_orig]),
                invariant_mass([lepton_vector(lep) for lep in selected_reco]),
            )

        z1, z2 = choose_z_pairs(orig_leptons)
        if z1 is not None:
            z1_orig = [orig_leptons[z1[0]], orig_leptons[z1[1]]]
            z1_reco = [reco_leptons_by_idx.get(lep["idx"]) for lep in z1_orig]
            if all(lep is not None for lep in z1_reco):
                out["mZ1"] = (
                    invariant_mass([lepton_vector(lep) for lep in z1_orig]),
                    invariant_mass([lepton_vector(lep) for lep in z1_reco]),
                )
        if z2 is not None:
            z2_orig = [orig_leptons[z2[0]], orig_leptons[z2[1]]]
            z2_reco = [reco_leptons_by_idx.get(lep["idx"]) for lep in z2_orig]
            if all(lep is not None for lep in z2_reco):
                out["mZ2"] = (
                    invariant_mass([lepton_vector(lep) for lep in z2_orig]),
                    invariant_mass([lepton_vector(lep) for lep in z2_reco]),
                )

    if "jets" in original and "jets" in reconstructed:
        names = names_by_object["jets"]
        pt_idx = feature_index(names, ("pt",))
        if pt_idx is not None:
            orig_values, orig_mask = original["jets"]
            reco_values, reco_mask = reconstructed["jets"]
            orig_pts = orig_values[orig_mask, pt_idx]
            reco_pts = reco_values[reco_mask, pt_idx]
            orig_selected = orig_pts[np.isfinite(orig_pts) & (orig_pts > jet_pt_threshold)]
            reco_selected = reco_pts[np.isfinite(reco_pts) & (reco_pts > jet_pt_threshold)]
            out["n_jets"] = (float(len(orig_selected)), float(len(reco_selected)))
            out["HT"] = (float(np.sum(orig_selected)), float(np.sum(reco_selected)))

    return out


def append_observables(storage: dict[str, list[tuple[float, float]]], observables: dict[str, tuple[float, float]]) -> None:
    for name, (orig, reco) in observables.items():
        if np.isfinite(orig) and np.isfinite(reco):
            storage[name].append((float(orig), float(reco)))


def metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    residual = reconstructed - original
    return {
        "n": int(len(original)),
        "mean_original": float(np.mean(original)),
        "mean_reconstructed": float(np.mean(reconstructed)),
        "bias": float(np.mean(residual)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "p16": float(np.percentile(residual, 16)),
        "p50": float(np.percentile(residual, 50)),
        "p84": float(np.percentile(residual, 84)),
    }


def histogram_bins(values: np.ndarray, bins: int = 70) -> np.ndarray:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.linspace(0, 1, bins)
    unique = np.unique(values)
    if len(unique) <= 25 and np.allclose(unique, np.round(unique), atol=1e-6):
        return np.arange(np.floor(unique.min()) - 0.5, np.ceil(unique.max()) + 1.5, 1.0)
    lo, hi = np.percentile(values, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(values.min()), float(values.max())
    if lo == hi:
        width = abs(lo) * 0.05 if lo else 1.0
        lo, hi = lo - width, hi + width
    return np.linspace(lo, hi, bins)


def apply_style(ax) -> None:
    ax.tick_params(direction="in", which="both", top=True, right=True, width=1.2, labelsize=12)
    ax.tick_params(which="major", length=7)
    ax.tick_params(which="minor", length=3.5)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def plot_observable(name: str, original: np.ndarray, reconstructed: np.ndarray, output_dir: Path) -> None:
    bins = histogram_bins(np.concatenate([original, reconstructed]))
    residual = reconstructed - original
    res_bins = histogram_bins(residual)
    rng = np.random.default_rng(42)
    n_plot = min(len(original), 50_000)
    idx = rng.choice(len(original), size=n_plot, replace=False)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    axes[0].hist(original, bins=bins, histtype="step", linewidth=1.9, label="original")
    axes[0].hist(reconstructed, bins=bins, histtype="step", linewidth=1.9, label="token reco")
    axes[0].set_title(name)
    axes[0].set_xlabel(name)
    axes[0].set_ylabel("events")
    axes[0].legend()
    apply_style(axes[0])

    axes[1].hist(residual, bins=res_bins, alpha=0.78)
    axes[1].axvline(0, color="black", linewidth=1.1)
    axes[1].set_title("residual")
    axes[1].set_xlabel("reco - original")
    axes[1].set_ylabel("events")
    apply_style(axes[1])

    line_lo, line_hi = histogram_bins(np.concatenate([original[idx], reconstructed[idx]])).take([0, -1])
    axes[2].scatter(original[idx], reconstructed[idx], s=3, alpha=0.22, rasterized=True)
    axes[2].plot([line_lo, line_hi], [line_lo, line_hi], color="black", linewidth=1.1)
    axes[2].set_xlim(line_lo, line_hi)
    axes[2].set_ylim(line_lo, line_hi)
    axes[2].set_title("correlation")
    axes[2].set_xlabel("original")
    axes[2].set_ylabel("token reco")
    apply_style(axes[2])

    fig.tight_layout()
    fig.savefig(output_dir / f"{name}_event_token_reco.png", dpi=180)
    plt.close(fig)


def process_group(
    *,
    source_file: str,
    rows: dict[str, np.ndarray],
    h5_path: Path,
    collections: dict[str, dict],
    names_by_object: dict[str, list[str]],
    vocabulary: dict,
    models: dict,
    preprocessors: dict,
    objects: list[str],
    max_objects: int,
    device: torch.device,
    jet_pt_threshold: float,
    storage: dict[str, list[tuple[float, float]]],
    event_index_mode: str,
) -> int:
    if event_index_mode == "parquet_column":
        event_indices = rows["event_index"].astype(np.int64)
    else:
        event_indices = np.arange(len(rows["event_index"]), dtype=np.int64)
    n_events = len(event_indices)

    original_by_object = {}
    reconstructed_by_object = {}

    with h5py.File(h5_path, "r") as handle:
        for object_name in objects:
            if object_name not in models or object_name not in vocabulary["objects"]:
                continue
            if object_name not in collections:
                continue

            original_csts, original_mask = read_original_objects(
                handle=handle,
                collection=collections[object_name],
                event_indices=event_indices,
                max_objects=max_objects,
            )
            n_obj = min(original_csts.shape[1], max_objects)
            token_indices = np.full(
                (n_events, n_obj, int(vocabulary["objects"][object_name]["num_quantizers"])),
                -1,
                dtype=np.int64,
            )
            token_mask = np.zeros((n_events, n_obj), dtype=bool)

            for row_idx in range(n_events):
                indices, object_mask = indices_from_event_tokens(
                    rows["tokens"][row_idx],
                    rows["mask"][row_idx],
                    rows["type_ids"][row_idx],
                    object_name,
                    vocabulary["objects"][object_name],
                    n_obj,
                )
                token_indices[row_idx] = indices[:n_obj]
                token_mask[row_idx] = object_mask[:n_obj]

            reco_csts = decode_object_tokens(
                model=models[object_name],
                indices=token_indices,
                mask=token_mask,
                transformer=preprocessors.get(object_name),
                device=device,
                n_features=original_csts.shape[-1],
            )

            original_by_object[object_name] = (original_csts, original_mask)
            reconstructed_by_object[object_name] = (reco_csts, token_mask)

    for event_idx in range(n_events):
        original_event = {
            object_name: (values[event_idx], mask[event_idx])
            for object_name, (values, mask) in original_by_object.items()
        }
        reconstructed_event = {
            object_name: (values[event_idx], mask[event_idx])
            for object_name, (values, mask) in reconstructed_by_object.items()
        }
        append_observables(
            storage,
            event_observables(
                original_event,
                reconstructed_event,
                names_by_object,
                jet_pt_threshold,
            ),
        )
    return n_events


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    device = choose_device(args.device)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.parquet).with_suffix("").with_name(Path(args.parquet).stem + "_physics_reco")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.datamodule_config)
    collections = collection_map(config)
    names_by_object = {object_name: feature_names(collections[object_name]) for object_name in args.objects if object_name in collections}
    h5_files = collect_h5_files(args)
    vocabulary = vocabulary_from_parquet(args.parquet)
    models = load_models(parse_path_map(args.tokenizer_checkpoints, "checkpoint"), args.objects, device)
    preprocessors = load_preprocessors(parse_path_map(args.preprocess_transformers, "preprocess transformer"), args.objects)

    parquet = pq.ParquetFile(args.parquet)
    required_columns = ["tokens", "mask", "type_ids", "source_file", "event_index"]
    if "is_mc" in parquet.schema_arrow.names:
        required_columns.append("is_mc")

    storage: dict[str, list[tuple[float, float]]] = defaultdict(list)
    grouped_rows: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    processed = 0

    for batch in parquet.iter_batches(batch_size=args.batch_size, columns=required_columns):
        table = batch.to_pydict()
        n_rows = len(table["source_file"])
        for row_idx in range(n_rows):
            if processed >= args.max_events:
                break
            source_file = table["source_file"][row_idx]
            if source_file not in h5_files:
                continue
            if "is_mc" in table and not bool(table["is_mc"][row_idx]):
                continue
            grouped_rows[source_file]["tokens"].append(table["tokens"][row_idx])
            grouped_rows[source_file]["mask"].append(table["mask"][row_idx])
            grouped_rows[source_file]["type_ids"].append(table["type_ids"][row_idx])
            grouped_rows[source_file]["event_index"].append(table["event_index"][row_idx])
            processed += 1
        if processed >= args.max_events:
            break

    if processed == 0:
        raise RuntimeError("No MC parquet rows matched the provided H5 files")

    log.info("Processing %s MC events from %s source files", processed, len(grouped_rows))
    for source_file, rows in grouped_rows.items():
        rows_np = {
            "tokens": np.asarray(rows["tokens"], dtype=np.int64),
            "mask": np.asarray(rows["mask"], dtype=bool),
            "type_ids": np.asarray(rows["type_ids"], dtype=np.int64),
            "event_index": np.asarray(rows["event_index"], dtype=np.int64),
        }
        log.info("Decoding %s rows from %s", len(rows_np["event_index"]), source_file)
        process_group(
            source_file=source_file,
            rows=rows_np,
            h5_path=h5_files[source_file],
            collections=collections,
            names_by_object=names_by_object,
            vocabulary=vocabulary,
            models=models,
            preprocessors=preprocessors,
            objects=args.objects,
            max_objects=args.max_objects,
            device=device,
            jet_pt_threshold=args.jet_pt_threshold,
            storage=storage,
            event_index_mode=args.event_index_mode,
        )

    summary = {}
    for name, pairs in storage.items():
        if not pairs:
            continue
        values = np.asarray(pairs, dtype=np.float64)
        original = values[:, 0]
        reconstructed = values[:, 1]
        summary[name] = metrics(original, reconstructed)
        plot_observable(name, original, reconstructed, output_dir)

    (output_dir / "event_physics_reco_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "event_physics_reco_metrics.csv").open("w", newline="") as handle:
        fieldnames = ["observable", "n", "mean_original", "mean_reconstructed", "bias", "mae", "rmse", "p16", "p50", "p84"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, row in summary.items():
            writer.writerow({"observable": name, **row})

    log.info("Wrote event-token physics diagnostics to %s", output_dir)


if __name__ == "__main__":
    main()
