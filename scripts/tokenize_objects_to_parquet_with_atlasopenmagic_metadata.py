"""Apply one VQ-VAE checkpoint per object type and write event token parquet files.

Each object type and residual quantizer receives its own token-ID range, so
large codebooks can be exported without merging distinct codes.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from joblib import load as joblib_load
from omegaconf import OmegaConf

from heptokens.models.vq_vae import LitVqVae

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

TYPE_IDS = {
    "electrons": 4,
    "electron": 4,
    "muons": 5,
    "muon": 5,
    "jets": 6,
    "jet": 6,
    "met": 7,
    "photons": 8,
    "photon": 8,
    "tracks": 9,
    "track": 9,
    "event": 10,
    "taus": 11,
    "tau": 11,
}

DEFAULT_EVENT_TOKEN_INPUTS = [
    "common/event/mu",
    "common/met/pt",
    "common/met/phi",
    "common/met/sumet",
]

DEFAULT_EVENT_TOKEN_RANGES = {
    "common/event/mu": (0.0, 100.0),
    "common/met/pt": (0.0, 500.0),
    "common/met/phi": (-math.pi, math.pi),
    "common/met/sumet": (0.0, 5000.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build old-style event token parquet from per-object VQ-VAE checkpoints."
    )
    parser.add_argument("--h5-files", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--datamodule-config",
        default="configs/datamodule/atlas_event_object.yaml",
        help="Config containing the same object feature paths used to train the checkpoints.",
    )
    parser.add_argument(
        "--tokenizer-checkpoints",
        nargs="+",
        required=True,
        help="Object checkpoint map, e.g. jets=/path/best.ckpt electrons=/path/best.ckpt.",
    )
    parser.add_argument(
        "--preprocess-transformers",
        nargs="+",
        default=[],
        help=(
            "Optional object preprocessing map, e.g. "
            "jets=/path/jets_log_standard.joblib electrons=/path/electrons_log_standard.joblib. "
            "Use this when checkpoints were trained with log-standard/standard preprocessing."
        ),
    )
    parser.add_argument(
        "--object-order",
        nargs="+",
        default=["electrons", "muons", "taus", "photons", "jets", "tracks", "met"],
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-events", type=int)
    parser.add_argument("--num-objects", type=int)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--pad-token-id", type=int, default=0)
    parser.add_argument("--cls-token-id", type=int, default=1)
    parser.add_argument("--sep-token-id", type=int, default=2)
    parser.add_argument("--mask-token-id", type=int, default=3)
    parser.add_argument("--event-bins", type=int, default=100)
    parser.add_argument(
        "--event-token-input",
        help=(
            "Backward-compatible single HDF5 feature used for the event-context token. "
            "Use --event-token-inputs for multiple event context tokens."
        ),
    )
    parser.add_argument(
        "--event-token-inputs",
        nargs="+",
        help=(
            "HDF5 features to bin into explicit event-context tokens. Defaults to "
            "common/event/mu plus common/met/{pt,phi,sumet}."
        ),
    )
    parser.add_argument(
        "--event-token-ranges",
        nargs="*",
        default=[],
        help=(
            "Optional binning ranges as path=min,max. Defaults are "
            "common/event/mu=0,100 common/met/pt=0,500 "
            "common/met/phi=-pi,pi common/met/sumet=0,5000."
        ),
    )
    parser.add_argument("--no-cls", action="store_true")
    parser.add_argument("--no-separators", action="store_true")
    parser.add_argument("--no-event-token", action="store_true")
    parser.add_argument("--write-legacy-columns", action="store_true")
    parser.add_argument(
        "--write-continuous-features",
        action="store_true",
        help=(
            "Write preprocessed continuous features aligned with grouped sequence "
            "positions for flat and hierarchical continuous baselines."
        ),
    )
    parser.add_argument(
        "--write-decoded-q8-features",
        action="store_true",
        help=(
            "Also decode each complete residual-code tuple with its object VQ-VAE "
            "and write decoded_continuous_features. Requires "
            "--write-continuous-features."
        ),
    )
    parser.add_argument(
        "--atlasopenmagic-release",
        default="2024r-pp",
        help=(
            "ATLAS Open Magic release used to fill DSID metadata. "
            "Set to an empty string to skip set_release()."
        ),
    )
    parser.add_argument(
        "--metadata-source",
        choices=["auto", "h5", "atlasopenmagic"],
        default="auto",
        help=(
            "Where sample metadata columns come from. auto uses H5 attrs first, "
            "then fills missing or zero cross sections from atlasopenmagic."
        ),
    )
    return parser.parse_args()


def choose_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def parse_checkpoint_map(items: list[str]) -> dict[str, str]:
    checkpoint_map = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                "Each tokenizer checkpoint must look like object=/path/checkpoint.ckpt"
            )
        key, value = item.split("=", 1)
        checkpoint_map[key.strip()] = value.strip()
    return checkpoint_map


def parse_path_map(items: list[str], *, item_name: str) -> dict[str, str]:
    path_map = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Each {item_name} must look like object=/path/file")
        key, value = item.split("=", 1)
        path_map[key.strip()] = value.strip()
    return path_map


def parse_event_range_map(items: list[str]) -> dict[str, tuple[float, float]]:
    range_map = {}
    for item in items:
        if "=" not in item or "," not in item:
            raise ValueError(
                "Each event token range must look like common/met/pt=0,500"
            )
        key, value = item.split("=", 1)
        low_text, high_text = value.split(",", 1)
        low = float(low_text.strip())
        high = float(high_text.strip())
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError(f"Invalid event token range for {key}: {value}")
        range_map[key.strip()] = (low, high)
    return range_map


def load_data_config(path: str) -> dict:
    cfg = OmegaConf.load(path)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    return {key: value for key, value in cfg.items() if key not in DATAMODULE_KEYS}


def collection_map(object_collections: list[dict]) -> dict[str, dict]:
    return {collection["object_name"]: collection for collection in object_collections}


def resolve_dataset(handle: h5py.File, h5_path: str):
    parts = h5_path.strip("/").split("/")
    node = handle
    for index, part in enumerate(parts):
        if isinstance(node, h5py.Dataset):
            field = "/".join(parts[index:])
            return node, field
        node = node[part]
    return node, None


def read_h5_path(
    handle: h5py.File,
    h5_path: str,
    event_slice: slice,
    num_objects: int | None = None,
) -> np.ndarray:
    dataset, field = resolve_dataset(handle, h5_path)
    if field is not None:
        array = dataset[field][event_slice]
    else:
        array = dataset[event_slice]
    if array.ndim == 2 and num_objects is not None:
        array = array[:, :num_objects]
    return array


def infer_n_events(handle: h5py.File, config: dict) -> int:
    if "metadata" in handle and "n_events" in handle["metadata"].attrs:
        return int(handle["metadata"].attrs["n_events"])
    if "n_events" in handle.attrs:
        return int(handle.attrs["n_events"])
    object_collections = config.get("object_collections") or []
    if object_collections:
        dataset, _ = resolve_dataset(handle, object_collections[0]["inputs"][0])
        return int(dataset.shape[0])
    event_inputs = config.get("event_inputs") or []
    if event_inputs:
        dataset, _ = resolve_dataset(handle, event_inputs[0])
        return int(dataset.shape[0])
    raise ValueError("Cannot infer number of events")


def read_collection_batch(
    handle: h5py.File,
    collection: dict,
    event_slice: slice,
    *,
    num_objects: int | None,
    global_mask_input: str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    feature_arrays = []
    n_objects = None
    for path in collection["inputs"]:
        array = read_h5_path(handle, path, event_slice)
        if array.ndim == 1:
            array = array[:, None]
        if n_objects is None:
            total_objects = array.shape[1]
            n_objects = min(total_objects, num_objects) if num_objects else total_objects
        feature_arrays.append(array[:, :n_objects].astype(np.float32))

    csts = np.stack(feature_arrays, axis=-1)
    csts = np.nan_to_num(csts, nan=0.0, posinf=0.0, neginf=0.0)
    csts = np.clip(csts, -1e4, 1e4)

    mask_input = collection.get("mask_input", global_mask_input)
    if mask_input is None:
        mask = np.ones(csts.shape[:2], dtype=bool)
    else:
        mask = read_h5_path(handle, mask_input, event_slice, n_objects).astype(bool)
        if mask.ndim == 1:
            mask = mask[:, None]
    return torch.from_numpy(csts).float(), torch.from_numpy(mask).bool()


def read_event_token_values(
    handle: h5py.File,
    event_token_specs: list[dict],
    event_slice: slice,
) -> np.ndarray | None:
    if not event_token_specs:
        return None

    tokens = []
    for spec in event_token_specs:
        values = read_h5_path(handle, spec["input"], event_slice).astype(np.float32)
        values = np.nan_to_num(values, nan=spec["range"][0], posinf=spec["range"][1], neginf=spec["range"][0])
        low, high = spec["range"]
        scaled = np.floor((values - low) / (high - low) * spec["size"])
        bins = np.clip(scaled, 0, spec["size"] - 1).astype(np.int64)
        tokens.append(spec["base"] + bins)
    return np.stack(tokens, axis=1)


def read_normalized_event_values(
    handle: h5py.File,
    event_token_specs: list[dict],
    event_slice: slice,
) -> np.ndarray | None:
    """Read event scalars and scale them with the token vocabulary ranges."""
    if not event_token_specs:
        return None

    normalized = []
    for spec in event_token_specs:
        low, high = (float(value) for value in spec["range"])
        if high <= low:
            raise ValueError(f"Invalid event range for {spec['input']}: {[low, high]}")
        values = read_h5_path(handle, spec["input"], event_slice).astype(np.float32)
        values = np.nan_to_num(values, nan=low, posinf=high, neginf=low)
        normalized.append(np.clip((values - low) / (high - low), 0.0, 1.0))
    return np.stack(normalized, axis=1).astype(np.float32)


def python_scalar(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    return value


def as_float(value, default: float = math.nan) -> float:
    value = python_scalar(value)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default: int = 0) -> int:
    value = python_scalar(value)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_str(value, default: str = "") -> str:
    value = python_scalar(value)
    if value is None:
        return default
    return str(value)


def h5_sample_metadata(handle: h5py.File, source_file: str) -> dict:
    attrs = handle["metadata"].attrs if "metadata" in handle else {}
    root_attrs = handle.attrs
    dsid = as_int(attrs.get("dsid"), 0)
    if dsid == 0 and "atlas/event/mcChannelNumber" in handle:
        mc = np.asarray(handle["atlas/event/mcChannelNumber"][:])
        unique = np.unique(mc[mc > 0])
        if len(unique) == 1:
            dsid = int(unique[0])

    return {
        "dsid": dsid,
        "process": as_str(attrs.get("process")),
        "generator": as_str(attrs.get("generator")),
        "cross_section_pb": as_float(attrs.get("cross_section_pb")),
        "filter_efficiency": as_float(attrs.get("filter_efficiency"), 1.0),
        "k_factor": as_float(attrs.get("k_factor"), 1.0),
        "sqrt_s_tev": as_float(attrs.get("sqrt_s_tev")),
        "is_mc": bool(python_scalar(attrs.get("is_mc", dsid > 0))),
        "sample_label": as_str(root_attrs.get("sample_label")),
        "data_taking_year": as_str(attrs.get("data_taking_year")),
        "mc_campaign": as_str(attrs.get("mc_campaign")),
        "source_file_events": as_int(attrs.get("n_events"), 0),
        "source_file": source_file,
    }


def lookup_atlasopenmagic_metadata(dsid: int, release: str, *, required: bool = False) -> dict:
    if dsid <= 0:
        return {}
    try:
        import atlasopenmagic as atom
    except ImportError:
        if required:
            raise RuntimeError(
                "atlasopenmagic is required because --metadata-source=atlasopenmagic, "
                "but it is not installed. Install it with: pixi add --pypi atlasopenmagic"
            )
        log.warning("atlasopenmagic is not installed; using H5 metadata only")
        return {}

    if release:
        try:
            atom.set_release(release)
        except Exception as exc:
            log.warning("atlasopenmagic set_release(%s) failed: %s", release, exc)

    dsid_str = str(dsid)
    try:
        metadata = atom.get_metadata(dsid_str)
    except Exception as exc:
        log.debug("atlasopenmagic full metadata lookup failed for DSID %s: %s", dsid, exc)
        metadata = {}

    def get_field(name: str, default=None):
        if isinstance(metadata, dict) and name in metadata:
            return metadata[name]
        try:
            value = atom.get_metadata(dsid_str, name)
        except Exception:
            return default
        return default if value is None else value

    cross_section = get_field("cross_section_pb")
    if cross_section in (None, ""):
        cross_section = get_field("cross_section")
    filter_efficiency = get_field("filter_efficiency")
    if filter_efficiency in (None, ""):
        filter_efficiency = get_field("genFiltEff")
    k_factor = get_field("k_factor")
    if k_factor in (None, ""):
        k_factor = get_field("kFactor")

    result = {
        "process": as_str(
            get_field("process")
            or get_field("physics_short")
            or get_field("dataset_name")
        ),
        "generator": as_str(get_field("generator") or get_field("generators")),
        "cross_section_pb": as_float(cross_section),
        "filter_efficiency": as_float(filter_efficiency, 1.0),
        "k_factor": as_float(k_factor, 1.0),
        "keywords": as_str(get_field("keywords")),
    }
    if (
        not result["process"]
        and not result["generator"]
        and not np.isfinite(result["cross_section_pb"])
    ):
        log.warning("atlasopenmagic returned no usable metadata for DSID %s", dsid)
        return {}
    return result


def should_replace_number(current: float, new: float) -> bool:
    return np.isfinite(new) and (not np.isfinite(current) or current == 0.0)


def sample_metadata_from_handle(
    handle: h5py.File,
    source_file: str,
    *,
    metadata_source: str,
    atlasopenmagic_release: str,
) -> dict:
    h5_metadata = h5_sample_metadata(handle, source_file)
    if metadata_source == "h5":
        return h5_metadata

    magic_metadata = lookup_atlasopenmagic_metadata(
        h5_metadata["dsid"],
        atlasopenmagic_release,
        required=metadata_source == "atlasopenmagic",
    )
    if metadata_source == "atlasopenmagic":
        merged = {**h5_metadata, **{k: v for k, v in magic_metadata.items() if v not in ("", None)}}
        return merged

    merged = dict(h5_metadata)
    for key in ("cross_section_pb", "filter_efficiency", "k_factor"):
        if should_replace_number(merged.get(key, math.nan), magic_metadata.get(key, math.nan)):
            merged[key] = magic_metadata[key]
    for key in ("process", "generator", "keywords"):
        if not merged.get(key) and magic_metadata.get(key):
            merged[key] = magic_metadata[key]
    return merged


def encode_object_batch(
    model: LitVqVae,
    csts: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    *,
    return_decoded: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    batch = {
        "csts": csts.to(device),
        "mask": mask.to(device),
        "jets": mask.sum(dim=1, keepdim=True).float().to(device),
    }
    if return_decoded:
        z_q, indices, _ = model.encode(batch)
        decoded = model.decode(z_q, batch).detach().cpu().float()
        if decoded.shape != csts.shape:
            raise ValueError(
                "Decoded VQ-VAE features do not match the input feature shape: "
                f"{tuple(decoded.shape)} != {tuple(csts.shape)}"
            )
        if not torch.isfinite(decoded[mask]).all():
            raise ValueError("VQ-VAE decoder produced non-finite object features")
    else:
        indices = model(batch)
    indices = indices.detach().cpu().long()
    if indices.dim() == 2:
        indices = indices.unsqueeze(-1)
    if return_decoded:
        return indices, decoded
    return indices


def apply_object_preprocessing(
    csts: torch.Tensor,
    mask: torch.Tensor,
    transformer,
    object_name: str,
) -> torch.Tensor:
    if transformer is None:
        return csts

    csts_np = csts.detach().cpu().numpy().copy()
    mask_np = mask.detach().cpu().numpy()
    if mask_np.any():
        expected_features = getattr(transformer, "n_features_in_", None)
        if expected_features is None:
            final_transformer = getattr(transformer, "final_transformer", None)
            expected_features = getattr(final_transformer, "n_features_in_", None)
        actual_features = csts_np.shape[-1]
        if expected_features is not None and int(expected_features) != actual_features:
            raise ValueError(
                f"{object_name} preprocessing expects {expected_features} features, "
                f"but the datamodule config provides {actual_features}. "
                "Use the matching tokenizer checkpoint, preprocessor, and datamodule config."
            )
        csts_np[mask_np] = transformer.transform(csts_np[mask_np])
    return torch.from_numpy(csts_np).float()


def build_vocabulary(
    models: dict[str, LitVqVae],
    event_token_inputs: list[str],
    event_range_map: dict[str, tuple[float, float]],
    args: argparse.Namespace,
) -> dict:
    """Assign a unique token range to every object quantizer and code."""
    special_tokens = {
        "pad": args.pad_token_id,
        "cls": args.cls_token_id,
        "sep": args.sep_token_id,
        "mask": args.mask_token_id,
    }
    if len(set(special_tokens.values())) != len(special_tokens):
        raise ValueError(f"Special token IDs must be unique: {special_tokens}")
    if args.event_bins < 1:
        raise ValueError("--event-bins must be at least 1")

    next_token = max(special_tokens.values()) + 1
    event_token_specs = []
    for event_input in event_token_inputs:
        value_range = event_range_map.get(
            event_input,
            DEFAULT_EVENT_TOKEN_RANGES.get(event_input, (0.0, float(args.event_bins))),
        )
        event_token_specs.append(
            {
                "base": next_token,
                "size": args.event_bins,
                "type_id": TYPE_IDS["event"],
                "input": event_input,
                "range": [float(value_range[0]), float(value_range[1])],
            }
        )
        next_token += args.event_bins

    event_vocab = {
        "type_id": TYPE_IDS["event"],
        "inputs": list(event_token_inputs),
        "tokens": event_token_specs,
    }
    if len(event_token_specs) == 1:
        event_vocab.update(event_token_specs[0])

    vocabulary = {
        "version": 1,
        "special_tokens": special_tokens,
        "event": event_vocab,
        "event_tokens": event_token_specs,
        "object_order": list(args.object_order),
        "objects": {},
    }

    for object_name in args.object_order:
        model = models.get(object_name)
        if model is None:
            continue
        if object_name not in TYPE_IDS:
            raise ValueError(f"No type ID is configured for object {object_name!r}")
        codebook_size = int(model.hparams.codebook_size)
        num_quantizers = int(model.hparams.num_quantizers)
        quantizers = []
        for quantizer_idx in range(num_quantizers):
            quantizers.append(
                {
                    "index": quantizer_idx,
                    "base": next_token,
                    "size": codebook_size,
                }
            )
            next_token += codebook_size
        vocabulary["objects"][object_name] = {
            "type_id": TYPE_IDS[object_name],
            "codebook_size": codebook_size,
            "num_quantizers": num_quantizers,
            "quantizers": quantizers,
        }

    vocabulary["vocab_size"] = next_token
    return vocabulary


def add_token(
    row_tokens: list[int],
    row_types: list[int],
    row_mask: list[bool],
    token: int,
    type_id: int,
) -> None:
    row_tokens.append(int(token))
    row_types.append(int(type_id))
    row_mask.append(True)


def assemble_rows(
    encoded_objects: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    event_tokens: np.ndarray | None,
    batch_size: int,
    vocabulary: dict,
    args: argparse.Namespace,
    event_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_tokens = np.full((batch_size, args.max_seq_length), args.pad_token_id, dtype=np.int64)
    all_types = np.zeros((batch_size, args.max_seq_length), dtype=np.int64)
    all_mask = np.zeros((batch_size, args.max_seq_length), dtype=bool)

    for event_idx in range(batch_size):
        row_tokens: list[int] = []
        row_types: list[int] = []
        row_mask: list[bool] = []

        if not args.no_cls:
            add_token(row_tokens, row_types, row_mask, args.cls_token_id, 0)

        if event_tokens is not None and not args.no_event_token:
            for token in np.atleast_1d(event_tokens[event_idx]):
                add_token(
                    row_tokens,
                    row_types,
                    row_mask,
                    int(token),
                    TYPE_IDS["event"],
                )

        for object_name in args.object_order:
            if object_name not in encoded_objects:
                continue
            indices, object_mask = encoded_objects[object_name][:2]
            type_id = TYPE_IDS[object_name]
            object_vocab = vocabulary["objects"][object_name]

            event_indices = indices[event_idx]
            event_mask = object_mask[event_idx]
            expected_quantizers = object_vocab["num_quantizers"]
            if event_indices.shape[-1] != expected_quantizers:
                raise ValueError(
                    f"{object_name} checkpoint reports {expected_quantizers} quantizers, "
                    f"but encoding produced {event_indices.shape[-1]} indices"
                )
            for object_idx in torch.nonzero(event_mask, as_tuple=False).flatten().tolist():
                for quantizer_idx in range(event_indices.shape[-1]):
                    code = int(event_indices[object_idx, quantizer_idx])
                    if code < 0:
                        continue
                    quantizer_vocab = object_vocab["quantizers"][quantizer_idx]
                    if code >= quantizer_vocab["size"]:
                        raise ValueError(
                            f"{object_name} q{quantizer_idx} produced code {code}, "
                            f"but its codebook size is {quantizer_vocab['size']}"
                        )
                    add_token(
                        row_tokens,
                        row_types,
                        row_mask,
                        quantizer_vocab["base"] + code,
                        type_id,
                    )

            if not args.no_separators:
                add_token(row_tokens, row_types, row_mask, args.sep_token_id, 0)

        length = min(len(row_tokens), args.max_seq_length)
        all_tokens[event_idx, :length] = row_tokens[:length]
        all_types[event_idx, :length] = row_types[:length]
        all_mask[event_idx, :length] = row_mask[:length]

    return all_tokens, all_mask, all_types


def make_table(
    tokens: np.ndarray,
    mask: np.ndarray,
    type_ids: np.ndarray,
    *,
    start_index: int,
    source_file: str,
    sample_metadata: dict,
    write_legacy_columns: bool,
    vocabulary: dict,
    extra_columns: dict[str, np.ndarray] | None = None,
) -> pa.Table:
    seq_len = tokens.shape[1]
    n_rows = len(tokens)
    cross_section_pb = as_float(sample_metadata.get("cross_section_pb"))
    filter_efficiency = as_float(sample_metadata.get("filter_efficiency"), 1.0)
    k_factor = as_float(sample_metadata.get("k_factor"), 1.0)
    effective_cross_section_pb = (
        cross_section_pb * filter_efficiency * k_factor
        if np.isfinite(cross_section_pb)
        else math.nan
    )
    fields = {
        "tokens": pa.array(tokens.tolist(), type=pa.list_(pa.int64(), seq_len)),
        "mask": pa.array(mask.tolist(), type=pa.list_(pa.bool_(), seq_len)),
        "type_ids": pa.array(type_ids.tolist(), type=pa.list_(pa.int64(), seq_len)),
        "event_index": pa.array(
            np.arange(start_index, start_index + len(tokens), dtype=np.int64)
        ),
        "source_file": pa.array([source_file] * len(tokens)),
        "dsid": pa.array([as_int(sample_metadata.get("dsid"))] * n_rows, type=pa.int64()),
        "process": pa.array([as_str(sample_metadata.get("process"))] * n_rows, type=pa.string()),
        "generator": pa.array(
            [as_str(sample_metadata.get("generator"))] * n_rows,
            type=pa.string(),
        ),
        "cross_section_pb": pa.array([cross_section_pb] * n_rows, type=pa.float64()),
        "filter_efficiency": pa.array([filter_efficiency] * n_rows, type=pa.float64()),
        "k_factor": pa.array([k_factor] * n_rows, type=pa.float64()),
        "effective_cross_section_pb": pa.array(
            [effective_cross_section_pb] * n_rows,
            type=pa.float64(),
        ),
        "sqrt_s_tev": pa.array(
            [as_float(sample_metadata.get("sqrt_s_tev"))] * n_rows,
            type=pa.float64(),
        ),
        "is_mc": pa.array([bool(sample_metadata.get("is_mc", False))] * n_rows, type=pa.bool_()),
        "sample_label": pa.array(
            [as_str(sample_metadata.get("sample_label"))] * n_rows,
            type=pa.string(),
        ),
        "data_taking_year": pa.array(
            [as_str(sample_metadata.get("data_taking_year"))] * n_rows,
            type=pa.string(),
        ),
        "mc_campaign": pa.array(
            [as_str(sample_metadata.get("mc_campaign"))] * n_rows,
            type=pa.string(),
        ),
        "source_file_events": pa.array(
            [as_int(sample_metadata.get("source_file_events"))] * n_rows,
            type=pa.int64(),
        ),
    }
    if write_legacy_columns:
        fields["input_ids"] = fields["tokens"]
        fields["attention_mask"] = fields["mask"]
        fields["token_type_ids"] = fields["type_ids"]
    if extra_columns:
        continuous = extra_columns.get("continuous_features")
        decoded_continuous = extra_columns.get("decoded_continuous_features")
        feature_mask = extra_columns.get("continuous_feature_mask")
        role_ids = extra_columns.get("position_role_ids")
        if continuous is not None:
            fields["continuous_features"] = pa.array(
                continuous.tolist(),
                type=pa.list_(
                    pa.list_(pa.float32(), continuous.shape[2]),
                    continuous.shape[1],
                ),
            )
        if decoded_continuous is not None:
            fields["decoded_continuous_features"] = pa.array(
                decoded_continuous.tolist(),
                type=pa.list_(
                    pa.list_(pa.float32(), decoded_continuous.shape[2]),
                    decoded_continuous.shape[1],
                ),
            )
        if feature_mask is not None:
            fields["continuous_feature_mask"] = pa.array(
                feature_mask.tolist(),
                type=pa.list_(
                    pa.list_(pa.bool_(), feature_mask.shape[2]),
                    feature_mask.shape[1],
                ),
            )
        if role_ids is not None:
            fields["position_role_ids"] = pa.array(
                role_ids.tolist(),
                type=pa.list_(pa.int64(), role_ids.shape[1]),
            )
    table = pa.table(fields)
    metadata = dict(table.schema.metadata or {})
    metadata[b"heptokens_token_vocabulary"] = json.dumps(vocabulary, sort_keys=True).encode()
    continuous_schema = vocabulary.get("continuous_schema")
    if continuous_schema is not None:
        metadata[b"heptokens_continuous_schema"] = json.dumps(
            continuous_schema, sort_keys=True
        ).encode()
    metadata[b"heptokens_sample_metadata_columns"] = json.dumps(
        [
            "dsid",
            "process",
            "generator",
            "cross_section_pb",
            "filter_efficiency",
            "k_factor",
            "effective_cross_section_pb",
            "sqrt_s_tev",
            "is_mc",
            "sample_label",
            "data_taking_year",
            "mc_campaign",
            "source_file_events",
        ]
    ).encode()
    return table.replace_schema_metadata(metadata)


def load_models(checkpoint_map: dict[str, str], device: torch.device) -> dict[str, LitVqVae]:
    models = {}
    for object_name, checkpoint in checkpoint_map.items():
        log.info("Loading %s tokenizer from %s", object_name, checkpoint)
        model = LitVqVae.load_from_checkpoint(checkpoint, map_location=device)
        model.to(device)
        model.eval()
        models[object_name] = model
    return models


def load_preprocessors(preprocess_map: dict[str, str]) -> dict[str, object]:
    preprocessors = {}
    for object_name, transformer_path in preprocess_map.items():
        path = Path(transformer_path)
        if not path.exists():
            raise FileNotFoundError(path)
        log.info("Loading %s preprocessing from %s", object_name, path)
        preprocessors[object_name] = joblib_load(path)
    return preprocessors


def validate_export_inputs(
    models: dict[str, LitVqVae],
    collections: dict[str, dict],
    object_order: list[str],
    preprocessors: dict[str, object] | None = None,
) -> None:
    if not models:
        raise ValueError("At least one tokenizer checkpoint is required")
    ordered_objects = set(object_order)
    if len(ordered_objects) != len(object_order):
        raise ValueError("--object-order must not contain duplicate object names")
    unknown_models = sorted(set(models) - ordered_objects)
    if unknown_models:
        raise ValueError(
            "Tokenizer checkpoints were provided for objects not present in --object-order: "
            + ", ".join(unknown_models)
        )
    missing_collections = sorted(set(models) - set(collections))
    if missing_collections:
        raise ValueError(
            "Tokenizer checkpoints were provided for objects missing from the datamodule config: "
            + ", ".join(missing_collections)
        )
    if preprocessors:
        unknown_preprocessors = sorted(set(preprocessors) - ordered_objects)
        if unknown_preprocessors:
            raise ValueError(
                "Preprocessing transformers were provided for objects not present in --object-order: "
                + ", ".join(unknown_preprocessors)
            )
        missing_model = sorted(set(preprocessors) - set(models))
        if missing_model:
            raise ValueError(
                "Preprocessing transformers were provided without matching checkpoints: "
                + ", ".join(missing_model)
            )


def choose_event_token_inputs(config: dict, args: argparse.Namespace) -> list[str]:
    if args.no_event_token:
        return []

    configured_event_inputs = set(config.get("event_inputs") or [])
    if args.event_token_inputs:
        return list(args.event_token_inputs)
    if args.event_token_input is not None:
        return [args.event_token_input]

    default_inputs = list(DEFAULT_EVENT_TOKEN_INPUTS)
    missing_from_config = sorted(set(default_inputs) - configured_event_inputs)
    if missing_from_config:
        log.info(
            "Using default event context tokens %s. Some are not listed in event_inputs "
            "but will be read directly from the H5 file: %s",
            default_inputs,
            missing_from_config,
        )
    return default_inputs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    config = load_data_config(args.datamodule_config)
    collections = collection_map(config.get("object_collections") or [])
    checkpoint_map = parse_checkpoint_map(args.tokenizer_checkpoints)
    preprocess_map = parse_path_map(args.preprocess_transformers, item_name="preprocess transformer")
    device = choose_device(args.device)
    models = load_models(checkpoint_map, device)
    preprocessors = load_preprocessors(preprocess_map)
    validate_export_inputs(models, collections, args.object_order, preprocessors)
    event_token_inputs = choose_event_token_inputs(config, args)
    event_range_map = parse_event_range_map(args.event_token_ranges)
    vocabulary = build_vocabulary(models, event_token_inputs, event_range_map, args)
    if args.write_continuous_features:
        from heptokens.data.continuous_schema import build_continuous_schema

        vocabulary["continuous_schema"] = build_continuous_schema(
            collections=collections,
            object_order=list(vocabulary["objects"]),
            event_token_specs=vocabulary["event_tokens"],
            type_ids=TYPE_IDS,
            include_decoded_q8=args.write_decoded_q8_features,
        )
    elif args.write_decoded_q8_features:
        raise ValueError(
            "--write-decoded-q8-features requires --write-continuous-features"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vocabulary_path = output_path.with_suffix(".vocab.json")
    vocabulary_path.write_text(json.dumps(vocabulary, indent=2) + "\n")
    log.info("Token vocabulary size: %s", vocabulary["vocab_size"])
    log.info("Wrote token vocabulary to %s", vocabulary_path)

    writer = None
    n_written = 0
    with torch.no_grad():
        for h5_path in args.h5_files:
            h5_path = str(h5_path)
            log.info("Tokenizing %s", h5_path)
            with h5py.File(h5_path, mode="r") as handle:
                source_file = Path(h5_path).name
                sample_metadata = sample_metadata_from_handle(
                    handle,
                    source_file,
                    metadata_source=args.metadata_source,
                    atlasopenmagic_release=args.atlasopenmagic_release,
                )
                log.info(
                    "Sample metadata for %s: DSID=%s xsec=%s pb filter_eff=%s k_factor=%s",
                    source_file,
                    sample_metadata.get("dsid"),
                    sample_metadata.get("cross_section_pb"),
                    sample_metadata.get("filter_efficiency"),
                    sample_metadata.get("k_factor"),
                )
                total_events = infer_n_events(handle, config)
                if args.num_events is not None:
                    total_events = min(total_events, args.num_events)

                for start in range(0, total_events, args.batch_size):
                    end = min(start + args.batch_size, total_events)
                    event_slice = slice(start, end)
                    batch_events = end - start
                    encoded_objects = {}

                    for object_name in args.object_order:
                        if object_name not in models or object_name not in collections:
                            continue
                        csts, object_mask = read_collection_batch(
                            handle,
                            collections[object_name],
                            event_slice,
                            num_objects=args.num_objects,
                            global_mask_input=config.get("mask_input"),
                        )
                        csts = apply_object_preprocessing(
                            csts,
                            object_mask,
                            preprocessors.get(object_name),
                            object_name,
                        )
                        try:
                            encoded = encode_object_batch(
                                models[object_name],
                                csts,
                                object_mask,
                                device,
                                return_decoded=args.write_decoded_q8_features,
                            )
                        except Exception as exc:
                            valid_objects = int(object_mask.sum().item())
                            raise RuntimeError(
                                f"Failed to encode {object_name} in {source_file} "
                                f"for events {start}:{end}; "
                                f"csts_shape={tuple(csts.shape)} "
                                f"valid_objects={valid_objects}"
                            ) from exc
                        if args.write_decoded_q8_features:
                            indices, decoded = encoded
                            encoded_objects[object_name] = (
                                indices,
                                object_mask,
                                csts,
                                decoded,
                            )
                        else:
                            encoded_objects[object_name] = (
                                encoded,
                                object_mask,
                                csts,
                            )

                    event_tokens = read_event_token_values(
                        handle,
                        vocabulary["event_tokens"],
                        event_slice,
                    )
                    event_values = (
                        read_normalized_event_values(
                            handle,
                            vocabulary["event_tokens"],
                            event_slice,
                        )
                        if args.write_continuous_features
                        else None
                    )
                    assembled = assemble_rows(
                        encoded_objects,
                        event_tokens,
                        batch_events,
                        vocabulary,
                        args,
                        event_values=event_values,
                    )
                    if len(assembled) == 3:
                        tokens, mask, type_ids = assembled
                        extra_columns = None
                    else:
                        tokens, mask, type_ids, extra_columns = assembled
                    table = make_table(
                        tokens,
                        mask,
                        type_ids,
                        start_index=n_written,
                        source_file=source_file,
                        sample_metadata=sample_metadata,
                        write_legacy_columns=args.write_legacy_columns,
                        vocabulary=vocabulary,
                        extra_columns=extra_columns,
                    )
                    if writer is None:
                        writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
                    writer.write_table(table)
                    n_written += len(tokens)
                    if n_written == len(tokens) or n_written % (10 * args.batch_size) == 0:
                        log.info("Encoded %s events", n_written)

    if writer is not None:
        writer.close()

    size_mb = output_path.stat().st_size / (1024 * 1024)
    log.info("Wrote %s events to %s (%.1f MB)", n_written, output_path, size_mb)


if __name__ == "__main__":
    main()
