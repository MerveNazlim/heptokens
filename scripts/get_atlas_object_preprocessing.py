"""Fit preprocessing transformers for ATLAS event object tokenizers."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
from joblib import dump
from omegaconf import OmegaConf

from heptokens.data.transforms import create_preprocessing_transformer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit an object-level preprocessing transformer from ATLAS event H5 files. "
            "For jets, use --mode log_standard and log pt,mass,n_trk,QG_nTracks."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--h5-files", nargs="+", required=True, help="Input ATLAS event H5 files.")
    parser.add_argument(
        "--datamodule-config",
        type=Path,
        default=Path("configs/datamodule/atlas_event_object.yaml"),
        help="Datamodule config with object_collections.",
    )
    parser.add_argument("--object-type", default="jets", help="Object collection to fit.")
    parser.add_argument(
        "--fit-split", choices=["all", "train"], default="all",
        help="Use train to match the ordered input list's seeded tokenizer training split.",
    )
    parser.add_argument(
        "--mode",
        choices=["quantile", "log_quantile", "log_standard", "standard"],
        default="log_standard",
        help="Preprocessing mode.",
    )
    parser.add_argument(
        "--log-features",
        default="pt,mass,n_trk,QG_nTracks",
        help="Comma-separated feature names to log before the final transformer.",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=1_000_000,
        help="Maximum valid objects used to fit the transformer.",
    )
    parser.add_argument(
        "--num-events-per-file",
        type=int,
        help="Optional cap on events read from each H5 file.",
    )
    parser.add_argument(
        "--n-quantiles",
        type=int,
        default=500,
        help="Number of quantiles for quantile modes.",
    )
    parser.add_argument(
        "--log-offset",
        type=float,
        default=1.0,
        help="Offset used by log transform: log(x + offset).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for subsampling valid objects inside each file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("resources/atlas_object_preprocessing"),
        help="Directory for transformer and metadata.",
    )
    parser.add_argument(
        "--output-name",
        help="Output basename without extension. Defaults to <object>_<mode>.",
    )
    return parser.parse_args()


def find_collection(config: dict, object_type: str) -> dict:
    for collection in config.get("object_collections") or []:
        if collection.get("object_name") == object_type:
            return collection
    available = ", ".join(c.get("object_name", "<unnamed>") for c in config.get("object_collections") or [])
    raise ValueError(f"Object collection {object_type!r} is not configured. Available: {available}")


def feature_name(path: str) -> str:
    return Path(path).name


def training_event_masks(
    h5_files: list[str], input_path: str, config: dict,
    num_events_per_file: int | None, seed: int,
) -> list[np.ndarray]:
    """Select the same global training events as the eager/iterable object loaders."""
    from heptokens.data.atlas_event_iterable import _assign_global_event_splits

    train_frac = float(config.get("train_frac", 0.7))
    val_frac = float(config.get("val_frac", 0.15))
    test_frac = float(config.get("test_frac", 0.15))
    if min(train_frac, val_frac, test_frac) < 0 or not np.isclose(
        train_frac + val_frac + test_frac, 1.0
    ):
        raise ValueError("Tokenizer train/val/test fractions must be nonnegative and sum to 1")
    specs = []
    for path in h5_files:
        with h5py.File(path, "r") as handle:
            count = len(handle[input_path])
        if num_events_per_file is not None:
            count = min(count, num_events_per_file)
        specs.append((path, count, "unspecified"))
    return [spec.split_ids == 0 for spec in _assign_global_event_splits(
        specs, train_frac, val_frac, seed
    )]


def read_valid_objects(
    *,
    h5_files: list[str],
    input_paths: list[str],
    mask_path: str,
    max_objects: int,
    num_events_per_file: int | None,
    rng: np.random.Generator,
    event_selections: list[np.ndarray] | None = None,
) -> np.ndarray:
    chunks = []
    collected = 0

    for file_index, file_path in enumerate(h5_files):
        if collected >= max_objects:
            break

        log.info("Reading %s", file_path)
        with h5py.File(file_path, "r") as handle:
            missing = [path for path in [*input_paths, mask_path] if path not in handle]
            if missing:
                raise KeyError(f"{file_path} is missing required H5 paths: {missing}")

            n_events = len(handle[input_paths[0]])
            if num_events_per_file is not None:
                n_events = min(n_events, num_events_per_file)
            sl = slice(0, n_events)

            feature_arrays = [handle[path][sl].astype(np.float32) for path in input_paths]
            csts = np.stack(feature_arrays, axis=-1)
            csts = np.nan_to_num(csts, nan=0.0, posinf=0.0, neginf=0.0)
            csts = np.clip(csts, -1e4, 1e4)

            mask = handle[mask_path][sl].astype(bool)
            if event_selections is not None:
                selected = event_selections[file_index]
                if len(selected) != n_events:
                    raise ValueError("Preprocessing event selection does not match the H5 event cap")
                csts = csts[selected]
                mask = mask[selected]
            valid = csts[mask]

        if len(valid) == 0:
            continue

        remaining = max_objects - collected
        if len(valid) > remaining:
            indices = rng.choice(len(valid), size=remaining, replace=False)
            valid = valid[indices]

        chunks.append(valid)
        collected += len(valid)
        log.info("  collected %d / %d valid objects", collected, max_objects)

    if not chunks:
        raise RuntimeError("No valid objects found for preprocessing fit.")
    return np.concatenate(chunks, axis=0)


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.datamodule_config), resolve=True)
    collection = find_collection(cfg, args.object_type)
    input_paths = list(collection["inputs"])
    mask_path = collection["mask_input"]
    feature_names = [feature_name(path) for path in input_paths]

    requested_log_features = [name.strip() for name in args.log_features.split(",") if name.strip()]
    name_to_index = {name: idx for idx, name in enumerate(feature_names)}
    missing_log_features = [name for name in requested_log_features if name not in name_to_index]
    if args.mode.startswith("log") and missing_log_features:
        raise ValueError(
            f"Requested log features are not in {args.object_type} inputs: {missing_log_features}. "
            f"Available: {feature_names}"
        )
    log_indices = [name_to_index[name] for name in requested_log_features if name in name_to_index]

    log.info("Object type: %s", args.object_type)
    log.info("Features: %s", feature_names)
    log.info("Log features: %s", [feature_names[idx] for idx in log_indices])

    rng = np.random.default_rng(args.seed)
    event_selections = None
    event_cap = args.num_events_per_file
    if args.fit_split == "train":
        configured_cap = cfg.get("num_events")
        if configured_cap is not None:
            event_cap = int(configured_cap) if event_cap is None else min(
                event_cap, int(configured_cap)
            )
        event_selections = training_event_masks(
            list(args.h5_files), input_paths[0], cfg, event_cap, args.seed
        )
    objects = read_valid_objects(
        h5_files=list(args.h5_files),
        input_paths=input_paths,
        mask_path=mask_path,
        max_objects=args.max_objects,
        num_events_per_file=event_cap,
        rng=rng,
        event_selections=event_selections,
    )
    log.info("Fitting %s transformer on %d objects with %d features", args.mode, len(objects), objects.shape[1])

    transformer = create_preprocessing_transformer(
        mode=args.mode,
        log_feature_indices=log_indices if log_indices else None,
        n_quantiles=args.n_quantiles,
        log_offset=args.log_offset,
        n_features=len(feature_names),
    )
    transformer.fit(objects)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    basename = args.output_name or f"{args.object_type}_{args.mode}"
    transformer_path = args.output_dir / f"{basename}.joblib"
    metadata_path = args.output_dir / f"{basename}.json"

    dump(transformer, transformer_path)
    metadata = {
        "object_type": args.object_type,
        "mode": args.mode,
        "feature_paths": input_paths,
        "feature_names": feature_names,
        "log_features": [feature_names[idx] for idx in log_indices],
        "log_feature_indices": log_indices,
        "log_offset": args.log_offset,
        "n_objects_fit": int(len(objects)),
        "h5_files": list(args.h5_files),
        "fit_split": args.fit_split,
        "seed": args.seed,
        "num_events_per_file": event_cap,
        "train_frac": cfg.get("train_frac", 0.7) if args.fit_split == "train" else None,
        "val_frac": cfg.get("val_frac", 0.15) if args.fit_split == "train" else None,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    log.info("Saved transformer to %s", transformer_path)
    log.info("Saved metadata to %s", metadata_path)


if __name__ == "__main__":
    main()
