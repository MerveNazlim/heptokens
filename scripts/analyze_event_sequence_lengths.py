"""Measure event-token sequence lengths before writing Parquet files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from omegaconf import OmegaConf


DEFAULT_OBJECT_ORDER = ["electrons", "muons", "taus", "photons", "jets", "tracks", "met"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure sequence truncation for per-object residual-VQ tokens."
    )
    parser.add_argument("--h5-files", nargs="+", required=True)
    parser.add_argument(
        "--datamodule-config",
        default="configs/datamodule/atlas_event_object.yaml",
    )
    parser.add_argument(
        "--object-quantizers",
        nargs="+",
        required=True,
        help="Quantizers per object type, e.g. jets=4 electrons=4.",
    )
    parser.add_argument("--object-order", nargs="+", default=DEFAULT_OBJECT_ORDER)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--num-events-per-file", type=int)
    parser.add_argument("--batch-size", type=int, default=100000)
    parser.add_argument("--no-cls", action="store_true")
    parser.add_argument("--no-event-token", action="store_true")
    parser.add_argument("--no-separators", action="store_true")
    parser.add_argument("--output-json")
    return parser.parse_args()


def parse_map(values: list[str]) -> dict[str, int]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected object=quantizers, got {value!r}")
        name, count = value.split("=", 1)
        result[name] = int(count)
    return result


def resolve_dataset(handle: h5py.File, h5_path: str):
    parts = h5_path.strip("/").split("/")
    node = handle
    for index, part in enumerate(parts):
        if isinstance(node, h5py.Dataset):
            return node, "/".join(parts[index:])
        node = node[part]
    return node, None


def read_path(handle: h5py.File, path: str, event_slice: slice) -> np.ndarray:
    dataset, field = resolve_dataset(handle, path)
    if field is None:
        return dataset[event_slice]
    return dataset[field][event_slice]


def infer_n_events(handle: h5py.File, collections: dict[str, dict]) -> int:
    if "metadata" in handle and "n_events" in handle["metadata"].attrs:
        return int(handle["metadata"].attrs["n_events"])
    if "n_events" in handle.attrs:
        return int(handle.attrs["n_events"])
    first_collection = next(iter(collections.values()))
    dataset, _ = resolve_dataset(handle, first_collection["inputs"][0])
    return int(dataset.shape[0])


def read_object_counts(
    handle: h5py.File,
    collection: dict,
    event_slice: slice,
    global_mask_input: str | None,
) -> np.ndarray:
    mask_input = collection.get("mask_input", global_mask_input)
    if mask_input is not None:
        mask = read_path(handle, mask_input, event_slice).astype(bool)
        if mask.ndim == 1:
            mask = mask[:, None]
        return mask.sum(axis=1, dtype=np.int64)

    values = read_path(handle, collection["inputs"][0], event_slice)
    if values.ndim == 1:
        return np.ones(len(values), dtype=np.int64)
    return np.full(len(values), values.shape[1], dtype=np.int64)


def new_object_stats() -> dict:
    return {
        "objects_total": 0,
        "objects_fully_kept": 0,
        "objects_fully_dropped": 0,
        "events_with_partial_object": 0,
        "partial_object_tokens_kept": 0,
    }


def main() -> None:
    args = parse_args()
    quantizers = parse_map(args.object_quantizers)
    cfg = OmegaConf.to_container(OmegaConf.load(args.datamodule_config), resolve=True)
    collections = {
        collection["object_name"]: collection
        for collection in cfg.get("object_collections") or []
    }

    configured_objects = [
        name for name in args.object_order if name in quantizers and name in collections
    ]
    if not configured_objects:
        raise ValueError("No requested object types were found in the datamodule config")

    fixed_tokens = int(not args.no_cls)
    fixed_tokens += int(not args.no_event_token and bool(cfg.get("event_inputs")))
    separator_tokens = int(not args.no_separators)

    required_lengths = []
    excess_tokens = []
    object_stats = {name: new_object_stats() for name in configured_objects}
    total_events = 0
    truncated_events = 0
    partial_events = 0

    for h5_file in args.h5_files:
        with h5py.File(h5_file, mode="r") as handle:
            n_events = infer_n_events(handle, collections)
            if args.num_events_per_file is not None:
                n_events = min(n_events, args.num_events_per_file)

            for start in range(0, n_events, args.batch_size):
                end = min(start + args.batch_size, n_events)
                event_slice = slice(start, end)
                batch_size = end - start
                counts = {
                    name: read_object_counts(
                        handle,
                        collections[name],
                        event_slice,
                        cfg.get("mask_input"),
                    )
                    for name in configured_objects
                }

                lengths = np.full(batch_size, fixed_tokens, dtype=np.int64)
                for name in configured_objects:
                    lengths += counts[name] * quantizers[name] + separator_tokens

                required_lengths.append(lengths)
                batch_excess = np.maximum(lengths - args.max_seq_length, 0)
                excess_tokens.append(batch_excess)
                truncated = lengths > args.max_seq_length
                total_events += batch_size
                truncated_events += int(truncated.sum())

                remaining = np.full(
                    batch_size,
                    max(args.max_seq_length - fixed_tokens, 0),
                    dtype=np.int64,
                )
                any_partial = np.zeros(batch_size, dtype=bool)

                for name in configured_objects:
                    q_count = quantizers[name]
                    object_count = counts[name]
                    object_tokens = object_count * q_count
                    kept_object_tokens = np.minimum(object_tokens, remaining)
                    fully_kept = kept_object_tokens // q_count
                    partial = (kept_object_tokens % q_count) > 0
                    fully_dropped = object_count - fully_kept - partial.astype(np.int64)

                    stats = object_stats[name]
                    stats["objects_total"] += int(object_count.sum())
                    stats["objects_fully_kept"] += int(fully_kept.sum())
                    stats["objects_fully_dropped"] += int(fully_dropped.sum())
                    stats["events_with_partial_object"] += int(partial.sum())
                    stats["partial_object_tokens_kept"] += int(
                        (kept_object_tokens % q_count).sum()
                    )

                    any_partial |= partial
                    segment_tokens = object_tokens + separator_tokens
                    remaining = np.maximum(remaining - segment_tokens, 0)

                partial_events += int(any_partial.sum())

    lengths = np.concatenate(required_lengths)
    excess = np.concatenate(excess_tokens)
    percentiles = {
        str(percentile): float(np.percentile(lengths, percentile))
        for percentile in (50, 90, 95, 99, 99.9, 100)
    }
    report = {
        "events": total_events,
        "max_seq_length": args.max_seq_length,
        "fixed_tokens_per_event": fixed_tokens,
        "object_order": configured_objects,
        "quantizers_per_object": quantizers,
        "required_length_percentiles": percentiles,
        "events_truncated": truncated_events,
        "events_truncated_fraction": truncated_events / total_events,
        "mean_tokens_removed_per_truncated_event": (
            float(excess[excess > 0].mean()) if truncated_events else 0.0
        ),
        "max_tokens_removed": int(excess.max()),
        "events_with_partial_object": partial_events,
        "events_with_partial_object_fraction": partial_events / total_events,
        "objects": object_stats,
    }

    print(json.dumps(report, indent=2))
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
