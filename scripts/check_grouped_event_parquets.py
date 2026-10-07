"""Validate grouped event-token parquet files without loading them fully into RAM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


REQUIRED_COLUMNS = {
    "tokens",
    "mask",
    "type_ids",
    "dsid",
    "cross_section_pb",
    "filter_efficiency",
    "k_factor",
    "effective_cross_section_pb",
    "process",
    "generator",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Stream through every row and validate metadata and sequence lengths.",
    )
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--signal-dsids", default="345060,344235")
    return parser.parse_args()


def classify_path(path: Path) -> str:
    name = path.name.lower()
    if "signal" in name:
        return "signal"
    if "background" in name:
        return "background"
    if "data" in name:
        return "data"
    return "unknown"


def full_validation(
    path: Path,
    parquet: pq.ParquetFile,
    batch_size: int,
    signal_dsids: set[int],
) -> None:
    sample = classify_path(path)
    columns = [
        "mask",
        "dsid",
        "cross_section_pb",
        "filter_efficiency",
        "k_factor",
        "effective_cross_section_pb",
        "process",
        "generator",
    ]
    rows_seen = 0
    min_length = None
    max_length = 0
    length_sum = 0
    truncated = 0
    metadata_by_dsid: dict[int, tuple[float, float, float, float, str, str]] = {}

    for batch_index, batch in enumerate(
        parquet.iter_batches(batch_size=batch_size, columns=columns), start=1
    ):
        mask = np.asarray(batch["mask"].to_pylist(), dtype=bool)
        lengths = mask.sum(axis=1)
        dsids = np.asarray(batch["dsid"], dtype=np.int64)
        xsec = np.asarray(batch["cross_section_pb"], dtype=np.float64)
        filt = np.asarray(batch["filter_efficiency"], dtype=np.float64)
        kfactor = np.asarray(batch["k_factor"], dtype=np.float64)
        effective = np.asarray(batch["effective_cross_section_pb"], dtype=np.float64)

        if not all(np.all(np.isfinite(values)) for values in (xsec, filt, kfactor, effective)):
            raise RuntimeError(f"{path}: non-finite metadata in batch {batch_index}")
        if np.any(xsec < 0) or np.any(filt < 0) or np.any(kfactor < 0):
            raise RuntimeError(f"{path}: negative metadata in batch {batch_index}")
        if not np.allclose(effective, xsec * filt * kfactor, rtol=1e-6, atol=1e-12):
            raise RuntimeError(
                f"{path}: inconsistent effective cross section in batch {batch_index}"
            )

        if sample == "data":
            if np.any(dsids != 0) or np.any(xsec != 0) or np.any(effective != 0):
                raise RuntimeError(f"{path}: invalid data metadata in batch {batch_index}")
        else:
            if np.any(dsids <= 0):
                raise RuntimeError(f"{path}: nonpositive MC DSID in batch {batch_index}")
            present = set(dsids.tolist())
            if sample == "signal" and not present <= signal_dsids:
                raise RuntimeError(f"{path}: non-signal DSIDs found: {sorted(present - signal_dsids)}")
            if sample == "background" and present & signal_dsids:
                raise RuntimeError(
                    f"{path}: signal DSIDs found in background: {sorted(present & signal_dsids)}"
                )

        processes = np.asarray(batch["process"].to_pylist(), dtype=object)
        generators = np.asarray(batch["generator"].to_pylist(), dtype=object)
        for dsid_value in np.unique(dsids):
            selected = np.flatnonzero(dsids == dsid_value)
            index = int(selected[0])
            dsid = int(dsid_value)
            values = (
                float(xsec[index]),
                float(filt[index]),
                float(kfactor[index]),
                float(effective[index]),
                processes[index] or "",
                generators[index] or "",
            )
            if not all(
                np.allclose(array[selected], value, rtol=1e-9, atol=1e-12)
                for array, value in zip((xsec, filt, kfactor, effective), values[:4])
            ):
                raise RuntimeError(
                    f"{path}: varying numeric metadata for DSID {dsid} in batch {batch_index}"
                )
            selected_processes = np.where(
                np.equal(processes[selected], None), "", processes[selected]
            )
            selected_generators = np.where(
                np.equal(generators[selected], None), "", generators[selected]
            )
            if np.any(selected_processes != values[4]) or np.any(
                selected_generators != values[5]
            ):
                raise RuntimeError(
                    f"{path}: varying process/generator for DSID {dsid} in batch {batch_index}"
                )
            previous = metadata_by_dsid.setdefault(dsid, values)
            numeric_match = np.allclose(previous[:4], values[:4], rtol=1e-9, atol=1e-12)
            if not numeric_match or previous[4:] != values[4:]:
                raise RuntimeError(
                    f"{path}: conflicting metadata for DSID {dsid}: {previous} versus {values}"
                )

        rows_seen += len(batch)
        batch_min = int(lengths.min())
        min_length = batch_min if min_length is None else min(min_length, batch_min)
        max_length = max(max_length, int(lengths.max()))
        length_sum += int(lengths.sum())
        truncated += int(np.count_nonzero(lengths == mask.shape[1]))
        if batch_index % 100 == 0:
            print(f"  scanned {rows_seen:,}/{parquet.metadata.num_rows:,} rows", flush=True)

    if rows_seen != parquet.metadata.num_rows:
        raise RuntimeError(
            f"{path}: scanned {rows_seen:,} rows, expected {parquet.metadata.num_rows:,}"
        )

    print("  full validation: OK")
    print(
        f"  global sequence lengths: min={min_length} "
        f"mean={length_sum / rows_seen:.2f} max={max_length}"
    )
    print(f"  sequences at max length: {truncated:,}")
    print("  complete DSID metadata:")
    for dsid, values in sorted(metadata_by_dsid.items()):
        xsec, filt, kfactor, effective, process, generator = values
        print(
            f"    {dsid}: xsec={xsec:.9g} pb filter={filt:.9g} "
            f"k={kfactor:.9g} effective={effective:.9g} pb "
            f"process={process!r} generator={generator!r}"
        )


def check_file(
    path: Path,
    rows: int,
    full: bool,
    batch_size: int,
    signal_dsids: set[int],
) -> None:
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    metadata = schema.metadata or {}
    missing = REQUIRED_COLUMNS - set(schema.names)
    if missing:
        raise RuntimeError(f"{path}: missing columns: {sorted(missing)}")
    if b"heptokens_token_vocabulary" not in metadata:
        raise RuntimeError(f"{path}: missing token vocabulary metadata")

    vocabulary = json.loads(metadata[b"heptokens_token_vocabulary"])
    layout = metadata.get(b"heptokens_sequence_layout", b"").decode()
    max_quantizers = int(vocabulary["max_quantizers"])
    vocab_size = int(vocabulary["vocab_size"])
    object_quantizers = {
        name: int(spec["num_quantizers"])
        for name, spec in vocabulary["objects"].items()
    }

    columns = sorted(REQUIRED_COLUMNS)
    batches = parquet.iter_batches(batch_size=min(rows, 2048), columns=columns)
    batch = next(batches)
    tokens = np.asarray(batch["tokens"].to_pylist(), dtype=np.int64)
    mask = np.asarray(batch["mask"].to_pylist(), dtype=bool)
    type_ids = np.asarray(batch["type_ids"].to_pylist(), dtype=np.int64)
    dsid = np.asarray(batch["dsid"])
    xsec = np.asarray(batch["cross_section_pb"])
    filt = np.asarray(batch["filter_efficiency"])
    kfactor = np.asarray(batch["k_factor"])
    effective = np.asarray(batch["effective_cross_section_pb"])

    if tokens.ndim != 3 or tokens.shape[2] != max_quantizers:
        raise RuntimeError(
            f"{path}: tokens shape {tokens.shape} is incompatible with "
            f"max_quantizers={max_quantizers}"
        )
    if mask.shape != tokens.shape[:2] or type_ids.shape != mask.shape:
        raise RuntimeError(
            f"{path}: incompatible shapes tokens={tokens.shape}, mask={mask.shape}, "
            f"type_ids={type_ids.shape}"
        )
    if vocabulary.get("includes_cls", False):
        cls_token_id = int(vocabulary["special_tokens"]["cls"])
        pad_token_id = int(vocabulary["special_tokens"]["pad"])
        if not np.all(mask[:, 0]):
            raise RuntimeError(f"{path}: CLS position is not active in every sampled row")
        if not np.all(tokens[:, 0, 0] == cls_token_id):
            raise RuntimeError(f"{path}: invalid CLS token in sampled rows")
        if not np.all(tokens[:, 0, 1:] == pad_token_id):
            raise RuntimeError(f"{path}: non-padding codes found in grouped CLS positions")
        if not np.all(type_ids[:, 0] == 0):
            raise RuntimeError(f"{path}: grouped CLS positions must use structural type ID 0")
    active = tokens[mask]
    if active.size and (active.min() < 0 or active.max() >= vocab_size):
        raise RuntimeError(
            f"{path}: active token range [{active.min()}, {active.max()}] "
            f"exceeds vocab_size={vocab_size}"
        )
    expected_effective = xsec * filt * kfactor
    if not np.allclose(effective, expected_effective, rtol=1e-6, atol=1e-12):
        raise RuntimeError(f"{path}: effective cross sections are inconsistent")

    is_data = np.all(dsid == 0)
    if is_data and (np.any(xsec != 0) or np.any(effective != 0)):
        raise RuntimeError(f"{path}: data rows contain nonzero cross sections")
    if not is_data and np.any(dsid <= 0):
        raise RuntimeError(f"{path}: MC rows contain nonpositive DSIDs")

    lengths = mask.sum(axis=1)
    print(f"\n{path}")
    print(f"  status: OK")
    print(f"  rows: {parquet.metadata.num_rows:,}")
    print(f"  layout: {layout}")
    print(f"  includes CLS: {bool(vocabulary.get('includes_cls', False))}")
    print(f"  sampled tokens shape: {tokens.shape}")
    print(f"  sequence lengths: min={lengths.min()} median={np.median(lengths):.0f} max={lengths.max()}")
    print(f"  vocab_size: {vocab_size:,}")
    print(f"  quantizers: {object_quantizers}")
    print(f"  sampled DSIDs: {sorted(set(dsid.tolist()))[:20]}")
    print(
        "  sampled metadata: "
        f"xsec=[{xsec.min():.6g}, {xsec.max():.6g}] pb, "
        f"filter=[{filt.min():.6g}, {filt.max():.6g}], "
        f"k=[{kfactor.min():.6g}, {kfactor.max():.6g}]"
    )
    if full:
        full_validation(path, parquet, batch_size, signal_dsids)


def main() -> None:
    args = parse_args()
    signal_dsids = {int(value) for value in args.signal_dsids.split(",") if value}
    for path in args.paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        check_file(path, args.rows, args.full, args.batch_size, signal_dsids)


if __name__ == "__main__":
    main()
