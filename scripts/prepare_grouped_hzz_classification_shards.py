#!/usr/bin/env python3
"""Prepare balanced grouped-token shards for ggF H->ZZ versus continuum ZZ."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SPLIT_NAMES = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream grouped token Parquets, retain one signal and one background "
            "DSID, downsample the larger class exactly, and write deterministic "
            "balanced train/validation/test shards."
        )
    )
    parser.add_argument("--signal-parquet", required=True)
    parser.add_argument("--background-parquet", required=True)
    parser.add_argument("--signal-dsid", type=int, default=345060)
    parser.add_argument("--background-dsid", type=int, default=700600)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--read-batch-size", type=int, default=4096)
    parser.add_argument("--shard-rows", type=int, default=50_000)
    parser.add_argument("--compression", default="snappy")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class ShardWriter:
    def __init__(
        self,
        output_dir: Path,
        *,
        shard_rows: int,
        row_group_rows: int,
        compression: str,
        seed: int,
    ) -> None:
        self.output_dir = output_dir
        self.shard_rows = shard_rows
        self.row_group_rows = row_group_rows
        self.compression = compression
        self.rng = np.random.default_rng(seed)
        self.buffer: list[pa.Table] = []
        self.buffered_rows = 0
        self.shard_index = 0
        self.written_rows = 0
        output_dir.mkdir(parents=True, exist_ok=True)

    def add(self, table: pa.Table) -> None:
        if not table.num_rows:
            return
        self.buffer.append(table)
        self.buffered_rows += table.num_rows
        if self.buffered_rows >= self.shard_rows:
            self._flush_full_shards()

    def _combined(self) -> pa.Table:
        return self.buffer[0] if len(self.buffer) == 1 else pa.concat_tables(self.buffer)

    def _flush_full_shards(self) -> None:
        combined = self._combined()
        offset = 0
        while combined.num_rows - offset >= self.shard_rows:
            self._write(combined.slice(offset, self.shard_rows))
            offset += self.shard_rows
        remainder = combined.slice(offset)
        self.buffer = [remainder] if remainder.num_rows else []
        self.buffered_rows = remainder.num_rows

    def finish(self) -> None:
        if self.buffered_rows:
            self._write(self._combined())
        self.buffer = []
        self.buffered_rows = 0

    def _write(self, table: pa.Table) -> None:
        order = self.rng.permutation(table.num_rows)
        table = table.take(pa.array(order, type=pa.int64()))
        path = self.output_dir / f"part-{self.shard_index:05d}.parquet"
        pq.write_table(
            table,
            path,
            compression=self.compression,
            row_group_size=self.row_group_rows,
            use_dictionary=True,
        )
        self.shard_index += 1
        self.written_rows += table.num_rows
        print(f"wrote {path} ({table.num_rows:,} rows)", flush=True)


def count_dsid(path: str, dsid: int, batch_size: int) -> int:
    parquet = pq.ParquetFile(path)
    if "dsid" not in parquet.schema_arrow.names:
        raise ValueError(f"{path} has no dsid column")
    count = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=["dsid"]):
        values = np.asarray(batch.column(0).to_numpy(zero_copy_only=False))
        count += int(np.count_nonzero(values == dsid))
    return count


def exact_split_codes(size: int, train_frac: float, val_frac: float, seed: int) -> np.ndarray:
    train = int(np.floor(size * train_frac))
    val = int(np.floor(size * val_frac))
    codes = np.full(size, 2, dtype=np.int8)
    codes[:train] = 0
    codes[train : train + val] = 1
    np.random.default_rng(seed).shuffle(codes)
    return codes


def identity_hash(source_file: str, event_index: int) -> int:
    value = f"{source_file}\0{event_index}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), "little")


def check_schema(path: str, reference: pa.Schema | None = None) -> pa.Schema:
    schema = pq.ParquetFile(path).schema_arrow
    required = {"tokens", "mask", "source_file", "event_index", "dsid"}
    missing = sorted(required - set(schema.names))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if reference is not None and not schema.equals(reference, check_metadata=False):
        raise ValueError("Signal and background Parquet schemas do not match")
    return schema


def stream_class(
    *,
    path: str,
    dsid: int,
    class_name: str,
    label: int,
    assignment_by_match: np.ndarray,
    writers: dict[tuple[str, str], ShardWriter],
    read_batch_size: int,
    seen_identities: set[int],
    token_shape: list[int] | None,
) -> list[int]:
    parquet = pq.ParquetFile(path)
    match_offset = 0
    observed_shape = token_shape

    for batch in parquet.iter_batches(batch_size=read_batch_size):
        dsids = np.asarray(
            batch.column(batch.schema.get_field_index("dsid")).to_numpy(
                zero_copy_only=False
            )
        )
        matching_rows = np.flatnonzero(dsids == dsid)
        if not matching_rows.size:
            continue

        assignments = assignment_by_match[
            match_offset : match_offset + matching_rows.size
        ]
        match_offset += matching_rows.size
        selected = assignments >= 0
        if not selected.any():
            continue

        selected_rows = matching_rows[selected]
        selected_assignments = assignments[selected]
        selected_batch = pa.Table.from_batches([batch]).take(
            pa.array(selected_rows, type=pa.int64())
        )

        if observed_shape is None:
            sample = np.asarray(selected_batch["tokens"].slice(0, 1).to_pylist())
            if sample.ndim != 3:
                raise ValueError(
                    f"Expected grouped tokens shaped [events, positions, quantizers], "
                    f"found {sample.shape} in {path}"
                )
            observed_shape = list(sample.shape[1:])
        sample_shape = np.asarray(selected_batch["tokens"].slice(0, 1).to_pylist()).shape
        if list(sample_shape[1:]) != observed_shape:
            raise ValueError(f"Token shape changed within inputs: {sample_shape[1:]}")

        sources = selected_batch["source_file"].to_pylist()
        events = selected_batch["event_index"].to_pylist()
        for source, event in zip(sources, events, strict=True):
            event_hash = identity_hash(str(source), int(event))
            if event_hash in seen_identities:
                raise ValueError(f"Duplicate selected event identity: {source}:{event}")
            seen_identities.add(event_hash)

        labels = pa.array(np.full(selected_batch.num_rows, label, dtype=np.int64))
        selected_batch = selected_batch.append_column("label", labels)
        for split_code, split_name in enumerate(SPLIT_NAMES):
            split_mask = selected_assignments == split_code
            if split_mask.any():
                writers[(split_name, class_name)].add(
                    selected_batch.filter(pa.array(split_mask))
                )

    if match_offset != assignment_by_match.size:
        raise RuntimeError(
            f"DSID {dsid} row count changed between passes: "
            f"expected {assignment_by_match.size:,}, found {match_offset:,}"
        )
    return observed_shape or []


def prepare(args: argparse.Namespace) -> None:
    if not 0 < args.train_frac < 1 or not 0 < args.val_frac < 1:
        raise ValueError("Split fractions must be between zero and one")
    if args.train_frac + args.val_frac >= 1:
        raise ValueError("--train-frac + --val-frac must be less than one")
    if args.read_batch_size <= 0 or args.shard_rows <= 0:
        raise ValueError("Batch and shard sizes must be positive")

    signal_schema = check_schema(args.signal_parquet)
    check_schema(args.background_parquet, signal_schema)
    signal_available = count_dsid(args.signal_parquet, args.signal_dsid, args.read_batch_size)
    background_available = count_dsid(
        args.background_parquet, args.background_dsid, args.read_batch_size
    )
    if not signal_available or not background_available:
        raise ValueError(
            f"No usable rows: signal={signal_available:,}, background={background_available:,}"
        )

    rows_per_class = min(signal_available, background_available)
    print(f"signal DSID {args.signal_dsid}: {signal_available:,} available", flush=True)
    print(
        f"background DSID {args.background_dsid}: {background_available:,} available",
        flush=True,
    )
    print(f"balanced selection: {rows_per_class:,} rows per class", flush=True)

    signal_assignment = np.full(signal_available, -1, dtype=np.int8)
    signal_selected = np.random.default_rng(args.seed).choice(
        signal_available, size=rows_per_class, replace=False
    )
    signal_assignment[signal_selected] = exact_split_codes(
        rows_per_class, args.train_frac, args.val_frac, args.seed + 1
    )
    background_assignment = exact_split_codes(
        rows_per_class, args.train_frac, args.val_frac, args.seed + 2
    )

    output_dir = args.output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    temporary_dir = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}"
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)

    writers = {}
    for split_index, split_name in enumerate(SPLIT_NAMES):
        for class_index, class_name in enumerate(("background", "signal")):
            writers[(split_name, class_name)] = ShardWriter(
                temporary_dir / split_name / class_name,
                shard_rows=args.shard_rows,
                row_group_rows=args.read_batch_size,
                compression=args.compression,
                seed=args.seed + 100 * split_index + class_index,
            )

    seen_identities: set[int] = set()
    try:
        token_shape = stream_class(
            path=args.signal_parquet,
            dsid=args.signal_dsid,
            class_name="signal",
            label=1,
            assignment_by_match=signal_assignment,
            writers=writers,
            read_batch_size=args.read_batch_size,
            seen_identities=seen_identities,
            token_shape=None,
        )
        token_shape = stream_class(
            path=args.background_parquet,
            dsid=args.background_dsid,
            class_name="background",
            label=0,
            assignment_by_match=background_assignment,
            writers=writers,
            read_batch_size=args.read_batch_size,
            seen_identities=seen_identities,
            token_shape=token_shape,
        )
        for writer in writers.values():
            writer.finish()

        split_counts = {}
        for split_name in SPLIT_NAMES:
            signal_rows = writers[(split_name, "signal")].written_rows
            background_rows = writers[(split_name, "background")].written_rows
            if signal_rows != background_rows:
                raise RuntimeError(f"{split_name} is not balanced")
            split_counts[split_name] = {
                "signal": signal_rows,
                "background": background_rows,
                "total": signal_rows + background_rows,
            }

        manifest = {
            "format_version": 1,
            "task": "ggF H->ZZ*->4l versus continuum VV/ZZ*->4l",
            "signal": {
                "dsid": args.signal_dsid,
                "label": 1,
                "input_parquet": str(Path(args.signal_parquet).resolve()),
                "available_rows": signal_available,
                "selected_rows": rows_per_class,
            },
            "background": {
                "dsid": args.background_dsid,
                "label": 0,
                "input_parquet": str(Path(args.background_parquet).resolve()),
                "available_rows": background_available,
                "selected_rows": rows_per_class,
            },
            "selection": "exact seeded random downsampling of the larger class",
            "split_method": "exact seeded event-level assignment independently per class",
            "seed": args.seed,
            "train_fraction": args.train_frac,
            "validation_fraction": args.val_frac,
            "test_fraction": 1.0 - args.train_frac - args.val_frac,
            "token_shape": token_shape,
            "selected_identity_count": len(seen_identities),
            "identity_overlap_detected": False,
            "split_counts": split_counts,
            "read_batch_size": args.read_batch_size,
            "shard_rows": args.shard_rows,
            "compression": args.compression,
            "notes": [
                "Classes are stored separately under each split for explicit "
                "balanced interleaving.",
                "This benchmark is unweighted; generator and pileup event weights are unavailable.",
                "Inputs intentionally match the completed no-CLS grouped foundation pretraining.",
            ],
        }
        (temporary_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        temporary_dir.rename(output_dir)
    except Exception:
        print(f"preparation failed; partial output retained at {temporary_dir}")
        raise

    print(f"prepared dataset: {output_dir}")
    for split_name, counts in split_counts.items():
        print(
            f"{split_name}: {counts['total']:,} rows "
            f"({counts['signal']:,} signal + {counts['background']:,} background)"
        )


def main() -> None:
    prepare(parse_args())


if __name__ == "__main__":
    main()
