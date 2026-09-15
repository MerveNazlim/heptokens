#!/usr/bin/env python3
"""Create bounded-memory, stratified train/validation Parquet shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream event-token Parquet files once and write source-file-stratified "
            "train/validation shards for foundation pretraining."
        )
    )
    parser.add_argument("--input-parquets", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-frac", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--read-batch-size", type=int, default=4096)
    parser.add_argument("--shard-rows", type=int, default=50_000)
    parser.add_argument(
        "--max-rows-per-input",
        type=int,
        help="Optional smoke-test cap applied independently to each input parquet.",
    )
    parser.add_argument("--compression", default="snappy")
    parser.add_argument(
        "--min-free-factor",
        type=float,
        default=1.1,
        help="Require this multiple of the compressed input size as free output space.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing prepared output directory.",
    )
    return parser.parse_args()


def stable_string_hash(value: str) -> np.uint64:
    digest = hashlib.blake2b(
        value.encode("utf-8"), digest_size=8, person=b"heptokens"
    ).digest()
    return np.uint64(int.from_bytes(digest, "little"))


def split_hash(values: np.ndarray, seed: int) -> np.ndarray:
    mixed = values.astype(np.uint64, copy=True) + np.uint64(seed)
    mixed ^= mixed >> np.uint64(30)
    mixed *= np.uint64(0xBF58476D1CE4E5B9)
    mixed ^= mixed >> np.uint64(27)
    mixed *= np.uint64(0x94D049BB133111EB)
    mixed ^= mixed >> np.uint64(31)
    return mixed


def input_domain(path: str) -> str:
    name = Path(path).stem.lower()
    for domain in ("signal", "background", "data"):
        if domain in name:
            return domain
    return Path(path).stem


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
        if table.num_rows == 0:
            return
        self.buffer.append(table)
        self.buffered_rows += table.num_rows
        if self.buffered_rows >= self.shard_rows:
            self._flush_full_shards()

    def _combined(self) -> pa.Table:
        if len(self.buffer) == 1:
            return self.buffer[0]
        return pa.concat_tables(self.buffer)

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
        permutation = self.rng.permutation(table.num_rows)
        shuffled = table.take(pa.array(permutation, type=pa.int64()))
        path = self.output_dir / f"part-{self.shard_index:05d}.parquet"
        pq.write_table(
            shuffled,
            path,
            compression=self.compression,
            row_group_size=self.row_group_rows,
            use_dictionary=True,
        )
        self.shard_index += 1
        self.written_rows += table.num_rows
        print(f"wrote {path} ({table.num_rows:,} rows)", flush=True)


def update_source_counts(
    source_stats: dict,
    sources: np.ndarray,
    train_mask: np.ndarray,
    *,
    input_path: str,
    batch: pa.RecordBatch,
) -> None:
    is_mc_values = (
        np.asarray(
            batch.column(batch.schema.get_field_index("is_mc")).to_numpy(
                zero_copy_only=False
            )
        )
        if "is_mc" in batch.schema.names
        else None
    )
    dsid_values = (
        np.asarray(
            batch.column(batch.schema.get_field_index("dsid")).to_numpy(
                zero_copy_only=False
            )
        )
        if "dsid" in batch.schema.names
        else None
    )
    for source in np.unique(sources):
        source_mask = sources == source
        stats = source_stats.setdefault(
            str(source),
            {
                "input_parquet": input_path,
                "is_mc": bool(is_mc_values[source_mask][0]) if is_mc_values is not None else None,
                "dsid": int(dsid_values[source_mask][0]) if dsid_values is not None else None,
                "total": 0,
                "train": 0,
                "val": 0,
            },
        )
        total = int(source_mask.sum())
        train = int((source_mask & train_mask).sum())
        stats["total"] += total
        stats["train"] += train
        stats["val"] += total - train


def prepare(args: argparse.Namespace) -> None:
    if not 0.0 < args.train_frac < 1.0:
        raise ValueError("--train-frac must be between zero and one")
    if args.read_batch_size <= 0 or args.shard_rows <= 0:
        raise ValueError("Batch and shard sizes must be positive")
    if args.max_rows_per_input is not None and args.max_rows_per_input <= 0:
        raise ValueError("--max-rows-per-input must be positive")
    if args.min_free_factor <= 0:
        raise ValueError("--min-free-factor must be positive")

    output_dir = args.output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    input_bytes = 0
    for path in args.input_parquets:
        file_bytes = Path(path).stat().st_size
        if args.max_rows_per_input is not None:
            available_rows = pq.ParquetFile(path).metadata.num_rows
            fraction = min(1.0, args.max_rows_per_input / max(available_rows, 1))
            file_bytes = int(file_bytes * fraction)
        input_bytes += file_bytes
    free_bytes = shutil.disk_usage(output_dir.parent).free
    required_bytes = int(input_bytes * args.min_free_factor)
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"Insufficient free space under {output_dir.parent}: "
            f"need at least {required_bytes / 2**30:.1f} GiB, "
            f"have {free_bytes / 2**30:.1f} GiB"
        )
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    temporary_dir = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}"
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True)

    train_writer = ShardWriter(
        temporary_dir / "train",
        shard_rows=args.shard_rows,
        row_group_rows=args.read_batch_size,
        compression=args.compression,
        seed=args.seed + 1,
    )
    val_writer = ShardWriter(
        temporary_dir / "val",
        shard_rows=args.shard_rows,
        row_group_rows=args.read_batch_size,
        compression=args.compression,
        seed=args.seed + 2,
    )

    threshold = np.uint64(int(args.train_frac * 2**64))
    source_hash_cache: dict[str, np.uint64] = {}
    source_stats: dict = {}
    domain_stats = defaultdict(lambda: {"total": 0, "train": 0, "val": 0})
    input_rows = {}
    input_available_rows = {}
    reference_schema = None

    try:
        for input_path in args.input_parquets:
            parquet = pq.ParquetFile(input_path)
            schema = parquet.schema_arrow
            required = {"source_file", "event_index", "tokens", "mask"}
            missing = sorted(required - set(schema.names))
            if missing:
                raise ValueError(f"{input_path} is missing required columns: {missing}")
            if reference_schema is None:
                reference_schema = schema
            elif not schema.equals(reference_schema, check_metadata=False):
                raise ValueError(f"Parquet schema does not match previous inputs: {input_path}")

            domain = input_domain(input_path)
            input_available_rows[input_path] = parquet.metadata.num_rows
            input_rows[input_path] = 0
            print(
                f"reading {input_path} ({parquet.metadata.num_rows:,} rows, domain={domain})",
                flush=True,
            )

            for batch in parquet.iter_batches(batch_size=args.read_batch_size):
                if (
                    args.max_rows_per_input is not None
                    and input_rows[input_path] >= args.max_rows_per_input
                ):
                    break
                if args.max_rows_per_input is not None:
                    remaining = args.max_rows_per_input - input_rows[input_path]
                    if batch.num_rows > remaining:
                        batch = batch.slice(0, remaining)
                input_rows[input_path] += batch.num_rows
                source_index = batch.schema.get_field_index("source_file")
                event_index = batch.schema.get_field_index("event_index")
                sources = np.asarray(batch.column(source_index).to_pylist(), dtype=object)
                events = np.asarray(
                    batch.column(event_index).to_numpy(zero_copy_only=False),
                    dtype=np.uint64,
                )
                source_hashes = np.empty(batch.num_rows, dtype=np.uint64)
                for source in np.unique(sources):
                    source_text = str(source)
                    source_hash = source_hash_cache.setdefault(
                        source_text, stable_string_hash(source_text)
                    )
                    source_hashes[sources == source] = source_hash

                train_mask = split_hash(events ^ source_hashes, args.seed) < threshold
                train_count = int(train_mask.sum())
                val_count = batch.num_rows - train_count
                domain_stats[domain]["total"] += batch.num_rows
                domain_stats[domain]["train"] += train_count
                domain_stats[domain]["val"] += val_count
                update_source_counts(
                    source_stats,
                    sources,
                    train_mask,
                    input_path=input_path,
                    batch=batch,
                )

                table = pa.Table.from_batches([batch])
                train_writer.add(table.filter(pa.array(train_mask)))
                val_writer.add(table.filter(pa.array(~train_mask)))

        train_writer.finish()
        val_writer.finish()
        manifest = {
            "format_version": 1,
            "seed": args.seed,
            "train_fraction_requested": args.train_frac,
            "split_method": "blake2b(source_file) xor event_index, then splitmix64",
            "stratification": "source_file",
            "read_batch_size": args.read_batch_size,
            "shard_rows": args.shard_rows,
            "compression": args.compression,
            "compressed_input_bytes_processed_estimate": input_bytes,
            "input_parquets": [str(Path(path).resolve()) for path in args.input_parquets],
            "input_available_rows": input_available_rows,
            "input_rows_processed": input_rows,
            "max_rows_per_input": args.max_rows_per_input,
            "train_rows": train_writer.written_rows,
            "val_rows": val_writer.written_rows,
            "total_rows": train_writer.written_rows + val_writer.written_rows,
            "train_shards": train_writer.shard_index,
            "val_shards": val_writer.shard_index,
            "domain_counts": dict(domain_stats),
            "source_counts": dict(sorted(source_stats.items())),
        }
        (temporary_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        temporary_dir.rename(output_dir)
    except Exception:
        print(f"preparation failed; partial output retained at {temporary_dir}")
        raise

    print(f"prepared dataset: {output_dir}")
    print(f"train: {train_writer.written_rows:,} rows in {train_writer.shard_index} shards")
    print(f"val:   {val_writer.written_rows:,} rows in {val_writer.shard_index} shards")


def main() -> None:
    prepare(parse_args())


if __name__ == "__main__":
    main()
