#!/usr/bin/env python3
"""Prepare deterministic grouped-token shards for H->4l production classes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class ClassSpec:
    name: str
    label: int
    parquet: Path
    dsids: tuple[int, ...]


@dataclass(frozen=True)
class ClassDefinition:
    name: str
    label: int
    sources: tuple[ClassSpec, ...]


def parse_class_spec(value: str) -> ClassSpec:
    try:
        name, label_text, parquet_text, dsid_text = value.split(":", 3)
        dsids = tuple(int(item) for item in dsid_text.split(",") if item)
        label = int(label_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "class specifications must be NAME:LABEL:PARQUET:DSID[,DSID...]"
        ) from exc
    if not name or not dsids:
        raise argparse.ArgumentTypeError("class name and DSID list must be non-empty")
    return ClassSpec(
        name=name,
        label=label,
        parquet=Path(parquet_text).resolve(),
        dsids=dsids,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select labelled DSIDs from grouped-token Parquets and write "
            "deterministic multiclass train/validation/test shards."
        )
    )
    parser.add_argument(
        "--class-spec",
        action="append",
        type=parse_class_spec,
        required=True,
        help="NAME:LABEL:PARQUET:DSID[,DSID...] (repeat once per class)",
    )
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
        shuffled = table.take(pa.array(order, type=pa.int64()))
        path = self.output_dir / f"part-{self.shard_index:05d}.parquet"
        pq.write_table(
            shuffled,
            path,
            compression=self.compression,
            row_group_size=self.row_group_rows,
            use_dictionary=True,
        )
        self.shard_index += 1
        self.written_rows += shuffled.num_rows
        print(f"wrote {path} ({shuffled.num_rows:,} rows)", flush=True)


def validate_specs(specs: list[ClassSpec]) -> list[ClassDefinition]:
    if len(specs) < 2:
        raise ValueError("At least two classes are required")

    name_to_label: dict[str, int] = {}
    label_to_name: dict[int, str] = {}
    for spec in specs:
        previous_label = name_to_label.setdefault(spec.name, spec.label)
        previous_name = label_to_name.setdefault(spec.label, spec.name)
        if previous_label != spec.label or previous_name != spec.name:
            raise ValueError(
                "Repeated class sources must use the same name and label"
            )
    labels = sorted(label_to_name)
    if labels != list(range(len(labels))):
        raise ValueError("Class labels must be contiguous integers starting at zero")

    assignments: dict[tuple[Path, int], str] = {}
    for spec in specs:
        if not spec.parquet.is_file():
            raise FileNotFoundError(spec.parquet)
        for dsid in spec.dsids:
            key = (spec.parquet, dsid)
            previous = assignments.setdefault(key, spec.name)
            if previous != spec.name:
                raise ValueError(
                    f"{spec.parquet} DSID {dsid} is assigned to both "
                    f"{previous} and {spec.name}"
                )
    return [
        ClassDefinition(
            name=label_to_name[label],
            label=label,
            sources=tuple(spec for spec in specs if spec.label == label),
        )
        for label in labels
    ]


def check_schemas(paths: list[Path]) -> pa.Schema:
    reference: pa.Schema | None = None
    for path in paths:
        schema = pq.ParquetFile(path).schema_arrow
        required = {"tokens", "mask", "source_file", "event_index", "dsid"}
        missing = sorted(required - set(schema.names))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        if reference is None:
            reference = schema
            continue
        if not schema.equals(reference, check_metadata=False):
            raise ValueError(f"Parquet schema mismatch: {path}")
        if (schema.metadata or {}).get(b"heptokens_continuous_schema") != (
            reference.metadata or {}
        ).get(b"heptokens_continuous_schema"):
            raise ValueError(f"Continuous feature schema mismatch: {path}")
    assert reference is not None
    return reference


def count_selected_rows(
    specs: list[ClassSpec], read_batch_size: int
) -> tuple[dict[str, int], dict[str, dict[int, int]], dict[tuple[Path, int], int]]:
    lookup = {
        (spec.parquet, dsid): spec.name
        for spec in specs
        for dsid in spec.dsids
    }
    class_counts = Counter()
    dsid_counts: dict[str, Counter] = defaultdict(Counter)
    source_counts: Counter = Counter()
    for path in sorted({spec.parquet for spec in specs}):
        parquet = pq.ParquetFile(path)
        target_dsids = {
            dsid for candidate, dsid in lookup if candidate == path
        }
        print(f"counting selected DSIDs in {path}", flush=True)
        for batch in parquet.iter_batches(
            batch_size=read_batch_size,
            columns=["dsid"],
            use_threads=True,
        ):
            values = np.asarray(batch.column(0).to_numpy(zero_copy_only=False))
            present, counts = np.unique(values, return_counts=True)
            for dsid_value, count in zip(present, counts, strict=True):
                dsid = int(dsid_value)
                if dsid not in target_dsids:
                    continue
                class_name = lookup[(path, dsid)]
                class_counts[class_name] += int(count)
                dsid_counts[class_name][dsid] += int(count)
                source_counts[(path, dsid)] += int(count)
    return (
        dict(class_counts),
        {name: dict(counts) for name, counts in dsid_counts.items()},
        dict(source_counts),
    )


def exact_split_codes(
    size: int, train_frac: float, val_frac: float, seed: int
) -> np.ndarray:
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


def prepare(args: argparse.Namespace) -> None:
    specs: list[ClassSpec] = sorted(args.class_spec, key=lambda spec: spec.label)
    classes = validate_specs(specs)
    if not 0 < args.train_frac < 1 or not 0 < args.val_frac < 1:
        raise ValueError("Split fractions must be between zero and one")
    if args.train_frac + args.val_frac >= 1:
        raise ValueError("--train-frac + --val-frac must be less than one")
    if args.read_batch_size <= 0 or args.shard_rows <= 0:
        raise ValueError("Batch and shard sizes must be positive")

    unique_paths = sorted({spec.parquet for spec in specs})
    check_schemas(unique_paths)
    class_counts, dsid_counts, source_counts = count_selected_rows(
        specs, args.read_batch_size
    )
    for spec in specs:
        missing_dsids = [
            dsid
            for dsid in spec.dsids
            if source_counts.get((spec.parquet, dsid), 0) <= 0
        ]
        if missing_dsids:
            raise ValueError(
                f"Class {spec.name} has no rows for DSIDs {missing_dsids} "
                f"in {spec.parquet}"
            )
    for class_definition in classes:
        count = class_counts.get(class_definition.name, 0)
        if count <= 0:
            raise ValueError(f"Class {class_definition.name} has no rows")
        print(
            f"class {class_definition.label} {class_definition.name}: "
            f"{count:,} rows from "
            f"{dsid_counts.get(class_definition.name, {})}",
            flush=True,
        )

    assignments = {
        (spec.parquet, dsid): exact_split_codes(
            source_counts[(spec.parquet, dsid)],
            args.train_frac,
            args.val_frac,
            args.seed + 1000 * spec.label + dsid,
        )
        for spec in specs
        for dsid in spec.dsids
    }
    offsets = {
        (spec.parquet, dsid): 0
        for spec in specs
        for dsid in spec.dsids
    }
    spec_by_path_dsid = {
        (spec.parquet, dsid): spec for spec in specs for dsid in spec.dsids
    }

    output_dir = args.output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    temporary_dir = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}"
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)

    writers: dict[tuple[str, str], ShardWriter] = {}
    for split_index, split_name in enumerate(SPLIT_NAMES):
        for class_definition in classes:
            writers[(split_name, class_definition.name)] = ShardWriter(
                temporary_dir / split_name / class_definition.name,
                shard_rows=args.shard_rows,
                row_group_rows=args.read_batch_size,
                compression=args.compression,
                seed=args.seed + 100 * split_index + class_definition.label,
            )

    seen_identities: set[int] = set()
    token_shape: list[int] | None = None
    try:
        for path in unique_paths:
            parquet = pq.ParquetFile(path)
            target_dsids = {
                dsid for candidate, dsid in spec_by_path_dsid if candidate == path
            }
            print(f"streaming selected rows from {path}", flush=True)
            for batch in parquet.iter_batches(batch_size=args.read_batch_size):
                dsids = np.asarray(
                    batch.column(batch.schema.get_field_index("dsid")).to_numpy(
                        zero_copy_only=False
                    )
                )
                for dsid in sorted(target_dsids.intersection(np.unique(dsids))):
                    spec = spec_by_path_dsid[(path, int(dsid))]
                    selected_rows = np.flatnonzero(dsids == dsid)
                    assignment_key = (path, int(dsid))
                    start = offsets[assignment_key]
                    stop = start + selected_rows.size
                    selected_assignments = assignments[assignment_key][start:stop]
                    offsets[assignment_key] = stop

                    table = pa.Table.from_batches([batch]).take(
                        pa.array(selected_rows, type=pa.int64())
                    )
                    if token_shape is None:
                        sample = np.asarray(table["tokens"].slice(0, 1).to_pylist())
                        if sample.ndim != 3:
                            raise ValueError(
                                "Expected grouped tokens [events, positions, quantizers], "
                                f"found {sample.shape} in {path}"
                            )
                        token_shape = list(sample.shape[1:])
                    sample_shape = np.asarray(
                        table["tokens"].slice(0, 1).to_pylist()
                    ).shape
                    if list(sample_shape[1:]) != token_shape:
                        raise ValueError(
                            f"Token shape changed within inputs: {sample_shape[1:]}"
                        )

                    for source, event in zip(
                        table["source_file"].to_pylist(),
                        table["event_index"].to_pylist(),
                        strict=True,
                    ):
                        event_hash = identity_hash(str(source), int(event))
                        if event_hash in seen_identities:
                            raise ValueError(
                                f"Duplicate selected event identity: {source}:{event}"
                            )
                        seen_identities.add(event_hash)

                    table = table.append_column(
                        "label",
                        pa.array(
                            np.full(table.num_rows, spec.label, dtype=np.int64)
                        ),
                    )
                    for split_code, split_name in enumerate(SPLIT_NAMES):
                        split_mask = selected_assignments == split_code
                        if split_mask.any():
                            writers[(split_name, spec.name)].add(
                                table.filter(pa.array(split_mask))
                            )

        for spec in specs:
            for dsid in spec.dsids:
                key = (spec.parquet, dsid)
                if offsets[key] != source_counts[key]:
                    raise RuntimeError(
                        f"Class {spec.name} DSID {dsid} row count changed between "
                        f"passes: expected {source_counts[key]:,}, "
                        f"found {offsets[key]:,}"
                    )
        for writer in writers.values():
            writer.finish()

        split_counts: dict[str, dict[str, int]] = {}
        for split_name in SPLIT_NAMES:
            by_class = {
                class_definition.name: writers[
                    (split_name, class_definition.name)
                ].written_rows
                for class_definition in classes
            }
            split_counts[split_name] = {
                **by_class,
                "total": sum(by_class.values()),
            }

        train_total = split_counts["train"]["total"]
        class_weights = {
            class_definition.name: train_total
            / (len(classes) * split_counts["train"][class_definition.name])
            for class_definition in classes
        }
        manifest = {
            "format_version": 1,
            "task": "H->ZZ*->4l production mode and continuum multiclass classification",
            "n_classes": len(classes),
            "classes": [
                {
                    "name": class_definition.name,
                    "label": class_definition.label,
                    "dsids": sorted(
                        {
                            dsid
                            for source in class_definition.sources
                            for dsid in source.dsids
                        }
                    ),
                    "input_parquets": [
                        str(source.parquet) for source in class_definition.sources
                    ],
                    "available_rows": class_counts[class_definition.name],
                    "dsid_rows": {
                        str(dsid): count
                        for dsid, count in sorted(
                            dsid_counts.get(class_definition.name, {}).items()
                        )
                    },
                }
                for class_definition in classes
            ],
            "selection": "all rows matching the configured DSIDs",
            "split_method": (
                "exact seeded event-level assignment independently per class and DSID"
            ),
            "seed": args.seed,
            "train_fraction": args.train_frac,
            "validation_fraction": args.val_frac,
            "test_fraction": 1.0 - args.train_frac - args.val_frac,
            "token_shape": token_shape,
            "selected_identity_count": len(seen_identities),
            "identity_overlap_detected": False,
            "split_counts": split_counts,
            "recommended_cross_entropy_weights": class_weights,
            "read_batch_size": args.read_batch_size,
            "shard_rows": args.shard_rows,
            "compression": args.compression,
            "notes": [
                "All available selected events are retained; classes are not downsampled.",
                "Class weights use the balanced inverse-frequency definition "
                "N_train / (n_classes * N_class).",
                "Metadata columns are retained for auditing but must not be model inputs.",
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
        class_summary = ", ".join(
            f"{class_definition.name}={counts[class_definition.name]:,}"
            for class_definition in classes
        )
        print(f"{split_name}: {counts['total']:,} rows ({class_summary})")
    print("recommended cross-entropy weights:")
    for class_definition in classes:
        print(
            f"  {class_definition.name}: "
            f"{class_weights[class_definition.name]:.8f}"
        )


def main() -> None:
    prepare(parse_args())


if __name__ == "__main__":
    main()
