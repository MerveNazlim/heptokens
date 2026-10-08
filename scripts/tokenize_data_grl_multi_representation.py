#!/usr/bin/env python3
"""Create aligned Q1, Q8, and continuous pretraining Parquets in one H5 pass.

The converter is intentionally group-scoped.  A production campaign assigns a
deterministic list of H5 files to each group, runs one converter process per
group, and merges the uniquely named Parquet outputs in object storage.  All
three representations use the same event-level train/validation decision.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf

import tokenize_objects_to_grouped_parquet as grouped_export
import tokenize_objects_to_parquet_with_atlasopenmagic_metadata as flat_export
from heptokens.data.continuous_schema import build_continuous_schema
from heptokens.data.data_grl_conversion import DataGrlConversionModule


log = logging.getLogger(__name__)

REPRESENTATIONS = ("q1", "q8", "continuous")
SPLITS = ("train", "val")
CONTINUOUS_COLUMNS = {
    "continuous_features",
    "continuous_feature_mask",
    "position_role_ids",
    "decoded_continuous_features",
}
TOKEN_COLUMNS = {"tokens", "input_ids"}
TOKEN_VOCABULARY_KEY = b"heptokens_token_vocabulary"
CONTINUOUS_SCHEMA_KEY = b"heptokens_continuous_schema"

DEFAULT_CONVERSION_CONFIG = (
    "configs/datamodule/data_grl_q1_q8_continuous_conversion.yaml"
)


def load_conversion_module(path: str | Path) -> DataGrlConversionModule:
    """Load the offline conversion policy from its declarative YAML."""
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(config, dict):
        raise TypeError(f"Conversion config must be a mapping: {path}")
    target = config.pop("_target_", None)
    expected_target = (
        "heptokens.data.data_grl_conversion.DataGrlConversionModule"
    )
    if target != expected_target:
        raise ValueError(
            f"Conversion config target is {target!r}; expected {expected_target!r}"
        )
    return DataGrlConversionModule(**config)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--conversion-config",
        default=DEFAULT_CONVERSION_CONFIG,
    )
    config_args, _ = config_parser.parse_known_args(argv)
    conversion_module = load_conversion_module(config_args.conversion_config)

    parser = argparse.ArgumentParser(
        description=(
            "Read each H5 once and write aligned final Q1, Q8, and continuous "
            "train/validation Parquet shards."
        )
    )
    parser.add_argument(
        "--conversion-config",
        default=config_args.conversion_config,
        help=(
            "YAML defining the shared H5-to-Parquet scientific policy, including "
            "tokenizer requirements and deterministic event split."
        ),
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        required=True,
        help=(
            "JSON containing group_id and files. Each file needs local_path and "
            "source_uri; source_uri is the stable event identity."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--datamodule-config",
        default=conversion_module.datamodule_config,
    )
    parser.add_argument("--q1-tokenizer-checkpoints", nargs="+", required=True)
    parser.add_argument("--q8-tokenizer-checkpoints", nargs="+", required=True)
    parser.add_argument("--preprocess-transformers", nargs="+", required=True)
    parser.add_argument(
        "--object-order",
        nargs="+",
        default=list(conversion_module.object_order),
    )
    parser.add_argument(
        "--batch-size", type=int, default=conversion_module.batch_size
    )
    parser.add_argument("--num-events-per-file", type=int)
    parser.add_argument("--num-objects", type=int)
    parser.add_argument(
        "--max-seq-length", type=int, default=conversion_module.max_seq_length
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=conversion_module.seed)
    parser.add_argument(
        "--train-frac", type=float, default=conversion_module.train_frac
    )
    parser.add_argument(
        "--shard-rows", type=int, default=conversion_module.shard_rows
    )
    parser.add_argument(
        "--row-group-rows", type=int, default=conversion_module.row_group_rows
    )
    parser.add_argument("--compression", default=conversion_module.compression)
    parser.add_argument(
        "--event-bins", type=int, default=conversion_module.event_bins
    )
    parser.add_argument(
        "--event-token-inputs",
        nargs="+",
        default=list(conversion_module.event_token_inputs),
    )
    parser.add_argument(
        "--event-token-ranges",
        nargs="*",
        default=list(conversion_module.event_token_ranges),
    )
    parser.add_argument(
        "--pad-token-id", type=int, default=conversion_module.pad_token_id
    )
    parser.add_argument(
        "--cls-token-id", type=int, default=conversion_module.cls_token_id
    )
    parser.add_argument(
        "--sep-token-id", type=int, default=conversion_module.sep_token_id
    )
    parser.add_argument(
        "--mask-token-id", type=int, default=conversion_module.mask_token_id
    )
    parser.add_argument(
        "--allow-mc",
        action="store_true",
        help="Allow an input whose H5 metadata identifies it as simulation.",
    )
    parser.add_argument("--overwrite-group", action="store_true")

    # These attributes are consumed by the established vocabulary/assembler
    # helpers.  The data-GRL campaign always keeps CLS, event tokens, and object
    # separators enabled.
    parser.set_defaults(
        no_cls=False,
        no_event_token=False,
        no_separators=False,
        event_token_input=None,
        write_legacy_columns=False,
    )
    args = parser.parse_args(argv)
    args.conversion_module = DataGrlConversionModule(
        datamodule_config=args.datamodule_config,
        seed=args.seed,
        train_frac=args.train_frac,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        shard_rows=args.shard_rows,
        row_group_rows=args.row_group_rows,
        compression=args.compression,
        event_bins=args.event_bins,
        object_order=tuple(args.object_order),
        event_token_inputs=tuple(args.event_token_inputs),
        event_token_ranges=tuple(args.event_token_ranges),
        pad_token_id=args.pad_token_id,
        cls_token_id=args.cls_token_id,
        sep_token_id=args.sep_token_id,
        mask_token_id=args.mask_token_id,
        q1_vocab_size=conversion_module.q1_vocab_size,
        q8_vocab_size=conversion_module.q8_vocab_size,
        require_real_data=(
            conversion_module.require_real_data and not args.allow_mc
        ),
        q1_tokenizer_spec=conversion_module.q1_tokenizer_spec,
        q8_tokenizer_spec=conversion_module.q8_tokenizer_spec,
    )
    return args


@dataclass(frozen=True)
class InputFile:
    local_path: Path
    source_uri: str


def preflight_input_files(
    input_files: list[InputFile],
) -> tuple[list[InputFile], dict[str, dict | None], list[dict]]:
    """Identify unreadable HDF5 inputs before expensive tokenization starts."""
    valid_files = []
    source_counts: dict[str, dict | None] = {
        input_file.source_uri: None for input_file in input_files
    }
    invalid_sources = []
    for input_file in input_files:
        input_bytes = input_file.local_path.stat().st_size
        error = None
        if input_bytes == 0:
            error = "zero-byte input file"
        else:
            try:
                with h5py.File(input_file.local_path, "r"):
                    pass
            except OSError as exc:
                error = f"unreadable HDF5 input: {exc}"
        if error is None:
            valid_files.append(input_file)
            continue

        log.warning("Skipping %s: %s", input_file.source_uri, error)
        record = {
            "source_uri": input_file.source_uri,
            "local_input_bytes": input_bytes,
            "status": "invalid_hdf5",
            "error": error,
        }
        invalid_sources.append(record)
        source_counts[input_file.source_uri] = {
            "available": 0,
            "processed": 0,
            "train": 0,
            "val": 0,
            "local_input_bytes": input_bytes,
            "status": "invalid_hdf5",
            "error": error,
        }
    return valid_files, source_counts, invalid_sources


def load_input_manifest(path: Path) -> tuple[str, list[InputFile]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError("Input manifest must be a JSON object")
    group_id = str(payload.get("group_id", "")).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", group_id):
        raise ValueError(f"Unsafe or missing group_id: {group_id!r}")
    entries = payload.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Input manifest must contain a non-empty files list")

    files = []
    seen_sources = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise TypeError(f"files[{index}] is not a JSON object")
        local_path = Path(str(entry.get("local_path", ""))).resolve()
        source_uri = str(entry.get("source_uri", "")).strip()
        if not local_path.is_file():
            raise FileNotFoundError(local_path)
        if not source_uri:
            raise ValueError(f"files[{index}] has no source_uri")
        if source_uri in seen_sources:
            raise ValueError(f"Duplicate source_uri in input manifest: {source_uri}")
        seen_sources.add(source_uri)
        files.append(InputFile(local_path=local_path, source_uri=source_uri))
    return group_id, files


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _replace_metadata(table: pa.Table, metadata: dict[bytes, bytes]) -> pa.Table:
    return table.replace_schema_metadata(metadata)


def split_q8_and_continuous_table(
    combined: pa.Table,
    *,
    q8_vocabulary: dict,
    continuous_schema: dict,
) -> tuple[pa.Table, pa.Table]:
    """Project a combined Arrow table without copying its column buffers."""
    q8_columns = [name for name in combined.column_names if name not in CONTINUOUS_COLUMNS]
    continuous_columns = [
        name
        for name in combined.column_names
        if name not in TOKEN_COLUMNS and name != "decoded_continuous_features"
    ]

    base_metadata = dict(combined.schema.metadata or {})
    q8_metadata = dict(base_metadata)
    q8_metadata[TOKEN_VOCABULARY_KEY] = json.dumps(
        q8_vocabulary, sort_keys=True
    ).encode()
    q8_metadata.pop(CONTINUOUS_SCHEMA_KEY, None)

    continuous_metadata = dict(base_metadata)
    continuous_metadata.pop(TOKEN_VOCABULARY_KEY, None)
    continuous_metadata[CONTINUOUS_SCHEMA_KEY] = json.dumps(
        continuous_schema, sort_keys=True
    ).encode()

    return (
        _replace_metadata(combined.select(q8_columns), q8_metadata),
        _replace_metadata(combined.select(continuous_columns), continuous_metadata),
    )


def assert_aligned_tables(tables: dict[str, pa.Table]) -> None:
    row_counts = {name: table.num_rows for name, table in tables.items()}
    if len(set(row_counts.values())) != 1:
        raise ValueError(f"Representation row counts are not aligned: {row_counts}")
    reference_name = "q1" if "q1" in tables else next(iter(tables))
    reference = tables[reference_name]
    for name, table in tables.items():
        for column in ("source_file", "event_index", "mask", "type_ids"):
            if not table[column].combine_chunks().equals(
                reference[column].combine_chunks()
            ):
                raise ValueError(f"{name} {column} values are not aligned with {reference_name}")


class AlignedShardWriter:
    """Write matching part files for all representations using one permutation."""

    def __init__(
        self,
        output_dir: Path,
        *,
        group_id: str,
        split: str,
        shard_rows: int,
        row_group_rows: int,
        compression: str,
        seed: int,
        representations: tuple[str, ...] = REPRESENTATIONS,
    ) -> None:
        if not representations or len(set(representations)) != len(representations):
            raise ValueError("Representations must be non-empty and unique")
        if any(not re.fullmatch(r"[a-z][a-z0-9_]*", name) for name in representations):
            raise ValueError("Unsafe representation name")
        self.representations = representations
        self.reference = representations[0]
        self.output_dir = output_dir
        self.group_id = group_id
        self.split = split
        self.shard_rows = shard_rows
        self.row_group_rows = row_group_rows
        self.compression = compression
        self.rng = np.random.default_rng(seed)
        self.buffers = {name: [] for name in representations}
        self.buffered_rows = 0
        self.written_rows = 0
        self.shard_index = 0
        self.files = {name: [] for name in representations}
        for representation in representations:
            (output_dir / representation / split).mkdir(parents=True, exist_ok=True)

    def add(self, tables: dict[str, pa.Table]) -> None:
        if set(tables) != set(self.representations):
            raise ValueError("Table names differ from the writer representations")
        assert_aligned_tables(tables)
        rows = tables[self.reference].num_rows
        if rows == 0:
            return
        for representation in self.representations:
            self.buffers[representation].append(tables[representation])
        self.buffered_rows += rows
        self._flush_full_shards()

    def _combined(self) -> dict[str, pa.Table]:
        return {
            representation: (
                values[0] if len(values) == 1 else pa.concat_tables(values)
            )
            for representation, values in self.buffers.items()
        }

    def _flush_full_shards(self) -> None:
        if self.buffered_rows < self.shard_rows:
            return
        combined = self._combined()
        offset = 0
        while combined[self.reference].num_rows - offset >= self.shard_rows:
            self._write(
                {
                    name: table.slice(offset, self.shard_rows)
                    for name, table in combined.items()
                }
            )
            offset += self.shard_rows
        remainder = {
            name: table.slice(offset) for name, table in combined.items()
        }
        remaining_rows = remainder[self.reference].num_rows
        self.buffers = {
            name: [table] if remaining_rows else []
            for name, table in remainder.items()
        }
        self.buffered_rows = remaining_rows

    def finish(self) -> None:
        if self.buffered_rows:
            self._write(self._combined())
        self.buffers = {name: [] for name in self.representations}
        self.buffered_rows = 0

    def _write(self, tables: dict[str, pa.Table]) -> None:
        assert_aligned_tables(tables)
        rows = tables[self.reference].num_rows
        permutation = pa.array(
            self.rng.permutation(rows), type=pa.int64()
        )
        stem = f"part-{self.group_id}-{self.shard_index:05d}.parquet"
        for representation, table in tables.items():
            path = self.output_dir / representation / self.split / stem
            if path.exists():
                raise FileExistsError(path)
            # Rebase sliced columns before take: some Arrow versions mishandle
            # non-zero offsets in nested fixed-size lists. This copies at most
            # one bounded shard, never a complete input file.
            table = pa.Table.from_arrays(
                [column.combine_chunks() for column in table.columns],
                schema=table.schema,
            )
            pq.write_table(
                table.take(permutation),
                path,
                compression=self.compression,
                row_group_size=self.row_group_rows,
                use_dictionary=True,
            )
            self.files[representation].append(
                {
                    "path": str(path.relative_to(self.output_dir)),
                    "rows": rows,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        self.shard_index += 1
        self.written_rows += rows
        log.info(
            "Wrote aligned %s shard %s with %s rows",
            self.split,
            stem,
            f"{rows:,}",
        )


def encode_batch(
    *,
    handle: h5py.File,
    event_slice: slice,
    args: argparse.Namespace,
    config: dict,
    collections: dict,
    preprocessors: dict,
    q1_models: dict,
    q8_models: dict,
    device: torch.device,
) -> tuple[dict, dict]:
    q1_encoded = {}
    q8_encoded = {}
    for object_name in args.object_order:
        csts, object_mask = flat_export.read_collection_batch(
            handle,
            collections[object_name],
            event_slice,
            num_objects=args.num_objects,
            global_mask_input=config.get("mask_input"),
        )
        csts = flat_export.apply_object_preprocessing(
            csts,
            object_mask,
            preprocessors[object_name],
            object_name,
        )
        q1_indices = flat_export.encode_object_batch(
            q1_models[object_name], csts, object_mask, device
        )
        q8_indices = flat_export.encode_object_batch(
            q8_models[object_name], csts, object_mask, device
        )
        q1_encoded[object_name] = (q1_indices, object_mask)
        q8_encoded[object_name] = (q8_indices, object_mask, csts)
    return q1_encoded, q8_encoded


def convert(args: argparse.Namespace) -> dict:
    conversion_module = args.conversion_module

    group_id, input_files = load_input_manifest(args.input_manifest)
    status_dir = args.output_dir / "conversion_status"
    status_path = status_dir / f"{group_id}.json"
    success_path = status_dir / f"{group_id}.SUCCESS.txt"
    if (status_path.exists() or success_path.exists()) and not args.overwrite_group:
        raise FileExistsError(
            f"Group {group_id} already has status output; use --overwrite-group"
        )
    status_dir.mkdir(parents=True, exist_ok=True)

    config = flat_export.load_data_config(args.datamodule_config)
    collections = flat_export.collection_map(config.get("object_collections") or [])
    missing_collections = sorted(set(args.object_order) - set(collections))
    if missing_collections:
        raise ValueError(f"Datamodule config is missing collections: {missing_collections}")

    device = flat_export.choose_device(args.device)
    q1_checkpoint_map = flat_export.parse_checkpoint_map(args.q1_tokenizer_checkpoints)
    q8_checkpoint_map = flat_export.parse_checkpoint_map(args.q8_tokenizer_checkpoints)
    preprocess_map = flat_export.parse_path_map(
        args.preprocess_transformers, item_name="preprocess transformer"
    )
    q1_models = flat_export.load_models(q1_checkpoint_map, device)
    q8_models = flat_export.load_models(q8_checkpoint_map, device)
    preprocessors = flat_export.load_preprocessors(preprocess_map)
    flat_export.validate_export_inputs(
        q1_models, collections, args.object_order, preprocessors
    )
    flat_export.validate_export_inputs(
        q8_models, collections, args.object_order, preprocessors
    )
    conversion_module.validate_tokenizers(q1_models, "q1")
    conversion_module.validate_tokenizers(q8_models, "q8")

    event_range_map = flat_export.parse_event_range_map(args.event_token_ranges)
    q1_vocabulary = grouped_export.build_grouped_vocabulary(
        q1_models, args.event_token_inputs, event_range_map, args
    )
    q8_vocabulary = grouped_export.build_grouped_vocabulary(
        q8_models, args.event_token_inputs, event_range_map, args
    )
    if int(q1_vocabulary["vocab_size"]) != conversion_module.q1_vocab_size:
        raise ValueError(
            f"Q1 vocabulary is {q1_vocabulary['vocab_size']}, expected "
            f"{conversion_module.q1_vocab_size}"
        )
    if int(q8_vocabulary["vocab_size"]) != conversion_module.q8_vocab_size:
        raise ValueError(
            f"Q8 vocabulary is {q8_vocabulary['vocab_size']}, expected "
            f"{conversion_module.q8_vocab_size}"
        )
    if q1_vocabulary["event_tokens"] != q8_vocabulary["event_tokens"]:
        raise ValueError("Q1 and Q8 event-token vocabularies differ")

    continuous_schema = build_continuous_schema(
        collections=collections,
        object_order=list(q8_vocabulary["objects"]),
        event_token_specs=q8_vocabulary["event_tokens"],
        type_ids=flat_export.TYPE_IDS,
        include_decoded_q8=False,
    )
    q8_assembly_vocabulary = copy.deepcopy(q8_vocabulary)
    q8_assembly_vocabulary["continuous_schema"] = continuous_schema

    train_writer = AlignedShardWriter(
        args.output_dir,
        group_id=group_id,
        split="train",
        shard_rows=args.shard_rows,
        row_group_rows=args.row_group_rows,
        compression=args.compression,
        seed=args.seed + 1,
    )
    val_writer = AlignedShardWriter(
        args.output_dir,
        group_id=group_id,
        split="val",
        shard_rows=args.shard_rows,
        row_group_rows=args.row_group_rows,
        compression=args.compression,
        seed=args.seed + 2,
    )
    membership_digest = hashlib.sha256()
    valid_input_files, source_counts, invalid_sources = preflight_input_files(
        input_files
    )
    log.info(
        "HDF5 preflight: %s valid, %s invalid",
        len(valid_input_files),
        len(invalid_sources),
    )

    with torch.inference_mode():
        for input_file in valid_input_files:
            log.info("Processing %s", input_file.source_uri)
            with h5py.File(input_file.local_path, "r") as handle:
                full_events = flat_export.infer_n_events(handle, config)
                total_events = full_events
                if args.num_events_per_file is not None:
                    total_events = min(total_events, args.num_events_per_file)
                sample_metadata = flat_export.sample_metadata_from_handle(
                    handle,
                    input_file.source_uri,
                    metadata_source="h5",
                    atlasopenmagic_release="",
                )
                sample_metadata["source_file_events"] = full_events
                conversion_module.validate_source_metadata(
                    input_file.source_uri, sample_metadata
                )

                source_train = 0
                source_val = 0
                encoded_source = input_file.source_uri.encode("utf-8")
                membership_digest.update(len(encoded_source).to_bytes(8, "little"))
                membership_digest.update(encoded_source)

                for start in range(0, total_events, args.batch_size):
                    end = min(start + args.batch_size, total_events)
                    event_slice = slice(start, end)
                    batch_events = end - start
                    q1_encoded, q8_encoded = encode_batch(
                        handle=handle,
                        event_slice=event_slice,
                        args=args,
                        config=config,
                        collections=collections,
                        preprocessors=preprocessors,
                        q1_models=q1_models,
                        q8_models=q8_models,
                        device=device,
                    )
                    event_tokens = flat_export.read_event_token_values(
                        handle, q8_vocabulary["event_tokens"], event_slice
                    )
                    event_values = flat_export.read_normalized_event_values(
                        handle, q8_vocabulary["event_tokens"], event_slice
                    )

                    q1_tokens, q1_mask, q1_types = grouped_export.assemble_grouped_rows(
                        q1_encoded,
                        event_tokens,
                        batch_events,
                        q1_vocabulary,
                        args,
                    )
                    q8_tokens, q8_mask, q8_types, continuous_columns = (
                        grouped_export.assemble_grouped_rows(
                            q8_encoded,
                            event_tokens,
                            batch_events,
                            q8_assembly_vocabulary,
                            args,
                            event_values=event_values,
                        )
                    )
                    if not np.array_equal(q1_mask, q8_mask):
                        raise ValueError(
                            f"Q1/Q8 sequence masks differ for {input_file.source_uri} "
                            f"events {start}:{end}"
                        )
                    if not np.array_equal(q1_types, q8_types):
                        raise ValueError(
                            f"Q1/Q8 type IDs differ for {input_file.source_uri} "
                            f"events {start}:{end}"
                        )

                    q1_table = grouped_export.make_grouped_table(
                        q1_tokens,
                        q1_mask,
                        q1_types,
                        start_index=start,
                        source_file=input_file.source_uri,
                        sample_metadata=sample_metadata,
                        write_legacy_columns=False,
                        vocabulary=q1_vocabulary,
                        extra_columns=None,
                    )
                    combined_q8_table = grouped_export.make_grouped_table(
                        q8_tokens,
                        q8_mask,
                        q8_types,
                        start_index=start,
                        source_file=input_file.source_uri,
                        sample_metadata=sample_metadata,
                        write_legacy_columns=False,
                        vocabulary=q8_assembly_vocabulary,
                        extra_columns=continuous_columns,
                    )
                    q8_table, continuous_table = split_q8_and_continuous_table(
                        combined_q8_table,
                        q8_vocabulary=q8_vocabulary,
                        continuous_schema=continuous_schema,
                    )
                    tables = {
                        "q1": q1_table,
                        "q8": q8_table,
                        "continuous": continuous_table,
                    }
                    assert_aligned_tables(tables)

                    event_indices = np.arange(start, end, dtype=np.uint64)
                    train_mask = conversion_module.train_mask(
                        input_file.source_uri, event_indices
                    )
                    val_mask = ~train_mask
                    source_train += int(train_mask.sum())
                    source_val += int(val_mask.sum())
                    membership_digest.update(event_indices.tobytes())
                    membership_digest.update(train_mask.astype(np.uint8).tobytes())

                    train_writer.add(
                        {
                            name: table.filter(pa.array(train_mask))
                            for name, table in tables.items()
                        }
                    )
                    val_writer.add(
                        {
                            name: table.filter(pa.array(val_mask))
                            for name, table in tables.items()
                        }
                    )

                source_counts[input_file.source_uri] = {
                    "available": full_events,
                    "processed": total_events,
                    "train": source_train,
                    "val": source_val,
                    "local_input_bytes": input_file.local_path.stat().st_size,
                    "status": "valid",
                }

    train_writer.finish()
    val_writer.finish()
    total_rows = train_writer.written_rows + val_writer.written_rows
    if total_rows == 0:
        raise RuntimeError(
            "Conversion produced zero rows across all input H5 files; "
            "refusing to write a success marker"
        )
    if any(item is None for item in source_counts.values()):
        raise RuntimeError("HDF5 preflight left unresolved source-count entries")
    if total_rows != sum(item["processed"] for item in source_counts.values()):
        raise RuntimeError("Written rows do not match processed H5 event count")

    manifest = {
        "format_version": 1,
        "group_id": group_id,
        "representations": list(REPRESENTATIONS),
        "conversion_config": str(args.conversion_config),
        "conversion_settings": conversion_module.manifest_settings(),
        "seed": args.seed,
        "train_fraction_requested": args.train_frac,
        "split_method": conversion_module.split_method,
        "membership_sha256": membership_digest.hexdigest(),
        "max_seq_length": args.max_seq_length,
        "q1_vocab_size": q1_vocabulary["vocab_size"],
        "q1_max_quantizers": q1_vocabulary["max_quantizers"],
        "q8_vocab_size": q8_vocabulary["vocab_size"],
        "q8_max_quantizers": q8_vocabulary["max_quantizers"],
        "continuous_max_feature_dim": continuous_schema["max_feature_dim"],
        "input_file_count": len(input_files),
        "valid_input_file_count": len(valid_input_files),
        "invalid_input_file_count": len(invalid_sources),
        "invalid_sources": invalid_sources,
        "source_counts": source_counts,
        "train_rows": train_writer.written_rows,
        "val_rows": val_writer.written_rows,
        "total_rows": total_rows,
        "outputs": {
            representation: {
                "train": train_writer.files[representation],
                "val": val_writer.files[representation],
            }
            for representation in REPRESENTATIONS
        },
    }
    status_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    success_path.write_text(
        f"group_id={group_id}\n"
        f"total_rows={total_rows}\n"
        f"train_rows={train_writer.written_rows}\n"
        f"val_rows={val_writer.written_rows}\n"
        f"invalid_input_file_count={len(invalid_sources)}\n"
        f"membership_sha256={membership_digest.hexdigest()}\n"
    )
    return manifest


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    manifest = convert(parse_args())
    log.info(
        "Completed group %s: total=%s train=%s val=%s",
        manifest["group_id"],
        f"{manifest['total_rows']:,}",
        f"{manifest['train_rows']:,}",
        f"{manifest['val_rows']:,}",
    )


if __name__ == "__main__":
    main()
