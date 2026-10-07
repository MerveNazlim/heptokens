#!/usr/bin/env python3
"""Full shard, vocabulary, feature-alignment, and event-leakage audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


IDENTITY_DTYPE = np.dtype([("first", "<u8"), ("second", "<u8")])
UINT64_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain-dir", type=Path)
    parser.add_argument("--classification-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("benchmark_data_audit.json"))
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument(
        "--rows-per-shard",
        type=int,
        default=128,
        help=(
            "Token/continuous rows validated per shard; metadata and identities "
            "are always complete."
        ),
    )
    parser.add_argument(
        "--full-row-validation",
        action="store_true",
        help="Validate token and continuous contents for every row; potentially very expensive.",
    )
    parser.add_argument("--temp-dir", type=Path)
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument(
        "--allow-pretrain-downstream-overlap",
        action="store_true",
        help=(
            "Report but do not fail when pretraining train rows occur in downstream "
            "validation/test. Within-dataset split leakage always fails."
        ),
    )
    args = parser.parse_args()
    if args.pretrain_dir is None and args.classification_dir is None:
        parser.error("Provide --pretrain-dir and/or --classification-dir")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.rows_per_shard < 1:
        parser.error("--rows-per-shard must be positive")
    return args


def split_files(root: Path, dataset: str) -> dict[str, list[Path]]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing prepared manifest: {manifest_path}")
    split_names = ("train", "val") if dataset == "pretrain" else ("train", "val", "test")
    result = {}
    for split in split_names:
        files = sorted((root / split).rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No {dataset} {split} Parquet shards under {root}")
        result[f"{dataset}_{split}"] = files
    return result


def decode_metadata(path: Path, key: bytes) -> dict | None:
    metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
    encoded = metadata.get(key)
    return json.loads(encoded) if encoded is not None else None


def validate_metadata_contract(vocabulary: dict, continuous_schema: dict, path: Path) -> None:
    special = {name: int(value) for name, value in vocabulary["special_tokens"].items()}
    if len(set(special.values())) != len(special):
        raise ValueError(f"Special token IDs are not unique in {path}")
    vocab_size = int(vocabulary["vocab_size"])
    ranges = []
    for index, spec in enumerate(vocabulary.get("event_tokens", [])):
        ranges.append((f"event_{index}", int(spec["base"]), int(spec["size"])))
    object_types = set()
    max_quantizers = 1
    for object_name, spec in vocabulary.get("objects", {}).items():
        type_id = int(spec["type_id"])
        if type_id in object_types:
            raise ValueError(f"Duplicate object type ID {type_id} in {path}")
        object_types.add(type_id)
        quantizers = spec["quantizers"]
        if int(spec["num_quantizers"]) != len(quantizers):
            raise ValueError(f"Quantizer count mismatch for {object_name} in {path}")
        max_quantizers = max(max_quantizers, len(quantizers))
        for index, quantizer in enumerate(quantizers):
            if int(quantizer["index"]) != index:
                raise ValueError(f"Non-sequential quantizer metadata for {object_name}")
            ranges.append(
                (
                    f"{object_name}_q{index}",
                    int(quantizer["base"]),
                    int(quantizer["size"]),
                )
            )
    ordered = sorted(ranges, key=lambda item: item[1])
    previous_end = max(special.values()) + 1
    for name, base, size in ordered:
        if size < 1 or base != previous_end or base + size > vocab_size:
            raise ValueError(
                f"Illegal or overlapping vocabulary range {name}=[{base}, {base + size}) "
                f"in {path}"
            )
        previous_end = base + size
    if previous_end != vocab_size:
        raise ValueError(
            f"Vocabulary ranges end at {previous_end}, but vocab_size={vocab_size} in {path}"
        )
    if int(vocabulary.get("max_quantizers", max_quantizers)) != max_quantizers:
        raise ValueError(f"max_quantizers metadata is inconsistent in {path}")

    columns = continuous_schema.get("feature_columns", {})
    expected_columns = {
        "raw": "continuous_features",
        "decoded_q8": "decoded_continuous_features",
        "mask": "continuous_feature_mask",
        "roles": "position_role_ids",
    }
    if any(columns.get(key) != value for key, value in expected_columns.items()):
        raise ValueError(f"Continuous feature-column metadata is incomplete in {path}")
    if continuous_schema.get("decoded_q8") is None:
        raise ValueError(f"Decoded-Q8 provenance metadata is missing in {path}")
    if set(continuous_schema.get("objects", {})) != set(vocabulary.get("objects", {})):
        raise ValueError(f"Continuous and token object sets differ in {path}")
    for object_name, spec in continuous_schema["objects"].items():
        if int(spec["type_id"]) != int(vocabulary["objects"][object_name]["type_id"]):
            raise ValueError(f"Type ID differs for {object_name} in {path}")


def stable_source_hash(value: str, person: bytes) -> np.uint64:
    digest = hashlib.blake2b(
        value.encode("utf-8"), digest_size=8, person=person
    ).digest()
    return np.uint64(int.from_bytes(digest, "little"))


def splitmix64(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.uint64, copy=True)
    values = (values + np.uint64(0x9E3779B97F4A7C15)) & UINT64_MASK
    values = (
        (values ^ (values >> np.uint64(30)))
        * np.uint64(0xBF58476D1CE4E5B9)
    ) & UINT64_MASK
    values = (
        (values ^ (values >> np.uint64(27)))
        * np.uint64(0x94D049BB133111EB)
    ) & UINT64_MASK
    return values ^ (values >> np.uint64(31))


def identity_hashes(
    sources: np.ndarray,
    events: np.ndarray,
    source_hashes: dict[str, tuple[np.uint64, np.uint64]],
    hash_owners: dict[tuple[int, int], str],
) -> np.ndarray:
    first_sources = np.empty(len(sources), dtype=np.uint64)
    second_sources = np.empty(len(sources), dtype=np.uint64)
    for source in np.unique(sources):
        source_text = str(source)
        pair = source_hashes.get(source_text)
        if pair is None:
            pair = (
                stable_source_hash(source_text, b"heptok-a"),
                stable_source_hash(source_text, b"heptok-b"),
            )
            owner = hash_owners.setdefault((int(pair[0]), int(pair[1])), source_text)
            if owner != source_text:
                raise RuntimeError(
                    f"Source-name hash collision between {owner!r} and {source_text!r}"
                )
            source_hashes[source_text] = pair
        selected = sources == source
        first_sources[selected] = pair[0]
        second_sources[selected] = pair[1]
    event_values = events.astype(np.uint64, copy=False)
    identities = np.empty(len(events), dtype=IDENTITY_DTYPE)
    identities["first"] = splitmix64(first_sources ^ event_values)
    identities["second"] = splitmix64(
        second_sources ^ event_values ^ np.uint64(0xD6E8FEB86659FD93)
    )
    return identities


def validate_token_rows(
    *,
    path: Path,
    tokens: np.ndarray,
    mask: np.ndarray,
    type_ids: np.ndarray,
    vocabulary: dict,
) -> None:
    if tokens.ndim != 3 or mask.shape != tokens.shape[:2] or type_ids.shape != mask.shape:
        raise ValueError(
            f"Grouped token shapes are inconsistent in {path}: "
            f"tokens={tokens.shape}, mask={mask.shape}, types={type_ids.shape}"
        )
    if tokens.shape[2] != int(vocabulary["max_quantizers"]):
        raise ValueError(f"Stored quantizer width differs from metadata in {path}")
    pad = int(vocabulary["special_tokens"]["pad"])
    if np.any(tokens < 0) or np.any(tokens >= int(vocabulary["vocab_size"])):
        raise ValueError(f"Out-of-vocabulary token ID found in {path}")
    if np.any(tokens[~mask] != pad) or np.any(type_ids[~mask] != 0):
        raise ValueError(f"Padded positions contain active token/type values in {path}")

    active = mask
    known = np.zeros(mask.shape, dtype=bool)
    structural = active & (type_ids == 0)
    legal_structural = {
        int(vocabulary["special_tokens"]["cls"]),
        int(vocabulary["special_tokens"]["sep"]),
    }
    if structural.any():
        if not np.isin(tokens[..., 0][structural], list(legal_structural)).all():
            raise ValueError(f"Illegal structural token found in {path}")
        if np.any(tokens[..., 1:][structural] != pad):
            raise ValueError(f"Structural position contains residual codes in {path}")
        known |= structural

    event_type = int(vocabulary["event"]["type_id"])
    event = active & (type_ids == event_type)
    event_specs = vocabulary.get("event_tokens", [])
    if event.any():
        event_ids = tokens[..., 0][event]
        legal = np.zeros(len(event_ids), dtype=bool)
        for spec in event_specs:
            base, size = int(spec["base"]), int(spec["size"])
            legal |= (event_ids >= base) & (event_ids < base + size)
        if not legal.all() or np.any(tokens[..., 1:][event] != pad):
            raise ValueError(f"Illegal event-input token group found in {path}")
        known |= event

    for spec in vocabulary.get("objects", {}).values():
        selected = active & (type_ids == int(spec["type_id"]))
        if not selected.any():
            continue
        quantizers = spec["quantizers"]
        for quantizer_index, quantizer in enumerate(quantizers):
            values = tokens[..., quantizer_index][selected]
            base, size = int(quantizer["base"]), int(quantizer["size"])
            if np.any(values < base) or np.any(values >= base + size):
                raise ValueError(
                    f"Illegal object q{quantizer_index} code found in {path}"
                )
        if np.any(tokens[..., len(quantizers) :][selected] != pad):
            raise ValueError(f"Object position has extra residual codes in {path}")
        known |= selected
    if np.any(active & ~known):
        unknown_types = np.unique(type_ids[active & ~known]).tolist()
        raise ValueError(f"Unknown active type IDs {unknown_types} in {path}")


def validate_continuous_rows(
    *,
    path: Path,
    raw: np.ndarray,
    decoded: np.ndarray,
    feature_mask: np.ndarray,
    roles: np.ndarray,
    sequence_mask: np.ndarray,
    schema: dict,
) -> None:
    if raw.shape != decoded.shape or raw.shape != feature_mask.shape:
        raise ValueError(f"Raw/decoded/mask shapes differ in {path}")
    if roles.shape != sequence_mask.shape or raw.shape[:2] != roles.shape:
        raise ValueError(f"Continuous sequence alignment differs in {path}")
    if not np.isfinite(raw[feature_mask]).all() or not np.isfinite(
        decoded[feature_mask]
    ).all():
        raise ValueError(f"Non-finite raw or decoded continuous feature found in {path}")
    if np.any(raw[~feature_mask] != 0.0) or np.any(decoded[~feature_mask] != 0.0):
        raise ValueError(f"Continuous feature padding is nonzero in {path}")
    role_ids = {key: int(value) for key, value in schema["role_ids"].items()}
    if np.any(roles[~sequence_mask] != role_ids["padding"]):
        raise ValueError(f"Sequence padding has non-padding roles in {path}")
    event = roles == role_ids["event"]
    objects = roles == role_ids["object"]
    if np.any((event | objects) & ~feature_mask.any(axis=-1)):
        raise ValueError(f"Maskable position has no continuous target in {path}")
    structural = (roles == role_ids["cls"]) | (roles == role_ids["separator"])
    if np.any(structural & feature_mask.any(axis=-1)):
        raise ValueError(f"Structural position carries continuous targets in {path}")
    if not np.array_equal(raw[event], decoded[event]):
        raise ValueError(f"Raw and decoded-Q8 event inputs differ in {path}")


def scan_split(
    *,
    name: str,
    files: list[Path],
    temp_root: Path,
    batch_size: int,
    rows_per_shard: int,
    full_row_validation: bool,
    canonical: dict,
    source_hashes: dict[str, tuple[np.uint64, np.uint64]],
    hash_owners: dict[tuple[int, int], str],
) -> tuple[np.memmap, dict]:
    rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    mmap_path = temp_root / f"{name}.identities.bin"
    identities = np.memmap(mmap_path, mode="w+", dtype=IDENTITY_DTYPE, shape=(rows,))
    offset = 0
    validated_rows = 0
    for file_index, path in enumerate(files):
        parquet = pq.ParquetFile(path)
        vocabulary = decode_metadata(path, b"heptokens_token_vocabulary")
        continuous_schema = decode_metadata(path, b"heptokens_continuous_schema")
        if vocabulary is None:
            raise ValueError(f"Missing token-vocabulary metadata in {path}")
        if continuous_schema is None:
            raise ValueError(f"Missing continuous-schema metadata in {path}")
        validate_metadata_contract(vocabulary, continuous_schema, path)
        if canonical.setdefault("vocabulary", vocabulary) != vocabulary:
            raise ValueError(f"Token-vocabulary metadata differs in {path}")
        if canonical.setdefault("continuous_schema", continuous_schema) != continuous_schema:
            raise ValueError(f"Continuous-schema metadata differs in {path}")
        required = {
            "source_file",
            "event_index",
            "tokens",
            "mask",
            "type_ids",
            "continuous_features",
            "decoded_continuous_features",
            "continuous_feature_mask",
            "position_role_ids",
        }
        missing = required - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{path} is missing benchmark columns: {sorted(missing)}")
        rows_to_validate = (
            parquet.metadata.num_rows
            if full_row_validation
            else min(rows_per_shard, parquet.metadata.num_rows)
        )
        file_validated = 0
        for batch in parquet.iter_batches(
            batch_size=min(batch_size, max(1, rows_to_validate)),
            columns=sorted(required - {"source_file", "event_index"}),
        ):
            if file_validated >= rows_to_validate:
                break
            if batch.num_rows > rows_to_validate - file_validated:
                batch = batch.slice(0, rows_to_validate - file_validated)
            arrays = {
                column: np.asarray(
                    batch.column(batch.schema.get_field_index(column)).to_pylist()
                )
                for column in required - {"source_file", "event_index"}
            }
            token_values = arrays["tokens"].astype(np.int64)
            sequence_mask = arrays["mask"].astype(bool)
            validate_token_rows(
                path=path,
                tokens=token_values,
                mask=sequence_mask,
                type_ids=arrays["type_ids"].astype(np.int64),
                vocabulary=vocabulary,
            )
            validate_continuous_rows(
                path=path,
                raw=arrays["continuous_features"].astype(np.float32),
                decoded=arrays["decoded_continuous_features"].astype(np.float32),
                feature_mask=arrays["continuous_feature_mask"].astype(bool),
                roles=arrays["position_role_ids"].astype(np.int64),
                sequence_mask=sequence_mask,
                schema=continuous_schema,
            )
            file_validated += batch.num_rows
            validated_rows += batch.num_rows

        for batch in parquet.iter_batches(
            batch_size=batch_size, columns=["source_file", "event_index"]
        ):
            sources = np.asarray(
                batch.column(batch.schema.get_field_index("source_file")).to_pylist(),
                dtype=object,
            )
            events = np.asarray(
                batch.column(batch.schema.get_field_index("event_index")).to_numpy(
                    zero_copy_only=False
                ),
                dtype=np.int64,
            )
            batch_identities = identity_hashes(
                sources,
                events,
                source_hashes,
                hash_owners,
            )
            identities[offset : offset + len(batch_identities)] = batch_identities
            offset += len(batch_identities)
        print(f"{name}: validated shard {file_index + 1}/{len(files)}: {path.name}")
    if offset != rows:
        raise RuntimeError(f"Identity row count mismatch for {name}: {offset} != {rows}")
    identities.flush()
    identities.sort(order=("first", "second"))
    duplicates = int(
        np.count_nonzero(
            (identities[1:]["first"] == identities[:-1]["first"])
            & (identities[1:]["second"] == identities[:-1]["second"])
        )
    )
    return identities, {
        "rows": rows,
        "shards": len(files),
        "content_rows_validated": validated_rows,
        "full_row_validation": full_row_validation,
        "duplicate_identities": duplicates,
    }


def overlap_count(left: np.ndarray, right: np.ndarray, chunk_size: int = 1_000_000) -> int:
    if len(left) > len(right):
        left, right = right, left
    total = 0
    for start in range(0, len(left), chunk_size):
        chunk = left[start : start + chunk_size]
        positions = np.searchsorted(right, chunk)
        in_bounds = positions < len(right)
        positions = positions[in_bounds]
        selected = chunk[in_bounds]
        total += int(
            np.count_nonzero(
                (right[positions]["first"] == selected["first"])
                & (right[positions]["second"] == selected["second"])
            )
        )
    return total


def main() -> None:
    args = parse_args()
    roots = []
    split_map = {}
    manifests = {}
    for dataset, root in (
        ("pretrain", args.pretrain_dir),
        ("classification", args.classification_dir),
    ):
        if root is None:
            continue
        root = root.resolve()
        roots.append(root)
        manifests[dataset] = json.loads((root / "manifest.json").read_text())
        split_map.update(split_files(root, dataset))

    temp_parent = args.temp_dir.resolve() if args.temp_dir else None
    if temp_parent is not None:
        temp_parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix="heptokens-audit-", dir=temp_parent))
    canonical: dict = {}
    source_hashes: dict[str, tuple[np.uint64, np.uint64]] = {}
    hash_owners: dict[tuple[int, int], str] = {}
    arrays = {}
    split_reports = {}
    try:
        for name, files in split_map.items():
            arrays[name], split_reports[name] = scan_split(
                name=name,
                files=files,
                temp_root=temp_root,
                batch_size=args.batch_size,
                rows_per_shard=args.rows_per_shard,
                full_row_validation=args.full_row_validation,
                canonical=canonical,
                source_hashes=source_hashes,
                hash_owners=hash_owners,
            )

        overlaps = {}
        for left_name, right_name in combinations(arrays, 2):
            count = overlap_count(arrays[left_name], arrays[right_name])
            overlaps[f"{left_name}__{right_name}"] = count

        hard_failures = []
        for name, report in split_reports.items():
            if report["duplicate_identities"]:
                hard_failures.append(f"{name} contains duplicate identities")
        for dataset in ("pretrain", "classification"):
            names = [name for name in arrays if name.startswith(f"{dataset}_")]
            for left_name, right_name in combinations(names, 2):
                key = f"{left_name}__{right_name}"
                if overlaps[key]:
                    hard_failures.append(f"{key} has {overlaps[key]} overlapping rows")

        cross_leakage = 0
        for downstream_split in ("classification_val", "classification_test"):
            key = f"pretrain_train__{downstream_split}"
            if key in overlaps:
                cross_leakage += overlaps[key]
        if cross_leakage and not args.allow_pretrain_downstream_overlap:
            hard_failures.append(
                "pretrain_train overlaps downstream validation/test; rerun with "
                "--allow-pretrain-downstream-overlap only if this is deliberate"
            )

        report = {
            "status": "FAIL" if hard_failures else "PASS",
            "prepared_roots": [str(root) for root in roots],
            "manifests": manifests,
            "splits": split_reports,
            "overlaps": overlaps,
            "pretrain_downstream_validation_test_overlap": cross_leakage,
            "pretrain_downstream_overlap_allowed": args.allow_pretrain_downstream_overlap,
            "unique_source_files": len(source_hashes),
            "token_vocabulary": canonical.get("vocabulary"),
            "continuous_schema": canonical.get("continuous_schema"),
            "failures": hard_failures,
            "identity_note": (
                "Row identities use two independent 64-bit hashes of the exact "
                "(source_file, event_index) pair; source-name hash collisions are checked."
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"{report['status']}: wrote audit to {args.output}")
        for key, count in overlaps.items():
            print(f"{key}: {count:,} overlapping identities")
        if hard_failures:
            raise SystemExit("; ".join(hard_failures))
    finally:
        for array in arrays.values():
            array.flush()
        arrays.clear()
        if args.keep_temp:
            print(f"Kept identity files in {temp_root}")
        else:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
