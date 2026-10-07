#!/usr/bin/env python3
"""Validate q-token and continuous columns aligned in prepared Parquet shards."""

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
    "continuous_features",
    "continuous_feature_mask",
    "position_role_ids",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Parquet files or prepared directories")
    parser.add_argument("--max-files", type=int, default=12)
    parser.add_argument("--rows-per-file", type=int, default=256)
    parser.add_argument(
        "--require-decoded-q8",
        action="store_true",
        help="Require and validate decoded_continuous_features in every shard.",
    )
    return parser.parse_args()


def expand_paths(paths: list[str], max_files: int) -> list[Path]:
    files = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.parquet")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    if not files:
        raise FileNotFoundError("No Parquet files found")
    return files[:max_files]


def main() -> None:
    args = parse_args()
    files = expand_paths(args.paths, args.max_files)
    reference_schema = None
    role_counts: dict[int, int] = {}
    rows_checked = 0
    decoded_files = 0

    for path in files:
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema_arrow.names)
        required_columns = set(REQUIRED_COLUMNS)
        if args.require_decoded_q8 or "decoded_continuous_features" in names:
            required_columns.add("decoded_continuous_features")
        missing = required_columns - names
        if missing:
            raise ValueError(f"{path} is missing paired columns: {sorted(missing)}")
        metadata = parquet.schema_arrow.metadata or {}
        encoded_schema = metadata.get(b"heptokens_continuous_schema")
        if encoded_schema is None:
            raise ValueError(f"{path} has no heptokens_continuous_schema metadata")
        schema = json.loads(encoded_schema)
        if reference_schema is None:
            reference_schema = schema
        elif schema != reference_schema:
            raise ValueError(f"Continuous schema differs in {path}")

        batch = next(
            parquet.iter_batches(
                batch_size=args.rows_per_file,
                columns=sorted(required_columns),
            )
        )
        arrays = {
            name: np.asarray(
                batch.column(batch.schema.get_field_index(name)).to_pylist()
            )
            for name in required_columns
        }
        tokens = arrays["tokens"]
        sequence_mask = arrays["mask"].astype(bool)
        type_ids = arrays["type_ids"]
        features = arrays["continuous_features"].astype(np.float32)
        feature_mask = arrays["continuous_feature_mask"].astype(bool)
        roles = arrays["position_role_ids"]

        if tokens.ndim != 3 or features.ndim != 3:
            raise ValueError(f"Unexpected paired shapes in {path}: {tokens.shape}, {features.shape}")
        if tokens.shape[:2] != features.shape[:2]:
            raise ValueError(f"Token/feature sequence shapes differ in {path}")
        if sequence_mask.shape != tokens.shape[:2] or type_ids.shape != sequence_mask.shape:
            raise ValueError(f"Sequence metadata shape mismatch in {path}")
        if feature_mask.shape != features.shape or roles.shape != sequence_mask.shape:
            raise ValueError(f"Continuous mask/role shape mismatch in {path}")
        if not np.isfinite(features).all():
            raise ValueError(f"Non-finite continuous feature found in {path}")
        decoded = arrays.get("decoded_continuous_features")
        if decoded is not None:
            decoded = decoded.astype(np.float32)
            decoded_files += 1
            if decoded.shape != features.shape:
                raise ValueError(f"Raw/decoded continuous shapes differ in {path}")
            if not np.isfinite(decoded[feature_mask]).all():
                raise ValueError(f"Non-finite decoded-Q8 feature found in {path}")

        role_ids = {name: int(value) for name, value in schema["role_ids"].items()}
        if np.any(roles[~sequence_mask] != role_ids["padding"]):
            raise ValueError(f"Padded positions have non-padding roles in {path}")
        maskable = (roles == role_ids["event"]) | (roles == role_ids["object"])
        if np.any(maskable & ~feature_mask.any(axis=-1)):
            raise ValueError(f"Event/object position without continuous target in {path}")
        structural = (roles == role_ids["cls"]) | (roles == role_ids["separator"])
        if np.any(structural & feature_mask.any(axis=-1)):
            raise ValueError(f"Structural position carries continuous features in {path}")
        if decoded is not None:
            event = roles == role_ids["event"]
            if not np.allclose(decoded[event], features[event], rtol=0.0, atol=0.0):
                raise ValueError(
                    f"Decoded-Q8 and raw event-level inputs are not identical in {path}"
                )
            if np.any(decoded[~feature_mask] != 0.0):
                raise ValueError(f"Decoded-Q8 padding contains nonzero values in {path}")

        unique_roles, counts = np.unique(roles[sequence_mask], return_counts=True)
        for role, count in zip(unique_roles, counts, strict=True):
            role_counts[int(role)] = role_counts.get(int(role), 0) + int(count)
        rows_checked += batch.num_rows

    role_names = {int(value): name for name, value in reference_schema["role_ids"].items()}
    readable_counts = {
        role_names.get(role, str(role)): count for role, count in sorted(role_counts.items())
    }
    print(f"PASS: checked {rows_checked:,} rows from {len(files)} paired shards")
    print(f"max_feature_dim: {reference_schema['max_feature_dim']}")
    print(f"objects: {', '.join(reference_schema['objects'])}")
    print(f"decoded-Q8 columns: {decoded_files}/{len(files)} files")
    print(f"valid role counts: {readable_counts}")


if __name__ == "__main__":
    main()
