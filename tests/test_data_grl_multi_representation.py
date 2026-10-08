"""Tests for aligned Q1/Q8/continuous data-GRL export helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from tokenize_data_grl_multi_representation import (  # noqa: E402
    AlignedShardWriter,
    CONTINUOUS_SCHEMA_KEY,
    InputFile,
    TOKEN_VOCABULARY_KEY,
    preflight_input_files,
    split_q8_and_continuous_table,
)


def _nested(values: np.ndarray, value_type: pa.DataType) -> pa.Array:
    if values.ndim == 2:
        return pa.array(values.tolist(), type=pa.list_(value_type, values.shape[1]))
    return pa.array(
        values.tolist(),
        type=pa.list_(
            pa.list_(value_type, values.shape[2]),
            values.shape[1],
        ),
    )


def _combined_table(rows: int = 5) -> pa.Table:
    sequence_length = 3
    tokens = np.arange(rows * sequence_length * 8, dtype=np.int64).reshape(
        rows, sequence_length, 8
    )
    mask = np.ones((rows, sequence_length), dtype=bool)
    type_ids = np.full((rows, sequence_length), 4, dtype=np.int64)
    features = np.arange(rows * sequence_length * 2, dtype=np.float32).reshape(
        rows, sequence_length, 2
    )
    table = pa.table(
        {
            "tokens": _nested(tokens, pa.int64()),
            "mask": _nested(mask, pa.bool_()),
            "type_ids": _nested(type_ids, pa.int64()),
            "event_index": pa.array(np.arange(rows, dtype=np.int64)),
            "source_file": pa.array(["gs://bucket/input.h5"] * rows),
            "continuous_features": _nested(features, pa.float32()),
            "continuous_feature_mask": _nested(
                np.ones_like(features, dtype=bool), pa.bool_()
            ),
            "position_role_ids": _nested(
                np.full((rows, sequence_length), 3, dtype=np.int64), pa.int64()
            ),
        }
    )
    return table.replace_schema_metadata(
        {
            TOKEN_VOCABULARY_KEY: json.dumps({"vocab_size": 131588}).encode(),
            CONTINUOUS_SCHEMA_KEY: json.dumps({"max_feature_dim": 2}).encode(),
        }
    )


def _q1_table(reference: pa.Table) -> pa.Table:
    rows = reference.num_rows
    sequence_length = 3
    tokens = np.arange(rows * sequence_length, dtype=np.int64).reshape(
        rows, sequence_length, 1
    )
    return pa.table(
        {
            "tokens": _nested(tokens, pa.int64()),
            "mask": reference["mask"],
            "type_ids": reference["type_ids"],
            "event_index": reference["event_index"],
            "source_file": reference["source_file"],
        }
    ).replace_schema_metadata(
        {TOKEN_VOCABULARY_KEY: json.dumps({"vocab_size": 98820}).encode()}
    )


class TestDataGrlMultiRepresentation(unittest.TestCase):
    def test_q4_only_writer_preserves_all_rows_schema_and_metadata(self) -> None:
        reference = _combined_table(rows=13)
        tokens = np.arange(13 * 3 * 4, dtype=np.int64).reshape(13, 3, 4)
        q4 = _q1_table(reference).set_column(
            0, "tokens", _nested(tokens, pa.int64())
        ).replace_schema_metadata(
            {TOKEN_VOCABULARY_KEY: b'{"vocab_size":2048,"max_quantizers":4}'}
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = AlignedShardWriter(
                root, group_id="group-00000", split="train", shard_rows=3,
                row_group_rows=2, compression="snappy", seed=43,
                representations=("q4",),
            )
            writer.add({"q4": q4.slice(0, 2)})
            writer.add({"q4": q4.slice(2)})
            writer.finish()
            self.assertEqual(writer.written_rows, 13)
            self.assertEqual(writer.shard_index, 5)
            self.assertEqual(set(writer.files), {"q4"})
            self.assertFalse((root / "q1").exists())
            parts = [pq.read_table(root / entry["path"]) for entry in writer.files["q4"]]
            self.assertEqual([part.num_rows for part in parts], [3, 3, 3, 3, 1])
            for part in parts:
                # Parquet normalizes nested child names from item to element.
                self.assertTrue(part.schema.equals(q4.schema))
                self.assertEqual(part.schema.metadata, q4.schema.metadata)
            actual = pq.read_table(root / "q4/train").sort_by("event_index")
            self.assertTrue(actual.equals(q4))

    def test_writer_rejects_missing_or_unexpected_representations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            writer = AlignedShardWriter(
                Path(directory), group_id="group-00000", split="train",
                shard_rows=3, row_group_rows=2, compression="snappy", seed=43,
                representations=("q4",),
            )
            with self.assertRaisesRegex(ValueError, "Table names"):
                writer.add({"q1": _q1_table(_combined_table())})
            self.assertEqual(writer.written_rows, 0)

    def test_writer_rejects_empty_duplicate_or_unsafe_representation_names(self) -> None:
        for names in ((), ("q4", "q4"), ("../q4",)):
            with self.subTest(representations=names), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    AlignedShardWriter(
                        Path(directory), group_id="group-00000", split="train",
                        shard_rows=3, row_group_rows=2, compression="snappy", seed=43,
                        representations=names,
                    )

    def test_preflight_records_zero_byte_and_unreadable_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid.h5"
            empty = root / "empty.h5"
            corrupt = root / "corrupt.h5"
            import h5py

            with h5py.File(valid, "w") as handle:
                handle.create_dataset("events", data=np.arange(2))
            empty.touch()
            corrupt.write_text("not HDF5")

            inputs = [
                InputFile(valid, "gs://bucket/valid.h5"),
                InputFile(empty, "gs://bucket/empty.h5"),
                InputFile(corrupt, "gs://bucket/corrupt.h5"),
            ]
            valid_inputs, counts, invalid = preflight_input_files(inputs)

            self.assertEqual(valid_inputs, [inputs[0]])
            self.assertIsNone(counts[inputs[0].source_uri])
            self.assertEqual(len(invalid), 2)
            self.assertEqual(counts[inputs[1].source_uri]["status"], "invalid_hdf5")
            self.assertEqual(counts[inputs[2].source_uri]["status"], "invalid_hdf5")

    def test_q8_and_continuous_projection_prunes_unused_columns(self) -> None:
        combined = _combined_table()
        q8, continuous = split_q8_and_continuous_table(
            combined,
            q8_vocabulary={"vocab_size": 131588, "max_quantizers": 8},
            continuous_schema={"max_feature_dim": 2},
        )

        self.assertIn("tokens", q8.column_names)
        self.assertNotIn("continuous_features", q8.column_names)
        self.assertNotIn(CONTINUOUS_SCHEMA_KEY, q8.schema.metadata)
        self.assertIn(TOKEN_VOCABULARY_KEY, q8.schema.metadata)

        self.assertNotIn("tokens", continuous.column_names)
        self.assertIn("continuous_features", continuous.column_names)
        self.assertNotIn(TOKEN_VOCABULARY_KEY, continuous.schema.metadata)
        self.assertIn(CONTINUOUS_SCHEMA_KEY, continuous.schema.metadata)

    def test_aligned_writer_uses_identical_membership_and_part_boundaries(self) -> None:
        combined = _combined_table()
        q8, continuous = split_q8_and_continuous_table(
            combined,
            q8_vocabulary={"vocab_size": 131588, "max_quantizers": 8},
            continuous_schema={"max_feature_dim": 2},
        )
        q1 = _q1_table(q8)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = AlignedShardWriter(
                root,
                group_id="group-00000",
                split="train",
                shard_rows=3,
                row_group_rows=2,
                compression="snappy",
                seed=43,
            )
            writer.add({"q1": q1, "q8": q8, "continuous": continuous})
            writer.finish()

            self.assertEqual(writer.written_rows, 5)
            self.assertEqual(writer.shard_index, 2)
            for part in range(2):
                name = f"part-group-00000-{part:05d}.parquet"
                identity_tables = []
                for representation in ("q1", "q8", "continuous"):
                    path = root / representation / "train" / name
                    self.assertTrue(path.is_file())
                    identity_tables.append(
                        pq.read_table(path, columns=["source_file", "event_index"])
                    )
                self.assertTrue(identity_tables[0].equals(identity_tables[1]))
                self.assertTrue(identity_tables[0].equals(identity_tables[2]))

            # Check actual code/feature values, not only row identities/counts.
            # This covers non-zero-offset full shards and the final remainder.
            for representation, expected in (
                ("q1", q1), ("q8", q8), ("continuous", continuous)
            ):
                actual = pq.read_table(root / representation / "train").sort_by("event_index")
                self.assertTrue(actual.equals(expected))
                self.assertEqual(actual.schema.metadata, expected.schema.metadata)


if __name__ == "__main__":
    unittest.main()
