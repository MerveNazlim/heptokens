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


if __name__ == "__main__":
    unittest.main()
