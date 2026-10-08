"""Regression coverage for sliced nested Arrow arrays in compatibility writers."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_grouped_hzz_classification_shards import ShardWriter as ClassificationWriter  # noqa: E402
from prepare_token_parquet_pretrain_shards import ShardWriter as PretrainWriter  # noqa: E402


def nested(values: np.ndarray, dtype: pa.DataType) -> pa.Array:
    return pa.array(values.tolist(), type=pa.list_(
        pa.list_(dtype, values.shape[2]), values.shape[1]
    ))


def fixture(rows: int = 192, quantizers: int = 8) -> pa.Table:
    return pa.table({
        "tokens": nested(np.arange(rows * 3 * quantizers).reshape(rows, 3, quantizers), pa.int64()),
        "continuous_features": nested(
            np.arange(rows * 3 * 14, dtype=np.float32).reshape(rows, 3, 14), pa.float32()
        ),
        "mask": pa.array([[True, True, False]] * rows, type=pa.list_(pa.bool_(), 3)),
        "event_index": pa.array(np.arange(rows)),
        "source_file": pa.array(["gs://bucket/input.h5"] * rows),
        "labels": pa.array(np.arange(rows) % 2),
    }).replace_schema_metadata({b"test": b"preserve metadata"})


class TestParquetShardWriters(unittest.TestCase):
    def check_writer(self, cls, expected: pa.Table, batch_size: int) -> None:
        with tempfile.TemporaryDirectory() as directory:
            writer = cls(Path(directory), shard_rows=41, row_group_rows=7,
                         compression="snappy", seed=42)
            for start in range(0, len(expected), batch_size):
                writer.add(expected.slice(start, batch_size))
            writer.finish()
            parts = sorted(Path(directory).glob("*.parquet"))
            restored = pa.concat_tables([pq.read_table(path) for path in parts]).sort_by("event_index")
            # Values, labels and metadata must survive, not merely row identities.
            # The reference is already ordered. Arrow sort_by would itself call
            # take on this non-zero-offset reference and invalidate the oracle.
            self.assertTrue(restored.equals(expected))
            self.assertEqual(restored.schema.metadata, expected.schema.metadata)
            self.assertEqual(writer.written_rows, len(expected))

    def test_nonzero_offset_input_and_final_remainder(self) -> None:
        for cls in (PretrainWriter, ClassificationWriter):
            for quantizers in (1, 4, 8):
                with self.subTest(writer=cls.__module__, quantizers=quantizers):
                    self.check_writer(cls, fixture(196, quantizers).slice(3, 192), batch_size=192)

    def test_short_input_batches_preserve_nested_values(self) -> None:
        for cls in (PretrainWriter, ClassificationWriter):
            with self.subTest(writer=cls.__module__):
                self.check_writer(cls, fixture(), batch_size=7)

    def test_pretraining_cli_preserves_all_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = fixture()
            path = root / "input.parquet"
            pq.write_table(expected, path)
            result = subprocess.run([
                sys.executable, str(ROOT / "scripts/prepare_token_parquet_pretrain_shards.py"),
                "--input-parquets", str(path), "--output-dir", str(root / "prepared"),
                "--read-batch-size", "7", "--shard-rows", "41", "--seed", "42",
            ], cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
                capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            parts = sorted((root / "prepared").glob("*/part-*.parquet"))
            restored = pa.concat_tables([pq.read_table(part) for part in parts]).sort_by("event_index")
            self.assertTrue(restored.equals(expected))


if __name__ == "__main__":
    unittest.main()
