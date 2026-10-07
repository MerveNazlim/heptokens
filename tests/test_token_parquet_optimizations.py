"""Regression tests for Arrow conversion and continuous-column loading."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from heptokens.data.benchmark_sequence import (
    CONTINUOUS_FEATURE_MASK_KEY,
    CONTINUOUS_FEATURES_KEY,
    MASK_KEY,
    POSITION_ROLE_IDS_KEY,
    TOKENS_KEY,
    TYPE_IDS_KEY,
)
from heptokens.data.benchmark_token_parquet import (
    StreamingTokenParquetDataset,
    TokenParquetPretrainModule,
    _sequence_array_to_numpy,
)


def _nested_array(values: np.ndarray, value_type: pa.DataType) -> pa.Array:
    return pa.array(
        values.tolist(),
        type=pa.list_(
            pa.list_(value_type, values.shape[2]),
            values.shape[1],
        ),
    )


def _paired_table(
    tokens: np.ndarray,
    features: np.ndarray,
    feature_mask: np.ndarray,
    mask: np.ndarray,
    type_ids: np.ndarray,
    role_ids: np.ndarray,
) -> pa.Table:
    sequence_length = tokens.shape[1]
    table = pa.table(
        {
            "tokens": _nested_array(tokens, pa.int64()),
            "mask": pa.array(
                mask.tolist(), type=pa.list_(pa.bool_(), sequence_length)
            ),
            "type_ids": pa.array(
                type_ids.tolist(), type=pa.list_(pa.int64(), sequence_length)
            ),
            "continuous_features": _nested_array(features, pa.float32()),
            "continuous_feature_mask": _nested_array(feature_mask, pa.bool_()),
            "position_role_ids": pa.array(
                role_ids.tolist(), type=pa.list_(pa.int64(), sequence_length)
            ),
        }
    )
    metadata = {
        b"heptokens_token_vocabulary": json.dumps({"vocab_size": 32}).encode(),
        b"heptokens_continuous_schema": json.dumps(
            {
                "max_feature_dim": features.shape[2],
                "role_ids": {
                    "padding": 0,
                    "cls": 1,
                    "event": 2,
                    "object": 3,
                    "separator": 4,
                },
                "event_inputs": [{"name": "mu"}],
                "objects": {
                    "electrons": {
                        "type_id": 4,
                        "feature_count": features.shape[2],
                        "groups": {
                            "kinematics": list(range(features.shape[2]))
                        },
                    }
                },
            },
            sort_keys=True,
        ).encode(),
    }
    return table.replace_schema_metadata(metadata)


class TestTokenParquetOptimizations(unittest.TestCase):
    def test_nested_fixed_size_arrow_conversion_preserves_sliced_values(self) -> None:
        values = np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2)
        array = _nested_array(values, pa.float32()).slice(1, 2)

        converted = _sequence_array_to_numpy(array, np.dtype(np.float32))

        np.testing.assert_array_equal(converted, values[1:3])
        self.assertEqual(converted.shape, (2, 3, 2))

    def test_continuous_stream_can_prune_q8_tokens(self) -> None:
        rows, sequence_length, feature_count, quantizers = 3, 4, 3, 2
        tokens = np.arange(rows * sequence_length * quantizers, dtype=np.int64).reshape(
            rows, sequence_length, quantizers
        )
        features = np.arange(
            rows * sequence_length * feature_count, dtype=np.float32
        ).reshape(rows, sequence_length, feature_count)
        feature_mask = np.ones_like(features, dtype=bool)
        mask = np.ones((rows, sequence_length), dtype=bool)
        type_ids = np.full((rows, sequence_length), 4, dtype=np.int64)
        role_ids = np.full((rows, sequence_length), 3, dtype=np.int64)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paired.parquet"
            pq.write_table(
                _paired_table(
                    tokens, features, feature_mask, mask, type_ids, role_ids
                ),
                path,
                row_group_size=2,
            )
            dataset = StreamingTokenParquetDataset(
                parquet_files=[str(path)],
                split_start=0.0,
                split_end=1.0,
                stream_batch_size=2,
                shuffle=False,
                require_continuous=True,
                include_tokens=False,
            )

            samples = list(dataset)

        self.assertEqual(len(samples), rows)
        self.assertTrue(all(TOKENS_KEY not in sample for sample in samples))
        np.testing.assert_array_equal(
            samples[1][CONTINUOUS_FEATURES_KEY].numpy(), features[1]
        )
        np.testing.assert_array_equal(
            samples[1][CONTINUOUS_FEATURE_MASK_KEY].numpy(), feature_mask[1]
        )
        np.testing.assert_array_equal(samples[1][MASK_KEY].numpy(), mask[1])
        np.testing.assert_array_equal(samples[1][TYPE_IDS_KEY].numpy(), type_ids[1])
        np.testing.assert_array_equal(
            samples[1][POSITION_ROLE_IDS_KEY].numpy(), role_ids[1]
        )

    def test_continuous_stream_does_not_require_a_token_column(self) -> None:
        rows, sequence_length, feature_count = 3, 4, 3
        features = np.arange(
            rows * sequence_length * feature_count, dtype=np.float32
        ).reshape(rows, sequence_length, feature_count)
        feature_mask = np.ones_like(features, dtype=bool)
        mask = np.ones((rows, sequence_length), dtype=bool)
        type_ids = np.full((rows, sequence_length), 4, dtype=np.int64)
        role_ids = np.full((rows, sequence_length), 3, dtype=np.int64)
        table = pa.table(
            {
                "mask": pa.array(
                    mask.tolist(), type=pa.list_(pa.bool_(), sequence_length)
                ),
                "type_ids": pa.array(
                    type_ids.tolist(), type=pa.list_(pa.int64(), sequence_length)
                ),
                "continuous_features": _nested_array(features, pa.float32()),
                "continuous_feature_mask": _nested_array(feature_mask, pa.bool_()),
                "position_role_ids": pa.array(
                    role_ids.tolist(),
                    type=pa.list_(pa.int64(), sequence_length),
                ),
            }
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "continuous-only.parquet"
            pq.write_table(table, path, row_group_size=2)
            dataset = StreamingTokenParquetDataset(
                parquet_files=[str(path)],
                split_start=0.0,
                split_end=1.0,
                stream_batch_size=2,
                shuffle=False,
                require_continuous=True,
                include_tokens=False,
            )

            samples = list(dataset)

        self.assertEqual(len(samples), rows)
        self.assertTrue(all(TOKENS_KEY not in sample for sample in samples))
        np.testing.assert_array_equal(
            samples[2][CONTINUOUS_FEATURES_KEY].numpy(), features[2]
        )

    def test_pretrain_module_propagates_continuous_token_pruning(self) -> None:
        rows, sequence_length, feature_count, quantizers = 4, 4, 3, 2
        tokens = np.arange(rows * sequence_length * quantizers, dtype=np.int64).reshape(
            rows, sequence_length, quantizers
        )
        features = np.arange(
            rows * sequence_length * feature_count, dtype=np.float32
        ).reshape(rows, sequence_length, feature_count)
        feature_mask = np.ones_like(features, dtype=bool)
        mask = np.ones((rows, sequence_length), dtype=bool)
        type_ids = np.full((rows, sequence_length), 4, dtype=np.int64)
        role_ids = np.full((rows, sequence_length), 3, dtype=np.int64)
        table = _paired_table(
            tokens, features, feature_mask, mask, type_ids, role_ids
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train").mkdir()
            (root / "val").mkdir()
            pq.write_table(table, root / "train" / "part-00000.parquet")
            pq.write_table(table, root / "val" / "part-00000.parquet")
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "total_rows": rows * 2,
                        "train_rows": rows,
                        "val_rows": rows,
                        "source_counts": {},
                    }
                )
            )
            module = TokenParquetPretrainModule(
                prepared_dir=str(root),
                require_continuous=True,
                continuous_feature_column="continuous_features",
                include_tokens=False,
                n_classes=1,
                num_workers=0,
                batch_size=2,
                shuffle_buffer_size=0,
            )
            sample = module.get_data_sample()

        self.assertFalse(module.train_set.include_tokens)
        self.assertNotIn(TOKENS_KEY, sample)
        self.assertIn(CONTINUOUS_FEATURES_KEY, sample)

if __name__ == "__main__":
    unittest.main()
