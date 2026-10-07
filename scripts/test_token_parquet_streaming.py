"""Regression checks for bounded-memory Parquet streaming."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import DataLoader

from heptokens.data.sequence import TOKENS_KEY
from heptokens.data.token_parquet import StreamingTokenParquetDataset, _bounded_shuffle


def _write_parquet(path: Path, values: list[int], *, row_group_size: int = 3) -> None:
    tokens = pa.array([[value, value + 1000] for value in values], type=pa.list_(pa.int64(), 2))
    masks = pa.array([[True, True] for _ in values], type=pa.list_(pa.bool_(), 2))
    types = pa.array([[0, 1] for _ in values], type=pa.list_(pa.int64(), 2))
    pq.write_table(
        pa.table({"tokens": tokens, "mask": masks, "type_ids": types}),
        path,
        row_group_size=row_group_size,
    )


class TestStreamingTokenParquetDataset(unittest.TestCase):
    def test_bounded_shuffle_preserves_coverage_and_mixes_grouped_input(self) -> None:
        grouped = list(range(40)) + list(range(100, 140)) + list(range(200, 240))
        first = list(
            _bounded_shuffle(
                iter(grouped),
                buffer_size=32,
                rng=np.random.default_rng(42),
            )
        )
        second = list(
            _bounded_shuffle(
                iter(grouped),
                buffer_size=32,
                rng=np.random.default_rng(42),
            )
        )

        self.assertEqual(sorted(first), sorted(grouped))
        self.assertEqual(len(first), len(set(first)))
        self.assertEqual(first, second)
        batches = [first[start : start + 16] for start in range(0, len(first), 16)]
        mixed_batches = sum(len({value // 100 for value in batch}) > 1 for batch in batches)
        self.assertGreater(mixed_batches, 0)

    def test_multiple_workers_yield_every_row_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            expected = []
            for file_index in range(5):
                values = list(range(file_index * 12, (file_index + 1) * 12))
                path = root / f"part-{file_index:02d}.parquet"
                _write_parquet(path, values)
                paths.append(str(path))
                expected.extend(values)

            dataset = StreamingTokenParquetDataset(
                parquet_files=paths,
                split_start=0.0,
                split_end=1.0,
                seed=42,
                stream_batch_size=2,
                shuffle=True,
                reshuffle_each_iteration=True,
            )
            loader = DataLoader(dataset, batch_size=None, num_workers=3)

            first_epoch = [int(sample[TOKENS_KEY][0]) for sample in loader]
            second_epoch = [int(sample[TOKENS_KEY][0]) for sample in loader]

            self.assertEqual(sorted(first_epoch), expected)
            self.assertEqual(sorted(second_epoch), expected)
            self.assertEqual(len(first_epoch), len(set(first_epoch)))
            self.assertEqual(len(second_epoch), len(set(second_epoch)))
            self.assertNotEqual(first_epoch, second_epoch)

    def test_fixed_validation_order_is_complete_and_repeatable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            expected = []
            for file_index in range(4):
                values = list(range(file_index * 9, (file_index + 1) * 9))
                path = root / f"part-{file_index:02d}.parquet"
                _write_parquet(path, values)
                paths.append(str(path))
                expected.extend(values)

            dataset = StreamingTokenParquetDataset(
                parquet_files=paths,
                split_start=0.0,
                split_end=1.0,
                seed=17,
                stream_batch_size=2,
                shuffle=True,
                reshuffle_each_iteration=False,
            )
            loader = DataLoader(dataset, batch_size=None, num_workers=2)

            first = [int(sample[TOKENS_KEY][0]) for sample in loader]
            second = [int(sample[TOKENS_KEY][0]) for sample in loader]

            self.assertEqual(sorted(first), expected)
            self.assertEqual(len(first), len(set(first)))
            self.assertEqual(first, second)

    def test_distributed_ranks_have_equal_complete_batches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            expected = []
            offset = 0
            for file_index, size in enumerate((11, 13, 17, 19, 23)):
                values = list(range(offset, offset + size))
                offset += size
                path = root / f"part-{file_index:02d}.parquet"
                _write_parquet(path, values, row_group_size=4)
                paths.append(str(path))
                expected.extend(values)

            datasets = [
                StreamingTokenParquetDataset(
                    parquet_files=paths,
                    split_start=0.0,
                    split_end=1.0,
                    seed=42,
                    stream_batch_size=3,
                    shuffle=True,
                    reshuffle_each_iteration=True,
                    loader_num_workers=1,
                    distributed_batch_size=4,
                    distributed_drop_last=True,
                )
                for _ in range(2)
            ]

            worker_limits = datasets[0]._distributed_worker_limits(2, 3)
            self.assertTrue(all(limit % 4 == 0 for limit in worker_limits))
            self.assertEqual(sum(worker_limits[0::2]), sum(worker_limits[1::2]))

            def rank_epoch(rank: int) -> list[int]:
                with mock.patch.dict(
                    "os.environ",
                    {"RANK": str(rank), "WORLD_SIZE": "2"},
                    clear=False,
                ):
                    return [int(sample[TOKENS_KEY][0]) for sample in datasets[rank]]

            first = [rank_epoch(0), rank_epoch(1)]
            second = [rank_epoch(0), rank_epoch(1)]

            self.assertEqual(len(first[0]), len(first[1]))
            self.assertEqual(len(first[0]) % 4, 0)
            self.assertTrue(set(first[0]).isdisjoint(first[1]))
            self.assertEqual(len(first[0] + first[1]), len(set(first[0] + first[1])))
            self.assertTrue(set(first[0] + first[1]).issubset(expected))
            self.assertNotEqual(set(first[0] + first[1]), set(second[0] + second[1]))


if __name__ == "__main__":
    unittest.main()
