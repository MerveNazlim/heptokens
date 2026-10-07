from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from heptokens.data.atlas_event_iterable import (
    AtlasEventObjectIterableDataset,
    AtlasEventObjectIterableModule,
    _assign_global_event_splits,
    _infer_domain,
)


class TestAtlasEventIterable(unittest.TestCase):
    def test_domain_inference_uses_sample_parent(self) -> None:
        self.assertEqual(_infer_domain("/scratch/h5/data/input.h5"), "data")
        self.assertEqual(_infer_domain("/scratch/h5/realdata/input.h5"), "data")
        self.assertEqual(_infer_domain("/home/zephyr/Data/sample/h5/MC/input.h5"), "mc")
        self.assertEqual(_infer_domain("/scratch/h5/MC/input.h5"), "mc")

    def test_global_split_matches_eager_random_split(self) -> None:
        lengths = [7, 5, 11]
        seed = 42
        specs = _assign_global_event_splits(
            [(f"file-{index}.h5", length, "mc") for index, length in enumerate(lengths)],
            train_frac=0.7,
            val_frac=0.15,
            seed=seed,
        )

        total = sum(lengths)
        train_size = int(total * 0.7)
        val_size = int(total * 0.15)
        permutation = torch.randperm(total, generator=torch.Generator().manual_seed(seed)).numpy()
        expected = np.full(total, 2, dtype=np.uint8)
        expected[permutation[:train_size]] = 0
        expected[permutation[train_size : train_size + val_size]] = 1

        actual = np.concatenate([spec.split_ids for spec in specs])
        np.testing.assert_array_equal(actual, expected)

    def test_chunked_iteration_returns_exact_split_members(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            offset = 0
            for file_index, length in enumerate((9, 8)):
                path = Path(directory) / "MC" / f"input-{file_index}.h5"
                path.parent.mkdir(exist_ok=True)
                with h5py.File(path, "w") as handle:
                    jets = handle.create_group("common/jets")
                    values = np.arange(offset, offset + length, dtype=np.float32)[:, None]
                    jets.create_dataset("pt", data=values)
                    jets.create_dataset("mask", data=np.ones_like(values, dtype=bool))
                paths.append((str(path), length, "mc"))
                offset += length

            specs = _assign_global_event_splits(paths, 0.7, 0.15, seed=9)
            common = dict(
                file_specs=specs,
                seed=9,
                event_inputs=[],
                object_collections=[
                    {
                        "object_name": "jets",
                        "mask_input": "common/jets/mask",
                        "inputs": ["common/jets/pt"],
                    }
                ],
                object_type="jets",
                chunk_size=4,
            )

            for split_id in (0, 1, 2):
                dataset = AtlasEventObjectIterableDataset(
                    split_id=split_id,
                    shuffle=False,
                    **common,
                )
                actual = [int(sample["csts"][0, 0]) for sample in dataset]
                split_ids = np.concatenate([spec.split_ids for spec in specs])
                expected = np.flatnonzero(split_ids == split_id).tolist()
                self.assertEqual(actual, expected)
                self.assertEqual(len(dataset), len(expected))

    def test_module_accepts_null_max_objects_from_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "MC" / "input.h5"
            path.parent.mkdir(exist_ok=True)
            with h5py.File(path, "w") as handle:
                jets = handle.create_group("common/jets")
                jets.create_dataset("pt", data=np.ones((10, 2), dtype=np.float32))
                jets.create_dataset("mask", data=np.ones((10, 2), dtype=bool))

            module = AtlasEventObjectIterableModule(
                data_paths=[str(path)],
                n_classes=1,
                object_type="jets",
                event_inputs=[],
                object_collections=[
                    {
                        "object_name": "jets",
                        "mask_input": "common/jets/mask",
                        "inputs": ["common/jets/pt"],
                    }
                ],
                max_objects=None,
                num_objects=2,
                num_workers=0,
                batch_size=2,
                transforms={},
                chunk_size=4,
                split_by_domain=False,
            )
            self.assertEqual(len(module.train_set), 7)
            self.assertEqual(len(module.valid_set), 1)
            self.assertEqual(len(module.test_set), 2)


if __name__ == "__main__":
    unittest.main()
