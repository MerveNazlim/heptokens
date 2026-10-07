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
    @staticmethod
    def _mixed_domain_dataset(
        directory: str, *, shuffle: bool = True, shuffle_mode: str = "buffered"
    ):
        raw_specs = []
        object_collections = [
            {
                "object_name": "jets",
                "mask_input": "common/jets/mask",
                "inputs": ["common/jets/pt"],
            }
        ]
        event_id = 0
        for domain, sign in (("MC", -1), ("realdata", 1)):
            for file_index in range(4):
                length = 32
                path = Path(directory) / domain / f"input-{file_index}.h5"
                path.parent.mkdir(exist_ok=True)
                values = sign * np.arange(
                    event_id + 1,
                    event_id + length + 1,
                    dtype=np.float32,
                )
                with h5py.File(path, "w") as handle:
                    jets = handle.create_group("common/jets")
                    jets.create_dataset("pt", data=values[:, None])
                    jets.create_dataset("mask", data=np.ones((length, 1), dtype=bool))
                raw_specs.append(
                    (str(path), length, "data" if domain == "realdata" else "mc")
                )
                event_id += length

        specs = _assign_global_event_splits(raw_specs, 1.0, 0.0, seed=17)
        return AtlasEventObjectIterableDataset(
            file_specs=specs,
            split_id=0,
            seed=17,
            shuffle=shuffle,
            event_inputs=[],
            object_collections=object_collections,
            object_type="jets",
            chunk_size=8,
            shuffle_mode=shuffle_mode,
            shuffle_buffer_size=32,
        )

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

            first_loader = module.train_dataloader()
            second_loader = module.train_dataloader()
            self.assertIsNone(first_loader.generator)
            self.assertIsNone(second_loader.generator)

            module = AtlasEventObjectIterableModule(
                data_paths=[str(path)],
                n_classes=1,
                object_type="jets",
                event_inputs=[],
                object_collections=module.data_config["object_collections"],
                num_objects=2,
                num_workers=0,
                batch_size=2,
                transforms={},
                shuffle_mode="buffered",
                split_by_domain=False,
            )
            first_loader = module.train_dataloader()
            second_loader = module.train_dataloader()
            self.assertEqual(first_loader.generator.initial_seed(), 42)
            self.assertEqual(second_loader.generator.initial_seed(), 43)

    def test_bounded_shuffle_preserves_exact_coverage_and_reshuffles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._mixed_domain_dataset(directory)
            first = [int(sample["csts"][0, 0]) for sample in dataset]
            second = [int(sample["csts"][0, 0]) for sample in dataset]

            expected = sorted(first)
            self.assertEqual(len(first), 256)
            self.assertEqual(len(set(first)), 256)
            self.assertEqual(sorted(second), expected)
            self.assertNotEqual(first, second)

    def test_legacy_default_matches_original_google_rng_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._mixed_domain_dataset(directory, shuffle_mode="legacy")
            # Independent reconstruction of Google's original file/chunk/event
            # shuffle, including the incremented seed on the next iteration.
            for iteration in (0, 1):
                rng = np.random.default_rng(dataset.seed + iteration)
                specs = list(dataset.file_specs)
                rng.shuffle(specs)
                expected = []
                for spec in specs:
                    chunks = list(range(0, spec.n_events, dataset.chunk_size))
                    rng.shuffle(chunks)
                    with h5py.File(spec.path, "r") as handle:
                        for start in chunks:
                            end = min(start + dataset.chunk_size, spec.n_events)
                            selected = np.flatnonzero(
                                spec.split_ids[start:end] == dataset.split_id
                            )
                            if not len(selected):
                                continue
                            rng.shuffle(selected)
                            values = handle["common/jets/pt"][start:end, 0]
                            expected.extend(int(values[i]) for i in selected)
                actual = [int(sample["csts"][0, 0]) for sample in dataset]
                self.assertEqual(actual, expected)

    def test_modes_share_membership_and_unshuffled_validation_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            legacy = self._mixed_domain_dataset(directory, shuffle_mode="legacy")
            buffered = self._mixed_domain_dataset(directory)
            a = [int(sample["csts"][0, 0]) for sample in legacy]
            b = [int(sample["csts"][0, 0]) for sample in buffered]
            self.assertEqual(sorted(a), sorted(b))
            self.assertNotEqual(a, b)
            legacy.shuffle = buffered.shuffle = False
            a = [int(sample["csts"][0, 0]) for sample in legacy]
            b = [int(sample["csts"][0, 0]) for sample in buffered]
            self.assertEqual(a, b)

    def test_buffer_owns_arrays_and_bounds_prefetch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._mixed_domain_dataset(directory)
            backing = np.arange(100, dtype=np.float32)
            consumed = 0

            def source():
                nonlocal consumed
                for value in backing:
                    consumed += 1
                    yield {"csts": backing[int(value) : int(value) + 1]}

            result = dataset._bounded_shuffle(source(), np.random.default_rng(9))
            first = next(result)
            self.assertEqual(consumed, dataset.shuffle_buffer_size + 1)
            self.assertIsNone(first["csts"].base)
            values = [float(first["csts"][0])] + [float(x["csts"][0]) for x in result]
            self.assertEqual(sorted(values), backing.tolist())

    def test_invalid_shuffle_mode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "shuffle_mode"):
                self._mixed_domain_dataset(directory, shuffle_mode="unknown")

    def test_bounded_shuffle_mixes_domains_before_batching(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._mixed_domain_dataset(directory)
            loader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=0)
            mixed_batches = 0
            total_batches = 0
            for batch in loader:
                values = batch["csts"][:, 0, 0]
                mixed_batches += int(torch.any(values < 0) and torch.any(values > 0))
                total_batches += 1

            self.assertGreaterEqual(mixed_batches, total_batches // 2)

    def test_multiworker_bounded_shuffle_has_no_duplicates_or_omissions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._mixed_domain_dataset(directory)
            # Read the same fixture instead of truncating H5 files potentially
            # still held by a worker finishing its shutdown.
            dataset.shuffle = False
            expected = sorted(int(sample["csts"][0, 0]) for sample in dataset)
            dataset.shuffle = True
            for mode in ("legacy", "buffered"):
                with self.subTest(shuffle_mode=mode):
                    dataset.shuffle_mode = mode
                    loader = torch.utils.data.DataLoader(
                        dataset, batch_size=13, num_workers=2
                    )
                    values = []
                    for batch in loader:
                        values.extend(int(value) for value in batch["csts"][:, 0, 0])
                    self.assertEqual(len(values), 256)
                    self.assertEqual(len(set(values)), 256)
                    self.assertEqual(sorted(values), expected)


if __name__ == "__main__":
    unittest.main()
