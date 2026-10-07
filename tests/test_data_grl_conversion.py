"""Tests for the reusable data-GRL H5-to-Parquet conversion policy."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from heptokens.data.data_grl_conversion import DataGrlConversionModule


class _Model:
    def __init__(self, codebook_size: int, num_quantizers: int) -> None:
        self.hparams = SimpleNamespace(
            codebook_size=codebook_size,
            num_quantizers=num_quantizers,
        )


class TestDataGrlConversionModule(unittest.TestCase):
    def test_split_is_stable_and_batch_boundary_independent(self) -> None:
        module = DataGrlConversionModule(seed=42, train_frac=0.9)
        indices = np.arange(10_000, dtype=np.uint64)
        whole = module.train_mask("gs://bucket/data/file.h5", indices)
        chunked = np.concatenate(
            [
                module.train_mask("gs://bucket/data/file.h5", part)
                for part in np.array_split(indices, 17)
            ]
        )
        np.testing.assert_array_equal(whole, chunked)
        self.assertGreater(int(whole.sum()), 8_800)
        self.assertLess(int(whole.sum()), 9_200)

    def test_source_uri_participates_in_event_identity(self) -> None:
        module = DataGrlConversionModule()
        indices = np.arange(1_000, dtype=np.uint64)
        left = module.train_mask("gs://bucket/data/a.h5", indices)
        right = module.train_mask("gs://bucket/data/b.h5", indices)
        self.assertFalse(np.array_equal(left, right))

    def test_real_data_policy_rejects_mc(self) -> None:
        module = DataGrlConversionModule(require_real_data=True)
        module.validate_source_metadata("data.h5", {"is_mc": False})
        with self.assertRaisesRegex(ValueError, "received MC input"):
            module.validate_source_metadata("mc.h5", {"is_mc": True})

    def test_tokenizer_specification_is_validated(self) -> None:
        module = DataGrlConversionModule()
        q1_models = {
            name: _Model(codebook_size, num_quantizers)
            for name, (codebook_size, num_quantizers) in (
                module.q1_tokenizer_spec.items()
            )
        }
        module.validate_tokenizers(q1_models, "q1")
        q1_models["jets"] = _Model(2_048, 8)
        with self.assertRaisesRegex(ValueError, "specification mismatch"):
            module.validate_tokenizers(q1_models, "q1")


if __name__ == "__main__":
    unittest.main()
