"""Continuous columns require the grouped, one-position-per-object layout."""

from __future__ import annotations

import sys
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import tokenize_objects_to_grouped_parquet as grouped  # noqa: E402
import tokenize_objects_to_parquet_with_atlasopenmagic_metadata as flat  # noqa: E402
from heptokens.data.continuous_schema import ROLE_IDS  # noqa: E402


class TestTokenizerExportLayout(unittest.TestCase):
    def test_flat_cli_rejects_continuous_output_before_loading_inputs(self) -> None:
        for decoded in (False, True):
            with self.subTest(decoded=decoded):
                args = Namespace(
                    write_continuous_features=True, write_decoded_q8_features=decoded
                )
                with patch.object(flat, "parse_args", return_value=args), patch.object(
                    flat, "load_data_config"
                ) as load_config, patch.object(flat, "load_models") as load_models:
                    with self.assertRaisesRegex(ValueError, "grouped_parquet.py"):
                        flat.main()
                    load_config.assert_not_called()
                    load_models.assert_not_called()

    def test_decoded_output_requires_continuous_features_for_both_layouts(self) -> None:
        args = Namespace(write_continuous_features=False, write_decoded_q8_features=True)
        for supported in (False, True):
            with self.subTest(grouped=supported), patch.object(
                flat, "parse_args", return_value=args
            ):
                with self.assertRaisesRegex(ValueError, "requires --write-continuous"):
                    flat.main(supports_continuous_features=supported)

    def test_grouped_entry_point_enables_continuous_output(self) -> None:
        with patch.object(flat, "build_vocabulary"), patch.object(
            flat, "assemble_rows"
        ), patch.object(flat, "make_table"), patch.object(flat, "main") as run:
            grouped.main()
            run.assert_called_once_with(supports_continuous_features=True)

    def test_flat_assembler_rejects_continuous_schema(self) -> None:
        with self.assertRaisesRegex(ValueError, "grouped exporter"):
            flat.assemble_rows({}, None, 1, {"continuous_schema": {}}, Namespace())

    def test_grouped_continuous_and_decoded_features_survive_parquet_roundtrip(self) -> None:
        import tempfile
        import pyarrow.parquet as pq

        args = Namespace(
            max_seq_length=5, pad_token_id=0, cls_token_id=1, sep_token_id=2,
            no_cls=False, no_event_token=False, no_separators=False,
            object_order=["jets"],
        )
        vocabulary = {
            "max_quantizers": 4,
            "objects": {"jets": {
                "num_quantizers": 4,
                "quantizers": [{"base": 20 + 10 * i, "size": 10} for i in range(4)],
            }},
            "continuous_schema": {
                "max_feature_dim": 2,
                "feature_columns": {"decoded_q8": "decoded_continuous_features"},
            },
        }
        features = torch.tensor([[[1.5, 2.5]]])
        decoded = torch.tensor([[[1.25, 2.25]]])
        assembled = grouped.assemble_grouped_rows(
            {"jets": (torch.tensor([[[1, 2, 3, 4]]]), torch.tensor([[True]]),
                      features, decoded)},
            np.array([[10]]), 1, vocabulary, args,
            event_values=np.array([[0.25]], dtype=np.float32),
        )
        tokens, mask, type_ids, extras = assembled
        table = grouped.make_grouped_table(
            tokens, mask, type_ids, start_index=0, source_file="fixture.h5",
            sample_metadata={"cross_section_pb": 1.0, "sqrt_s_tev": 13.6},
            write_legacy_columns=False,
            vocabulary=vocabulary, extra_columns=extras,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grouped.parquet"
            pq.write_table(table, path)
            restored = pq.read_table(path)
        # Compare data/schema and metadata separately: Parquet renames list children.
        self.assertTrue(restored.equals(table))
        self.assertEqual(restored.schema.metadata, table.schema.metadata)
        self.assertEqual(tokens[0, 2].tolist(), [21, 32, 43, 54])
        np.testing.assert_array_equal(extras["continuous_features"][0, 2], [1.5, 2.5])
        np.testing.assert_array_equal(extras["decoded_continuous_features"][0, 2], [1.25, 2.25])
        np.testing.assert_array_equal(extras["continuous_feature_mask"][0, 2], [True, True])
        self.assertEqual(extras["position_role_ids"][0].tolist(), [
            ROLE_IDS["cls"], ROLE_IDS["event"], ROLE_IDS["object"],
            ROLE_IDS["separator"], ROLE_IDS["padding"],
        ])

    def test_flat_tokens_without_continuous_options_are_unchanged(self) -> None:
        args = Namespace(
            max_seq_length=3, pad_token_id=0, cls_token_id=1, no_cls=False,
            no_event_token=True, no_separators=True, object_order=[],
        )
        result = flat.assemble_rows({}, None, 1, {"objects": {}}, args)
        self.assertEqual(len(result), 3)
        np.testing.assert_array_equal(result[0], [[1, 0, 0]])
        np.testing.assert_array_equal(result[1], [[True, False, False]])


if __name__ == "__main__":
    unittest.main()
