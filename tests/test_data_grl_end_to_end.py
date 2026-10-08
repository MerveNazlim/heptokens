"""Exercise the direct Google H5-to-Parquet CLI, not legacy resharing.

The checkpoints are small, untrained fixtures: these tests establish data and
runtime correctness, not reconstruction quality or production GPU throughput.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from functools import partial
from pathlib import Path
from unittest.mock import patch

import h5py
import joblib
import lightning.pytorch as pl
import numpy as np
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler

from heptokens.data.data_grl_conversion import DataGrlConversionModule
from heptokens.data.token_parquet import StreamingTokenParquetDataset, TokenParquetPretrainModule
from heptokens.models.coders import CoderModel, Decoder, Encoder
from heptokens.models.vq_vae import LitVqVae


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tokenize_data_grl_multi_representation as conversion  # noqa: E402
from finalize_data_grl_conversion_manifests import finalize  # noqa: E402


class TestDirectDataGrlConversion(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.config = ROOT / "configs/datamodule/atlas_event_object.yaml"
        cls.policy = DataGrlConversionModule()
        cls.collections = OmegaConf.to_container(OmegaConf.load(cls.config))["object_collections"]
        cls.inputs = []
        for file_index, count in enumerate((67, 53)):
            path = cls.root / f"data-{file_index}.h5"
            with h5py.File(path, "w") as handle:
                meta = handle.create_group("metadata")
                meta.attrs["is_mc"] = False
                meta.attrs["n_events"] = count
                for index, key in enumerate(cls.policy.event_token_inputs):
                    handle.create_dataset(key, data=np.arange(count, dtype=np.float32) + index)
                for index, collection in enumerate(cls.collections):
                    values = np.arange(count * 2, dtype=np.float32).reshape(count, 2)
                    mask = (values.astype(np.int64) + index) % 5 != 0
                    # Include events with no valid objects in a collection.
                    mask[::11] = False
                    handle.create_dataset(collection["mask_input"], data=mask)
                    for feature_index, key in enumerate(collection["inputs"]):
                        handle.create_dataset(
                            key, data=values / 20 + file_index + index + feature_index / 10
                        )
            cls.inputs.append({"local_path": str(path), "source_uri": f"gs://test/data-{file_index}.h5"})
        cls.input_manifest = cls.root / "group.json"
        cls.input_manifest.write_text(json.dumps({"group_id": "group-00000", "files": cls.inputs}))
        cls.checkpoints = {"q1": [], "q8": []}
        cls.preprocessors = []
        torch.manual_seed(42)
        for collection in cls.collections:
            name = collection["object_name"]
            features = collection["inputs"]
            # Fit fixture preprocessing before conversion; conversion may only transform.
            scaler = StandardScaler().fit(
                np.arange(30 * len(features), dtype=np.float32).reshape(30, len(features)) / 10
            )
            preprocess_path = cls.root / f"{name}.joblib"
            joblib.dump(scaler, preprocess_path)
            cls.preprocessors.append(f"{name}={preprocess_path}")
            for representation, spec in (
                ("q1", cls.policy.q1_tokenizer_spec),
                ("q8", cls.policy.q8_tokenizer_spec),
            ):
                size, depth = spec[name]
                model = LitVqVae(
                    encoder=partial(Encoder, model=partial(CoderModel, hidden_dims=[])),
                    decoder=partial(Decoder, model=partial(CoderModel, hidden_dims=[])),
                    codebook_size=size, codebook_dim=8, num_quantizers=depth,
                    feature_names=[key.split("/")[-1] for key in features],
                    data_sample={"csts": torch.zeros(1, len(features))},
                )
                checkpoint = cls.root / f"{name}-{representation}.ckpt"
                torch.save({
                    "state_dict": model.state_dict(),
                    "hyper_parameters": dict(model.hparams),
                    "pytorch-lightning_version": pl.__version__,
                }, checkpoint)
                cls.checkpoints[representation].append(f"{name}={checkpoint}")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def arguments(self, output: Path, batch: int, shard: int) -> list[str]:
        return [
            "--conversion-config", str(ROOT / conversion.DEFAULT_CONVERSION_CONFIG),
            "--input-manifest", str(self.input_manifest),
            "--datamodule-config", str(self.config),
            "--output-dir", str(output), "--device", "cpu",
            "--batch-size", str(batch), "--shard-rows", str(shard),
            "--row-group-rows", "7",
            "--q1-tokenizer-checkpoints", *self.checkpoints["q1"],
            "--q8-tokenizer-checkpoints", *self.checkpoints["q8"],
            "--preprocess-transformers", *self.preprocessors,
        ]

    @staticmethod
    def read_rows(output: Path, representation: str, split: str) -> list[dict]:
        paths = sorted((output / representation / split).glob("*.parquet"))
        rows = [row for path in paths for row in pq.read_table(path).to_pylist()]
        # Scalar metadata can legitimately contain NaNs for unknown cross sections.
        values = ("source_file", "event_index", "tokens", "mask", "type_ids",
                  "continuous_features", "continuous_feature_mask", "position_role_ids")
        return sorted(
            [{key: row[key] for key in values if key in row} for row in rows],
            key=lambda row: (row["source_file"], row["event_index"]),
        )

    def test_real_cli_exact_roundtrip_across_batch_and_shard_boundaries(self) -> None:
        environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"), OMP_NUM_THREADS="2")
        artifacts = [Path(item.split("=", 1)[1]) for item in self.preprocessors]
        before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts]
        reference = None
        for batch, shard in ((1000, 1000), (7, 13), (11, 41)):
            with self.subTest(batch=batch, shard=shard):
                output = self.root / f"cli-{batch}-{shard}"
                result = subprocess.run(
                    [sys.executable, str(ROOT / "scripts/tokenize_data_grl_multi_representation.py"),
                     *self.arguments(output, batch, shard)],
                    cwd=ROOT, env=environment, capture_output=True, text=True, timeout=120,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                status = json.loads((output / "conversion_status/group-00000.json").read_text())
                self.assertEqual(status["total_rows"], 120)
                self.assertEqual(status["invalid_input_file_count"], 0)
                self.assertTrue((output / "conversion_status/group-00000.SUCCESS.txt").is_file())
                actual = {}
                for split in conversion.SPLITS:
                    identities = []
                    for representation in conversion.REPRESENTATIONS:
                        rows = self.read_rows(output, representation, split)
                        actual[representation, split] = rows
                        identities.append([(row["source_file"], row["event_index"]) for row in rows])
                        self.assertEqual(len(rows), status[f"{split}_rows"])
                        for row in rows:
                            self.assertEqual(len(row["mask"]), 256)
                            self.assertTrue(row["mask"][0])
                            if representation != "continuous":
                                depth = 1 if representation == "q1" else 8
                                self.assertEqual(len(row["tokens"][0]), depth)
                                self.assertEqual(row["tokens"][0], [1] + [0] * (depth - 1))
                            else:
                                self.assertTrue(np.isfinite(row["continuous_features"]).all())
                        for part in status["outputs"][representation][split]:
                            path = output / part["path"]
                            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), part["sha256"])
                            self.assertLessEqual(part["rows"], shard)
                            metadata = pq.ParquetFile(path).metadata
                            self.assertTrue(all(metadata.row_group(i).num_rows <= 7
                                                for i in range(metadata.num_row_groups)))
                    self.assertEqual(identities[0], identities[1])
                    self.assertEqual(identities[0], identities[2])
                    expected = []
                    for item, count in zip(self.inputs, (67, 53)):
                        mask = self.policy.train_mask(item["source_uri"], np.arange(count))
                        expected.extend((item["source_uri"], int(i)) for i in np.flatnonzero(
                            mask if split == "train" else ~mask))
                    self.assertEqual(identities[0], sorted(expected))
                if reference is None:
                    reference = actual
                else:
                    self.assertEqual(actual, reference)
        self.assertEqual(before, [hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts])

    def test_saved_transforms_only_and_bounded_reads_with_frozen_models(self) -> None:
        args = conversion.parse_args(self.arguments(self.root / "frozen", 7, 13))
        original_load = conversion.flat_export.load_models
        loaded = []

        def record_models(*arguments):
            models = original_load(*arguments)
            for model in models.values():
                self.assertFalse(model.training)
                loaded.append((model, {key: value.clone() for key, value in model.state_dict().items()}))
            return models

        with patch.object(conversion.flat_export, "load_models", side_effect=record_models), \
                patch.object(conversion.flat_export, "read_collection_batch",
                             wraps=conversion.flat_export.read_collection_batch) as read, \
                patch.object(StandardScaler, "fit", side_effect=AssertionError("Must not refit")):
            conversion.convert(args)
        self.assertGreater(read.call_count, 12)
        for call in read.call_args_list:
            event_slice = call.args[2]
            self.assertLessEqual(event_slice.stop - event_slice.start, 7)
        for model, before in loaded:
            self.assertTrue(all(torch.equal(value, model.state_dict()[key])
                                for key, value in before.items()))

    def test_finalized_outputs_stream_exact_values_without_resharding(self) -> None:
        output = self.root / "prepared"
        status = conversion.convert(conversion.parse_args(self.arguments(output, 7, 13)))
        plan = self.root / "campaign.json"
        plan.write_text(json.dumps({
            "groups": [{"group_id": "group-00000", "manifest": self.input_manifest.name}],
            "input_file_count": 2,
            "inventory_sha256": hashlib.sha256(self.input_manifest.read_bytes()).hexdigest(),
        }))
        summary = finalize(argparse.Namespace(
            campaign_plan=plan, status_dir=output / "conversion_status",
            output_dir=output / "manifests",
        ))
        self.assertEqual(summary["total_rows"], 120)
        for representation in conversion.REPRESENTATIONS:
            prepared = output / representation
            prepared.joinpath("manifest.json").write_bytes(
                output.joinpath("manifests", f"{representation}_manifest.json").read_bytes()
            )
            module = TokenParquetPretrainModule(
                prepared_dir=str(prepared), batch_size=16, num_workers=0, n_classes=1,
                include_tokens=representation != "continuous",
                require_continuous=representation == "continuous",
            )
            self.assertEqual(module.train_set.total_rows, status["train_rows"])
            self.assertEqual(module.valid_set.total_rows, status["val_rows"])
            for split in conversion.SPLITS:
                paths = sorted(prepared.joinpath(split).glob("*.parquet"))
                rows = [row for path in paths for row in pq.read_table(path).to_pylist()]
                dataset = StreamingTokenParquetDataset(
                    parquet_files=[str(path) for path in paths], shuffle=False,
                    split_start=0.0, split_end=1.0, stream_batch_size=3,
                    include_tokens=representation != "continuous",
                    require_continuous=representation == "continuous",
                )
                samples = list(dataset)
                self.assertEqual(len(samples), len(rows))
                for sample, row in zip(samples, rows):
                    for key, value in sample.items():
                        np.testing.assert_array_equal(value.numpy(), row[key])
                    self.assertEqual("tokens" in sample, representation != "continuous")

    def test_missing_saved_transform_fails_before_loading_or_writing(self) -> None:
        output = self.root / "missing-transform"
        args = conversion.parse_args(self.arguments(output, 7, 13))
        args.preprocess_transformers = args.preprocess_transformers[:-1]
        with patch.object(conversion.flat_export, "load_models") as load:
            with self.assertRaisesRegex(ValueError, "saved preprocessing.*missing"):
                conversion.convert(args)
        load.assert_not_called()
        self.assertFalse(list(output.rglob("*.parquet")))
        self.assertFalse(list(output.rglob("*.SUCCESS.txt")))

    def test_nonfinite_preprocessing_does_not_write_success(self) -> None:
        output = self.root / "nonfinite"
        args = conversion.parse_args(self.arguments(output, 7, 13))
        with patch.object(StandardScaler, "transform", side_effect=lambda values: values * np.nan):
            with self.assertRaisesRegex(ValueError, "non-finite valid features"):
                conversion.convert(args)
        self.assertFalse(list(output.rglob("*.SUCCESS.txt")))


if __name__ == "__main__":
    unittest.main()
