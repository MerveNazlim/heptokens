"""Saved-checkpoint evaluation with the canonical tokenizer datamodule."""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock, patch

import h5py
import joblib
import lightning
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_vqvae_tokenizer as analysis  # noqa: E402
from heptokens.models.coders import CoderModel, Decoder, Encoder  # noqa: E402
from heptokens.models.vq_vae import LitVqVae  # noqa: E402


class TestAnalyzeVqvaeTokenizer(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.run = self.root / "run"
        (self.run / "checkpoints").mkdir(parents=True)
        self.names = ["pt", "eta", "phi"]
        ids = np.arange(37, dtype=np.float32)
        leading = np.column_stack((ids + 1, (ids - 10) * 0.05, (ids % 5) * 0.1))
        self.values = np.stack((leading, leading + [0.25, 0.01, -0.02]), axis=1)
        self.values = self.values.astype(np.float32)
        self.mask = np.ones((37, 2), dtype=bool)
        self.mask[:, 1] = np.arange(37) % 2 == 0
        self.mask[::7] = False
        self.paths = []
        start = 0
        for index, length in enumerate((20, 17)):
            path = self.root / f"input-{index}.h5"
            with h5py.File(path, "w") as handle:
                jets = handle.create_group("common/jets")
                for feature, name in enumerate(self.names):
                    jets.create_dataset(name, data=self.values[start : start + length, :, feature])
                jets.create_dataset("mask", data=self.mask[start : start + length])
            self.paths.append(str(path))
            start += length
        permutation = torch.randperm(37, generator=torch.Generator().manual_seed(42)).numpy()
        self.members = {"train": permutation[:22], "val": permutation[22:29],
                        "test": permutation[29:]}
        train = self.members["train"]
        self.scaler = StandardScaler().fit(self.values[train][self.mask[train]])
        self.preprocessor = self.root / "preprocessor.joblib"
        joblib.dump(self.scaler, self.preprocessor)
        self.cfg = OmegaConf.create({
            "datamodule": {
                "_target_": "heptokens.data.atlas_event_iterable.AtlasEventObjectIterableModule",
                "data_paths": self.paths, "n_classes": 1, "event_inputs": [],
                "object_type": "jets", "output_mode": "object", "num_objects": 2,
                "num_events": None, "train_frac": 0.6, "val_frac": 0.2, "test_frac": 0.2,
                "seed": 42, "split_by_domain": False, "chunk_size": 4,
                "shuffle_buffer_size": 8, "batch_size": 4, "num_workers": 1,
                "persistent_workers": True, "multiprocessing_context": "spawn",
                "object_collections": [{
                    "object_name": "jets", "mask_input": "common/jets/mask",
                    "inputs": [f"common/jets/{name}" for name in self.names],
                }],
                "transforms": {"preprocess": {
                    "_target_": "heptokens.data.collation.preprocess_objects_batch",
                    "_partial_": True,
                    "cst_fn": {"_target_": "joblib.load", "filename": str(self.preprocessor)},
                }},
            },
            "model": {"codebook_size": 8},
        })
        OmegaConf.save(self.cfg, self.run / "full_config.yaml")
        self.checkpoint = self.run / "checkpoints/best.ckpt"
        self.decoder_bias = [1.25, -0.5, 0.75]
        self.save_checkpoint(2)

    def save_checkpoint(self, quantizers: int) -> LitVqVae:
        torch.manual_seed(42)
        coder = functools.partial(CoderModel, hidden_dims=[8])
        model = LitVqVae(
            encoder=functools.partial(Encoder, model=coder),
            decoder=functools.partial(Decoder, model=coder),
            codebook_size=8, codebook_dim=3, num_quantizers=quantizers,
            feature_names=self.names,
            data_sample={"csts": torch.zeros(1, 2, 3)},
        ).eval()
        # A known normalized-space prediction makes inverse reconstruction
        # verifiable independently of the evaluator implementation.
        with torch.no_grad():
            output_layer = model.decoder.coder.model[-1]
            output_layer.weight.zero_()
            output_layer.bias.copy_(torch.tensor(self.decoder_bias))
        torch.save({
            "state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": lightning.__version__,
        }, self.checkpoint)
        return model

    @staticmethod
    def loader_args(**overrides) -> Namespace:
        return Namespace(**{
            "h5_files": None, "num_events_per_file": None, "batch_size": 4,
            "num_workers": 0, **overrides,
        })

    def test_cli_defaults_to_validation(self) -> None:
        with patch.object(sys, "argv", ["analyze_vqvae_tokenizer.py", "--run-dir", str(self.run)]):
            args = analysis.parse_args()
        self.assertEqual(args.split, "val")
        self.assertIsNone(args.h5_files)
        self.assertIsNone(args.num_events_per_file)

    def test_root_entrypoint_delegates_cli_and_helpers_to_canonical_module(self) -> None:
        from scripts import analyze_vqvae_tokenizer as canonical

        spec = importlib.util.spec_from_file_location(
            "tokenizer_evaluation_compat", ROOT / "analyze_vqvae_tokenizer.py"
        )
        wrapper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wrapper)
        self.assertIs(wrapper.main, canonical.main)
        self.assertIs(wrapper.collect_diagnostics_for_h5_files,
                      canonical.collect_diagnostics_for_h5_files)
        self.assertIs(wrapper.codebook_summary, canonical.codebook_summary)

    def test_saved_split_and_preprocessing_config_are_not_modified(self) -> None:
        self.cfg.datamodule.shuffle_mode = "legacy"
        self.cfg.datamodule.num_events = 19
        before = OmegaConf.to_container(self.cfg, resolve=True)
        actual = analysis.analysis_datamodule_cfg(self.cfg, self.loader_args())
        for key in ("data_paths", "seed", "train_frac", "val_frac", "test_frac",
                    "num_events", "transforms", "shuffle_mode"):
            self.assertEqual(OmegaConf.to_container(actual, resolve=True)[key],
                             before["datamodule"][key])
        self.assertEqual(actual.num_workers, 0)
        self.assertFalse(actual.persistent_workers)
        self.assertIsNone(actual.multiprocessing_context)
        self.assertEqual(OmegaConf.to_container(self.cfg, resolve=True), before)

    def test_file_override_clears_stale_domain_sampling_settings(self) -> None:
        self.cfg.datamodule.data_domains = ["mc", "data"]
        self.cfg.datamodule.sampling_domain_fractions = {"mc": 0.5, "data": 0.5}
        self.cfg.datamodule.sampling_num_samples = 100
        actual = analysis.analysis_datamodule_cfg(
            self.cfg, self.loader_args(h5_files=[self.paths[0]], num_events_per_file=7)
        )
        self.assertEqual(list(actual.data_paths), [self.paths[0]])
        self.assertIsNone(actual.data_domains)
        self.assertIsNone(actual.sampling_domain_fractions)
        self.assertIsNone(actual.sampling_num_samples)
        self.assertEqual(actual.num_events, 7)

    def test_split_dispatch_uses_the_requested_dataloader(self) -> None:
        for split, method in ("train", "train_dataloader"), ("val", "val_dataloader"), (
            "test", "test_dataloader"
        ):
            with self.subTest(split=split):
                module = Mock()
                result = analysis.dataloader_from_datamodule(module, split)
                module.setup.assert_called_once_with(split)
                getattr(module, method).assert_called_once_with()
                self.assertIs(result, getattr(module, method).return_value)

    def test_validation_and_test_use_original_event_membership_and_preprocessing(self) -> None:
        for split in ("val", "test"):
            with self.subTest(split=split):
                loader = analysis.canonical_dataloader_for_h5_files(
                    cfg=self.cfg, h5_files=None, split=split, batch_size=4,
                    num_workers=0, num_events_per_file=None,
                )
                batches = list(loader)
                values = torch.cat([b["csts"] for b in batches]).numpy()
                mask = torch.cat([b["mask"] for b in batches]).numpy()
                restored = self.scaler.inverse_transform(values[mask])
                members = self.members[split]
                expected = self.values[members][self.mask[members]]
                np.testing.assert_allclose(
                    restored[np.argsort(restored[:, 0])],
                    expected[np.argsort(expected[:, 0])], atol=1e-5,
                )
                self.assertEqual(len(values), len(members))

    def test_saved_eager_atlas_loader_preserves_splits_and_inverse_reconstruction(self) -> None:
        from heptokens.data.atlas_event_mappable import AtlasEventMapDataset

        cfg = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=True))
        cfg.datamodule._target_ = "heptokens.data.atlas_event_mappable.AtlasEventMapModule"
        for key in ("chunk_size", "shuffle_buffer_size", "split_by_domain"):
            del cfg.datamodule[key]
        model = analysis.load_analysis_model(self.run, None, torch.device("cpu"))
        _, inverse = analysis.transform_list_and_cst_fn_from_cfg(cfg)
        prediction = np.asarray(self.decoder_bias) * self.scaler.scale_ + self.scaler.mean_
        for split in ("val", "test"):
            with self.subTest(split=split):
                loader = analysis.canonical_dataloader_for_h5_files(
                    cfg=cfg, h5_files=None, split=split, batch_size=4,
                    num_workers=0, num_events_per_file=None,
                )
                self.assertIsInstance(loader.dataset.dataset.datasets[0], AtlasEventMapDataset)
                original, reconstructed, codes, n_seen = analysis.collect_diagnostics_from_loader(
                    model=model, loader=loader, cst_inverse_transformer=inverse,
                    device=torch.device("cpu"), max_valid_objects=200,
                )
                members = self.members[split]
                expected = self.values[members][self.mask[members]]
                self.assertEqual(n_seen, len(expected))
                self.assertEqual(codes.shape, (len(expected), 2))
                np.testing.assert_allclose(
                    original[np.argsort(original[:, 0])],
                    expected[np.argsort(expected[:, 0])], atol=1e-5,
                )
                np.testing.assert_allclose(
                    reconstructed, np.tile(prediction, (len(expected), 1)), atol=1e-5
                )

    def test_real_q1_q4_q8_checkpoint_evaluation_excludes_padding_and_keeps_weights(self) -> None:
        _, inverse = analysis.transform_list_and_cst_fn_from_cfg(self.cfg)
        members = self.members["val"]
        expected = self.values[members][self.mask[members]]
        for quantizers in (1, 4, 8):
            with self.subTest(quantizers=quantizers):
                self.save_checkpoint(quantizers)
                model = analysis.load_analysis_model(self.run, None, torch.device("cpu"))
                self.assertFalse(model.training)
                before = {key: value.clone() for key, value in model.state_dict().items()}
                original, reconstructed, codes, n_seen = analysis.collect_diagnostics_for_h5_files(
                    cfg=self.cfg, model=model, h5_files=None, split="val",
                    num_events_per_file=None, batch_size=4, num_workers=0,
                    cst_inverse_transformer=inverse, device=torch.device("cpu"),
                    max_valid_objects=200,
                )
                self.assertEqual(n_seen, len(expected))
                self.assertEqual(reconstructed.shape, expected.shape)
                physical_prediction = (
                    np.asarray(self.decoder_bias) * self.scaler.scale_ + self.scaler.mean_
                )
                np.testing.assert_allclose(
                    reconstructed, np.tile(physical_prediction, (len(expected), 1)), atol=1e-5
                )
                self.assertEqual(codes.shape, (len(expected), quantizers))
                self.assertTrue(np.all((codes >= 0) & (codes < 8)))
                np.testing.assert_allclose(
                    original[np.argsort(original[:, 0])],
                    expected[np.argsort(expected[:, 0])], atol=1e-5,
                )
                counts = analysis.codebook_counts(codes, 8)
                np.testing.assert_array_equal(counts.sum(axis=1), [len(expected)] * quantizers)
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_object_cap_does_not_redefine_the_event_split(self) -> None:
        model = analysis.load_analysis_model(self.run, None, torch.device("cpu"))
        _, inverse = analysis.transform_list_and_cst_fn_from_cfg(self.cfg)
        original, reconstructed, codes, n_seen = analysis.collect_diagnostics_for_h5_files(
            cfg=self.cfg, model=model, h5_files=None, split="test", num_events_per_file=None,
            batch_size=4, num_workers=0, cst_inverse_transformer=inverse,
            device=torch.device("cpu"), max_valid_objects=3,
        )
        self.assertEqual(len(original), 3)
        self.assertEqual(len(reconstructed), 3)
        self.assertEqual(len(codes), 3)
        self.assertGreaterEqual(n_seen, 3)
        expected = self.values[self.members["test"]][self.mask[self.members["test"]]][:, 0]
        for value in original[:, 0]:
            self.assertTrue(np.any(np.isclose(expected, value)))

    def test_feature_names_come_from_the_saved_collection_order(self) -> None:
        for object_name, names in (
            ("muons", ["pt", "eta", "phi", "charge", "ptvarcone30", "topoetcone20"]),
            ("photons", ["pt", "eta", "phi", "isTight", "ptcone20", "topoetcone20",
                         "topoetcone40"]),
        ):
            with self.subTest(object_name=object_name):
                cfg = OmegaConf.create({"datamodule": {
                    "output_mode": "object", "object_type": object_name,
                    "object_collections": [{
                        "object_name": object_name,
                        "inputs": [f"common/{object_name}/{name}" for name in names],
                    }],
                }})
                self.assertEqual(analysis.feature_names_from_cfg(cfg, len(names)), names)
                self.assertNotIn("trk_iso03", analysis.feature_names_from_cfg(cfg, len(names)))

    def test_codebook_statistics_include_perplexity_and_ignore_masked_codes(self) -> None:
        codes = np.array([[0, 0], [0, 1], [0, 0], [0, 1], [-1, -1]])
        counts = analysis.codebook_counts(codes, 4)
        summary = analysis.codebook_summary(counts)
        self.assertEqual(summary["quantizer_0"]["assignments"], 4)
        self.assertEqual(summary["quantizer_0"]["dead_codes"], 3)
        self.assertAlmostEqual(summary["quantizer_0"]["perplexity"], 1.0)
        self.assertAlmostEqual(summary["quantizer_1"]["perplexity"], 2.0)
        self.assertAlmostEqual(summary["quantizer_1"]["entropy_nats"], np.log(2))
        empty = analysis.codebook_summary(np.zeros((1, 4), dtype=np.int64))
        self.assertEqual(empty["quantizer_0"]["perplexity"], 0.0)

    def test_cli_evaluates_real_checkpoint_and_writes_metrics_and_plots(self) -> None:
        before = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(ROOT / "src"),
                                                        str(ROOT / "scripts"))))
        entries = ("scripts/analyze_vqvae_tokenizer.py", "analyze_vqvae_tokenizer.py")
        physical_prediction = (
            np.asarray(self.decoder_bias) * self.scaler.scale_ + self.scaler.mean_
        )
        for split in ("val", "test"):
            snapshots = []
            members = self.members[split]
            reference = self.values[members][self.mask[members]]
            expected = len(reference)
            for entry_index, entry in enumerate(entries):
                with self.subTest(entrypoint=entry, split=split):
                    output = self.root / f"figures-{split}-{entry_index}"
                    command = [sys.executable, str(ROOT / entry),
                               "--run-dir", str(self.run), "--output-dir", str(output),
                               "--split", split, "--device", "cpu", "--num-workers", "0",
                               "--batch-size", "4"]
                    result = subprocess.run(command, cwd=ROOT, env=env, text=True,
                                            capture_output=True, timeout=60)
                    self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
                    self.assertEqual(
                        before, hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
                    )
                    counts = np.load(output / "codebook_counts.npy")
                    self.assertEqual(counts.shape, (2, 8))
                    np.testing.assert_array_equal(counts.sum(axis=1), [expected, expected])
                    metrics = json.loads((output / "reconstruction_metrics.json").read_text())
                    self.assertEqual(set(metrics), set(self.names))
                    for feature_index, name in enumerate(self.names):
                        metric = metrics[name]
                        residual = physical_prediction[feature_index] - reference[:, feature_index]
                        self.assertEqual(metric["n"], expected)
                        # Verify physical-unit outputs, not merely shapes or files.
                        self.assertAlmostEqual(
                            metric["mean_original"], float(reference[:, feature_index].mean()),
                            delta=2e-5,
                        )
                        self.assertAlmostEqual(
                            metric["mean_reconstructed"], physical_prediction[feature_index],
                            delta=2e-5,
                        )
                        self.assertAlmostEqual(metric["bias"], float(residual.mean()), delta=2e-5)
                        self.assertAlmostEqual(
                            metric["mae"], float(np.abs(residual).mean()), delta=2e-5
                        )
                    stats = json.loads((output / "codebook_usage_summary.json").read_text())
                    self.assertEqual(set(stats), {"quantizer_0", "quantizer_1"})
                    self.assertIn("perplexity", stats["quantizer_0"])
                    self.assertIn(
                        f"valid_objects_analyzed: {expected}", (output / "summary.txt").read_text()
                    )
                    for filename in ("pt_reconstruction_triptych.png", "pt_overlay.png",
                                     "codebook_frequency.png", "dead_tokens.png",
                                     "residual_summary.png"):
                        self.assertGreater((output / filename).stat().st_size, 0)
                    snapshots.append((metrics, stats, counts.tolist()))
            self.assertEqual(len(snapshots), 2)
            self.assertEqual(snapshots[0], snapshots[1])


if __name__ == "__main__":
    unittest.main()
