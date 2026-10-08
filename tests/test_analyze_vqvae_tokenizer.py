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
from heptokens.data.transforms import create_preprocessing_transformer  # noqa: E402


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

    def lepton_runs(self, eager: bool = False) -> dict:
        """Real, differently shaped ATLAS collections and separate saved scalers."""
        ids = np.arange(37, dtype=np.float32)
        runs = {}
        for object_type, features, slots, offset in (
            ("electrons", ["pt", "eta", "phi", "charge", "ptvarcone20", "topoetcone20",
                           "LHMedium", "LHTight"], 2, 0.0),
            ("muons", ["pt", "eta", "phi", "charge", "ptvarcone30", "topoetcone20"],
             3, 0.7),
        ):
            raw = np.column_stack(
                [20 + offset * 20 + ids, 0.02 * ids - offset, 0.03 * ids + offset]
                + [ids * (index + 1) / 100 for index in range(len(features) - 3)]
            ).astype(np.float32)
            mask = np.zeros((37, slots), dtype=bool)
            mask[:, 0] = True
            start = 0
            for path, length in zip(self.paths, (20, 17)):
                with h5py.File(path, "a") as handle:
                    if f"common/{object_type}" in handle:
                        del handle[f"common/{object_type}"]
                    group = handle.create_group(f"common/{object_type}")
                    for feature_index, name in enumerate(features):
                        values = np.full((length, slots), -999.0, dtype=np.float32)
                        values[:, 0] = raw[start:start + length, feature_index]
                        group.create_dataset(name, data=values)
                    group.create_dataset("mask", data=mask[start:start + length])
                start += length

            preprocessor = create_preprocessing_transformer(
                mode="log_standard", log_feature_indices=[0], n_features=len(features)
            ).fit(raw[self.members["train"]])
            run = self.root / f"{object_type}-{'eager' if eager else 'stream'}"
            (run / "checkpoints").mkdir(parents=True, exist_ok=True)
            joblib.dump(preprocessor, run / "preprocessor.joblib")
            cfg = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=True))
            cfg.datamodule.object_type = object_type
            cfg.datamodule.num_objects = slots
            cfg.datamodule.object_collections = [{
                "object_name": object_type, "mask_input": f"common/{object_type}/mask",
                "inputs": [f"common/{object_type}/{name}" for name in features],
            }]
            cfg.datamodule.transforms.preprocess.cst_fn.filename = str(run / "preprocessor.joblib")
            if eager:
                cfg.datamodule._target_ = "heptokens.data.atlas_event_mappable.AtlasEventMapModule"
                for key in ("chunk_size", "shuffle_buffer_size", "split_by_domain"):
                    del cfg.datamodule[key]
            OmegaConf.save(cfg, run / "full_config.yaml")
            bias = np.linspace(-0.2, 0.4, len(features), dtype=np.float32) + offset
            coder = functools.partial(CoderModel, hidden_dims=[8])
            model = LitVqVae(
                encoder=functools.partial(Encoder, model=coder),
                decoder=functools.partial(Decoder, model=coder), codebook_size=8,
                codebook_dim=3, num_quantizers=2, feature_names=features,
                data_sample={"csts": torch.zeros(1, slots, len(features))},
            ).eval()
            with torch.no_grad():
                output_layer = model.decoder.coder.model[-1]
                output_layer.weight.zero_()
                output_layer.bias.copy_(torch.from_numpy(bias))
            torch.save({
                "state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams),
                "pytorch-lightning_version": lightning.__version__,
            }, run / "checkpoints/best.ckpt")
            # Independently invert log+standard preprocessing for known decoder output.
            final = preprocessor.final_transformer
            prediction = bias * final.scale_ + final.mean_
            prediction[0] = np.exp(prediction[0]) - 1
            runs[object_type] = dict(
                cfg=cfg, run=run, raw=raw, features=features,
                preprocessor=preprocessor, prediction=prediction,
                model=analysis.load_analysis_model(run, None, torch.device("cpu")),
            )
        return runs

    @staticmethod
    def expected_lepton_observables(electron, muon) -> dict:
        deta = electron[1] - muon[1]
        dphi = (electron[2] - muon[2] + np.pi) % (2 * np.pi) - np.pi
        vectors = []
        for values, mass in ((electron, 0.000511), (muon, 0.10566)):
            pt, eta, phi = values[:3]
            momentum = np.array([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)])
            vectors.append(np.r_[np.sqrt(momentum @ momentum + mass ** 2), momentum])
        total = vectors[0] + vectors[1]
        return {"dR_ll": np.hypot(deta, dphi),
                "m_ll": np.sqrt(max(total[0] ** 2 - total[1:] @ total[1:], 0))}

    def derived_loaders(self, runs, split="val", **overrides):
        return analysis.aligned_lepton_dataloaders(
            electron_cfg=runs["electrons"]["cfg"], muon_cfg=runs["muons"]["cfg"],
            **{"h5_files": None, "split": split, "batch_size": 4, "num_workers": 0,
               "num_events_per_file": None, **overrides},
        )

    def derived_values(self, runs, loaders, max_events=200):
        return analysis.collect_lepton_pair_diagnostics(
            electron_model=runs["electrons"]["model"], muon_model=runs["muons"]["model"],
            electron_loader=loaders[0], muon_loader=loaders[1],
            electron_inverse_transformer=runs["electrons"]["preprocessor"],
            muon_inverse_transformer=runs["muons"]["preprocessor"],
            electron_feature_names=runs["electrons"]["features"],
            muon_feature_names=runs["muons"]["features"],
            device=torch.device("cpu"), max_events=max_events,
        )

    def test_derived_leptons_use_saved_preprocessing_splits_and_physical_units(self) -> None:
        for eager in (False, True):
            runs = self.lepton_runs(eager=eager)
            for split in ("val", "test"):
                with self.subTest(eager=eager, split=split):
                    loaders = self.derived_loaders(runs, split=split, num_workers=2)
                    self.assertTrue(all(loader.num_workers == 0 for loader in loaders))
                    snapshots = [{k: v.clone() for k, v in run["model"].state_dict().items()}
                                 for run in runs.values()]
                    with patch.object(runs["electrons"]["model"], "encode",
                                      wraps=runs["electrons"]["model"].encode) as e_encode, \
                         patch.object(runs["muons"]["model"], "encode",
                                      wraps=runs["muons"]["model"].encode) as m_encode:
                        derived = self.derived_values(runs, loaders)
                    members = self.members[split] if eager else np.sort(self.members[split])
                    references = [self.expected_lepton_observables(
                        runs["electrons"]["raw"][i], runs["muons"]["raw"][i]
                    ) for i in members]
                    prediction = self.expected_lepton_observables(
                        runs["electrons"]["prediction"], runs["muons"]["prediction"]
                    )
                    for name, (original, reconstructed) in derived.items():
                        self.assertEqual(len(original), len(members))  # 7 val, 8 test, NOT 37.
                        np.testing.assert_allclose(original, [r[name] for r in references],
                                                   rtol=2e-5, atol=2e-5)
                        np.testing.assert_allclose(reconstructed, prediction[name],
                                                   rtol=2e-5, atol=2e-5)
                    for run, encode, snapshot in zip(
                        runs.values(), (e_encode, m_encode), snapshots
                    ):
                        batches = [call.args[0] for call in encode.call_args_list]
                        normalized = torch.cat([b["csts"][b["mask"]] for b in batches]).numpy()
                        np.testing.assert_allclose(
                            normalized, run["preprocessor"].transform(run["raw"][members]),
                            rtol=2e-5, atol=2e-5,
                        )
                        for key, value in run["model"].state_dict().items():
                            torch.testing.assert_close(value, snapshot[key], rtol=0, atol=0)

    def test_derived_leptons_skip_missing_objects_and_cap_after_split(self) -> None:
        runs = self.lepton_runs()
        excluded = np.sort(self.members["val"])[:2]
        start = 0
        for path, length in zip(self.paths, (20, 17)):
            with h5py.File(path, "a") as handle:
                for index in excluded:
                    if start <= index < start + length:
                        handle["common/electrons/mask"][index - start, 0] = False
            start += length
        values = self.derived_values(runs, self.derived_loaders(runs), max_events=3)
        members = [i for i in np.sort(self.members["val"]) if i not in excluded][:3]
        for name, (original, _) in values.items():
            expected = [self.expected_lepton_observables(
                runs["electrons"]["raw"][i], runs["muons"]["raw"][i]
            )[name] for i in members]
            np.testing.assert_allclose(original, expected, rtol=2e-5, atol=2e-5)

    def test_derived_alignment_rejects_different_inputs_and_membership(self) -> None:
        for eager in (False, True):
            runs = self.lepton_runs(eager=eager)
            baseline = OmegaConf.to_container(runs["muons"]["cfg"], resolve=True)
            for field, value, message in (
                ("data_paths", list(reversed(self.paths)), "ordered H5 files"),
                ("seed", 43, "event splits"),
                ("num_events", 19, "event splits"),
            ):
                with self.subTest(eager=eager, field=field):
                    runs["muons"]["cfg"] = OmegaConf.create(baseline)
                    runs["muons"]["cfg"].datamodule[field] = value
                    with self.assertRaisesRegex(ValueError, message):
                        self.derived_loaders(runs)
        runs = self.lepton_runs()
        with self.assertRaisesRegex(ValueError, "val or test"):
            self.derived_loaders(runs, split="train")
        runs["muons"]["cfg"].datamodule._target_ = (
            "heptokens.data.atlas_event_mappable.AtlasEventMapModule"
        )
        for key in ("chunk_size", "shuffle_buffer_size", "split_by_domain"):
            del runs["muons"]["cfg"].datamodule[key]
        with self.assertRaisesRegex(ValueError, "mixing loader types"):
            self.derived_loaders(runs)

    def test_derived_collection_rejects_silent_loader_truncation(self) -> None:
        runs = self.lepton_runs()
        e_batches, m_batches = [list(loader) for loader in self.derived_loaders(runs)]
        with self.assertRaisesRegex(ValueError, "numbers of batches"):
            self.derived_values(runs, (e_batches, m_batches[:-1]))
        m_batches[0] = {key: value[:1] for key, value in m_batches[0].items()}
        with self.assertRaisesRegex(ValueError, "event counts"):
            self.derived_values(runs, (e_batches, m_batches))

    def test_derived_file_override_and_event_cap_use_canonical_split(self) -> None:
        runs = self.lepton_runs()
        loaders = self.derived_loaders(
            runs, h5_files=[self.paths[0]], num_events_per_file=12
        )
        selected = loaders[0].dataset.file_specs[0].split_ids == 1
        members = np.flatnonzero(selected)
        self.assertEqual(len(selected), 12)
        values = self.derived_values(runs, loaders)
        for name, (original, _) in values.items():
            expected = [self.expected_lepton_observables(
                runs["electrons"]["raw"][i], runs["muons"]["raw"][i]
            )[name] for i in members]
            np.testing.assert_allclose(original, expected, rtol=2e-5, atol=2e-5)
        self.assertIsNone(runs["electrons"]["cfg"].datamodule.num_events)
        self.assertEqual(list(runs["muons"]["cfg"].datamodule.data_paths), self.paths)

    def test_derived_cli_rejects_incomplete_or_nonheldout_requests(self) -> None:
        for extra in (
            ["--derived-electron-run-dir", "electron"],
            ["--derived-electron-run-dir", "electron", "--derived-muon-run-dir", "muon",
             "--split", "train"],
            ["--derived-electron-run-dir", "electron", "--derived-muon-run-dir", "muon",
             "--max-derived-events", "0"],
        ):
            with self.subTest(extra=extra), patch.object(
                sys, "argv", ["analyze_vqvae_tokenizer.py", "--run-dir", str(self.run), *extra]
            ), patch("sys.stderr"):
                with self.assertRaises(SystemExit) as error:
                    analysis.parse_args()
                self.assertEqual(error.exception.code, 2)

    def test_derived_cli_writes_physical_metrics_from_saved_runs(self) -> None:
        runs = self.lepton_runs()
        output = self.root / "derived-cli"
        saved_files = [
            run["run"] / name for run in runs.values()
            for name in ("full_config.yaml", "preprocessor.joblib", "checkpoints/best.ckpt")
        ]
        hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in saved_files]
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(
            (str(ROOT / "src"), str(ROOT / "scripts"))
        ))
        result = subprocess.run(
            [sys.executable, str(ROOT / "analyze_vqvae_tokenizer.py"), "--run-dir", str(self.run),
             "--output-dir", str(output), "--device", "cpu", "--split", "test",
             "--batch-size", "4", "--derived-electron-run-dir", str(runs["electrons"]["run"]),
             "--derived-muon-run-dir", str(runs["muons"]["run"])],
            cwd=ROOT, env=env, text=True, capture_output=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads((output / "derived_leptons/derived_lepton_metrics.json").read_text())
        prediction = self.expected_lepton_observables(
            runs["electrons"]["prediction"], runs["muons"]["prediction"]
        )
        for name in ("dR_ll", "m_ll"):
            reference = np.array([self.expected_lepton_observables(
                runs["electrons"]["raw"][i], runs["muons"]["raw"][i]
            )[name] for i in self.members["test"]])
            self.assertEqual(summary[name]["n"], 8)
            self.assertAlmostEqual(summary[name]["mae"],
                                   float(np.abs(prediction[name] - reference).mean()), delta=2e-5)
            self.assertGreater((output / f"derived_leptons/{name}_reconstruction_triptych.png")
                               .stat().st_size, 0)
        self.assertEqual(hashes, [
            hashlib.sha256(path.read_bytes()).hexdigest() for path in saved_files
        ])

    def test_derived_refuses_uninvertible_saved_preprocessing(self) -> None:
        runs = self.lepton_runs()
        cfg = runs["muons"]["cfg"]
        cfg.datamodule.transforms = {"preprocess": {
            "_target_": "heptokens.data.collation.collate_and_transform",
            "_partial_": True, "do_default_collate": False,
        }}
        OmegaConf.save(cfg, runs["muons"]["run"] / "full_config.yaml")
        args = Namespace(
            derived_electron_run_dir=str(runs["electrons"]["run"]),
            derived_muon_run_dir=str(runs["muons"]["run"]), h5_files=None,
            split="val", batch_size=4, num_workers=0, num_events_per_file=None,
            max_derived_events=200,
        )
        with self.assertRaisesRegex(ValueError, "invertible saved object preprocessing"):
            analysis.run_derived_lepton_diagnostics(
                args=args, output_dir=self.root, device=torch.device("cpu")
            )


if __name__ == "__main__":
    unittest.main()
