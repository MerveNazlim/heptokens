"""Train-only preprocessing must match tokenizer event membership."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import joblib
import numpy as np
from omegaconf import OmegaConf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from get_atlas_object_preprocessing import training_event_masks  # noqa: E402


class TestObjectPreprocessingSplit(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.paths = []
        self.values = []
        for index, count in enumerate((12, 11)):
            path = self.root / f"input-{index}.h5"
            values = (np.arange(count * 4, dtype=np.float32).reshape(count, 2, 2) + index * 100)
            with h5py.File(path, "w") as handle:
                handle.create_dataset("common/jets/pt", data=values[:, :, 0])
                handle.create_dataset("common/jets/eta", data=values[:, :, 1])
                handle.create_dataset("common/jets/mask", data=np.ones((count, 2), dtype=bool))
            self.paths.append(str(path))
            self.values.append(values)
        self.config = {
            "train_frac": 0.7, "val_frac": 0.15, "test_frac": 0.15, "num_events": None,
            "object_collections": [{"object_name": "jets", "mask_input": "common/jets/mask",
                                    "inputs": ["common/jets/pt", "common/jets/eta"]}],
        }
        self.config_path = self.root / "data.yaml"

    @staticmethod
    def expected_train(values: list[np.ndarray]) -> np.ndarray:
        combined = np.concatenate(values)
        order = torch.randperm(len(combined), generator=torch.Generator().manual_seed(42)).numpy()
        return combined[order[:int(len(combined) * 0.7)]].reshape(-1, 2)

    def run_fit(self, split: str | None, cap: int | None = None):
        OmegaConf.save(OmegaConf.create(self.config), self.config_path)
        command = [
            sys.executable, str(ROOT / "scripts/get_atlas_object_preprocessing.py"),
            "--h5-files", *self.paths, "--datamodule-config", str(self.config_path),
            "--object-type", "jets", "--mode", "standard", "--seed", "42",
            "--max-objects", "1000", "--output-dir", str(self.root / "fit"),
        ]
        if split is not None:
            command.extend(["--fit-split", split])
        if cap is not None:
            command.extend(["--num-events-per-file", str(cap)])
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60,
                                env=dict(os.environ, PYTHONPATH=str(ROOT / "src")))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        scaler = joblib.load(self.root / "fit/jets_standard.joblib")
        metadata = json.loads((self.root / "fit/jets_standard.json").read_text())
        return scaler, metadata

    def test_train_fit_excludes_validation_and_test_objects(self) -> None:
        scaler, metadata = self.run_fit("train")
        expected = self.expected_train(self.values)
        np.testing.assert_allclose(scaler.mean_, expected.mean(axis=0))
        self.assertEqual(metadata["n_objects_fit"], len(expected))
        self.assertEqual(metadata["fit_split"], "train")
        self.assertEqual(metadata["h5_files"], self.paths)

    def test_configured_and_explicit_event_caps_match_training(self) -> None:
        self.config["num_events"] = 5
        for cap, effective in ((None, 5), (3, 3)):
            with self.subTest(cap=cap):
                scaler, metadata = self.run_fit("train", cap)
                expected = self.expected_train([values[:effective] for values in self.values])
                np.testing.assert_allclose(scaler.mean_, expected.mean(axis=0))
                self.assertEqual(metadata["num_events_per_file"], effective)

    def test_default_all_mode_preserves_existing_fit_behavior(self) -> None:
        scaler, metadata = self.run_fit(None)
        expected = np.concatenate(self.values).reshape(-1, 2)
        np.testing.assert_allclose(scaler.mean_, expected.mean(axis=0))
        self.assertEqual(metadata["fit_split"], "all")

    def test_invalid_split_fractions_are_rejected(self) -> None:
        self.config["train_frac"] = 0.8
        with self.assertRaisesRegex(ValueError, "fractions"):
            training_event_masks(self.paths, "common/jets/pt", self.config, None, 42)


if __name__ == "__main__":
    unittest.main()
