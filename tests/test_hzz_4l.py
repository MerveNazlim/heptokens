"""Tests for the integrated, treatment-matched HZZ downstream setup."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import lightning as pl
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from heptokens.models.foundation_hzz import HZZClassifier, backbone_fingerprint

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from hzz_4l_batches import RECORD, ScheduledBatches, StudyData, batch_digest, pinned  # noqa: E402
from train_hzz_4l import CONFIG, RUNS, FitAudit, optimizer_audit, read_json  # noqa: E402


def toy_stream(tmp_path):
    shards = []
    records = np.array([(0, 1, 0), (0, 2, 0), (0, 3, 1), (0, 4, 1)], dtype=RECORD)
    for label, physical_order in enumerate(([1, 0], [3, 2])):
        arr = records[physical_order]
        path = tmp_path / f"{label}.parquet"
        pq.write_table(pa.table({
            "source_file": ["h5"] * len(arr),
            "event_index": arr["event_index"],
            "label": arr["label"],
            "mask": [[True, True]] * len(arr),
            "type_ids": [[1, 3]] * len(arr),
            "tokens": [[[1] + [0] * 7, [int(i) + 3] + [0] * 7] for i in arr["event_index"]],
        }), path, row_group_size=1)
        shards.append((label, path))
    order = np.array([2, 0, 3, 1])
    namespace = bytes(32)
    data = SimpleNamespace(identities={"train": records, "val": records}, shards={"train": shards},
                           source_ids={"h5": 0}, columns={"mask": bool, "type_ids": np.int64, "tokens": np.int64},
                           batch_size=2, steps=2, namespace=namespace, orders=[order, order.copy()],
                           batch_hashes=[[batch_digest(namespace, records[selection])
                                          for selection in order.reshape(-1, 2)]] * 2)
    data.read_ids = StudyData.read_ids.__get__(data)
    data.read_inputs = StudyData.read_inputs.__get__(data)
    data.train = ScheduledBatches(data)
    return data


def model_kwargs():
    return dict(family="q8",
                hparams=dict(object_projection_mode="linear", vocab_size=64, max_quantizers=8,
                             quantizer_embedding_dim=2, hidden_dim=16, num_heads=2, num_layers=1,
                             dropout=0.1, max_seq_length=2, num_type_ids=5,
                             use_position_embedding=True, use_type_embedding=True),
                schema=None, vocabulary=dict(special_tokens={"pad": 0, "cls": 1},
                                             max_quantizers=8, vocab_size=64),
                weights=[1., 2., 3., 4., 5.],
                protocol=dict(learning_rate=1e-4, weight_decay=0.01, warmup_epochs=1,
                              min_lr=1e-6, epochs=2))


class HZZTests(unittest.TestCase):
    def test_study_keeps_only_two_families_and_eighteen_matched_runs(self):
        cfg = read_json(CONFIG)
        self.assertEqual(set(cfg["representations"]), {"q8", "continuous_flat"})
        self.assertEqual(len(RUNS), 18)
        for family, condition, seed, preseed in RUNS.values():
            if condition == "scratch":
                self.assertIn(seed, (42, 43, 44))
                self.assertIsNone(preseed)
            else:
                self.assertEqual(seed, 42)
                self.assertEqual(cfg["sources"][f"{family}_p{preseed}"]["pretrain_seed"], preseed)
        self.assertEqual(sum(cfg["split_counts"]["test"]), 64403)

    def test_training_batches_match_archive_order_across_parquet_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            data = toy_stream(Path(directory))
            data.train.epoch = 0
            batches = list(data.train)
            self.assertEqual([batch["tokens"][:, 1, 0].tolist() for batch in batches], [[6, 4], [7, 5]])
            self.assertEqual([batch["labels"].tolist() for batch in batches], [[1, 0], [1, 0]])
            with self.assertRaisesRegex(ValueError, "one explicitly armed"):
                list(data.train)
            data.train.epoch = 1
            data.batch_hashes[1][0] = "bad"
            with self.assertRaisesRegex(ValueError, "Canonical batch identity"):
                list(data.train)
            with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                pinned(data.shards["train"][0][1], "0" * 64)

    def test_two_epoch_fit_treatments(self):
        for condition in ("scratch", "frozen", "finetune"):
            with self.subTest(condition=condition), tempfile.TemporaryDirectory() as directory:
                self._fit(Path(directory), condition)

    def _fit(self, tmp_path, condition):
        data = toy_stream(tmp_path)

        class ToyDataModule(pl.LightningDataModule):
            def train_dataloader(self):
                data.train.epoch = self.trainer.current_epoch
                return torch.utils.data.DataLoader(data.train, batch_size=None)

            def val_dataloader(self):
                # All five classes for macro-OvR AUC, plus a partial last batch.
                vals = [{"tokens": torch.tensor([[1] + [0] * 7, [5] + [0] * 7]),
                         "mask": torch.tensor([True, True]), "type_ids": torch.tensor([1, 3]),
                         "labels": label} for label in torch.arange(5)]
                return torch.utils.data.DataLoader(vals, batch_size=2)

        data.identities["val"] = range(5)
        kwargs = model_kwargs()
        initial = HZZClassifier(**kwargs, condition="scratch")
        initial_state = initial.backbone.state_dict()
        initial_fingerprint = backbone_fingerprint(initial_state)
        model = HZZClassifier(**kwargs, condition=condition,
                              backbone_state=None if condition == "scratch" else initial_state)
        if condition != "scratch":
            self.assertEqual(backbone_fingerprint(model.backbone.state_dict()), initial_fingerprint)
        audit = FitAudit(data, {"protocol": kwargs["protocol"]},
                         None if condition == "scratch" else initial_fingerprint)
        trainer = pl.Trainer(
            default_root_dir=tmp_path, accelerator="cpu", devices=1, max_epochs=2,
            reload_dataloaders_every_n_epochs=1, enable_checkpointing=False, logger=False,
            enable_model_summary=False, enable_progress_bar=False, callbacks=[audit],
        )
        trainer.fit(model, datamodule=ToyDataModule())
        self.assertEqual(audit.receipt["global_step"], 4)
        self.assertEqual(audit.receipt["validation_rows"], {0: 5, 1: 5})
        self.assertTrue(trainer.callback_metrics["valid/auc"].isfinite())
        excluded = optimizer_audit(trainer, model)["backbone_parameter_tensors"] == 0
        self.assertEqual(excluded, condition == "frozen")
