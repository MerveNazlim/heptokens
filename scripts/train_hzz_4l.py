#!/usr/bin/env python3
"""Train parallel Q8 or flat continuous on the pinned ATLAS HZZ four-lepton cohort."""

import argparse
import json
from pathlib import Path

import lightning as pl
import numpy as np
import pyarrow.parquet as pq
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score

from heptokens.data.token_parquet import _read_token_vocabulary
from heptokens.models.foundation_continuous import _load_checkpoint_state
from heptokens.models.foundation_hzz import HZZClassifier, backbone_fingerprint
from hzz_4l_batches import IDENTITY, StudyData, StudyDataModule, pinned, read_json, sha256

CONFIG = Path(__file__).resolve().parents[1] / "configs/downstream/hzz_4l_core.json"
SEEDS = (42, 43, 44)
RUNS = {
    f"{family}_{condition}_{'s' if condition == 'scratch' else 'p'}{seed}":
        (family, condition, seed if condition == "scratch" else 42,
         None if condition == "scratch" else seed)
    for family in ("q8", "continuous_flat")
    for condition in ("finetune", "scratch", "frozen")
    for seed in SEEDS
}
VOCABULARY_KEYS = ("objects", "event", "event_tokens", "object_order", "special_tokens",
                   "vocab_size", "max_quantizers", "includes_cls", "sequence_layout")
SCHEMA_KEYS = ("layout", "preprocessing", "max_feature_dim", "role_ids", "event_inputs", "objects")


def write_json(path, value):
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def source_backbone(source, directory, data, hparams):
    if source is None:
        if directory is not None:
            raise ValueError("Scratch must not be given a source directory or checkpoint")
        return None
    checkpoint = pinned(directory / "checkpoint.ckpt", source["checkpoint_sha256"])
    cfg = OmegaConf.load(pinned(directory / "config.yaml", source["pretrain_config_sha256"]))
    status_path = pinned(directory / "status.txt", source["pretrain_status_sha256"])
    status = dict(line.split("=", 1) for line in status_path.read_text().splitlines() if "=" in line)
    for key, expected in {
            "seed": source["pretrain_seed"], "datamodule.seed": source["pretraining_data_seed"],
            "datamodule.batch_size": source["pretraining_batch_size"],
            "trainer.devices": source["pretraining_num_devices"], "trainer.max_epochs": 10}.items():
        if OmegaConf.select(cfg, key) != expected:
            raise ValueError(f"Original pretraining config differs: {key}")
    expected_status = {"exit_code": "0", "seed": str(source["pretrain_seed"]), "epochs": "10"}
    if any(status.get(k) != v for k, v in expected_status.items()):
        raise ValueError("Pinned pretraining source did not finish with the expected seed")
    # Historical Lightning/Hydra checkpoints contain pickled metadata. Do not
    # deserialize until all three files match the committed artifact hashes.
    saved = _load_checkpoint_state(str(checkpoint))
    if saved["epoch"] != 9 or saved["global_step"] <= 0:
        raise ValueError("Expected a completed ten-epoch pretraining checkpoint")
    hp = saved["hyper_parameters"]
    if hp.get("seed", source["pretrain_seed"]) != source["pretrain_seed"]:
        raise ValueError("Saved pretraining seed differs")
    defaults = {"object_projection_mode": "linear", "feature_grouping": "schema"}
    for key, expected in hparams.items():
        observed = hp.get(key, defaults.get(key))
        if observed != expected or OmegaConf.select(cfg, f"model.{key}", default=observed) != observed:
            raise ValueError(f"Original pretraining architecture differs: {key}")
    if data.kind == "q8":
        vocabulary = hp.get("token_vocabulary")
        if vocabulary is None:
            if status.get("manifest_sha256") != source["pretraining_manifest_sha256"]:
                raise ValueError("Legacy Q8 vocabulary not tied to the original manifest")
            metadata = pinned(directory / "vocabulary.parquet", source["pretraining_vocabulary_shard_sha256"])
            vocabulary = _read_token_vocabulary(str(metadata))
        if vocabulary is None or any(vocabulary.get(k) != data.vocabulary.get(k) for k in VOCABULARY_KEYS):
            raise ValueError("Pretraining and downstream Q8 token ID mappings differ")
    elif any(hp["continuous_schema"][k] != data.schema[k] for k in SCHEMA_KEYS):
        raise ValueError("Pretraining and downstream continuous inputs differ")
    prefix = "model.backbone."
    state = {key.removeprefix(prefix): tensor for key, tensor in saved["state_dict"].items()
             if key.startswith(prefix)}
    if not state:
        raise ValueError("No pretrained backbone found")
    return state


def optimizer_audit(trainer, model):
    if len(trainer.optimizers) != 1:
        raise ValueError("Expected exactly one supervised optimizer")
    optimizer = trainer.optimizers[0]
    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    backbone = {id(p) for p in model.backbone.parameters()}
    if (len(ids) != len(set(ids)) or set(ids) != trainable
            or (set(ids) & backbone) != (set() if model.frozen else backbone)):
        raise ValueError("Optimizer trainable parameter set differs from the selected mode")
    return {"optimizer": type(optimizer).__name__, "parameter_tensors": len(ids),
            "backbone_parameter_tensors": len(set(ids) & backbone)}


class FitAudit(pl.Callback):
    def __init__(self, data, cfg, source_fingerprint):
        self.data, self.cfg, self.source_fingerprint = data, cfg, source_fingerprint
        self.receipt, self.validation_rows = None, {}

    def on_fit_start(self, trainer, model):
        self.initial = backbone_fingerprint(model.backbone.state_dict())
        if self.source_fingerprint is not None and self.initial != self.source_fingerprint:
            raise ValueError("Source backbone fingerprint changed on transfer")
        optimizer_audit(trainer, model)

    def on_train_epoch_start(self, trainer, model):
        if model.frozen and any(m.training for m in model.backbone.modules()):
            raise ValueError("Frozen backbone entered training mode")

    def on_after_backward(self, trainer, model):
        if model.frozen and any(p.grad is not None for p in model.backbone.parameters()):
            raise ValueError("Frozen backbone received gradients")

    def on_validation_batch_end(self, trainer, model, outputs, batch, batch_idx, dataloader_idx=0):
        if not trainer.sanity_checking:
            epoch = trainer.current_epoch
            self.validation_rows[epoch] = self.validation_rows.get(epoch, 0) + len(batch["labels"])

    def on_fit_end(self, trainer, model):
        protocol = self.cfg["protocol"]
        epochs = range(protocol["epochs"])
        expected_steps = protocol["epochs"] * self.data.steps
        final = backbone_fingerprint(model.backbone.state_dict())
        if ((final == self.initial) != model.frozen or trainer.global_step != expected_steps
                or self.data.train.consumed != {epoch: self.data.steps for epoch in epochs}
                or self.validation_rows != {epoch: len(self.data.identities["val"]) for epoch in epochs}):
            raise ValueError("Incomplete scheduled training/validation or wrong backbone trainability")
        self.receipt = dict(passed=True, global_step=int(trainer.global_step),
                            initial_backbone_sha256=self.initial, final_backbone_sha256=final,
                            schedule_batch_counts=self.data.train.consumed, validation_rows=self.validation_rows,
                            optimizer=optimizer_audit(trainer, model))


@torch.inference_mode()
def predict(model, data, device):
    records, probabilities = [], []
    names = np.asarray(data.sources)
    for label, path in data.shards["test"]:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=data.batch_size,
                                                       columns=[*IDENTITY, *data.columns]):
            import pyarrow as pa
            table = pa.Table.from_batches([batch])
            ids = data.read_ids(table)
            if np.any(ids["label"] != label):
                raise ValueError("Test labels differ from the shard class")
            inputs = {k: torch.from_numpy(v.copy()).to(device) for k, v in data.read_inputs(table).items()}
            prediction = model(inputs).softmax(-1).cpu().numpy()
            if prediction.shape != (len(ids), 5) or not np.isfinite(prediction).all():
                raise ValueError("Invalid held-out probabilities")
            records.append(ids)
            probabilities.append(prediction)
    actual, expected = np.concatenate(records), data.identities["test"]
    if not np.array_equal(np.sort(actual), np.sort(expected)):
        raise ValueError("Test identities are missing, duplicate or incorrect")
    return dict(probabilities=np.concatenate(probabilities), source_file=names[actual["source_id"]],
                event_index=actual["event_index"], labels=actual["label"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--run-key", choices=RUNS, required=True)
    parser.add_argument("--source-dir", type=Path, help="Staged files for the pinned pretraining seed (never for scratch)")
    parser.add_argument("--output-dir", type=Path, help="New output directory; required for a full fit")
    parser.add_argument("--preflight-only", action="store_true", help="Verify data and forward one canonical batch on CPU")
    args = parser.parse_args()
    family, condition, seed, preseed = RUNS[args.run_key]
    if (preseed is None) != (args.source_dir is None):
        parser.error("--source-dir is required for fine-tuning/frozen; forbidden for scratch")
    if not args.preflight_only and (args.output_dir is None or args.output_dir.exists()):
        parser.error("full fits require a new --output-dir")
    if not args.preflight_only and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        parser.error("full fits require exactly one visible CUDA GPU")
    cfg = read_json(CONFIG)
    rep = cfg["representations"][family]
    source = None if preseed is None else cfg["sources"][f"{family}_p{preseed}"]
    protocol = cfg["protocol"]
    data = StudyData(args.dataset_root, cfg, rep["data_kind"], seed)
    datamodule = StudyDataModule(data)
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("medium")
    state = source_backbone(source, args.source_dir, data, rep["expected_hparams"])
    source_fingerprint = None if state is None else backbone_fingerprint(state)
    model = HZZClassifier(family=family, condition=condition, hparams=rep["expected_hparams"],
                          schema=data.schema, vocabulary=data.vocabulary, weights=data.weights,
                          protocol=protocol, backbone_state=state)
    del state
    if source_fingerprint is not None and backbone_fingerprint(model.backbone.state_dict()) != source_fingerprint:
        raise ValueError("Model does not contain the pinned pretrained backbone")
    if args.preflight_only:
        data.train.epoch = 0
        model.eval()
        with torch.inference_mode():
            scores = model(next(iter(data.train)))
            valid_scores = model(next(iter(datamodule.val_dataloader())))
        if any(t.shape != (data.batch_size, 5) or not torch.isfinite(t).all() for t in (scores, valid_scores)):
            raise ValueError("First training or validation batch could not be scored")
        print(f"PASS {args.run_key}: verified selection, schedules, source and training/validation forward")
        return

    root = args.output_dir
    root.mkdir(parents=True)
    ckpts = root / "checkpoints"
    ckpts.mkdir()
    common = dict(run_key=args.run_key, model_family=family, configuration_id=rep["configuration_id"],
                  condition=condition, training_seed=seed, pretrain_seed=preseed,
                  class_names=cfg["class_names"], class_weights=data.weights, protocol=protocol,
                  membership_sha256=cfg["dataset"]["manifest_sha256"]["selection/membership.npz"],
                  index_identity_sha256=data.index["index_identity_sha256"],
                  selected_manifest_sha256=data.view["manifest_sha256"],
                  pretraining_checkpoint_sha256=None if source is None else source["checkpoint_sha256"])
    write_json(root / "run_plan.json", dict(**common, model_configuration=rep, pretraining_source=source))
    best = ModelCheckpoint(dirpath=ckpts, filename="best", monitor="valid/auc", mode="max",
                           save_on_train_epoch_end=False, auto_insert_metric_name=False,
                           enable_version_counter=False)
    last = ModelCheckpoint(dirpath=ckpts, filename="last", save_on_train_epoch_end=True,
                           auto_insert_metric_name=False, enable_version_counter=False)
    audit = FitAudit(data, cfg, source_fingerprint)
    trainer = pl.Trainer(default_root_dir=root, accelerator="gpu", devices=1, num_nodes=1,
                         max_epochs=protocol["epochs"], limit_train_batches=1.0, limit_val_batches=1.0,
                         val_check_interval=1.0, check_val_every_n_epoch=1, num_sanity_val_steps=2,
                         reload_dataloaders_every_n_epochs=1, log_every_n_steps=50,
                         logger=CSVLogger(root, name="training_metrics", version="fit"),
                         enable_progress_bar=False, callbacks=[best, last, audit])
    trainer.fit(model, datamodule=datamodule)
    if (audit.receipt is None or not best.best_model_path or not torch.isfinite(best.best_model_score)
            or not (ckpts / "last.ckpt").is_file()):
        raise ValueError("Fit did not produce a complete validation-best checkpoint")
    write_json(root / "fit_audit.json", audit.receipt)

    model.load_state_dict(_load_checkpoint_state(best.best_model_path)["state_dict"], strict=True)
    arrays = predict(model.eval().to("cuda"), data, torch.device("cuda"))
    labels, probabilities = arrays["labels"], arrays["probabilities"]
    predictions = probabilities.argmax(1)
    names = common["class_names"]
    target = root / "test_evaluation"
    target.mkdir()
    path = target / "test_predictions.npz"
    np.savez_compressed(path, **arrays, class_names=np.asarray(names), predictions=predictions)
    auc = float(roc_auc_score(labels, probabilities, average="macro", multi_class="ovr"))
    summary = dict(**common, test_events=len(labels), prediction_sha256=sha256(path),
                   checkpoint_sha256=sha256(best.best_model_path), checkpoint_selection="validation-best",
                   macro_ovr_auc=auc, test_auc=auc, accuracy=float(accuracy_score(labels, predictions)),
                   balanced_accuracy=float(balanced_accuracy_score(labels, predictions)),
                   per_class_auc={name: float(roc_auc_score(labels == i, probabilities[:, i])) for i, name in enumerate(names)},
                   confusion_matrix=confusion_matrix(labels, predictions, labels=list(range(len(names)))).tolist())
    write_json(target / "evaluation_summary.json", summary)
    write_json(root / "completion.json", dict(passed=True, run_key=args.run_key,
               evaluation_summary_sha256=sha256(target / "evaluation_summary.json")))
    print(f"PASS {args.run_key}: {len(labels)} matched test events, macro AUC {auc:.6f}")


if __name__ == "__main__":
    main()
