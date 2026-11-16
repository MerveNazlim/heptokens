#!/usr/bin/env python3
"""Train one of the 45 matched downstream fits on the fixed >=4 e/mu cohort."""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import lightning as pl
import numpy as np
import pyarrow.parquet as pq
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score

import multiclass_controls_study as controls
import q4_latest_hier_seed_multiclass as q4_study
import seed_multiclass_study as seed_study
import projected_q1_seed_multiclass as q1_study
from heptokens.data.canonical_batch_classification import (
    CanonicalBatchAuditCallback, CanonicalBatchClassificationModule,
)
from heptokens.data.canonical_batch_schedule import load_index, load_schedule
from heptokens.models.foundation_continuous_matched_multiclass import LitMatchedContinuousMulticlass
from heptokens.models.foundation_grouped_cls_classifier import LitGroupedCLSClassifier, _load_checkpoint_state
from heptokens.models.utils import warmup_cosine_scheduler
from hybrid_q4_downstream import sequence_numpy, sha256, vocabulary_semantics
from prepare_hzz_4l_multiclass import load_selection, verify_view


NAMES = ("ggf", "vbf", "vh", "tth", "zz_continuum")
FAMILIES = ("q1_projected", "q4", "q8", "continuous_flat", "hierarchical_kin_aux")
CONDITIONS = ("finetune", "scratch", "frozen")
SEEDS = (42, 43, 44)
COUNTS = {
    "train": (236209, 24787, 31421, 48894, 7642),
    "val": (50773, 5253, 6520, 10460, 1665),
    "test": (50518, 5367, 6815, 10491, 1730),
}
BATCH_SIZE = 256
EPOCHS = 10
STEPS_PER_EPOCH = sum(COUNTS["train"]) // BATCH_SIZE
PROTOCOL = dict(
    cohort="at_least_four_selected_reconstructed_electrons_or_muons",
    taus_retained_as_inputs=True, original_splits_preserved=True, epochs=EPOCHS,
    batch_size=BATCH_SIZE, optimizer="AdamW", learning_rate=1e-4, weight_decay=1e-2,
    warmup_epochs=1, min_lr=1e-6, loss="inverse_training_frequency_weighted_cross_entropy",
    validation="all_selected_validation_events_each_epoch",
    checkpoint="maximum_validation_macro_one_vs_rest_AUC",
    full_optimizer_steps=EPOCHS * STEPS_PER_EPOCH,
)


def run_keys():
    return {f"{family}_{condition}_{'s' if condition == 'scratch' else 'p'}{seed}":
            (family, condition, seed if condition == "scratch" else 42,
             None if condition == "scratch" else seed)
            for family in FAMILIES for condition in CONDITIONS for seed in SEEDS}


def read_run(plan_path, key):
    plan = json.loads(plan_path.read_text())
    family, condition, seed, preseed = run_keys()[key]
    representation = plan["representations"][family]
    if representation["configuration_id"] != controls.CONFIGURATION_IDS[family]:
        raise ValueError("Original paper model configuration changed")
    source = None if preseed is None else plan["sources"][f"{family}_p{preseed}"]
    if source is not None and (source["model_family"] != family or source["pretrain_seed"] != preseed
                              or source["pretraining_configuration_id"] != representation["configuration_id"]):
        raise ValueError("Pretraining source differs from the paper configuration")
    return dict(run_key=key, model_family=family, condition=condition, training_seed=seed,
                pretrain_seed=preseed), representation, source


def class_weights():
    return (sum(COUNTS["train"]) / (len(NAMES) * np.asarray(COUNTS["train"]))).tolist()


def check_inputs(root, kind, membership_hash):
    index_dir = root / "canonical_index"
    selected = root / "selection" / "membership.npz"
    prepared = root / kind
    if sha256(selected) != membership_hash:
        raise ValueError("Wrong selected event membership")
    receipt, membership = load_selection(root / "selection")
    view = verify_view(prepared, root / "selection", representation=kind)
    index = load_index(index_dir)
    if tuple(index.manifest["class_names"]) != NAMES:
        raise ValueError("Unexpected class order")
    ids = {name: i for i, name in enumerate(receipt["sources"])}
    remap = np.asarray([ids[name] for name in index.sources])
    for split, counts in COUNTS.items():
        values = index.splits[split]
        if not np.array_equal(np.bincount(values.labels, minlength=5), counts):
            raise ValueError(f"Wrong canonical selected membership counts: {split}")
        original = membership[split]
        if not (np.array_equal(remap[values.source_id], original["source_id"])
                and np.array_equal(values.event_index, original["event_index"])
                and np.array_equal(values.labels, original["labels"])):
            raise ValueError(f"Canonical index is not the frozen 4l selection: {split}")
    manifest = json.loads((prepared / "manifest.json").read_text())
    for split, counts in COUNTS.items():
        if manifest["split_counts"][split] != dict(zip(NAMES, counts)) | {"total": sum(counts)}:
            raise ValueError(f"Wrong selected dataset manifest counts: {split}")
    if not np.allclose([manifest["recommended_cross_entropy_weights"][name] for name in NAMES],
                       class_weights(), rtol=1e-10, atol=0):
        raise ValueError("Wrong selected training class weights")
    return index, index_dir, prepared, view


def make_data(prepared, index_dir, kind, seed):
    # The base production guard fixes the *old* inclusive row count. Apply the
    # new count gate here, while retaining its audited schedule/batch path.
    data = CanonicalBatchClassificationModule(
        canonical_index_dir=index_dir, prepared_dir=str(prepared), production=False,
        n_classes=5, batch_size=BATCH_SIZE, num_workers=0, seed=seed, label_column="label",
        stream_batch_size=4096, shuffle_buffer_size=8192, require_continuous=kind == "continuous",
        continuous_feature_column="continuous_features" if kind == "continuous" else None,
        persistent_workers=False, pin_memory=True,
    )
    if data.train_set.total_rows != sum(COUNTS["train"]):
        raise ValueError("Wrong selected training row count")
    data.train_set.production = True
    data.production = True
    return data


def validate_source(run, source, source_dir, vocabulary):
    if source is None:
        if source_dir is not None:
            raise ValueError("Scratch cannot read pretrained weights")
        return None, None
    if source_dir is None:
        raise ValueError("Missing pinned source")
    ckpt, cfg, status = (source_dir / name for name in ("checkpoint.ckpt", "config.yaml", "status.txt"))
    family = run["model_family"]
    if source.get("pretrain_completion_sha256"):
        saved, _ = q4_study.validate_source_receipt(ckpt, cfg, status, source_dir / "completion.json", source)
    else:
        validator = q1_study.validate_pretrain_provenance if family == "q1_projected" else seed_study.validate_pretrain_provenance
        saved = validator(ckpt, cfg, status, source)
    if controls.DATA_KINDS[family] != "continuous":
        original = saved["hyper_parameters"].get("token_vocabulary")
        if original is None:
            recover = q1_study.verified_vocabulary if family == "q1_projected" else seed_study.legacy_pretraining_vocabulary
            original = recover(source_dir / "vocabulary.parquet", status, source, vocabulary)
        if vocabulary_semantics(original) != vocabulary_semantics(vocabulary):
            raise ValueError("Pretraining and selected classification vocabularies differ")
    return ckpt, controls.backbone_fingerprint(controls.checkpoint_backbone(saved))


def make_model(run, representation, schema, vocab, weights, checkpoint=None, *, training=True):
    hp = dict(representation["expected_hparams"])
    if training:
        hp.update(optimizer=partial(torch.optim.AdamW, lr=1e-4, weight_decay=1e-2),
                  scheduler=partial(warmup_cosine_scheduler, warmup_epochs=1, min_lr=1e-6))
    condition = run["condition"]
    if condition != "finetune":
        model = controls.classifier(dict(run, control_mode=condition), hp, schema, vocab, weights, checkpoint)
    else:
        continuous = representation["data_kind"] == "continuous"
        cls = LitMatchedContinuousMulticlass if continuous else LitGroupedCLSClassifier
        hp.update(continuous_schema=schema) if continuous else hp.update(token_vocabulary=vocab)
        model = cls(**hp, n_classes=5, class_weights=weights, classifier_hidden_dim=None,
                    freeze_backbone=False, backbone_ckpt_path=str(checkpoint) if checkpoint else None,
                    learning_rate=1e-4)
    controls.validate_model(model, dict(run, control_mode=condition), weights)
    for name, expected in representation["expected_hparams"].items():
        if model.hparams.get(name) != expected:
            raise ValueError(f"Actual model architecture drift: {name}")
    if representation["data_kind"] == "continuous" and model.backbone.role_embedding is None:
        raise ValueError("Continuous role embedding unexpectedly disabled")
    return model


class FitAudit(Callback):
    def __init__(self, run, source_hash, expected_steps):
        self.run, self.source_hash, self.expected_steps = run, source_hash, expected_steps
        self.frozen = run["condition"] == "frozen"
        self.receipt = None

    def on_fit_start(self, trainer, pl_module):
        self.initial = controls.backbone_fingerprint(pl_module.backbone.state_dict())
        if self.source_hash is not None and self.initial != self.source_hash:
            raise ValueError("Strict transfer of source backbone failed")
        controls.optimizer_audit(trainer.optimizers, pl_module)

    def on_train_epoch_start(self, trainer, pl_module):
        if self.frozen and any(m.training for m in pl_module.backbone.modules()):
            raise ValueError("Frozen backbone did not remain in evaluation mode")

    def on_after_backward(self, trainer, pl_module):
        if self.frozen and any(p.grad is not None for p in pl_module.backbone.parameters()):
            raise ValueError("Frozen backbone received gradients")

    def on_fit_end(self, trainer, pl_module):
        final = controls.backbone_fingerprint(pl_module.backbone.state_dict())
        if (final == self.initial) != self.frozen or trainer.global_step != self.expected_steps:
            raise ValueError("Wrong backbone trainability or optimizer-step count")
        self.receipt = dict(passed=True, global_step=int(trainer.global_step), initial_backbone_sha256=self.initial,
                            final_backbone_sha256=final, optimizer=controls.optimizer_audit(trainer.optimizers, pl_module))


@torch.inference_mode()
def predict(model, prepared, index, continuous, device):
    keys = {"mask": bool, "type_ids": np.int64}
    if continuous:
        keys.update(continuous_features=np.float32, continuous_feature_mask=bool, position_role_ids=np.int64)
    else:
        keys.update(tokens=np.int64)
    parts = {name: [] for name in ("probabilities", "source_file", "event_index", "labels")}
    for label, name in enumerate(NAMES):
        for path in sorted((prepared / "test" / name).glob("*.parquet")):
            for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_SIZE,
                    columns=[*keys, "source_file", "event_index", "label"]):
                column = lambda field: batch.column(batch.schema.get_field_index(field))
                inp = {k: torch.from_numpy(sequence_numpy(column(k), dt).copy()).to(device) for k, dt in keys.items()}
                scores = model(inp).softmax(-1).cpu().numpy()
                if scores.shape != (len(batch), 5) or not np.isfinite(scores).all():
                    raise ValueError("Invalid held-out probabilities")
                labels = column("label").to_numpy()
                if not np.all(labels == label):
                    raise ValueError("Test labels do not match class directory")
                for k, v in dict(probabilities=scores, labels=labels, source_file=column("source_file").to_pylist(),
                                 event_index=column("event_index").to_numpy()).items():
                    parts[k].append(np.asarray(v))
    arrays = {k: np.concatenate(v) for k, v in parts.items()}
    expected = index.splits["test"]
    expected_source = np.asarray(index.sources, dtype=str)[expected.source_id]
    actual_order = np.lexsort((arrays["event_index"], arrays["source_file"]))
    expected_order = np.lexsort((expected.event_index, expected_source))
    for key, value in (("source_file", expected_source), ("event_index", expected.event_index),
                       ("labels", expected.labels)):
        if not np.array_equal(arrays[key][actual_order], value[expected_order]):
            raise ValueError(f"Test predictions differ from the selected canonical test identities: {key}")
    if (len(actual_order) != sum(COUNTS["test"])
        or np.any((arrays["source_file"][actual_order][1:] == arrays["source_file"][actual_order][:-1])
                  & (arrays["event_index"][actual_order][1:] == arrays["event_index"][actual_order][:-1]))):
        raise ValueError("Missing or duplicate selected test events")
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "dataset-root", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--run-key", choices=run_keys(), required=True)
    parser.add_argument("--membership-sha256", required=True)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    run, rep, source = read_run(args.plan, args.run_key)
    kind, seed = rep["data_kind"], run["training_seed"]
    index, index_dir, prepared, view = check_inputs(args.dataset_root, kind, args.membership_sha256)
    for epoch in range(EPOCHS):
        schedule = load_schedule(index_dir / "schedules" / f"train_seed{seed}_epoch{epoch:02d}.npz", index_dir=index)
        expected = dict(seed=seed, epoch=epoch, full_rows=sum(COUNTS["train"]), batches=STEPS_PER_EPOCH,
                        dropped_rows=sum(COUNTS["train"]) % BATCH_SIZE, batch_size=BATCH_SIZE)
        if any(schedule.metadata.get(k) != v for k, v in expected.items()):
            raise ValueError("Unexpected canonical schedule")
    if not args.preflight_only and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError("Production fits require exactly one CUDA GPU")
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("medium")
    data = make_data(prepared, index_dir, kind, seed)
    schema, vocab = data.get_continuous_schema(), data.get_token_vocabulary()
    checkpoint, source_hash = validate_source(run, source, args.source_dir, vocab)
    weights = class_weights()
    model = make_model(run, rep, schema, vocab, weights, checkpoint)
    if source_hash is not None and controls.backbone_fingerprint(model.backbone.state_dict()) != source_hash:
        raise ValueError("Wrong source backbone loaded")
    if args.preflight_only:
        data.set_canonical_epoch(0, 1)
        batch = next(iter(data.train_dataloader()))
        model.eval()
        with torch.inference_mode():
            scores = model(batch)
        if scores.shape != (BATCH_SIZE, 5) or not torch.isfinite(scores).all():
            raise ValueError("First canonical batch could not be scored")
        print(f"PASS {args.run_key}: one selected batch, architecture, source, and membership")
        return
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    ckpts = root / "checkpoints"
    ckpts.mkdir(exist_ok=False)
    best = ModelCheckpoint(dirpath=ckpts, filename="best", monitor="valid/auc", mode="max",
                           save_on_train_epoch_end=False, auto_insert_metric_name=False,
                           enable_version_counter=False)
    last = ModelCheckpoint(dirpath=ckpts, filename="last", save_on_train_epoch_end=True,
                           auto_insert_metric_name=False, enable_version_counter=False)
    batch_audit = CanonicalBatchAuditCallback()
    fit_audit = FitAudit(run, source_hash, PROTOCOL["full_optimizer_steps"])
    controls.write_json(root / "run_plan.json", dict(**run, protocol=PROTOCOL,
        model_configuration=rep, pretraining_source=source, class_weights=weights,
        membership_sha256=args.membership_sha256, index_identity_sha256=index.identity_sha256,
        selected_manifest_sha256=view["manifest_sha256"]))
    trainer = pl.Trainer(default_root_dir=root, accelerator="gpu", devices=1, num_nodes=1,
                         max_epochs=EPOCHS, limit_train_batches=1.0, limit_val_batches=1.0,
                         val_check_interval=1.0, check_val_every_n_epoch=1, num_sanity_val_steps=2,
                         reload_dataloaders_every_n_epochs=1, log_every_n_steps=50,
                         logger=CSVLogger(root, name="training_metrics", version="fit"),
                         enable_progress_bar=False,
                         callbacks=[best, last, batch_audit, fit_audit])
    trainer.fit(model, datamodule=data)
    if (fit_audit.receipt is None or not best.best_model_path
        or not torch.isfinite(best.best_model_score) or not (ckpts / "last.ckpt").is_file()):
        raise ValueError("Fit did not complete with a finite validation-best checkpoint")
    controls.write_json(root / "fit_audit.json", fit_audit.receipt)
    saved = _load_checkpoint_state(best.best_model_path)
    restored = make_model(run, rep, schema, vocab, weights, training=False)
    restored.load_state_dict(saved["state_dict"], strict=True)
    del saved
    model.cpu()
    del model, trainer
    torch.cuda.empty_cache()
    restored = restored.eval().to("cuda")
    arrays = predict(restored, prepared, index, kind == "continuous", torch.device("cuda"))
    labels, probabilities = arrays["labels"], arrays["probabilities"]
    predicted = probabilities.argmax(axis=1)
    destination = root / "test_evaluation"
    destination.mkdir(exist_ok=False)
    path = destination / "test_predictions.npz"
    np.savez_compressed(path, **arrays, class_names=np.asarray(NAMES), predictions=predicted)
    auc = float(roc_auc_score(labels, probabilities, average="macro", multi_class="ovr"))
    summary = dict(**run, configuration_id=rep["configuration_id"], class_names=NAMES,
                   class_weights=weights, test_events=len(labels), protocol=PROTOCOL,
                   membership_sha256=args.membership_sha256, index_identity_sha256=index.identity_sha256,
                   selected_manifest_sha256=view["manifest_sha256"],
                   prediction_sha256=sha256(path),
                   checkpoint_sha256=sha256(best.best_model_path), checkpoint_selection="validation-best",
                   pretraining_checkpoint_sha256=None if source is None else source["checkpoint_sha256"],
                   macro_ovr_auc=auc, test_auc=auc, accuracy=float(accuracy_score(labels, predicted)),
                   balanced_accuracy=float(balanced_accuracy_score(labels, predicted)),
                   per_class_auc={n: float(roc_auc_score(labels == i, probabilities[:, i])) for i, n in enumerate(NAMES)},
                   confusion_matrix=confusion_matrix(labels, predicted, labels=list(range(5))).tolist())
    controls.write_json(destination / "evaluation_summary.json", summary)
    controls.write_json(root / "completion.json", dict(passed=True, **run, evaluation_summary_sha256=sha256(destination / "evaluation_summary.json")))
    print(f"PASS {args.run_key}: {len(labels)} matched test events, macro AUC {auc:.6f}")


if __name__ == "__main__":
    main()
