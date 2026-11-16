#!/usr/bin/env python3
"""Original projected-Q1 pretraining replicates, fixed fine-tuning seed 42.

This opt-in adapter leaves the six Q8/continuous/hybrid launch files untouched.
The original independent Q1 tokenizers are applied to the canonical five-class
events; they are not the first level of the Q8 tokenizer.
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf

from heptokens.models.foundation_grouped_cls_classifier import LitGroupedCLSClassifier, _load_checkpoint_state
from matched_multiclass_study import check_fit, validate_alignment
from seed_multiclass_study import (MODEL_KEYS, TOKEN_KEYS, NAMES, classes, compare,
    load_predictions, read_status, sha256, validate_export_provenance, vocabulary_semantics)

FAMILY = "q1_projected"
CONFIGURATION_ID = "q1_projected_v1_linear_1x32"
RUN_KEYS = tuple(f"q1_projected_p{seed}" for seed in (42, 43, 44))


def read_plan(path, key):
    plan = json.loads(Path(path).read_text())
    spec = plan["models"][key]
    if (key not in RUN_KEYS or spec["model_family"] != FAMILY or
            spec["pretrain_seed"] not in (42, 43, 44) or spec["finetune_seed"] != 42 or
            key != f"q1_projected_p{spec['pretrain_seed']}"):
        raise ValueError("Require original projected-Q1 pretraining seed 42/43/44 and fine-tuning seed 42")
    if spec["pretraining_configuration_id"] != CONFIGURATION_ID:
        raise ValueError("Wrong projected-Q1 configuration ID; direct-256 is not interchangeable")
    return plan, spec


def semantic_hparams(hparams):
    result = dict(hparams)
    # Original Q1 always used Linear(32, hidden_dim), before this option existed.
    # Strict backbone-state loading below verifies the actual saved projection.
    result.setdefault("object_projection_mode", "linear")
    return result


def validate_pretrain_provenance(checkpoint, config, status, spec):
    for path, key in ((checkpoint, "checkpoint_sha256"), (config, "pretrain_config_sha256"),
                      (status, "pretrain_status_sha256")):
        if sha256(path) != spec[key]:
            raise ValueError(f"Pinned pretraining checksum mismatch: {key}")
    cfg, result = OmegaConf.load(config), read_status(status)
    for key, expected in {"seed": spec["pretrain_seed"], "datamodule.seed": spec["pretraining_data_seed"],
            "datamodule.batch_size": spec["pretraining_batch_size"],
            "trainer.devices": spec["pretraining_num_devices"], "trainer.max_epochs": 10}.items():
        if OmegaConf.select(cfg, key) != expected:
            raise ValueError(f"Wrong pretraining configuration: {key}")
    if spec["pretraining_global_batch_size"] != spec["pretraining_batch_size"] * spec["pretraining_num_devices"]:
        raise ValueError("Inconsistent pretraining global batch")
    for key, expected in (("exit_code", 0), ("seed", spec["pretrain_seed"]), ("epochs", 10)):
        if str(result.get(key)) != str(expected):
            raise ValueError(f"Pretraining status did not succeed/match: {key}")
    saved = _load_checkpoint_state(str(checkpoint))
    hp = semantic_hparams(saved["hyper_parameters"])
    if saved.get("epoch") != 9 or saved.get("global_step", 0) <= 0:
        raise ValueError("Pretraining did not complete ten epochs")
    for key, expected in spec["expected_hparams"].items():
        if hp.get(key) != expected:
            raise ValueError(f"Wrong pretrained {key}: {hp.get(key)!r}, expected {expected!r}")
    if "seed" in hp and hp["seed"] != spec["pretrain_seed"]:
        raise ValueError("Checkpoint pretraining seed differs")
    for key in MODEL_KEYS + TOKEN_KEYS:
        observed = OmegaConf.select(cfg, f"model.{key}")
        if observed is not None and observed != hp.get(key):
            raise ValueError(f"Pretraining configuration/checkpoint model mismatch: {key}")
    saved["hyper_parameters"] = hp
    return saved


def verified_vocabulary(path, status, spec, classification_vocabulary):
    if path is None or sha256(path) != spec["pretraining_vocabulary_shard_sha256"]:
        raise ValueError("Original Q1 pretraining vocabulary shard checksum mismatch")
    if read_status(status).get("manifest_sha256") != spec["pretraining_manifest_sha256"]:
        raise ValueError("Original Q1 pretraining status/manifest provenance mismatch")
    encoded = (pq.ParquetFile(path).schema_arrow.metadata or {}).get(b"heptokens_token_vocabulary")
    if encoded is None:
        raise ValueError("Original Q1 pretraining shard lacks vocabulary metadata")
    vocabulary = json.loads(encoded)
    if vocabulary.get("max_quantizers") != 1 or vocabulary.get("vocab_size") != 98820:
        raise ValueError("Not the original independent Q1 tokenizer vocabulary")
    if vocabulary_semantics(vocabulary) != vocabulary_semantics(classification_vocabulary):
        raise ValueError("Original Q1 pretraining/classification token-ID mapping differs")
    return vocabulary


def classifier(spec, hp, vocabulary, weights, checkpoint=None, legacy_vocabulary=None):
    hp = semantic_hparams(hp)
    required = {"vocab_size": 98820, "max_quantizers": 1,
                "quantizer_embedding_dim": 32, "object_projection_mode": "linear"}
    if any(hp.get(key) != value for key, value in required.items()):
        raise ValueError("Require projected Q1: one 32D embedding and learned 32-to-hidden projection")
    pretrained_vocabulary = hp.get("token_vocabulary")
    if pretrained_vocabulary is None:
        if legacy_vocabulary is None:
            raise ValueError("Missing pretrained token-ID mapping; require verified original Q1 shard")
        pretrained_vocabulary = legacy_vocabulary
    if vocabulary_semantics(pretrained_vocabulary) != vocabulary_semantics(vocabulary):
        raise ValueError("Classification token-ID mapping differs from projected-Q1 pretraining")
    model = LitGroupedCLSClassifier(n_classes=5, class_weights=weights, freeze_backbone=False,
        classifier_hidden_dim=None, learning_rate=1e-4,
        backbone_ckpt_path=str(checkpoint) if checkpoint else None, token_vocabulary=vocabulary,
        **{key: hp[key] for key in MODEL_KEYS + TOKEN_KEYS if key in hp})
    if not isinstance(model.backbone.object_projection, torch.nn.Linear):
        raise ValueError("Projected-Q1 linear projection was removed")
    if model.backbone.object_projection.in_features != 32:
        raise ValueError("Projected-Q1 input width is not 32")
    if checkpoint:
        state = _load_checkpoint_state(str(checkpoint))["state_dict"]
        transferred = {name.removeprefix("model.backbone."): value for name, value in state.items()
                       if name.startswith("model.backbone.")}
        if not transferred or set(transferred) != set(model.backbone.state_dict()):
            raise ValueError("Incomplete projected-Q1 backbone transfer")
        for name, value in transferred.items():
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError("Non-finite projected-Q1 backbone tensor")
            if not torch.equal(model.backbone.state_dict()[name], value):
                raise ValueError(f"Pretraining backbone transfer changed tensor: {name}")
    return model


def preflight(args):
    plan, spec = read_plan(args.plan, args.model_name)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    # The frozen function's Q1 branch name identifies data shape, not architecture.
    manifest, vocabulary, _ = validate_alignment(args.source_dir, args.prepared_dir, args.report_dir,
        "q1_direct256", plan, args.input_inventory)
    saved = validate_pretrain_provenance(args.checkpoint, args.pretrain_config, args.pretrain_status, spec)
    original = verified_vocabulary(args.pretrain_vocabulary_shard, args.pretrain_status, spec, vocabulary)
    weights = [manifest["recommended_cross_entropy_weights"][name] for name in NAMES]
    model = classifier(spec, saved["hyper_parameters"], vocabulary, weights, args.checkpoint, original)
    if not all(p.requires_grad for p in model.backbone.parameters()) or not isinstance(model.model.classifier, torch.nn.Linear):
        raise ValueError("Require fully unfrozen backbone and linear classification head")
    report = dict(passed=True, model=args.model_name, model_family=FAMILY,
        pretrain_seed=spec["pretrain_seed"], finetune_seed=42, plan_sha256=sha256(args.plan),
        checkpoint_sha256=spec["checkpoint_sha256"], checkpoint_uri=spec["checkpoint"],
        checkpoint_local_path=str(args.checkpoint.resolve()),
        pretrain_config_sha256=spec["pretrain_config_sha256"], pretrain_status_sha256=spec["pretrain_status_sha256"],
        pretraining_job=spec["pretraining_job"], pretraining_configuration_id=CONFIGURATION_ID,
        pretraining_num_devices=spec["pretraining_num_devices"], pretraining_batch_size=spec["pretraining_batch_size"],
        pretraining_global_batch_size=spec["pretraining_global_batch_size"], pretraining_data_seed=spec["pretraining_data_seed"],
        pretraining_topology=dict(num_devices=spec["pretraining_num_devices"],
            batch_size_per_gpu=spec["pretraining_batch_size"], global_batch_size=spec["pretraining_global_batch_size"],
            data_seed=spec["pretraining_data_seed"]),
        pretraining_vocabulary_provenance=dict(source="original pretraining shard metadata",
            pretraining_manifest_sha256=spec["pretraining_manifest_sha256"],
            shard_uri=spec["pretraining_vocabulary_shard"], shard_sha256=spec["pretraining_vocabulary_shard_sha256"],
            original_status_manifest_verified=True, classification_mapping_verified=True),
        pretrain_epoch=saved["epoch"], pretrain_global_step=saved["global_step"],
        source_manifest_sha256=plan["source_manifest_sha256"], manifest_sha256=sha256(args.prepared_dir / "manifest.json"),
        classes=NAMES, class_weights=weights, split_counts=manifest["split_counts"], classifier_head="linear_256_to_5",
        backbone_frozen=False, protocol=plan["protocol"], caveats=[])
    (args.report_dir / "transfer_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS original projected-Q1 mapping, exact five-class samples, strict backbone transfer, fine-tuning seed 42")
    return report


def export_test(args):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, roc_auc_score
    plan, spec = read_plan(args.plan, args.model_name)
    manifest = json.loads((args.prepared_dir / "manifest.json").read_text())
    transfer = validate_export_provenance(args, plan, spec, manifest)
    saved = _load_checkpoint_state(str(args.checkpoint)); hp = dict(saved["hyper_parameters"])
    if str(Path(hp.get("backbone_ckpt_path", "")).resolve()) != transfer["checkpoint_local_path"]:
        raise ValueError("Fine-tuned checkpoint saved a different pretrained source")
    allowed = inspect.signature(LitGroupedCLSClassifier.__init__).parameters
    kwargs = {key: value for key, value in hp.items() if key in allowed}
    kwargs.update(backbone_ckpt_path=None, optimizer=None, scheduler=None)
    model = LitGroupedCLSClassifier(**kwargs); model.load_state_dict(saved["state_dict"], strict=True)
    weights = [manifest["recommended_cross_entropy_weights"][name] for name in NAMES]
    if model.n_classes != 5 or not isinstance(model.model.classifier, torch.nn.Linear) or model.model.classifier.out_features != 5:
        raise ValueError("Not a linear five-class classifier")
    if not all(p.requires_grad for p in model.backbone.parameters()) or not np.allclose(model.class_weights.cpu(), weights):
        raise ValueError("Wrong class weights or frozen backbone in fine-tuned checkpoint")
    for key in MODEL_KEYS + TOKEN_KEYS:
        if key in spec["expected_hparams"] and hp.get(key) != spec["expected_hparams"][key]:
            raise ValueError(f"Fine-tuned projected-Q1 representation mismatch: {key}")
    model.eval().to(args.device); torch.set_float32_matmul_precision("medium")
    keys = [("tokens", np.int64), ("mask", bool), ("type_ids", np.int64)]
    parts = {name: [] for name in ("probabilities", "labels", "source_file", "event_index")}
    from hybrid_q4_downstream import sequence_numpy
    with torch.inference_mode():
        for label, name in enumerate(NAMES):
            count = 0
            for path in sorted((args.prepared_dir / "test" / name).glob("*.parquet")):
                for batch in pq.ParquetFile(path).iter_batches(batch_size=args.batch_size,
                        columns=[key for key, _ in keys] + ["label", "source_file", "event_index"]):
                    column = lambda key: batch.column(batch.schema.get_field_index(key))
                    labels = np.asarray(column("label").to_numpy(), dtype=np.int64)
                    if not np.all(labels == label):
                        raise ValueError("Invalid held-out labels")
                    inputs = {key: torch.from_numpy(sequence_numpy(column(key), dtype).copy()).to(args.device) for key, dtype in keys}
                    values = dict(probabilities=model(inputs).softmax(-1).cpu().numpy(), labels=labels,
                        source_file=np.asarray(column("source_file").to_pylist(), dtype=str),
                        event_index=np.asarray(column("event_index").to_numpy(), dtype=np.int64))
                    for key, value in values.items(): parts[key].append(value)
                    count += len(labels)
            if count != manifest["split_counts"]["test"][name]:
                raise ValueError(f"Incomplete held-out split: {name}")
    arrays = {key: np.concatenate(value) for key, value in parts.items()}
    arrays.update(class_names=np.asarray(NAMES), predictions=arrays["probabilities"].argmax(1))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output_dir / "test_predictions.npz", **arrays)
    checked = load_predictions(args.output_dir); labels, probabilities = checked["labels"], checked["probabilities"]
    if len(labels) != manifest["split_counts"]["test"]["total"]:
        raise ValueError("Incomplete/duplicated held-out test events")
    predictions = probabilities.argmax(1)
    rows = compare([(f"Q1 projected pretrain seed {spec['pretrain_seed']}", args.output_dir)],
                   args.output_dir / "plots", expected_events=len(labels))
    report = {key: transfer[key] for key in ("model", "model_family", "pretrain_seed", "finetune_seed",
        "checkpoint_uri", "checkpoint_local_path", "pretrain_config_sha256", "pretrain_status_sha256",
        "pretraining_job", "pretraining_configuration_id", "pretraining_num_devices", "pretraining_batch_size",
        "pretraining_global_batch_size", "pretraining_data_seed", "pretraining_topology", "pretraining_vocabulary_provenance",
        "source_manifest_sha256", "caveats")}
    report.update(pretraining_checkpoint_sha256=spec["checkpoint_sha256"], checkpoint_selection="validation-best",
        checkpoint_sha256=sha256(args.checkpoint), finetune_config_sha256=sha256(args.run_dir / "full_config.yaml"),
        manifest_sha256=sha256(args.prepared_dir / "manifest.json"), class_names=NAMES, test_events=len(labels),
        macro_ovr_auc=float(roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")),
        weighted_ovr_auc=float(roc_auc_score(labels, probabilities, multi_class="ovr", average="weighted")),
        accuracy=float(accuracy_score(labels, predictions)), balanced_accuracy=float(balanced_accuracy_score(labels, predictions)),
        per_class_auc={row["class_name"]: row["auc"] for row in rows},
        confusion_matrix=confusion_matrix(labels, predictions, labels=range(5)).tolist(),
        normalized_confusion_matrix=confusion_matrix(labels, predictions, labels=range(5), normalize="true").tolist())
    report["test_auc"] = report["macro_ovr_auc"]
    (args.output_dir / "classification_report.json").write_text(json.dumps(classification_report(
        labels, predictions, labels=range(5), target_names=NAMES, output_dict=True, zero_division=0), indent=2) + "\n")
    for name in ("summary.json", "evaluation_summary.json"):
        (args.output_dir / name).write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    for name in ("source-dir", "prepared-dir", "checkpoint", "pretrain-config", "pretrain-status", "pretrain-vocabulary-shard",
                 "report-dir", "input-inventory", "plan"):
        pre.add_argument(f"--{name}", type=Path, required=True)
    pre.add_argument("--model-name", required=True)
    fit = sub.add_parser("check-fit"); fit.add_argument("--run-dir", type=Path, required=True)
    fit.add_argument("--epochs", type=int, required=True)
    export = sub.add_parser("export-test")
    for name in ("prepared-dir", "checkpoint", "output-dir", "plan", "transfer-report", "run-dir"):
        export.add_argument(f"--{name}", type=Path, required=True)
    export.add_argument("--model-name", required=True); export.add_argument("--batch-size", type=int, default=256)
    export.add_argument("--device", default="cuda")
    args = parser.parse_args()
    {"preflight": preflight, "check-fit": check_fit, "export-test": export_test}[args.command](args)


if __name__ == "__main__":
    main()
