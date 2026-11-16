#!/usr/bin/env python3
"""Gates and exports for fixed-fine-tuning-seed pretraining replicates.

The Q8, continuous-flat and hybrid backbones are unchanged.  Only pretrained
backbone weights are transferred; categorical and hybrid latent heads are not.
All classifiers use the frozen cluster-294 five-class fine-tuning protocol.
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

from heptokens.models.foundation_continuous_matched_multiclass import LitMatchedContinuousMulticlass
from heptokens.models.foundation_grouped_cls_classifier import LitGroupedCLSClassifier, _load_checkpoint_state
from hybrid_q4_downstream import classes, sequence_numpy, sha256, validate_data, vocabulary_semantics
from matched_multiclass_study import check_fit, validate_alignment
from plot_matched_multiclass_comparison import NAMES, compare, load_predictions
from prepare_q4_downstream_from_h5 import verify_inventory

FAMILIES = {"q8", "continuous_flat", "hybrid"}
CONFIGURATION_IDS = {
    "q8": "q8_parallel_v1_linear_8x32",
    "continuous_flat": "continuous_flat_v1_schema",
    "hybrid": "hybrid_coarse4_latent_v1_identity_8x32",
}
MODEL_KEYS = ("hidden_dim", "num_heads", "num_layers", "dropout", "max_seq_length",
              "num_type_ids", "use_type_embedding", "use_position_embedding")
TOKEN_KEYS = ("vocab_size", "max_quantizers", "quantizer_embedding_dim", "object_projection_mode")


def semantic_hparams(hparams, family):
    result = dict(hparams)
    if family == "q8":
        # The original Q8 exporter predates this option: its projection was
        # unconditionally linear.  Strict state loading still checks its weights.
        result.setdefault("object_projection_mode", "linear")
    if family == "continuous_flat":
        result.setdefault("feature_grouping", "schema")
    return result


def read_plan(path, key):
    plan = json.loads(Path(path).read_text())
    spec = plan["models"][key]
    if spec["model_family"] not in FAMILIES or spec["pretrain_seed"] not in (43, 44) or spec["finetune_seed"] != 42:
        raise ValueError("Require a Q8/continuous/hybrid pretraining seed 43 or 44 and fine-tuning seed 42")
    if spec["pretraining_configuration_id"] != CONFIGURATION_IDS[spec["model_family"]]:
        raise ValueError("Wrong pretrained representation/configuration ID")
    return plan, spec


def read_status(path):
    text = Path(path).read_text()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = dict(line.split("=", 1) for line in text.splitlines() if "=" in line)
    if not isinstance(value, dict):
        raise ValueError("Invalid pretraining job status")
    return value


def validate_pretrain_provenance(checkpoint, config, status, spec):
    for path, key in ((checkpoint, "checkpoint_sha256"), (config, "pretrain_config_sha256"),
                      (status, "pretrain_status_sha256")):
        if sha256(path) != spec[key]:
            raise ValueError(f"Pinned pretraining checksum mismatch: {key}")
    cfg, result = OmegaConf.load(config), read_status(status)
    required = {
        "seed": spec["pretrain_seed"], "datamodule.seed": spec["pretraining_data_seed"],
        "datamodule.batch_size": spec["pretraining_batch_size"],
        "trainer.devices": spec["pretraining_num_devices"], "trainer.max_epochs": 10,
    }
    for key, expected in required.items():
        if OmegaConf.select(cfg, key) != expected:
            raise ValueError(f"Wrong pretraining configuration: {key}")
    if spec["pretraining_batch_size"] * spec["pretraining_num_devices"] != spec["pretraining_global_batch_size"]:
        raise ValueError("Inconsistent declared pretraining global batch")
    for key, expected in (("exit_code", 0), ("seed", spec["pretrain_seed"]), ("epochs", 10)):
        if str(result.get(key)) != str(expected):
            raise ValueError(f"Pretraining status did not succeed/match: {key}")
    saved = _load_checkpoint_state(str(checkpoint))
    hp = semantic_hparams(saved["hyper_parameters"], spec["model_family"])
    if saved.get("epoch") != 9 or saved.get("global_step", 0) <= 0:
        raise ValueError("Pretraining did not complete ten epochs")
    for key, expected in spec["expected_hparams"].items():
        if key == "seed":
            observed = OmegaConf.select(cfg, "seed")
        else:
            observed = hp.get(key, "schema" if key == "feature_grouping" else None)
        if observed != expected:
            raise ValueError(f"Wrong pretrained {key}: {observed!r}, expected {expected!r}")
    if "seed" in hp and hp["seed"] != spec["pretrain_seed"]:
        raise ValueError("Checkpoint pretraining seed differs")
    # The saved resolved configuration and checkpoint must describe one model.
    for key in MODEL_KEYS + (TOKEN_KEYS if spec["model_family"] != "continuous_flat" else ("architecture",)):
        observed = OmegaConf.select(cfg, f"model.{key}")
        if observed is not None and observed != hp.get(key):
            raise ValueError(f"Pretraining configuration/checkpoint model mismatch: {key}")
    saved["hyper_parameters"] = hp
    return saved


def legacy_pretraining_vocabulary(path, status, spec, classification_vocabulary):
    """Recover the original Q8 mapping only from a checksum-pinned train shard.

    Legacy Q8 checkpoints did not embed vocabulary metadata.  The original
    successful job status pins its pretraining manifest, and that manifest's
    pinned first training shard supplies the mapping.  Current classification
    metadata alone is not evidence of which mapping pretraining consumed.
    """
    if spec["model_family"] != "q8" or path is None:
        raise ValueError("Legacy Q8 needs a verified original pretraining vocabulary shard")
    if sha256(path) != spec["pretraining_vocabulary_shard_sha256"]:
        raise ValueError("Original pretraining vocabulary shard checksum mismatch")
    if read_status(status).get("manifest_sha256") != spec["pretraining_manifest_sha256"]:
        raise ValueError("Original pretraining status/manifest provenance mismatch")
    encoded = (pq.ParquetFile(path).schema_arrow.metadata or {}).get(b"heptokens_token_vocabulary")
    if encoded is None:
        raise ValueError("Original pretraining shard lacks token-vocabulary metadata")
    vocabulary = json.loads(encoded)
    if vocabulary_semantics(vocabulary) != vocabulary_semantics(classification_vocabulary):
        raise ValueError("Original pretraining shard token-ID mapping differs from classification")
    return vocabulary


def validate_inputs(source, prepared, report, key, plan, spec, inventory):
    if spec["model_family"] == "continuous_flat":
        return validate_alignment(source, prepared, report, "continuous_flat", plan, inventory)
    verify_inventory(source, plan["source_manifest_sha256"], inventory, plan["input_inventory_sha256"])
    manifest, definitions, vocabulary = validate_data(source, plan["source_manifest_sha256"], "hybrid", report)
    if tuple(name for name, _ in definitions) != NAMES or manifest["split_counts"] != plan["split_counts"]:
        raise ValueError("Wrong matched five-class membership/counts")
    # Token classifiers read the original Q8 shards verbatim, not a conversion.
    if sha256(prepared / "manifest.json") != plan["source_manifest_sha256"]:
        raise ValueError("Token fine-tuning did not retain the canonical source manifest")
    expected = {p.relative_to(source): sha256(p) for split in ("train", "val", "test")
                for p in (source / split).rglob("*.parquet")}
    actual = {p.relative_to(prepared): sha256(p) for split in ("train", "val", "test")
              for p in (prepared / split).rglob("*.parquet")}
    if actual != expected:
        raise ValueError("Token fine-tuning shards/event identities differ from canonical source")
    return manifest, vocabulary, None


def classifier(spec, hp, schema, vocabulary, weights, checkpoint=None, legacy_pretraining_vocabulary=None):
    hp = semantic_hparams(hp, spec["model_family"])
    kwargs = {key: hp[key] for key in MODEL_KEYS if key in hp}
    common = dict(n_classes=5, class_weights=weights, freeze_backbone=False,
                  classifier_hidden_dim=None, learning_rate=1e-4,
                  backbone_ckpt_path=str(checkpoint) if checkpoint else None, **kwargs)
    if spec["model_family"] == "continuous_flat":
        return LitMatchedContinuousMulticlass(**common, continuous_schema=schema,
            architecture="flat", feature_grouping="schema")
    expected_projection = "linear" if spec["model_family"] == "q8" else "identity"
    if hp.get("object_projection_mode") != expected_projection or hp.get("max_quantizers") != 8:
        raise ValueError("Wrong Q8/hybrid projection or number of quantizers")
    pretrained_vocabulary = hp.get("token_vocabulary")
    if pretrained_vocabulary is None:
        if spec["model_family"] != "q8" or legacy_pretraining_vocabulary is None:
            raise ValueError("Missing pretrained token-ID mapping; a verified original Q8 vocabulary shard is required")
        pretrained_vocabulary = legacy_pretraining_vocabulary
    if vocabulary_semantics(pretrained_vocabulary) != vocabulary_semantics(vocabulary):
        raise ValueError("Classification token-ID mapping differs from pretrained vocabulary")
    model = LitGroupedCLSClassifier(**common, token_vocabulary=vocabulary,
        **{key: hp[key] for key in TOKEN_KEYS})
    if checkpoint:
        state = _load_checkpoint_state(str(checkpoint))["state_dict"]
        transferred = {name.removeprefix("model.backbone."): value for name, value in state.items()
                       if name.startswith("model.backbone.")}
        if not transferred or any(not torch.isfinite(v).all() for v in transferred.values() if v.is_floating_point()):
            raise ValueError("Missing/non-finite pretrained backbone weights")
        # Explicitly assert exact transfer, excluding every pretraining head.
        for name, value in transferred.items():
            if not torch.equal(model.backbone.state_dict()[name], value):
                raise ValueError("Pretraining backbone transfer changed a tensor")
    return model


def preflight(args):
    plan, spec = read_plan(args.plan, args.model_name)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    manifest, vocabulary, schema = validate_inputs(args.source_dir, args.prepared_dir, args.report_dir,
        args.model_name, plan, spec, args.input_inventory)
    saved = validate_pretrain_provenance(args.checkpoint, args.pretrain_config, args.pretrain_status, spec)
    weights = [manifest["recommended_cross_entropy_weights"][name] for name in NAMES]
    legacy_vocabulary, vocabulary_provenance = None, None
    if spec["model_family"] == "q8" and saved["hyper_parameters"].get("token_vocabulary") is None:
        legacy_vocabulary = legacy_pretraining_vocabulary(getattr(args, "pretrain_vocabulary_shard", None),
            args.pretrain_status, spec, vocabulary)
        vocabulary_provenance = dict(source="original pretraining shard metadata, not checkpoint metadata",
            pretraining_manifest_sha256=spec["pretraining_manifest_sha256"],
            shard_uri=spec["pretraining_vocabulary_shard"],
            shard_sha256=spec["pretraining_vocabulary_shard_sha256"],
            original_status_manifest_verified=True, classification_mapping_verified=True)
    model = classifier(spec, saved["hyper_parameters"], schema, vocabulary, weights, args.checkpoint,
                       legacy_pretraining_vocabulary=legacy_vocabulary)
    if not all(parameter.requires_grad for parameter in model.backbone.parameters()) or not isinstance(model.model.classifier, torch.nn.Linear):
        raise ValueError("Require a fully unfrozen backbone and linear classifier")
    report = dict(passed=True, model=args.model_name, model_family=spec["model_family"],
        pretrain_seed=spec["pretrain_seed"], finetune_seed=42, plan_sha256=sha256(args.plan),
        checkpoint_sha256=spec["checkpoint_sha256"], checkpoint_uri=spec["checkpoint"],
        checkpoint_local_path=str(args.checkpoint.resolve()),
        pretrain_config_sha256=spec["pretrain_config_sha256"], pretrain_status_sha256=spec["pretrain_status_sha256"],
        pretraining_job=spec["pretraining_job"], pretraining_configuration_id=spec["pretraining_configuration_id"],
        pretraining_num_devices=spec["pretraining_num_devices"], pretraining_batch_size=spec["pretraining_batch_size"],
        pretraining_global_batch_size=spec["pretraining_global_batch_size"], pretraining_data_seed=spec["pretraining_data_seed"],
        pretraining_vocabulary_provenance=vocabulary_provenance,
        pretraining_topology=dict(num_devices=spec["pretraining_num_devices"],
            batch_size_per_gpu=spec["pretraining_batch_size"], global_batch_size=spec["pretraining_global_batch_size"],
            data_seed=spec["pretraining_data_seed"]),
        pretrain_epoch=saved["epoch"], pretrain_global_step=saved["global_step"],
        source_manifest_sha256=plan["source_manifest_sha256"], manifest_sha256=sha256(args.prepared_dir / "manifest.json"),
        classes=NAMES, class_weights=weights, split_counts=manifest["split_counts"], classifier_head="linear_256_to_5",
        backbone_frozen=False, protocol=plan["protocol"],
        caveats=(["Hybrid seeds 43/44 used global batch 3582 (6 x 597); seed 42 used 3584 (8 x 448). This is not a strictly seed-only pretraining variation."]
                 if spec["model_family"] == "hybrid" else []))
    (args.report_dir / "transfer_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS canonical samples, pinned pretraining seed/configuration, strict backbone-only transfer and fixed fine-tuning seed 42")
    return report


def validate_export_provenance(args, plan, spec, manifest):
    report = json.loads(args.transfer_report.read_text())
    for key, expected in (("passed", True), ("model", args.model_name), ("plan_sha256", sha256(args.plan)),
                          ("pretrain_seed", spec["pretrain_seed"]), ("finetune_seed", 42),
                          ("checkpoint_sha256", spec["checkpoint_sha256"]),
                          ("manifest_sha256", sha256(args.prepared_dir / "manifest.json"))):
        if report.get(key) != expected:
            raise ValueError(f"Unverified pretraining/fine-tuning provenance: {key}")
    if manifest["split_counts"] != plan["split_counts"] or tuple(name for name, _ in classes(manifest)) != NAMES:
        raise ValueError("Held-out manifest/classes changed")
    cfg = OmegaConf.load(args.run_dir / "full_config.yaml")
    expected = {"seed": 42, "datamodule.seed": 42, "datamodule.batch_size": 256, "datamodule.num_workers": 0,
        "datamodule.n_classes": 5, "datamodule.stream_batch_size": 4096,
        "datamodule.shuffle_buffer_size": 8192, "model.freeze_backbone": False,
        "model.classifier_hidden_dim": None, "model.learning_rate": 1e-4,
        "model.optimizer._target_": "torch.optim.AdamW", "model.optimizer.lr": 1e-4,
        "model.optimizer.weight_decay": 1e-2, "model.scheduler.warmup_epochs": 1,
        "model.scheduler.min_lr": 1e-6, "trainer.max_epochs": 10, "trainer.devices": 1,
        "trainer.val_check_interval": 1.0, "trainer.limit_val_batches": 1.0,
        "callbacks.best_checkpoint.monitor": "valid/auc", "callbacks.best_checkpoint.mode": "max"}
    for key, value in expected.items():
        if OmegaConf.select(cfg, key) != value:
            raise ValueError(f"Fine-tuning protocol drift in export: {key}")
    if str(Path(OmegaConf.select(cfg, "model.backbone_ckpt_path")).resolve()) != report["checkpoint_local_path"]:
        raise ValueError("Fine-tuning loaded a different pretrained checkpoint path")
    if sha256(Path(report["checkpoint_local_path"])) != spec["checkpoint_sha256"]:
        raise ValueError("Pretrained source checkpoint changed after transfer")
    if args.checkpoint.resolve() != (args.run_dir / "checkpoints/best.ckpt").resolve():
        raise ValueError("Test export must use this run's validation-best checkpoint")
    return report


def export_test(args):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, roc_auc_score
    plan, spec = read_plan(args.plan, args.model_name)
    manifest = json.loads((args.prepared_dir / "manifest.json").read_text())
    transfer = validate_export_provenance(args, plan, spec, manifest)
    saved = _load_checkpoint_state(str(args.checkpoint)); hp = dict(saved["hyper_parameters"])
    if str(Path(hp.get("backbone_ckpt_path", "")).resolve()) != transfer["checkpoint_local_path"]:
        raise ValueError("Fine-tuned checkpoint saved a different pretrained source")
    cls = LitMatchedContinuousMulticlass if spec["model_family"] == "continuous_flat" else LitGroupedCLSClassifier
    allowed = inspect.signature(cls.__init__).parameters
    kwargs = {key: value for key, value in hp.items() if key in allowed}
    kwargs.update(backbone_ckpt_path=None, optimizer=None, scheduler=None)
    model = cls(**kwargs); model.load_state_dict(saved["state_dict"], strict=True)
    weights = [manifest["recommended_cross_entropy_weights"][name] for name in NAMES]
    if model.n_classes != 5 or not isinstance(model.model.classifier, torch.nn.Linear) or model.model.classifier.out_features != 5:
        raise ValueError("Not a linear five-class classifier")
    if not all(p.requires_grad for p in model.backbone.parameters()) or not np.allclose(model.class_weights.cpu(), weights):
        raise ValueError("Wrong class weights or frozen backbone in fine-tuned checkpoint")
    for key in MODEL_KEYS + (TOKEN_KEYS if spec["model_family"] != "continuous_flat" else ("architecture",)):
        if key in spec["expected_hparams"] and hp.get(key) != spec["expected_hparams"][key]:
            raise ValueError(f"Fine-tuned checkpoint representation mismatch: {key}")
    model.eval().to(args.device); torch.set_float32_matmul_precision("medium")
    keys = [("tokens", np.int64), ("mask", bool), ("type_ids", np.int64)]
    if spec["model_family"] == "continuous_flat":
        keys = [("continuous_features", np.float32), ("continuous_feature_mask", bool),
                ("position_role_ids", np.int64), ("mask", bool), ("type_ids", np.int64)]
    parts = {name: [] for name in ("probabilities", "labels", "source_file", "event_index")}
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
                    for key, value in values.items():
                        parts[key].append(value)
                    count += len(labels)
            if count != manifest["split_counts"]["test"][name]:
                raise ValueError(f"Incomplete held-out split: {name}")
    arrays = {key: np.concatenate(value) for key, value in parts.items()}
    arrays.update(class_names=np.asarray(NAMES), predictions=arrays["probabilities"].argmax(1))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output_dir / "test_predictions.npz", **arrays)
    checked = load_predictions(args.output_dir)
    labels, probabilities = checked["labels"], checked["probabilities"]
    if len(labels) != manifest["split_counts"]["test"]["total"]:
        raise ValueError("Incomplete/duplicated held-out test events")
    predictions = probabilities.argmax(1)
    rows = compare([(f'{spec["model_family"]} pretrain seed {spec["pretrain_seed"]}', args.output_dir)],
                   args.output_dir / "plots", expected_events=len(labels))
    # Export the scalar/run metadata separately; no prediction-score averaging.
    report = {key: transfer[key] for key in ("model", "model_family", "pretrain_seed", "finetune_seed",
        "checkpoint_uri", "checkpoint_local_path", "pretrain_config_sha256", "pretrain_status_sha256",
        "pretraining_job", "pretraining_configuration_id", "pretraining_num_devices", "pretraining_batch_size",
        "pretraining_global_batch_size", "pretraining_data_seed", "pretraining_topology", "pretraining_vocabulary_provenance",
        "source_manifest_sha256", "caveats")}
    report.update(pretraining_checkpoint_sha256=spec["checkpoint_sha256"], checkpoint_selection="validation-best",
        checkpoint_sha256=sha256(args.checkpoint), finetune_config_sha256=sha256(args.run_dir / "full_config.yaml"),
        manifest_sha256=sha256(args.prepared_dir / "manifest.json"), class_names=NAMES,
        test_events=len(labels), test_auc=float(roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")),
        macro_ovr_auc=float(roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")),
        weighted_ovr_auc=float(roc_auc_score(labels, probabilities, multi_class="ovr", average="weighted")),
        accuracy=float(accuracy_score(labels, predictions)), balanced_accuracy=float(balanced_accuracy_score(labels, predictions)),
        per_class_auc={row["class_name"]: row["auc"] for row in rows},
        confusion_matrix=confusion_matrix(labels, predictions, labels=range(5)).tolist(),
        normalized_confusion_matrix=confusion_matrix(labels, predictions, labels=range(5), normalize="true").tolist())
    classification = classification_report(labels, predictions, labels=range(5), target_names=NAMES,
                                         output_dict=True, zero_division=0)
    (args.output_dir / "classification_report.json").write_text(json.dumps(classification, indent=2) + "\n")
    for name in ("summary.json", "evaluation_summary.json"):
        (args.output_dir / name).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    for name in ("source-dir", "prepared-dir", "checkpoint", "pretrain-config", "pretrain-status",
                 "report-dir", "input-inventory", "plan"):
        pre.add_argument(f"--{name}", type=Path, required=True)
    pre.add_argument("--model-name", required=True)
    pre.add_argument("--pretrain-vocabulary-shard", type=Path,
                     help="Pinned original training shard, required only for legacy Q8 checkpoints without vocabulary metadata")
    fit = sub.add_parser("check-fit")
    fit.add_argument("--run-dir", type=Path, required=True); fit.add_argument("--epochs", type=int, required=True)
    export = sub.add_parser("export-test")
    for name in ("prepared-dir", "checkpoint", "output-dir", "plan", "transfer-report", "run-dir"):
        export.add_argument(f"--{name}", type=Path, required=True)
    export.add_argument("--model-name", required=True)
    export.add_argument("--batch-size", type=int, default=256); export.add_argument("--device", default="cuda")
    args = parser.parse_args()
    {"preflight": preflight, "check-fit": check_fit, "export-test": export_test}[args.command](args)


if __name__ == "__main__":
    main()
