#!/usr/bin/env python3
"""Isolated, provenance-checked scratch and frozen five-class controls.

No old unfrozen fine-tuning gates are reused.  Scratch never opens a pretrained
checkpoint; frozen controls transfer only the exact pinned backbone, keep it in
evaluation mode, and optimize the linear five-class head alone.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf

from heptokens.models.foundation_continuous_matched_multiclass import LitMatchedContinuousMulticlass
from heptokens.models.foundation_grouped_cls_classifier import LitGroupedCLSClassifier, _load_checkpoint_state
from heptokens.models.foundation_multiclass_controls import (
    LitContinuousMulticlassControl, LitTokenMulticlassControl,
)
from hybrid_q4_downstream import classes, sequence_numpy, sha256, vocabulary_semantics
from matched_multiclass_study import validate_alignment
from plot_matched_multiclass_comparison import NAMES, compare, load_predictions
import projected_q1_seed_multiclass as q1_study
import q4_latest_hier_seed_multiclass as q4_study
import seed_multiclass_study as seed_study

CONFIGURATION_IDS = {
    "q1_projected": "q1_projected_v1_linear_1x32",
    "q4": "q4_independent_tokenizers_v1",
    "q8": "q8_parallel_v1_linear_8x32",
    "continuous_flat": "continuous_flat_v1_schema",
    "hierarchical_kin_aux": "hierarchical_kin_aux_287",
}
DATA_KINDS = {"q1_projected": "q1", "q4": "q4", "q8": "q8",
              "continuous_flat": "continuous", "hierarchical_kin_aux": "continuous"}
BACKBONE_KEYS = seed_study.MODEL_KEYS + seed_study.TOKEN_KEYS + (
    "architecture", "feature_grouping", "inner_num_heads", "inner_num_layers")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def read_plan(path, run_key):
    plan = json.loads(Path(path).read_text())
    run = plan["runs"][run_key]
    family, mode = run["model_family"], run["control_mode"]
    if family not in CONFIGURATION_IDS or mode not in {"scratch", "frozen"}:
        raise ValueError("Only the five selected scratch/frozen representations are supported")
    if run.get("training_seed") not in (42, 43, 44):
        raise ValueError("Require training seed 42, 43 or 44")
    representation = plan["representations"][family]
    if representation.get("configuration_id") != CONFIGURATION_IDS[family]:
        raise ValueError("Wrong representation configuration ID")
    if representation.get("data_kind") != DATA_KINDS[family]:
        raise ValueError("Wrong representation data kind")
    source = None
    if mode == "scratch":
        if run.get("source_key") is not None or run.get("pretrain_seed") is not None:
            raise ValueError("Scratch must have no pretraining source or pretraining seed")
    else:
        source = plan["sources"][run["source_key"]]
        if (run.get("pretrain_seed") not in (42, 43, 44)
                or source.get("pretrain_seed") != run["pretrain_seed"]
                or source.get("model_family") != family
                or run["source_key"] != f"{family}_p{run['pretrain_seed']}"):
            raise ValueError("Frozen source family/seed does not match this run")
        if source.get("pretraining_configuration_id") != CONFIGURATION_IDS[family]:
            raise ValueError("Wrong frozen source configuration ID")
        if family in {"q4", "hierarchical_kin_aux"} and run["pretrain_seed"] in (43, 44) \
                and not source.get("pretrain_completion_sha256"):
            raise ValueError("Q4/latest-hierarchical seeds 43/44 require a strict completion receipt")
        for key, expected in representation["expected_hparams"].items():
            if source["expected_hparams"].get(key) != expected:
                raise ValueError(f"Source/representation configuration differs: {key}")
    return plan, run, representation, source


def backbone_fingerprint(state):
    """Hash every tensor name, type, shape and exact byte, independent of torch.save."""
    if not state:
        raise ValueError("Missing backbone tensors")
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Non-finite backbone tensor: {name}")
        descriptor = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":"))
        digest.update(descriptor.encode())
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def checkpoint_backbone(saved):
    prefix = "model.backbone."
    return {key[len(prefix):]: value for key, value in saved["state_dict"].items()
            if key.startswith(prefix)}


def validate_inputs(source, prepared, report_dir, family, plan, inventory):
    # Only input-validation functions are shared with the old unfrozen studies.
    # The historical q1_direct256 branch denotes the data shape, not our model.
    if family == "q1_projected":
        return validate_alignment(source, prepared, report_dir, "q1_direct256", plan, inventory)
    spec = {"model_family": family}
    if family in {"q4", "hierarchical_kin_aux"}:
        return q4_study.validate_inputs(source, prepared, report_dir, family, plan, spec, inventory)
    return seed_study.validate_inputs(source, prepared, report_dir, family, plan, spec, inventory)


def _semantic_hparams(hp, family):
    if family == "q1_projected":
        return q1_study.semantic_hparams(hp)
    return seed_study.semantic_hparams(hp, family)


def _constructor_keys(family):
    base = LitMatchedContinuousMulticlass if DATA_KINDS[family] == "continuous" else LitGroupedCLSClassifier
    control = LitContinuousMulticlassControl if DATA_KINDS[family] == "continuous" else LitTokenMulticlassControl
    return set(inspect.signature(base.__init__).parameters) | set(inspect.signature(control.__init__).parameters)


def classifier(run, hp, schema, vocabulary, weights, checkpoint=None):
    family, mode = run["model_family"], run["control_mode"]
    hp = _semantic_hparams(hp, family)
    if mode == "scratch" and checkpoint is not None:
        raise ValueError("Scratch must never read or load a pretrained checkpoint")
    kwargs = {key: value for key, value in hp.items() if key in _constructor_keys(family)}
    for key in ("self", "data_sample", "args", "kwargs"):
        kwargs.pop(key, None)
    kwargs.update(n_classes=5, class_weights=weights, control_mode=mode,
        freeze_backbone=mode == "frozen", classifier_hidden_dim=None, learning_rate=1e-4,
        backbone_ckpt_path=str(checkpoint) if checkpoint else None)
    if DATA_KINDS[family] == "continuous":
        expected = ("flat", "schema") if family == "continuous_flat" else ("hierarchical", "kinematics_auxiliary_v1")
        if (hp.get("architecture"), hp.get("feature_grouping", "schema")) != expected:
            raise ValueError("Require the selected continuous architecture/grouping")
        kwargs.update(continuous_schema=schema, architecture=expected[0], feature_grouping=expected[1])
        model = LitContinuousMulticlassControl(**kwargs)
    else:
        required = {"q1_projected": (1, "linear"), "q4": (4, "identity"), "q8": (8, "linear")}[family]
        if (hp.get("max_quantizers"), hp.get("object_projection_mode")) != required:
            raise ValueError("Wrong selected token quantizers/projection")
        if vocabulary is None or vocabulary.get("max_quantizers") != required[0]:
            raise ValueError("Wrong classification vocabulary")
        if hp.get("vocab_size") != vocabulary.get("vocab_size"):
            raise ValueError("Classification vocabulary size differs from the architecture")
        if family == "q1_projected" and hp.get("quantizer_embedding_dim") != 32:
            raise ValueError("Projected Q1 must retain its 32D embedding and learned projection")
        kwargs["token_vocabulary"] = vocabulary
        model = LitTokenMulticlassControl(**kwargs)
    validate_model(model, run, weights)
    return model


def validate_model(model, run, weights):
    frozen = run["control_mode"] == "frozen"
    head = model.model.classifier
    if (model.n_classes != 5 or not isinstance(head, torch.nn.Linear)
            or head.out_features != 5 or not np.allclose(model.class_weights.cpu(), weights)):
        raise ValueError("Require the matched weighted-CE linear five-class head")
    if any(parameter.requires_grad == frozen for parameter in model.backbone.parameters()):
        raise ValueError("Wrong backbone trainability for this control")
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if frozen and set(names) != {"model.classifier.weight", "model.classifier.bias"}:
        raise ValueError("Frozen optimizer must train only the linear classifier")
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if frozen and count != 5 * (head.in_features + 1):
        raise ValueError("Frozen trainable parameter count differs from the linear head")
    return names, count


def preflight(args):
    plan, run, representation, source = read_plan(args.plan, args.run_key)
    if source is None:
        for name in ("checkpoint", "pretrain_config", "pretrain_status", "pretrain_completion", "pretrain_vocabulary_shard"):
            if getattr(args, name, None) is not None:
                raise ValueError(f"Scratch cannot receive a pretrained source: {name}")
    else:
        for name in ("checkpoint", "pretrain_config", "pretrain_status"):
            if getattr(args, name, None) is None:
                raise ValueError(f"Frozen control requires pinned pretraining input: {name}")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    manifest, vocabulary, schema = validate_inputs(args.source_dir, args.prepared_dir,
        args.report_dir, run["model_family"], plan, args.input_inventory)
    weights = [manifest["recommended_cross_entropy_weights"][name] for name in NAMES]
    saved, receipt, vocabulary_provenance, checkpoint = None, None, None, None
    model_config = representation.get("model_config", {})
    hp = dict(model_config) if isinstance(model_config, dict) else {}
    hp.update(representation["expected_hparams"])
    if source is not None:
        checkpoint = args.checkpoint
        if source.get("pretrain_completion_sha256"):
            if getattr(args, "pretrain_completion", None) is None:
                raise ValueError("Frozen source requires its pinned completion receipt")
            saved, receipt = q4_study.validate_source_receipt(checkpoint, args.pretrain_config,
                args.pretrain_status, args.pretrain_completion, source)
        else:
            validator = q1_study.validate_pretrain_provenance if run["model_family"] == "q1_projected" else seed_study.validate_pretrain_provenance
            saved = validator(checkpoint, args.pretrain_config, args.pretrain_status, source)
        hp = _semantic_hparams(saved["hyper_parameters"], run["model_family"])
        if DATA_KINDS[run["model_family"]] != "continuous":
            original = hp.get("token_vocabulary")
            if original is None:
                shard = getattr(args, "pretrain_vocabulary_shard", None)
                recover = q1_study.verified_vocabulary if run["model_family"] == "q1_projected" else seed_study.legacy_pretraining_vocabulary
                original = recover(shard, args.pretrain_status, source, vocabulary)
                vocabulary_provenance = dict(source="original pretraining shard metadata, not classification metadata",
                    pretraining_manifest_sha256=source["pretraining_manifest_sha256"],
                    shard_uri=source["pretraining_vocabulary_shard"],
                    shard_sha256=source["pretraining_vocabulary_shard_sha256"],
                    original_status_manifest_verified=True, classification_mapping_verified=True)
            if vocabulary_semantics(original) != vocabulary_semantics(vocabulary):
                raise ValueError("Pretraining/classification token-ID mapping differs")
    model = classifier(run, hp, schema, vocabulary, weights, checkpoint)
    source_hash = backbone_fingerprint(checkpoint_backbone(saved)) if saved else None
    observed = backbone_fingerprint(model.backbone.state_dict())
    if source_hash is not None and observed != source_hash:
        raise ValueError("Strict backbone transfer changed or omitted tensors")
    names, count = validate_model(model, run, weights)
    provenance = None if source is None else dict(source, checkpoint_local_path=str(checkpoint.resolve()),
        pretraining_completion=receipt, pretraining_vocabulary_provenance=vocabulary_provenance,
        pretrain_epoch=saved["epoch"], pretrain_global_step=saved["global_step"])
    report = dict(passed=True, run_key=args.run_key, model=args.run_key,
        model_family=run["model_family"], control_mode=run["control_mode"],
        training_seed=run["training_seed"], pretrain_seed=run.get("pretrain_seed"),
        source_key=run.get("source_key"), source_provenance=provenance,
        plan_sha256=sha256(args.plan), source_manifest_sha256=plan["source_manifest_sha256"],
        input_inventory_sha256=plan["input_inventory_sha256"],
        prepared_dir_local_path=str(args.prepared_dir.resolve()),
        manifest_sha256=sha256(args.prepared_dir / "manifest.json"), classes=NAMES,
        class_weights=weights, split_counts=manifest["split_counts"], protocol=plan["protocol"],
        configuration_id=representation["configuration_id"], backbone_frozen=source is not None,
        classifier_head=f"linear_{model.model.classifier.in_features}_to_5",
        trainable_parameter_names=names, trainable_parameters=count,
        initial_backbone_sha256=observed, source_backbone_sha256=source_hash)
    write_json(args.report_dir / "transfer_validation.json", report)
    print(f"PASS {args.run_key}: canonical five-class inputs and {run['control_mode']} initialization")
    return report


def validate_transfer(plan_path, run_key, transfer_path):
    plan, run, representation, source = read_plan(plan_path, run_key)
    report = json.loads(Path(transfer_path).read_text())
    for key, expected in (("passed", True), ("run_key", run_key), ("plan_sha256", sha256(plan_path)),
            ("model_family", run["model_family"]), ("control_mode", run["control_mode"]),
            ("training_seed", run["training_seed"]), ("pretrain_seed", run.get("pretrain_seed")),
            ("source_key", run.get("source_key")), ("source_manifest_sha256", plan["source_manifest_sha256"]),
            ("input_inventory_sha256", plan["input_inventory_sha256"]), ("split_counts", plan["split_counts"]),
            ("protocol", plan["protocol"]), ("configuration_id", representation["configuration_id"]),
            ("classes", list(NAMES)), ("backbone_frozen", source is not None)):
        if report.get(key) != expected:
            raise ValueError(f"Unverified control transfer provenance: {key}")
    if source is None:
        if report.get("source_provenance") is not None or report.get("source_backbone_sha256") is not None:
            raise ValueError("Scratch report illegally identifies a pretrained source")
    else:
        provenance = report.get("source_provenance") or {}
        if any(provenance.get(key) != value for key, value in source.items()):
            raise ValueError("Frozen transfer report source differs from the pinned plan")
        if not report.get("source_backbone_sha256"):
            raise ValueError("Missing verified frozen source backbone fingerprint")
    return plan, run, representation, source, report


def optimizer_audit(optimizers, model):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if len(optimizers) != 1:
        raise ValueError("Require exactly one control optimizer")
    optimizer = optimizers[0]
    selected = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    identities = [id(parameter) for parameter in selected]
    if len(identities) != len(set(identities)) or set(identities) != trainable:
        raise ValueError("Optimizer parameters differ from the exact trainable-only parameter set")
    if any(id(parameter) not in trainable for parameter in optimizer.state):
        raise ValueError("Optimizer state contains a frozen/non-training parameter")
    return dict(parameter_groups=[[names[id(parameter)] for parameter in group["params"]]
                                 for group in optimizer.param_groups],
        state_parameter_names=sorted(names[id(parameter)] for parameter in optimizer.state),
        trainable_parameter_names=sorted(names[identity] for identity in trainable),
        trainable_parameters=sum(parameter.numel() for parameter in selected))


class ControlsAuditCallback(Callback):
    """Record actual fit initialization/finalization, not a fresh probe model."""

    def __init__(self, plan, run_key, transfer_report):
        super().__init__()
        self.plan_path, self.run_key, self.transfer_path = Path(plan), run_key, Path(transfer_report)
        self.initial = None
        self.audit = None

    def on_fit_start(self, trainer, pl_module):
        _, run, _, source, report = validate_transfer(self.plan_path, self.run_key, self.transfer_path)
        hp = pl_module.hparams
        if hp.get("control_mode") != run["control_mode"] or bool(hp.get("freeze_backbone")) != (source is not None):
            raise ValueError("Actual fitted model is not the requested control mode")
        validate_model(pl_module, run, report["class_weights"])
        configured_source = hp.get("backbone_ckpt_path")
        config_path = Path(trainer.default_root_dir) / "full_config.yaml"
        if config_path.exists():
            _reject_alternate_weights(OmegaConf.load(config_path))
        if getattr(trainer, "ckpt_path", None) is not None:
            raise ValueError("Control fit cannot resume a classifier checkpoint")
        if source is None:
            if configured_source is not None:
                raise ValueError("Scratch fit cannot load a source or resume checkpoint")
        else:
            expected = report["source_provenance"]["checkpoint_local_path"]
            if configured_source is None or Path(configured_source).resolve() != Path(expected).resolve():
                raise ValueError("Actual frozen fit loaded a different source checkpoint")
            if sha256(Path(expected)) != source["checkpoint_sha256"]:
                raise ValueError("Pinned frozen checkpoint changed after preflight")
            source_hash = backbone_fingerprint(checkpoint_backbone(_load_checkpoint_state(expected)))
            if source_hash != report["source_backbone_sha256"]:
                raise ValueError("Actual frozen source backbone fingerprint differs")
        self.initial = backbone_fingerprint(pl_module.backbone.state_dict())
        if source is not None and self.initial != report["source_backbone_sha256"]:
            raise ValueError("Actual frozen initialization differs from the pinned source")
        self.audit = dict(passed=False, run_key=self.run_key, model_family=run["model_family"],
            control_mode=run["control_mode"], training_seed=run["training_seed"],
            pretrain_seed=run.get("pretrain_seed"), plan_sha256=sha256(self.plan_path),
            transfer_report_sha256=sha256(self.transfer_path),
            source_backbone_sha256=report["source_backbone_sha256"], initial_backbone_sha256=self.initial,
            optimizer=optimizer_audit(trainer.optimizers, pl_module))
        self._write(trainer)

    def on_train_epoch_start(self, trainer, pl_module):
        if pl_module.hparams.get("control_mode") == "frozen":
            if any(module.training for module in pl_module.backbone.modules()):
                raise ValueError("Frozen backbone must remain in evaluation mode, including dropout")

    def on_after_backward(self, trainer, pl_module):
        if pl_module.hparams.get("control_mode") == "frozen" and any(
                parameter.grad is not None for parameter in pl_module.backbone.parameters()):
            raise ValueError("Frozen backbone received a gradient")

    def on_fit_end(self, trainer, pl_module):
        if self.audit is None or self.initial is None:
            raise ValueError("Missing actual control initialization audit")
        final = backbone_fingerprint(pl_module.backbone.state_dict())
        frozen = self.audit["control_mode"] == "frozen"
        if (final != self.initial) == frozen:
            raise ValueError("Frozen backbone changed" if frozen else "Scratch backbone did not train/change")
        self.audit.update(passed=True, final_backbone_sha256=final, backbone_changed=final != self.initial,
            global_step=int(trainer.global_step), optimizer=optimizer_audit(trainer.optimizers, pl_module))
        self._write(trainer)

    def _write(self, trainer):
        if getattr(trainer, "is_global_zero", True):
            root = Path(trainer.default_root_dir)
            root.mkdir(parents=True, exist_ok=True)
            write_json(root / "control_integrity.json", self.audit)


def _reject_alternate_weights(cfg):
    for key in ("ckpt_path", "weight_ckpt_path", "full_resume", "model.weight_ckpt_path", "model.full_resume"):
        if OmegaConf.select(cfg, key) not in (None, False):
            raise ValueError(f"Control cannot resume or load alternate weights: {key}")


def validate_config(path, run, representation, report, epochs):
    cfg = OmegaConf.load(path)
    frozen = run["control_mode"] == "frozen"
    expected = {"seed": run["training_seed"], "datamodule.seed": run["training_seed"],
        "datamodule.batch_size": 256, "datamodule.num_workers": 0, "datamodule.n_classes": 5,
        "datamodule.stream_batch_size": 4096, "datamodule.shuffle_buffer_size": 8192,
        "model.control_mode": run["control_mode"], "model.freeze_backbone": frozen,
        "model.classifier_hidden_dim": None, "model.n_classes": 5, "model.learning_rate": 1e-4,
        "model.optimizer._target_": "torch.optim.AdamW", "model.optimizer.lr": 1e-4,
        "model.optimizer.weight_decay": 1e-2, "model.scheduler.warmup_epochs": 1,
        "model.scheduler.min_lr": 1e-6, "trainer.max_epochs": epochs, "trainer.devices": 1,
        "trainer.val_check_interval": 1.0, "trainer.limit_val_batches": 1.0,
        "callbacks.best_checkpoint.monitor": "valid/auc", "callbacks.best_checkpoint.mode": "max"}
    for key, value in expected.items():
        if OmegaConf.select(cfg, key) != value:
            raise ValueError(f"Control training protocol drift: {key}")
    _reject_alternate_weights(cfg)
    for key in BACKBONE_KEYS:
        if key in representation["expected_hparams"]:
            observed = OmegaConf.select(cfg, f"model.{key}")
            if observed != representation["expected_hparams"][key]:
                raise ValueError(f"Control architecture drift: {key}")
    if not np.allclose(OmegaConf.select(cfg, "model.class_weights"), report["class_weights"]):
        raise ValueError("Control class weights changed")
    path = OmegaConf.select(cfg, "model.backbone_ckpt_path")
    if frozen:
        if path is None or Path(path).resolve() != Path(report["source_provenance"]["checkpoint_local_path"]).resolve():
            raise ValueError("Frozen config loaded another source checkpoint")
    elif path is not None:
        raise ValueError("Scratch config cannot load a pretrained backbone")
    prepared = OmegaConf.select(cfg, "datamodule.prepared_dir")
    if prepared is None:
        raise ValueError("Missing actual control prepared-data path")
    prepared, canonical = Path(prepared).resolve(), Path(report["prepared_dir_local_path"]).resolve()
    if epochs == 10:
        if prepared != canonical or sha256(prepared / "manifest.json") != report["manifest_sha256"]:
            raise ValueError("Actual full fit did not read the verified canonical prepared inputs")
    else:
        smoke = json.loads((prepared / "manifest.json").read_text())
        if smoke.get("smoke_only", {}).get("source_manifest_sha256") != report["manifest_sha256"]:
            raise ValueError("Smoke inputs do not identify the verified canonical prepared source")
        if any((prepared / split).resolve() != (canonical / split).resolve() for split in ("train", "test")):
            raise ValueError("Smoke training/test inputs are not read-only canonical references")
    return cfg


def _validate_checkpoint_optimizer(saved, model, integrity):
    states = saved.get("optimizer_states", [])
    if len(states) != 1:
        raise ValueError("Missing/extra saved control optimizer")
    optimizer = states[0]
    groups = optimizer.get("param_groups", [])
    ids = [identity for group in groups for identity in group["params"]]
    expected_groups = integrity["optimizer"]["parameter_groups"]
    expected_names = [name for group in expected_groups for name in group]
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if ([len(group["params"]) for group in groups] != [len(group) for group in expected_groups]
            or len(ids) != len(set(ids))
            or len(ids) != len(trainable) or len(expected_names) != len(set(expected_names))
            or set(expected_names) != set(trainable)):
        raise ValueError("Saved optimizer groups include frozen or omit trainable parameters")
    if not set(optimizer.get("state", {})).issubset(ids):
        raise ValueError("Saved optimizer state contains frozen/non-training parameters")
    parameter_by_id = {identity: trainable[name] for identity, name in zip(ids, expected_names, strict=True)}
    for identity, state in optimizer.get("state", {}).items():
        if any(isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all()
               for value in state.values()):
            raise ValueError("Saved optimizer contains non-finite state")
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if key in state and state[key].shape != parameter_by_id[identity].shape:
                raise ValueError("Saved optimizer state shape differs from its trainable-only parameter")


def restore_classifier(saved, run, representation, weights):
    hp = _semantic_hparams(saved["hyper_parameters"], run["model_family"])
    if hp.get("control_mode") != run["control_mode"] or bool(hp.get("freeze_backbone")) != (run["control_mode"] == "frozen"):
        raise ValueError("Saved classifier control mode differs")
    for key in BACKBONE_KEYS:
        if key in representation["expected_hparams"] and hp.get(key) != representation["expected_hparams"][key]:
            raise ValueError(f"Saved classifier architecture differs: {key}")
    kwargs = {key: value for key, value in hp.items() if key in _constructor_keys(run["model_family"])}
    kwargs.update(backbone_ckpt_path=None, optimizer=None, scheduler=None)
    cls = LitContinuousMulticlassControl if DATA_KINDS[run["model_family"]] == "continuous" else LitTokenMulticlassControl
    model = cls(**kwargs)
    model.load_state_dict(saved["state_dict"], strict=True)
    validate_model(model, run, weights)
    return model


def check_fit(args):
    if args.epochs not in (1, 10):
        raise ValueError("Only the 1-epoch smoke or 10-epoch full control protocol is supported")
    _, run, representation, source, report = validate_transfer(args.plan, args.run_key, args.transfer_report)
    validate_config(args.run_dir / "full_config.yaml", run, representation, report, args.epochs)
    integrity = json.loads((args.run_dir / "control_integrity.json").read_text())
    for key, expected in (("passed", True), ("run_key", args.run_key), ("control_mode", run["control_mode"]),
            ("training_seed", run["training_seed"]), ("pretrain_seed", run.get("pretrain_seed")),
            ("plan_sha256", sha256(args.plan)), ("transfer_report_sha256", sha256(args.transfer_report)),
            ("source_backbone_sha256", report["source_backbone_sha256"])):
        if integrity.get(key) != expected:
            raise ValueError(f"Missing/invalid actual fit integrity: {key}")
    frozen = source is not None
    initial, final = integrity.get("initial_backbone_sha256"), integrity.get("final_backbone_sha256")
    if not initial or not final or (initial == final) != frozen or integrity.get("backbone_changed") != (not frozen):
        raise ValueError("Actual backbone initialization/finalization failed the control invariant")
    if frozen:
        checkpoint = getattr(args, "source_checkpoint", None) or Path(report["source_provenance"]["checkpoint_local_path"])
        if sha256(checkpoint) != source["checkpoint_sha256"]:
            raise ValueError("Pinned source checkpoint changed")
        if backbone_fingerprint(checkpoint_backbone(_load_checkpoint_state(str(checkpoint)))) != initial or initial != report["source_backbone_sha256"]:
            raise ValueError("Frozen source/initial/final backbone fingerprints differ")
    elif getattr(args, "source_checkpoint", None) is not None:
        raise ValueError("Scratch check-fit must not read a pretrained checkpoint")
    expected_steps = 200 if args.epochs == 1 else 66430
    if integrity.get("global_step") != expected_steps:
        raise ValueError("Actual fit audit has an incomplete optimizer-step budget")
    loaded = {}
    for name in ("last", "best"):
        saved = _load_checkpoint_state(str(args.run_dir / f"checkpoints/{name}.ckpt"))
        if any(value.is_floating_point() and not torch.isfinite(value).all() for value in saved["state_dict"].values()):
            raise ValueError("Non-finite control checkpoint")
        if name == "last" and (saved.get("epoch") != args.epochs - 1 or saved.get("global_step") != expected_steps):
            raise ValueError("Incomplete control epochs/optimizer-step budget")
        hp_source = saved["hyper_parameters"].get("backbone_ckpt_path")
        if frozen:
            if hp_source is None or Path(hp_source).resolve() != Path(report["source_provenance"]["checkpoint_local_path"]).resolve():
                raise ValueError("Saved classifier identifies another source checkpoint")
        elif hp_source is not None:
            raise ValueError("Scratch checkpoint illegally identifies a source backbone")
        model = restore_classifier(saved, run, representation, report["class_weights"])
        names, count = validate_model(model, run, report["class_weights"])
        recorded = integrity["optimizer"]
        if sorted(names) != recorded["trainable_parameter_names"] or count != recorded["trainable_parameters"]:
            raise ValueError("Actual optimizer audit differs from trainable classifier parameters")
        if not set(recorded["state_parameter_names"]).issubset(names):
            raise ValueError("Actual optimizer state includes frozen parameters")
        _validate_checkpoint_optimizer(saved, model, integrity)
        fingerprint = backbone_fingerprint(checkpoint_backbone(saved))
        if (frozen and fingerprint != initial) or (name == "last" and fingerprint != final):
            raise ValueError("Saved best/last backbone fingerprint violates the control invariant")
        loaded[name] = saved
    scores = [state["best_model_score"] for state in loaded["best"].get("callbacks", {}).values()
              if isinstance(state, dict) and state.get("monitor") == "valid/auc" and state.get("best_model_score") is not None]
    if len(scores) != 1 or not torch.isfinite(torch.as_tensor(scores[0])) or not 0 <= float(scores[0]) <= 1:
        raise ValueError("Missing/non-finite validation-best AUC checkpoint")
    result = dict(passed=True, run_key=args.run_key, control_mode=run["control_mode"],
        training_seed=run["training_seed"], pretrain_seed=run.get("pretrain_seed"),
        epochs=args.epochs, expected_global_steps=expected_steps, initial_backbone_sha256=initial,
        final_backbone_sha256=final, best_checkpoint_sha256=sha256(args.run_dir / "checkpoints/best.ckpt"),
        last_checkpoint_sha256=sha256(args.run_dir / "checkpoints/last.ckpt"))
    write_json(args.run_dir / "control_fit_validation.json", result)
    print(f"PASS {args.run_key}: exact control protocol, step budget and actual backbone/optimizer integrity")
    return result


def export_test(args):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, roc_auc_score
    plan, run, representation, _, transfer = validate_transfer(args.plan, args.run_key, args.transfer_report)
    # Export never bypasses the full-run checkpoint and control-integrity gates.
    fit_args = argparse.Namespace(**vars(args))
    fit_args.epochs = 10
    fit = check_fit(fit_args)
    if args.checkpoint.resolve() != (args.run_dir / "checkpoints/best.ckpt").resolve():
        raise ValueError("Test export must use this run's validation-best checkpoint")
    manifest = json.loads((args.prepared_dir / "manifest.json").read_text())
    if (sha256(args.prepared_dir / "manifest.json") != transfer["manifest_sha256"]
            or manifest["split_counts"] != plan["split_counts"]
            or tuple(name for name, _ in classes(manifest)) != NAMES):
        raise ValueError("Held-out canonical manifest/classes changed")
    saved = _load_checkpoint_state(str(args.checkpoint))
    model = restore_classifier(saved, run, representation, transfer["class_weights"])
    model.eval().to(args.device)
    torch.set_float32_matmul_precision("medium")
    keys = [("tokens", np.int64), ("mask", bool), ("type_ids", np.int64)]
    if DATA_KINDS[run["model_family"]] == "continuous":
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
                    probabilities = model(inputs).softmax(-1).cpu().numpy()
                    if probabilities.shape != (len(labels), 5) or not np.isfinite(probabilities).all():
                        raise ValueError("Invalid held-out classifier probabilities")
                    values = dict(probabilities=probabilities, labels=labels,
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
    rows = compare([(args.run_key, args.output_dir)], args.output_dir / "plots", expected_events=len(labels))
    predictions = probabilities.argmax(1)
    macro = float(roc_auc_score(labels, probabilities, multi_class="ovr", average="macro"))
    summary = {key: transfer[key] for key in ("run_key", "model", "model_family", "control_mode", "training_seed",
        "pretrain_seed", "source_key", "source_provenance", "configuration_id", "source_manifest_sha256",
        "input_inventory_sha256", "class_weights", "protocol", "source_backbone_sha256", "backbone_frozen")}
    summary.update(checkpoint_selection="validation-best", checkpoint_sha256=sha256(args.checkpoint),
        finetune_config_sha256=sha256(args.run_dir / "full_config.yaml"),
        manifest_sha256=transfer["manifest_sha256"], class_names=NAMES, test_events=len(labels),
        test_auc=macro, macro_ovr_auc=macro,
        weighted_ovr_auc=float(roc_auc_score(labels, probabilities, multi_class="ovr", average="weighted")),
        accuracy=float(accuracy_score(labels, predictions)), balanced_accuracy=float(balanced_accuracy_score(labels, predictions)),
        per_class_auc={row["class_name"]: row["auc"] for row in rows},
        confusion_matrix=confusion_matrix(labels, predictions, labels=range(5)).tolist(),
        normalized_confusion_matrix=confusion_matrix(labels, predictions, labels=range(5), normalize="true").tolist(),
        control_fit_validation=fit)
    write_json(args.output_dir / "classification_report.json", classification_report(labels, predictions,
        labels=range(5), target_names=NAMES, output_dict=True, zero_division=0))
    for name in ("summary.json", "evaluation_summary.json"):
        write_json(args.output_dir / name, summary)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    for name in ("plan", "source-dir", "prepared-dir", "input-inventory", "report-dir"):
        pre.add_argument(f"--{name}", type=Path, required=True)
    for name in ("checkpoint", "pretrain-config", "pretrain-status", "pretrain-completion", "pretrain-vocabulary-shard"):
        pre.add_argument(f"--{name}", type=Path)
    pre.add_argument("--run-key", required=True)
    fit = sub.add_parser("check-fit")
    export = sub.add_parser("export-test")
    for command in (fit, export):
        for name in ("plan", "run-dir", "transfer-report"):
            command.add_argument(f"--{name}", type=Path, required=True)
        command.add_argument("--run-key", required=True)
        command.add_argument("--source-checkpoint", type=Path)
    fit.add_argument("--epochs", type=int, choices=(1, 10), required=True)
    for name in ("checkpoint", "prepared-dir", "output-dir"):
        export.add_argument(f"--{name}", type=Path, required=True)
    export.add_argument("--device", default="cuda")
    export.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    {"preflight": preflight, "check-fit": check_fit, "export-test": export_test}[args.command](args)


if __name__ == "__main__":
    main()
