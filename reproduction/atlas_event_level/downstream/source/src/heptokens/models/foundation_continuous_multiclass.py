"""Multiclass adapter with the exact grouped-token downstream loss/metrics.

The original binary classifier and all pretraining classes are unchanged.
Only the input backbone differs from the hybrid/Q4 multiclass classifier.
"""
from __future__ import annotations

import torch
from lightning import LightningModule
from torchmetrics import AUROC, Accuracy

from heptokens.data.continuous_schema import regroup_continuous_schema
from heptokens.models.foundation_continuous import (
    ContinuousCLSClassifier, ContinuousSequenceBackbone, _load_checkpoint_state,
)
from heptokens.models.foundation_grouped_cls_classifier import LitGroupedCLSClassifier

PRESET = "kinematics_auxiliary_v1"


def input_semantics(schema):
    """Ignore optional decoded-target columns, never input features/scales/groups."""
    return {key: schema[key] for key in (
        "layout", "preprocessing", "max_feature_dim", "role_ids", "event_inputs", "objects"
    )}


class LitContinuousMulticlassClassifier(LitGroupedCLSClassifier):
    """Reuse the matched weighted CE and macro-OvR metrics, not a token backbone."""

    def __init__(self, *, continuous_schema: dict, data_sample=None, n_classes=5,
                 class_weights=None, backbone_ckpt_path=None, freeze_backbone=False,
                 architecture="hierarchical", feature_grouping=PRESET,
                 hidden_dim=256, num_heads=8, num_layers=4, inner_num_heads=4,
                 inner_num_layers=1, dropout=0.1, max_seq_length=256, num_type_ids=12,
                 classifier_hidden_dim=None, use_type_embedding=True,
                 use_position_embedding=True, learning_rate=1e-4,
                 optimizer=None, scheduler=None):
        # Deliberately do not construct LitGroupedCLSClassifier's token backbone.
        LightningModule.__init__(self)
        if n_classes != 5 or architecture != "hierarchical" or feature_grouping != PRESET:
            raise ValueError("This control requires five classes and the new hierarchical grouping")
        if classifier_hidden_dim is not None:
            raise ValueError("Matched hybrid/Q4 multiclass fine-tuning uses a linear CLS head")
        continuous_schema = regroup_continuous_schema(continuous_schema, grouping=feature_grouping)
        # Explicit capture also works when a subclass bypasses its token-only
        # parent's __init__; Lightning's frame inference would miss new fields.
        self.save_hyperparameters({key: value for key, value in locals().copy().items()
                                   if key not in {"self", "data_sample", "__class__"}})
        self.learning_rate, self.n_classes = learning_rate, n_classes
        weights = torch.as_tensor(class_weights, dtype=torch.float32) if class_weights is not None else torch.empty(0)
        if weights.numel() not in (0, n_classes) or not torch.isfinite(weights).all() or torch.any(weights <= 0):
            raise ValueError("Expected five finite positive class weights")
        self.register_buffer("class_weights", weights, persistent=True)
        backbone = ContinuousSequenceBackbone(
            continuous_schema=continuous_schema, architecture=architecture,
            hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers,
            inner_num_heads=inner_num_heads, inner_num_layers=inner_num_layers,
            dropout=dropout, max_seq_length=max_seq_length, num_type_ids=num_type_ids,
            use_type_embedding=use_type_embedding, use_position_embedding=use_position_embedding)
        if backbone_ckpt_path:
            saved = _load_checkpoint_state(str(backbone_ckpt_path))
            hp = saved.get("hyper_parameters", {})
            if hp.get("feature_grouping") != PRESET or input_semantics(hp["continuous_schema"]) != input_semantics(continuous_schema):
                raise ValueError("Pretraining and classification input features/scales/groups differ")
            for key in ("architecture", "hidden_dim", "num_heads", "num_layers", "inner_num_heads",
                        "inner_num_layers", "dropout", "max_seq_length", "num_type_ids",
                        "use_type_embedding", "use_position_embedding"):
                if hp.get(key) != getattr(self.hparams, key):
                    raise ValueError(f"Pretrained backbone setting differs: {key}")
            prefix = "model.backbone."
            state = {key.removeprefix(prefix): value for key, value in saved["state_dict"].items()
                     if key.startswith(prefix)}
            if not state or any(not torch.isfinite(value).all() for value in state.values() if value.is_floating_point()):
                raise ValueError("Missing/non-finite pretrained backbone")
            backbone.load_state_dict(state, strict=True)
        self.model = ContinuousCLSClassifier(backbone, hidden_dim=hidden_dim, n_classes=n_classes,
            freeze_backbone=freeze_backbone, classifier_hidden_dim=classifier_hidden_dim)
        for split in ("train", "valid", "test"):
            setattr(self, f"{split}_acc", Accuracy(task="multiclass", num_classes=n_classes))
            setattr(self, f"{split}_auc", AUROC(task="multiclass", num_classes=n_classes, average="macro"))

    def forward(self, batch):
        return self.model(batch)

