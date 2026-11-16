"""Opt-in flat/hierarchical adapters for the existing five-class protocol.

Reuse the grouped classifier's optimizer, weighted CE and metrics verbatim.
No existing pretraining, binary classifier or hierarchical adapter is changed.
"""
from __future__ import annotations

import torch
from lightning import LightningModule
from torchmetrics import AUROC, Accuracy

from heptokens.data.continuous_schema import regroup_continuous_schema
from heptokens.models.foundation_continuous import (
    ContinuousCLSClassifier, ContinuousSequenceBackbone, _load_checkpoint_state,
)
from heptokens.models.foundation_continuous_multiclass import input_semantics
from heptokens.models.foundation_grouped_cls_classifier import LitGroupedCLSClassifier


class LitMatchedContinuousMulticlass(LitGroupedCLSClassifier):
    def __init__(self, *, continuous_schema: dict, data_sample=None, n_classes=5,
                 class_weights=None, backbone_ckpt_path=None, freeze_backbone=False,
                 architecture="flat", feature_grouping="schema", hidden_dim=256,
                 num_heads=8, num_layers=4, inner_num_heads=4, inner_num_layers=1,
                 dropout=0.1, max_seq_length=256, num_type_ids=12,
                 classifier_hidden_dim=None, use_type_embedding=True,
                 use_position_embedding=True, learning_rate=1e-4,
                 optimizer=None, scheduler=None):
        LightningModule.__init__(self)
        if n_classes != 5 or freeze_backbone or classifier_hidden_dim is not None:
            raise ValueError("Require an unfrozen backbone and a linear five-class CLS head")
        if (architecture, feature_grouping) not in {
            ("flat", "schema"), ("hierarchical", "kinematics_auxiliary_v1")
        }:
            raise ValueError("Only the selected flat and latest hierarchical baselines are supported")
        continuous_schema = regroup_continuous_schema(continuous_schema, grouping=feature_grouping)
        self.save_hyperparameters({key: value for key, value in locals().copy().items()
                                   if key not in {"self", "data_sample", "__class__"}})
        self.learning_rate, self.n_classes = learning_rate, n_classes
        weights = torch.as_tensor(class_weights, dtype=torch.float32) if class_weights is not None else torch.empty(0)
        if weights.numel() not in (0, 5) or not torch.isfinite(weights).all() or torch.any(weights <= 0):
            raise ValueError("Expected five finite positive class weights")
        self.register_buffer("class_weights", weights, persistent=True)
        backbone = ContinuousSequenceBackbone(
            continuous_schema=continuous_schema, architecture=architecture, hidden_dim=hidden_dim,
            num_heads=num_heads, num_layers=num_layers, inner_num_heads=inner_num_heads,
            inner_num_layers=inner_num_layers, dropout=dropout, max_seq_length=max_seq_length,
            num_type_ids=num_type_ids, use_type_embedding=use_type_embedding,
            use_position_embedding=use_position_embedding)
        if backbone_ckpt_path:
            saved = _load_checkpoint_state(str(backbone_ckpt_path))
            hp = saved["hyper_parameters"]
            if hp.get("feature_grouping", "schema") != feature_grouping:
                raise ValueError("Checkpoint feature grouping differs")
            if input_semantics(hp["continuous_schema"]) != input_semantics(continuous_schema):
                raise ValueError("Pretraining input features, scales or groups differ")
            for key in ("architecture", "hidden_dim", "num_heads", "num_layers", "dropout",
                        "max_seq_length", "num_type_ids", "use_type_embedding", "use_position_embedding"):
                if hp.get(key) != getattr(self.hparams, key):
                    raise ValueError(f"Pretrained backbone setting differs: {key}")
            if architecture == "hierarchical":
                for key in ("inner_num_heads", "inner_num_layers"):
                    if hp.get(key) != getattr(self.hparams, key):
                        raise ValueError(f"Pretrained backbone setting differs: {key}")
            prefix = "model.backbone."
            state = {key[len(prefix):]: value for key, value in saved["state_dict"].items()
                     if key.startswith(prefix)}
            if not state or any(not torch.isfinite(value).all() for value in state.values() if value.is_floating_point()):
                raise ValueError("Missing/non-finite pretrained backbone")
            backbone.load_state_dict(state, strict=True)
        self.model = ContinuousCLSClassifier(backbone, hidden_dim=hidden_dim, n_classes=5,
            freeze_backbone=False, classifier_hidden_dim=None)
        for split in ("train", "valid", "test"):
            setattr(self, f"{split}_acc", Accuracy(task="multiclass", num_classes=5))
            setattr(self, f"{split}_auc", AUROC(task="multiclass", num_classes=5, average="macro"))

    def forward(self, batch):
        return self.model(batch)
