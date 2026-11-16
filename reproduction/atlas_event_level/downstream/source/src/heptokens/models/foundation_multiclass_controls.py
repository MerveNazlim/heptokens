"""Opt-in scratch and frozen-backbone controls for the matched five-class study.

Existing classifier implementations, state keys and initialization order are
unchanged. Frozen models may omit their source only for checkpoint restoration;
the production preflight separately requires a checksum-pinned source.
"""
from __future__ import annotations

import torch

from heptokens.models.foundation_continuous_matched_multiclass import LitMatchedContinuousMulticlass
from heptokens.models.foundation_grouped_cls_classifier import LitGroupedCLSClassifier


def _validate_control(control_mode, n_classes, classifier_hidden_dim,
                      backbone_ckpt_path, freeze_backbone):
    if control_mode not in {"scratch", "frozen"}:
        raise ValueError("control_mode must be scratch or frozen")
    if n_classes != 5 or classifier_hidden_dim is not None:
        raise ValueError("Controls require a linear five-class CLS head")
    frozen = control_mode == "frozen"
    if freeze_backbone is not None and freeze_backbone is not frozen:
        raise ValueError("freeze_backbone disagrees with control_mode")
    if control_mode == "scratch" and backbone_ckpt_path is not None:
        raise ValueError("Scratch controls must never load a pretrained backbone")
    return frozen


class _ControlOptimiserMixin:
    def configure_optimizers(self):
        # Unlike the legacy mixin, exclude frozen tensors from param groups as
        # well as gradients. This guarantees head-only optimizer state.
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        hp = self.hparams
        opt_partial = hp.get("optimizer")
        if callable(opt_partial):
            optimizer = opt_partial(parameters)
        else:
            learning_rate = self.__dict__.get("learning_rate") or hp.get("learning_rate", 1e-3)
            optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        sched_partial = hp.get("scheduler")
        if callable(sched_partial):
            scheduler = sched_partial(optimizer=optimizer, model=self)
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        return optimizer


class LitTokenMulticlassControl(_ControlOptimiserMixin, LitGroupedCLSClassifier):
    def __init__(self, *, control_mode, data_sample=None, n_classes=5,
                 class_weights=None, token_vocabulary=None, vocab_size=131588,
                 backbone_ckpt_path=None, freeze_backbone=None,
                 classifier_hidden_dim=None, **kwargs):
        frozen = _validate_control(control_mode, n_classes, classifier_hidden_dim,
                                   backbone_ckpt_path, freeze_backbone)
        super().__init__(data_sample=data_sample, n_classes=n_classes,
                         class_weights=class_weights, token_vocabulary=token_vocabulary,
                         vocab_size=vocab_size, backbone_ckpt_path=backbone_ckpt_path,
                         freeze_backbone=frozen, classifier_hidden_dim=None, **kwargs)
        self.save_hyperparameters({"control_mode": control_mode, "freeze_backbone": frozen})


class LitContinuousMulticlassControl(_ControlOptimiserMixin, LitMatchedContinuousMulticlass):
    def __init__(self, *, continuous_schema, control_mode, data_sample=None,
                 n_classes=5, class_weights=None, backbone_ckpt_path=None,
                 freeze_backbone=None, classifier_hidden_dim=None,
                 use_role_embedding=True, **kwargs):
        frozen = _validate_control(control_mode, n_classes, classifier_hidden_dim,
                                   backbone_ckpt_path, freeze_backbone)
        if use_role_embedding is not True:
            raise ValueError("Matched continuous controls require role embeddings ON")
        # The historical matched adapter only permits an unfrozen constructor.
        # Its inner classifier already implements correct eval/no_grad freezing.
        super().__init__(continuous_schema=continuous_schema, data_sample=data_sample,
                         n_classes=n_classes, class_weights=class_weights,
                         backbone_ckpt_path=backbone_ckpt_path, freeze_backbone=False,
                         classifier_hidden_dim=None, **kwargs)
        if self.backbone.role_embedding is None:
            raise ValueError("Matched continuous controls require the role embedding table")
        self.model.freeze_backbone = frozen
        self.backbone.requires_grad_(not frozen)
        if frozen:
            self.backbone.eval()
        self.save_hyperparameters({"control_mode": control_mode, "freeze_backbone": frozen,
                                   "use_role_embedding": True})
