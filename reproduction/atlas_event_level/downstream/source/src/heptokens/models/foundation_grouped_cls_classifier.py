"""Grouped foundation classifier using the dedicated CLS representation."""

from __future__ import annotations

import functools
import logging
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import AUROC, Accuracy

from heptokens.data.sequence import LABELS_KEY, MASK_KEY, TOKENS_KEY, TYPE_IDS_KEY
from heptokens.models.foundation_grouped import GroupedSequenceBackbone
from heptokens.models.utils import ScheduledOptimiserMixin

log = logging.getLogger(__name__)


def _load_checkpoint_state(checkpoint_path: str) -> dict:
    """Load checkpoints written by current or newer Hydra installations."""
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as error:
        module_name = "hydra._internal.target_policy"
        if error.name != module_name:
            raise

        import hydra._internal

        compatibility_module = types.ModuleType(module_name)

        class _DeferredTarget(functools.partial):
            _hydra_call_context = None

        _DeferredTarget.__module__ = module_name
        _DeferredTarget.__qualname__ = "_DeferredTarget"
        compatibility_module._DeferredTarget = _DeferredTarget
        sys.modules[module_name] = compatibility_module
        hydra._internal.target_policy = compatibility_module
        log.info("Using Hydra target-policy compatibility while loading %s", checkpoint_path)
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


class GroupedCLSClassifier(nn.Module):
    """Apply a linear classifier to the contextualized position-zero CLS state."""

    def __init__(
        self,
        backbone: GroupedSequenceBackbone,
        *,
        hidden_dim: int,
        n_classes: int,
        freeze_backbone: bool,
        cls_token_id: int,
        pad_token_id: int,
        classifier_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.cls_token_id = cls_token_id
        self.pad_token_id = pad_token_id
        self.classifier = (
            nn.Sequential(
                nn.Linear(hidden_dim, classifier_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(classifier_hidden_dim, n_classes),
            )
            if classifier_hidden_dim is not None
            else nn.Linear(hidden_dim, n_classes)
        )
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _validate_cls(self, tokens: torch.Tensor, mask: torch.Tensor) -> None:
        if tokens.ndim != 3 or tokens.shape[1] < 1:
            raise ValueError(f"Expected grouped tokens [batch, sequence, Q], got {tokens.shape}")
        if not torch.all(mask[:, 0].bool()):
            raise ValueError("CLS position 0 is not active in every event")
        if not torch.all(tokens[:, 0, 0].eq(self.cls_token_id)):
            raise ValueError(
                f"Position 0 does not contain CLS token ID {self.cls_token_id}"
            )
        if tokens.shape[-1] > 1 and not torch.all(
            tokens[:, 0, 1:].eq(self.pad_token_id)
        ):
            raise ValueError("CLS position contains non-padding residual quantizer codes")

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_cls(tokens, mask)
        if self.freeze_backbone:
            with torch.no_grad():
                hidden = self.backbone(tokens, mask, type_ids)
        else:
            hidden = self.backbone(tokens, mask, type_ids)
        return self.classifier(hidden[:, 0])


class LitGroupedCLSClassifier(ScheduledOptimiserMixin, LightningModule):
    """Grouped-token classifier using the contextualized CLS state."""

    def __init__(
        self,
        *,
        data_sample: dict | None = None,
        n_classes: int = 2,
        class_weights: list[float] | None = None,
        token_vocabulary: dict | None = None,
        backbone_ckpt_path: str | None = None,
        freeze_backbone: bool = True,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_seq_length: int = 256,
        vocab_size: int = 131588,
        max_quantizers: int = 8,
        quantizer_embedding_dim: int | None = 32,
        num_type_ids: int = 12,
        pad_token_id: int = 0,
        cls_token_id: int = 1,
        classifier_hidden_dim: int | None = None,
        use_type_embedding: bool = True,
        use_position_embedding: bool = True,
        object_projection_mode: str = "linear",
        learning_rate: float = 1e-4,
        optimizer=None,
        scheduler=None,
    ) -> None:
        super().__init__()
        if n_classes < 2:
            raise ValueError(f"Classification requires at least two classes, got {n_classes}")
        if token_vocabulary is not None:
            vocabulary_quantizers = int(token_vocabulary["max_quantizers"])
            if vocabulary_quantizers != max_quantizers:
                raise ValueError(
                    f"Configured max_quantizers={max_quantizers}, but the token "
                    f"vocabulary requires {vocabulary_quantizers}"
                )
            vocab_size = int(token_vocabulary["vocab_size"])
        if class_weights is not None:
            weights = torch.as_tensor(class_weights, dtype=torch.float32)
            if weights.numel() != n_classes:
                raise ValueError(
                    f"Expected {n_classes} class weights, got {weights.numel()}"
                )
            if not torch.isfinite(weights).all() or torch.any(weights <= 0):
                raise ValueError("Class weights must be finite and positive")
        else:
            weights = torch.empty(0, dtype=torch.float32)
        self.save_hyperparameters(ignore=["data_sample", "token_vocabulary"])
        self.learning_rate = learning_rate
        self.n_classes = n_classes
        self.register_buffer("class_weights", weights, persistent=True)

        backbone = GroupedSequenceBackbone(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            max_quantizers=max_quantizers,
            quantizer_embedding_dim=quantizer_embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_seq_length=max_seq_length,
            num_type_ids=num_type_ids,
            pad_token_id=pad_token_id,
            use_type_embedding=use_type_embedding,
            use_position_embedding=use_position_embedding,
            object_projection_mode=object_projection_mode,
        )
        if backbone_ckpt_path:
            self._load_pretrained_backbone(backbone, backbone_ckpt_path)
        else:
            log.warning("No backbone checkpoint supplied; using random initialization")

        self.model = GroupedCLSClassifier(
            backbone,
            hidden_dim=hidden_dim,
            n_classes=n_classes,
            freeze_backbone=freeze_backbone,
            cls_token_id=cls_token_id,
            pad_token_id=pad_token_id,
            classifier_hidden_dim=classifier_hidden_dim,
        )
        metric_kwargs = (
            {"task": "binary"}
            if n_classes == 2
            else {"task": "multiclass", "num_classes": n_classes}
        )
        auc_kwargs = (
            metric_kwargs
            if n_classes == 2
            else {**metric_kwargs, "average": "macro"}
        )
        self.train_acc = Accuracy(**metric_kwargs)
        self.valid_acc = Accuracy(**metric_kwargs)
        self.test_acc = Accuracy(**metric_kwargs)
        self.train_auc = AUROC(**auc_kwargs)
        self.valid_auc = AUROC(**auc_kwargs)
        self.test_auc = AUROC(**auc_kwargs)

    @property
    def backbone(self) -> GroupedSequenceBackbone:
        return self.model.backbone

    @staticmethod
    def _load_pretrained_backbone(
        backbone: GroupedSequenceBackbone,
        checkpoint_path: str,
    ) -> None:
        checkpoint = _load_checkpoint_state(checkpoint_path)
        state_dict = checkpoint.get("state_dict", checkpoint)
        prefix = "model.backbone."
        backbone_state = {
            key.removeprefix(prefix): value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if not backbone_state:
            raise ValueError(
                f"No grouped backbone weights with prefix '{prefix}' found in "
                f"{checkpoint_path}"
            )
        backbone.load_state_dict(backbone_state, strict=True)
        log.info("Loaded grouped foundation backbone from %s", checkpoint_path)

    def forward(self, batch: dict) -> torch.Tensor:
        return self.model(
            batch[TOKENS_KEY],
            batch[MASK_KEY],
            batch.get(TYPE_IDS_KEY),
        )

    def _shared_step(self, batch: dict, prefix: str) -> torch.Tensor:
        labels = batch[LABELS_KEY].long()
        logits = self.forward(batch)
        loss = F.cross_entropy(
            logits,
            labels,
            weight=self.class_weights if self.class_weights.numel() else None,
        )
        probabilities = torch.softmax(logits, dim=-1)
        metric_probabilities = probabilities[:, 1] if self.n_classes == 2 else probabilities
        predictions = logits.argmax(dim=-1)
        accuracy = getattr(self, f"{prefix}_acc")
        auc = getattr(self, f"{prefix}_auc")
        accuracy.update(predictions, labels)
        auc.update(metric_probabilities, labels)
        self.log(
            f"{prefix}/total_loss", loss,
            prog_bar=True, on_step=prefix == "train", on_epoch=True,
            sync_dist=prefix != "train",
        )
        self.log(
            f"{prefix}/acc", accuracy,
            prog_bar=True, on_step=False, on_epoch=True, sync_dist=True,
        )
        self.log(
            f"{prefix}/auc", auc,
            prog_bar=prefix != "train", on_step=False, on_epoch=True, sync_dist=True,
        )
        return loss

    def training_step(self, batch: dict) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict) -> torch.Tensor:
        return self._shared_step(batch, "valid")

    def test_step(self, batch: dict) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def predict_step(self, batch: dict) -> dict:
        logits = self.forward(batch)
        probabilities = torch.softmax(logits, dim=-1)
        return {
            "score": probabilities[:, 1] if self.n_classes == 2 else probabilities,
            "probabilities": probabilities,
            "label": batch[LABELS_KEY],
        }
