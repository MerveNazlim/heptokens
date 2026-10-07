"""Grouped foundation classifier with masked mean pooling only."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import AUROC, Accuracy

from heptokens.data.sequence import LABELS_KEY, MASK_KEY, TOKENS_KEY, TYPE_IDS_KEY
from heptokens.models.foundation_grouped import GroupedSequenceBackbone
from heptokens.models.utils import ScheduledOptimiserMixin

log = logging.getLogger(__name__)


class GroupedMaskedMeanClassifier(nn.Module):
    """Apply a linear classifier to the mean of valid backbone positions."""

    def __init__(
        self,
        backbone: GroupedSequenceBackbone,
        *,
        hidden_dim: int,
        n_classes: int,
        freeze_backbone: bool,
        classifier_hidden_dim: int | None = None,
        exclude_first_position: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.exclude_first_position = exclude_first_position
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

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.freeze_backbone:
            with torch.no_grad():
                hidden = self.backbone(tokens, mask, type_ids)
        else:
            hidden = self.backbone(tokens, mask, type_ids)
        valid = mask.bool().unsqueeze(-1)
        if self.exclude_first_position:
            valid = valid.clone()
            valid[:, 0] = False
        pooled = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        return self.classifier(pooled)


class LitGroupedMaskedMeanClassifier(ScheduledOptimiserMixin, LightningModule):
    """Binary grouped-token classifier using masked mean event pooling."""

    def __init__(
        self,
        *,
        data_sample: dict | None = None,
        n_classes: int = 2,
        backbone_ckpt_path: str | None = None,
        freeze_backbone: bool = True,
        classifier_hidden_dim: int | None = None,
        exclude_first_position: bool = False,
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
        use_type_embedding: bool = True,
        use_position_embedding: bool = True,
        learning_rate: float = 1e-4,
        optimizer=None,
        scheduler=None,
    ) -> None:
        super().__init__()
        if n_classes != 2:
            raise ValueError(f"This binary classifier requires n_classes=2, got {n_classes}")
        self.save_hyperparameters(ignore=["data_sample"])
        self.learning_rate = learning_rate
        self.n_classes = n_classes

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
        )
        if backbone_ckpt_path:
            self._load_pretrained_backbone(backbone, backbone_ckpt_path)
        else:
            log.warning("No backbone checkpoint supplied; using random initialization")

        self.model = GroupedMaskedMeanClassifier(
            backbone,
            hidden_dim=hidden_dim,
            n_classes=n_classes,
            freeze_backbone=freeze_backbone,
            classifier_hidden_dim=classifier_hidden_dim,
            exclude_first_position=exclude_first_position,
        )
        self.train_acc = Accuracy(task="binary")
        self.valid_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")
        self.train_auc = AUROC(task="binary")
        self.valid_auc = AUROC(task="binary")
        self.test_auc = AUROC(task="binary")

    @property
    def backbone(self) -> GroupedSequenceBackbone:
        return self.model.backbone

    @staticmethod
    def _load_pretrained_backbone(
        backbone: GroupedSequenceBackbone,
        checkpoint_path: str,
    ) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
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
        loss = F.cross_entropy(logits, labels)
        probabilities = torch.softmax(logits, dim=-1)[:, 1]
        predictions = logits.argmax(dim=-1)

        accuracy = getattr(self, f"{prefix}_acc")
        auc = getattr(self, f"{prefix}_auc")
        accuracy.update(predictions, labels)
        auc.update(probabilities, labels)
        self.log(
            f"{prefix}/total_loss",
            loss,
            prog_bar=True,
            on_step=prefix == "train",
            on_epoch=True,
            sync_dist=prefix != "train",
        )
        self.log(
            f"{prefix}/acc",
            accuracy,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"{prefix}/auc",
            auc,
            prog_bar=prefix != "train",
            on_step=False,
            on_epoch=True,
            sync_dist=True,
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
        return {
            "score": torch.softmax(logits, dim=-1)[:, 1],
            "label": batch[LABELS_KEY],
        }
