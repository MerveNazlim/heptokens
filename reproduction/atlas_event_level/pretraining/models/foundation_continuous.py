"""Flat and hierarchical continuous baselines for grouped event sequences."""

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

from heptokens.data.continuous_schema import validate_continuous_schema
from heptokens.data.benchmark_sequence import (
    CONTINUOUS_FEATURE_MASK_KEY,
    CONTINUOUS_FEATURES_KEY,
    LABELS_KEY,
    MASK_KEY,
    POSITION_ROLE_IDS_KEY,
    TYPE_IDS_KEY,
)
from heptokens.models.transformer import Transformer
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

        compat_module = types.ModuleType(module_name)

        class _DeferredTarget(functools.partial):
            _hydra_call_context = None

        _DeferredTarget.__module__ = module_name
        _DeferredTarget.__qualname__ = "_DeferredTarget"
        compat_module._DeferredTarget = _DeferredTarget
        sys.modules[module_name] = compat_module
        hydra._internal.target_policy = compat_module
        log.info("Using Hydra target-policy compatibility while loading %s", checkpoint_path)
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


class ContinuousSequenceBackbone(nn.Module):
    """Encode aligned continuous positions before event-level self-attention."""

    def __init__(
        self,
        *,
        continuous_schema: dict,
        architecture: str,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        inner_num_heads: int,
        inner_num_layers: int,
        dropout: float,
        max_seq_length: int,
        num_type_ids: int,
        use_type_embedding: bool,
        use_position_embedding: bool,
    ) -> None:
        super().__init__()
        validate_continuous_schema(continuous_schema)
        if architecture not in {"flat", "hierarchical"}:
            raise ValueError(
                f"architecture must be 'flat' or 'hierarchical', got {architecture!r}"
            )
        self.architecture = architecture
        self.hidden_dim = hidden_dim
        self.schema = continuous_schema
        self.role_ids = {
            name: int(value)
            for name, value in continuous_schema["role_ids"].items()
        }
        self.object_specs = continuous_schema["objects"]
        self.type_to_object = {
            int(spec["type_id"]): object_name
            for object_name, spec in self.object_specs.items()
        }

        self.role_embedding = nn.Embedding(
            max(self.role_ids.values()) + 1, hidden_dim, padding_idx=self.role_ids["padding"]
        )
        self.event_projection = nn.Linear(1, hidden_dim)
        self.object_projections = nn.ModuleDict()
        self.group_projections = nn.ModuleDict()
        for object_name, spec in self.object_specs.items():
            if architecture == "flat":
                self.object_projections[object_name] = nn.Linear(
                    int(spec["feature_count"]), hidden_dim
                )
            else:
                for group_name, indices in spec["groups"].items():
                    self.group_projections[
                        self.group_key(object_name, group_name)
                    ] = nn.Linear(len(indices), hidden_dim)

        self.inner_transformer = None
        if architecture == "hierarchical":
            self.inner_transformer = Transformer(
                input_dim=hidden_dim,
                output_dim=hidden_dim,
                d_model=hidden_dim,
                n_heads=inner_num_heads,
                num_layers=inner_num_layers,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
            )

        self.mask_embedding = nn.Parameter(torch.zeros(hidden_dim))
        self.type_embedding = (
            nn.Embedding(num_type_ids, hidden_dim) if use_type_embedding else None
        )
        self.position_embedding = (
            nn.Embedding(max_seq_length, hidden_dim) if use_position_embedding else None
        )
        self.transformer = Transformer(
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            d_model=hidden_dim,
            n_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def group_key(object_name: str, group_name: str) -> str:
        return f"{object_name}__{group_name}"

    def _encode_flat_objects(
        self,
        features: torch.Tensor,
        type_ids: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        for type_id, object_name in self.type_to_object.items():
            selected = type_ids.eq(type_id)
            count = int(self.object_specs[object_name]["feature_count"])
            output[selected] = self.object_projections[object_name](
                features[selected, :count]
            )

    def _encode_hierarchical_objects(
        self,
        features: torch.Tensor,
        type_ids: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert self.inner_transformer is not None
        for type_id, object_name in self.type_to_object.items():
            selected = type_ids.eq(type_id)
            if not selected.any():
                continue
            spec = self.object_specs[object_name]
            group_vectors = []
            for group_name, indices in spec["groups"].items():
                index = torch.as_tensor(indices, dtype=torch.long, device=features.device)
                group_values = features[selected].index_select(1, index)
                group_vectors.append(
                    self.group_projections[self.group_key(object_name, group_name)](
                        group_values
                    )
                )
            groups = torch.stack(group_vectors, dim=1)
            group_mask = torch.ones(
                groups.shape[:2], dtype=torch.bool, device=groups.device
            )
            encoded = self.inner_transformer(groups, mask=group_mask)
            output[selected] = encoded.mean(dim=1)

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        type_ids: torch.Tensor,
        role_ids: torch.Tensor,
        masked_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(
                f"Expected continuous features [B, N, F], got {tuple(features.shape)}"
            )
        batch_size, seq_len, _ = features.shape
        flat_features = features.reshape(-1, features.shape[-1])
        flat_types = type_ids.reshape(-1)
        flat_roles = role_ids.reshape(-1)
        content = features.new_zeros((flat_features.shape[0], self.hidden_dim))

        event_positions = flat_roles.eq(self.role_ids["event"])
        content[event_positions] = self.event_projection(
            flat_features[event_positions, :1]
        )
        object_positions = flat_roles.eq(self.role_ids["object"])
        object_features = flat_features[object_positions]
        object_types = flat_types[object_positions]
        object_output = content.new_zeros((object_features.shape[0], self.hidden_dim))
        if self.architecture == "flat":
            # Invoke every projection, including on empty selections. This keeps
            # every parameter in the DDP graph and avoids GPU-to-host .any()
            # synchronizations without changing any non-empty result.
            self._encode_flat_objects(object_features, object_types, object_output)
        elif object_features.shape[0] > 0:
            self._encode_hierarchical_objects(
                object_features, object_types, object_output
            )
        content[object_positions] = object_output

        x = content.reshape(batch_size, seq_len, self.hidden_dim)
        if masked_positions is not None:
            x = x.clone()
            x[masked_positions] = self.mask_embedding
        x = x + self.role_embedding(role_ids)
        if self.type_embedding is not None:
            x = x + self.type_embedding(type_ids)
        if self.position_embedding is not None:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            x = x + self.position_embedding(positions)
        x = self.dropout(x)
        return self.layer_norm(self.transformer(x, mask=mask.bool()))


class ContinuousMaskedSequenceModel(nn.Module):
    """Mask aligned positions and reconstruct their continuous feature targets."""

    def __init__(self, *, continuous_schema: dict, architecture: str, **backbone_kwargs) -> None:
        super().__init__()
        self.schema = continuous_schema
        self.architecture = architecture
        self.backbone = ContinuousSequenceBackbone(
            continuous_schema=continuous_schema,
            architecture=architecture,
            **backbone_kwargs,
        )
        hidden_dim = int(backbone_kwargs["hidden_dim"])
        self.event_head = nn.Linear(hidden_dim, 1)
        self.object_heads = nn.ModuleDict()
        self.group_heads = nn.ModuleDict()
        for object_name, spec in continuous_schema["objects"].items():
            if architecture == "flat":
                self.object_heads[object_name] = nn.Linear(
                    hidden_dim, int(spec["feature_count"])
                )
            else:
                for group_name, indices in spec["groups"].items():
                    self.group_heads[
                        ContinuousSequenceBackbone.group_key(object_name, group_name)
                    ] = nn.Linear(hidden_dim, len(indices))

    def reconstruct(
        self,
        *,
        features: torch.Tensor,
        mask: torch.Tensor,
        type_ids: torch.Tensor,
        role_ids: torch.Tensor,
        masked_positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.backbone(
            features, mask, type_ids, role_ids, masked_positions=masked_positions
        )[masked_positions]
        selected_types = type_ids[masked_positions]
        selected_roles = role_ids[masked_positions]
        prediction = features.new_zeros((hidden.shape[0], features.shape[-1]))

        event = selected_roles.eq(self.backbone.role_ids["event"])
        prediction[event, :1] = self.event_head(hidden[event])

        for type_id, object_name in self.backbone.type_to_object.items():
            selected = selected_types.eq(type_id)
            spec = self.schema["objects"][object_name]
            if self.architecture == "flat":
                count = int(spec["feature_count"])
                prediction[selected, :count] = self.object_heads[object_name](
                    hidden[selected]
                )
            else:
                for group_name, indices in spec["groups"].items():
                    values = self.group_heads[
                        ContinuousSequenceBackbone.group_key(object_name, group_name)
                    ](hidden[selected])
                    index = torch.as_tensor(
                        indices, dtype=torch.long, device=prediction.device
                    )
                    selected_rows = selected.nonzero(as_tuple=False).flatten()
                    prediction[selected_rows.unsqueeze(1), index.unsqueeze(0)] = values
        return prediction


class LitContinuousMaskedSequenceModel(ScheduledOptimiserMixin, LightningModule):
    """Matched masked-event pretraining for flat or hierarchical features."""

    def __init__(
        self,
        *,
        continuous_schema: dict,
        architecture: str,
        data_sample: dict | None = None,
        n_classes: int | None = None,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        inner_num_heads: int = 4,
        inner_num_layers: int = 1,
        dropout: float = 0.1,
        max_seq_length: int = 256,
        num_type_ids: int = 12,
        mask_prob: float = 0.15,
        use_type_embedding: bool = True,
        use_position_embedding: bool = True,
        learning_rate: float = 1e-4,
        optimizer=None,
        scheduler=None,
    ) -> None:
        super().__init__()
        if continuous_schema is None:
            raise ValueError("continuous_schema is required for continuous pretraining")
        self.save_hyperparameters(ignore=["data_sample"])
        self.learning_rate = learning_rate
        self.mask_prob = mask_prob
        self.model = ContinuousMaskedSequenceModel(
            continuous_schema=continuous_schema,
            architecture=architecture,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            inner_num_heads=inner_num_heads,
            inner_num_layers=inner_num_layers,
            dropout=dropout,
            max_seq_length=max_seq_length,
            num_type_ids=num_type_ids,
            use_type_embedding=use_type_embedding,
            use_position_embedding=use_position_embedding,
        )

    @property
    def backbone(self) -> ContinuousSequenceBackbone:
        return self.model.backbone

    def _shared_step(self, batch: dict, prefix: str) -> torch.Tensor:
        features = batch[CONTINUOUS_FEATURES_KEY].float()
        feature_mask = batch[CONTINUOUS_FEATURE_MASK_KEY].bool()
        sequence_mask = batch[MASK_KEY].bool()
        type_ids = batch[TYPE_IDS_KEY].long()
        role_ids = batch[POSITION_ROLE_IDS_KEY].long()
        can_mask = sequence_mask & (
            role_ids.eq(self.backbone.role_ids["event"])
            | role_ids.eq(self.backbone.role_ids["object"])
        )
        probabilities = torch.full(
            can_mask.shape, self.mask_prob, dtype=torch.float32, device=features.device
        )
        masked_positions = torch.bernoulli(probabilities).bool() & can_mask
        if not masked_positions.any():
            first_valid = can_mask.flatten().nonzero(as_tuple=False)
            if first_valid.numel() == 0:
                raise RuntimeError("Continuous batch contains no maskable positions")
            masked_positions.view(-1)[first_valid[0, 0]] = True

        prediction = self.model.reconstruct(
            features=features,
            mask=sequence_mask,
            type_ids=type_ids,
            role_ids=role_ids,
            masked_positions=masked_positions,
        )
        target = features[masked_positions]
        valid = feature_mask[masked_positions]
        squared_error = (prediction - target).square() * valid
        position_loss = squared_error.sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)
        loss = position_loss.mean()
        sync_dist = prefix == "valid"
        self.log(
            f"{prefix}/total_loss", loss, prog_bar=True, sync_dist=sync_dist
        )
        self.log(
            f"{prefix}/reconstruction_rmse",
            loss.sqrt(),
            prog_bar=True,
            sync_dist=sync_dist,
        )
        return loss

    def training_step(self, batch: dict) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict) -> torch.Tensor:
        return self._shared_step(batch, "valid")


class ContinuousCLSClassifier(nn.Module):
    """Classify an event from the continuous backbone's CLS state."""

    def __init__(
        self,
        backbone: ContinuousSequenceBackbone,
        *,
        hidden_dim: int,
        n_classes: int,
        freeze_backbone: bool,
        classifier_hidden_dim: int | None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
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

    def forward(self, batch: dict) -> torch.Tensor:
        role_ids = batch[POSITION_ROLE_IDS_KEY].long()
        if not torch.all(role_ids[:, 0].eq(self.backbone.role_ids["cls"])):
            raise ValueError("Position zero is not CLS in every continuous event")
        arguments = (
            batch[CONTINUOUS_FEATURES_KEY].float(),
            batch[MASK_KEY].bool(),
            batch[TYPE_IDS_KEY].long(),
            role_ids,
        )
        if self.freeze_backbone:
            with torch.no_grad():
                hidden = self.backbone(*arguments)
        else:
            hidden = self.backbone(*arguments)
        return self.classifier(hidden[:, 0])


class LitContinuousCLSClassifier(ScheduledOptimiserMixin, LightningModule):
    """Matched binary classifier for flat or hierarchical continuous inputs."""

    def __init__(
        self,
        *,
        continuous_schema: dict,
        architecture: str,
        data_sample: dict | None = None,
        n_classes: int = 2,
        backbone_ckpt_path: str | None = None,
        freeze_backbone: bool = False,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        inner_num_heads: int = 4,
        inner_num_layers: int = 1,
        dropout: float = 0.1,
        max_seq_length: int = 256,
        num_type_ids: int = 12,
        classifier_hidden_dim: int | None = 256,
        use_type_embedding: bool = True,
        use_position_embedding: bool = True,
        learning_rate: float = 1e-4,
        optimizer=None,
        scheduler=None,
    ) -> None:
        super().__init__()
        if n_classes != 2:
            raise ValueError(f"This binary classifier requires n_classes=2, got {n_classes}")
        if continuous_schema is None:
            raise ValueError("continuous_schema is required for continuous classification")
        self.save_hyperparameters(ignore=["data_sample"])
        self.learning_rate = learning_rate
        self.n_classes = n_classes
        backbone = ContinuousSequenceBackbone(
            continuous_schema=continuous_schema,
            architecture=architecture,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            inner_num_heads=inner_num_heads,
            inner_num_layers=inner_num_layers,
            dropout=dropout,
            max_seq_length=max_seq_length,
            num_type_ids=num_type_ids,
            use_type_embedding=use_type_embedding,
            use_position_embedding=use_position_embedding,
        )
        if backbone_ckpt_path:
            self._load_pretrained_backbone(backbone, backbone_ckpt_path)
        else:
            log.warning("No continuous backbone checkpoint supplied; using random initialization")
        self.model = ContinuousCLSClassifier(
            backbone,
            hidden_dim=hidden_dim,
            n_classes=n_classes,
            freeze_backbone=freeze_backbone,
            classifier_hidden_dim=classifier_hidden_dim,
        )
        self.train_acc = Accuracy(task="binary")
        self.valid_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")
        self.train_auc = AUROC(task="binary")
        self.valid_auc = AUROC(task="binary")
        self.test_auc = AUROC(task="binary")

    @property
    def backbone(self) -> ContinuousSequenceBackbone:
        return self.model.backbone

    @staticmethod
    def _load_pretrained_backbone(
        backbone: ContinuousSequenceBackbone, checkpoint_path: str
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
                f"No continuous backbone weights with prefix {prefix!r} found in "
                f"{checkpoint_path}"
            )
        backbone.load_state_dict(backbone_state, strict=True)

    def forward(self, batch: dict) -> torch.Tensor:
        return self.model(batch)

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
            f"{prefix}/acc", accuracy, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
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
