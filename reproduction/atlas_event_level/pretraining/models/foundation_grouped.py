"""Masked pretraining with one sequence position per physics object."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule

from heptokens.data.sequence import MASK_KEY, TOKENS_KEY, TYPE_IDS_KEY
from heptokens.models.transformer import Transformer
from heptokens.models.utils import ScheduledOptimiserMixin


class GroupedSequenceBackbone(nn.Module):
    """Concatenate residual-quantizer embeddings before event attention."""

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_dim: int,
        max_quantizers: int,
        quantizer_embedding_dim: int | None,
        num_heads: int,
        num_layers: int,
        dropout: float,
        max_seq_length: int,
        num_type_ids: int,
        pad_token_id: int,
        use_type_embedding: bool,
        use_position_embedding: bool,
    ) -> None:
        super().__init__()
        self.max_quantizers = max_quantizers
        self.pad_token_id = pad_token_id
        embedding_dim = quantizer_embedding_dim or math.ceil(hidden_dim / max_quantizers)
        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_token_id,
        )
        self.object_projection = nn.Linear(max_quantizers * embedding_dim, hidden_dim)
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

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected grouped tokens [B, N, Q], got {tuple(tokens.shape)}")
        if tokens.shape[-1] != self.max_quantizers:
            raise ValueError(
                f"Expected Q={self.max_quantizers}, got grouped tokens with Q={tokens.shape[-1]}"
            )

        batch_size, seq_len, _ = tokens.shape
        quantizer_mask = tokens.ne(self.pad_token_id)
        embedded = self.token_embedding(tokens)
        embedded = embedded * quantizer_mask.unsqueeze(-1)
        x = self.object_projection(embedded.flatten(start_dim=2))

        if self.type_embedding is not None:
            if type_ids is None:
                type_ids = torch.zeros_like(mask, dtype=torch.long)
            x = x + self.type_embedding(type_ids)
        if self.position_embedding is not None:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
            x = x + self.position_embedding(positions)

        x = self.dropout(x)
        return self.layer_norm(self.transformer(x, mask=mask.bool()))


class GroupedMaskedSequenceModel(nn.Module):
    """Mask one object position and reconstruct all of its quantizer codes."""

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_dim: int,
        max_quantizers: int,
        quantizer_embedding_dim: int | None,
        num_heads: int,
        num_layers: int,
        dropout: float,
        max_seq_length: int,
        num_type_ids: int,
        pad_token_id: int,
        use_type_embedding: bool,
        use_position_embedding: bool,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_quantizers = max_quantizers
        self.pad_token_id = pad_token_id
        self.backbone = GroupedSequenceBackbone(
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
        self.quantizer_prediction_embedding = nn.Embedding(max_quantizers, hidden_dim)
        self.mlm_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, vocab_size),
        )

    def masked_logits(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        type_ids: torch.Tensor | None,
        masked_positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.backbone(tokens, mask, type_ids)
        selected = hidden[masked_positions]
        quantizer_ids = torch.arange(self.max_quantizers, device=tokens.device)
        states = selected.unsqueeze(1) + self.quantizer_prediction_embedding(quantizer_ids)
        return self.mlm_head(states)


class LitGroupedMaskedSequenceModel(ScheduledOptimiserMixin, LightningModule):
    """Lightning wrapper for grouped object-level masked pretraining."""

    def __init__(
        self,
        *,
        data_sample: dict | None = None,
        n_classes: int | None = None,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_seq_length: int = 256,
        vocab_size: int = 20000,
        max_quantizers: int = 8,
        quantizer_embedding_dim: int | None = None,
        num_type_ids: int = 12,
        mask_token_id: int = 3,
        pad_token_id: int = 0,
        mask_prob: float = 0.15,
        use_type_embedding: bool = True,
        use_position_embedding: bool = True,
        learning_rate: float = 1e-4,
        optimizer=None,
        scheduler=None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["data_sample"])
        self.learning_rate = learning_rate
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.model = GroupedMaskedSequenceModel(
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

    @property
    def backbone(self) -> GroupedSequenceBackbone:
        return self.model.backbone

    def _shared_step(self, batch: dict, prefix: str) -> torch.Tensor:
        original = batch[TOKENS_KEY]
        object_mask = batch[MASK_KEY].bool()
        type_ids = batch.get(TYPE_IDS_KEY)
        quantizer_mask = original.ne(self.pad_token_id)

        probabilities = torch.full(object_mask.shape, self.mask_prob, device=original.device)
        contains_special_token = ((original <= 3) & quantizer_mask).any(dim=-1)
        can_mask = object_mask & ~contains_special_token
        masked_positions = torch.bernoulli(probabilities).bool() & can_mask
        if not masked_positions.any():
            first_valid = can_mask.flatten().nonzero(as_tuple=False)
            if first_valid.numel() == 0:
                raise RuntimeError("Grouped token batch contains no maskable object positions")
            masked_positions.view(-1)[first_valid[0, 0]] = True

        tokens = original.clone()
        replace = masked_positions.unsqueeze(-1) & quantizer_mask
        tokens[replace] = self.mask_token_id
        labels = original[masked_positions]
        valid_labels = quantizer_mask[masked_positions]
        logits = self.model.masked_logits(tokens, object_mask, type_ids, masked_positions)
        loss = F.cross_entropy(logits[valid_labels], labels[valid_labels])

        accuracy = (logits.argmax(dim=-1)[valid_labels] == labels[valid_labels]).float().mean()
        sync_dist = prefix == "valid"
        self.log(
            f"{prefix}/mask_acc",
            accuracy,
            prog_bar=True,
            sync_dist=sync_dist,
        )
        self.log(
            f"{prefix}/total_loss",
            loss,
            prog_bar=True,
            sync_dist=sync_dist,
        )
        return loss

    def training_step(self, batch: dict) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict) -> torch.Tensor:
        return self._shared_step(batch, "valid")
