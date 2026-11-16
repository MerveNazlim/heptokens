"""Transformer building block for sequence-aware encoding/decoding."""

from __future__ import annotations

import torch
import torch.nn as nn


class Transformer(nn.Module):
    """Transformer encoder stack with input/output projections.

    This module preserves the input sequence length and supports a padding mask.
    It is designed to be used inside Encoder/Decoder wrappers.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        d_model: int = 128,
        n_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass through the transformer.

        Args:
            x: Tensor of shape [batch, n_csts, input_dim].
            mask: Boolean tensor of shape [batch, n_csts] where True means valid.

        Returns:
            Tensor of shape [batch, n_csts, output_dim].
        """
        squeeze_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_batch = True

        key_padding_mask = None
        empty_sequences = None
        if mask is not None:
            mask = mask.bool()
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            empty_sequences = ~mask.any(dim=1)
            x = x.clone()
            # Zero out padded positions to avoid NaNs in attention/FFN paths.
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
            if empty_sequences.any():
                mask_for_encoder = mask.clone()
                # Ensure at least one valid token so attention doesn't see all padding.
                mask_for_encoder[empty_sequences, 0] = True
                key_padding_mask = ~mask_for_encoder
                x[empty_sequences] = 0.0
            else:
                key_padding_mask = ~mask
        x = self.input_proj(x)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.output_proj(x)

        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
            if empty_sequences is not None and empty_sequences.any():
                x[empty_sequences] = 0.0

        if squeeze_batch:
            x = x.squeeze(0)

        return x


class SetToVectorTransformer(nn.Module):
    """Maps a set (variable-length sequence) to a single vector."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        d_model: int = 128,
        n_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        activation: str = "gelu",
        pooling: str = "mean",  # "mean", "max", or "cls"
    ) -> None:
        super().__init__()
        self.pooling = pooling

        # Use your existing Transformer
        self.transformer = Transformer(
            input_dim=input_dim,
            output_dim=d_model,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.output_proj = nn.Linear(d_model, output_dim)
        if pooling not in ["mean", "max", "cls"]:
            raise ValueError(f"Unknown pooling method: {pooling}")
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, input_dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [batch, n_csts, input_dim]
            mask: [batch, n_csts] where True means valid

        Returns:
            [batch, output_dim]
        """
        # Prepend CLS token if using cls pooling
        if self.pooling == "cls":
            x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
            if mask is not None:
                cls_mask = torch.ones((mask.shape[0], 1), device=mask.device, dtype=mask.dtype)
                mask = torch.cat([cls_mask, mask], dim=1)
        # Get sequence representations
        seq_out = self.transformer(x, mask)  # [batch, n_csts, d_model]

        # Pool to single vector
        if self.pooling == "mean":
            if mask is not None:
                # Mean over valid positions only
                valid_sum = (seq_out * mask.unsqueeze(-1)).sum(dim=1)
                valid_count = mask.sum(dim=1, keepdim=True)
                pooled = valid_sum / valid_count.clamp(min=1)
            else:
                pooled = seq_out.mean(dim=1)
        elif self.pooling == "max":
            if mask is not None:
                seq_out_masked = seq_out.masked_fill(~mask.unsqueeze(-1), float("-inf"))
                pooled = seq_out_masked.max(dim=1)[0]
            else:
                pooled = seq_out.max(dim=1)[0]
        elif self.pooling == "cls":
            pooled = seq_out[:, 0, :]  # First token is [CLS]

        # Project to output and give singleton sequence dimension
        return self.output_proj(pooled).unsqueeze(1)


class VectorToSetTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        max_set_size: int,
        *,
        d_model: int = 128,
        n_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.max_set_size = max_set_size

        # Learnable output queries (one per set element)
        self.output_queries = nn.Parameter(torch.randn(1, max_set_size, d_model))

        # Project input vector
        self.input_proj = nn.Linear(input_dim, d_model)

        # Decoder with cross-attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]

        # Project and expand input to serve as memory
        memory = self.input_proj(x)  # [batch, 1, d_model]

        # Expand queries
        queries = self.output_queries.expand(batch_size, -1, -1)  # [batch, max_set_size, d_model]

        # Decoder cross-attends to input vector
        tgt_key_padding_mask = ~mask if mask is not None else None
        out = self.decoder(queries, memory, tgt_key_padding_mask=tgt_key_padding_mask)

        return self.output_proj(out)
