"""PyTorch Lightning wrapper for VQ-VAE with learned Product Quantization."""

import logging
from typing import Tuple

import torch
from torch import nn
from vector_quantize_pytorch import VectorQuantize

from heptokens.models.vq_vae import LitVqVae

log = logging.getLogger(__name__)


class ProductQuantizer(nn.Module):
    """Product quantizer: split the embedding into M sub-vectors and quantize each
    with its own independent codebook.

    Mirrors the ResidualVQ interface (``layers`` attribute and a
    ``(quantized, indices[M, ...], losses[M])`` forward output) so it can be used
    as a drop-in replacement inside LitVqVae.

    Args:
        dim: Total embedding dimension; must be divisible by num_quantizers
        num_quantizers: Number of independent sub-spaces M
        codebook_size: Number of codes in each sub-space codebook
        commitment: Weight for commitment loss (matches VectorQuantize and ResidualVQ)
    """

    def __init__(
        self,
        *,
        dim: int,
        num_quantizers: int,
        codebook_size: int,
        commitment: float = 1.0,
    ):
        super().__init__()
        if dim % num_quantizers != 0:
            raise ValueError(
                f"codebook_dim ({dim}) must be divisible by num_quantizers ({num_quantizers}). "
                f"Valid choices: {[m for m in range(1, dim + 1) if dim % m == 0]}"
            )
        self.dim = dim
        self.sub_dim = dim // num_quantizers
        self.layers = nn.ModuleList([
            VectorQuantize(
                dim=self.sub_dim,
                codebook_size=codebook_size,
                commitment=commitment,
            )
            for _ in range(num_quantizers)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chunks = x.split(self.sub_dim, dim=-1)
        quantized, indices, losses = zip(
            *(layer(chunk) for layer, chunk in zip(self.layers, chunks))
        )
        return torch.cat(quantized, dim=-1), torch.stack(indices), torch.stack(losses)


class LitPqVae(LitVqVae):
    """VQ-VAE with learned Product Quantization.

    Inherits all training, validation, logging, and loss logic from LitVqVae.
    Overrides only the quantization-related methods.

    num_quantizers here means number of independent sub-spaces M, not residual
    levels as in LitVqVae. Each object is still tokenized into M indices, so the
    output shape [batch, n_csts, M] matches LitVqVae.

    Requires: codebook_dim % num_quantizers == 0

    Args:
        encoder: Encoder class to instantiate
        decoder: Decoder class to instantiate
        codebook_size: Number of codes in each sub-space codebook
        codebook_dim: Total embedding dimension; must be divisible by num_quantizers
        num_quantizers: Number of sub-spaces M to split codebook_dim into
        commitment_weight: Weight for commitment loss
        learning_rate: Learning rate for optimizer
        reconstruction_weight: Weight for reconstruction loss
        feature_names: Optional list of feature names for logging
        feature_loss_weights: Optional per-feature loss weights
        data_codebook_init: Whether to initialise codebooks from data
        data_codebook_init_samples: Number of samples to collect for init
        data_codebook_init_quantizers: Which sub-spaces to initialise (default: all)
        dead_code_reset: Whether to periodically reset dead codes
        dead_code_reset_interval: Steps between dead code resets
        dead_code_reset_min_count: Codes used <= this are considered dead
        dead_code_reset_max_fraction: Max fraction of codebook to reset at once
        dead_code_reset_quantizers: Which sub-spaces to reset (default: all)
        dead_code_reset_sample_size: Number of encoder outputs to keep for reset
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Replace the parent's ResidualVQ with a product quantizer.
        self.vector_quantization = ProductQuantizer(
            dim=int(self.hparams.codebook_dim),
            num_quantizers=int(self.hparams.num_quantizers),
            codebook_size=int(self.hparams.codebook_size),
            commitment=float(self.hparams.commitment_weight),
        )

    def _all_quantizers(self) -> set[int]:
        return set(range(int(self.hparams.num_quantizers)))

    # Unlike residual levels, all sub-spaces are equivalent, so default to all.
    def _quantizers_to_reset(self) -> set[int]:
        configured = self.hparams.dead_code_reset_quantizers
        if configured is None:
            return self._all_quantizers()
        return {int(idx) for idx in configured}

    def _quantizers_to_initialize(self) -> set[int]:
        configured = self.hparams.data_codebook_init_quantizers
        if configured is None:
            return self._all_quantizers()
        return {int(idx) for idx in configured}

    def _reset_layer_codes(
        self,
        layer: torch.nn.Module,
        dead_indices: torch.Tensor,
        replacements: torch.Tensor,
    ) -> int:
        """ Like the parent but:
        1. slices full-dim replacements to this sub-space's chunk
        2. shape-checks against sub-dim rather than codebook_dim
        Calling super() would fail because it checks (codebook_size, codebook_dim)
        but each sub-codebook is (codebook_size, sub_dim)
        """
        sub_dime = self.vector_quantization.sub_dim

        if replacements.shape[-1] == int(self.hparams.codebook_dim):
            idx = next(
                i for i, l in enumerate(self.vector_quantization.layers) if l is layer
            )
            replacements = replacements[..., idx * sub_dim : (idx + 1) * sub_dim]

        # Shape to match against — sub_dim, not codebook_dim
        shape = (int(self.hparams.codebook_size), sub_dim)

        codebook = getattr(layer, "_codebook", layer)
        row_attrs = ("embed", "codebook", "weight", "embed_avg", "ema_embed", "ema_embedding")
        cluster_attrs = ("cluster_size", "ema_cluster_size")
        touched = 0

        for attr in row_attrs:
            tensor = getattr(codebook, attr, None)
            if not torch.is_tensor(tensor) or tensor.ndim not in (2, 3):
                continue
            if tensor.shape[-2:] == shape:
                self._assign_codebook_rows(tensor, dead_indices, replacements)
                touched += 1
            elif tensor.shape[-2:] == shape[::-1]:
                self._assign_codebook_rows(tensor.transpose(-1, -2), dead_indices, replacements)
                touched += 1

        embedding = getattr(codebook, "embedding", None)
        if embedding is not None and hasattr(embedding, "weight"):
            self._assign_codebook_rows(embedding.weight, dead_indices, replacements)
            touched += 1

        for attr in cluster_attrs:
            tensor = getattr(codebook, attr, None)
            if torch.is_tensor(tensor) and tensor.shape[-1] == int(self.hparams.codebook_size):
                self._assign_cluster_rows(tensor, dead_indices)

        return touched


