"""PyTorch Lightning wrapper for an FSQ autoencoder."""

import logging
from typing import Dict, Tuple
import re

import torch
from lightning import LightningModule

from heptokens.models.coders import Decoder, Encoder
from heptokens.models.utils import ScheduledOptimiserMixin

log = logging.getLogger(__name__)


class LitFSQ(ScheduledOptimiserMixin, LightningModule):
    """Lightning Module wrapper for an FSQ autoencoder.

    Args:
        encoder: Encoder, built with output_dim = len(levels)
        decoder: Decoder, built with input_dim = len(levels)
        levels: Number of quantization levels per latent dimension, e.g. [8, 5, 5, 5]
            (codebook size = prod(levels))
        learning_rate: Learning rate for optimizer
        feature_names: Feature names used for per-feature metric logging
        data_sample: Sample data used to infer the input dimension
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        levels: list[int],
        learning_rate: float = 1e-3,
        optimizer=None,
        scheduler=None,
        feature_names: list[str] | None = None,
        data_sample: torch.Tensor = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.feature_names = feature_names or []

        # Declare FSQ
        self.fsq = FSQ(levels)
        self.levels_dim = len(levels)


        n = self.fsq.codebook_size
        self.register_buffer("_train_code_counts", torch.zeros(n, dtype=torch.long), persistent=False)
        self.register_buffer("_val_code_counts",   torch.zeros(n, dtype=torch.long), persistent=False)



        # Infer input dimension from data_sample if provided
        if data_sample is not None:
            input_dim = data_sample["csts"].shape[-1]
        else:
            input_dim = 3  # Default for now

        # Declare encoder
        self.encoder = encoder(input_dim=input_dim, output_dim=self.levels_dim)
        # Declare decoder
        self.decoder = decoder(input_dim=self.levels_dim, output_dim=input_dim)



    @staticmethod
    def _metric_name(text: str) -> str:
        text = text.split("/")[-1]
        text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
        return text.strip("_") or "feature"

    def _feature_name(self, idx: int) -> str:
        if idx < len(self.feature_names):
            return self._metric_name(str(self.feature_names[idx]))
        return f"feature_{idx}"

    def _reconstruction_loss_and_prediction(
        self,
        z_hat: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        reconstruction = self.decode(z_hat, batch)
        mask = batch["mask"].bool()
        reconstructed = reconstruction[mask]
        original = batch["csts"][mask]

        reconstruction_loss = ((reconstructed - original) ** 2).mean(dim=-1).mean()
        return reconstruction_loss, reconstruction

    def _log_feature_reconstruction(
        self,
        prefix: str,
        reconstruction: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> None:
        mask = batch["mask"].bool()
        original = batch["csts"][mask]
        reconstructed = reconstruction[mask]
        if original.numel() == 0:
            return

        residual = reconstructed - original
        mae = residual.abs().mean(dim=0)
        rmse = torch.sqrt((residual**2).mean(dim=0))
        bias = residual.mean(dim=0)
        for idx in range(original.shape[-1]):
            name = self._feature_name(idx)
            self.log(f"{prefix}/feature_mae/{name}", mae[idx], on_step=False, on_epoch=True)
            self.log(f"{prefix}/feature_rmse/{name}", rmse[idx], on_step=False, on_epoch=True)
            self.log(f"{prefix}/feature_bias/{name}", bias[idx], on_step=False, on_epoch=True)

    @staticmethod
    def _accumulate_code_counts(counts, indices):
        valid = indices[indices >= 0]                          # drop padding (-1)
        counts += torch.bincount(valid, minlength=counts.numel())

    def _log_codebook_usage(self, prefix, counts):
        total = counts.sum()
        if total == 0:
            return
        used = (counts > 0).sum().float()
        probs = counts[counts > 0].float() / total
        perplexity = torch.exp(-(probs * probs.log()).sum())
        self.log(f"{prefix}/codebook/used_codes", used)
        self.log(f"{prefix}/codebook/used_fraction", used / counts.numel())
        self.log(f"{prefix}/codebook/perplexity", perplexity)
        counts.zero_()

    def encode(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode and quantize input data.

        Args:
            batch: Dictionary containing 'csts' tensor of shape [batch_size, n_csts, d_vector]
                and 'mask' tensor of shape [batch_size, n_csts]

        Returns:
            z_hat: quantized codes [batch_size, n_csts, levels_dim]
            indices: codebook indices [batch_size, n_csts], -1 at masked positions
        """
        z = self.encoder(batch) # [batch_size, n_csts, d_vector] -> [batch_size, n_csts, levels_dim] 
        z_hat = self.fsq.quantize(z)
        indices = self.fsq.codes_to_indexes(z_hat)   # [B, N]
        mask = batch["mask"].bool()
        indices = indices.masked_fill(~mask, -1)     # padding -> -1
        return z_hat, indices

    def decode(self, z_hat: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode quantized codes.

        Args:
            z_hat: Quantized codes [batch_size, n_csts, levels_dim]
            batch: Original batch dict with 'csts' and 'mask'

        Returns:
            Reconstructed csts [batch_size, n_csts, d_vector], zeros at masked positions
        """
        return self.decoder(z_hat, batch)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return codebook indices for input data.

        Args:
            batch: Dictionary containing 'csts' and 'mask'

        Returns:
            Indices of shape [batch_size, n_csts], -1 at masked positions
        """
        return self.encode(batch)[1]

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with reconstruction loss.

        Args:
            batch: Dictionary containing data tensors
            batch_idx: Index of current batch

        Returns:
            Total loss tensor
        """
        z_hat, indices = self.encode(batch)
        self._accumulate_code_counts(self._train_code_counts, indices)
        recon_loss, reconstruction = self._reconstruction_loss_and_prediction(z_hat, batch)

        total_loss = recon_loss

        # Log metrics
        self.log("train/total_loss", total_loss, prog_bar=True)
        self.log("train/recon_loss", recon_loss, prog_bar=True)

        self._log_feature_reconstruction("train", reconstruction, batch)

        return total_loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:

        z_hat, indices = self.encode(batch)
        self._accumulate_code_counts(self._val_code_counts, indices)
        recon_loss, reconstruction = self._reconstruction_loss_and_prediction(z_hat, batch)

        total_loss = recon_loss

        # Log metrics
        self.log("val/total_loss", total_loss, prog_bar=True)
        self.log("val/recon_loss", recon_loss, prog_bar=True)

        self._log_feature_reconstruction("val", reconstruction, batch)

        return total_loss

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Predict step returns codebook indices."""
        return self(batch)

    def on_train_epoch_end(self):
        self._log_codebook_usage("train", self._train_code_counts)

    def on_validation_epoch_end(self):
        self._log_codebook_usage("val", self._val_code_counts)

def round_ste(z):
        """Round with straight through gradients."""
        return z + (z.round() - z).detach()

class FSQ(torch.nn.Module):
    """
    Pytorch implementation of a simple FSQ adapted from original JAX version:
    https://github.com/google-research/google-research/blob/master/fsq/fsq.ipynb
    """
    def __init__(self, levels: list[int]):
        super().__init__()
        self.levels = [int(level) for level in levels]  # plain list (also handles Hydra ListConfig)
        levels_tensor = torch.tensor(self.levels, dtype=torch.int32)
        basis = torch.cat(
            (torch.ones(1, dtype=torch.int32), torch.cumprod(levels_tensor[:-1], dim=0))
        ).to(torch.int32)
        # Buffers move with the module to the training device
        self.register_buffer("levels_tensor", levels_tensor, persistent=False)
        self.register_buffer("basis", basis, persistent=False)
        self.implicit_codebook = self.indexes_to_codes(torch.arange(self.codebook_size))

    @property
    def num_dimensions(self) -> int:
        """Number of dimensions expected from inputs."""
        return len(self.levels)

    @property
    def codebook_size(self) -> int:
        """Size of the codebook."""
        return int(torch.prod(self.levels_tensor))

    @property
    def codebook(self) -> torch.tensor:
        """Returns the implicit codebook. Shape (prod(levels), num_dimensions)."""
        return self.implicit_codebook


    def bound(self, z) -> torch.tensor:
        """
        Apply a bounded transformation f(z), approximately centered at zero, with a range chosen to yield L_i integer levels after rounding.
        Example:
        L = 5: f(z) in (-2, 2) approximately
               -> round(f(z)) in {-2, -1, 0, 1, 2}

        L = 8: f(z) in (-4, 3) approximately
               -> round(f(z)) in {-4, -3, -2, -1, 0, 1, 2, 3}

        The offset handles even L, while the shift compensates near z=0.
        Returns:
            f(z) after bounded transformation
        """
        eps = 1e-3
        half_l = (self.levels_tensor - 1) * (1 - eps) / 2
        offset = torch.where(self.levels_tensor % 2 == 1, 0.0, 0.5)
        shift = torch.atanh(offset / half_l)
        return  torch.tanh(z + shift) * half_l - offset

    def quantize(self, z):
        """
        Quanitzes z, returns quantized zhat, same shape as z.
        Returns:
            f(z) after bounded transformation
        """
        quantized = round_ste(self.bound(z))
        half_width = self.levels_tensor // 2 # Renormalize to [-1, 1].
        return quantized / half_width
    
    def _scale_and_shift(self, zhat_normalized) -> torch.Tensor:
        """
        Converts normalized FSQ codes into nonnegative integer levels
        Example:
            L = 5, half_width = 2
            [-1, -0.5, 0, 0.5, 1] -> [0, 1, 2, 3, 4]
        Returns:
            Integer levels tensor
        """
        half_width = self.levels_tensor // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat) -> torch.Tensor:
        """
        Converts onnegative integer levels back to normalized FSQ codes
        Example:
            L = 5, half_width = 2
            [0, 1, 2, 3, 4] -> [-1, -0.5, 0, 0.5, 1]
        Returns:
            normalized FSQ codes
        """
        half_width = self.levels_tensor // 2
        return (zhat - half_width) / half_width

    def codes_to_indexes(self, zhat) -> torch.Tensor:
        """
        Converts a 'code' to an index in the codebook.
        """
        assert zhat.shape[-1] == len(self.levels)

        digits = self._scale_and_shift(zhat).round().long()
        return (digits * self.basis).sum(dim=-1)

    def indexes_to_codes(self, indices) -> torch.Tensor:
        """Convert discrete token indices back into normalized FSQ codes.

        Example:
            levels = [8, 5, 5, 3]
            basis  = [1, 8, 40, 200]
            index  = 186
            digits = [2, 3, 4, 0]
            zhat   = [-0.5, 0.5, 1.0, -1.0]

        Input:  [B, N]
        Output: [B, N, D]
        """
        indices = indices.unsqueeze(-1)
        codes_non_centered = torch.remainder(
            torch.div(
                indices,
                self.basis,
                rounding_mode="floor"
            ),
            self.levels_tensor
        )

        return self._scale_and_shift_inverse(codes_non_centered)
