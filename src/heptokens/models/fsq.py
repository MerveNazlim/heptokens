"""PyTorch Lightning wrappers for FSQ and residual FSQ autoencoders."""

import logging
import math
from typing import Dict, Tuple
import re

import torch
from lightning import LightningModule
from torch.amp import autocast

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

        self.learning_rate = learning_rate
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
        entropy = -(probs * probs.log()).sum()               # nats
        perplexity = torch.exp(entropy)
        self.log(f"{prefix}/codebook/used_codes", used)
        self.log(f"{prefix}/codebook/used_fraction", used / counts.numel())
        self.log(f"{prefix}/codebook/entropy", entropy)
        # entropy / log(codebook size): 1 = uniform usage, comparable across levels choices
        self.log(f"{prefix}/codebook/normalized_entropy", entropy / math.log(counts.numel()))
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


class LitResidualFSQ(ScheduledOptimiserMixin, LightningModule):
    """Lightning Module wrapper for a residual FSQ autoencoder.

    Args:
        encoder: Encoder, built with output_dim = len(levels)
        decoder: Decoder, built with input_dim = len(levels)
        levels: Number of quantization levels per latent dimension, e.g. [8, 5, 5, 5]
            (codebook size per quantizer = prod(levels))
        num_quantizers: Number of residual FSQ stages; each token is num_quantizers indices
        dim_normalization: Normalize each latent dim over valid objects (BatchNorm without
            affine, running stats in eval) before the soft clamp, so no dim can saturate
        target_std: Std of each latent dim after normalization
        learning_rate: Learning rate for optimizer
        feature_names: Feature names used for per-feature metric logging
        data_sample: Sample data used to infer the input dimension
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        levels: list[int],
        num_quantizers: int = 1,
        dim_normalization: bool = False,
        target_std: float = 0.5,
        learning_rate: float = 1e-3,
        optimizer=None,
        scheduler=None,
        feature_names: list[str] | None = None,
        data_sample: torch.Tensor = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.feature_names = feature_names or []
        self.dim_normalization = dim_normalization
        self.target_std = target_std

        # Infer input dimension from data_sample if provided
        if data_sample is not None:
            input_dim = data_sample["csts"].shape[-1]
        else:
            input_dim = 3  # Default for now

        # Residual FSQ
        self.vector_quantization = ResidualFSQ(
            num_quantizers=num_quantizers,
            levels=levels,
        )

        levels_dim = self.vector_quantization.num_dimensions
        n = self.vector_quantization.codebook_size
        self.register_buffer(
            "_train_code_counts", torch.zeros(num_quantizers, n, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_val_code_counts", torch.zeros(num_quantizers, n, dtype=torch.long), persistent=False
        )

        # Per-dim latent normalization; affine=False so a learned scale cannot push a dim back
        # into saturation. running_mean / running_var are saved with the checkpoint.
        self.latent_norm = (
            torch.nn.BatchNorm1d(levels_dim, affine=False, eps=1e-5, momentum=0.1)
            if dim_normalization
            else None
        )

        # Declare encoder
        self.encoder = encoder(input_dim=input_dim, output_dim=levels_dim)
        # Declare decoder
        self.decoder = decoder(input_dim=levels_dim, output_dim=input_dim)

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
        """counts: [Q, codebook_size]; indices: [..., Q] with -1 at padding."""
        for quantizer_idx in range(counts.shape[0]):
            values = indices[..., quantizer_idx]
            valid = values[values >= 0]                        # drop padding (-1)
            counts[quantizer_idx] += torch.bincount(valid, minlength=counts.shape[1])

    def _log_codebook_usage(self, prefix, counts):
        for quantizer_idx in range(counts.shape[0]):
            stage_counts = counts[quantizer_idx]
            total = stage_counts.sum()
            if total == 0:
                continue
            used = (stage_counts > 0).sum().float()
            probs = stage_counts[stage_counts > 0].float() / total
            entropy = -(probs * probs.log()).sum()           # nats
            perplexity = torch.exp(entropy)
            name = f"{prefix}/codebook/q{quantizer_idx}"
            self.log(f"{name}_used_codes", used)
            self.log(f"{name}_used_fraction", used / stage_counts.numel())
            self.log(f"{name}_entropy", entropy)
            # entropy / log(codebook size): 1 = uniform usage, comparable across levels choices
            self.log(f"{name}_normalized_entropy", entropy / math.log(stage_counts.numel()))
            self.log(f"{name}_perplexity", perplexity)
        counts.zero_()

    def latent(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encoder output, normalized per dim if dim_normalization: [batch_size, n_csts, levels_dim].

        Statistics come from valid objects only; padded positions are set to 0.
        """
        z = self.encoder(batch)
        if self.latent_norm is None:
            return z
        mask = batch["mask"].bool()
        valid = z[mask]  # [n_valid, levels_dim]
        if self.training and valid.shape[0] < 2:
            # batch statistics need at least two objects; use the running ones instead
            normed = torch.nn.functional.batch_norm(
                valid, self.latent_norm.running_mean, self.latent_norm.running_var,
                training=False, eps=self.latent_norm.eps,
            )
        else:
            normed = self.latent_norm(valid)
        z_norm = normed.new_zeros(z.shape)
        z_norm[mask] = normed * self.target_std
        return z_norm

    def encode(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode and quantize input data.

        Args:
            batch: Dictionary containing 'csts' tensor of shape [batch_size, n_csts, d_vector]
                and 'mask' tensor of shape [batch_size, n_csts]

        Returns:
            z_hat: quantized codes summed over stages [batch_size, n_csts, levels_dim]
            indices: per-stage codebook indices [batch_size, n_csts, num_quantizers],
                -1 at masked positions
        """
        z = self.latent(batch)  # [batch_size, n_csts, d_vector] -> [batch_size, n_csts, levels_dim]
        z_hat, indices = self.vector_quantization.n_quantize(z)
        mask = batch["mask"].bool()
        indices = indices.masked_fill(~mask.unsqueeze(-1), -1)  # padding -> -1
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
            Indices of shape [batch_size, n_csts, num_quantizers], -1 at masked positions
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
        half_l = (self.levels_tensor - 1) * (1 + eps) / 2
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


class SymmetricFSQ(FSQ):
    """
    FSQ stage for residual quantization:
    Use symmetric grid {-1, ..., 1} with step 2/(L-1) for every L (the base FSQ grid is asymmetric
    for even L), and a hard clamp instead of tanh, since the residual input is already in range.
    For odd L the grid is the same as FSQ's.
    """
    def quantize(self, z):
        levels_minus_1 = self.levels_tensor - 1
        x = z.clamp(-1.0, 1.0)
        return round_ste(levels_minus_1 * (x + 1) / 2) * 2 / levels_minus_1 - 1

    def _scale_and_shift(self, zhat_normalized) -> torch.Tensor:
        """[-1, 1] -> {0, ..., L-1}"""
        return (zhat_normalized + 1) * (self.levels_tensor - 1) / 2

    def _scale_and_shift_inverse(self, zhat) -> torch.Tensor:
        """{0, ..., L-1} -> [-1, 1]"""
        return zhat * 2 / (self.levels_tensor - 1) - 1


class ResidualFSQ(torch.nn.Module):
    """
    Pytorch implementation of a residual FSQ simplified from:
    https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/residual_fsq.py

    Input is soft-clamped once to tanh(z / c) * c with c = 1 + 1/(L-1).
    Each stage then quantizes residual / scale on a symmetric grid (step 2/(L-1)) with a hard clamp, scale = L ** -stage. 
    The residual after a stage is at most 1/(L-1) * scale, so residual / next_scale lies in [-c, c].
    """
    def __init__(self, num_quantizers: int, levels: list[int]):
        super().__init__()
        levels_tensor = torch.tensor([int(level) for level in levels], dtype=torch.float32)

        self.layers = torch.nn.ModuleList()
        scales = []
        for ind in range(num_quantizers):
            scales.append(levels_tensor ** -ind)
            self.layers.append(SymmetricFSQ(levels=levels))

        self.register_buffer("scales", torch.stack(scales), persistent=False)  # [Q, D]
        self.register_buffer("soft_clamp_value", 1 + 1 / (levels_tensor - 1), persistent=False)

    @property
    def levels(self) -> list[int]:
        return self.layers[0].levels

    @property
    def num_quantizers(self) -> int:
        return len(self.layers)

    @property
    def num_dimensions(self) -> int:
        """Number of dimensions expected from inputs."""
        return self.layers[0].num_dimensions

    @property
    def codebook_size(self) -> int:
        return self.layers[0].codebook_size

    @property
    def codebooks(self) -> torch.Tensor :
        """Per-stage implicit codebooks (unscaled).  [Q, prod(levels), num_dimensions]"""
        codebooks = [layer.implicit_codebook for layer in self.layers]
        codebooks = torch.stack(codebooks, dim = 0)
        return codebooks

    def n_quantize(self, z):
        """
        Quantizes z with num_quantizers residual stages.
        Returns:
            z_hat: sum of scaled stage codes, same shape as z
            indices: per-stage indices [..., num_quantizers]
        """
        z_hat_out = 0.
        all_indices = []

        with autocast('cuda', enabled = False):
            # soft clamp once; later stages see residuals already in range
            c = self.soft_clamp_value
            residual = torch.tanh(z.float() / c) * c

            for layer, scale in zip(self.layers, self.scales):
                z_hat = layer.quantize(residual / scale)
                indices = layer.codes_to_indexes(z_hat)
                z_hat = z_hat * scale

                residual = residual - z_hat.detach()
                z_hat_out = z_hat_out + z_hat

                all_indices.append(indices)

        all_indices = torch.stack(all_indices, dim = -1)
        return z_hat_out, all_indices

    def indexes_to_codes(self, indices) -> torch.Tensor:
        """Per-stage indices [..., Q] -> summed codes [..., num_dimensions]."""
        codes = [
            layer.indexes_to_codes(indices[..., q]) * scale
            for q, (layer, scale) in enumerate(zip(self.layers, self.scales))
        ]
        return torch.stack(codes, dim=0).sum(dim=0)
