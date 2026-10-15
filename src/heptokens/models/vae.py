"""PyTorch Lightning wrapper for VAE."""

import logging
from typing import Dict, Tuple
import re

import torch
from lightning import LightningModule

from heptokens.models.coders import Decoder, Encoder
from heptokens.models.utils import ScheduledOptimiserMixin

log = logging.getLogger(__name__)


class LitStandardVae(ScheduledOptimiserMixin, LightningModule):
    """Lightning Module wrapper for a per-constituent Gaussian VAE.

    Args:
        encoder: Encoder, built with output_dim = 2 * latent_dim (mu, logvar)
        decoder: Decoder, built with input_dim = latent_dim
        latent_dim: Dimension of the latent variable z
        beta: KL weight relative to the standard ELBO with a unit-variance Gaussian
            decoder. The reconstruction loss is a per-feature mean, so the applied
            KL weight is beta * 2 / input_dim; beta = 1 recovers the sigma = 1 ELBO
        kl_scale: Applied KL weight without the 2 / input_dim rescaling. If set, it
            overrides beta, and beta is recorded as kl_scale * input_dim / 2
        kl_warmup_steps: Linearly ramp the KL weight from 0 to its full value over
            this many optimizer steps (0 disables warmup)
        learning_rate: Learning rate for optimizer
        feature_names: Feature names used for per-feature metric logging
        data_sample: Sample data used to infer the input dimension
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        latent_dim: int = 16,
        beta: float = 1.0,
        kl_scale: float | None = None,
        kl_warmup_steps: int = 0,
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
        self.kl_warmup_steps = kl_warmup_steps
        self.feature_names = feature_names or []

        # Infer input dimension from data_sample if provided
        if data_sample is not None:
            input_dim = data_sample["csts"].shape[-1]
        else:
            input_dim = 3  # Default for now

        # Recon loss is mean_d (x - x_hat)^2, the sigma = 1 Gaussian NLL scaled by 2 / D,
        # KL term is scaled by the same factor to match the ELBO at beta = 1
        self.elbo_kl_factor = 2.0 / input_dim
        if kl_scale is None:
            kl_scale = beta * self.elbo_kl_factor
        else:
            beta = kl_scale / self.elbo_kl_factor
        self.beta = beta
        self.kl_scale = kl_scale
        # Stored in hparams (checkpoint + logger) so runs can be compared in either convention
        self.hparams["beta"] = self.beta
        self.hparams["elbo_kl_factor"] = self.elbo_kl_factor
        self.hparams["kl_scale"] = self.kl_scale

        # Declare encoder
        self.encoder = encoder(input_dim=input_dim, output_dim=2 * latent_dim)
        # Declare decoder
        self.decoder = decoder(input_dim=latent_dim, output_dim=input_dim)

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
        z: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        reconstruction = self.decode(z, batch)
        mask = batch["mask"].bool()
        reconstructed = reconstruction[mask]
        original = batch["csts"][mask]

        reconstruction_loss = ((reconstructed - original) ** 2).mean(dim=-1).mean()
        return reconstruction_loss, reconstruction

    @staticmethod
    def _kl_divergence(
        z_mu: torch.Tensor,
        z_logvar: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """KL(q(z|x) || N(0, I)) averaged over valid constituents.

        Returns:
            kl_loss: KL summed over latent dims, averaged over valid constituents
            kl_per_dim: KL per latent dim, averaged over valid constituents
        """
        kl = -0.5 * (1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
        kl_per_dim = kl[batch["mask"].bool()].mean(dim=0)
        return kl_per_dim.sum(), kl_per_dim

    def _kl_weight(self) -> float:
        if self.kl_warmup_steps <= 0:
            return self.kl_scale
        return self.kl_scale * min(1.0, self.global_step / self.kl_warmup_steps)

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

    def _log_active_latent_dims(self, prefix: str, kl_per_dim: torch.Tensor) -> None:
        # Latent dims with KL ~ 0 have collapsed to the prior and carry no information
        active = (kl_per_dim > 0.01).sum().float()
        self.log(f"{prefix}/active_latent_dims", active, on_step=False, on_epoch=True)

    def encode(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input data to the parameters of q(z|x).

        Args:
            batch: Dictionary containing 'csts' tensor of shape [batch_size, n_csts, d_vector]
                and 'mask' tensor of shape [batch_size, n_csts]

        Returns:
            z_mu: Mean of latent variable distribution [batch_size, n_csts, latent_dim]
            z_logvar: Log variance of latent variable distribution [batch_size, n_csts, latent_dim]
        """
        latent_params = self.encoder(batch)
        z_mu, z_logvar = latent_params.chunk(2, dim=-1)
        # Keep exp(logvar) finite early in training
        z_logvar = z_logvar.clamp(-20.0, 10.0)
        return z_mu, z_logvar

    def reparameterize(self, z_mu: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:
        """Sample z = mu + sigma * eps with eps ~ N(0, I).

        Args:
            z_mu: Mean of latent variable distribution
            z_logvar: Log variance of latent variable distribution

        Returns:
            z: Sampled latent variable
        """
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        return z_mu + std * eps

    def decode(self, z: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode latent variable.

        Args:
            z: Latent variable [batch_size, n_csts, latent_dim]
            batch: Original batch dict with 'csts' and 'mask'

        Returns:
            Reconstructed csts [batch_size, n_csts, d_vector], zeros at masked positions
        """
        return self.decoder(z, batch)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the latent mean for input data.

        Args:
            batch: Dictionary containing 'csts' and 'mask'

        Returns:
            z_mu of shape [batch_size, n_csts, latent_dim], zeros at masked positions
        """
        return self.encode(batch)[0]

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with reconstruction and KL loss.

        Args:
            batch: Dictionary containing data tensors
            batch_idx: Index of current batch

        Returns:
            Total loss tensor
        """
        z_mu, z_logvar = self.encode(batch)
        z = self.reparameterize(z_mu, z_logvar)

        recon_loss, reconstruction = self._reconstruction_loss_and_prediction(z, batch)
        kl_loss, kl_per_dim = self._kl_divergence(z_mu, z_logvar, batch)

        kl_weight = self._kl_weight()
        total_loss = recon_loss + kl_weight * kl_loss

        # Log metrics
        self.log("train/total_loss", total_loss, prog_bar=True)
        # kl_loss is the raw KL; kl_weighted is the term actually added to the loss
        self.log("train/kl_loss", kl_loss, prog_bar=True)
        self.log("train/kl_weighted", kl_weight * kl_loss)
        self.log("train/recon_loss", recon_loss, prog_bar=True)
        self.log("train/kl_weight", kl_weight)

        self._log_feature_reconstruction("train", reconstruction, batch)
        self._log_active_latent_dims("train", kl_per_dim)

        return total_loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:

        z_mu, z_logvar = self.encode(batch)
        z = self.reparameterize(z_mu, z_logvar)

        recon_loss, reconstruction = self._reconstruction_loss_and_prediction(z, batch)
        kl_loss, kl_per_dim = self._kl_divergence(z_mu, z_logvar, batch)

        # Report the full-weight objective so val loss is comparable across epochs
        total_loss = recon_loss + self.kl_scale * kl_loss

        # Log metrics
        self.log("val/total_loss", total_loss, prog_bar=True)
        self.log("val/kl_loss", kl_loss, prog_bar=True)
        self.log("val/kl_weighted", self.kl_scale * kl_loss)
        self.log("val/recon_loss", recon_loss, prog_bar=True)

        self._log_feature_reconstruction("val", reconstruction, batch)
        self._log_active_latent_dims("val", kl_per_dim)

        return {"val_loss": total_loss}

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Predict step returns the latent mean z_mu."""
        return self(batch)
