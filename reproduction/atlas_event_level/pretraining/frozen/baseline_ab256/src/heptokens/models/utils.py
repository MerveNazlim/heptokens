import logging
import math

import torch as T
from lightning import LightningModule
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)


class ScheduledOptimiserMixin:
    """Mixin providing ``configure_optimizers`` via Hydra-injected partials.

    Models using this mixin may accept ``optimizer`` and ``scheduler`` as
    ``functools.partial`` objects (typically via Hydra ``_partial_: true``).
    If no optimizer partial is provided, falls back to Adam with
    ``self.learning_rate``.  If no scheduler partial is provided, no
    scheduler is used.
    """

    def configure_optimizers(self):
        hp = getattr(self, "hparams", {})

        # --- optimizer ---
        opt_partial = hp.get("optimizer", None)
        if callable(opt_partial):
            opt = opt_partial(self.parameters())
        else:
            lr = self.__dict__.get("learning_rate") or hp.get("learning_rate", 1e-3)
            opt = T.optim.Adam(self.parameters(), lr=lr)

        # --- scheduler (optional) ---
        sched_partial = hp.get("scheduler", None)
        if callable(sched_partial):
            sched = sched_partial(optimizer=opt, model=self)
            return {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
            }

        return opt


def get_max_steps(model: LightningModule) -> int:
    """Get the maximum number of steps from the model trainer."""
    try:
        logger.info("Attempting to get the max steps from the model trainer")
        max_steps = model.trainer.max_steps
        if max_steps < 1:
            steps_per_epoch = len(model.trainer.datamodule.train_dataloader())
            max_epochs = model.trainer.max_epochs
            max_steps = steps_per_epoch * max_epochs
        logger.info(f"Success:  max_steps = {max_steps}")
    except Exception as e:
        logger.info(f"Failed to get max steps from the model trainer: {e}")
        max_steps = 0
    return max_steps


def warmup_cosine_scheduler(
    optimizer: Optimizer,
    warmup_epochs: int = 5,
    min_lr: float = 1e-6,
    model: LightningModule | None = None,
    max_epochs: int = -1,
) -> T.optim.lr_scheduler.SequentialLR:
    """Linear warmup followed by cosine annealing, configured in epochs.

    If ``max_epochs`` is -1 (default), it is read from ``model.trainer``.
    """
    if max_epochs < 1 and model is not None:
        max_epochs = model.trainer.max_epochs
    if max_epochs < 1:
        raise ValueError("max_epochs must be positive (set it or pass a model with a trainer).")

    warmup = min(warmup_epochs, max_epochs)
    warmup_sched = T.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-2, total_iters=warmup,
    )
    cosine_sched = T.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(max_epochs - warmup, 1), eta_min=min_lr,
    )
    return T.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup],
    )


def linear_warmup_cosine_decay(
    optimizer: Optimizer,
    warmup_steps: int = 1000,
    total_steps: int = 10000,
    final_factor: float = 5e-2,
    init_factor: float = 1e-5,
    model: LightningModule | None = None,
) -> LambdaLR:
    """Return a scheduler with a linear warmup and a cosine decay."""
    # Attempt to get the max steps from the model trainer
    if total_steps == -1 and model is not None:
        total_steps = get_max_steps(model)

    warmup_steps = max(1, warmup_steps)  # Avoid division by zero
    assert 0 < final_factor < 1, "Final factor must be less than 1"
    assert 0 < init_factor < 1, "Initial factor must be less than 1"
    assert 0 < warmup_steps < total_steps, "Total steps must be greater than warmup"

    def fn(x: int) -> float:
        if x <= warmup_steps:
            return init_factor + x * (1 - init_factor) / warmup_steps
        if x >= total_steps:
            return final_factor
        t = (x - warmup_steps) / (total_steps - warmup_steps) * math.pi
        return (1 + math.cos(t)) * (1 - final_factor) / 2 + final_factor

    return LambdaLR(optimizer, fn)


class JetBackbone(nn.Module):
    """Generalised backbone for the jet models.

    Simply wraps the constituent embedding and encoder together in a single module.
    Easy for saving and loading using the pickle module.
    """

    def __init__(
        self,
        cst_emb: nn.Module,
        jet_emb: nn.Module,
        encoder,  # type: Transformer  # TODO: write this class!
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.cst_emb = cst_emb
        self.jet_emb = jet_emb
        self.encoder = encoder
        self.causal = causal
        self.dim = encoder.dim
        self.outp_dim = encoder.outp_dim

    def forward(
        self,
        csts: T.Tensor,
        mask: T.Tensor,
        jets: T.Tensor,
    ) -> T.Tensor:
        """Pass through the complete backbone."""
        csts = self.cst_emb(csts)
        jets = self.jet_emb(jets)
        x = self.encoder(csts, mask=mask, ctxt=jets, causal=self.causal)
        new_mask = self.encoder.get_combined_mask(mask)  # Registers
        return x, new_mask
