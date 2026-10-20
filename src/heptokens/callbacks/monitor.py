"""Callbacks for monitoring model training."""

import logging
import re
import time
from typing import Any

from lightning import Callback, LightningModule, Trainer
from torch.optim import Optimizer

from heptokens.utils.torch_utils import (
    get_activations,
    get_submodules,
    gradient_norm,
)

log = logging.getLogger(__name__)


class ActivationMonitor(Callback):
    """Callback to monitor the activations magnitudes at select layers in a model."""

    def __init__(
        self,
        logging_interval: int = 100,
        layer_types: list | None = None,
        layer_regex: list | None = None,
        param_regex: list | None = None,
    ) -> None:
        self.logging_interval = logging_interval
        self.layer_types = layer_types
        self.layer_regex = layer_regex
        self.param_regex = param_regex
        self.act_dict = {}

    def on_train_batch_start(
        self,
        _trainer: Trainer,
        pl_module: LightningModule,
        _batch: Any,
        batch_idx: int,
    ) -> None:
        """Add hooks to the model to monitor the layer activations."""
        if batch_idx % self.logging_interval != 0:
            return
        pl_module.hooks = get_activations(
            pl_module,
            self.act_dict,
            types=self.layer_types,
            regex=self.layer_regex,
        )

    def on_train_batch_end(
        self,
        _trainer: Trainer,
        pl_module: LightningModule,
        _outputs: Any,
        _batch: Any,
        batch_idx: int,
    ) -> None:
        """Remove the hooks after the batch and log the activations."""
        if batch_idx % self.logging_interval != 0:
            return
        for key, value in self.act_dict.items():
            pl_module.log(f"activations/{key}", value)
        self.act_dict = {}
        for hook in pl_module.hooks:
            hook.remove()
        for n, p in pl_module.named_parameters():
            if any(re.match(r, n) for r in self.param_regex):
                self.log(f"param/{n}", p.detach().abs().mean())


class LogGradNorm(Callback):
    """Logs the gradient norm."""

    def __init__(self, logging_interval: int = 1, depth: int = 0):
        self.logging_interval = logging_interval
        self.depth = depth

    def on_before_optimizer_step(
        self, _trainer: Trainer, pl_module: LightningModule, _optimizer: Optimizer
    ):
        if pl_module.global_step % self.logging_interval == 0:
            sub_modules = get_submodules(pl_module, self.depth)
            for subname, module in sub_modules:
                grad = gradient_norm(module)
                if grad > 0:
                    self.log("grad/" + subname, gradient_norm(module))


class EpochSummary(Callback):
    """Log one line of metrics per epoch, a compact replacement for the progress bar in batch logs.

    Args:
        exclude_regex: Metrics whose name matches are left out, e.g. per-feature metrics
        every_n_steps: Also log a short line every n optimizer steps (0 = only per epoch)
    """

    def __init__(self, exclude_regex: str | None = "/feature_", every_n_steps: int = 0) -> None:
        self.exclude = re.compile(exclude_regex) if exclude_regex else None
        self.every_n_steps = every_n_steps
        self.start = None

    def _metrics(self, trainer: Trainer, prefixes: tuple[str, ...] = ("",)) -> str:
        items = []
        for name, value in sorted(trainer.callback_metrics.items()):
            if self.exclude is not None and self.exclude.search(name):
                continue
            if not name.startswith(prefixes) or getattr(value, "numel", lambda: 1)() != 1:
                continue
            items.append(f"{name}={float(value):.5g}")
        return " ".join(items)

    def _header(self, trainer: Trainer) -> str:
        elapsed = int(time.monotonic() - self.start) if self.start is not None else 0
        hours, rest = divmod(elapsed, 3600)
        return (
            f"epoch {trainer.current_epoch + 1}/{trainer.max_epochs} step {trainer.global_step} "
            f"[{hours}:{rest // 60:02d}:{rest % 60:02d}]"
        )

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.start = time.monotonic()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        _pl_module: LightningModule,
        _outputs: Any,
        _batch: Any,
        _batch_idx: int,
    ) -> None:
        if not self.every_n_steps or trainer.global_step % self.every_n_steps or not trainer.is_global_zero:
            return
        log.info("%s | %s", self._header(trainer), self._metrics(trainer, ("train/",)))

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Runs after the end-of-epoch validation, so val/* metrics are from this epoch.
        if trainer.is_global_zero:
            log.info("%s | %s", self._header(trainer), self._metrics(trainer))
