"""Callbacks for monitoring model training."""

import csv
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from lightning import Callback, LightningModule, Trainer
from torch.optim import Optimizer

from heptokens.utils.torch_utils import (
    get_activations,
    get_submodules,
    gradient_norm,
)

log = logging.getLogger(__name__)


class MaskingPolicyAudit(Callback):
    """Audit actual masks without dropping events or changing their weights.

    Counts are summed across ranks at shared logging/epoch boundaries. Ratios
    therefore use global denominators, including events with zero objects.
    """

    COUNT_NAMES = (
        "events", "eligible_events", "forced_events", "physical_maskless_before",
        "physical_maskless_after", "physical_positions", "masked_physical_positions",
        "context_positions", "masked_context_positions",
    )

    def __init__(self, path: str, logging_interval: int = 100) -> None:
        if logging_interval < 1:
            raise ValueError("logging_interval must be positive")
        self.path = Path(path)
        self.logging_interval = logging_interval
        self.counts: dict[str, torch.Tensor] = {}

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self.counts.pop("train", None)

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self.counts.pop("valid", None)

    def _accumulate(self, pl_module, split: str) -> None:
        prefix, counts = pl_module.last_masking_counts
        if prefix != split:
            raise RuntimeError(f"Masking audit split mismatch: {prefix} vs {split}")
        if split not in self.counts:
            self.counts[split] = counts.clone()
        else:
            self.counts[split] += counts

    def _snapshot(self, trainer, pl_module, split: str) -> None:
        counts = self.counts[split].clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        values = dict(zip(self.COUNT_NAMES, counts.cpu().tolist()))
        if values["physical_maskless_after"]:
            raise RuntimeError("Per-event physical-object masking guarantee was violated")
        if not trainer.is_global_zero:
            return
        values.update(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            epoch=trainer.current_epoch,
            global_step=trainer.global_step,
            split=split,
            mask_prob=pl_module.mask_prob,
            zero_physical_events=values["events"] - values["eligible_events"],
            effective_physical_mask_fraction=(
                values["masked_physical_positions"] / values["physical_positions"]
                if values["physical_positions"] else ""
            ),
        )
        fields = ["timestamp_utc", "epoch", "global_step", "split", "mask_prob",
                  *self.COUNT_NAMES, "zero_physical_events", "effective_physical_mask_fraction"]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        header = not self.path.exists()
        with self.path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if header:
                writer.writeheader()
            writer.writerow(values)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self._accumulate(pl_module, "train")
        if trainer.global_step and trainer.global_step % self.logging_interval == 0:
            self._snapshot(trainer, pl_module, "train")

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if "train" in self.counts:
            self._snapshot(trainer, pl_module, "train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx,
                                dataloader_idx=0) -> None:
        if not trainer.sanity_checking:
            self._accumulate(pl_module, "valid")

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not trainer.sanity_checking and "valid" in self.counts:
            self._snapshot(trainer, pl_module, "valid")


class LiveMetricsCSV(Callback):
    """Write rank-averaged training and validation metrics to a live CSV file."""

    def __init__(self, path: str, logging_interval: int = 100) -> None:
        self.path = Path(path)
        self.logging_interval = logging_interval

    @staticmethod
    def _distributed_mean(value: torch.Tensor) -> float:
        metric = value.detach().float().clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(metric, op=dist.ReduceOp.SUM)
            metric /= dist.get_world_size()
        return float(metric.cpu())

    def _write(
        self,
        trainer: Trainer,
        *,
        split: str,
        loss: float,
        accuracy: float | None,
    ) -> None:
        if not trainer.is_global_zero:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.path.exists()
        with self.path.open("a", newline="") as stream:
            writer = csv.writer(stream)
            if write_header:
                writer.writerow(
                    ["timestamp_utc", "epoch", "global_step", "split", "loss", "mask_acc", "lr"]
                )
            lr = trainer.optimizers[0].param_groups[0]["lr"] if trainer.optimizers else ""
            writer.writerow(
                [
                    datetime.now(timezone.utc).isoformat(),
                    trainer.current_epoch,
                    trainer.global_step,
                    split,
                    loss,
                    "" if accuracy is None else accuracy,
                    lr,
                ]
            )
            stream.flush()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        _pl_module: LightningModule,
        outputs: Any,
        _batch: Any,
        _batch_idx: int,
    ) -> None:
        if trainer.global_step == 0 or trainer.global_step % self.logging_interval:
            return
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if not isinstance(loss, torch.Tensor):
            return
        loss_value = self._distributed_mean(loss)
        accuracy = trainer.callback_metrics.get("train/mask_acc")
        accuracy_value = (
            self._distributed_mean(accuracy)
            if isinstance(accuracy, torch.Tensor)
            else None
        )
        self._write(
            trainer,
            split="train",
            loss=loss_value,
            accuracy=accuracy_value,
        )

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        _pl_module: LightningModule,
    ) -> None:
        loss = trainer.callback_metrics.get("valid/total_loss")
        accuracy = trainer.callback_metrics.get("valid/mask_acc")
        if not isinstance(loss, torch.Tensor):
            return
        # These validation values were already synchronized by self.log.
        self._write(
            trainer,
            split="valid",
            loss=float(loss.detach().cpu()),
            accuracy=(
                float(accuracy.detach().cpu())
                if isinstance(accuracy, torch.Tensor)
                else None
            ),
        )


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
