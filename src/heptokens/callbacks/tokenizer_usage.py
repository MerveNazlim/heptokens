"""Validation-wide code counts for the controlled tokenizer experiment."""

import hashlib
import json
from pathlib import Path

import torch
from lightning.pytorch import Callback


class TokenizerUsageAudit(Callback):
    """Read existing validation outputs, without extra inference or random draws.

    Counts cover the trainer's actual validation pass (including its configured
    batch limit), not the whole dataset and not averages of per-batch occupancy.
    This experiment callback intentionally supports only a single device.
    """

    def __init__(self, stop_after_epochs: int | None = None):
        if stop_after_epochs is not None and (
            type(stop_after_epochs) is not int or stop_after_epochs < 1
        ):
            raise ValueError("stop_after_epochs must be a positive integer or None")
        self.stop_after_epochs = stop_after_epochs
        self.completed_epochs = 0

    def on_fit_start(self, trainer, pl_module):
        if self.stop_after_epochs is not None:
            if not self.stop_after_epochs < trainer.max_epochs:
                raise ValueError("Pilot must stop before the original schedule horizon")
            if (trainer.min_epochs or 0) > self.stop_after_epochs or (trainer.min_steps or 0) > 0:
                raise ValueError("Trainer minimums would prevent the requested pilot stop")

    def on_train_epoch_end(self, trainer, pl_module):
        self.completed_epochs = trainer.current_epoch + 1
        if self.stop_after_epochs is not None and self.completed_epochs >= self.stop_after_epochs:
            # Keep max_epochs unchanged: the scheduler uses the original horizon.
            trainer.should_stop = True

    def on_validation_epoch_start(self, trainer, pl_module):
        if trainer.world_size != 1:
            raise ValueError("TokenizerUsageAudit requires a single-device experiment")
        self.counts = torch.zeros(
            int(pl_module.hparams.num_quantizers),
            int(pl_module.hparams.codebook_size),
            dtype=torch.int64,
        )
        self.input_digest = hashlib.sha256()
        self.batches = 0

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if trainer.sanity_checking:
            return
        indices = outputs["indices"].detach().cpu()
        for q in range(self.counts.shape[0]):
            valid = indices[..., q][indices[..., q] >= 0]
            self.counts[q] += torch.bincount(valid, minlength=self.counts.shape[1])
        for key in ("csts", "mask"):
            array = batch[key].detach().cpu().contiguous().numpy()
            self.input_digest.update(str((key, array.shape, str(array.dtype))).encode())
            self.input_digest.update(array.tobytes())
        self.batches += 1

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        metrics = []
        for q, counts in enumerate(self.counts):
            n = int(counts.sum())
            if n == 0:
                raise ValueError("No valid objects in the controlled validation pass")
            p = counts[counts > 0].double() / n
            used = int((counts > 0).sum())
            perplexity = float(torch.exp(-(p * p.log()).sum()))
            metrics.append(
                {
                    "q": q,
                    "objects": n,
                    "used_codes": used,
                    "used_fraction": used / len(counts),
                    "perplexity": perplexity,
                    "normalized_perplexity": perplexity / len(counts),
                }
            )
            pl_module.log(f"val/aggregate/q{q}_used_codes", float(used))
            pl_module.log(f"val/aggregate/q{q}_perplexity", perplexity)
        record = {
            "epoch": trainer.current_epoch,
            "step": trainer.global_step,
            "validation_batches": self.batches,
            "input_tensor_sha256": self.input_digest.hexdigest(),
            "input_note": "Ordered preprocessed validation tensors, not an event-ID audit.",
            "data_codebook_init": bool(pl_module.hparams.data_codebook_init),
            "initialization_completed": (
                bool(pl_module._data_codebook_initialized)
                if pl_module.hparams.data_codebook_init
                else None
            ),
            "codebook_metrics": metrics,
            "counts": self.counts.tolist(),
            "validation_metrics": {
                str(key): float(value)
                for key, value in trainer.callback_metrics.items()
                if str(key).startswith("val/") and value.numel() == 1
            },
        }
        out = Path(trainer.default_root_dir) / "validation_usage"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"epoch{trainer.current_epoch:03d}_step{trainer.global_step:09d}.json"
        path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")

    def on_fit_end(self, trainer, pl_module):
        if pl_module.hparams.data_codebook_init and not bool(pl_module._data_codebook_initialized):
            raise RuntimeError("Treatment finished without reaching data-codebook initialization")
        if self.stop_after_epochs is not None:
            if self.completed_epochs != self.stop_after_epochs:
                raise RuntimeError("Pilot did not complete exactly the requested number of epochs")
            out = Path(trainer.default_root_dir)
            checkpoint = out / "checkpoints/pilot_end.ckpt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(checkpoint)
            receipt = {
                "completed_epochs": self.completed_epochs,
                "stop_after_epochs": self.stop_after_epochs,
                "schedule_max_epochs": trainer.max_epochs,
                "global_step": trainer.global_step,
                "checkpoint": str(checkpoint),
                "note": "Completed early-behavior pilot, not full 20-epoch training. No automatic resume.",
            }
            (out / "pilot_completion.json").write_text(json.dumps(receipt, indent=2) + "\n")
