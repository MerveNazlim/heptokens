"""Five-class HZZ supervision over the parallel Q8 or flat continuous CLS state."""

import hashlib
import json
from contextlib import nullcontext

import torch
from lightning import LightningModule
from torch import nn
from torch.nn import functional as F
from torchmetrics import AUROC, Accuracy

from heptokens.models.foundation_continuous import ContinuousSequenceBackbone
from heptokens.models.foundation_grouped import GroupedSequenceBackbone
from heptokens.models.utils import warmup_cosine_scheduler


def backbone_fingerprint(state):
    if not state:
        raise ValueError("Missing backbone weights")
    fingerprint = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite backbone tensor: {name}")
        descriptor = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":"))
        fingerprint.update(descriptor.encode() + b"\0")
        fingerprint.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return fingerprint.hexdigest()


class HZZClassifier(LightningModule):
    def __init__(self, *, family, condition, hparams, schema, vocabulary, weights, protocol, backbone_state=None):
        super().__init__()
        if condition not in {"finetune", "frozen", "scratch"}:
            raise ValueError(f"Unknown training condition: {condition}")
        if (condition == "scratch") != (backbone_state is None):
            raise ValueError("Only scratch can initialize without a source backbone")
        self.family = family
        self.frozen = condition == "frozen"
        self.protocol = protocol
        args = dict(hparams)
        if family == "q8":
            if args.pop("object_projection_mode") != "linear" or schema is not None:
                raise ValueError("Only parallel Q8 with a learned linear projection is supported")
            tokens = vocabulary["special_tokens"]
            self.pad_token_id, self.cls_token_id = int(tokens["pad"]), int(tokens["cls"])
            if (self.pad_token_id != 0 or self.cls_token_id != 1
                    or not vocabulary["max_quantizers"] == args["max_quantizers"] == 8
                    or vocabulary["vocab_size"] != args["vocab_size"]):
                raise ValueError("Vocabulary differs from the original Q8 model")
            self.backbone = GroupedSequenceBackbone(pad_token_id=self.pad_token_id, **args)
        elif family == "continuous_flat":
            if args.pop("feature_grouping") != "schema" or args["architecture"] != "flat" or vocabulary is not None:
                raise ValueError("Only the original flat continuous schema is supported")
            self.backbone = ContinuousSequenceBackbone(
                continuous_schema=schema, inner_num_heads=4, inner_num_layers=1, **args)
            if self.backbone.role_embedding is None:
                raise ValueError("The flat baseline requires the role embedding")
        else:
            raise ValueError(f"Unsupported representation: {family}")
        if backbone_state is not None:
            self.backbone.load_state_dict(backbone_state, strict=True)
        self.classifier = nn.Linear(args["hidden_dim"], 5)
        if self.frozen:
            self.backbone.requires_grad_(False)
            self.backbone.eval()
        weights = torch.as_tensor(weights, dtype=torch.float32)
        if weights.shape != (5,) or not torch.isfinite(weights).all() or torch.any(weights <= 0):
            raise ValueError("Expected five positive finite training class weights")
        self.register_buffer("class_weights", weights, persistent=True)
        for prefix in ("train", "valid"):
            setattr(self, f"{prefix}_acc", Accuracy(task="multiclass", num_classes=5))
            setattr(self, f"{prefix}_auc", AUROC(task="multiclass", num_classes=5, average="macro"))

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, batch):
        mask, types = batch["mask"].bool(), batch["type_ids"].long()
        if self.family == "q8":
            tokens = batch["tokens"]
            if (tokens.ndim != 3 or not torch.all(mask[:, 0])
                    or not torch.all(tokens[:, 0, 0] == self.cls_token_id)
                    or not torch.all(tokens[:, 0, 1:] == self.pad_token_id)):
                raise ValueError("Q8 position zero must contain one active CLS code")
            arguments = tokens, mask, types
        else:
            roles = batch["position_role_ids"].long()
            if not torch.all(roles[:, 0] == self.backbone.role_ids["cls"]):
                raise ValueError("Continuous position zero must be CLS")
            arguments = batch["continuous_features"].float(), mask, types, roles
        with torch.no_grad() if self.frozen else nullcontext():
            hidden = self.backbone(*arguments)
        return self.classifier(hidden[:, 0])

    def step(self, batch, prefix):
        logits, labels = self(batch), batch["labels"].long()
        loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        auc, acc = getattr(self, f"{prefix}_auc"), getattr(self, f"{prefix}_acc")
        auc.update(logits.softmax(-1), labels)
        acc.update(logits.argmax(-1), labels)
        self.log(f"{prefix}/total_loss", loss, on_step=prefix == "train", on_epoch=True)
        self.log(f"{prefix}/acc", acc, on_step=False, on_epoch=True)
        self.log(f"{prefix}/auc", auc, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "valid")

    def configure_optimizers(self):
        config = self.protocol
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=config["learning_rate"], weight_decay=config["weight_decay"])
        scheduler = warmup_cosine_scheduler(
            optimizer, warmup_epochs=config["warmup_epochs"], min_lr=config["min_lr"], max_epochs=config["epochs"])
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
