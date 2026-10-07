#!/usr/bin/env python3
"""Evaluate grouped HZZ CLS controls using the common held-out test logic."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

import evaluate_grouped_hzz_mean_pool_controls as common
from heptokens.data.sequence import LABELS_KEY
from heptokens.models.foundation_grouped_cls_classifier import LitGroupedCLSClassifier


common.RUNS = {
    "pretrained_frozen_cls": "Pretrained, frozen",
    "random_frozen_cls": "Random, frozen",
    "pretrained_finetuned_cls": "Pretrained, fine-tuned",
    "random_finetuned_cls": "Random, from scratch",
}


def evaluate_checkpoint(checkpoint: Path, dataloader, device: torch.device):
    model = LitGroupedCLSClassifier.load_from_checkpoint(checkpoint, map_location="cpu")
    model.to(device).eval()
    scores = []
    labels = []
    loss_sum = 0.0
    event_count = 0
    with torch.inference_mode():
        for batch in dataloader:
            batch = {
                key: value.to(device, non_blocking=True)
                if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            logits = model(batch)
            targets = batch[LABELS_KEY].long()
            loss_sum += F.cross_entropy(logits, targets, reduction="sum").item()
            event_count += targets.numel()
            scores.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
            labels.append(targets.cpu())

    score = torch.cat(scores).numpy()
    label = torch.cat(labels).numpy()
    fpr, tpr, thresholds = roc_curve(label, score)
    n_background = int(np.count_nonzero(label == 0))
    minimum_fpr = 1.0 / max(n_background, 1)
    result = {
        "checkpoint": str(checkpoint.resolve()),
        "events": int(event_count),
        "signal_events": int(np.count_nonzero(label == 1)),
        "background_events": n_background,
        "test_loss": loss_sum / event_count,
        "test_accuracy": accuracy_score(label, score >= 0.5),
        "test_auc": roc_auc_score(label, score),
    }
    for target in common.TARGET_SIGNAL_EFFICIENCIES:
        index = min(int(np.searchsorted(tpr, target, side="left")), len(tpr) - 1)
        suffix = int(target * 100)
        result[f"background_efficiency_at_signal_efficiency_{suffix}"] = float(fpr[index])
        result[f"background_rejection_at_signal_efficiency_{suffix}"] = float(
            1.0 / max(fpr[index], minimum_fpr)
        )
        result[f"threshold_at_signal_efficiency_{suffix}"] = float(thresholds[index])
    return result, (tpr, fpr)


common.evaluate_checkpoint = evaluate_checkpoint


def add_default(flag: str, value: str) -> None:
    if flag not in sys.argv:
        sys.argv.extend([flag, value])


if __name__ == "__main__":
    add_default(
        "--project-dir",
        "results/atlas_hzz_grouped_cls_classification",
    )
    add_default(
        "--prepared-dir",
        "results/event_tokens_grouped_cls_final_new_mcdata/"
        "hzz_ggf_vs_zz_cls_classification_shards",
    )
    common.main()
