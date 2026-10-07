#!/usr/bin/env python3
"""Evaluate CLS and mean-pooled two-layer MLP HZZ controls."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from heptokens.data.sequence import LABELS_KEY
from heptokens.data.token_parquet import GroupedTokenParquetClassificationModule
from heptokens.models.foundation_grouped_cls_classifier import LitGroupedCLSClassifier
from heptokens.models.foundation_grouped_mean_classifier import LitGroupedMaskedMeanClassifier


RUNS = {
    "pretrained_frozen_cls_mlp": ("cls", "Pretrained, frozen", LitGroupedCLSClassifier),
    "random_frozen_cls_mlp": ("cls", "Random, frozen", LitGroupedCLSClassifier),
    "pretrained_finetuned_cls_mlp": ("cls", "Pretrained, fine-tuned", LitGroupedCLSClassifier),
    "random_finetuned_cls_mlp": ("cls", "Random, from scratch", LitGroupedCLSClassifier),
    "pretrained_frozen_mean_mlp": ("mean", "Pretrained, frozen", LitGroupedMaskedMeanClassifier),
    "random_frozen_mean_mlp": ("mean", "Random, frozen", LitGroupedMaskedMeanClassifier),
    "pretrained_finetuned_mean_mlp": ("mean", "Pretrained, fine-tuned", LitGroupedMaskedMeanClassifier),
    "random_finetuned_mean_mlp": ("mean", "Random, from scratch", LitGroupedMaskedMeanClassifier),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("results/atlas_hzz_grouped_mlp_pooling_controls"))
    parser.add_argument("--prepared-dir", type=Path, default=Path("results/event_tokens_grouped_cls_final_new_mcdata/hzz_ggf_vs_zz_cls_classification_shards"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--pooling", choices=("all", "cls", "mean"), default="all")
    return parser.parse_args()


def checkpoint_for(directory: Path) -> Path:
    for name in ("best.ckpt", "last.ckpt"):
        path = directory / "checkpoints" / name
        if path.is_file(): return path
    raise FileNotFoundError(f"No checkpoint under {directory}")


def evaluate(model_class, checkpoint, loader, device):
    model = model_class.load_from_checkpoint(checkpoint, map_location="cpu").to(device).eval()
    scores, labels = [], []
    loss_sum = 0.0
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            logits = model(batch)
            target = batch[LABELS_KEY].long()
            loss_sum += F.cross_entropy(logits, target, reduction="sum").item()
            scores.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
            labels.append(target.cpu())
    score, label = torch.cat(scores).numpy(), torch.cat(labels).numpy()
    fpr, tpr, thresholds = roc_curve(label, score)
    n_background = int((label == 0).sum())
    row = {
        "checkpoint": str(checkpoint.resolve()), "events": len(label),
        "test_loss": loss_sum / len(label), "test_accuracy": accuracy_score(label, score >= 0.5),
        "test_auc": roc_auc_score(label, score),
    }
    for target_eff in (0.5, 0.7, 0.8):
        index = min(int(np.searchsorted(tpr, target_eff, side="left")), len(tpr) - 1)
        suffix = int(target_eff * 100)
        row[f"background_rejection_at_signal_efficiency_{suffix}"] = 1.0 / max(fpr[index], 1.0 / n_background)
        row[f"threshold_at_signal_efficiency_{suffix}"] = float(thresholds[index])
    return row, (tpr, fpr)


def main():
    args = parse_args()
    device = torch.device(args.device)
    default_output_name = (
        "test_evaluation" if args.pooling == "all"
        else f"test_evaluation_{args.pooling}"
    )
    output = args.output_dir or args.project_dir / default_output_name
    output.mkdir(parents=True, exist_ok=True)
    dm = GroupedTokenParquetClassificationModule(
        prepared_dir=str(args.prepared_dir), n_classes=2,
        batch_size=args.batch_size, num_workers=args.num_workers,
        stream_batch_size=4096, shuffle_buffer_size=8192,
        persistent_workers=False, pin_memory=device.type == "cuda", label_column="label",
    )
    rows, curves = [], {}
    selected_runs = {
        name: spec
        for name, spec in RUNS.items()
        if args.pooling == "all" or spec[0] == args.pooling
    }
    for name, (pooling, label, model_class) in selected_runs.items():
        checkpoint = checkpoint_for(args.project_dir / name)
        print(f"Evaluating {name}: {checkpoint}", flush=True)
        row, curve = evaluate(model_class, checkpoint, dm.test_dataloader(), device)
        display_label = label if args.pooling != "all" else f"{pooling.upper()}: {label}"
        row.update(run=name, label=display_label)
        rows.append(row); curves[name] = curve
    with (output / "test_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output / "test_metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    fig, axis = plt.subplots(figsize=(9, 7))
    for row in rows:
        tpr, fpr = curves[row["run"]]
        axis.plot(tpr, fpr, linewidth=2, label=f"{row['label']} (AUC={row['test_auc']:.3f})")
    axis.set(xlabel="Signal efficiency", ylabel="Background efficiency", xlim=(0, 1), ylim=(1e-5, 1))
    axis.set_yscale("log"); axis.grid(True, which="both", alpha=0.25)
    axis.legend(loc="lower left", fontsize=9)
    fig.tight_layout(); fig.savefig(output / "test_roc.png", dpi=180); fig.savefig(output / "test_roc.pdf"); plt.close(fig)
    for row in rows:
        print(f"{row['run']:34s} AUC={row['test_auc']:.4f} R50={row['background_rejection_at_signal_efficiency_50']:.2f} R70={row['background_rejection_at_signal_efficiency_70']:.2f} R80={row['background_rejection_at_signal_efficiency_80']:.2f}")
    print(f"Wrote evaluation to {output.resolve()}")


if __name__ == "__main__":
    main()
