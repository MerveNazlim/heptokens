#!/usr/bin/env python3
"""Evaluate matched HZZ classifiers initialized from the two decoder runs."""

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

from heptokens.data.sequence import LABELS_KEY, TOKENS_KEY
from heptokens.data.token_parquet import GroupedTokenParquetClassificationModule
from heptokens.models.foundation_grouped_cls_classifier import LitGroupedCLSClassifier


RUNS = {
    "autoregressive_frozen_cls_mlp": "Autoregressive pretrained, frozen",
    "parallel_frozen_cls_mlp": "Parallel pretrained, frozen",
    "autoregressive_finetuned_cls_mlp": "Autoregressive pretrained, fine-tuned",
    "parallel_finetuned_cls_mlp": "Parallel pretrained, fine-tuned",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument(
        "--quantizer-cutoffs",
        type=int,
        nargs="+",
        default=[7],
        help=(
            "Largest residual quantizer retained at evaluation time. "
            "For example, 0 1 3 7 evaluates Q0, Q0-Q1, Q0-Q3, and Q0-Q7."
        ),
    )
    return parser.parse_args()


def checkpoint_for(directory: Path) -> Path:
    for name in ("best.ckpt", "last.ckpt"):
        path = directory / "checkpoints" / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No checkpoint under {directory}")


def evaluate(model, loader, device, *, max_quantizer: int):
    scores, labels = [], []
    loss_sum = 0.0
    with torch.inference_mode():
        for batch in loader:
            batch = {
                key: value.to(device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }
            tokens = batch[TOKENS_KEY]
            if max_quantizer + 1 < tokens.shape[-1]:
                tokens = tokens.clone()
                tokens[..., max_quantizer + 1 :] = model.model.pad_token_id
                batch[TOKENS_KEY] = tokens
            logits = model(batch)
            target = batch[LABELS_KEY].long()
            loss_sum += F.cross_entropy(logits, target, reduction="sum").item()
            scores.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
            labels.append(target.cpu())
    score = torch.cat(scores).numpy()
    label = torch.cat(labels).numpy()
    fpr, tpr, thresholds = roc_curve(label, score)
    n_background = int((label == 0).sum())
    row = {
        "max_quantizer": max_quantizer,
        "quantizers_retained": max_quantizer + 1,
        "events": len(label),
        "test_loss": loss_sum / len(label),
        "test_accuracy": accuracy_score(label, score >= 0.5),
        "test_auc": roc_auc_score(label, score),
    }
    for target_efficiency in (0.5, 0.7, 0.8):
        index = min(
            int(np.searchsorted(tpr, target_efficiency, side="left")), len(tpr) - 1
        )
        suffix = int(target_efficiency * 100)
        row[f"background_rejection_at_signal_efficiency_{suffix}"] = 1.0 / max(
            fpr[index], 1.0 / n_background
        )
        row[f"threshold_at_signal_efficiency_{suffix}"] = float(thresholds[index])
    return row, (tpr, fpr)


def main() -> None:
    args = parse_args()
    cutoffs = sorted(set(args.quantizer_cutoffs))
    invalid = [cutoff for cutoff in cutoffs if cutoff < 0 or cutoff > 7]
    if invalid:
        raise ValueError(f"Quantizer cutoffs must be between 0 and 7, got {invalid}")
    device = torch.device(args.device)
    output = args.output_dir or (
        args.project_dir
        / ("quantizer_input_ablation" if cutoffs != [7] else "test_evaluation")
    )
    output.mkdir(parents=True, exist_ok=True)
    datamodule = GroupedTokenParquetClassificationModule(
        prepared_dir=str(args.prepared_dir),
        n_classes=2,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        stream_batch_size=4096,
        shuffle_buffer_size=8192,
        persistent_workers=False,
        pin_memory=device.type == "cuda",
        label_column="label",
    )

    rows, curves = [], {}
    for run, label in RUNS.items():
        checkpoint = checkpoint_for(args.project_dir / run)
        print(f"Loading {run}: {checkpoint}", flush=True)
        model = LitGroupedCLSClassifier.load_from_checkpoint(
            checkpoint, map_location="cpu"
        ).to(device).eval()
        for cutoff in cutoffs:
            print(f"  evaluating Q0-Q{cutoff}", flush=True)
            row, curve = evaluate(
                model,
                datamodule.test_dataloader(),
                device,
                max_quantizer=cutoff,
            )
            row.update(
                checkpoint=str(checkpoint.resolve()),
                run=run,
                label=label,
            )
            rows.append(row)
            curves[(run, cutoff)] = curve
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output_stem = "quantizer_cutoff_metrics" if cutoffs != [7] else "test_metrics"
    with (output / f"{output_stem}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / f"{output_stem}.json").write_text(json.dumps(rows, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), squeeze=False)
    colors = {
        0: "#4c78a8",
        1: "#72b7b2",
        3: "#f58518",
        7: "#e45756",
    }
    for axis, (run, label) in zip(axes.flat, RUNS.items()):
        run_rows = [row for row in rows if row["run"] == run]
        for row in run_rows:
            cutoff = row["max_quantizer"]
            tpr, fpr = curves[(run, cutoff)]
            axis.plot(
                tpr,
                fpr,
                linewidth=2,
                color=colors.get(cutoff),
                label=f"Q0-Q{cutoff} (AUC={row['test_auc']:.3f})",
            )
        axis.set(
            title=label,
            xlabel="Signal efficiency",
            ylabel="Background efficiency",
            xlim=(0, 1),
            ylim=(1e-5, 1),
        )
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    roc_stem = "quantizer_cutoff_roc" if cutoffs != [7] else "test_roc"
    fig.savefig(output / f"{roc_stem}.png", dpi=180)
    fig.savefig(output / f"{roc_stem}.pdf")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 6.5))
    run_colors = {
        "autoregressive_frozen_cls_mlp": "#4c78a8",
        "parallel_frozen_cls_mlp": "#f58518",
        "autoregressive_finetuned_cls_mlp": "#e45756",
        "parallel_finetuned_cls_mlp": "#9d4f8c",
    }
    for run, label in RUNS.items():
        run_rows = sorted(
            (row for row in rows if row["run"] == run),
            key=lambda row: row["max_quantizer"],
        )
        axis.plot(
            [row["max_quantizer"] for row in run_rows],
            [row["test_auc"] for row in run_rows],
            marker="o",
            linewidth=2,
            color=run_colors[run],
            label=label,
        )
    axis.set_xticks(cutoffs, [f"Q0-Q{cutoff}" for cutoff in cutoffs])
    axis.set_xlabel("Quantizer levels retained at inference")
    axis.set_ylabel("Test AUC")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "quantizer_cutoff_auc.png", dpi=180)
    fig.savefig(output / "quantizer_cutoff_auc.pdf")
    plt.close(fig)

    for row in rows:
        print(
            f"{row['run']:38s} Q0-Q{row['max_quantizer']} "
            f"AUC={row['test_auc']:.4f} "
            f"R50={row['background_rejection_at_signal_efficiency_50']:.2f} "
            f"R70={row['background_rejection_at_signal_efficiency_70']:.2f} "
            f"R80={row['background_rejection_at_signal_efficiency_80']:.2f}",
            flush=True,
        )
    print(f"Wrote evaluation to {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
