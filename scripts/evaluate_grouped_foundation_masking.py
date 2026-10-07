#!/usr/bin/env python3
"""Evaluate grouped MLM predictions against frequency baselines."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from heptokens.data.sequence import MASK_KEY, TOKENS_KEY, TYPE_IDS_KEY
from heptokens.data.token_parquet import StreamingTokenParquetDataset
from heptokens.models.foundation_grouped import LitGroupedMaskedSequenceModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--stream-batch-size", type=int, default=1024)
    parser.add_argument("--train-events", type=int, default=200_000)
    parser.add_argument("--validation-events", type=int, default=50_000)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", default="1,5,10,50")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def parquet_files(directory: Path, split: str) -> list[str]:
    paths = sorted(str(path) for path in (directory / split).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No Parquet shards found under {directory / split}")
    return paths


def load_vocabulary(path: str) -> dict:
    metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
    key = b"heptokens_token_vocabulary"
    if key not in metadata:
        raise RuntimeError(f"Missing grouped vocabulary metadata in {path}")
    return json.loads(metadata[key])


def type_names(vocabulary: dict) -> dict[int, str]:
    names = {0: "structural"}
    event = vocabulary.get("event", {})
    if "type_id" in event:
        names[int(event["type_id"])] = "event_context"
    for name, specification in vocabulary.get("objects", {}).items():
        names[int(specification["type_id"])] = name
    return names


def object_code_ranges(vocabulary: dict) -> dict[tuple[int, int], tuple[int, int]]:
    """Return the legal global-token interval for each object type and RVQ level."""
    ranges = {}
    for specification in vocabulary.get("objects", {}).values():
        type_id = int(specification["type_id"])
        for quantizer in specification.get("quantizers", []):
            index = int(quantizer["index"])
            base = int(quantizer["base"])
            size = int(quantizer["size"])
            ranges[(type_id, index)] = (base, base + size)
    return ranges


def make_loader(
    paths: list[str],
    *,
    max_rows: int,
    batch_size: int,
    stream_batch_size: int,
) -> DataLoader:
    dataset = StreamingTokenParquetDataset(
        parquet_files=paths,
        split_start=0.0,
        split_end=1.0,
        max_rows=max_rows,
        stream_batch_size=stream_batch_size,
        shuffle=False,
        loader_num_workers=0,
    )
    return DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True)


def valid_code_mask(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    quantizer_mask = tokens.ne(0)
    special_position = ((tokens <= 3) & quantizer_mask).any(dim=-1)
    return mask.bool() & ~special_position


def estimate_frequency_modes(
    loader: DataLoader,
    *,
    max_events: int,
) -> tuple[dict[tuple[int, int], int], int]:
    counts: dict[tuple[int, int], Counter] = defaultdict(Counter)
    events = 0
    for batch in loader:
        tokens = batch[TOKENS_KEY]
        mask = batch[MASK_KEY]
        types = batch[TYPE_IDS_KEY]
        usable = valid_code_mask(tokens, mask)
        for type_id in torch.unique(types[usable]).tolist():
            positions = usable & types.eq(type_id)
            selected = tokens[positions]
            for quantizer in range(tokens.shape[-1]):
                values = selected[:, quantizer]
                values = values[values.ne(0)]
                if values.numel():
                    unique, frequency = torch.unique(values, return_counts=True)
                    counts[(int(type_id), quantizer)].update(
                        dict(zip(unique.tolist(), frequency.tolist()))
                    )
        events += tokens.shape[0]
        if events >= max_events:
            break
    modes = {
        group: frequencies.most_common(1)[0][0]
        for group, frequencies in counts.items()
        if frequencies
    }
    return modes, min(events, max_events)


def load_model(run_dir: Path, device: torch.device) -> tuple[LitGroupedMaskedSequenceModel, Path]:
    checkpoints = sorted((run_dir / "checkpoints").glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found under {run_dir / 'checkpoints'}")
    checkpoint = checkpoints[-1]
    model = LitGroupedMaskedSequenceModel.load_from_checkpoint(
        checkpoint,
        map_location="cpu",
    )
    model.eval().to(device)
    return model, checkpoint


@torch.inference_mode()
def evaluate(
    model: LitGroupedMaskedSequenceModel,
    loader: DataLoader,
    *,
    modes: dict[tuple[int, int], int],
    code_ranges: dict[tuple[int, int], tuple[int, int]],
    max_events: int,
    mask_prob: float,
    top_ks: list[int],
    seed: int,
    device: torch.device,
) -> tuple[pd.DataFrame, dict]:
    generator = torch.Generator(device=device).manual_seed(seed)
    stats: dict[tuple[int, int], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    events = 0

    for batch in loader:
        original = batch[TOKENS_KEY].to(device, non_blocking=True)
        object_mask = batch[MASK_KEY].to(device, non_blocking=True).bool()
        types = batch[TYPE_IDS_KEY].to(device, non_blocking=True)
        can_mask = valid_code_mask(original, object_mask)
        masked_positions = (
            torch.rand(can_mask.shape, generator=generator, device=device) < mask_prob
        ) & can_mask
        if not masked_positions.any():
            first = can_mask.flatten().nonzero(as_tuple=False)
            if first.numel() == 0:
                continue
            masked_positions.view(-1)[first[0, 0]] = True

        quantizer_mask = original.ne(model.pad_token_id)
        corrupted = original.clone()
        corrupted[masked_positions.unsqueeze(-1) & quantizer_mask] = model.mask_token_id
        labels = original[masked_positions]
        valid_labels = quantizer_mask[masked_positions]
        selected_types = types[masked_positions]
        logits = model.model.masked_logits(
            corrupted,
            object_mask,
            types,
            masked_positions,
        )
        max_k = min(max(top_ks), logits.shape[-1])
        predictions = logits.topk(max_k, dim=-1).indices
        losses = F.cross_entropy(
            logits[valid_labels],
            labels[valid_labels],
            reduction="none",
        )
        loss_matrix = torch.zeros_like(labels, dtype=losses.dtype)
        loss_matrix[valid_labels] = losses

        for quantizer in range(labels.shape[1]):
            present = valid_labels[:, quantizer]
            if not present.any():
                continue
            group_labels = labels[present, quantizer]
            group_types = selected_types[present]
            group_predictions = predictions[present, quantizer]
            group_logits = logits[present, quantizer]
            group_losses = loss_matrix[present, quantizer]
            for type_id in torch.unique(group_types).tolist():
                selected = group_types.eq(type_id)
                truth = group_labels[selected]
                predicted = group_predictions[selected]
                selected_logits = group_logits[selected]
                values = stats[(int(type_id), quantizer)]
                count = int(selected.sum())
                values["count"] += count
                values["loss_sum"] += float(group_losses[selected].sum())
                for top_k in top_ks:
                    values[f"top{top_k}_correct"] += int(
                        predicted[:, :top_k].eq(truth.unsqueeze(1)).any(dim=1).sum()
                    )
                mode = modes.get((int(type_id), quantizer))
                if mode is not None:
                    values["mode_correct"] += int(truth.eq(mode).sum())

                code_range = code_ranges.get((int(type_id), quantizer))
                if code_range is not None:
                    start, stop = code_range
                    unrestricted_top1 = predicted[:, 0]
                    values["range_count"] += count
                    values["global_top1_in_range"] += int(
                        ((unrestricted_top1 >= start) & (unrestricted_top1 < stop)).sum()
                    )
                    local_logits = selected_logits[:, start:stop]
                    local_truth = truth - start
                    constrained_losses = F.cross_entropy(
                        local_logits,
                        local_truth,
                        reduction="none",
                    )
                    values["constrained_loss_sum"] += float(constrained_losses.sum())
                    constrained_k = min(max(top_ks), stop - start)
                    constrained_predictions = (
                        local_logits.topk(constrained_k, dim=-1).indices + start
                    )
                    for top_k in top_ks:
                        effective_k = min(top_k, constrained_k)
                        values[f"constrained_top{top_k}_correct"] += int(
                            constrained_predictions[:, :effective_k]
                            .eq(truth.unsqueeze(1))
                            .any(dim=1)
                            .sum()
                        )

        events += original.shape[0]
        if events >= max_events:
            break
        if events % 10_000 < original.shape[0]:
            print(f"evaluated {events:,}/{max_events:,} validation events", flush=True)

    rows = []
    for (type_id, quantizer), values in sorted(stats.items()):
        count = int(values["count"])
        row = {
            "type_id": type_id,
            "quantizer": quantizer,
            "masked_codes": count,
            "cross_entropy": values["loss_sum"] / count,
            "perplexity": float(np.exp(min(values["loss_sum"] / count, 50))),
            "mode_baseline_acc": values["mode_correct"] / count,
        }
        for top_k in top_ks:
            row[f"model_top{top_k}_acc"] = values[f"top{top_k}_correct"] / count
        range_count = int(values["range_count"])
        row["global_top1_valid_range_rate"] = (
            values["global_top1_in_range"] / range_count if range_count else np.nan
        )
        row["global_top1_invalid_range_rate"] = (
            1.0 - row["global_top1_valid_range_rate"] if range_count else np.nan
        )
        row["constrained_cross_entropy"] = (
            values["constrained_loss_sum"] / range_count if range_count else np.nan
        )
        row["constrained_perplexity"] = (
            float(np.exp(min(row["constrained_cross_entropy"], 50)))
            if range_count
            else np.nan
        )
        for top_k in top_ks:
            row[f"constrained_top{top_k}_acc"] = (
                values[f"constrained_top{top_k}_correct"] / range_count
                if range_count
                else np.nan
            )
        rows.append(row)
    frame = pd.DataFrame(rows)

    totals = frame["masked_codes"].to_numpy(dtype=float)
    summary = {
        "validation_events": min(events, max_events),
        "masked_codes": int(totals.sum()),
        "cross_entropy": float(np.average(frame["cross_entropy"], weights=totals)),
        "mode_baseline_acc": float(np.average(frame["mode_baseline_acc"], weights=totals)),
    }
    for top_k in top_ks:
        summary[f"model_top{top_k}_acc"] = float(
            np.average(frame[f"model_top{top_k}_acc"], weights=totals)
        )
    summary["perplexity"] = float(np.exp(min(summary["cross_entropy"], 50)))
    constrained = frame[frame["global_top1_valid_range_rate"].notna()]
    constrained_totals = constrained["masked_codes"].to_numpy(dtype=float)
    summary["object_masked_codes"] = int(constrained_totals.sum())
    summary["object_global_cross_entropy"] = float(
        np.average(constrained["cross_entropy"], weights=constrained_totals)
    )
    summary["object_global_perplexity"] = float(
        np.exp(min(summary["object_global_cross_entropy"], 50))
    )
    summary["global_top1_valid_range_rate"] = float(
        np.average(
            constrained["global_top1_valid_range_rate"],
            weights=constrained_totals,
        )
    )
    summary["global_top1_invalid_range_rate"] = (
        1.0 - summary["global_top1_valid_range_rate"]
    )
    summary["constrained_cross_entropy"] = float(
        np.average(
            constrained["constrained_cross_entropy"],
            weights=constrained_totals,
        )
    )
    summary["constrained_perplexity"] = float(
        np.exp(min(summary["constrained_cross_entropy"], 50))
    )
    for top_k in top_ks:
        summary[f"object_global_top{top_k}_acc"] = float(
            np.average(
                constrained[f"model_top{top_k}_acc"],
                weights=constrained_totals,
            )
        )
        summary[f"constrained_top{top_k}_acc"] = float(
            np.average(
                constrained[f"constrained_top{top_k}_acc"],
                weights=constrained_totals,
            )
        )
    return frame, summary


def plot_accuracy(frame: pd.DataFrame, output: Path, names: dict[int, str]) -> None:
    frame = frame.copy()
    frame["object"] = frame["type_id"].map(names).fillna(
        frame["type_id"].map(lambda value: f"type_{value}")
    )
    frame = frame[frame["constrained_top1_acc"].notna()]
    objects = list(dict.fromkeys(frame["object"]))
    fig, axes = plt.subplots(
        len(objects), 1, figsize=(11, max(4, 2.7 * len(objects))), squeeze=False,
        constrained_layout=True,
    )
    for axis, object_name in zip(axes[:, 0], objects):
        selected = frame[frame["object"] == object_name]
        x = np.arange(len(selected))
        axis.bar(x - 0.25, selected["mode_baseline_acc"], 0.25, label="mode baseline")
        axis.bar(x, selected["model_top1_acc"], 0.25, label="global top-1")
        axis.bar(
            x + 0.25,
            selected["constrained_top1_acc"],
            0.25,
            label="constrained top-1",
        )
        axis.set_xticks(x, [f"q{value}" for value in selected["quantizer"]])
        axis.set_ylabel("Accuracy")
        axis.set_title(object_name)
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False, ncols=3)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not 0 < args.mask_prob <= 1:
        raise ValueError("--mask-prob must be in (0, 1]")
    top_ks = sorted({int(value) for value in args.top_k.split(",")})
    if not top_ks or top_ks[0] < 1:
        raise ValueError("--top-k values must be positive")

    run_dir = args.run_dir.resolve()
    prepared_dir = args.prepared_dir.resolve()
    output_dir = (args.output_dir or run_dir / "masking_evaluation").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_paths = parquet_files(prepared_dir, "train")
    validation_paths = parquet_files(prepared_dir, "val")
    vocabulary = load_vocabulary(train_paths[0])
    names = type_names(vocabulary)
    code_ranges = object_code_ranges(vocabulary)

    print("estimating frequency baselines", flush=True)
    train_loader = make_loader(
        train_paths,
        max_rows=args.train_events,
        batch_size=max(args.batch_size, 64),
        stream_batch_size=args.stream_batch_size,
    )
    modes, train_events = estimate_frequency_modes(
        train_loader,
        max_events=args.train_events,
    )
    print(f"frequency baseline used {train_events:,} training events", flush=True)

    device = torch.device(args.device)
    model, checkpoint = load_model(run_dir, device)
    validation_loader = make_loader(
        validation_paths,
        max_rows=args.validation_events,
        batch_size=args.batch_size,
        stream_batch_size=args.stream_batch_size,
    )
    frame, summary = evaluate(
        model,
        validation_loader,
        modes=modes,
        code_ranges=code_ranges,
        max_events=args.validation_events,
        mask_prob=args.mask_prob,
        top_ks=top_ks,
        seed=args.seed,
        device=device,
    )
    frame["object"] = frame["type_id"].map(names).fillna("unknown")
    columns = ["object", *[column for column in frame.columns if column != "object"]]
    frame = frame[columns]
    summary.update(
        {
            "checkpoint": str(checkpoint),
            "training_events_for_baseline": train_events,
            "mask_probability": args.mask_prob,
        }
    )

    frame.to_csv(output_dir / "per_type_quantizer_metrics.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_accuracy(frame, output_dir / "global_vs_constrained_top1.png", names)
    print("\nOverall summary:")
    print(json.dumps(summary, indent=2))
    print("\nPer-type/quantizer metrics:")
    print(frame.to_string(index=False))
    print(f"\nWrote evaluation to {output_dir}")


if __name__ == "__main__":
    main()
