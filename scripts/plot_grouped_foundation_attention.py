#!/usr/bin/env python3
"""Plot real self-attention weights from a grouped foundation checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from heptokens.data.sequence import MASK_KEY, TOKENS_KEY, TYPE_IDS_KEY
from heptokens.data.token_parquet import StreamingTokenParquetDataset
from heptokens.models.foundation_grouped import LitGroupedMaskedSequenceModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--event-offset", type=int, default=0)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument(
        "--attention-mode",
        choices=("layer", "rollout", "heads"),
        default="layer",
        help="Use one layer or attention rollout through all layers.",
    )
    parser.add_argument("--max-positions", type=int, default=32)
    parser.add_argument(
        "--aggregate-events",
        type=int,
        default=0,
        help="Average attention by physics category over this many events.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def load_vocabulary(path: str) -> dict:
    metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
    key = b"heptokens_token_vocabulary"
    if key not in metadata:
        raise RuntimeError(f"Missing grouped vocabulary metadata in {path}")
    return json.loads(metadata[key])


def make_type_names(vocabulary: dict) -> dict[int, str]:
    names = {0: "special"}
    event = vocabulary.get("event", {})
    if "type_id" in event:
        names[int(event["type_id"])] = "evt"
    for name, specification in vocabulary.get("objects", {}).items():
        names[int(specification["type_id"])] = name.rstrip("s")
    return names


def event_context_names(vocabulary: dict) -> list[str]:
    pretty = {
        "common/event/mu": "Pileup μ",
        "common/met/pt": "MET pT",
        "common/met/phi": "MET φ",
        "common/met/sumet": "sumET",
    }
    inputs = vocabulary.get("event", {}).get("inputs", [])
    return [pretty.get(value, value.rsplit("/", 1)[-1]) for value in inputs]


def load_event(paths: list[str], offset: int) -> dict[str, torch.Tensor]:
    dataset = StreamingTokenParquetDataset(
        parquet_files=paths,
        split_start=0.0,
        split_end=1.0,
        max_rows=offset + 1,
        stream_batch_size=min(max(offset + 1, 1), 4096),
        shuffle=False,
        loader_num_workers=0,
    )
    for index, event in enumerate(dataset):
        if index == offset:
            return event
    raise IndexError(f"Event offset {offset} exceeds the available rows")


def load_model(run_dir: Path, device: torch.device) -> tuple[LitGroupedMaskedSequenceModel, Path]:
    checkpoints = sorted((run_dir / "checkpoints").glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found under {run_dir / 'checkpoints'}")
    checkpoint = checkpoints[-1]
    model = LitGroupedMaskedSequenceModel.load_from_checkpoint(
        checkpoint, map_location="cpu"
    )
    model.eval().to(device)
    return model, checkpoint


@torch.inference_mode()
def extract_attention(
    model: LitGroupedMaskedSequenceModel,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    type_ids: torch.Tensor,
) -> list[torch.Tensor]:
    """Run the encoder explicitly and retain [batch, head, query, key] weights."""
    backbone = model.backbone
    quantizer_mask = tokens.ne(backbone.pad_token_id)
    embedded = backbone.token_embedding(tokens) * quantizer_mask.unsqueeze(-1)
    x = backbone.object_projection(embedded.flatten(start_dim=2))

    if backbone.type_embedding is not None:
        x = x + backbone.type_embedding(type_ids)
    if backbone.position_embedding is not None:
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        x = x + backbone.position_embedding(positions)
    x = backbone.dropout(x)

    transformer = backbone.transformer
    x = transformer.input_proj(x)
    key_padding_mask = ~mask.bool()
    attention = []
    for layer in transformer.encoder.layers:
        if layer.norm_first:
            normalized = layer.norm1(x)
            attended, weights = layer.self_attn(
                normalized,
                normalized,
                normalized,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            x = x + layer.dropout1(attended)
            normalized = layer.norm2(x)
            feedforward = layer.linear2(
                layer.dropout(layer.activation(layer.linear1(normalized)))
            )
            x = x + layer.dropout2(feedforward)
        else:
            attended, weights = layer.self_attn(
                x,
                x,
                x,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            x = layer.norm1(x + layer.dropout1(attended))
            feedforward = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
            x = layer.norm2(x + layer.dropout2(feedforward))
        attention.append(weights.detach().cpu())
    return attention


def position_labels(type_ids: torch.Tensor, names: dict[int, str]) -> list[str]:
    counters: dict[str, int] = {}
    labels = []
    for type_id in type_ids.tolist():
        name = names.get(int(type_id), f"type{type_id}")
        counters[name] = counters.get(name, 0) + 1
        labels.append(f"{name}{counters[name]}")
    return labels


def position_categories(
    type_ids: torch.Tensor,
    *,
    type_names: dict[int, str],
    event_type_id: int | None,
    context_names: list[str],
) -> list[str | None]:
    event_index = 0
    categories: list[str | None] = []
    for type_id in type_ids.tolist():
        type_id = int(type_id)
        if type_id == 0:
            categories.append(None)
        elif event_type_id is not None and type_id == event_type_id:
            name = (
                context_names[event_index]
                if event_index < len(context_names)
                else f"Event context {event_index + 1}"
            )
            categories.append(name)
            event_index += 1
        else:
            categories.append(type_names.get(type_id, f"type {type_id}"))
    return categories


@torch.inference_mode()
def aggregate_attention(
    model: LitGroupedMaskedSequenceModel,
    paths: list[str],
    *,
    vocabulary: dict,
    max_events: int,
    batch_size: int,
    layer_index: int,
    attention_mode: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    dataset = StreamingTokenParquetDataset(
        parquet_files=paths,
        split_start=0.0,
        split_end=1.0,
        max_rows=max_events,
        stream_batch_size=min(max(max_events, 1), 4096),
        shuffle=False,
        loader_num_workers=0,
    )
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    names = make_type_names(vocabulary)
    context = event_context_names(vocabulary)
    event_type = vocabulary.get("event", {}).get("type_id")
    object_order = [
        names[int(vocabulary["objects"][name]["type_id"])]
        for name in vocabulary.get("object_order", [])
        if name in vocabulary.get("objects", {})
    ]
    categories = list(dict.fromkeys(context + object_order))
    category_index = {name: index for index, name in enumerate(categories)}
    sums = np.zeros((len(categories), len(categories)), dtype=np.float64)
    counts = np.zeros(len(categories), dtype=np.int64)
    enrichment_sums = np.zeros_like(sums)
    enrichment_counts = np.zeros_like(sums, dtype=np.int64)
    events = 0

    for batch in loader:
        tokens = batch[TOKENS_KEY].to(device, non_blocking=True)
        mask = batch[MASK_KEY].bool().to(device, non_blocking=True)
        type_ids = batch[TYPE_IDS_KEY].long().to(device, non_blocking=True)
        layer_weights = extract_attention(model, tokens, mask, type_ids)
        if attention_mode == "rollout":
            sequence_length = tokens.shape[1]
            identity = torch.eye(sequence_length).unsqueeze(0)
            rollout = identity.expand(tokens.shape[0], -1, -1).clone()
            for values in layer_weights:
                values = values.mean(dim=1)
                values = values + identity
                values = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                rollout = torch.bmm(values, rollout)
            weights = rollout.numpy()
        else:
            weights = layer_weights[layer_index].mean(dim=1).numpy()

        for event in range(tokens.shape[0]):
            valid = mask[event].cpu().numpy().astype(bool)
            event_categories = position_categories(
                type_ids[event].cpu(),
                type_names=names,
                event_type_id=int(event_type) if event_type is not None else None,
                context_names=context,
            )
            retained = np.array(
                [valid[i] and category is not None for i, category in enumerate(event_categories)]
            )
            if not retained.any():
                continue
            matrix = weights[event]
            for query_name in categories:
                query_positions = np.array(
                    [retained[i] and value == query_name for i, value in enumerate(event_categories)]
                )
                if not query_positions.any():
                    continue
                row = matrix[query_positions].mean(axis=0)
                grouped = np.array(
                    [
                        row[
                            np.array(
                                [
                                    retained[i] and value == key_name
                                    for i, value in enumerate(event_categories)
                                ]
                            )
                        ].sum()
                        for key_name in categories
                    ]
                )
                total = grouped.sum()
                if total > 0:
                    observed = grouped / total
                    query_index = category_index[query_name]
                    sums[query_index] += observed
                    counts[query_index] += 1
                    available = np.array(
                        [
                            sum(
                                retained[i] and value == key_name
                                for i, value in enumerate(event_categories)
                            )
                            for key_name in categories
                        ],
                        dtype=np.float64,
                    )
                    expected = available / available.sum()
                    present = expected > 0
                    enrichment_sums[query_index, present] += (
                        observed[present] / expected[present]
                    )
                    enrichment_counts[query_index, present] += 1
            events += 1
        if events >= max_events:
            break
        if events % 1000 < tokens.shape[0]:
            print(f"aggregated {events:,}/{max_events:,} events", flush=True)

    populated = counts > 0
    averaged = sums[populated] / counts[populated, None]
    retained_columns = averaged.sum(axis=0) > 0
    averaged = averaged[:, retained_columns]
    enrichment = np.divide(
        enrichment_sums,
        enrichment_counts,
        out=np.full_like(enrichment_sums, np.nan),
        where=enrichment_counts > 0,
    )
    enrichment = enrichment[populated][:, retained_columns]
    row_labels = [name for name, keep in zip(categories, populated) if keep]
    column_labels = [name for name, keep in zip(categories, retained_columns) if keep]
    if row_labels != column_labels:
        raise RuntimeError("Attention category rows and columns are inconsistent")
    return averaged, enrichment, row_labels, events


@torch.inference_mode()
def aggregate_head_enrichment(
    model: LitGroupedMaskedSequenceModel,
    paths: list[str],
    *,
    vocabulary: dict,
    max_events: int,
    batch_size: int,
    layer_index: int,
    device: torch.device,
) -> tuple[np.ndarray, list[str], int]:
    dataset = StreamingTokenParquetDataset(
        parquet_files=paths,
        split_start=0.0,
        split_end=1.0,
        max_rows=max_events,
        stream_batch_size=min(max(max_events, 1), 4096),
        shuffle=False,
        loader_num_workers=0,
    )
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    names = make_type_names(vocabulary)
    context = event_context_names(vocabulary)
    event_type = vocabulary.get("event", {}).get("type_id")
    object_order = [
        names[int(vocabulary["objects"][name]["type_id"])]
        for name in vocabulary.get("object_order", [])
        if name in vocabulary.get("objects", {})
    ]
    categories = list(dict.fromkeys(context + object_order))
    category_index = {name: index for index, name in enumerate(categories)}
    num_heads = model.backbone.transformer.encoder.layers[layer_index].self_attn.num_heads
    sums = np.zeros(
        (num_heads, len(categories), len(categories)), dtype=np.float64
    )
    counts = np.zeros_like(sums, dtype=np.int64)
    events = 0

    for batch in loader:
        tokens = batch[TOKENS_KEY].to(device, non_blocking=True)
        mask = batch[MASK_KEY].bool().to(device, non_blocking=True)
        type_ids = batch[TYPE_IDS_KEY].long().to(device, non_blocking=True)
        weights = extract_attention(model, tokens, mask, type_ids)[layer_index].numpy()

        for event in range(tokens.shape[0]):
            valid = mask[event].cpu().numpy().astype(bool)
            event_categories = position_categories(
                type_ids[event].cpu(),
                type_names=names,
                event_type_id=int(event_type) if event_type is not None else None,
                context_names=context,
            )
            category_ids = np.array(
                [category_index.get(value, -1) if valid[i] else -1 for i, value in enumerate(event_categories)]
            )
            retained = category_ids >= 0
            if not retained.any():
                continue
            available = np.bincount(
                category_ids[retained], minlength=len(categories)
            ).astype(np.float64)
            expected = available / available.sum()
            present_keys = expected > 0

            for query_index in np.unique(category_ids[retained]):
                query_positions = category_ids == query_index
                for head in range(num_heads):
                    row = weights[event, head, query_positions].mean(axis=0)
                    grouped = np.bincount(
                        category_ids[retained],
                        weights=row[retained],
                        minlength=len(categories),
                    )
                    total = grouped.sum()
                    if total <= 0:
                        continue
                    observed = grouped / total
                    sums[head, query_index, present_keys] += (
                        observed[present_keys] / expected[present_keys]
                    )
                    counts[head, query_index, present_keys] += 1
            events += 1
        if events >= max_events:
            break
        if events % 1000 < tokens.shape[0]:
            print(f"aggregated heads for {events:,}/{max_events:,} events", flush=True)

    enrichment = np.divide(
        sums,
        counts,
        out=np.full_like(sums, np.nan),
        where=counts > 0,
    )
    populated = np.isfinite(enrichment).any(axis=(0, 2))
    columns = np.isfinite(enrichment).any(axis=(0, 1))
    enrichment = enrichment[:, populated][:, :, columns]
    row_labels = [name for name, keep in zip(categories, populated) if keep]
    column_labels = [name for name, keep in zip(categories, columns) if keep]
    if row_labels != column_labels:
        raise RuntimeError("Per-head attention categories are inconsistent")
    return enrichment, row_labels, events


def plot_aggregate(
    matrix: np.ndarray,
    labels: list[str],
    *,
    layer_index: int,
    attention_mode: str,
    events: int,
    output: Path,
) -> None:
    size = max(7.5, 0.62 * len(labels) + 3.0)
    fig, axis = plt.subplots(figsize=(size, size), constrained_layout=True)
    image = axis.imshow(matrix, cmap="magma", vmin=0.0, aspect="equal")
    axis.set_xticks(
        np.arange(len(labels)), labels, rotation=45, ha="right", fontsize=16
    )
    axis.set_yticks(np.arange(len(labels)), labels, fontsize=16)
    axis.set_xlabel("Category receiving attention", fontsize=18)
    axis.set_ylabel("Query category", fontsize=18)
    scope = (
        "Attention rollout through all Transformer blocks"
        if attention_mode == "rollout"
        else f"Transformer block {layer_index + 1}"
    )
    axis.set_title(
        "Mean information flow by token category\n"
        f"{scope}; {events:,} validation events",
        fontsize=20,
        pad=12,
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            color = "black" if value > 0.20 else "white"
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=13,
            )
    colorbar = fig.colorbar(image, ax=axis, shrink=0.82)
    colorbar.set_label("Mean attention fraction", fontsize=16)
    colorbar.ax.tick_params(labelsize=14)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_enrichment(
    matrix: np.ndarray,
    labels: list[str],
    *,
    layer_index: int,
    attention_mode: str,
    events: int,
    output: Path,
) -> None:
    finite = matrix[np.isfinite(matrix)]
    upper = max(2.0, float(np.percentile(finite, 98))) if finite.size else 2.0
    size = max(7.5, 0.62 * len(labels) + 3.0)
    fig, axis = plt.subplots(figsize=(size, size), constrained_layout=True)
    image = axis.imshow(
        matrix,
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=upper),
        aspect="equal",
    )
    axis.set_xticks(
        np.arange(len(labels)), labels, rotation=45, ha="right", fontsize=16
    )
    axis.set_yticks(np.arange(len(labels)), labels, fontsize=16)
    axis.set_xlabel("Category receiving attention", fontsize=18)
    axis.set_ylabel("Query category", fontsize=18)
    scope = (
        "attention rollout through all Transformer blocks"
        if attention_mode == "rollout"
        else f"Transformer block {layer_index + 1}"
    )
    axis.set_title(
        "Information-flow enrichment over availability\n"
        f"{scope}; {events:,} validation events",
        fontsize=20,
        pad=12,
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if not np.isfinite(value):
                label = "–"
                color = "black"
            else:
                label = f"{value:.1f}×"
                color = "white" if value < 0.45 or value > 1.65 else "black"
            axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                color=color,
                fontsize=13,
            )
    colorbar = fig.colorbar(image, ax=axis, shrink=0.82)
    colorbar.set_label("Attention / availability baseline", fontsize=16)
    colorbar.ax.tick_params(labelsize=14)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_head_grid(
    matrices: np.ndarray,
    labels: list[str],
    *,
    layer_index: int,
    events: int,
    output: Path,
) -> None:
    finite = matrices[np.isfinite(matrices)]
    upper = max(2.0, float(np.percentile(finite, 98))) if finite.size else 2.0
    columns = 4
    rows = int(np.ceil(matrices.shape[0] / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.4 * columns, 5.2 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    for head, axis in enumerate(axes.flat):
        if head >= matrices.shape[0]:
            axis.set_visible(False)
            continue
        image = axis.imshow(
            matrices[head],
            cmap="coolwarm",
            norm=TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=upper),
            aspect="equal",
        )
        axis.set_title(f"Head {head + 1}", fontsize=17)
        axis.set_xticks(
            np.arange(len(labels)), labels, rotation=55, ha="right", fontsize=10
        )
        axis.set_yticks(np.arange(len(labels)), labels, fontsize=10)
    fig.suptitle(
        f"Per-head attention enrichment, Transformer block {layer_index + 1}\n"
        f"{events:,} validation events; 1× = availability baseline",
        fontsize=22,
    )
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, shrink=0.78, location="right")
        colorbar.set_label("Attention / availability baseline", fontsize=16)
        colorbar.ax.tick_params(labelsize=13)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_presentation_enrichment(
    matrix: np.ndarray,
    labels: list[str],
    *,
    context_categories: int,
    events: int,
    output: Path,
) -> None:
    query_labels = labels[context_categories:]
    compact = matrix[context_categories:].copy()
    for row, query_name in enumerate(query_labels):
        if query_name in labels:
            compact[row, labels.index(query_name)] = np.nan

    finite = compact[np.isfinite(compact)]
    upper = max(2.0, float(np.percentile(finite, 98))) if finite.size else 2.0
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#eeeeee")
    fig, axis = plt.subplots(figsize=(13.5, 7.8), constrained_layout=True)
    image = axis.imshow(
        compact,
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=upper),
        aspect="auto",
    )
    axis.set_xticks(
        np.arange(len(labels)), labels, rotation=42, ha="right", fontsize=15
    )
    axis.set_yticks(np.arange(len(query_labels)), query_labels, fontsize=16)
    axis.set_xlabel("Information received from", fontsize=18)
    axis.set_ylabel("Object representation", fontsize=18)
    axis.set_title("Learned information flow across the event", fontsize=22, pad=12)
    for row in range(compact.shape[0]):
        for column in range(compact.shape[1]):
            value = compact[row, column]
            if not np.isfinite(value):
                label = ""
                color = "black"
            else:
                label = f"{value:.1f}×"
                color = "white" if value < 0.45 or value > 1.65 else "black"
            axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                color=color,
                fontsize=13,
            )
    colorbar = fig.colorbar(image, ax=axis, shrink=0.88, pad=0.02)
    colorbar.set_label("Enrichment", fontsize=16)
    colorbar.ax.tick_params(labelsize=13)
    fig.text(
        0.5,
        0.005,
        (
            f"Attention rollout over all Transformer blocks, averaged over {events:,} "
            "validation events and normalized to category availability. "
            "Same-type diagonal hidden."
        ),
        ha="center",
        fontsize=12,
    )
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.event_offset < 0 or args.max_positions < 1 or args.aggregate_events < 0:
        raise ValueError("Event offset must be non-negative and max positions positive")

    prepared_dir = args.prepared_dir.resolve()
    paths = sorted(str(path) for path in (prepared_dir / args.split).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No Parquet shards under {prepared_dir / args.split}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(args.run_dir.resolve(), device)
    vocabulary = load_vocabulary(paths[0])
    output_dir = (args.output_dir or args.run_dir / "attention_analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_index = args.layer % len(model.backbone.transformer.encoder.layers)

    if args.aggregate_events:
        if args.attention_mode == "heads":
            enrichment, labels, events = aggregate_head_enrichment(
                model,
                paths,
                vocabulary=vocabulary,
                max_events=args.aggregate_events,
                batch_size=args.batch_size,
                layer_index=layer_index,
                device=device,
            )
            output = output_dir / f"attention_heads_{args.split}_layer{layer_index}.png"
            plot_head_grid(
                enrichment,
                labels,
                layer_index=layer_index,
                events=events,
                output=output,
            )
            metadata = {
                "checkpoint": str(checkpoint),
                "prepared_dir": str(prepared_dir),
                "split": args.split,
                "events": events,
                "layer": layer_index,
                "attention_mode": "heads",
                "heads": int(enrichment.shape[0]),
                "categories": labels,
                "output": str(output),
            }
            (output_dir / f"attention_heads_layer{layer_index}_metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(metadata, indent=2))
            return

        matrix, enrichment, labels, events = aggregate_attention(
            model,
            paths,
            vocabulary=vocabulary,
            max_events=args.aggregate_events,
            batch_size=args.batch_size,
            layer_index=layer_index,
            attention_mode=args.attention_mode,
            device=device,
        )
        suffix = "rollout" if args.attention_mode == "rollout" else f"layer{layer_index}"
        output = output_dir / f"attention_by_category_{args.split}_{suffix}.png"
        plot_aggregate(
            matrix,
            labels,
            layer_index=layer_index,
            attention_mode=args.attention_mode,
            events=events,
            output=output,
        )
        enrichment_output = (
            output_dir / f"attention_enrichment_{args.split}_{suffix}.png"
        )
        plot_enrichment(
            enrichment,
            labels,
            layer_index=layer_index,
            attention_mode=args.attention_mode,
            events=events,
            output=enrichment_output,
        )
        presentation_output = None
        if args.attention_mode == "rollout":
            presentation_output = (
                output_dir / f"attention_enrichment_presentation_{args.split}_rollout.png"
            )
            plot_presentation_enrichment(
                enrichment,
                labels,
                context_categories=len(event_context_names(vocabulary)),
                events=events,
                output=presentation_output,
            )
        metadata = {
            "checkpoint": str(checkpoint),
            "prepared_dir": str(prepared_dir),
            "split": args.split,
            "events": events,
            "layer": layer_index,
            "attention_mode": args.attention_mode,
            "heads_averaged": int(
                model.backbone.transformer.encoder.layers[layer_index].self_attn.num_heads
            ),
            "special_positions_excluded": True,
            "categories": labels,
            "output": str(output),
            "enrichment_output": str(enrichment_output),
            "presentation_output": (
                str(presentation_output) if presentation_output is not None else None
            ),
            "enrichment_definition": (
                "observed attention fraction divided by the fraction of retained "
                "positions in the receiving category; averaged only over events "
                "where that receiving category is present"
            ),
        }
        (output_dir / "attention_by_category_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(metadata, indent=2))
        return

    event = load_event(paths, args.event_offset)
    tokens = event[TOKENS_KEY].unsqueeze(0).to(device)
    mask = event[MASK_KEY].unsqueeze(0).bool().to(device)
    type_ids = event[TYPE_IDS_KEY].unsqueeze(0).long().to(device)
    attention = extract_attention(model, tokens, mask, type_ids)

    layer_index = args.layer % len(attention)
    valid = mask[0].nonzero(as_tuple=False).flatten()
    valid = valid[: args.max_positions]
    if not valid.numel():
        raise RuntimeError("Selected event contains no valid positions")
    matrix = attention[layer_index][0].mean(dim=0)
    matrix = matrix[valid.cpu()][:, valid.cpu()].numpy()
    names = make_type_names(vocabulary)
    labels = position_labels(type_ids[0, valid].cpu(), names)

    output = output_dir / f"attention_{args.split}_event{args.event_offset}_layer{layer_index}.png"

    size = max(7.0, min(12.0, 0.34 * len(labels) + 3.0))
    fig, axis = plt.subplots(figsize=(size, size), constrained_layout=True)
    image = axis.imshow(matrix, cmap="magma", vmin=0.0, aspect="equal")
    axis.set_xticks(np.arange(len(labels)), labels, rotation=60, ha="right")
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.set_xlabel("Attended-to position")
    axis.set_ylabel("Query position")
    axis.set_title(f"Grouped foundation model self-attention, layer {layer_index + 1}\n(mean over heads; one {args.split} event)")
    fig.colorbar(image, ax=axis, label="Attention weight", shrink=0.82)
    fig.savefig(output, dpi=220)
    plt.close(fig)

    metadata = {
        "checkpoint": str(checkpoint),
        "prepared_dir": str(prepared_dir),
        "split": args.split,
        "event_offset": args.event_offset,
        "layer": layer_index,
        "heads_averaged": int(attention[layer_index].shape[1]),
        "positions_shown": len(labels),
        "labels": labels,
        "output": str(output),
    }
    (output_dir / "attention_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
