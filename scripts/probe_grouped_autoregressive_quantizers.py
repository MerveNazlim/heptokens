#!/usr/bin/env python3
"""Compare parallel and teacher-forced RVQ prediction on a frozen backbone."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def parquet_files(directory: Path, split: str) -> list[str]:
    paths = sorted(str(path) for path in (directory / split).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No Parquet shards under {directory / split}")
    return paths


def load_vocabulary(path: str) -> dict:
    metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
    key = b"heptokens_token_vocabulary"
    if key not in metadata:
        raise RuntimeError(f"Missing vocabulary metadata in {path}")
    return json.loads(metadata[key])


def codebook_specs(vocabulary: dict) -> dict[tuple[int, int], dict]:
    specs = {}
    for object_name, object_spec in vocabulary.get("objects", {}).items():
        type_id = int(object_spec["type_id"])
        for quantizer in object_spec.get("quantizers", []):
            specs[(type_id, int(quantizer["index"]))] = {
                "object": object_name,
                "base": int(quantizer["base"]),
                "size": int(quantizer["size"]),
            }
    return specs


def make_loader(
    paths: list[str],
    *,
    max_rows: int,
    batch_size: int,
    stream_batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = StreamingTokenParquetDataset(
        parquet_files=paths,
        split_start=0.0,
        split_end=1.0,
        max_rows=max_rows,
        stream_batch_size=stream_batch_size,
        shuffle=shuffle,
        shuffle_buffer_size=8192 if shuffle else 0,
        seed=seed,
        loader_num_workers=0,
    )
    return DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True)


def load_backbone(
    run_dir: Path, device: torch.device
) -> tuple[LitGroupedMaskedSequenceModel, Path]:
    checkpoints = sorted((run_dir / "checkpoints").glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint under {run_dir / 'checkpoints'}")
    checkpoint = checkpoints[-1]
    model = LitGroupedMaskedSequenceModel.load_from_checkpoint(
        checkpoint, map_location="cpu"
    )
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


class QuantizerProbes(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        vocab_size: int,
        embedding_dim: int,
        max_quantizers: int,
        specs: dict[tuple[int, int], dict],
        token_embedding: torch.Tensor,
    ) -> None:
        super().__init__()
        self.max_quantizers = max_quantizers
        self.specs = specs
        self.previous_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        with torch.no_grad():
            self.previous_embedding.weight.copy_(token_embedding)
        self.previous_projection = nn.Linear(embedding_dim, hidden_dim)
        self.quantizer_embedding = nn.Embedding(max_quantizers, hidden_dim)
        self.autoregressive_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.parallel_heads = nn.ModuleDict()
        self.autoregressive_heads = nn.ModuleDict()
        for (type_id, quantizer), specification in specs.items():
            key = self.key(type_id, quantizer)
            size = int(specification["size"])
            self.parallel_heads[key] = nn.Linear(hidden_dim, size)
            self.autoregressive_heads[key] = nn.Linear(hidden_dim, size)

    @staticmethod
    def key(type_id: int, quantizer: int) -> str:
        return f"type{type_id}_q{quantizer}"

    def states(
        self, context: torch.Tensor, true_codes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        quantizer_ids = torch.arange(
            self.max_quantizers, device=context.device
        ).unsqueeze(0)
        q_embedding = self.quantizer_embedding(quantizer_ids)
        parallel = context.unsqueeze(1) + q_embedding

        previous = torch.zeros_like(true_codes)
        previous[:, 1:] = true_codes[:, :-1]
        previous_embedded = self.previous_projection(self.previous_embedding(previous))
        decoder_input = previous_embedded + q_embedding
        autoregressive, _ = self.autoregressive_gru(
            decoder_input,
            context.unsqueeze(0),
        )
        autoregressive = autoregressive + q_embedding
        return parallel, autoregressive

    def logits(
        self,
        states: torch.Tensor,
        *,
        type_id: int,
        quantizer: int,
        autoregressive: bool,
    ) -> torch.Tensor:
        heads = self.autoregressive_heads if autoregressive else self.parallel_heads
        return heads[self.key(type_id, quantizer)](states)


def mask_object_positions(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    type_ids: torch.Tensor,
    *,
    object_type_ids: torch.Tensor,
    mask_prob: float,
    generator: torch.Generator,
) -> torch.Tensor:
    is_object = (type_ids.unsqueeze(-1) == object_type_ids).any(dim=-1)
    can_mask = mask.bool() & is_object
    selected = (
        torch.rand(can_mask.shape, generator=generator, device=tokens.device) < mask_prob
    ) & can_mask
    if not selected.any() and can_mask.any():
        selected.view(-1)[can_mask.view(-1).nonzero(as_tuple=False)[0, 0]] = True
    return selected


def probe_loss_and_stats(
    probes: QuantizerProbes,
    parallel_states: torch.Tensor,
    autoregressive_states: torch.Tensor,
    labels: torch.Tensor,
    type_ids: torch.Tensor,
    valid_codes: torch.Tensor,
    specs: dict[tuple[int, int], dict],
    stats: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    parallel_losses = []
    autoregressive_losses = []
    for quantizer in range(labels.shape[1]):
        present = valid_codes[:, quantizer]
        if not present.any():
            continue
        for type_id in torch.unique(type_ids[present]).tolist():
            specification = specs.get((int(type_id), quantizer))
            if specification is None:
                continue
            selected = present & type_ids.eq(type_id)
            truth = labels[selected, quantizer] - int(specification["base"])
            parallel_logits = probes.logits(
                parallel_states[selected, quantizer],
                type_id=int(type_id),
                quantizer=quantizer,
                autoregressive=False,
            )
            autoregressive_logits = probes.logits(
                autoregressive_states[selected, quantizer],
                type_id=int(type_id),
                quantizer=quantizer,
                autoregressive=True,
            )
            parallel_loss = F.cross_entropy(parallel_logits, truth)
            autoregressive_loss = F.cross_entropy(autoregressive_logits, truth)
            parallel_losses.append(parallel_loss)
            autoregressive_losses.append(autoregressive_loss)
            if stats is not None:
                key = (int(type_id), quantizer)
                count = int(selected.sum())
                values = stats[key]
                values["count"] += count
                values["parallel_loss"] += float(parallel_loss) * count
                values["autoregressive_loss"] += float(autoregressive_loss) * count
                values["parallel_correct"] += int(
                    parallel_logits.argmax(dim=-1).eq(truth).sum()
                )
                values["autoregressive_correct"] += int(
                    autoregressive_logits.argmax(dim=-1).eq(truth).sum()
                )
    if not parallel_losses:
        zero = parallel_states.sum() * 0.0
        return zero, zero
    return torch.stack(parallel_losses).mean(), torch.stack(autoregressive_losses).mean()


def prepare_context(
    backbone: LitGroupedMaskedSequenceModel,
    batch: dict,
    *,
    object_type_ids: torch.Tensor,
    mask_prob: float,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    tokens = batch[TOKENS_KEY].to(device, non_blocking=True)
    mask = batch[MASK_KEY].bool().to(device, non_blocking=True)
    type_ids = batch[TYPE_IDS_KEY].long().to(device, non_blocking=True)
    selected = mask_object_positions(
        tokens,
        mask,
        type_ids,
        object_type_ids=object_type_ids,
        mask_prob=mask_prob,
        generator=generator,
    )
    if not selected.any():
        return None
    valid_codes = tokens.ne(backbone.pad_token_id)[selected]
    labels = tokens[selected]
    selected_types = type_ids[selected]
    corrupted = tokens.clone()
    corrupted[selected.unsqueeze(-1) & tokens.ne(backbone.pad_token_id)] = (
        backbone.mask_token_id
    )
    with torch.no_grad():
        hidden = backbone.backbone(corrupted, mask, type_ids)[selected]
    return hidden, labels, selected_types, valid_codes


def evaluate(
    backbone: LitGroupedMaskedSequenceModel,
    probes: QuantizerProbes,
    loader: DataLoader,
    *,
    specs: dict[tuple[int, int], dict],
    object_type_ids: torch.Tensor,
    mask_prob: float,
    seed: int,
    device: torch.device,
) -> dict:
    probes.eval()
    stats = defaultdict(lambda: defaultdict(float))
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.inference_mode():
        for batch in loader:
            prepared = prepare_context(
                backbone,
                batch,
                object_type_ids=object_type_ids,
                mask_prob=mask_prob,
                generator=generator,
                device=device,
            )
            if prepared is None:
                continue
            context, labels, selected_types, valid_codes = prepared
            parallel, autoregressive = probes.states(context, labels)
            probe_loss_and_stats(
                probes,
                parallel,
                autoregressive,
                labels,
                selected_types,
                valid_codes,
                specs,
                stats,
            )
    return stats


def stats_frame(stats: dict, specs: dict[tuple[int, int], dict]) -> pd.DataFrame:
    rows = []
    for key, values in sorted(stats.items()):
        count = int(values["count"])
        specification = specs[key]
        rows.append(
            {
                "object": specification["object"],
                "type_id": key[0],
                "quantizer": key[1],
                "masked_codes": count,
                "parallel_cross_entropy": values["parallel_loss"] / count,
                "teacher_forced_cross_entropy": values["autoregressive_loss"] / count,
                "parallel_top1_acc": values["parallel_correct"] / count,
                "teacher_forced_top1_acc": values["autoregressive_correct"] / count,
            }
        )
    frame = pd.DataFrame(rows)
    frame["top1_absolute_gain"] = (
        frame["teacher_forced_top1_acc"] - frame["parallel_top1_acc"]
    )
    return frame


def plot_comparison(frame: pd.DataFrame, output: Path) -> None:
    objects = list(dict.fromkeys(frame["object"]))
    fig, axes = plt.subplots(
        len(objects),
        1,
        figsize=(11, max(4, 2.7 * len(objects))),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, object_name in zip(axes[:, 0], objects):
        selected = frame[frame["object"] == object_name]
        x = np.arange(len(selected))
        axis.bar(x - 0.18, selected["parallel_top1_acc"], 0.36, label="parallel probe")
        axis.bar(
            x + 0.18,
            selected["teacher_forced_top1_acc"],
            0.36,
            label="teacher-forced autoregressive probe",
        )
        axis.set_xticks(x, [f"q{value}" for value in selected["quantizer"]])
        axis.set_ylabel("Top-1 accuracy")
        axis.set_title(object_name)
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False, ncols=2)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    run_dir = args.run_dir.resolve()
    prepared_dir = args.prepared_dir.resolve()
    output_dir = (
        args.output_dir or run_dir / "teacher_forced_quantizer_probe"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_paths = parquet_files(prepared_dir, "train")
    val_paths = parquet_files(prepared_dir, "val")
    vocabulary = load_vocabulary(train_paths[0])
    specs = codebook_specs(vocabulary)
    backbone, checkpoint = load_backbone(run_dir, device)
    hidden_dim = int(backbone.hparams.hidden_dim)
    probes = QuantizerProbes(
        hidden_dim=hidden_dim,
        vocab_size=int(backbone.hparams.vocab_size),
        embedding_dim=backbone.backbone.token_embedding.embedding_dim,
        max_quantizers=int(backbone.hparams.max_quantizers),
        specs=specs,
        token_embedding=backbone.backbone.token_embedding.weight.detach().cpu(),
    ).to(device)
    optimizer = torch.optim.AdamW(
        probes.parameters(), lr=args.learning_rate, weight_decay=1e-2
    )
    object_type_ids = torch.tensor(
        sorted({key[0] for key in specs}), device=device, dtype=torch.long
    )

    history = []
    for epoch in range(args.epochs):
        probes.train()
        loader = make_loader(
            train_paths,
            max_rows=args.train_events,
            batch_size=args.batch_size,
            stream_batch_size=args.stream_batch_size,
            shuffle=True,
            seed=args.seed + epoch,
        )
        generator = torch.Generator(device=device).manual_seed(args.seed + epoch)
        running_parallel = 0.0
        running_autoregressive = 0.0
        steps = 0
        for batch in loader:
            prepared = prepare_context(
                backbone,
                batch,
                object_type_ids=object_type_ids,
                mask_prob=args.mask_prob,
                generator=generator,
                device=device,
            )
            if prepared is None:
                continue
            context, labels, selected_types, valid_codes = prepared
            parallel, autoregressive = probes.states(context, labels)
            parallel_loss, autoregressive_loss = probe_loss_and_stats(
                probes,
                parallel,
                autoregressive,
                labels,
                selected_types,
                valid_codes,
                specs,
            )
            optimizer.zero_grad(set_to_none=True)
            (parallel_loss + autoregressive_loss).backward()
            torch.nn.utils.clip_grad_norm_(probes.parameters(), 1.0)
            optimizer.step()
            running_parallel += float(parallel_loss.detach())
            running_autoregressive += float(autoregressive_loss.detach())
            steps += 1
            if steps % 500 == 0:
                print(
                    f"epoch {epoch + 1} step {steps}: "
                    f"parallel={running_parallel / steps:.4f} "
                    f"teacher_forced={running_autoregressive / steps:.4f}",
                    flush=True,
                )
        history.append(
            {
                "epoch": epoch + 1,
                "steps": steps,
                "parallel_loss": running_parallel / max(steps, 1),
                "teacher_forced_loss": running_autoregressive / max(steps, 1),
            }
        )

    validation_loader = make_loader(
        val_paths,
        max_rows=args.validation_events,
        batch_size=args.batch_size,
        stream_batch_size=args.stream_batch_size,
        shuffle=False,
        seed=args.seed,
    )
    stats = evaluate(
        backbone,
        probes,
        validation_loader,
        specs=specs,
        object_type_ids=object_type_ids,
        mask_prob=args.mask_prob,
        seed=args.seed + 10_000,
        device=device,
    )
    frame = stats_frame(stats, specs)
    frame.to_csv(output_dir / "per_type_quantizer_probe_metrics.csv", index=False)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    plot_comparison(frame, output_dir / "parallel_vs_teacher_forced_top1.png")
    torch.save(probes.state_dict(), output_dir / "probe_state.pt")

    weights = frame["masked_codes"].to_numpy(dtype=float)
    summary = {
        "checkpoint": str(checkpoint),
        "train_events_per_epoch": args.train_events,
        "validation_events": args.validation_events,
        "epochs": args.epochs,
        "parallel_top1_acc": float(
            np.average(frame["parallel_top1_acc"], weights=weights)
        ),
        "teacher_forced_top1_acc": float(
            np.average(frame["teacher_forced_top1_acc"], weights=weights)
        ),
        "parallel_cross_entropy": float(
            np.average(frame["parallel_cross_entropy"], weights=weights)
        ),
        "teacher_forced_cross_entropy": float(
            np.average(frame["teacher_forced_cross_entropy"], weights=weights)
        ),
        "note": (
            "Teacher-forced metrics condition on true earlier RVQ codes and are an "
            "upper-bound diagnostic, not free-running autoregressive performance."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print("\nPer-type/quantizer metrics:")
    print(frame.to_string(index=False))
    print(f"\nWrote probe evaluation to {output_dir}")


if __name__ == "__main__":
    main()
