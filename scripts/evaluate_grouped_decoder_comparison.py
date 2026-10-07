#!/usr/bin/env python3
"""Compare trained parallel and autoregressive grouped-token decoders.

The two checkpoints are evaluated on exactly the same masked physics objects.
Metrics include the parallel decoder over both the global vocabulary and the
legal object/quantizer range, teacher-forced autoregressive decoding, and
free-running autoregressive decoding.  Optionally, predicted code tuples are
decoded through the object VQ-VAEs to compare feature-level resolution.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from heptokens.data.sequence import MASK_KEY, TOKENS_KEY, TYPE_IDS_KEY
from heptokens.models.foundation_grouped import LitGroupedMaskedSequenceModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_grouped_quantizer_sampling import (  # noqa: E402
    canonical_object_name,
    circular_residual,
    decode_in_batches,
    load_tokenizer,
    local_indices,
    parse_tokenizer_runs,
    resolution_rows,
)
from probe_grouped_autoregressive_quantizers import (  # noqa: E402
    codebook_specs,
    load_vocabulary,
    make_loader,
    mask_object_positions,
    parquet_files,
)


TOP_K = (1, 5, 10, 50)
METHODS = (
    "parallel_global",
    "parallel_constrained",
    "autoregressive_teacher_forced",
    "autoregressive_free_running",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--autoregressive-run", type=Path, required=True)
    parser.add_argument("--parallel-run", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument(
        "--tokenizer-run",
        action="append",
        default=[],
        metavar="OBJECT=RUN_DIR",
        help="Repeat for each object type whose predicted codes should be decoded.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--stream-batch-size", type=int, default=1024)
    parser.add_argument("--validation-events", type=int, default=50_000)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--max-objects-per-type", type=int, default=10_000)
    parser.add_argument("--samples-per-object", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--decode-batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def checkpoint_for_run(run_dir: Path) -> Path:
    preferred = run_dir / "checkpoints" / "last.ckpt"
    if preferred.exists():
        return preferred
    checkpoints = sorted((run_dir / "checkpoints").glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint under {run_dir / 'checkpoints'}")
    return checkpoints[-1]


def load_model(run_dir: Path, device: torch.device):
    checkpoint = checkpoint_for_run(run_dir)
    model = LitGroupedMaskedSequenceModel.load_from_checkpoint(
        checkpoint, map_location="cpu"
    ).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


def validate_models(parallel, autoregressive, vocabulary: dict) -> None:
    if parallel.model.quantizer_decoder != "parallel":
        raise ValueError("--parallel-run does not contain a parallel decoder")
    if autoregressive.model.quantizer_decoder != "autoregressive":
        raise ValueError(
            "--autoregressive-run does not contain an autoregressive decoder"
        )
    for name in ("vocab_size", "hidden_dim", "max_quantizers", "max_seq_length"):
        left = int(getattr(parallel.hparams, name))
        right = int(getattr(autoregressive.hparams, name))
        if left != right:
            raise ValueError(f"Model mismatch for {name}: {left} != {right}")
    expected_vocab = int(vocabulary["vocab_size"])
    if int(parallel.hparams.vocab_size) != expected_vocab:
        raise ValueError(
            f"Checkpoint vocabulary {parallel.hparams.vocab_size} does not match "
            f"validation vocabulary {expected_vocab}"
        )


def update_metrics(stats, key, method: str, logits, truth) -> None:
    values = stats[key][method]
    count = int(truth.numel())
    values["count"] += count
    values["cross_entropy_sum"] += float(
        F.cross_entropy(logits, truth, reduction="sum")
    )
    for k in TOP_K:
        actual_k = min(k, logits.shape[-1])
        correct = logits.topk(actual_k, dim=-1).indices.eq(truth[:, None]).any(1)
        values[f"top{k}_correct"] += int(correct.sum())


def sample_local(logits, temperature: float, generator) -> torch.Tensor:
    probabilities = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(
        probabilities, num_samples=1, generator=generator
    ).squeeze(1)


def parallel_codes(
    logits,
    type_ids,
    specs,
    *,
    sample: bool,
    temperature: float,
    generator,
) -> torch.Tensor:
    codes = torch.zeros(
        logits.shape[:2], dtype=torch.long, device=logits.device
    )
    for (type_id, quantizer), specification in specs.items():
        selected = type_ids.eq(type_id)
        if not selected.any():
            continue
        base = int(specification["base"])
        size = int(specification["size"])
        local_logits = logits[selected, quantizer, base : base + size]
        if sample:
            local = sample_local(local_logits, temperature, generator)
        else:
            local = local_logits.argmax(dim=-1)
        codes[selected, quantizer] = local + base
    return codes


def free_running_autoregressive(
    model,
    context,
    type_ids,
    labels,
    valid_labels,
    specs,
    stats,
) -> torch.Tensor:
    decoder = model.model.autoregressive_decoder
    codes = torch.full_like(labels, model.pad_token_id)
    previous = torch.full(
        (len(context),), model.pad_token_id, dtype=torch.long, device=context.device
    )
    hidden = context.unsqueeze(0)
    for quantizer in range(model.model.max_quantizers):
        decoder_input = decoder.previous_projection(
            model.backbone.token_embedding(previous)
        )
        q_embedding = decoder.quantizer_prediction_embedding.weight[quantizer]
        output, hidden = decoder.autoregressive_gru(
            (decoder_input + q_embedding).unsqueeze(1), hidden
        )
        state = output[:, 0] + q_embedding
        for type_id in torch.unique(type_ids).tolist():
            specification = specs.get((int(type_id), quantizer))
            if specification is None:
                continue
            selected = type_ids.eq(type_id) & valid_labels[:, quantizer]
            if not selected.any():
                continue
            base = int(specification["base"])
            local_truth = labels[selected, quantizer] - base
            local_logits = decoder.object_heads[
                decoder.object_key(int(type_id), quantizer)
            ](state[selected])
            update_metrics(
                stats,
                (int(type_id), quantizer),
                "autoregressive_free_running",
                local_logits,
                local_truth,
            )
            codes[selected, quantizer] = local_logits.argmax(dim=-1) + base
        previous = codes[:, quantizer]
    return codes


def append_decode_records(
    records,
    counts,
    *,
    labels,
    valid_labels,
    type_ids,
    parallel_logits,
    parallel_greedy,
    autoregressive_context,
    autoregressive_greedy,
    autoregressive_model,
    specs,
    max_per_type,
    samples_per_object,
    temperature,
    generator,
) -> None:
    decoder = autoregressive_model.model.autoregressive_decoder
    max_q = autoregressive_model.model.max_quantizers
    for type_id in torch.unique(type_ids).tolist():
        remaining = max_per_type - counts[int(type_id)]
        if remaining <= 0:
            continue
        complete = type_ids.eq(type_id) & valid_labels[:, :max_q].all(dim=1)
        indices = complete.nonzero(as_tuple=False).flatten()[:remaining]
        if not len(indices):
            continue
        truth = labels[indices, :max_q]
        records[int(type_id)]["truth_greedy"].append(truth.cpu())
        records[int(type_id)]["parallel_greedy"].append(
            parallel_greedy[indices, :max_q].cpu()
        )
        records[int(type_id)]["autoregressive_greedy"].append(
            autoregressive_greedy[indices, :max_q].cpu()
        )

        repeated_indices = indices.repeat_interleave(samples_per_object)
        repeated_types = type_ids[indices].repeat_interleave(samples_per_object)
        repeated_context = autoregressive_context[indices].repeat_interleave(
            samples_per_object, dim=0
        )
        sampled_parallel = parallel_codes(
            parallel_logits[repeated_indices],
            repeated_types,
            specs,
            sample=True,
            temperature=temperature,
            generator=generator,
        )
        sampled_autoregressive = decoder.generate_object_codes(
            context=repeated_context,
            type_ids=repeated_types,
            token_embedding=autoregressive_model.backbone.token_embedding,
            sample=True,
            temperature=temperature,
            generator=generator,
        )
        records[int(type_id)]["truth_sample"].append(
            truth.repeat_interleave(samples_per_object, dim=0).cpu()
        )
        records[int(type_id)]["parallel_sample"].append(sampled_parallel.cpu())
        records[int(type_id)]["autoregressive_sample"].append(
            sampled_autoregressive.cpu()
        )
        counts[int(type_id)] += int(len(indices))


def evaluate_models(
    parallel,
    autoregressive,
    loader,
    *,
    specs,
    object_type_ids,
    mask_prob,
    max_per_type,
    samples_per_object,
    temperature,
    seed,
    device,
):
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    tuple_stats = defaultdict(lambda: defaultdict(float))
    records = defaultdict(lambda: defaultdict(list))
    stored_counts = defaultdict(int)
    mask_generator = torch.Generator(device=device).manual_seed(seed)
    sample_generator = torch.Generator(device=device).manual_seed(seed + 1)

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            original = batch[TOKENS_KEY].to(device, non_blocking=True)
            attention_mask = batch[MASK_KEY].bool().to(device, non_blocking=True)
            all_type_ids = batch[TYPE_IDS_KEY].long().to(device, non_blocking=True)
            selected = mask_object_positions(
                original,
                attention_mask,
                all_type_ids,
                object_type_ids=object_type_ids,
                mask_prob=mask_prob,
                generator=mask_generator,
            )
            if not selected.any():
                continue
            labels = original[selected]
            valid_labels = labels.ne(parallel.pad_token_id)
            selected_types = all_type_ids[selected]
            corrupted = original.clone()
            corrupted[selected.unsqueeze(-1) & original.ne(parallel.pad_token_id)] = (
                parallel.mask_token_id
            )

            parallel_logits = parallel.model.masked_logits(
                corrupted, attention_mask, all_type_ids, selected
            )
            for (type_id, quantizer), specification in specs.items():
                chosen = selected_types.eq(type_id) & valid_labels[:, quantizer]
                if not chosen.any():
                    continue
                global_truth = labels[chosen, quantizer]
                global_logits = parallel_logits[chosen, quantizer]
                update_metrics(
                    stats,
                    (type_id, quantizer),
                    "parallel_global",
                    global_logits,
                    global_truth,
                )
                base = int(specification["base"])
                size = int(specification["size"])
                update_metrics(
                    stats,
                    (type_id, quantizer),
                    "parallel_constrained",
                    global_logits[:, base : base + size],
                    global_truth - base,
                )

            parallel_greedy = parallel_codes(
                parallel_logits,
                selected_types,
                specs,
                sample=False,
                temperature=temperature,
                generator=sample_generator,
            )

            autoregressive_hidden = autoregressive.backbone(
                corrupted, attention_mask, all_type_ids
            )[selected]
            decoder = autoregressive.model.autoregressive_decoder
            teacher_states = decoder._teacher_forced_states(
                autoregressive_hidden,
                labels,
                autoregressive.backbone.token_embedding,
            )
            for (type_id, quantizer), specification in specs.items():
                chosen = selected_types.eq(type_id) & valid_labels[:, quantizer]
                if not chosen.any():
                    continue
                base = int(specification["base"])
                local_truth = labels[chosen, quantizer] - base
                local_logits = decoder.object_heads[
                    decoder.object_key(type_id, quantizer)
                ](teacher_states[chosen, quantizer])
                update_metrics(
                    stats,
                    (type_id, quantizer),
                    "autoregressive_teacher_forced",
                    local_logits,
                    local_truth,
                )

            autoregressive_greedy = free_running_autoregressive(
                autoregressive,
                autoregressive_hidden,
                selected_types,
                labels,
                valid_labels,
                specs,
                stats,
            )
            for type_id in torch.unique(selected_types).tolist():
                complete = selected_types.eq(type_id) & valid_labels.all(dim=1)
                count = int(complete.sum())
                if not count:
                    continue
                tuple_stats[int(type_id)]["count"] += count
                tuple_stats[int(type_id)]["parallel_correct"] += int(
                    parallel_greedy[complete].eq(labels[complete]).all(dim=1).sum()
                )
                tuple_stats[int(type_id)]["autoregressive_correct"] += int(
                    autoregressive_greedy[complete]
                    .eq(labels[complete])
                    .all(dim=1)
                    .sum()
                )

            if max_per_type > 0 and samples_per_object > 0:
                append_decode_records(
                    records,
                    stored_counts,
                    labels=labels,
                    valid_labels=valid_labels,
                    type_ids=selected_types,
                    parallel_logits=parallel_logits,
                    parallel_greedy=parallel_greedy,
                    autoregressive_context=autoregressive_hidden,
                    autoregressive_greedy=autoregressive_greedy,
                    autoregressive_model=autoregressive,
                    specs=specs,
                    max_per_type=max_per_type,
                    samples_per_object=samples_per_object,
                    temperature=temperature,
                    generator=sample_generator,
                )
            if (batch_index + 1) % 100 == 0:
                print(
                    f"evaluated {batch_index + 1} batches; stored objects "
                    f"{dict(stored_counts)}",
                    flush=True,
                )
    return stats, tuple_stats, records


def metrics_frame(stats, specs) -> pd.DataFrame:
    rows = []
    for key in sorted(stats):
        row = {
            "object": specs[key]["object"],
            "type_id": key[0],
            "quantizer": key[1],
        }
        for method in METHODS:
            values = stats[key][method]
            count = int(values["count"])
            row[f"{method}_count"] = count
            row[f"{method}_cross_entropy"] = (
                values["cross_entropy_sum"] / count if count else np.nan
            )
            for k in TOP_K:
                row[f"{method}_top{k}_acc"] = (
                    values[f"top{k}_correct"] / count if count else np.nan
                )
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["teacher_forced_top1_gain_over_parallel"] = (
        frame["autoregressive_teacher_forced_top1_acc"]
        - frame["parallel_constrained_top1_acc"]
    )
    frame["free_running_top1_gain_over_parallel"] = (
        frame["autoregressive_free_running_top1_acc"]
        - frame["parallel_constrained_top1_acc"]
    )
    return frame


def weighted_summary(frame: pd.DataFrame) -> dict:
    summary = {}
    for method in METHODS:
        weights = frame[f"{method}_count"].to_numpy(dtype=float)
        summary[method] = {
            "masked_codes": int(weights.sum()),
            "cross_entropy": float(
                np.average(frame[f"{method}_cross_entropy"], weights=weights)
            ),
        }
        for k in TOP_K:
            summary[method][f"top{k}_acc"] = float(
                np.average(frame[f"{method}_top{k}_acc"], weights=weights)
            )
    return summary


def mean_topk_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for quantizer, selected in frame.groupby("quantizer"):
        row = {"quantizer": int(quantizer)}
        for method in METHODS:
            weights = selected[f"{method}_count"].to_numpy(dtype=float)
            for k in TOP_K:
                row[f"{method}_top{k}_acc"] = np.average(
                    selected[f"{method}_top{k}_acc"], weights=weights
                )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("quantizer")


def tuple_frame(tuple_stats, specs) -> pd.DataFrame:
    names = {}
    for (type_id, _), specification in specs.items():
        names[type_id] = specification["object"]
    rows = []
    for type_id, values in sorted(tuple_stats.items()):
        count = int(values["count"])
        rows.append(
            {
                "object": names[type_id],
                "type_id": type_id,
                "complete_objects": count,
                "parallel_exact_q8_acc": values["parallel_correct"] / count,
                "autoregressive_exact_q8_acc": (
                    values["autoregressive_correct"] / count
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_top1(frame: pd.DataFrame, output: Path) -> None:
    objects = list(dict.fromkeys(frame["object"]))
    fig, axes = plt.subplots(
        len(objects),
        1,
        figsize=(11, max(4, 2.7 * len(objects))),
        squeeze=False,
        constrained_layout=True,
    )
    styles = (
        ("parallel_constrained", "Parallel, legal range", "#4c78a8"),
        ("autoregressive_teacher_forced", "AR, teacher forced", "#f58518"),
        ("autoregressive_free_running", "AR, free running", "#e45756"),
    )
    for axis, object_name in zip(axes[:, 0], objects):
        selected = frame[frame.object.eq(object_name)].sort_values("quantizer")
        for method, label, color in styles:
            axis.plot(
                selected.quantizer,
                selected[f"{method}_top1_acc"],
                marker="o",
                label=label,
                color=color,
            )
        axis.set_xticks(range(8), [f"q{q}" for q in range(8)])
        axis.set_ylabel("Top-1 accuracy")
        axis.set_title(object_name)
        axis.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Residual quantizer level")
    axes[0, 0].legend(frameon=False, ncols=3, fontsize=9)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_resolution(frame: pd.DataFrame, output: Path) -> None:
    objects = list(dict.fromkeys(frame["object"]))
    methods = (
        ("parallel_greedy", "Parallel greedy", "#4c78a8"),
        ("autoregressive_greedy", "AR greedy", "#72b7b2"),
        ("parallel_sample", "Parallel sampled", "#f58518"),
        ("autoregressive_sample", "AR sampled", "#e45756"),
    )
    fig, axes = plt.subplots(
        len(objects),
        1,
        figsize=(13, max(4, 3.1 * len(objects))),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, object_name in zip(axes[:, 0], objects):
        selected = frame[frame.object.eq(object_name)]
        features = list(dict.fromkeys(selected.feature))
        x = np.arange(len(features))
        width = 0.2
        for index, (method, label, color) in enumerate(methods):
            values = [
                selected[
                    selected.method.eq(method) & selected.feature.eq(feature)
                ].resolution_over_reference_iqr.iloc[0]
                for feature in features
            ]
            axis.bar(
                x + (index - 1.5) * width,
                values,
                width,
                label=label,
                color=color,
            )
        axis.set_xticks(x, features, rotation=35, ha="right")
        axis.set_ylabel("Residual IQR / true-decode IQR")
        axis.set_title(object_name)
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False, ncols=4, fontsize=9)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def finite_range(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -1.0, 1.0
    lo, hi = np.percentile(values, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(values.min()), float(values.max())
    if lo == hi:
        width = abs(lo) * 0.05 if lo else 1.0
        lo, hi = lo - width, hi + width
    return float(lo), float(hi)


def histogram_bins(values: np.ndarray, n_bins: int = 70) -> np.ndarray:
    lo, hi = finite_range(values)
    return np.linspace(lo, hi, n_bins + 1)


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "feature"


def residual_for_feature(
    predicted: np.ndarray, reference: np.ndarray, feature_name: str
) -> np.ndarray:
    name = feature_name.lower()
    if name == "phi" or name.endswith("/phi"):
        return circular_residual(predicted, reference)
    return predicted - reference


def apply_plot_style(axis) -> None:
    axis.tick_params(direction="in", which="both", top=True, right=True)
    axis.minorticks_on()
    axis.grid(alpha=0.18)


def plot_decoder_triptychs(
    *,
    object_name: str,
    feature_names: list[str],
    reference: np.ndarray,
    parallel: np.ndarray,
    autoregressive: np.ndarray,
    mode: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for feature_index, feature_name in enumerate(feature_names):
        truth = reference[:, feature_index]
        parallel_values = parallel[:, feature_index]
        autoregressive_values = autoregressive[:, feature_index]
        finite = (
            np.isfinite(truth)
            & np.isfinite(parallel_values)
            & np.isfinite(autoregressive_values)
        )
        truth = truth[finite]
        parallel_values = parallel_values[finite]
        autoregressive_values = autoregressive_values[finite]
        if truth.size == 0:
            continue

        parallel_residual = residual_for_feature(
            parallel_values, truth, feature_name
        )
        autoregressive_residual = residual_for_feature(
            autoregressive_values, truth, feature_name
        )
        value_bins = histogram_bins(
            np.concatenate([truth, parallel_values, autoregressive_values])
        )
        residual_bins = histogram_bins(
            np.concatenate([parallel_residual, autoregressive_residual])
        )

        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
        axes[0].hist(
            truth,
            bins=value_bins,
            histtype="step",
            linewidth=2.0,
            color="black",
            label="True-code decode",
        )
        axes[0].hist(
            parallel_values,
            bins=value_bins,
            histtype="step",
            linewidth=1.8,
            color="#4c78a8",
            label="Parallel",
        )
        axes[0].hist(
            autoregressive_values,
            bins=value_bins,
            histtype="step",
            linewidth=1.8,
            color="#e45756",
            label="Autoregressive",
        )
        axes[0].set_title("Decoded distribution")
        axes[0].set_xlabel(feature_name)
        axes[0].set_ylabel("Objects")
        axes[0].legend(frameon=False, fontsize=9)

        parallel_iqr = np.subtract(
            *np.percentile(parallel_residual, [75, 25])
        )
        autoregressive_iqr = np.subtract(
            *np.percentile(autoregressive_residual, [75, 25])
        )
        axes[1].hist(
            parallel_residual,
            bins=residual_bins,
            histtype="step",
            linewidth=1.8,
            color="#4c78a8",
            label=f"Parallel, IQR={parallel_iqr:.3g}",
        )
        axes[1].hist(
            autoregressive_residual,
            bins=residual_bins,
            histtype="step",
            linewidth=1.8,
            color="#e45756",
            label=f"AR, IQR={autoregressive_iqr:.3g}",
        )
        axes[1].axvline(0, color="black", linewidth=1.1)
        axes[1].set_title("Prediction residual")
        axes[1].set_xlabel("Predicted-code decode - true-code decode")
        axes[1].set_ylabel("Objects")
        axes[1].legend(frameon=False, fontsize=9)

        n_plot = min(truth.size, 8_000)
        indices = rng.choice(truth.size, size=n_plot, replace=False)
        axes[2].scatter(
            truth[indices],
            parallel_values[indices],
            s=4,
            alpha=0.18,
            color="#4c78a8",
            label="Parallel",
            rasterized=True,
        )
        axes[2].scatter(
            truth[indices],
            autoregressive_values[indices],
            s=4,
            alpha=0.18,
            color="#e45756",
            label="Autoregressive",
            rasterized=True,
        )
        lo, hi = finite_range(
            np.concatenate(
                [truth[indices], parallel_values[indices], autoregressive_values[indices]]
            )
        )
        axes[2].plot([lo, hi], [lo, hi], color="black", linewidth=1.1)
        axes[2].set_xlim(lo, hi)
        axes[2].set_ylim(lo, hi)
        axes[2].set_title("Event-by-event correlation")
        axes[2].set_xlabel("True-code decode")
        axes[2].set_ylabel("Predicted-code decode")
        axes[2].legend(frameon=False, fontsize=9)

        for axis in axes:
            apply_plot_style(axis)
        fig.suptitle(f"{object_name}: {feature_name} ({mode})", fontsize=16)
        fig.tight_layout()
        fig.savefig(
            output_dir
            / f"{safe_filename(feature_name)}_{safe_filename(mode)}_triptych.png",
            dpi=180,
        )
        plt.close(fig)


def decode_records(
    records,
    *,
    specs,
    tokenizer_runs,
    device,
    decode_batch_size,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    summary = {}
    for type_id, values in sorted(records.items()):
        object_specs = [
            specs[(type_id, q)]
            for q in range(8)
            if (type_id, q) in specs
        ]
        object_name = canonical_object_name(object_specs[0]["object"])
        tokenizer_run = tokenizer_runs.get(object_name)
        if tokenizer_run is None:
            print(f"Skipping {object_name}: no tokenizer run supplied", flush=True)
            continue
        model, transformer, n_features, feature_names, checkpoint = load_tokenizer(
            tokenizer_run, object_name, object_specs, device
        )
        method_pairs = (
            ("parallel_greedy", "truth_greedy"),
            ("autoregressive_greedy", "truth_greedy"),
            ("parallel_sample", "truth_sample"),
            ("autoregressive_sample", "truth_sample"),
        )
        decoded_references = {}
        decoded_predictions = {}
        for method, truth_key in method_pairs:
            truth_global = torch.cat(values[truth_key])
            predicted_global = torch.cat(values[method])
            reference_key = truth_key
            if reference_key not in decoded_references:
                truth_local = local_indices(truth_global, object_specs)
                decoded_references[reference_key] = decode_in_batches(
                    model,
                    truth_local,
                    transformer=transformer,
                    device=device,
                    n_features=n_features,
                    batch_size=decode_batch_size,
                )
            predicted_local = local_indices(predicted_global, object_specs)
            predicted = decode_in_batches(
                model,
                predicted_local,
                transformer=transformer,
                device=device,
                n_features=n_features,
                batch_size=decode_batch_size,
            )
            decoded_predictions[method] = predicted
            rows.extend(
                resolution_rows(
                    object_name,
                    feature_names,
                    decoded_references[reference_key],
                    predicted,
                    method,
                )
            )

        array_dir = output_dir / "decoded_arrays"
        array_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            array_dir / f"{safe_filename(object_name)}.npz",
            feature_names=np.asarray(feature_names),
            truth_greedy=decoded_references["truth_greedy"],
            parallel_greedy=decoded_predictions["parallel_greedy"],
            autoregressive_greedy=decoded_predictions["autoregressive_greedy"],
            truth_sample=decoded_references["truth_sample"],
            parallel_sample=decoded_predictions["parallel_sample"],
            autoregressive_sample=decoded_predictions["autoregressive_sample"],
        )
        triptych_dir = output_dir / "decoded_feature_triptychs" / safe_filename(
            object_name
        )
        plot_decoder_triptychs(
            object_name=object_name,
            feature_names=feature_names,
            reference=decoded_references["truth_greedy"],
            parallel=decoded_predictions["parallel_greedy"],
            autoregressive=decoded_predictions["autoregressive_greedy"],
            mode="greedy",
            output_dir=triptych_dir,
        )
        plot_decoder_triptychs(
            object_name=object_name,
            feature_names=feature_names,
            reference=decoded_references["truth_sample"],
            parallel=decoded_predictions["parallel_sample"],
            autoregressive=decoded_predictions["autoregressive_sample"],
            mode="sampled",
            output_dir=triptych_dir,
        )
        summary[object_name] = {
            "greedy_objects": int(sum(len(x) for x in values["truth_greedy"])),
            "sampled_decodes": int(sum(len(x) for x in values["truth_sample"])),
            "tokenizer_checkpoint": str(checkpoint),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return pd.DataFrame(rows), summary


def main() -> None:
    args = parse_args()
    if not 0 < args.mask_prob <= 1:
        raise ValueError("--mask-prob must be in (0, 1]")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    val_paths = parquet_files(args.prepared_dir.expanduser().resolve(), "val")
    vocabulary = load_vocabulary(val_paths[0])
    specs = codebook_specs(vocabulary)
    parallel, parallel_checkpoint = load_model(
        args.parallel_run.expanduser().resolve(), device
    )
    autoregressive, autoregressive_checkpoint = load_model(
        args.autoregressive_run.expanduser().resolve(), device
    )
    validate_models(parallel, autoregressive, vocabulary)
    object_type_ids = torch.tensor(
        sorted({key[0] for key in specs}), dtype=torch.long, device=device
    )
    loader = make_loader(
        val_paths,
        max_rows=args.validation_events,
        batch_size=args.batch_size,
        stream_batch_size=args.stream_batch_size,
        shuffle=False,
        seed=args.seed,
    )
    stats, q8_stats, records = evaluate_models(
        parallel,
        autoregressive,
        loader,
        specs=specs,
        object_type_ids=object_type_ids,
        mask_prob=args.mask_prob,
        max_per_type=args.max_objects_per_type,
        samples_per_object=args.samples_per_object,
        temperature=args.temperature,
        seed=args.seed,
        device=device,
    )

    metrics = metrics_frame(stats, specs)
    metrics.to_csv(output_dir / "per_object_quantizer_metrics.csv", index=False)
    mean_topk = mean_topk_frame(metrics)
    mean_topk.to_csv(output_dir / "mean_topk_by_quantizer.csv", index=False)
    exact_q8 = tuple_frame(q8_stats, specs)
    exact_q8.to_csv(output_dir / "exact_q8_accuracy.csv", index=False)
    plot_top1(metrics, output_dir / "top1_accuracy_by_object_and_quantizer.png")

    decoded_summary = {}
    if args.tokenizer_run:
        tokenizer_runs = parse_tokenizer_runs(args.tokenizer_run)
        resolution, decoded_summary = decode_records(
            records,
            specs=specs,
            tokenizer_runs=tokenizer_runs,
            device=device,
            decode_batch_size=args.decode_batch_size,
            output_dir=output_dir,
        )
        if not resolution.empty:
            resolution.to_csv(
                output_dir / "decoded_feature_resolution.csv", index=False
            )
            plot_resolution(
                resolution, output_dir / "decoded_feature_resolution.png"
            )

    summary = {
        "parallel_checkpoint": str(parallel_checkpoint),
        "autoregressive_checkpoint": str(autoregressive_checkpoint),
        "prepared_dir": str(args.prepared_dir.expanduser().resolve()),
        "validation_events": args.validation_events,
        "mask_prob": args.mask_prob,
        "temperature": args.temperature,
        "samples_per_object": args.samples_per_object,
        "weighted_metrics": weighted_summary(metrics),
        "decoded": decoded_summary,
        "decode_reference": "VQ-VAE decode of the true Q0-Q7 tuple",
        "interpretation": {
            "parallel_global": "Original parallel head over the full global vocabulary.",
            "parallel_constrained": "Same logits restricted to the legal object/quantizer code range.",
            "autoregressive_teacher_forced": "Conditions on true earlier codes; an upper-bound diagnostic.",
            "autoregressive_free_running": "Conditions on its own earlier greedy predictions.",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote evaluation to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
