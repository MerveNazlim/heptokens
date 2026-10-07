#!/usr/bin/env python3
"""Evaluate RVQ top-k prediction and decoded sampling resolution."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from heptokens.models.vq_vae import LitVqVae

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_event_token_physics_reco import decode_object_tokens  # noqa: E402
from analyze_vqvae_tokenizer import (  # noqa: E402
    feature_count_from_cfg,
    feature_names_from_cfg,
    find_checkpoint,
    transform_list_and_cst_fn_from_cfg,
)
from probe_grouped_autoregressive_quantizers import (  # noqa: E402
    QuantizerProbes,
    codebook_specs,
    load_backbone,
    load_vocabulary,
    make_loader,
    parquet_files,
    prepare_context,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--probe-state", type=Path, required=True)
    parser.add_argument(
        "--tokenizer-run",
        action="append",
        default=[],
        metavar="OBJECT=RUN_DIR",
        help="Repeat for every object type to decode (for example electrons=results/...).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--stream-batch-size", type=int, default=1024)
    parser.add_argument("--validation-events", type=int, default=50_000)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--max-objects-per-type", type=int, default=10_000)
    parser.add_argument("--samples-per-object", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--decode-batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def parse_tokenizer_runs(values: list[str]) -> dict[str, Path]:
    runs = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected OBJECT=RUN_DIR, received {value!r}")
        name, path = value.split("=", 1)
        runs[name.strip().lower()] = Path(path).expanduser().resolve()
    if not runs:
        raise ValueError("At least one --tokenizer-run OBJECT=RUN_DIR is required")
    return runs


def canonical_object_name(name: str) -> str:
    name = name.lower().strip()
    return name if name.endswith("s") else f"{name}s"


def build_probes(backbone, specs, probe_state: Path, device: torch.device):
    probes = QuantizerProbes(
        hidden_dim=int(backbone.hparams.hidden_dim),
        vocab_size=int(backbone.hparams.vocab_size),
        embedding_dim=backbone.backbone.token_embedding.embedding_dim,
        max_quantizers=int(backbone.hparams.max_quantizers),
        specs=specs,
        token_embedding=backbone.backbone.token_embedding.weight.detach().cpu(),
    )
    state = torch.load(probe_state, map_location="cpu", weights_only=True)
    probes.load_state_dict(state)
    return probes.eval().to(device)


def update_topk_stats(stats, logits, truth, key, prefix: str) -> None:
    for k in (1, 5, 10, 50):
        actual_k = min(k, logits.shape[-1])
        correct = logits.topk(actual_k, dim=-1).indices.eq(truth[:, None]).any(dim=1)
        stats[key][f"{prefix}_top{k}_correct"] += int(correct.sum())


def collect_contexts_and_topk(
    backbone,
    probes,
    loader,
    *,
    specs,
    object_type_ids,
    mask_prob,
    max_per_type,
    seed,
    device,
):
    stats = defaultdict(lambda: defaultdict(float))
    stored = defaultdict(lambda: {"context": [], "labels": []})
    counts = defaultdict(int)
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
            parallel_states, teacher_states = probes.states(context, labels)
            for type_id in torch.unique(selected_types).tolist():
                type_selected = selected_types.eq(type_id)
                for q in range(labels.shape[1]):
                    selected = type_selected & valid_codes[:, q]
                    specification = specs.get((int(type_id), q))
                    if specification is None or not selected.any():
                        continue
                    truth = labels[selected, q] - int(specification["base"])
                    parallel_logits = probes.logits(
                        parallel_states[selected, q],
                        type_id=int(type_id), quantizer=q, autoregressive=False,
                    )
                    teacher_logits = probes.logits(
                        teacher_states[selected, q],
                        type_id=int(type_id), quantizer=q, autoregressive=True,
                    )
                    key = (int(type_id), q)
                    stats[key]["count"] += int(truth.numel())
                    update_topk_stats(stats, parallel_logits, truth, key, "parallel")
                    update_topk_stats(stats, teacher_logits, truth, key, "teacher_forced")

                remaining = max_per_type - counts[int(type_id)]
                if remaining > 0:
                    indices = type_selected.nonzero(as_tuple=False).flatten()[:remaining]
                    stored[int(type_id)]["context"].append(context[indices].cpu())
                    stored[int(type_id)]["labels"].append(labels[indices].cpu())
                    counts[int(type_id)] += int(indices.numel())
            if counts and all(counts[int(value)] >= max_per_type for value in object_type_ids.tolist()):
                break
    packed = {}
    for type_id, values in stored.items():
        if values["context"]:
            packed[type_id] = {
                "context": torch.cat(values["context"]),
                "labels": torch.cat(values["labels"]),
            }
    return stats, packed


def topk_frame(stats, specs) -> pd.DataFrame:
    rows = []
    for key, values in sorted(stats.items()):
        count = int(values["count"])
        row = {
            "object": specs[key]["object"],
            "type_id": key[0],
            "quantizer": key[1],
            "masked_codes": count,
        }
        for prefix in ("parallel", "teacher_forced"):
            for k in (1, 5, 10, 50):
                row[f"{prefix}_top{k}_acc"] = values[f"{prefix}_top{k}_correct"] / count
        rows.append(row)
    return pd.DataFrame(rows)


def sample_local(logits: torch.Tensor, temperature: float, generator) -> torch.Tensor:
    probabilities = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probabilities, 1, generator=generator).squeeze(1)


def sample_code_tuples(
    probes,
    context,
    *,
    type_id,
    specs,
    samples_per_object,
    temperature,
    seed,
    device,
):
    context = context.to(device).repeat_interleave(samples_per_object, dim=0)
    n = context.shape[0]
    max_q = probes.max_quantizers
    parallel_codes = torch.zeros((n, max_q), dtype=torch.long, device=device)
    autoregressive_codes = torch.zeros_like(parallel_codes)
    generator = torch.Generator(device=device).manual_seed(seed)

    quantizer_ids = torch.arange(max_q, device=device).unsqueeze(0)
    q_embedding = probes.quantizer_embedding(quantizer_ids)
    parallel_states = context.unsqueeze(1) + q_embedding
    hidden = context.unsqueeze(0)
    previous = torch.zeros(n, dtype=torch.long, device=device)
    for q in range(max_q):
        specification = specs.get((type_id, q))
        if specification is None:
            continue
        base = int(specification["base"])
        parallel_logits = probes.logits(
            parallel_states[:, q], type_id=type_id, quantizer=q, autoregressive=False
        )
        parallel_codes[:, q] = sample_local(
            parallel_logits, temperature, generator
        ) + base

        previous_embedded = probes.previous_projection(probes.previous_embedding(previous))
        decoder_input = previous_embedded + q_embedding[:, q]
        output, hidden = probes.autoregressive_gru(decoder_input.unsqueeze(1), hidden)
        state = output[:, 0] + q_embedding[:, q]
        logits = probes.logits(
            state, type_id=type_id, quantizer=q, autoregressive=True
        )
        autoregressive_codes[:, q] = sample_local(logits, temperature, generator) + base
        previous = autoregressive_codes[:, q]
    return parallel_codes.cpu(), autoregressive_codes.cpu()


def load_tokenizer(run_dir: Path, expected_name: str, expected_specs, device):
    config_path = run_dir / "full_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    cfg = OmegaConf.load(config_path)
    checkpoint = find_checkpoint(run_dir, None)
    model = LitVqVae.load_from_checkpoint(checkpoint, map_location="cpu").eval().to(device)
    _, inverse_transformer = transform_list_and_cst_fn_from_cfg(cfg)
    n_features = feature_count_from_cfg(cfg)
    feature_names = feature_names_from_cfg(cfg, n_features)
    expected_q = len(expected_specs)
    if int(model.hparams.num_quantizers) != expected_q:
        raise ValueError(
            f"{expected_name}: tokenizer has {model.hparams.num_quantizers} quantizers, "
            f"but vocabulary defines {expected_q}"
        )
    sizes = {int(value["size"]) for value in expected_specs}
    if sizes != {int(model.hparams.codebook_size)}:
        raise ValueError(
            f"{expected_name}: tokenizer codebook={model.hparams.codebook_size}, "
            f"vocabulary sizes={sorted(sizes)}"
        )
    return model, inverse_transformer, n_features, feature_names, checkpoint


def local_indices(global_codes: torch.Tensor, object_specs: list[dict]) -> np.ndarray:
    result = global_codes.numpy().copy()
    for q, specification in enumerate(object_specs):
        result[:, q] -= int(specification["base"])
    return result.astype(np.int64)


def decode_in_batches(
    model, codes, *, transformer, device, n_features, batch_size
) -> np.ndarray:
    outputs = []
    for start in range(0, len(codes), batch_size):
        batch_codes = codes[start : start + batch_size, None, :]
        mask = np.ones(batch_codes.shape[:2], dtype=bool)
        decoded = decode_object_tokens(
            model=model,
            indices=batch_codes,
            mask=mask,
            transformer=transformer,
            device=device,
            n_features=n_features,
        )
        outputs.append(decoded[:, 0])
    return np.concatenate(outputs)


def circular_residual(predicted: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(predicted - reference), np.cos(predicted - reference))


def resolution_rows(object_name, feature_names, reference, predictions, method):
    rows = []
    for index, feature in enumerate(feature_names):
        if feature.lower() == "phi" or feature.lower().endswith("/phi"):
            residual = circular_residual(predictions[:, index], reference[:, index])
        else:
            residual = predictions[:, index] - reference[:, index]
        reference_iqr = np.subtract(*np.percentile(reference[:, index], [75, 25]))
        residual_iqr = np.subtract(*np.percentile(residual, [75, 25]))
        rows.append(
            {
                "object": object_name,
                "feature": feature,
                "method": method,
                "samples": len(residual),
                "median_residual": float(np.median(residual)),
                "median_absolute_error": float(np.median(np.abs(residual))),
                "residual_iqr": float(residual_iqr),
                "reference_iqr": float(reference_iqr),
                "resolution_over_reference_iqr": float(
                    residual_iqr / max(abs(reference_iqr), 1e-12)
                ),
            }
        )
    return rows


def plot_topk(frame: pd.DataFrame, output: Path) -> None:
    summary = []
    for (object_name, quantizer), selected in frame.groupby(["object", "quantizer"]):
        weights = selected["masked_codes"]
        for method in ("parallel", "teacher_forced"):
            for k in (1, 5, 10, 50):
                summary.append(
                    {
                        "object": object_name,
                        "quantizer": quantizer,
                        "method": method,
                        "k": k,
                        "accuracy": np.average(selected[f"{method}_top{k}_acc"], weights=weights),
                    }
                )
    plot_frame = pd.DataFrame(summary)
    objects = list(dict.fromkeys(plot_frame["object"]))
    fig, axes = plt.subplots(len(objects), 1, figsize=(11, max(4, 2.8 * len(objects))), squeeze=False, constrained_layout=True)
    colors = {5: "#4c78a8", 10: "#f58518", 50: "#54a24b"}
    for axis, object_name in zip(axes[:, 0], objects):
        selected = plot_frame[plot_frame.object.eq(object_name)]
        quantizers = sorted(selected.quantizer.unique())
        x = np.arange(len(quantizers))
        width = 0.12
        offset = -2.5 * width
        for method, hatch in (("parallel", ""), ("teacher_forced", "//")):
            for k in (5, 10, 50):
                values = [selected[(selected.method == method) & (selected.k == k) & (selected.quantizer == q)].accuracy.iloc[0] for q in quantizers]
                axis.bar(x + offset, values, width, color=colors[k], hatch=hatch, label=f"{method}, top-{k}")
                offset += width
        axis.set_xticks(x, [f"q{q}" for q in quantizers])
        axis.set_ylabel("Accuracy")
        axis.set_title(object_name)
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False, ncols=3, fontsize=9)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_resolution(frame: pd.DataFrame, output: Path) -> None:
    objects = list(dict.fromkeys(frame["object"]))
    fig, axes = plt.subplots(len(objects), 1, figsize=(12, max(4, 3.0 * len(objects))), squeeze=False, constrained_layout=True)
    for axis, object_name in zip(axes[:, 0], objects):
        selected = frame[frame.object.eq(object_name)]
        features = list(dict.fromkeys(selected.feature))
        x = np.arange(len(features))
        for shift, method, color in ((-0.2, "parallel_sample", "#4c78a8"), (0.2, "autoregressive_sample", "#f58518")):
            values = [selected[(selected.method == method) & (selected.feature == feature)].resolution_over_reference_iqr.iloc[0] for feature in features]
            axis.bar(x + shift, values, 0.4, label=method.replace("_", " "), color=color)
        axis.set_xticks(x, features, rotation=35, ha="right")
        axis.set_ylabel("Residual IQR / true-decode IQR")
        axis.set_title(object_name)
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False, ncols=2)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    run_dir = args.run_dir.resolve()
    prepared_dir = args.prepared_dir.resolve()
    output_dir = (args.output_dir or run_dir / "quantizer_sampling_evaluation").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    val_paths = parquet_files(prepared_dir, "val")
    vocabulary = load_vocabulary(val_paths[0])
    specs = codebook_specs(vocabulary)
    backbone, backbone_checkpoint = load_backbone(run_dir, device)
    probes = build_probes(backbone, specs, args.probe_state.resolve(), device)
    object_type_ids = torch.tensor(sorted({key[0] for key in specs}), device=device)
    loader = make_loader(
        val_paths,
        max_rows=args.validation_events,
        batch_size=args.batch_size,
        stream_batch_size=args.stream_batch_size,
        shuffle=False,
        seed=args.seed,
    )
    stats, packed = collect_contexts_and_topk(
        backbone, probes, loader,
        specs=specs,
        object_type_ids=object_type_ids,
        mask_prob=args.mask_prob,
        max_per_type=args.max_objects_per_type,
        seed=args.seed,
        device=device,
    )
    topk = topk_frame(stats, specs)
    topk.to_csv(output_dir / "parallel_vs_teacher_forced_topk.csv", index=False)
    plot_topk(topk, output_dir / "parallel_vs_teacher_forced_topk.png")

    tokenizer_runs = parse_tokenizer_runs(args.tokenizer_run)
    resolution = []
    decoded_summary = {}
    for type_id, values in packed.items():
        object_specs = [specs[(type_id, q)] for q in range(probes.max_quantizers) if (type_id, q) in specs]
        object_name = canonical_object_name(object_specs[0]["object"])
        run_dir_for_object = tokenizer_runs.get(object_name)
        if run_dir_for_object is None:
            print(f"Skipping decoded resolution for {object_name}: no --tokenizer-run supplied")
            continue
        model, transformer, n_features, feature_names, tokenizer_checkpoint = load_tokenizer(
            run_dir_for_object, object_name, object_specs, device
        )
        parallel_global, autoregressive_global = sample_code_tuples(
            probes,
            values["context"],
            type_id=type_id,
            specs=specs,
            samples_per_object=args.samples_per_object,
            temperature=args.temperature,
            seed=args.seed + type_id,
            device=device,
        )
        truth_global = values["labels"].repeat_interleave(args.samples_per_object, dim=0)
        truth_local = local_indices(truth_global, object_specs)
        parallel_local = local_indices(parallel_global, object_specs)
        autoregressive_local = local_indices(autoregressive_global, object_specs)
        truth_decoded = decode_in_batches(model, truth_local, transformer=transformer, device=device, n_features=n_features, batch_size=args.decode_batch_size)
        parallel_decoded = decode_in_batches(model, parallel_local, transformer=transformer, device=device, n_features=n_features, batch_size=args.decode_batch_size)
        autoregressive_decoded = decode_in_batches(model, autoregressive_local, transformer=transformer, device=device, n_features=n_features, batch_size=args.decode_batch_size)
        resolution.extend(resolution_rows(object_name, feature_names, truth_decoded, parallel_decoded, "parallel_sample"))
        resolution.extend(resolution_rows(object_name, feature_names, truth_decoded, autoregressive_decoded, "autoregressive_sample"))
        decoded_summary[object_name] = {
            "objects": int(len(values["context"])),
            "samples_per_object": args.samples_per_object,
            "decoded_samples": int(len(truth_decoded)),
            "tokenizer_checkpoint": str(tokenizer_checkpoint),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    resolution_frame = pd.DataFrame(resolution)
    if not resolution_frame.empty:
        resolution_frame.to_csv(output_dir / "decoded_feature_resolution.csv", index=False)
        plot_resolution(resolution_frame, output_dir / "decoded_feature_resolution.png")

    summary = {
        "backbone_checkpoint": str(backbone_checkpoint),
        "probe_state": str(args.probe_state.resolve()),
        "validation_events": args.validation_events,
        "temperature": args.temperature,
        "samples_per_object": args.samples_per_object,
        "decoded": decoded_summary,
        "reference": "VQ-VAE decode of the true Q8 tuple",
        "note": "Teacher-forced top-k uses true earlier codes; sampled autoregressive decoding is free-running and uses its own earlier samples.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Wrote evaluation to {output_dir}")


if __name__ == "__main__":
    main()
