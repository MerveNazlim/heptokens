#!/usr/bin/env python3
"""Compare Q1, Q8, and flat-continuous HZZ classifiers on matched events."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from heptokens.data.benchmark_sequence import sequence_batch_to_device
from heptokens.data.benchmark_token_parquet import (
    GroupedTokenParquetClassificationModule,
)
from heptokens.models.foundation_continuous import LitContinuousCLSClassifier
from heptokens.models.foundation_grouped_cls_classifier import LitGroupedCLSClassifier


@dataclass(frozen=True)
class RunSpec:
    name: str
    representation: str
    checkpoint: Path
    prepared_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q1-checkpoint", type=Path, required=True)
    parser.add_argument("--q1-prepared-dir", type=Path, required=True)
    parser.add_argument("--q1-data-only-checkpoint", type=Path)
    parser.add_argument("--q1-data-only-prepared-dir", type=Path)
    parser.add_argument("--q8-checkpoint", type=Path, required=True)
    parser.add_argument("--q8-prepared-dir", type=Path, required=True)
    parser.add_argument("--q8-data-only-checkpoint", type=Path)
    parser.add_argument("--q8-data-only-prepared-dir", type=Path)
    parser.add_argument("--continuous-checkpoint", type=Path, required=True)
    parser.add_argument("--continuous-prepared-dir", type=Path, required=True)
    parser.add_argument("--continuous-data-only-checkpoint", type=Path)
    parser.add_argument("--continuous-data-only-prepared-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--stream-batch-size", type=int, default=4096)
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def validate_paths(spec: RunSpec) -> None:
    if not spec.checkpoint.is_file():
        raise FileNotFoundError(f"Missing {spec.name} checkpoint: {spec.checkpoint}")
    manifest = spec.prepared_dir / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing {spec.name} manifest: {manifest}")
    if not list((spec.prepared_dir / "test" / "signal").glob("*.parquet")):
        raise FileNotFoundError(f"Missing {spec.name} signal test shards")
    if not list((spec.prepared_dir / "test" / "background").glob("*.parquet")):
        raise FileNotFoundError(f"Missing {spec.name} background test shards")


def read_test_identities(prepared_dir: Path) -> set[tuple[str, int, int]]:
    import pyarrow.parquet as pq

    identities: set[tuple[str, int, int]] = set()
    rows = 0
    for class_name, fallback_label in (("signal", 1), ("background", 0)):
        for path in sorted((prepared_dir / "test" / class_name).glob("*.parquet")):
            parquet = pq.ParquetFile(path)
            names = set(parquet.schema_arrow.names)
            required = {"source_file", "event_index"}
            missing = required - names
            if missing:
                raise ValueError(f"{path} is missing identity columns {sorted(missing)}")
            columns = ["source_file", "event_index"]
            if "label" in names:
                columns.append("label")
            for batch in parquet.iter_batches(batch_size=65536, columns=columns):
                source_files = batch.column(0).to_pylist()
                event_indices = batch.column(1).to_pylist()
                labels = (
                    batch.column(2).to_pylist()
                    if len(columns) == 3
                    else [fallback_label] * batch.num_rows
                )
                for source_file, event_index, label in zip(
                    source_files, event_indices, labels, strict=True
                ):
                    identity = (str(source_file), int(event_index), int(label))
                    if identity in identities:
                        raise ValueError(f"Duplicate test identity in {prepared_dir}: {identity}")
                    identities.add(identity)
                rows += batch.num_rows
    if rows != len(identities):
        raise RuntimeError(f"Identity count mismatch for {prepared_dir}")
    return identities


def verify_matched_events(specs: list[RunSpec]) -> int:
    reference_spec = specs[0]
    reference = read_test_identities(reference_spec.prepared_dir)
    for spec in specs[1:]:
        observed = read_test_identities(spec.prepared_dir)
        if observed != reference:
            missing = sorted(reference - observed)[:3]
            extra = sorted(observed - reference)[:3]
            raise RuntimeError(
                f"{spec.name} test events differ from {reference_spec.name}: "
                f"missing={len(reference - observed)}, extra={len(observed - reference)}, "
                f"missing_examples={missing}, extra_examples={extra}"
            )
    return len(reference)


def load_model(spec: RunSpec, device: torch.device):
    model_class = (
        LitContinuousCLSClassifier
        if spec.representation == "continuous"
        else LitGroupedCLSClassifier
    )
    model = model_class.load_from_checkpoint(
        spec.checkpoint,
        map_location="cpu",
        backbone_ckpt_path=None,
    )
    model.eval().to(device)
    return model


@torch.inference_mode()
def predict(model, dataloader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    scores = []
    labels = []
    for batch in dataloader:
        batch = sequence_batch_to_device(batch, str(device))
        prediction = model.predict_step(batch)
        scores.append(prediction["score"].detach().cpu().numpy())
        labels.append(prediction["label"].detach().cpu().numpy())
    if not scores:
        raise RuntimeError("Test dataloader produced no batches")
    return np.concatenate(scores), np.concatenate(labels)


def rejection_at_efficiency(
    labels: np.ndarray, scores: np.ndarray, efficiency: float
) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    background_events = max(int(np.count_nonzero(labels == 0)), 1)
    false_positive_rate = max(
        float(np.interp(efficiency, tpr, fpr)), 1.0 / background_events
    )
    return 1.0 / false_positive_rate


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def write_roc_plots(
    output_dir: Path,
    curves: list[tuple[str, float, np.ndarray, np.ndarray, int]],
) -> None:
    colors = {
        "Q1 tokens": "#0072B2",
        "Q1 tokens (MC+data)": "#0072B2",
        "Q1 tokens (200M data-only pretrain)": "#CC79A7",
        "Q8 tokens": "#D55E00",
        "Q8 tokens (MC+data)": "#D55E00",
        "Q8 tokens (200M data-only pretrain)": "#E69F00",
        "Continuous": "#009E73",
        "Continuous (MC+data)": "#009E73",
        "Continuous (200M data-only pretrain)": "#333333",
    }

    figure, axis = plt.subplots(figsize=(8, 6))
    for name, auc, fpr, tpr, _ in curves:
        axis.plot(
            tpr,
            fpr,
            linewidth=2.2,
            color=colors.get(name),
            label=f"{name} (AUC={auc:.3f})",
        )
    axis.set_yscale("log")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(1e-5, 1.0)
    axis.set_xlabel("Signal efficiency")
    axis.set_ylabel("Background efficiency")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_dir / "roc_comparison.png", dpi=200)
    figure.savefig(output_dir / "roc_comparison.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 6))
    for name, auc, fpr, tpr, background_events in curves:
        rejection = 1.0 / np.maximum(fpr, 1.0 / max(background_events, 1))
        axis.plot(
            tpr,
            rejection,
            linewidth=2.2,
            color=colors.get(name),
            label=f"{name} (AUC={auc:.3f})",
        )
    axis.set_yscale("log")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(1.0, None)
    axis.set_xlabel("Signal efficiency")
    axis.set_ylabel("Background rejection")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_dir / "background_rejection_comparison.png", dpi=200)
    figure.savefig(output_dir / "background_rejection_comparison.pdf")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if (args.q1_data_only_checkpoint is None) != (
        args.q1_data_only_prepared_dir is None
    ):
        raise ValueError(
            "Provide both --q1-data-only-checkpoint and "
            "--q1-data-only-prepared-dir, or neither"
        )
    if (args.q8_data_only_checkpoint is None) != (
        args.q8_data_only_prepared_dir is None
    ):
        raise ValueError(
            "Provide both --q8-data-only-checkpoint and "
            "--q8-data-only-prepared-dir, or neither"
        )
    if (args.continuous_data_only_checkpoint is None) != (
        args.continuous_data_only_prepared_dir is None
    ):
        raise ValueError(
            "Provide both --continuous-data-only-checkpoint and "
            "--continuous-data-only-prepared-dir, or neither"
        )

    q1_name = "Q1 tokens (MC+data)" if args.q1_data_only_checkpoint else "Q1 tokens"
    q8_name = "Q8 tokens (MC+data)" if args.q8_data_only_checkpoint else "Q8 tokens"
    continuous_name = (
        "Continuous (MC+data)"
        if args.continuous_data_only_checkpoint
        else "Continuous"
    )
    specs = [
        RunSpec(q1_name, "q1", args.q1_checkpoint, args.q1_prepared_dir),
    ]
    if args.q1_data_only_checkpoint:
        specs.append(
            RunSpec(
                "Q1 tokens (200M data-only pretrain)",
                "q1",
                args.q1_data_only_checkpoint,
                args.q1_data_only_prepared_dir,
            )
        )
    specs.append(RunSpec(q8_name, "q8", args.q8_checkpoint, args.q8_prepared_dir))
    if args.q8_data_only_checkpoint:
        specs.append(
            RunSpec(
                "Q8 tokens (200M data-only pretrain)",
                "q8",
                args.q8_data_only_checkpoint,
                args.q8_data_only_prepared_dir,
            )
        )
    specs.append(
        RunSpec(
            continuous_name,
            "continuous",
            args.continuous_checkpoint,
            args.continuous_prepared_dir,
        )
    )
    if args.continuous_data_only_checkpoint:
        specs.append(
            RunSpec(
                "Continuous (200M data-only pretrain)",
                "continuous",
                args.continuous_data_only_checkpoint,
                args.continuous_data_only_prepared_dir,
            )
        )
    for spec in specs:
        validate_paths(spec)

    matched_events = verify_matched_events(specs)
    print(f"PASS: all representations contain the same {matched_events:,} test events")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    curves = []
    curve_arrays = {}

    for spec in specs:
        require_continuous = spec.representation == "continuous"
        datamodule = GroupedTokenParquetClassificationModule(
            prepared_dir=str(spec.prepared_dir),
            n_classes=2,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_sequences=args.max_sequences,
            seed=args.seed,
            stream_batch_size=args.stream_batch_size,
            shuffle_buffer_size=8192,
            label_column="label",
            pin_memory=device.type == "cuda",
            persistent_workers=False,
            require_continuous=require_continuous,
            continuous_feature_column=("continuous_features" if require_continuous else None),
        )
        model = load_model(spec, device)
        scores, labels = predict(model, datamodule.test_dataloader(), device)
        fpr, tpr, thresholds = roc_curve(labels, scores)
        auc = float(roc_auc_score(labels, scores))
        background_events = int(np.count_nonzero(labels == 0))
        row = {
            "run": spec.name,
            "representation": spec.representation,
            "events": int(len(labels)),
            "signal_events": int(np.count_nonzero(labels == 1)),
            "background_events": background_events,
            "auc": auc,
            "rejection_at_50pct": rejection_at_efficiency(labels, scores, 0.50),
            "rejection_at_70pct": rejection_at_efficiency(labels, scores, 0.70),
            "rejection_at_80pct": rejection_at_efficiency(labels, scores, 0.80),
            "checkpoint": str(spec.checkpoint.resolve()),
            "prepared_dir": str(spec.prepared_dir.resolve()),
        }
        rows.append(row)
        curves.append((spec.name, auc, fpr, tpr, background_events))
        prefix = safe_name(spec.name)
        curve_arrays[f"{prefix}_fpr"] = fpr
        curve_arrays[f"{prefix}_tpr"] = tpr
        curve_arrays[f"{prefix}_thresholds"] = thresholds
        curve_arrays[f"{prefix}_scores"] = scores
        curve_arrays[f"{prefix}_labels"] = labels
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    np.savez_compressed(args.output_dir / "roc_arrays.npz", **curve_arrays)
    write_roc_plots(args.output_dir, curves)

    print("\nrun                            AUC      R@50%    R@70%    R@80%")
    print("-" * 68)
    for row in rows:
        print(
            f"{row['run']:<30} {row['auc']:.4f}   "
            f"{row['rejection_at_50pct']:8.2f} "
            f"{row['rejection_at_70pct']:8.2f} "
            f"{row['rejection_at_80pct']:8.2f}"
        )
    print(f"\nWrote comparison to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
