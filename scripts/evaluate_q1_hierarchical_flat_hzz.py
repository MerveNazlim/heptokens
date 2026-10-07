#!/usr/bin/env python3
"""Compare Q1, hierarchical-continuous, and flat-continuous HZZ classifiers."""

from __future__ import annotations

import argparse
import csv
import functools
import importlib
import json
import re
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

if TYPE_CHECKING:
    import torch

    from evaluate_q1_q8_continuous_hzz import RunSpec


def install_hydra_checkpoint_compatibility() -> None:
    """Provide the class path embedded by newer Hydra checkpoint metadata."""
    module_name = "hydra._internal.target_policy"
    if module_name in sys.modules:
        return
    try:
        importlib.import_module(module_name)
        return
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise

    import hydra._internal

    compatibility_module = types.ModuleType(module_name)

    class _DeferredTarget(functools.partial):
        _hydra_call_context = None

    _DeferredTarget.__module__ = module_name
    _DeferredTarget.__qualname__ = "_DeferredTarget"
    compatibility_module._DeferredTarget = _DeferredTarget
    sys.modules[module_name] = compatibility_module
    hydra._internal.target_policy = compatibility_module


def load_model(spec: RunSpec, device: torch.device):
    from evaluate_q1_q8_continuous_hzz import load_model as _load_model

    try:
        return _load_model(spec, device)
    except ModuleNotFoundError as error:
        if error.name != "hydra._internal.target_policy":
            raise
        install_hydra_checkpoint_compatibility()
        return _load_model(spec, device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q1-checkpoint", type=Path)
    parser.add_argument("--q1-prepared-dir", type=Path)
    parser.add_argument("--hierarchical-checkpoint", type=Path)
    parser.add_argument("--flat-checkpoint", type=Path)
    parser.add_argument("--continuous-prepared-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--replot-dir",
        type=Path,
        help="Replot saved summary.json and roc_arrays.npz without inference",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--stream-batch-size", type=int, default=4096)
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoints-only", action="store_true")
    parser.add_argument("--device")
    args = parser.parse_args()
    if args.replot_dir is not None:
        if args.checkpoints_only:
            parser.error("--replot-dir cannot be combined with --checkpoints-only")
    else:
        required = (
            "q1_checkpoint", "q1_prepared_dir", "hierarchical_checkpoint",
            "flat_checkpoint", "continuous_prepared_dir", "output_dir",
        )
        missing = ["--" + name.replace("_", "-") for name in required
                   if getattr(args, name) is None]
        if missing:
            parser.error("required for evaluation: " + ", ".join(missing))
    return args


def write_plots(output_dir: Path, curves: list[dict]) -> None:
    # Model imports previously applied ROOT styling; replotting skips those imports.
    plt.style.use(hep.style.ROOT)
    colors = {
        "Q1 tokens": "#0072B2",
        "Hierarchical continuous": "#CC79A7",
        "Flat continuous": "#009E73",
    }

    figure, axis = plt.subplots(figsize=(8, 6))
    for curve in curves:
        axis.plot(
            curve["tpr"],
            curve["fpr"],
            linewidth=2.2,
            color=curve.get("color", colors.get(curve["name"])),
            linestyle=curve.get("linestyle", "-"),
            label=f"{curve['name']} (AUC={curve['auc']:.3f})",
        )
    axis.set_yscale("log")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(1e-5, 1.0)
    axis.set_xlabel("Signal efficiency")
    axis.set_ylabel("Background efficiency")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(
        loc="upper left", fontsize=12, frameon=False, labelspacing=0.4,
    )
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"background_efficiency.{suffix}", dpi=240)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 6))
    for curve in curves:
        floor = 1.0 / max(curve["background_events"], 1)
        rejection = 1.0 / np.maximum(curve["fpr"], floor)
        axis.plot(
            curve["tpr"],
            rejection,
            linewidth=2.2,
            color=curve.get("color", colors.get(curve["name"])),
            linestyle=curve.get("linestyle", "-"),
            label=f"{curve['name']} (AUC={curve['auc']:.3f})",
        )
    axis.set_yscale("log")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(1.0, None)
    axis.set_xlabel("Signal efficiency")
    axis.set_ylabel("Background rejection")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(
        loc="upper right", fontsize=12, frameon=False, labelspacing=0.4,
    )
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"background_rejection.{suffix}", dpi=240)
    plt.close(figure)


def replot_saved_results(source_dir: Path, output_dir: Path) -> None:
    rows = json.loads((source_dir / "summary.json").read_text())
    if not rows:
        raise ValueError("Saved summary contains no runs")
    curves = []
    with np.load(source_dir / "roc_arrays.npz", allow_pickle=False) as arrays:
        for row in rows:
            prefix = re.sub(r"[^A-Za-z0-9]+", "_", row["run"]).strip("_").lower()
            curves.append({
                "name": row["run"],
                "auc": row["auc"],
                "fpr": arrays[f"{prefix}_fpr"],
                "tpr": arrays[f"{prefix}_tpr"],
                "background_events": int(np.count_nonzero(arrays[f"{prefix}_labels"] == 0)),
            })
    output_dir.mkdir(parents=True, exist_ok=True)
    write_plots(output_dir, curves)
    print(f"Replotted saved results to {output_dir.resolve()} (no inference)")


def main() -> None:
    args = parse_args()
    if args.replot_dir is not None:
        replot_saved_results(args.replot_dir, args.output_dir or args.replot_dir)
        return

    import torch
    from sklearn.metrics import roc_auc_score, roc_curve

    from evaluate_q1_q8_continuous_hzz import (
        RunSpec, predict, rejection_at_efficiency, safe_name, validate_paths,
        verify_matched_events,
    )
    from heptokens.data.benchmark_token_parquet import (
        GroupedTokenParquetClassificationModule,
    )

    print(f"Evaluator: hydra-compat-v2 ({Path(__file__).resolve()})", flush=True)
    install_hydra_checkpoint_compatibility()
    specs = [
        RunSpec("Q1 tokens", "q1", args.q1_checkpoint, args.q1_prepared_dir),
        RunSpec(
            "Hierarchical continuous",
            "continuous",
            args.hierarchical_checkpoint,
            args.continuous_prepared_dir,
        ),
        RunSpec(
            "Flat continuous",
            "continuous",
            args.flat_checkpoint,
            args.continuous_prepared_dir,
        ),
    ]
    for spec in specs:
        validate_paths(spec)
    if args.checkpoints_only:
        for spec in specs:
            model = load_model(spec, torch.device("cpu"))
            print(f"PASS: loaded {spec.name}: {spec.checkpoint}", flush=True)
            del model
        return
    matched_events = verify_matched_events(specs)
    print(f"PASS: all representations contain the same {matched_events:,} test events")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    curves = []
    arrays = {}
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
            continuous_feature_column=(
                "continuous_features" if require_continuous else None
            ),
        )
        model = load_model(spec, device)
        scores, labels = predict(model, datamodule.test_dataloader(), device)
        fpr, tpr, thresholds = roc_curve(labels, scores)
        run_auc = float(roc_auc_score(labels, scores))
        background_events = int(np.count_nonzero(labels == 0))
        row = {
            "run": spec.name,
            "events": int(labels.size),
            "auc": run_auc,
            "rejection_at_50pct": rejection_at_efficiency(labels, scores, 0.50),
            "rejection_at_70pct": rejection_at_efficiency(labels, scores, 0.70),
            "rejection_at_80pct": rejection_at_efficiency(labels, scores, 0.80),
            "checkpoint": str(spec.checkpoint.resolve()),
            "prepared_dir": str(spec.prepared_dir.resolve()),
        }
        rows.append(row)
        curves.append(
            {
                "name": spec.name,
                "auc": run_auc,
                "fpr": fpr,
                "tpr": tpr,
                "background_events": background_events,
            }
        )
        prefix = safe_name(spec.name)
        arrays[f"{prefix}_fpr"] = fpr
        arrays[f"{prefix}_tpr"] = tpr
        arrays[f"{prefix}_thresholds"] = thresholds
        arrays[f"{prefix}_scores"] = scores
        arrays[f"{prefix}_labels"] = labels
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    np.savez_compressed(args.output_dir / "roc_arrays.npz", **arrays)
    write_plots(args.output_dir, curves)

    print("\nrun                         AUC      R@50%    R@70%    R@80%")
    print("-" * 66)
    for row in rows:
        print(
            f"{row['run']:<28} {row['auc']:.4f}   "
            f"{row['rejection_at_50pct']:8.2f} "
            f"{row['rejection_at_70pct']:8.2f} "
            f"{row['rejection_at_80pct']:8.2f}"
        )
    print(f"\nWrote comparison to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
