#!/usr/bin/env python3
"""Compare electron-tokenizer sampling studies on fixed MC and data objects.

Every checkpoint is evaluated with the same ordered H5 files, preprocessing,
official ``model.encode`` path, and cumulative four-quantizer reconstruction.
The script verifies that the physical inputs are identical before comparing
reconstruction quality or codebook use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import OrderedDict
from pathlib import Path

import hydra
import matplotlib
import numpy as np
import torch
from omegaconf import OmegaConf

from analyze_vqvae_tokenizer import (
    analysis_datamodule_cfg,
    apply_hep_style,
    binned_residual_iqr_over_median_truth,
    choose_device,
    codebook_counts,
    codebook_summary,
    collect_diagnostics_from_loader,
    dataloader_from_datamodule,
    feature_names_from_cfg,
    find_checkpoint,
    reconstruction_metrics,
    safe_filename,
    transform_list_and_cst_fn_from_cfg,
)
from heptokens.models.vq_vae import LitVqVae

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


log = logging.getLogger(__name__)


DEFAULT_MODELS = OrderedDict(
    [
        (
            "MC-only reference",
            "results/atlas_event_tokenizers_1806_nonjet_logstd_capacity_scan/"
            "electrons_logstd_dim8_cb4096_q4",
        ),
        (
            "Natural MC+data",
            "results/atlas_electron_sampling_study/"
            "electrons_logstd_dim8_cb4096_q4_natural",
        ),
        (
            "Balanced 50/50",
            "results/atlas_electron_sampling_study/"
            "electrons_logstd_dim8_cb4096_q4_balanced50",
        ),
        (
            "MC-heavy 75/25",
            "results/atlas_electron_sampling_study/"
            "electrons_logstd_dim8_cb4096_q4_mcheavy75",
        ),
    ]
)

COLORS = {
    "MC-only reference": "#3F7FE5",
    "Natural MC+data": "#F28E2B",
    "Balanced 50/50": "#2CA58D",
    "MC-heavy 75/25": "#8E5BD9",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare completed electron domain-sampling tokenizer runs."
    )
    parser.add_argument(
        "--mc-dir", default="/home/zephyr/Data/viviana/bnl-treasure/data/h5"
    )
    parser.add_argument(
        "--data-dir",
        default="/home/zephyr/Data/viviana/bnl-treasure/data/h5/realdata",
    )
    parser.add_argument(
        "--output-dir", default="results/atlas_electron_sampling_study/comparison"
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="LABEL=RUN_DIR",
        help="Override defaults with a labelled run; repeat for every model.",
    )
    parser.add_argument("--samples", nargs="+", choices=["mc", "realdata"], default=["mc", "realdata"])
    parser.add_argument("--n-files", type=int, default=20)
    parser.add_argument("--num-events-per-file", type=int)
    parser.add_argument("--max-valid-objects", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--n-bins", type=int, default=14)
    parser.add_argument("--min-bin-count", type=int, default=50)
    parser.add_argument("--min-denominator", type=float, default=1e-8)
    parser.add_argument("--force", action="store_true", help="Ignore cached arrays.")
    return parser.parse_args()


def parse_models(values: list[str]) -> OrderedDict[str, Path]:
    raw = OrderedDict()
    if values:
        for value in values:
            if "=" not in value:
                raise ValueError(f"Expected LABEL=RUN_DIR, got {value!r}")
            label, path = value.split("=", 1)
            raw[label.strip()] = path.strip()
    else:
        raw.update(DEFAULT_MODELS)
    return OrderedDict((label, Path(path).resolve()) for label, path in raw.items())


def fixed_files(directory: str, n_files: int) -> list[str]:
    return [
        str(path)
        for path in sorted(Path(directory).glob("*.h5"))
        if path.is_file() and path.stat().st_size > 0
    ][:n_files]


def sample_files(args: argparse.Namespace) -> OrderedDict[str, list[str]]:
    samples: OrderedDict[str, list[str]] = OrderedDict()
    if "mc" in args.samples:
        samples["MC sample"] = fixed_files(args.mc_dir, args.n_files)
    if "realdata" in args.samples:
        samples["real-data sample"] = fixed_files(args.data_dir, args.n_files)
    for label, files in samples.items():
        if not files:
            raise FileNotFoundError(f"No non-empty H5 files found for {label}")
    return samples


def cache_path(
    output_dir: Path,
    sample_label: str,
    model_label: str,
    run_dir: Path,
    checkpoint: Path,
    files: list[str],
    args: argparse.Namespace,
) -> Path:
    payload = "\n".join(
        [
            str(run_dir),
            str(checkpoint),
            str(checkpoint.stat().st_mtime_ns),
            str(args.num_events_per_file),
            str(args.max_valid_objects),
            f"split={args.split}",
            "analysis_path=shared_reference_datamodule_v2",
            *files,
        ]
    )
    digest = hashlib.sha1(payload.encode()).hexdigest()[:14]
    name = f"{safe_filename(sample_label)}_{safe_filename(model_label)}_{digest}.npz"
    return output_dir / "cache" / name


def collect_one(
    *,
    sample_label: str,
    model_label: str,
    run_dir: Path,
    reference_datamodule_cfg,
    files: list[str],
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    cfg_path = run_dir / "full_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    checkpoint = find_checkpoint(run_dir, None)
    cached = cache_path(
        output_dir, sample_label, model_label, run_dir, checkpoint, files, args
    )
    if cached.exists() and not args.force:
        log.info("Reading cache %s", cached)
        item = np.load(cached, allow_pickle=True)
        return {
            "original": item["original"],
            "reconstruction": item["reconstruction"],
            "indices": item["indices"],
            "feature_names": list(item["feature_names"]),
        }

    cfg = OmegaConf.load(cfg_path)
    _, inverse_transformer = transform_list_and_cst_fn_from_cfg(cfg)

    log.info("Loading %s for %s", checkpoint, sample_label)
    model = LitVqVae.load_from_checkpoint(checkpoint, map_location=device)
    model.to(device)
    model.eval()

    loader_args = argparse.Namespace(
        h5_files=list(files),
        num_events_per_file=args.num_events_per_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    # Build every evaluation loader from one shared datamodule definition so
    # saved run-specific loading caps, masks, and split settings cannot change
    # the evaluated objects. Keep only this run's fitted preprocessing transform.
    evaluation_cfg = OmegaConf.create(
        {
            "datamodule": OmegaConf.to_container(
                reference_datamodule_cfg, resolve=False
            )
        }
    )
    evaluation_cfg.datamodule.transforms = OmegaConf.to_container(
        cfg.datamodule.get("transforms", {}), resolve=False
    )
    datamodule_cfg = analysis_datamodule_cfg(evaluation_cfg, loader_args)
    # A fair fixed-file evaluation must not inherit a run-specific event cap.
    # None means read every event in each selected H5 file.
    datamodule_cfg.num_events = args.num_events_per_file
    datamodule = hydra.utils.instantiate(datamodule_cfg)
    loader = dataloader_from_datamodule(datamodule, args.split)

    original, reconstruction, indices, _ = collect_diagnostics_from_loader(
        model=model,
        loader=loader,
        cst_inverse_transformer=inverse_transformer,
        device=device,
        max_valid_objects=args.max_valid_objects,
    )
    feature_names = feature_names_from_cfg(cfg, original.shape[1])
    cached.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cached,
        original=original,
        reconstruction=reconstruction,
        indices=indices,
        feature_names=np.asarray(feature_names),
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "original": original,
        "reconstruction": reconstruction,
        "indices": indices,
        "feature_names": feature_names,
    }


def verify_shared_inputs(sample_label: str, outputs: OrderedDict[str, dict]) -> None:
    reference_label, reference = next(iter(outputs.items()))
    for label, item in list(outputs.items())[1:]:
        if item["feature_names"] != reference["feature_names"]:
            raise RuntimeError(
                f"{sample_label}: feature order differs for {reference_label} and {label}: "
                f"{reference['feature_names']} vs {item['feature_names']}"
            )
        left = np.asarray(reference["original"], dtype=np.float64)
        right = np.asarray(item["original"], dtype=np.float64)
        if left.shape != right.shape:
            raise RuntimeError(
                f"{sample_label}: object shapes differ for {reference_label} and {label}: "
                f"{left.shape} vs {right.shape}"
            )
        finite = np.isfinite(left) & np.isfinite(right)
        max_delta = float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else 0.0
        same_nonfinite = np.array_equal(np.isfinite(left), np.isfinite(right))
        if max_delta > 1e-6 or not same_nonfinite:
            raise RuntimeError(
                f"{sample_label}: checkpoints did not receive identical ordered objects; "
                f"max physical-input delta={max_delta:.6g}"
            )
    log.info("%s: verified identical ordered inputs across %d models", sample_label, len(outputs))


def make_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    lo, hi = np.percentile(values, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(values.min()), float(values.max())
    if lo == hi:
        raise ValueError("Cannot define bins for a constant feature")
    return np.linspace(lo, hi, n_bins + 1)


def model_color(label: str, index: int) -> str:
    fallback = ["#3F7FE5", "#F28E2B", "#2CA58D", "#8E5BD9", "#D64B8C"]
    return COLORS.get(label, fallback[index % len(fallback)])


def summarize(
    samples: OrderedDict[str, OrderedDict[str, dict]],
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    resolution_rows: list[dict] = []
    for sample_label, models in samples.items():
        reference = next(iter(models.values()))
        feature_names = reference["feature_names"]
        pt_idx = next(i for i, name in enumerate(feature_names) if name.lower() == "pt")
        bins = make_bins(reference["original"][:, pt_idx], args.n_bins)
        for model_label, item in models.items():
            metrics = reconstruction_metrics(
                item["original"], item["reconstruction"], feature_names
            )
            code_size = int(np.max(item["indices"])) + 1
            # All study runs use 4096 codes. Preserve that denominator even if
            # the fixed evaluation sample does not reach the highest code ID.
            code_size = max(code_size, 4096)
            usage = codebook_summary(codebook_counts(item["indices"], code_size))
            used_values = [
                float(value["percent_used"])
                for key, value in usage.items()
                if key.startswith("quantizer_")
            ]
            centers, values, counts, _, _ = binned_residual_iqr_over_median_truth(
                item["original"][:, pt_idx],
                item["reconstruction"][:, pt_idx],
                bins,
                min_bin_count=args.min_bin_count,
                min_denominator=args.min_denominator,
            )
            for center, value, count in zip(centers, values, counts):
                resolution_rows.append(
                    {
                        "sample": sample_label,
                        "model": model_label,
                        "pt_bin_center": float(center),
                        "relative_resolution": float(value),
                        "objects": int(count),
                    }
                )
            for feature_name, feature_metrics in metrics.items():
                rows.append(
                    {
                        "sample": sample_label,
                        "model": model_label,
                        "feature": feature_name,
                        "n": int(feature_metrics["n"]),
                        "mae": float(feature_metrics["mae"]),
                        "rmse": float(feature_metrics["rmse"]),
                        "bias": float(feature_metrics["bias"]),
                        "mean_codebook_used_percent": float(np.mean(used_values)),
                        **{
                            f"q{q}_used_percent": float(usage[f"quantizer_{q}"]["percent_used"])
                            for q in range(len(usage))
                        },
                    }
                )
    return rows, resolution_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_pt_resolution(
    samples: OrderedDict[str, OrderedDict[str, dict]],
    resolution_rows: list[dict],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, len(samples), figsize=(6.5 * len(samples), 5.2), sharey=True)
    axes = np.atleast_1d(axes)
    model_labels = list(next(iter(samples.values())))
    for ax, sample_label in zip(axes, samples):
        for model_idx, model_label in enumerate(model_labels):
            selected = [
                row
                for row in resolution_rows
                if row["sample"] == sample_label and row["model"] == model_label
            ]
            x = np.asarray([row["pt_bin_center"] for row in selected])
            y = np.asarray([row["relative_resolution"] for row in selected])
            valid = np.isfinite(y)
            ax.plot(
                x[valid],
                y[valid],
                marker="o",
                linewidth=2.0,
                label=model_label,
                color=model_color(model_label, model_idx),
            )
        ax.set_title(sample_label, fontsize=18)
        ax.set_xlabel(r"Original electron $p_T$", fontsize=14)
        ax.grid(alpha=0.22)
        apply_hep_style(ax)
    axes[0].set_ylabel(r"IQR(reco - original) / |median(original)|", fontsize=14)
    axes[-1].legend(frameon=False, fontsize=10)
    fig.suptitle(r"Electron $p_T$ reconstruction by training mixture", fontsize=18)
    fig.tight_layout()
    fig.savefig(output_dir / "electron_pt_resolution_sampling_comparison.png", dpi=200)
    plt.close(fig)


def plot_global_pt_metrics(rows: list[dict], output_dir: Path) -> None:
    pt_rows = [row for row in rows if row["feature"].lower() == "pt"]
    samples = list(OrderedDict.fromkeys(row["sample"] for row in pt_rows))
    models = list(OrderedDict.fromkeys(row["model"] for row in pt_rows))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    width = 0.8 / len(models)
    x = np.arange(len(samples), dtype=float)
    for model_idx, model in enumerate(models):
        selected = {
            row["sample"]: row for row in pt_rows if row["model"] == model
        }
        offset = (model_idx - (len(models) - 1) / 2) * width
        for ax, metric in zip(axes, ["mae", "rmse"]):
            ax.bar(
                x + offset,
                [selected[sample][metric] for sample in samples],
                width=width,
                label=model,
                color=model_color(model, model_idx),
            )
    for ax, metric in zip(axes, ["MAE", "RMSE"]):
        ax.set_xticks(x, samples)
        ax.set_ylabel(rf"Electron $p_T$ {metric}")
        ax.grid(axis="y", alpha=0.22)
        apply_hep_style(ax)
    axes[-1].legend(frameon=False, fontsize=9)
    fig.suptitle(r"Global electron $p_T$ reconstruction", fontsize=18)
    fig.tight_layout()
    fig.savefig(output_dir / "electron_pt_global_metrics_sampling_comparison.png", dpi=200)
    plt.close(fig)


def plot_codebook_usage(rows: list[dict], output_dir: Path) -> None:
    pt_rows = [row for row in rows if row["feature"].lower() == "pt"]
    samples = list(OrderedDict.fromkeys(row["sample"] for row in pt_rows))
    models = list(OrderedDict.fromkeys(row["model"] for row in pt_rows))
    fig, axes = plt.subplots(1, len(samples), figsize=(6.5 * len(samples), 5.0), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(4, dtype=float)
    width = 0.8 / len(models)
    for ax, sample in zip(axes, samples):
        for model_idx, model in enumerate(models):
            row = next(
                item for item in pt_rows if item["sample"] == sample and item["model"] == model
            )
            offset = (model_idx - (len(models) - 1) / 2) * width
            ax.bar(
                x + offset,
                [row[f"q{q}_used_percent"] for q in range(4)],
                width=width,
                label=model,
                color=model_color(model, model_idx),
            )
        ax.set_xticks(x, [f"q{q}" for q in range(4)])
        ax.set_title(sample, fontsize=18)
        ax.set_xlabel("Residual quantizer stage")
        ax.grid(axis="y", alpha=0.22)
        apply_hep_style(ax)
    axes[0].set_ylabel("Codebook used [%]")
    axes[-1].legend(frameon=False, fontsize=9)
    fig.suptitle("Electron codebook usage on fixed evaluation objects", fontsize=18)
    fig.tight_layout()
    fig.savefig(output_dir / "electron_codebook_usage_sampling_comparison.png", dpi=200)
    plt.close(fig)


def write_report(
    rows: list[dict], resolution_rows: list[dict], output_dir: Path
) -> None:
    pt_rows = [row for row in rows if row["feature"].lower() == "pt"]
    lines = ["# Electron sampling-study comparison", ""]
    for sample in OrderedDict.fromkeys(row["sample"] for row in pt_rows):
        sample_rows = [row for row in pt_rows if row["sample"] == sample]
        natural = next(
            (row for row in sample_rows if row["model"] == "Natural MC+data"),
            None,
        )
        binned_means = {}
        for model in OrderedDict.fromkeys(row["model"] for row in sample_rows):
            values = np.asarray(
                [
                    row["relative_resolution"]
                    for row in resolution_rows
                    if row["sample"] == sample and row["model"] == model
                ],
                dtype=np.float64,
            )
            binned_means[model] = float(np.mean(values[np.isfinite(values)]))

        lines.extend(
            [
                f"## {sample}",
                "",
                "| model | pT MAE | vs natural | pT RMSE | mean binned pT resolution | mean codebook used | q0 | q1 | q2 | q3 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sample_rows:
            if natural is None:
                delta_text = "-"
            else:
                delta = 100.0 * (row["mae"] / natural["mae"] - 1.0)
                delta_text = f"{delta:+.1f}%"
            lines.append(
                "| {model} | {mae:.5g} | {delta} | {rmse:.5g} | {binned:.5g} | {mean:.2f}% | "
                "{q0:.2f}% | {q1:.2f}% | {q2:.2f}% | {q3:.2f}% |".format(
                    model=row["model"],
                    mae=row["mae"],
                    delta=delta_text,
                    rmse=row["rmse"],
                    binned=binned_means[row["model"]],
                    mean=row["mean_codebook_used_percent"],
                    q0=row["q0_used_percent"],
                    q1=row["q1_used_percent"],
                    q2=row["q2_used_percent"],
                    q3=row["q3_used_percent"],
                )
            )
        best_mae = min(sample_rows, key=lambda row: row["mae"])
        best_rmse = min(sample_rows, key=lambda row: row["rmse"])
        best_binned = min(sample_rows, key=lambda row: binned_means[row["model"]])
        lines.extend(
            [
                "",
                f"Best pT MAE: **{best_mae['model']}** ({best_mae['mae']:.5g})  ",
                f"Best pT RMSE: **{best_rmse['model']}** ({best_rmse['rmse']:.5g})  ",
                f"Best mean binned pT resolution: **{best_binned['model']}** "
                f"({binned_means[best_binned['model']]:.5g})",
                "",
            ]
        )
    (output_dir / "comparison_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    models = parse_models(args.model)
    samples = sample_files(args)
    device = choose_device(args.device)

    for label, run_dir in models.items():
        if not (run_dir / "full_config.yaml").exists():
            raise FileNotFoundError(f"Missing completed run for {label}: {run_dir}")
        find_checkpoint(run_dir, None)

    reference_run_dir = next(iter(models.values()))
    reference_cfg = OmegaConf.load(reference_run_dir / "full_config.yaml")
    reference_datamodule_cfg = reference_cfg.datamodule

    manifest = {
        "models": {label: str(path) for label, path in models.items()},
        "samples": samples,
        "max_valid_objects": args.max_valid_objects,
        "num_events_per_file": args.num_events_per_file,
        "metric": "IQR(reco-original) / abs(median(original)) in fixed original-value bins",
    }
    (output_dir / "comparison_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    outputs: OrderedDict[str, OrderedDict[str, dict]] = OrderedDict()
    for sample_label, files in samples.items():
        log.info("Evaluating %s from %d files", sample_label, len(files))
        outputs[sample_label] = OrderedDict()
        for model_label, run_dir in models.items():
            outputs[sample_label][model_label] = collect_one(
                sample_label=sample_label,
                model_label=model_label,
                run_dir=run_dir,
                reference_datamodule_cfg=reference_datamodule_cfg,
                files=files,
                output_dir=output_dir,
                args=args,
                device=device,
            )
        verify_shared_inputs(sample_label, outputs[sample_label])

    rows, resolution_rows = summarize(outputs, args, output_dir)
    write_csv(output_dir / "all_feature_metrics.csv", rows)
    write_csv(output_dir / "pt_binned_resolution.csv", resolution_rows)
    write_report(rows, resolution_rows, output_dir)
    plot_pt_resolution(outputs, resolution_rows, output_dir)
    plot_global_pt_metrics(rows, output_dir)
    plot_codebook_usage(rows, output_dir)
    log.info("Wrote comparison to %s", output_dir)


if __name__ == "__main__":
    main()
