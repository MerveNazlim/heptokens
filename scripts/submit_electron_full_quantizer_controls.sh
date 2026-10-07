#!/usr/bin/env python3
"""Focused comparison of full-feature electron quantizer controls."""

from __future__ import annotations

import argparse
import json
import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

import compare_electron_domain_controls as domain


DEFAULT_MC_ONLY_RUN = (
    "results/atlas_event_tokenizers_1806_nonjet_logstd_capacity_scan/"
    "electrons_logstd_dim8_cb4096_q4"
)
DEFAULT_Q4_BASELINE_RUN = (
    "results/atlas_event_tokenizers_0107_logstd_mc_realdata/"
    "electrons_logstd_dim8_cb4096_q4"
)
DEFAULT_Q6_RUN = (
    "results/atlas_electron_full_quantizer_controls/"
    "electrons_full_dim8_cb4096_q6_e20_mcdata"
)
DEFAULT_Q6_2K_RUN = (
    "results/atlas_electron_full_quantizer_controls/"
    "electrons_full_dim8_cb2048_q6_e20_mcdata"
)
DEFAULT_Q8_RUN = (
    "results/atlas_electron_full_training_controls/"
    "electrons_full_dim8_cb4096_q8_e20_mcdata"
)
DEFAULT_Q8_2K_RUN = (
    "results/atlas_electron_full_quantizer_controls/"
    "electrons_full_dim8_cb2048_q8_e20_mcdata"
)
DEFAULT_NEW_SAMPLES_MCDATA_RUN = (
    "results/atlas_electron_new_samples_mcdata/"
    "electrons_full_dim8_cb4096_q4_e20_mcdata_new_samples_offline"
)
DEFAULT_NEW_SAMPLES_MC_ONLY_RUN = "results/atlas_electron_new_samples_mconly"

LINE_RESOLUTION_FEATURES = ("pt", "ptvarcone30", "topoetcone20")
CATEGORICAL_FEATURES = ("LHMedium", "LHTight")


def plot_feature_metric_bars(
    *,
    test_title: str,
    feature_name: str,
    samples: OrderedDict[str, OrderedDict[str, dict]],
    metrics_rows: list[dict],
    output_path: Path,
) -> None:
    fig, axes = domain.plt.subplots(
        1,
        len(samples),
        figsize=(7.0 * len(samples), 5.0),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    width = 0.36

    for ax, sample_label in zip(axes, samples):
        labels = list(samples[sample_label])
        x = np.arange(len(labels))
        mae = []
        rmse = []
        for model_label in labels:
            selected = [
                row
                for row in metrics_rows
                if row["sample"] == sample_label
                and row["model"] == model_label
                and row["feature"] == feature_name
            ]
            if selected:
                mae.append(float(selected[0]["mae"]))
                rmse.append(float(selected[0]["rmse"]))
            else:
                mae.append(np.nan)
                rmse.append(np.nan)

        ax.bar(x - width / 2, mae, width=width, label="MAE", color="#4C83F1")
        ax.bar(x + width / 2, rmse, width=width, label="RMSE", color="#F28E2B")
        ax.set_title(sample_label, fontsize=18)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_xlabel("model", fontsize=13)
        ax.grid(axis="y", alpha=0.22)
        domain.apply_hep_style(ax)

    axes[0].set_ylabel(f"{feature_name} reconstruction error", fontsize=13)
    axes[-1].legend(frameon=False, fontsize=9)
    fig.suptitle(f"{test_title}: {feature_name}", fontsize=18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    domain.plt.close(fig)


def resolve_run_dir(path: str | Path, label: str) -> Path:
    """Resolve either a run directory or a project directory containing runs."""
    run_dir = Path(path).resolve()
    if (run_dir / "full_config.yaml").exists():
        return run_dir

    if not run_dir.exists():
        raise FileNotFoundError(f"Missing completed run for {label}: {run_dir}")

    candidates = sorted(
        {config.parent for config in run_dir.rglob("full_config.yaml")},
        key=lambda candidate: (candidate.stat().st_mtime, str(candidate)),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"Could not find full_config.yaml under {run_dir} for {label}"
        )
    if len(candidates) > 1:
        logging.warning(
            "Found %d possible run directories under %s for %s; using newest: %s",
            len(candidates),
            run_dir,
            label,
            candidates[0],
        )
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare full-feature electron tokenizer quantizer controls: "
            "q4 baseline, q6, q6/cb2048, q8, q8/cb2048, MC-only full, "
            "and new-sample runs."
        )
    )
    parser.add_argument("--mc-only-run", default=DEFAULT_MC_ONLY_RUN)
    parser.add_argument("--q4-baseline-run", default=DEFAULT_Q4_BASELINE_RUN)
    parser.add_argument("--q6-run", default=DEFAULT_Q6_RUN)
    parser.add_argument("--q6-2k-run", default=DEFAULT_Q6_2K_RUN)
    parser.add_argument("--q8-run", default=DEFAULT_Q8_RUN)
    parser.add_argument("--q8-2k-run", default=DEFAULT_Q8_2K_RUN)
    parser.add_argument("--new-samples-mcdata-run", default=DEFAULT_NEW_SAMPLES_MCDATA_RUN)
    parser.add_argument(
        "--new-samples-mconly-run",
        default=DEFAULT_NEW_SAMPLES_MC_ONLY_RUN,
        help=(
            "New-sample MC-only run directory. This may be either the exact run "
            "directory or a project directory containing one or more run directories."
        ),
    )
    parser.add_argument(
        "--mc-dir", default="/home/zephyr/Data/viviana/bnl-treasure/data/h5"
    )
    parser.add_argument(
        "--data-dir",
        default="/home/zephyr/Data/viviana/bnl-treasure/data/h5/realdata",
    )
    parser.add_argument(
        "--output-dir",
        default="results/atlas_electron_full_quantizer_controls/comparison",
    )
    parser.add_argument("--n-files", type=int, default=20)
    parser.add_argument("--num-events-per-file", type=int)
    parser.add_argument("--max-valid-objects", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--checkpoint-name", default="last.ckpt")
    parser.add_argument("--n-bins", type=int, default=14)
    parser.add_argument("--min-bin-count", type=int, default=50)
    parser.add_argument("--min-denominator", type=float, default=1e-8)
    parser.add_argument("--force", action="store_true", help="Ignore cached arrays.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = domain.choose_device(args.device)
    files_by_sample = domain.sample_files(args)

    run_dirs = OrderedDict(
        [
            ("MC-only (full)", resolve_run_dir(args.mc_only_run, "MC-only (full)")),
            (
                "MC+data (q4 baseline)",
                resolve_run_dir(args.q4_baseline_run, "MC+data (q4 baseline)"),
            ),
            ("MC+data (q6)", resolve_run_dir(args.q6_run, "MC+data (q6)")),
            (
                "MC+data (q6, cb2048)",
                resolve_run_dir(args.q6_2k_run, "MC+data (q6, cb2048)"),
            ),
            ("MC+data (q8)", resolve_run_dir(args.q8_run, "MC+data (q8)")),
            (
                "MC+data (q8, cb2048)",
                resolve_run_dir(args.q8_2k_run, "MC+data (q8, cb2048)"),
            ),
            (
                "MC+data new samples (q4)",
                resolve_run_dir(
                    args.new_samples_mcdata_run,
                    "MC+data new samples (q4)",
                ),
            ),
            (
                "MC-only new samples (q4)",
                resolve_run_dir(
                    args.new_samples_mconly_run,
                    "MC-only new samples (q4)",
                ),
            ),
        ]
    )

    domain.COLORS.update(
        {
            "MC+data (q4 baseline)": "#F28E2B",
            "MC+data (q6)": "#B45F06",
            "MC+data (q6, cb2048)": "#E15759",
            "MC+data (q8)": "#CC79A7",
            "MC+data (q8, cb2048)": "#009E73",
            "MC+data new samples (q4)": "#7A5195",
            "MC-only new samples (q4)": "#2F4B7C",
        }
    )
    domain.LINESTYLES.update(
        {
            "MC+data (q6)": "--",
            "MC+data (q6, cb2048)": (0, (6, 2, 1, 2)),
            "MC+data (q8)": ":",
            "MC+data (q8, cb2048)": "-.",
            "MC+data new samples (q4)": (0, (3, 1, 1, 1)),
            "MC-only new samples (q4)": (0, (5, 2)),
        }
    )

    checkpoints = {}
    for label, run_dir in run_dirs.items():
        if not (run_dir / "full_config.yaml").exists():
            raise FileNotFoundError(f"Missing completed run for {label}: {run_dir}")
        checkpoints[label] = str(domain.explicit_checkpoint(run_dir, args.checkpoint_name))

    reference_cfg = OmegaConf.load(run_dirs["MC+data (q4 baseline)"] / "full_config.yaml")
    reference_datamodule_cfg = reference_cfg.datamodule

    all_outputs: OrderedDict[str, OrderedDict[str, dict]] = OrderedDict()
    for sample_label, files in files_by_sample.items():
        all_outputs[sample_label] = OrderedDict()
        for model_label, run_dir in run_dirs.items():
            all_outputs[sample_label][model_label] = domain.collect_one(
                sample_label=sample_label,
                model_label=model_label,
                run_dir=run_dir,
                reference_datamodule_cfg=reference_datamodule_cfg,
                files=files,
                output_dir=output_dir,
                args=args,
                device=device,
            )

    test_key = "full_quantizer_controls"
    test_title = "Electron full-feature quantizer controls"
    model_labels = list(run_dirs)

    manifest = {
        "runs": {label: str(path) for label, path in run_dirs.items()},
        "checkpoints": checkpoints,
        "samples": files_by_sample,
        "split": args.split,
        "max_valid_objects": args.max_valid_objects,
        "num_events_per_file": args.num_events_per_file,
        "binned_resolution_features": list(domain.BINNED_RESOLUTION_FEATURES),
        "metric": (
            "IQR(reco-original) / abs(median(original)) "
            "in fixed original-value bins"
        ),
        "encode_path": "canonical model.encode via collect_diagnostics_from_loader",
    }
    (output_dir / "comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    selected_samples: OrderedDict[str, OrderedDict[str, dict]] = OrderedDict()
    for sample_label, outputs in all_outputs.items():
        selected_samples[sample_label] = OrderedDict(
            (label, outputs[label]) for label in model_labels
        )

    first_models = next(iter(selected_samples.values()))
    shared_features = domain.common_feature_names(*first_models.values())
    shared_feature_keys = [name.lower() for name in shared_features]
    if "pt" not in shared_feature_keys:
        raise RuntimeError("shared feature list has no pT")

    metrics_rows, resolution_rows = domain.evaluate_test(
        test_key=test_key,
        test_title=test_title,
        samples=selected_samples,
        feature_names=shared_features,
        args=args,
    )

    domain.write_csv(output_dir / f"{test_key}_metrics.csv", metrics_rows)
    domain.write_csv(output_dir / f"{test_key}_binned_resolution.csv", resolution_rows)
    for feature_name in domain.BINNED_RESOLUTION_FEATURES:
        if feature_name.lower() not in shared_feature_keys:
            continue
        if feature_name in CATEGORICAL_FEATURES:
            stale_path = output_dir / f"{test_key}_{feature_name}_resolution.png"
            stale_csv_path = output_dir / f"{test_key}_{feature_name}_binned_resolution.csv"
            if stale_path.exists():
                stale_path.unlink()
            if stale_csv_path.exists():
                stale_csv_path.unlink()
            plot_feature_metric_bars(
                test_title=test_title,
                feature_name=feature_name,
                samples=selected_samples,
                metrics_rows=metrics_rows,
                output_path=output_dir / f"{test_key}_{feature_name}_mae_rmse.png",
            )
            continue
        if feature_name not in LINE_RESOLUTION_FEATURES:
            continue
        feature_rows = [row for row in resolution_rows if row["feature"] == feature_name]
        domain.write_csv(
            output_dir / f"{test_key}_{feature_name}_binned_resolution.csv",
            feature_rows,
        )
        domain.plot_feature_resolution(
            test_title=test_title,
            feature_name=feature_name,
            samples=selected_samples,
            resolution_rows=feature_rows,
            output_path=output_dir / f"{test_key}_{feature_name}_resolution.png",
        )

    report_lines = domain.report_section(
        title=test_title,
        models=model_labels,
        metrics_rows=metrics_rows,
        resolution_rows=resolution_rows,
    )
    (output_dir / "comparison_report.md").write_text(
        "\n".join(report_lines).rstrip() + "\n"
    )
    logging.info("Wrote full quantizer-control comparison to %s", output_dir)


if __name__ == "__main__":
    main()
