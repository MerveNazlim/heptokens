#!/usr/bin/env python3
"""Compare old-sample and new-sample full electron tokenizers in one plot.

Unlike the strict fair-comparison scripts, this intentionally evaluates old
training runs on the old H5 samples and new training runs on the new H5 samples,
then overlays the results in the same MC/data panels. This is useful for
checking whether the new-sample trainings behave similarly to the old-sample
baselines and capacity controls.
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import json
import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

import compare_electron_domain_controls as domain
from compare_electron_full_quantizer_controls import (
    CATEGORICAL_FEATURES,
    LINE_RESOLUTION_FEATURES,
    DEFAULT_MC_ONLY_RUN,
    DEFAULT_NEW_SAMPLES_MCDATA_RUN,
    DEFAULT_NEW_SAMPLES_MC_ONLY_RUN,
    DEFAULT_Q4_BASELINE_RUN,
    DEFAULT_Q6_RUN,
    DEFAULT_Q8_2K_RUN,
    DEFAULT_Q8_RUN,
    plot_feature_metric_bars,
    resolve_run_dir,
)


log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay old-sample and new-sample full-feature electron tokenizer "
            "controls. Old runs are evaluated on old H5 files; new runs are "
            "evaluated on new H5 files."
        )
    )
    parser.add_argument("--old-mc-only-run", default=DEFAULT_MC_ONLY_RUN)
    parser.add_argument("--old-mcdata-run", default=DEFAULT_Q4_BASELINE_RUN)
    parser.add_argument("--old-q6-run", default=DEFAULT_Q6_RUN)
    parser.add_argument("--old-q8-run", default=DEFAULT_Q8_RUN)
    parser.add_argument("--old-q8-2k-run", default=DEFAULT_Q8_2K_RUN)
    parser.add_argument("--new-mcdata-run", default=DEFAULT_NEW_SAMPLES_MCDATA_RUN)
    parser.add_argument("--new-mconly-run", default=DEFAULT_NEW_SAMPLES_MC_ONLY_RUN)
    parser.add_argument(
        "--include-new-mconly",
        action="store_true",
        help="Also include the new-sample MC-only run. Disabled by default.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "old-mconly",
            "old-mcdata",
            "old-q6",
            "old-q8",
            "old-q8-2k",
            "new-mcdata",
        ],
        choices=[
            "old-mconly",
            "old-mcdata",
            "old-q6",
            "old-q8",
            "old-q8-2k",
            "new-mcdata",
            "new-mconly",
        ],
        help="Subset of tokenizer runs to draw.",
    )
    parser.add_argument(
        "--old-mc-dir",
        nargs="+",
        default=["/home/zephyr/Data/viviana/bnl-treasure/data/h5"],
        help="Old MC H5 directory, file, or quoted glob. Multiple values are accepted.",
    )
    parser.add_argument(
        "--old-data-dir",
        nargs="+",
        default=["/home/zephyr/Data/viviana/bnl-treasure/data/h5/realdata"],
        help="Old real-data H5 directory, file, or quoted glob. Multiple values are accepted.",
    )
    parser.add_argument(
        "--new-mc-dir",
        nargs="+",
        default=["/home/zephyr/Data/viviana/bnl-treasure/data_new/h5"],
        help="New MC H5 directory, file, or quoted glob. Multiple values are accepted.",
    )
    parser.add_argument(
        "--new-data-dir",
        nargs="+",
        default=["/home/zephyr/Data/viviana/bnl-treasure/data_new/h5/realdata"],
        help="New real-data H5 directory, file, or quoted glob. Multiple values are accepted.",
    )
    parser.add_argument(
        "--old-mc-exclude",
        nargs="*",
        default=[],
        help="Filename or path glob patterns to exclude from old MC inputs.",
    )
    parser.add_argument(
        "--old-data-exclude",
        nargs="*",
        default=[],
        help="Filename or path glob patterns to exclude from old real-data inputs.",
    )
    parser.add_argument(
        "--new-mc-exclude",
        nargs="*",
        default=[],
        help="Filename or path glob patterns to exclude from new MC inputs.",
    )
    parser.add_argument(
        "--new-data-exclude",
        nargs="*",
        default=[],
        help="Filename or path glob patterns to exclude from new real-data inputs.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/atlas_electron_full_quantizer_controls/old_new_sample_comparison",
    )
    parser.add_argument(
        "--n-files",
        type=int,
        default=20,
        help="Number of files per sample panel. Use 0 or a negative value for all files.",
    )
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


def expand_h5_input(value: str) -> list[Path]:
    if any(char in value for char in "*?[]"):
        candidates = [Path(path) for path in glob.glob(value)]
    else:
        path = Path(value)
        if path.is_dir():
            candidates = list(path.glob("*.h5"))
        elif path.is_file():
            candidates = [path]
        else:
            candidates = []
    return sorted(
        path
        for path in candidates
        if path.is_file() and path.name.endswith(".h5") and path.stat().st_size > 0
    )


def excluded(path: Path, patterns: list[str]) -> bool:
    path_text = str(path)
    return any(
        fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(path_text, pattern)
        for pattern in patterns
    )


def fixed_files(inputs: list[str], n_files: int, exclude_patterns: list[str]) -> list[str]:
    paths: list[Path] = []
    for value in inputs:
        paths.extend(expand_h5_input(value))
    files = [
        str(path)
        for path in sorted(dict.fromkeys(paths))
        if not excluded(path, exclude_patterns)
    ]
    if n_files > 0:
        files = files[:n_files]
    if not files:
        raise FileNotFoundError(f"No non-empty H5 files found in: {inputs}")
    return files


def model_specs(args: argparse.Namespace) -> OrderedDict[str, dict]:
    builders = OrderedDict(
        [
            (
                "old-mconly",
                lambda: (
                    "MC-only old (q4)",
                    {
                        "run_dir": resolve_run_dir(
                            args.old_mc_only_run,
                            "MC-only old (q4)",
                        ),
                        "source": "old",
                    },
                ),
            ),
            (
                "old-mcdata",
                lambda: (
                    "MC+data old (q4)",
                    {
                        "run_dir": resolve_run_dir(
                            args.old_mcdata_run,
                            "MC+data old (q4)",
                        ),
                        "source": "old",
                    },
                ),
            ),
            (
                "old-q6",
                lambda: (
                    "MC+data old (q6)",
                    {
                        "run_dir": resolve_run_dir(args.old_q6_run, "MC+data old (q6)"),
                        "source": "old",
                    },
                ),
            ),
            (
                "old-q8",
                lambda: (
                    "MC+data old (q8)",
                    {
                        "run_dir": resolve_run_dir(args.old_q8_run, "MC+data old (q8)"),
                        "source": "old",
                    },
                ),
            ),
            (
                "old-q8-2k",
                lambda: (
                    "MC+data old (q8, cb2048)",
                    {
                        "run_dir": resolve_run_dir(
                            args.old_q8_2k_run,
                            "MC+data old (q8, cb2048)",
                        ),
                        "source": "old",
                    },
                ),
            ),
            (
                "new-mcdata",
                lambda: (
                    "MC+data new (q4)",
                    {
                        "run_dir": resolve_run_dir(
                            args.new_mcdata_run,
                            "MC+data new (q4)",
                        ),
                        "source": "new",
                    },
                ),
            ),
        ]
    )
    if args.include_new_mconly and "new-mconly" not in args.models:
        args.models.append("new-mconly")
    builders["new-mconly"] = lambda: (
        "MC-only new (q4)",
        {
            "run_dir": resolve_run_dir(args.new_mconly_run, "MC-only new (q4)"),
            "source": "new",
        },
    )

    specs: OrderedDict[str, dict] = OrderedDict()
    for key in args.models:
        label, spec = builders[key]()
        specs[label] = spec
    return specs


def files_by_source(args: argparse.Namespace) -> dict[str, OrderedDict[str, list[str]]]:
    return {
        "old": OrderedDict(
            [
                (
                    "MC sample",
                    fixed_files(args.old_mc_dir, args.n_files, args.old_mc_exclude),
                ),
                (
                    "real-data sample",
                    fixed_files(args.old_data_dir, args.n_files, args.old_data_exclude),
                ),
            ]
        ),
        "new": OrderedDict(
            [
                (
                    "MC sample",
                    fixed_files(args.new_mc_dir, args.n_files, args.new_mc_exclude),
                ),
                (
                    "real-data sample",
                    fixed_files(args.new_data_dir, args.n_files, args.new_data_exclude),
                ),
            ]
        ),
    }


def collect_outputs(
    *,
    specs: OrderedDict[str, dict],
    source_files: dict[str, OrderedDict[str, list[str]]],
    output_dir: Path,
    args: argparse.Namespace,
    device,
) -> OrderedDict[str, OrderedDict[str, dict]]:
    all_outputs: OrderedDict[str, OrderedDict[str, dict]] = OrderedDict(
        [("MC sample", OrderedDict()), ("real-data sample", OrderedDict())]
    )
    for model_label, spec in specs.items():
        run_dir = spec["run_dir"]
        cfg = OmegaConf.load(run_dir / "full_config.yaml")
        reference_datamodule_cfg = cfg.datamodule
        sample_files = source_files[spec["source"]]
        for sample_label, files in sample_files.items():
            all_outputs[sample_label][model_label] = domain.collect_one(
                sample_label=f"{sample_label} ({spec['source']} files)",
                model_label=model_label,
                run_dir=run_dir,
                reference_datamodule_cfg=reference_datamodule_cfg,
                files=files,
                output_dir=output_dir,
                args=args,
                device=device,
            )
    return all_outputs


def evaluate_without_shared_input_check(
    *,
    test_key: str,
    samples: OrderedDict[str, OrderedDict[str, dict]],
    feature_names: list[str],
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict]]:
    metrics_rows: list[dict] = []
    resolution_rows: list[dict] = []

    for sample_label, models in samples.items():
        feature_lookup = {name.lower(): idx for idx, name in enumerate(feature_names)}
        binned_features = [
            name
            for name in domain.BINNED_RESOLUTION_FEATURES
            if name.lower() in feature_lookup
        ]

        bins_by_feature = {}
        for feature_name in binned_features:
            feature_idx = feature_lookup[feature_name.lower()]
            values = []
            for item in models.values():
                original, _ = domain.aligned_arrays(item, feature_names)
                values.append(original[:, feature_idx])
            try:
                bins_by_feature[feature_name] = domain.make_bins(
                    np.concatenate(values),
                    args.n_bins,
                )
            except ValueError as exc:
                log.warning(
                    "%s: skipping binned plot for %s: %s",
                    sample_label,
                    feature_name,
                    exc,
                )
        binned_features = list(bins_by_feature)

        for model_label, item in models.items():
            original, reconstruction = domain.aligned_arrays(item, feature_names)
            metrics = domain.reconstruction_metrics(original, reconstruction, feature_names)
            mean_used, usage = domain.codebook_usage(item)
            for feature_name, values in metrics.items():
                row = {
                    "test": test_key,
                    "sample": sample_label,
                    "model": model_label,
                    "feature": feature_name,
                    "n": int(values["n"]),
                    "mae": float(values["mae"]),
                    "rmse": float(values["rmse"]),
                    "bias": float(values["bias"]),
                    "mean_codebook_used_percent": mean_used,
                }
                for quantizer_idx in range(item["indices"].shape[1]):
                    row[f"q{quantizer_idx}_used_percent"] = float(
                        usage[f"quantizer_{quantizer_idx}"]["percent_used"]
                    )
                metrics_rows.append(row)

            for feature_name in binned_features:
                feature_idx = feature_lookup[feature_name.lower()]
                centers, values, counts, medians, residual_iqrs = (
                    domain.binned_residual_iqr_over_median_truth(
                        original[:, feature_idx],
                        reconstruction[:, feature_idx],
                        bins_by_feature[feature_name],
                        min_bin_count=args.min_bin_count,
                        min_denominator=args.min_denominator,
                    )
                )
                for center, value, count, median, residual_iqr in zip(
                    centers,
                    values,
                    counts,
                    medians,
                    residual_iqrs,
                ):
                    resolution_rows.append(
                        {
                            "test": test_key,
                            "sample": sample_label,
                            "model": model_label,
                            "feature": feature_name,
                            "bin_center": float(center),
                            "relative_resolution": float(value),
                            "objects": int(count),
                            "median_original": float(median),
                            "residual_iqr": float(residual_iqr),
                        }
                    )

    return metrics_rows, resolution_rows


def configure_styles() -> None:
    domain.COLORS.update(
        {
            "MC-only old (q4)": "#4C83F1",
            "MC+data old (q4)": "#F28E2B",
            "MC+data old (q6)": "#B45F06",
            "MC+data old (q8)": "#CC79A7",
            "MC+data old (q8, cb2048)": "#009E73",
            "MC-only new (q4)": "#2F4B7C",
            "MC+data new (q4)": "#7A5195",
        }
    )
    domain.LINESTYLES.update(
        {
            "MC+data old (q6)": "--",
            "MC+data old (q8)": ":",
            "MC+data old (q8, cb2048)": "-.",
            "MC-only new (q4)": (0, (5, 2)),
            "MC+data new (q4)": (0, (3, 1, 1, 1)),
        }
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = domain.choose_device(args.device)
    configure_styles()

    specs = model_specs(args)
    source_files = files_by_source(args)

    checkpoints = {
        label: str(domain.explicit_checkpoint(spec["run_dir"], args.checkpoint_name))
        for label, spec in specs.items()
    }

    all_outputs = collect_outputs(
        specs=specs,
        source_files=source_files,
        output_dir=output_dir,
        args=args,
        device=device,
    )

    first_models = next(iter(all_outputs.values()))
    shared_features = domain.common_feature_names(*first_models.values())
    shared_feature_keys = [name.lower() for name in shared_features]
    if "pt" not in shared_feature_keys:
        raise RuntimeError("shared feature list has no pT")

    test_key = "old_new_sample_quantizer_controls"
    test_title = "Electron full-feature old/new sample tokenizer controls"
    model_labels = list(specs)

    metrics_rows, resolution_rows = evaluate_without_shared_input_check(
        test_key=test_key,
        samples=all_outputs,
        feature_names=shared_features,
        args=args,
    )

    domain.write_csv(output_dir / f"{test_key}_metrics.csv", metrics_rows)
    domain.write_csv(output_dir / f"{test_key}_binned_resolution.csv", resolution_rows)

    for feature_name in domain.BINNED_RESOLUTION_FEATURES:
        if feature_name.lower() not in shared_feature_keys:
            continue
        if feature_name in CATEGORICAL_FEATURES:
            plot_feature_metric_bars(
                test_title=test_title,
                feature_name=feature_name,
                samples=all_outputs,
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
            samples=all_outputs,
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

    manifest = {
        "runs": {label: str(spec["run_dir"]) for label, spec in specs.items()},
        "run_sample_source": {label: spec["source"] for label, spec in specs.items()},
        "checkpoints": checkpoints,
        "source_files": source_files,
        "split": args.split,
        "max_valid_objects": args.max_valid_objects,
        "num_events_per_file": args.num_events_per_file,
        "binned_resolution_features": list(domain.BINNED_RESOLUTION_FEATURES),
        "metric": (
            "IQR(reco-original) / abs(median(original)); old runs use old H5 "
            "files and new runs use new H5 files. Bins are shared within each "
            "panel using the union of model inputs."
        ),
    }
    (output_dir / "comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    logging.info("Wrote old/new sample comparison to %s", output_dir)


if __name__ == "__main__":
    main()
