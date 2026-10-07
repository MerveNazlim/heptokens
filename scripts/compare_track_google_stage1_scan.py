#!/usr/bin/env python3
"""Compare Google/Condor track stage-1 tokenizers on MC and real data."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import OrderedDict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import compare_electron_domain_controls as domain  # noqa: E402
import compare_electron_google_stage1_scan as scan  # noqa: E402
from analyze_vqvae_tokenizer import find_checkpoint  # noqa: E402
from compare_electron_full_quantizer_controls import plot_feature_metric_bars  # noqa: E402


log = logging.getLogger(__name__)

DEFAULT_GOOGLE_BASE = "google_results/tracks"
DEFAULT_FULL_REFERENCE_RUN = (
    "results/atlas_object_final_tokenizers_new_mcdata/"
    "tracks_full_dim8_cb4096_q8_e20_new_mcdata"
)

STAGE1_RUNS = OrderedDict(
    [
        ("stage1 q2 cb32768 dim16", "tracks_stage1_q2_cb32768_dim16_e20"),
        ("stage1 q4 cb8192 dim16", "tracks_stage1_q4_cb8192_dim16_e20"),
        ("stage1 q6 cb2048 dim8", "tracks_stage1_q6_cb2048_dim8_e20"),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare copied Google/Condor track stage-1 tokenizers plus one "
            "full-sample reference run on fixed MC and real-data H5 files."
        )
    )
    parser.add_argument("--google-base", default=DEFAULT_GOOGLE_BASE)
    parser.add_argument("--full-reference-run", default=DEFAULT_FULL_REFERENCE_RUN)
    parser.add_argument("--full-reference-label", default="full q8 cb4096 dim8")
    parser.add_argument(
        "--mc-dir",
        default="/home/zephyr/Data/viviana/bnl-treasure/data_new/h5",
    )
    parser.add_argument(
        "--data-dir",
        default="/home/zephyr/Data/viviana/bnl-treasure/data_new/h5/realdata",
    )
    parser.add_argument("--exclude-mc-pattern", default="DAOD_PHYSLITE.370016*")
    parser.add_argument(
        "--output-dir",
        default="results/track_google_stage1_scan_comparison",
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
    parser.add_argument(
        "--features",
        default="all",
        help="Comma-separated features to plot, or 'all' for all shared features.",
    )
    parser.add_argument(
        "--line-features",
        default="pt,eta,phi,chiSquared",
        help=(
            "Comma-separated features to draw as binned resolution curves. Other "
            "requested features are drawn as MAE/RMSE summary bars."
        ),
    )
    parser.add_argument(
        "--google-preprocessor",
        default="",
        help=(
            "Optional fallback preprocessor for Google runs. If omitted, the script "
            "searches each copied cluster result for track preprocessing joblibs."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Ignore cached arrays.")
    return parser.parse_args()


def track_preprocessor_candidates(args: argparse.Namespace, run_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    if args.google_preprocessor:
        candidates.append(Path(args.google_preprocessor))
    names = (
        "tracks_log_standard_no_ndoflog.joblib",
        "tracks_log_standard_no_ndof.joblib",
        "tracks_log_standard.joblib",
    )
    for base in [run_dir, *run_dir.parents]:
        for name in names:
            candidates.extend(
                [
                    base / "preprocessing" / name,
                    base / "results" / "preprocessing" / name,
                    base
                    / "results"
                    / "preprocessing"
                    / "atlas_object_final_tokenizers_new_mcdata"
                    / name,
                ]
            )
    for name in names:
        candidates.extend(
            [
                Path("results/preprocessing/atlas_object_final_tokenizers_new_mcdata") / name,
                Path("results/preprocessing") / name,
            ]
        )
    return candidates


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = domain.choose_device(args.device)

    scan.STAGE1_RUNS = STAGE1_RUNS
    scan.google_preprocessor_candidates = track_preprocessor_candidates
    args.include_e1 = False

    run_dirs = scan.stage1_run_dirs(args)
    files_by_sample = scan.sample_files(args)

    domain.COLORS.update(
        {
            "stage1 q2 cb32768 dim16": "#4C78A8",
            "stage1 q4 cb8192 dim16": "#F58518",
            "stage1 q6 cb2048 dim8": "#54A24B",
            args.full_reference_label: "#111111",
        }
    )
    domain.LINESTYLES.update(
        {
            "stage1 q4 cb8192 dim16": "--",
            "stage1 q6 cb2048 dim8": "-.",
            args.full_reference_label: ":",
        }
    )

    checkpoints = {
        label: str(find_checkpoint(run_dir, str(run_dir / "checkpoints" / args.checkpoint_name)))
        for label, run_dir in run_dirs.items()
    }

    all_outputs: OrderedDict[str, OrderedDict[str, dict]] = OrderedDict()
    for sample_label, files in files_by_sample.items():
        all_outputs[sample_label] = OrderedDict()
        for model_label, run_dir in run_dirs.items():
            all_outputs[sample_label][model_label] = scan.collect_one(
                sample_label=sample_label,
                model_label=model_label,
                run_dir=run_dir,
                files=files,
                output_dir=output_dir,
                args=args,
                device=device,
            )

    first_models = next(iter(all_outputs.values()))
    shared_features = domain.common_feature_names(*first_models.values())
    features_to_plot = scan.selected_features(args.features, shared_features)
    line_features = {
        feature.strip().lower()
        for feature in args.line_features.split(",")
        if feature.strip()
    }

    domain.BINNED_RESOLUTION_FEATURES = tuple(
        feature for feature in features_to_plot if feature.lower() in line_features
    )
    metrics_rows, resolution_rows = domain.evaluate_test(
        test_key="track_google_stage1_scan",
        test_title="Track Google stage-1 tokenizer scan",
        samples=all_outputs,
        feature_names=shared_features,
        args=args,
    )

    scan.write_csv(output_dir / "track_google_stage1_scan_metrics.csv", metrics_rows)
    scan.write_csv(
        output_dir / "track_google_stage1_scan_binned_resolution.csv",
        resolution_rows,
    )

    binned_feature_keys = {
        feature.lower() for feature in domain.BINNED_RESOLUTION_FEATURES
    }
    for feature_name in features_to_plot:
        feature_key = feature_name.lower()
        if feature_key not in line_features or feature_key not in binned_feature_keys:
            plot_feature_metric_bars(
                test_title="Track Google stage-1 tokenizer scan",
                feature_name=feature_name,
                samples=all_outputs,
                metrics_rows=metrics_rows,
                output_path=output_dir / f"track_google_stage1_scan_{feature_name}_mae_rmse.png",
            )
            continue
        feature_rows = [row for row in resolution_rows if row["feature"] == feature_name]
        scan.write_csv(
            output_dir / f"track_google_stage1_scan_{feature_name}_binned_resolution.csv",
            feature_rows,
        )
        domain.plot_feature_resolution(
            test_title="Track Google stage-1 tokenizer scan",
            feature_name=feature_name,
            samples=all_outputs,
            resolution_rows=feature_rows,
            output_path=output_dir / f"track_google_stage1_scan_{feature_name}_resolution.png",
        )

    report_lines = domain.report_section(
        title="Track Google stage-1 tokenizer scan",
        models=list(run_dirs),
        metrics_rows=metrics_rows,
        resolution_rows=resolution_rows,
    )
    (output_dir / "comparison_report.md").write_text("\n".join(report_lines).rstrip() + "\n")
    (output_dir / "comparison_manifest.json").write_text(
        json.dumps(
            {
                "runs": {label: str(path) for label, path in run_dirs.items()},
                "checkpoints": checkpoints,
                "samples": files_by_sample,
                "features": features_to_plot,
                "line_features": sorted(line_features),
                "metric": "IQR(reco-original) / abs(median(original)) in fixed original-value bins",
                "common_eval_split": "train_frac=0, val_frac=1, test_frac=0",
            },
            indent=2,
        )
        + "\n"
    )
    log.info("Wrote track Google stage-1 comparison to %s", output_dir)


if __name__ == "__main__":
    main()
