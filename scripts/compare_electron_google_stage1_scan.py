#!/usr/bin/env python3
"""Compare Google/Condor electron stage-1 tokenizers on MC and real data.

This overlays the copied Google stage-1 scan runs and one full-sample reference
run on the usual two-panel MC/data electron resolution plots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import OrderedDict
from pathlib import Path

import hydra
import matplotlib
import numpy as np
import torch
from omegaconf import OmegaConf

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import compare_electron_domain_controls as domain  # noqa: E402
from analyze_vqvae_tokenizer import (  # noqa: E402
    collect_diagnostics_from_loader,
    dataloader_from_datamodule,
    feature_names_from_cfg,
    find_checkpoint,
    reconstruction_metrics,
    safe_filename,
    transform_list_and_cst_fn_from_cfg,
)
from compare_electron_full_quantizer_controls import plot_feature_metric_bars  # noqa: E402
from heptokens.models.vq_vae import LitVqVae  # noqa: E402

matplotlib.use("Agg")


log = logging.getLogger(__name__)

DEFAULT_GOOGLE_BASE = "google_results/electrons"
DEFAULT_FULL_REFERENCE_RUN = (
    "results/atlas_object_final_tokenizers_new_mcdata/"
    "electrons_full_dim8_cb2048_q8_e20_new_mcdata"
)

STAGE1_RUNS = OrderedDict(
    [
        ("stage1 q2 cb32768 dim16", "electrons_stage1_q2_cb32768_dim16_e20"),
        ("stage1 q4 cb8192 dim16", "electrons_stage1_q4_cb8192_dim16_e20"),
        ("stage1 q4 cb16384 dim16", "electrons_stage1_q4_cb16384_dim16_e20"),
        ("stage1 q6 cb2048 dim8", "electrons_stage1_q6_cb2048_dim8_e20"),
        ("stage1 q6 cb4096 dim8", "electrons_stage1_q6_cb4096_dim8_e20"),
        ("stage1 q8 cb2048 dim8", "electrons_stage1_q8_cb2048_dim8_e20"),
        ("stage1 q8 cb4096 dim8", "electrons_stage1_q8_cb4096_dim8_e20"),
    ]
)

LINE_RESOLUTION_FEATURES = ("pt", "ptvarcone30", "topoetcone20")
CATEGORICAL_FEATURES = ("LHMedium", "LHTight")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare copied Google/Condor electron stage-1 scan tokenizers plus "
            "one full-sample reference run on fixed MC and real-data H5 files."
        )
    )
    parser.add_argument("--google-base", default=DEFAULT_GOOGLE_BASE)
    parser.add_argument("--full-reference-run", default=DEFAULT_FULL_REFERENCE_RUN)
    parser.add_argument("--full-reference-label", default="full q8 cb2048 dim8")
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
        default="results/electron_google_stage1_scan_comparison",
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
        help=(
            "Comma-separated features to plot, or 'all' for all features shared by "
            "the selected models."
        ),
    )
    parser.add_argument(
        "--line-features",
        default="pt,ptvarcone30,topoetcone20",
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
            "searches each copied cluster result for preprocessing/electrons_log_standard.joblib."
        ),
    )
    parser.add_argument(
        "--include-e1",
        action="store_true",
        help="Also include the copied q4/cb8192/dim16 one-epoch smoke run.",
    )
    parser.add_argument("--force", action="store_true", help="Ignore cached arrays.")
    return parser.parse_args()


def fixed_files(directory: str, n_files: int, *, exclude_pattern: str = "") -> list[str]:
    files = [
        path
        for path in sorted(Path(directory).glob("*.h5"))
        if path.is_file() and path.stat().st_size > 0
    ]
    if exclude_pattern:
        files = [
            path
            for path in files
            if not path.match(f"*/{exclude_pattern}") and not path.match(exclude_pattern)
        ]
    return [str(path) for path in files[:n_files]]


def sample_files(args: argparse.Namespace) -> OrderedDict[str, list[str]]:
    samples: OrderedDict[str, list[str]] = OrderedDict(
        [
            (
                "MC sample",
                fixed_files(
                    args.mc_dir,
                    args.n_files,
                    exclude_pattern=args.exclude_mc_pattern,
                ),
            ),
            ("real-data sample", fixed_files(args.data_dir, args.n_files)),
        ]
    )
    for label, files in samples.items():
        if not files:
            raise FileNotFoundError(f"No non-empty H5 files found for {label}")
    return samples


def resolve_run_dir(path: str | Path, label: str) -> Path:
    run_dir = Path(path).resolve()
    if (run_dir / "full_config.yaml").exists():
        return run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing run path for {label}: {run_dir}")

    candidates = sorted(
        {config.parent for config in run_dir.rglob("full_config.yaml")},
        key=lambda candidate: (candidate.stat().st_mtime, str(candidate)),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No full_config.yaml found under {run_dir} for {label}")
    if len(candidates) > 1:
        log.warning(
            "Found %d candidate run dirs under %s for %s; using newest: %s",
            len(candidates),
            run_dir,
            label,
            candidates[0],
        )
    return candidates[0]


def stage1_run_dirs(args: argparse.Namespace) -> OrderedDict[str, Path]:
    google_base = Path(args.google_base)
    definitions = OrderedDict(STAGE1_RUNS)
    if args.include_e1:
        definitions["stage1 q4 cb8192 dim16 e1"] = "electrons_stage1_q4_cb8192_dim16_e1"

    runs: OrderedDict[str, Path] = OrderedDict()
    for label, relative in definitions.items():
        runs[label] = resolve_run_dir(google_base / relative, label)
    runs[args.full_reference_label] = resolve_run_dir(
        args.full_reference_run,
        args.full_reference_label,
    )
    return runs


def preprocessor_path_from_cfg(cfg) -> Path | None:
    filename = OmegaConf.select(cfg, "datamodule.transforms.preprocess.cst_fn.filename")
    if not filename:
        return None
    path = Path(str(filename))
    return path if path.exists() else None


def google_preprocessor_candidates(args: argparse.Namespace, run_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    if args.google_preprocessor:
        candidates.append(Path(args.google_preprocessor))
    for base in [run_dir, *run_dir.parents]:
        candidates.extend(
            [
                base / "preprocessing" / "electrons_log_standard.joblib",
                base / "results" / "preprocessing" / "electrons_log_standard.joblib",
                base
                / "results"
                / "preprocessing"
                / "atlas_object_final_tokenizers_new_mcdata"
                / "electrons_log_standard.joblib",
            ]
        )
    candidates.extend(
        [
            Path("results/preprocessing/atlas_object_final_tokenizers_new_mcdata/electrons_log_standard.joblib"),
            Path("results/preprocessing/electrons_log_standard.joblib"),
        ]
    )
    return candidates


def patch_missing_preprocessor(cfg, args: argparse.Namespace, run_dir: Path) -> Path | None:
    existing = preprocessor_path_from_cfg(cfg)
    if existing is not None:
        return existing.resolve()

    saved = OmegaConf.select(cfg, "datamodule.transforms.preprocess.cst_fn.filename")
    for candidate in google_preprocessor_candidates(args, run_dir):
        if candidate.exists():
            resolved = candidate.resolve()
            OmegaConf.update(
                cfg,
                "datamodule.transforms.preprocess.cst_fn.filename",
                str(resolved),
                force_add=True,
            )
            log.info(
                "Using local preprocessor %s for %s; saved path was %s",
                resolved,
                run_dir,
                saved,
            )
            return resolved

    if saved:
        raise FileNotFoundError(
            "Saved preprocessor path is missing and no local replacement was found. "
            f"Saved path: {saved}. Use --google-preprocessor."
        )
    return None


def force_common_eval_split(cfg) -> None:
    OmegaConf.set_struct(cfg.datamodule, False)
    cfg.datamodule._target_ = "heptokens.data.atlas_event_mappable.AtlasEventMapModule"
    cfg.datamodule.train_frac = 0.0
    cfg.datamodule.val_frac = 1.0
    cfg.datamodule.test_frac = 0.0
    for key in (
        "chunk_size",
        "split_by_domain",
        "sampling_domain_fractions",
        "sampling_balance_by",
        "sampling_num_samples",
    ):
        if key in cfg.datamodule:
            del cfg.datamodule[key]


def analysis_datamodule_cfg(cfg, args: argparse.Namespace, files: list[str]):
    datamodule_cfg = OmegaConf.create(OmegaConf.to_container(cfg.datamodule, resolve=True))
    OmegaConf.set_struct(datamodule_cfg, False)
    datamodule_cfg.data_paths = list(files)
    if "data_path" in datamodule_cfg:
        del datamodule_cfg["data_path"]
    datamodule_cfg.data_domains = None
    datamodule_cfg.sampling_domain_fractions = None
    datamodule_cfg.sampling_num_samples = None
    datamodule_cfg.num_events = args.num_events_per_file
    datamodule_cfg.batch_size = args.batch_size
    datamodule_cfg.num_workers = args.num_workers
    if args.num_workers == 0:
        datamodule_cfg.persistent_workers = False
        datamodule_cfg.multiprocessing_context = None
    return datamodule_cfg


def cache_path(
    *,
    output_dir: Path,
    sample_label: str,
    model_label: str,
    run_dir: Path,
    checkpoint: Path,
    files: list[str],
    preprocessor: Path | None,
    args: argparse.Namespace,
) -> Path:
    checkpoint_stat = checkpoint.stat()
    cache_input = "\n".join(
        files
        + [
            f"run={run_dir.resolve()}",
            f"checkpoint={checkpoint.resolve()}",
            f"checkpoint_size={checkpoint_stat.st_size}",
            f"checkpoint_mtime_ns={checkpoint_stat.st_mtime_ns}",
            f"preprocessor={preprocessor}",
            f"split={args.split}",
            f"num_events_per_file={args.num_events_per_file}",
            f"batch_size={args.batch_size}",
            f"max_valid_objects={args.max_valid_objects}",
            "analysis_path=electron_google_stage1_scan_v1",
        ]
    )
    digest = hashlib.sha1(cache_input.encode("utf-8")).hexdigest()[:12]
    return (
        output_dir
        / "cache_arrays"
        / (
            f"{safe_filename(sample_label)}_{safe_filename(model_label)}_"
            f"{safe_filename(run_dir.name)}_{len(files)}files_{digest}.npz"
        )
    )


def collect_one(
    *,
    sample_label: str,
    model_label: str,
    run_dir: Path,
    files: list[str],
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    cfg = OmegaConf.load(run_dir / "full_config.yaml")
    checkpoint = find_checkpoint(
        run_dir,
        str((run_dir / "checkpoints" / args.checkpoint_name).resolve()),
    )
    preprocessor = patch_missing_preprocessor(cfg, args, run_dir)
    force_common_eval_split(cfg)

    cached = cache_path(
        output_dir=output_dir,
        sample_label=sample_label,
        model_label=model_label,
        run_dir=run_dir,
        checkpoint=checkpoint,
        files=files,
        preprocessor=preprocessor,
        args=args,
    )
    if cached.exists() and not args.force:
        log.info("Reading cache %s", cached)
        item = np.load(cached, allow_pickle=True)
        return {
            "original": item["original"],
            "reconstruction": item["reconstruction"],
            "indices": item["indices"],
            "feature_names": [str(value) for value in item["feature_names"]],
            "codebook_size": int(item["codebook_size"]),
            "checkpoint": str(item["checkpoint"]),
        }

    _, inverse_transformer = transform_list_and_cst_fn_from_cfg(cfg)
    log.info("Loading %s for %s / %s", checkpoint, sample_label, model_label)
    model = LitVqVae.load_from_checkpoint(checkpoint, map_location=device)
    model.to(device)
    model.eval()

    datamodule_cfg = analysis_datamodule_cfg(cfg, args, files)
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
    codebook_size = int(cfg.model.codebook_size)

    cached.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cached,
        original=original,
        reconstruction=reconstruction,
        indices=indices,
        feature_names=np.asarray(feature_names),
        codebook_size=np.asarray(codebook_size),
        checkpoint=np.asarray(str(checkpoint)),
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "original": original,
        "reconstruction": reconstruction,
        "indices": indices,
        "feature_names": feature_names,
        "codebook_size": codebook_size,
        "checkpoint": str(checkpoint),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    domain.write_csv(path, rows)


def selected_features(requested: str, shared_features: list[str]) -> list[str]:
    if requested.strip().lower() == "all":
        return list(shared_features)
    lookup = {name.lower(): name for name in shared_features}
    features = []
    for raw in requested.split(","):
        key = raw.strip().lower()
        if not key:
            continue
        if key not in lookup:
            log.warning("Skipping requested feature %s; shared features are %s", raw, shared_features)
            continue
        features.append(lookup[key])
    return features


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = domain.choose_device(args.device)

    run_dirs = stage1_run_dirs(args)
    files_by_sample = sample_files(args)

    domain.COLORS.update(
        {
            "stage1 q2 cb32768 dim16": "#4C78A8",
            "stage1 q4 cb8192 dim16": "#F58518",
            "stage1 q4 cb16384 dim16": "#B45F06",
            "stage1 q6 cb2048 dim8": "#54A24B",
            "stage1 q6 cb4096 dim8": "#00876C",
            "stage1 q8 cb2048 dim8": "#CC79A7",
            "stage1 q8 cb4096 dim8": "#7A5195",
            args.full_reference_label: "#111111",
        }
    )
    domain.LINESTYLES.update(
        {
            "stage1 q4 cb16384 dim16": "--",
            "stage1 q6 cb4096 dim8": "--",
            "stage1 q8 cb4096 dim8": "--",
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
            all_outputs[sample_label][model_label] = collect_one(
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
    features_to_plot = selected_features(args.features, shared_features)
    line_features = {
        feature.strip().lower()
        for feature in args.line_features.split(",")
        if feature.strip()
    }
    metrics_rows, resolution_rows = domain.evaluate_test(
        test_key="electron_google_stage1_scan",
        test_title="Electron Google stage-1 tokenizer scan",
        samples=all_outputs,
        feature_names=shared_features,
        args=args,
    )

    write_csv(output_dir / "electron_google_stage1_scan_metrics.csv", metrics_rows)
    write_csv(
        output_dir / "electron_google_stage1_scan_binned_resolution.csv",
        resolution_rows,
    )

    binned_feature_keys = {
        feature.lower() for feature in domain.BINNED_RESOLUTION_FEATURES
    }
    for feature_name in features_to_plot:
        feature_key = feature_name.lower()
        if (
            feature_name in CATEGORICAL_FEATURES
            or feature_key not in line_features
            or feature_key not in binned_feature_keys
        ):
            plot_feature_metric_bars(
                test_title="Electron Google stage-1 tokenizer scan",
                feature_name=feature_name,
                samples=all_outputs,
                metrics_rows=metrics_rows,
                output_path=output_dir / f"electron_google_stage1_scan_{feature_name}_mae_rmse.png",
            )
            continue
        feature_rows = [row for row in resolution_rows if row["feature"] == feature_name]
        write_csv(
            output_dir / f"electron_google_stage1_scan_{feature_name}_binned_resolution.csv",
            feature_rows,
        )
        domain.plot_feature_resolution(
            test_title="Electron Google stage-1 tokenizer scan",
            feature_name=feature_name,
            samples=all_outputs,
            resolution_rows=feature_rows,
            output_path=output_dir / f"electron_google_stage1_scan_{feature_name}_resolution.png",
        )

    report_lines = domain.report_section(
        title="Electron Google stage-1 tokenizer scan",
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
    log.info("Wrote electron Google stage-1 comparison to %s", output_dir)


if __name__ == "__main__":
    main()
