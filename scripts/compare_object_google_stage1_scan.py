#!/usr/bin/env python3
"""Compare copied Google/Condor stage-1 object tokenizer scans.

For one object type, this overlays the copied Google stage-1 scan runs and,
optionally, one full-sample reference run on the usual two-panel MC/data
resolution plots.
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
    apply_hep_style,
    collect_diagnostics_from_loader,
    dataloader_from_datamodule,
    feature_label,
    feature_names_from_cfg,
    find_checkpoint,
    safe_filename,
    transform_list_and_cst_fn_from_cfg,
)
from compare_electron_full_quantizer_controls import plot_feature_metric_bars  # noqa: E402
from heptokens.models.vq_vae import LitVqVae  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


log = logging.getLogger(__name__)

OBJECTS = ("jets", "electrons", "muons", "photons", "taus", "tracks")

ORIGINAL_STAGE1_SPECS = OrderedDict(
    [
        ("stage1 q2 cb32768 dim16", "q2_cb32768_dim16_e20"),
        ("stage1 q4 cb8192 dim16", "q4_cb8192_dim16_e20"),
        ("stage1 q4 cb16384 dim16", "q4_cb16384_dim16_e20"),
        ("stage1 q6 cb2048 dim8", "q6_cb2048_dim8_e20"),
        ("stage1 q6 cb4096 dim8", "q6_cb4096_dim8_e20"),
        ("stage1 q8 cb2048 dim8", "q8_cb2048_dim8_e20"),
        ("stage1 q8 cb4096 dim8", "q8_cb4096_dim8_e20"),
    ]
)

EXTENDED_STAGE1_SPECS = OrderedDict(
    [
        ("stage1 q1 cb16384 dim8", "q1_cb16384_dim8_e20"),
        ("stage1 q1 cb32768 dim8", "q1_cb32768_dim8_e20"),
        ("stage1 q4 cb4096 dim8", "q4_cb4096_dim8_e20"),
        ("stage1 q4 cb8192 dim8", "q4_cb8192_dim8_e20"),
        ("stage1 q4 cb16384 dim8", "q4_cb16384_dim8_e20"),
        ("stage1 q6 cb8192 dim8", "q6_cb8192_dim8_e20"),
        ("stage1 q8 cb8192 dim8", "q8_cb8192_dim8_e20"),
    ]
)

EXTENDED_Q8_STAGE1_SPECS = OrderedDict(
    [
        ("stage1 q8 cb2048 dim8", "q8_cb2048_dim8_e20"),
        ("stage1 q8 cb4096 dim8", "q8_cb4096_dim8_e20"),
        *EXTENDED_STAGE1_SPECS.items(),
    ]
)

STAGE1_SPECS = OrderedDict([*ORIGINAL_STAGE1_SPECS.items(), *EXTENDED_STAGE1_SPECS.items()])

DEFAULT_FULL_REFERENCE_RUNS = {
    "jets": (
        "results/atlas_object_final_tokenizers_new_mcdata/"
        "jets_full_dim8_cb2048_q8_e20_new_mcdata"
    ),
    "electrons": (
        "results/atlas_object_final_tokenizers_new_mcdata/"
        "electrons_full_dim8_cb2048_q8_e20_new_mcdata"
    ),
    "muons": (
        "results/atlas_object_final_tokenizers_new_mcdata/"
        "muons_full_dim8_cb2048_q8_e20_new_mcdata"
    ),
    "photons": (
        "results/atlas_object_final_tokenizers_new_mcdata/"
        "photons_full_dim8_cb2048_q8_e20_new_mcdata"
    ),
    "taus": (
        "results/atlas_object_final_tokenizers_new_mcdata/"
        "taus_full_dim8_cb4096_q8_e20_new_mcdata"
    ),
    "tracks": (
        "results/atlas_object_final_tokenizers_new_mcdata/"
        "tracks_full_dim8_cb4096_q8_e20_new_mcdata"
    ),
}

DEFAULT_FULL_REFERENCE_LABELS = {
    "jets": "full q8 cb2048 dim8",
    "electrons": "full q8 cb2048 dim8",
    "muons": "full q8 cb2048 dim8",
    "photons": "full q8 cb2048 dim8",
    "taus": "full q8 cb4096 dim8",
    "tracks": "full q8 cb4096 dim8",
}

DEFAULT_LINE_FEATURES = {
    "jets": "pt,mass,n_trk,QG_nTracks,QG_tracksWidth,QG_tracksC1,DL1d_pb,DL1d_pc,DL1d_pu,GN2_pb,GN2_pc,GN2_pu",
    "electrons": "pt,ptvarcone30,topoetcone20",
    "muons": "pt,ptvarcone30,topoetcone20",
    "photons": "pt,ptcone20,topoetcone20,topoetcone40",
    "taus": "pt,RNNJetScore,RNNEleScore",
    "tracks": "pt,nDoF,chiSquared",
}

PREPROCESSOR_NAMES = {
    "jets": ("jets_log_standard.joblib",),
    "electrons": ("electrons_log_standard.joblib",),
    "muons": ("muons_log_standard.joblib",),
    "photons": ("photons_log_standard.joblib",),
    "taus": ("taus_log_standard.joblib",),
    "tracks": (
        "tracks_log_standard_no_ndoflog.joblib",
        "tracks_log_standard_no_ndof.joblib",
        "tracks_log_standard.joblib",
    ),
}

COLORS = {
    "stage1 q2 cb32768 dim16": "#4C78A8",
    "stage1 q4 cb8192 dim16": "#F58518",
    "stage1 q4 cb16384 dim16": "#B45F06",
    "stage1 q6 cb2048 dim8": "#54A24B",
    "stage1 q6 cb4096 dim8": "#00876C",
    "stage1 q8 cb2048 dim8": "#CC79A7",
    "stage1 q8 cb4096 dim8": "#7A5195",
    "stage1 q1 cb16384 dim8": "#2F4B7C",
    "stage1 q1 cb32768 dim8": "#665191",
    "stage1 q4 cb4096 dim8": "#A05195",
    "stage1 q4 cb8192 dim8": "#D45087",
    "stage1 q4 cb16384 dim8": "#F95D6A",
    "stage1 q6 cb8192 dim8": "#FF7C43",
    "stage1 q8 cb8192 dim8": "#FFA600",
}

LINESTYLES = {
    "stage1 q2 cb32768 dim16": "--",
    "stage1 q4 cb8192 dim16": "--",
    "stage1 q4 cb16384 dim16": "--",
    "stage1 q6 cb2048 dim8": "-",
    "stage1 q6 cb4096 dim8": "-",
    "stage1 q8 cb2048 dim8": "-",
    "stage1 q8 cb4096 dim8": "-",
    "stage1 q1 cb16384 dim8": "-",
    "stage1 q1 cb32768 dim8": "-",
    "stage1 q4 cb4096 dim8": "-",
    "stage1 q4 cb8192 dim8": "-",
    "stage1 q4 cb16384 dim8": "-",
    "stage1 q6 cb8192 dim8": "-",
    "stage1 q8 cb8192 dim8": "-",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Google/Condor stage-1 tokenizer scan runs for one object."
    )
    parser.add_argument("--object", required=True, choices=OBJECTS)
    parser.add_argument(
        "--google-base",
        help="Defaults to google_results/OBJECT.",
    )
    parser.add_argument(
        "--full-reference-run",
        default="none",
        help=(
            "Full-sample reference run to overlay. Use 'auto' for the known "
            "Zephyr q8 run, or 'none' to disable."
        ),
    )
    parser.add_argument(
        "--full-reference-label",
        default="auto",
        help="Label for --full-reference-run. Use 'auto' for an object-specific label.",
    )
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
        help="Defaults to results/google_stage1_scan_plots/OBJECT.",
    )
    parser.add_argument("--n-files", type=int, default=20)
    parser.add_argument("--num-events-per-file", type=int)
    parser.add_argument("--max-valid-objects", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--checkpoint-name", default="last.ckpt")
    parser.add_argument(
        "--scan-set",
        choices=["original", "extended", "extended-q8", "all"],
        default="original",
        help=(
            "Choose the original seven points, the seven extended points, the "
            "extended points plus the two old q8 references, or all points."
        ),
    )
    parser.add_argument("--n-bins", type=int, default=14)
    parser.add_argument("--min-bin-count", type=int, default=50)
    parser.add_argument("--min-denominator", type=float, default=1e-8)
    parser.add_argument(
        "--log-y",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a logarithmic y-axis for binned resolution plots.",
    )
    parser.add_argument(
        "--features",
        default="all",
        help="Comma-separated features to plot, or 'all' for all shared features.",
    )
    parser.add_argument(
        "--line-features",
        default="auto",
        help=(
            "Comma-separated features to draw as binned resolution curves. Other "
            "requested/shared features are drawn as MAE/RMSE summary bars. "
            "Use 'auto' for object-specific defaults."
        ),
    )
    parser.add_argument(
        "--google-preprocessor",
        default="",
        help=(
            "Optional fallback preprocessor for Google runs. If omitted, the script "
            "searches inside each copied cluster result."
        ),
    )
    parser.add_argument(
        "--skip-missing-runs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip missing stage-1 run directories instead of failing.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Only replot from existing cached arrays; fail if any cache is missing.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate completed runs, checkpoints, run-local preprocessors, and "
            "common evaluation samples without evaluating models."
        ),
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
            ("data", fixed_files(args.data_dir, args.n_files)),
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

    completed_candidates = [
        candidate for candidate in candidates if (candidate / "SUCCESS.txt").exists()
    ]
    if completed_candidates:
        selected = completed_candidates[0]
    else:
        selected = candidates[0]

    if len(candidates) > 1:
        log.warning(
            "Found %d candidate run dirs under %s for %s; using newest %s candidate: %s",
            len(candidates),
            run_dir,
            label,
            "completed" if completed_candidates else "available",
            selected,
        )
    return selected


def stage1_dir_name(object_name: str, suffix: str) -> str:
    return f"{object_name}_stage1_{suffix}"


def display_label(model_label: str) -> str:
    label = model_label.removeprefix("stage1 ")
    label = label.replace(" cb", " ")
    label = label.replace(" dim", " d")
    return label


def feature_unit(feature_name: str) -> str:
    key = feature_name.lower()
    if key in {
        "pt",
        "mass",
        "ptvarcone30",
        "topoetcone20",
        "topoetcone40",
        "ptcone20",
        "trk_iso03",
    }:
        return "GeV"
    if key in {"n_trk", "qg_ntracks", "ndof", "nDoF".lower()}:
        return "count"
    return ""


def truth_axis_label(feature_name: str) -> str:
    label = feature_label(feature_name)
    unit = feature_unit(feature_name)
    if unit:
        return f"Truth {label} [{unit}]"
    return f"Truth {label}"


def stage1_run_dirs(args: argparse.Namespace) -> OrderedDict[str, Path]:
    google_base = Path(args.google_base or f"google_results/{args.object}")
    runs: OrderedDict[str, Path] = OrderedDict()
    specs = {
        "original": ORIGINAL_STAGE1_SPECS,
        "extended": EXTENDED_STAGE1_SPECS,
        "extended-q8": EXTENDED_Q8_STAGE1_SPECS,
        "all": STAGE1_SPECS,
    }[args.scan_set]
    for label, suffix in specs.items():
        path = google_base / stage1_dir_name(args.object, suffix)
        try:
            resolved = resolve_run_dir(path, label)
            if not (resolved / "SUCCESS.txt").exists():
                raise FileNotFoundError(f"No SUCCESS.txt found for {label} under {resolved}")
            runs[label] = resolved
        except FileNotFoundError:
            if args.skip_missing_runs:
                log.warning("Skipping missing %s at %s", label, path)
                continue
            raise

    full_reference_run = args.full_reference_run
    if full_reference_run == "auto":
        full_reference_run = DEFAULT_FULL_REFERENCE_RUNS[args.object]
    if full_reference_run.lower() != "none":
        full_reference_label = (
            DEFAULT_FULL_REFERENCE_LABELS[args.object]
            if args.full_reference_label == "auto"
            else args.full_reference_label
        )
        try:
            runs[full_reference_label] = resolve_run_dir(full_reference_run, full_reference_label)
        except FileNotFoundError:
            if args.skip_missing_runs:
                log.warning("Skipping missing full reference %s", full_reference_run)
            else:
                raise
    return runs


def resolve_checkpoint(run_dir: Path, checkpoint_name: str) -> Path:
    requested = run_dir / "checkpoints" / checkpoint_name
    if requested.exists():
        return find_checkpoint(run_dir, str(requested.resolve()))

    for fallback_name in ("last.ckpt", "best.ckpt"):
        fallback = run_dir / "checkpoints" / fallback_name
        if fallback.exists():
            log.warning(
                "Requested checkpoint %s is missing for %s; using %s",
                checkpoint_name,
                run_dir,
                fallback.name,
            )
            return find_checkpoint(run_dir, str(fallback.resolve()))

    candidates = sorted(
        (run_dir / "checkpoints").glob("*.ckpt"),
        key=lambda path: (path.stat().st_mtime, str(path)),
        reverse=True,
    )
    if candidates:
        log.warning(
            "Requested checkpoint %s is missing for %s; using newest checkpoint %s",
            checkpoint_name,
            run_dir,
            candidates[0].name,
        )
        return find_checkpoint(run_dir, str(candidates[0].resolve()))
    raise FileNotFoundError(f"No checkpoint found under {run_dir / 'checkpoints'}")


def preprocessor_path_from_cfg(cfg) -> Path | None:
    filename = OmegaConf.select(cfg, "datamodule.transforms.preprocess.cst_fn.filename")
    if not filename:
        return None
    path = Path(str(filename))
    return path if path.exists() else None


def copied_stage1_results_dir(run_dir: Path) -> Path | None:
    if "google_results" not in run_dir.parts or "_stage1_" not in run_dir.name:
        return None
    return next(
        (parent for parent in run_dir.parents if parent.name == "results"),
        None,
    )


def verified_stage1_preprocessor(args: argparse.Namespace, run_dir: Path) -> Path | None:
    results_dir = copied_stage1_results_dir(run_dir)
    if results_dir is None:
        return None

    candidates = [
        results_dir / "preprocessing" / name
        for name in PREPROCESSOR_NAMES[args.object]
    ]
    preprocessor = next((path for path in candidates if path.is_file()), None)
    if preprocessor is None:
        expected = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            "Copied stage-1 run is missing its run-local preprocessor. "
            f"Expected one of: {expected}"
        )

    checksum_path = results_dir / "preprocessor.sha256"
    if not checksum_path.is_file():
        raise FileNotFoundError(
            f"Copied stage-1 run is missing its checksum file: {checksum_path}"
        )
    checksum_entries = dict(
        line.strip().split("=", 1)
        for line in checksum_path.read_text().splitlines()
        if "=" in line
    )
    expected_checksum = checksum_entries.get("preprocessor_sha256")
    if not expected_checksum:
        raise ValueError(f"No preprocessor_sha256 entry found in {checksum_path}")
    actual_checksum = hashlib.sha256(preprocessor.read_bytes()).hexdigest()
    if actual_checksum != expected_checksum:
        raise ValueError(
            "Run-local preprocessor checksum mismatch: "
            f"expected={expected_checksum}, actual={actual_checksum}, path={preprocessor}"
        )

    log.info(
        "Using verified run-local preprocessor %s (sha256=%s)",
        preprocessor,
        actual_checksum,
    )
    return preprocessor.resolve()


def verify_common_stage1_preprocessing(
    args: argparse.Namespace,
    run_dirs: OrderedDict[str, Path],
) -> None:
    checksums: OrderedDict[str, str] = OrderedDict()
    for label, run_dir in run_dirs.items():
        preprocessor = verified_stage1_preprocessor(args, run_dir)
        if preprocessor is not None:
            checksums[label] = hashlib.sha256(preprocessor.read_bytes()).hexdigest()

    unique_checksums = set(checksums.values())
    if len(unique_checksums) > 1:
        details = ", ".join(
            f"{label}={checksum}" for label, checksum in checksums.items()
        )
        raise ValueError(
            "Stage-1 runs do not use an identical preprocessing transform: "
            f"{details}"
        )
    if checksums:
        log.info(
            "Verified identical run-local preprocessing for %d stage-1 runs "
            "(sha256=%s)",
            len(checksums),
            next(iter(unique_checksums)),
        )


def preprocessor_candidates(args: argparse.Namespace, run_dir: Path) -> list[Path]:
    names = PREPROCESSOR_NAMES[args.object]
    candidates: list[Path] = []
    if args.google_preprocessor:
        candidates.append(Path(args.google_preprocessor))
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


def patch_missing_preprocessor(cfg, args: argparse.Namespace, run_dir: Path) -> Path | None:
    stage1_preprocessor = verified_stage1_preprocessor(args, run_dir)
    if stage1_preprocessor is not None:
        OmegaConf.update(
            cfg,
            "datamodule.transforms.preprocess.cst_fn.filename",
            str(stage1_preprocessor),
            force_add=True,
        )
        return stage1_preprocessor

    existing = preprocessor_path_from_cfg(cfg)
    if existing is not None:
        return existing.resolve()

    saved = OmegaConf.select(cfg, "datamodule.transforms.preprocess.cst_fn.filename")
    for candidate in preprocessor_candidates(args, run_dir):
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
        names = ", ".join(PREPROCESSOR_NAMES[args.object])
        raise FileNotFoundError(
            "Saved preprocessor path is missing and no local replacement was found. "
            f"Saved path: {saved}. Expected one of: {names}. Use --google-preprocessor."
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
    object_name: str,
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
            f"object={object_name}",
            f"run={run_dir.resolve()}",
            f"checkpoint={checkpoint.resolve()}",
            f"checkpoint_size={checkpoint_stat.st_size}",
            f"checkpoint_mtime_ns={checkpoint_stat.st_mtime_ns}",
            f"preprocessor={preprocessor}",
            f"split={args.split}",
            f"num_events_per_file={args.num_events_per_file}",
            f"batch_size={args.batch_size}",
            f"max_valid_objects={args.max_valid_objects}",
            "analysis_path=object_google_stage1_scan_v1",
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


def cache_path_candidates(
    *,
    output_dir: Path,
    object_name: str,
    sample_label: str,
    model_label: str,
    run_dir: Path,
    checkpoint: Path,
    files: list[str],
    preprocessor: Path | None,
    args: argparse.Namespace,
) -> list[Path]:
    labels = [sample_label]
    if sample_label == "data":
        labels.append("real-data sample")
    return [
        cache_path(
            output_dir=output_dir,
            object_name=object_name,
            sample_label=label,
            model_label=model_label,
            run_dir=run_dir,
            checkpoint=checkpoint,
            files=files,
            preprocessor=preprocessor,
            args=args,
        )
        for label in labels
    ]


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
    checkpoint = resolve_checkpoint(run_dir, args.checkpoint_name)
    preprocessor = patch_missing_preprocessor(cfg, args, run_dir)
    force_common_eval_split(cfg)

    cache_candidates = cache_path_candidates(
        output_dir=output_dir,
        object_name=args.object,
        sample_label=sample_label,
        model_label=model_label,
        run_dir=run_dir,
        checkpoint=checkpoint,
        files=files,
        preprocessor=preprocessor,
        args=args,
    )
    cached = cache_candidates[0]
    existing_cache = next((path for path in cache_candidates if path.exists()), None)
    if existing_cache is not None and not args.force:
        log.info("Reading cache %s", existing_cache)
        item = np.load(existing_cache, allow_pickle=True)
        return {
            "original": item["original"],
            "reconstruction": item["reconstruction"],
            "indices": item["indices"],
            "feature_names": [str(value) for value in item["feature_names"]],
            "codebook_size": int(item["codebook_size"]),
            "checkpoint": str(item["checkpoint"]),
        }
    if args.cache_only:
        raise FileNotFoundError(
            f"Cache is missing for {sample_label} / {model_label}: {cached}. "
            "Rerun without --cache-only only if you want to evaluate the model again."
        )

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


def line_feature_set(args: argparse.Namespace) -> set[str]:
    raw = DEFAULT_LINE_FEATURES[args.object] if args.line_features == "auto" else args.line_features
    return {feature.strip().lower() for feature in raw.split(",") if feature.strip()}


def plot_feature_resolution(
    *,
    object_name: str,
    test_title: str,
    feature_name: str,
    samples: OrderedDict[str, OrderedDict[str, dict]],
    resolution_rows: list[dict],
    output_path: Path,
    log_y: bool,
) -> None:
    fig, axes = plt.subplots(1, len(samples), figsize=(6.7 * len(samples), 5.2), sharey=True)
    axes = np.atleast_1d(axes)
    legend_handles = None
    legend_labels = None
    for ax, sample_label in zip(axes, samples):
        for model_label in samples[sample_label]:
            selected = [
                row
                for row in resolution_rows
                if row["sample"] == sample_label
                and row["model"] == model_label
                and row["feature"] == feature_name
            ]
            x = np.asarray([row["bin_center"] for row in selected])
            y = np.asarray([row["relative_resolution"] for row in selected])
            valid = np.isfinite(y)
            if log_y:
                valid &= y > 0.0
            ax.plot(
                x[valid],
                y[valid],
                marker="o",
                linewidth=2.0,
                label=display_label(model_label),
                color=domain.COLORS.get(model_label),
                linestyle=domain.LINESTYLES.get(model_label, "-"),
            )
        ax.set_title(sample_label, fontsize=18)
        ax.set_xlabel(truth_axis_label(feature_name), fontsize=14)
        if log_y:
            ax.set_yscale("log")
        ax.grid(alpha=0.22)
        apply_hep_style(ax)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
    metric_kind = (
        resolution_rows[0].get("metric", binned_metric_kind(feature_name))
        if resolution_rows
        else binned_metric_kind(feature_name)
    )
    axes[0].set_ylabel(binned_metric_label(metric_kind, feature_name), fontsize=13)
    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.005),
            frameon=False,
            fontsize=8,
            ncol=4,
            handlelength=2.2,
            columnspacing=1.4,
        )
    fig.suptitle(f"{object_name}: {test_title}", fontsize=18)
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def apply_styles(full_reference_label: str | None) -> None:
    domain.COLORS.update(COLORS)
    domain.LINESTYLES.update(LINESTYLES)
    if full_reference_label:
        domain.COLORS[full_reference_label] = "#111111"
        domain.LINESTYLES[full_reference_label] = ":"


def outputs_with_display_labels(
    samples: OrderedDict[str, OrderedDict[str, dict]],
) -> OrderedDict[str, OrderedDict[str, dict]]:
    renamed: OrderedDict[str, OrderedDict[str, dict]] = OrderedDict()
    for sample_label, models in samples.items():
        renamed[sample_label] = OrderedDict(
            (display_label(model_label), item) for model_label, item in models.items()
        )
    return renamed


def apply_display_styles(model_labels: list[str]) -> None:
    for model_label in model_labels:
        shown = display_label(model_label)
        if shown == model_label:
            continue
        if model_label in domain.COLORS:
            domain.COLORS[shown] = domain.COLORS[model_label]
        if model_label in domain.LINESTYLES:
            domain.LINESTYLES[shown] = domain.LINESTYLES[model_label]


def binned_metric_kind(feature_name: str) -> str:
    """Choose a stable binned metric for the feature.

    pT and mass use the ratio convention from the reference jet-tokenizer
    plots. Scores, ID flags, isolation variables, and counts use residual IQR
    in their own units, which is better behaved than dividing by a tiny median.
    """
    if feature_name.lower() in {"pt", "mass"}:
        return "ratio_iqr_over_median"
    return "residual_iqr"


def binned_metric_label(metric_kind: str, feature_name: str) -> str:
    label = feature_label(feature_name)
    if metric_kind == "ratio_iqr_over_median":
        return rf"IQR / median of {label}$^{{reco}}$/{label}$^{{truth}}$"
    if metric_kind == "residual_iqr":
        return rf"IQR({label}$^{{reco}}$ - {label}$^{{truth}}$)"
    raise ValueError(metric_kind)


def binned_metric_values(
    original: np.ndarray,
    reconstruction: np.ndarray,
    bins: np.ndarray,
    *,
    metric_kind: str,
    min_bin_count: int,
    min_denominator: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    original = np.asarray(original, dtype=np.float64)
    reconstruction = np.asarray(reconstruction, dtype=np.float64)
    bins = np.asarray(bins, dtype=np.float64)
    centers = 0.5 * (bins[:-1] + bins[1:])
    values = np.full(len(centers), np.nan, dtype=np.float64)
    counts = np.zeros(len(centers), dtype=np.int64)
    medians = np.full(len(centers), np.nan, dtype=np.float64)
    residual_iqrs = np.full(len(centers), np.nan, dtype=np.float64)

    for bin_idx in range(len(centers)):
        in_bin = (original >= bins[bin_idx]) & (original < bins[bin_idx + 1])
        if bin_idx == len(centers) - 1:
            in_bin = (original >= bins[bin_idx]) & (original <= bins[bin_idx + 1])
        finite = in_bin & np.isfinite(original) & np.isfinite(reconstruction)
        if metric_kind == "ratio_iqr_over_median":
            finite &= np.abs(original) >= min_denominator
        counts[bin_idx] = int(np.count_nonzero(finite))
        if counts[bin_idx] < min_bin_count:
            continue

        truth = original[finite]
        reco = reconstruction[finite]
        medians[bin_idx] = float(np.median(truth))

        if metric_kind == "ratio_iqr_over_median":
            ratio = reco / truth
            ratio = ratio[np.isfinite(ratio)]
            if len(ratio) < min_bin_count:
                continue
            denominator = abs(float(np.median(ratio)))
            if denominator < min_denominator:
                continue
            q25, q75 = np.percentile(ratio, [25, 75])
            ratio_iqr = float(q75 - q25)
            residual_iqrs[bin_idx] = ratio_iqr
            values[bin_idx] = ratio_iqr / denominator
        elif metric_kind == "residual_iqr":
            residual = reco - truth
            q25, q75 = np.percentile(residual, [25, 75])
            residual_iqr = float(q75 - q25)
            residual_iqrs[bin_idx] = residual_iqr
            values[bin_idx] = residual_iqr
        else:
            raise ValueError(metric_kind)

    return centers, values, counts, medians, residual_iqrs


def evaluate_stage1_scan(
    *,
    test_key: str,
    test_title: str,
    samples: OrderedDict[str, OrderedDict[str, dict]],
    feature_names: list[str],
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict]]:
    metrics_rows: list[dict] = []
    resolution_rows: list[dict] = []
    for sample_label, models in samples.items():
        domain.verify_shared_inputs(sample_label, test_title, models, feature_names)
        reference_original, _ = domain.aligned_arrays(next(iter(models.values())), feature_names)
        feature_lookup = {name.lower(): idx for idx, name in enumerate(feature_names)}
        binned_features = [
            name for name in domain.BINNED_RESOLUTION_FEATURES if name.lower() in feature_lookup
        ]
        bins_by_feature = {}
        for name in binned_features:
            try:
                bins_by_feature[name] = domain.make_bins(
                    reference_original[:, feature_lookup[name.lower()]],
                    args.n_bins,
                )
            except ValueError as exc:
                log.warning(
                    "%s / %s: skipping binned plot for %s: %s",
                    test_title,
                    sample_label,
                    name,
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
                metric_kind = binned_metric_kind(feature_name)
                centers, values, counts, medians, residual_iqrs = binned_metric_values(
                    original[:, feature_idx],
                    reconstruction[:, feature_idx],
                    bins_by_feature[feature_name],
                    metric_kind=metric_kind,
                    min_bin_count=args.min_bin_count,
                    min_denominator=args.min_denominator,
                )
                for center, value, count, median, residual_iqr in zip(
                    centers, values, counts, medians, residual_iqrs
                ):
                    resolution_rows.append(
                        {
                            "test": test_key,
                            "sample": sample_label,
                            "model": model_label,
                            "feature": feature_name,
                            "metric": metric_kind,
                            "bin_center": float(center),
                            "relative_resolution": float(value),
                            "objects": int(count),
                            "median_original": float(median),
                            "residual_iqr": float(residual_iqr),
                        }
                    )
    return metrics_rows, resolution_rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    output_dir = Path(args.output_dir or f"results/google_stage1_scan_plots/{args.object}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = domain.choose_device(args.device)

    run_dirs = stage1_run_dirs(args)
    if not run_dirs:
        raise RuntimeError("No runs selected")
    verify_common_stage1_preprocessing(args, run_dirs)
    files_by_sample = sample_files(args)
    for sample_label, files in files_by_sample.items():
        file_list_digest = hashlib.sha256("\n".join(files).encode("utf-8")).hexdigest()
        log.info(
            "Common evaluation sample %s: %d files (list sha256=%s)",
            sample_label,
            len(files),
            file_list_digest,
        )

    full_reference_label = None
    for label in run_dirs:
        if label not in STAGE1_SPECS:
            full_reference_label = label
            break
    apply_styles(full_reference_label)
    apply_display_styles(list(run_dirs))

    checkpoints = {
        label: str(resolve_checkpoint(run_dir, args.checkpoint_name))
        for label, run_dir in run_dirs.items()
    }
    if args.validate_only:
        log.info(
            "Validation successful for %d runs and %d evaluation samples; "
            "no models were evaluated.",
            len(run_dirs),
            len(files_by_sample),
        )
        return

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

    plot_outputs = outputs_with_display_labels(all_outputs)
    first_models = next(iter(plot_outputs.values()))
    shared_features = domain.common_feature_names(*first_models.values())
    features_to_plot = selected_features(args.features, shared_features)
    if args.line_features.strip().lower() == "all":
        line_features = {feature.lower() for feature in features_to_plot}
    else:
        line_features = line_feature_set(args)
    domain.BINNED_RESOLUTION_FEATURES = tuple(
        feature for feature in features_to_plot if feature.lower() in line_features
    )

    test_key = f"{args.object}_google_stage1_scan"
    test_title = f"{args.object}: Google stage-1 tokenizer scan"
    metrics_rows, resolution_rows = evaluate_stage1_scan(
        test_key=test_key,
        test_title=test_title,
        samples=plot_outputs,
        feature_names=shared_features,
        args=args,
    )

    write_csv(output_dir / f"{test_key}_metrics.csv", metrics_rows)
    write_csv(output_dir / f"{test_key}_binned_resolution.csv", resolution_rows)

    binned_feature_keys = {feature.lower() for feature in domain.BINNED_RESOLUTION_FEATURES}
    for feature_name in features_to_plot:
        feature_key = feature_name.lower()
        if feature_key not in binned_feature_keys:
            plot_feature_metric_bars(
                test_title=test_title,
                feature_name=feature_name,
                samples=plot_outputs,
                metrics_rows=metrics_rows,
                output_path=output_dir / f"{test_key}_{safe_filename(feature_name)}_mae_rmse.png",
            )
            continue
        feature_rows = [row for row in resolution_rows if row["feature"] == feature_name]
        write_csv(
            output_dir / f"{test_key}_{safe_filename(feature_name)}_binned_resolution.csv",
            feature_rows,
        )
        plot_feature_resolution(
            object_name=args.object,
            test_title=f"{feature_label(feature_name)} resolution",
            feature_name=feature_name,
            samples=plot_outputs,
            resolution_rows=feature_rows,
            output_path=output_dir / f"{test_key}_{safe_filename(feature_name)}_resolution.png",
            log_y=args.log_y,
        )

    report_lines = domain.report_section(
        title=test_title,
        models=[display_label(label) for label in run_dirs],
        metrics_rows=metrics_rows,
        resolution_rows=resolution_rows,
    )
    (output_dir / "comparison_report.md").write_text("\n".join(report_lines).rstrip() + "\n")
    (output_dir / "comparison_manifest.json").write_text(
        json.dumps(
            {
                "object": args.object,
                "runs": {label: str(path) for label, path in run_dirs.items()},
                "display_labels": {
                    label: display_label(label) for label in run_dirs
                },
                "checkpoints": checkpoints,
                "samples": files_by_sample,
                "features": features_to_plot,
                "line_features": sorted(line_features),
                "metric": (
                    "Auto binned metric: pt and mass use "
                    "IQR(reco/original)/median(reco/original); scores, ID, "
                    "isolation variables, and counts use IQR(reco-original)."
                ),
                "common_eval_split": "train_frac=0, val_frac=1, test_frac=0",
            },
            indent=2,
        )
        + "\n"
    )
    log.info("Wrote %s Google stage-1 comparison to %s", args.object, output_dir)


if __name__ == "__main__":
    main()
