#!/usr/bin/env python3
"""Locate decoder values that overflow the inverse electron preprocessing.

The tokenizer is trained in standardized space.  For log-transformed features,
decoding to physical units applies

    standardized -> logged physical value -> exp(logged value)

This diagnostic stops before ``exp`` and reports any decoder outputs outside
the finite exponential range.  It therefore identifies the run, checkpoint,
sample, feature, and object rows responsible for an overflow without hiding
the problem by clipping.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from analyze_vqvae_tokenizer import (
    choose_device,
    collect_diagnostics_for_h5_files,
    feature_names_from_cfg,
    transform_list_and_cst_fn_from_cfg,
)
from compare_electron_sampling_study import DEFAULT_MODELS
from heptokens.models.vq_vae import LitVqVae


log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find values that overflow electron inverse log preprocessing."
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
        default="results/atlas_electron_sampling_study/inverse_overflow_debug",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="LABEL=RUN_DIR",
        help="Model to inspect; repeat to override the default sampling-study runs.",
    )
    parser.add_argument(
        "--samples", nargs="+", choices=["mc", "realdata"], default=["mc", "realdata"]
    )
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best")
    parser.add_argument("--n-files", type=int, default=20)
    parser.add_argument("--num-events-per-file", type=int)
    parser.add_argument("--max-valid-objects", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def parse_models(values: list[str]) -> OrderedDict[str, Path]:
    raw: OrderedDict[str, str] = OrderedDict()
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


def checkpoint_path(run_dir: Path, name: str) -> Path:
    path = run_dir / "checkpoints" / f"{name}.ckpt"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def logged_feature_indices(transformer) -> list[int]:
    log_transformer = getattr(transformer, "log_transformer", None)
    configs = getattr(log_transformer, "feature_configs", None)
    if configs is None:
        return []

    indices: list[int] = []
    for config in configs:
        raw = config["indices"]
        indices.extend([raw] if isinstance(raw, int) else list(raw))
    return sorted(set(int(index) for index in indices))


def before_log_inverse(transformer, values: np.ndarray) -> np.ndarray:
    final_transformer = getattr(transformer, "final_transformer", None)
    if final_transformer is None:
        raise TypeError(
            f"Expected a CompositeTransformer, found {type(transformer).__name__}"
        )
    # Preserve float32 here: this is the dtype used by the normal analysis path,
    # and np.exp(float32) overflows near 88.7 rather than near 709.
    return final_transformer.inverse_transform(values)


def inspect_arrays(
    *,
    model_label: str,
    sample_label: str,
    checkpoint: Path,
    transformer,
    feature_names: list[str],
    original_std: np.ndarray,
    reconstruction_std: np.ndarray,
    top_k: int,
) -> dict:
    original_preexp = before_log_inverse(transformer, original_std)
    reconstruction_preexp = before_log_inverse(transformer, reconstruction_std)
    log_indices = logged_feature_indices(transformer)

    dtype = reconstruction_preexp.dtype
    exp_limit = float(np.log(np.finfo(dtype).max))
    result = {
        "model": model_label,
        "sample": sample_label,
        "checkpoint": str(checkpoint),
        "n_objects": int(len(original_std)),
        "pre_exp_dtype": str(dtype),
        "finite_exp_limit": exp_limit,
        "logged_feature_indices": log_indices,
        "features": [],
    }

    for index, feature_name in enumerate(feature_names):
        reco_std = reconstruction_std[:, index]
        reco_preexp = reconstruction_preexp[:, index]
        original_pre = original_preexp[:, index]
        is_logged = index in log_indices
        unsafe_original = (~np.isfinite(original_pre)) | (original_pre > exp_limit)
        unsafe_reconstruction = (~np.isfinite(reco_preexp)) | (
            reco_preexp > exp_limit
        )
        if not is_logged:
            unsafe_original = ~np.isfinite(original_pre)
            unsafe_reconstruction = ~np.isfinite(reco_preexp)

        finite_reco_preexp = reco_preexp[np.isfinite(reco_preexp)]
        feature = {
            "name": feature_name,
            "index": index,
            "log_transformed": is_logged,
            "original_standardized_min": float(np.nanmin(original_std[:, index])),
            "original_standardized_max": float(np.nanmax(original_std[:, index])),
            "reconstruction_standardized_min": float(np.nanmin(reco_std)),
            "reconstruction_standardized_max": float(np.nanmax(reco_std)),
            "original_pre_exp_min": float(np.nanmin(original_pre)),
            "original_pre_exp_max": float(np.nanmax(original_pre)),
            "reconstruction_pre_exp_min": float(np.min(finite_reco_preexp))
            if len(finite_reco_preexp)
            else None,
            "reconstruction_pre_exp_max": float(np.max(finite_reco_preexp))
            if len(finite_reco_preexp)
            else None,
            "unsafe_original_count": int(np.count_nonzero(unsafe_original)),
            "unsafe_reconstruction_count": int(
                np.count_nonzero(unsafe_reconstruction)
            ),
        }

        unsafe_rows = np.flatnonzero(unsafe_reconstruction)
        if len(unsafe_rows):
            order = unsafe_rows[
                np.argsort(reco_preexp[unsafe_rows], kind="stable")[::-1]
            ][:top_k]
            feature["largest_unsafe_rows"] = [
                {
                    "row": int(row),
                    "original_standardized": float(original_std[row, index]),
                    "reconstruction_standardized": float(reco_std[row]),
                    "original_pre_exp": float(original_pre[row]),
                    "reconstruction_pre_exp": float(reco_preexp[row]),
                }
                for row in order
            ]
        result["features"].append(feature)

    unsafe_features = [
        item
        for item in result["features"]
        if item["unsafe_original_count"] or item["unsafe_reconstruction_count"]
    ]
    result["unsafe_feature_count"] = len(unsafe_features)
    result["unsafe_original_value_count"] = int(
        sum(item["unsafe_original_count"] for item in unsafe_features)
    )
    result["unsafe_reconstruction_value_count"] = int(
        sum(item["unsafe_reconstruction_count"] for item in unsafe_features)
    )
    return result


def collect_standardized(
    *,
    run_dir: Path,
    checkpoint: Path,
    files: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str], object]:
    cfg = OmegaConf.load(run_dir / "full_config.yaml")
    _, transformer = transform_list_and_cst_fn_from_cfg(cfg)
    if transformer is None:
        raise RuntimeError(f"No preprocessing transformer found in {run_dir}")

    model = LitVqVae.load_from_checkpoint(checkpoint, map_location=device)
    model.to(device)
    model.eval()
    original_std, reconstruction_std, _, _ = collect_diagnostics_for_h5_files(
        cfg=cfg,
        model=model,
        h5_files=files,
        split=args.split,
        num_events_per_file=args.num_events_per_file,
        # Deliberately stop before inverse_transform so overflow cannot be hidden.
        cst_inverse_transformer=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        max_valid_objects=args.max_valid_objects,
    )
    feature_names = feature_names_from_cfg(cfg, original_std.shape[1])
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return original_std, reconstruction_std, feature_names, transformer


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    models = parse_models(args.model)

    samples: OrderedDict[str, list[str]] = OrderedDict()
    if "mc" in args.samples:
        samples["MC sample"] = fixed_files(args.mc_dir, args.n_files)
    if "realdata" in args.samples:
        samples["real-data sample"] = fixed_files(args.data_dir, args.n_files)
    for sample_label, files in samples.items():
        if not files:
            raise FileNotFoundError(f"No H5 files found for {sample_label}")

    summaries = []
    for model_label, run_dir in models.items():
        checkpoint = checkpoint_path(run_dir, args.checkpoint)
        for sample_label, files in samples.items():
            log.info("Inspecting %s on %s using %s", model_label, sample_label, checkpoint)
            original_std, reconstruction_std, feature_names, transformer = collect_standardized(
                run_dir=run_dir,
                checkpoint=checkpoint,
                files=files,
                args=args,
                device=device,
            )
            summary = inspect_arrays(
                model_label=model_label,
                sample_label=sample_label,
                checkpoint=checkpoint,
                transformer=transformer,
                feature_names=feature_names,
                original_std=original_std,
                reconstruction_std=reconstruction_std,
                top_k=args.top_k,
            )
            summaries.append(summary)

            print(f"\n{model_label} | {sample_label} | {checkpoint.name}")
            print(
                f"  objects={summary['n_objects']:,} dtype={summary['pre_exp_dtype']} "
                f"exp_limit={summary['finite_exp_limit']:.5g}"
            )
            for feature in summary["features"]:
                if (
                    feature["log_transformed"]
                    or feature["unsafe_original_count"]
                    or feature["unsafe_reconstruction_count"]
                ):
                    print(
                        f"  {feature['name']}: pre-exp original "
                        f"[{feature['original_pre_exp_min']:.5g}, "
                        f"{feature['original_pre_exp_max']:.5g}], reco "
                        f"[{feature['reconstruction_pre_exp_min']:.5g}, "
                        f"{feature['reconstruction_pre_exp_max']:.5g}], "
                        f"unsafe input={feature['unsafe_original_count']}, "
                        f"unsafe reco={feature['unsafe_reconstruction_count']}"
                    )

    report_path = output_dir / f"inverse_overflow_{args.checkpoint}.json"
    report_path.write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"\nWrote {report_path}")

    unsafe = [
        item
        for item in summaries
        if item["unsafe_original_value_count"]
        or item["unsafe_reconstruction_value_count"]
    ]
    if unsafe:
        print("\nOverflow source found:")
        for item in unsafe:
            names = [
                f"{feature['name']} "
                f"(input={feature['unsafe_original_count']}, "
                f"reco={feature['unsafe_reconstruction_count']})"
                for feature in item["features"]
                if feature["unsafe_original_count"]
                or feature["unsafe_reconstruction_count"]
            ]
            print(
                f"  {item['model']} | {item['sample']} | "
                f"{Path(item['checkpoint']).name}: {', '.join(names)}"
            )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
