#!/usr/bin/env python3
"""Report decoder outputs that would explode during inverse log preprocessing."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_vqvae_tokenizer import (  # noqa: E402
    choose_device,
    dataset_kwargs_from_cfg,
    feature_names_from_cfg,
    find_checkpoint,
    to_device,
    transform_list_and_cst_fn_from_cfg,
)
from heptokens.data.atlas_event_mappable import AtlasEventMapDataset  # noqa: E402
from heptokens.models.vq_vae import LitVqVae  # noqa: E402


log = logging.getLogger(__name__)


DEFAULT_RUNS = {
    "electrons": {
        "mc_only": "results/atlas_event_tokenizers_1806_nonjet_logstd_capacity_scan/electrons_logstd_dim8_cb4096_q4",
        "mcdata": "results/atlas_event_tokenizers_0107_logstd_mc_realdata/electrons_logstd_dim8_cb4096_q4",
    },
    "muons": {
        "mc_only": "results/atlas_event_tokenizers_1806_nonjet_logstd_capacity_scan/muons_logstd_dim8_cb4096_q4",
        "mcdata": "results/atlas_event_tokenizers_0107_logstd_mc_realdata/muons_logstd_dim8_cb4096_q4",
    },
    "photons": {
        "mc_only": "results/atlas_event_tokenizers_1806_nonjet_logstd_capacity_scan/photons_logstd_dim8_cb4096_q4",
        "mcdata": "results/atlas_event_tokenizers_0107_logstd_mc_realdata/photons_logstd_dim8_cb4096_q4",
    },
    "jets": {
        "mc_only": "results/atlas_event_tokenizers_1606_jets_logstd_capacity_scan/jets_logstd_dim8_cb4096_q4",
        "mcdata": "results/atlas_event_tokenizers_0107_logstd_mc_realdata/jets_logstd_dim8_cb4096_q4",
    },
    "tracks": {
        "mc_only": "results/atlas_event_tokenizers_1906_tracks_logstd_tests/tracks_logstd_no_ndoflog_dim8_cb8192_q4",
        "mcdata": "results/atlas_event_tokenizers_0107_logstd_mc_realdata/tracks_logstd_dim8_cb8192_q4",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find reconstructed log-space values that cause inverse-transform overflow."
    )
    parser.add_argument("--object", default="electrons", choices=sorted(DEFAULT_RUNS))
    parser.add_argument("--mc-only-run-dir", help="Override MC-only run directory.")
    parser.add_argument("--mcdata-run-dir", help="Override MC+data run directory.")
    parser.add_argument("--h5-files", nargs="+")
    parser.add_argument("--mc-dir", default="/home/zephyr/Data/viviana/bnl-treasure/data/h5")
    parser.add_argument("--n-files", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-events-per-file", type=int)
    parser.add_argument("--max-valid-objects", type=int, default=500_000)
    parser.add_argument(
        "--log-threshold",
        type=float,
        default=20.0,
        help="Report reconstructed log(feature+offset) values above this threshold.",
    )
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--output-dir",
        default="results/fair_mc_vs_mcdata_eval/decoder_outliers",
    )
    return parser.parse_args()


def fixed_files(directory: str, n_files: int) -> list[str]:
    return sorted(
        str(path)
        for path in Path(directory).glob("*.h5")
        if path.is_file() and path.stat().st_size > 0
    )[:n_files]


def log_indices(transformer) -> list[int]:
    log_transformer = getattr(transformer, "log_transformer", None)
    if log_transformer is None:
        return []
    out: list[int] = []
    for config in getattr(log_transformer, "feature_configs", []):
        raw = config.get("indices", [])
        if isinstance(raw, int):
            out.append(raw)
        else:
            out.extend(int(value) for value in raw)
    return sorted(set(out))


def final_transformer(transformer):
    return getattr(transformer, "final_transformer", transformer)


def n_features_from_preprocessor(transformer) -> int:
    if getattr(transformer, "n_features_in_", None) is not None:
        return int(transformer.n_features_in_)
    final = final_transformer(transformer)
    if getattr(final, "n_features_in_", None) is not None:
        return int(final.n_features_in_)
    if hasattr(final, "mean_"):
        return int(len(final.mean_))
    raise ValueError(f"Could not infer feature count from {type(transformer).__name__}")


def inverse_logged_space(transformer, values: np.ndarray) -> np.ndarray:
    final = final_transformer(transformer)
    if hasattr(final, "inverse_transform"):
        return final.inverse_transform(values)
    return values


def physical_from_logged(logged_values: np.ndarray, offset: float = 1.0) -> np.ndarray:
    return np.exp(np.clip(logged_values, -50, 50)) - offset


def analyze_run(
    *,
    label: str,
    run_dir: Path,
    h5_files: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict], dict]:
    cfg = OmegaConf.load(run_dir / "full_config.yaml")
    transforms, preprocessor = transform_list_and_cst_fn_from_cfg(cfg)
    if preprocessor is None:
        raise RuntimeError(f"{label}: no preprocessing transformer found")
    log_feature_indices = log_indices(preprocessor)
    if not log_feature_indices:
        raise RuntimeError(f"{label}: preprocessor has no log-transformed feature indices")

    checkpoint = find_checkpoint(run_dir, None)
    model = LitVqVae.load_from_checkpoint(checkpoint, map_location=device)
    model.to(device)
    model.eval()

    loader_args = SimpleNamespace(
        h5_files=h5_files,
        num_events_per_file=args.num_events_per_file,
    )
    data_paths, dataset_kwargs = dataset_kwargs_from_cfg(cfg, loader_args, run_dir)
    feature_names = feature_names_from_cfg(cfg, n_features_from_preprocessor(preprocessor))

    rows: list[dict] = []
    max_by_feature = {
        feature_names[idx]: {
            "max_reco_logged": float("-inf"),
            "n_above_threshold": 0,
            "n_valid": 0,
        }
        for idx in log_feature_indices
    }
    global_object_index = 0

    with torch.no_grad():
        for data_path in data_paths:
            if global_object_index >= args.max_valid_objects:
                break
            dataset = AtlasEventMapDataset(data_path, **dataset_kwargs)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
            )
            for batch in loader:
                if global_object_index >= args.max_valid_objects:
                    break
                for transform in transforms:
                    batch = transform(batch)
                batch = to_device(batch, device)

                z_q, indices, _ = model.encode(batch)
                reco = model.decode(z_q, batch)
                mask = batch["mask"].bool()

                original_std = batch["csts"][mask].detach().cpu().float().numpy()
                reco_std = reco[mask].detach().cpu().float().numpy()
                code_indices = indices[mask].detach().cpu().long().numpy()
                n_batch = len(reco_std)
                if n_batch == 0:
                    continue

                remaining = args.max_valid_objects - global_object_index
                if n_batch > remaining:
                    original_std = original_std[:remaining]
                    reco_std = reco_std[:remaining]
                    code_indices = code_indices[:remaining]
                    n_batch = remaining

                original_logged = inverse_logged_space(preprocessor, original_std)
                reco_logged = inverse_logged_space(preprocessor, reco_std)

                for idx in log_feature_indices:
                    name = feature_names[idx]
                    values = reco_logged[:, idx]
                    finite = np.isfinite(values)
                    max_by_feature[name]["n_valid"] += int(len(values))
                    if finite.any():
                        max_by_feature[name]["max_reco_logged"] = max(
                            max_by_feature[name]["max_reco_logged"],
                            float(np.max(values[finite])),
                        )
                    bad = (~finite) | (values > args.log_threshold)
                    max_by_feature[name]["n_above_threshold"] += int(np.count_nonzero(bad))
                    bad_indices = np.where(bad)[0]
                    for local_idx in bad_indices[: args.top_k]:
                        logged_value = float(reco_logged[local_idx, idx])
                        rows.append(
                            {
                                "model": label,
                                "feature": name,
                                "global_object_index": global_object_index + int(local_idx),
                                "source_file": str(data_path),
                                "original_logged": float(original_logged[local_idx, idx]),
                                "reco_logged": logged_value,
                                "original_physical": float(
                                    physical_from_logged(original_logged[local_idx : local_idx + 1, idx])[0]
                                ),
                                "reco_physical_clipped": float(
                                    physical_from_logged(np.array([logged_value]))[0]
                                ),
                                "reco_standardized": float(reco_std[local_idx, idx]),
                                "original_standardized": float(original_std[local_idx, idx]),
                                "codes": " ".join(str(int(x)) for x in code_indices[local_idx]),
                            }
                        )

                global_object_index += n_batch

    rows = sorted(
        rows,
        key=lambda row: (
            row["model"],
            row["feature"],
            -row["reco_logged"] if np.isfinite(row["reco_logged"]) else float("-inf"),
        ),
    )
    return rows, max_by_feature


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    device = choose_device(args.device)
    old_run = Path(args.mc_only_run_dir or DEFAULT_RUNS[args.object]["mc_only"]).resolve()
    new_run = Path(args.mcdata_run_dir or DEFAULT_RUNS[args.object]["mcdata"]).resolve()
    h5_files = list(args.h5_files) if args.h5_files else fixed_files(args.mc_dir, args.n_files)
    if not h5_files:
        raise FileNotFoundError("No H5 files selected")

    output_dir = Path(args.output_dir) / args.object
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "files_used.txt").write_text("\n".join(h5_files) + "\n")

    all_rows: list[dict] = []
    summaries: dict[str, dict] = {}
    for label, run_dir in [("MC-only", old_run), ("MC+data", new_run)]:
        rows, summary = analyze_run(
            label=label,
            run_dir=run_dir,
            h5_files=h5_files,
            args=args,
            device=device,
        )
        all_rows.extend(rows)
        summaries[label] = summary

    outlier_path = output_dir / "decoder_log_outliers.csv"
    with outlier_path.open("w", newline="") as handle:
        fieldnames = [
            "model",
            "feature",
            "global_object_index",
            "source_file",
            "original_logged",
            "reco_logged",
            "original_physical",
            "reco_physical_clipped",
            "original_standardized",
            "reco_standardized",
            "codes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    summary_path = output_dir / "decoder_log_outlier_summary.csv"
    with summary_path.open("w", newline="") as handle:
        fieldnames = ["model", "feature", "n_valid", "n_above_threshold", "fraction", "max_reco_logged"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for label, summary in summaries.items():
            for feature, item in summary.items():
                n_valid = item["n_valid"]
                writer.writerow(
                    {
                        "model": label,
                        "feature": feature,
                        "n_valid": n_valid,
                        "n_above_threshold": item["n_above_threshold"],
                        "fraction": item["n_above_threshold"] / n_valid if n_valid else 0.0,
                        "max_reco_logged": item["max_reco_logged"],
                    }
                )

    print(f"Wrote {outlier_path}")
    print(f"Wrote {summary_path}")
    for label, summary in summaries.items():
        print(f"\n{label}")
        for feature, item in summary.items():
            n_valid = item["n_valid"]
            print(
                f"  {feature}: above={item['n_above_threshold']}/{n_valid} "
                f"({item['n_above_threshold'] / n_valid if n_valid else 0:.3e}), "
                f"max logged reco={item['max_reco_logged']:.5g}"
            )


if __name__ == "__main__":
    main()
