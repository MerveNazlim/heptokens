"""Select one reproducible MC+data file subset balanced across all object masks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path

import h5py
import numpy as np
from omegaconf import OmegaConf

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select about 25% of MC and 25% of data files while minimizing the "
            "MC/data valid-object count imbalance across every configured object type."
        )
    )
    parser.add_argument("--mc-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--datamodule-config",
        type=Path,
        default=Path("configs/datamodule/atlas_event_object.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("stage1_subsets"))
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-trials", type=int, default=50_000)
    parser.add_argument("--swap-trials", type=int, default=50_000)
    parser.add_argument("--mask-chunk-events", type=int, default=16_384)
    parser.add_argument("--exclude-mc-prefix", default="DAOD_PHYSLITE.370016")
    parser.add_argument(
        "--gcs-prefix",
        default="gs://fcc-teststorage/input_files/h5",
    )
    parser.add_argument(
        "--reuse-counts",
        action="store_true",
        help="Reuse output-dir/stage1_file_counts.csv instead of rescanning HDF5 masks.",
    )
    return parser.parse_args()


def configured_masks(config_path: Path) -> dict[str, str]:
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    masks = {
        str(collection["object_name"]): str(collection["mask_input"])
        for collection in config.get("object_collections") or []
    }
    if not masks:
        raise ValueError(f"No object masks found in {config_path}")
    return masks


def h5_accessor(handle: h5py.File, h5_path: str):
    """Return a sliceable dataset/field accessor and its number of events."""
    parts = h5_path.strip("/").split("/")
    node = handle
    field = None
    for index, part in enumerate(parts):
        if isinstance(node, h5py.Dataset):
            field = "/".join(parts[index:])
            break
        if part not in node:
            raise KeyError(f"Missing HDF5 path {h5_path!r}")
        node = node[part]
    if not isinstance(node, h5py.Dataset):
        raise ValueError(f"HDF5 path {h5_path!r} is not a dataset or field")
    return (node.fields(field) if field else node), int(node.shape[0])


def count_mask(handle: h5py.File, mask_path: str, chunk_events: int) -> int:
    accessor, n_events = h5_accessor(handle, mask_path)
    total = 0
    for start in range(0, n_events, chunk_events):
        total += int(np.count_nonzero(accessor[start : start + chunk_events]))
    return total


def discover_files(args: argparse.Namespace) -> list[tuple[str, Path]]:
    mc_files = sorted(
        path
        for path in args.mc_dir.glob("*.h5")
        if path.stat().st_size > 0 and not path.name.startswith(args.exclude_mc_prefix)
    )
    data_files = sorted(
        path for path in args.data_dir.glob("*.h5") if path.stat().st_size > 0
    )
    if not mc_files or not data_files:
        raise RuntimeError(f"Missing input files: MC={len(mc_files)} data={len(data_files)}")
    return [("mc", path) for path in mc_files] + [("data", path) for path in data_files]


def scan_counts(
    files: list[tuple[str, Path]],
    masks: dict[str, str],
    chunk_events: int,
) -> list[dict[str, object]]:
    rows = []
    for index, (domain, path) in enumerate(files, start=1):
        log.info("[%d/%d] %s", index, len(files), path)
        with h5py.File(path, "r") as handle:
            counts = {
                object_name: count_mask(handle, mask_path, chunk_events)
                for object_name, mask_path in masks.items()
            }
        rows.append({"domain": domain, "path": str(path), **counts})
    return rows


def write_counts(path: Path, rows: list[dict[str, object]], objects: list[str]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["domain", "path", *objects])
        writer.writeheader()
        writer.writerows(rows)


def read_counts(path: Path, objects: list[str]) -> list[dict[str, object]]:
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        for object_name in objects:
            row[object_name] = int(row[object_name])
    return rows


def imbalance_score(mc_counts: np.ndarray, data_counts: np.ndarray) -> tuple[float, float]:
    """Minimize worst then RMS symmetric log-ratio across object types."""
    log_ratios = np.abs(np.log((mc_counts + 1.0) / (data_counts + 1.0)))
    return float(log_ratios.max()), float(np.sqrt(np.mean(log_ratios**2)))


def optimize_subset(
    mc_values: np.ndarray,
    data_values: np.ndarray,
    n_mc: int,
    n_data: int,
    seed: int,
    random_trials: int,
    swap_trials: int,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    best_mc = None
    best_data = None
    best_score = (float("inf"), float("inf"))

    for _ in range(random_trials):
        mc_indices = rng.choice(len(mc_values), size=n_mc, replace=False)
        data_indices = rng.choice(len(data_values), size=n_data, replace=False)
        score = imbalance_score(
            mc_values[mc_indices].sum(axis=0),
            data_values[data_indices].sum(axis=0),
        )
        if score < best_score:
            best_mc, best_data, best_score = mc_indices, data_indices, score

    if best_mc is None or best_data is None:
        raise RuntimeError("No candidate subsets were generated")

    selected_mc = set(int(index) for index in best_mc)
    selected_data = set(int(index) for index in best_data)
    mc_sum = mc_values[sorted(selected_mc)].sum(axis=0)
    data_sum = data_values[sorted(selected_data)].sum(axis=0)

    for _ in range(swap_trials):
        domain = "mc" if rng.random() < 0.5 else "data"
        if domain == "mc":
            remove = int(rng.choice(sorted(selected_mc)))
            available = np.asarray(
                [index for index in range(len(mc_values)) if index not in selected_mc]
            )
            add = int(rng.choice(available))
            candidate_mc_sum = mc_sum - mc_values[remove] + mc_values[add]
            candidate_data_sum = data_sum
        else:
            remove = int(rng.choice(sorted(selected_data)))
            available = np.asarray(
                [index for index in range(len(data_values)) if index not in selected_data]
            )
            add = int(rng.choice(available))
            candidate_mc_sum = mc_sum
            candidate_data_sum = data_sum - data_values[remove] + data_values[add]

        score = imbalance_score(candidate_mc_sum, candidate_data_sum)
        if score < best_score:
            best_score = score
            if domain == "mc":
                selected_mc.remove(remove)
                selected_mc.add(add)
                mc_sum = candidate_mc_sum
            else:
                selected_data.remove(remove)
                selected_data.add(add)
                data_sum = candidate_data_sum

    return np.asarray(sorted(selected_mc)), np.asarray(sorted(selected_data)), best_score


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not 0 < args.fraction <= 1:
        raise ValueError("--fraction must be in (0, 1]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    masks = configured_masks(args.datamodule_config)
    objects = list(masks)
    counts_path = args.output_dir / "stage1_file_counts.csv"

    if args.reuse_counts and counts_path.exists():
        log.info("Reusing %s", counts_path)
        rows = read_counts(counts_path, objects)
    else:
        files = discover_files(args)
        rows = scan_counts(files, masks, args.mask_chunk_events)
        write_counts(counts_path, rows, objects)
        log.info("Wrote %s", counts_path)

    mc_rows = [row for row in rows if row["domain"] == "mc"]
    data_rows = [row for row in rows if row["domain"] == "data"]
    mc_values = np.asarray([[row[name] for name in objects] for row in mc_rows], dtype=np.int64)
    data_values = np.asarray([[row[name] for name in objects] for row in data_rows], dtype=np.int64)
    n_mc = max(1, int(len(mc_rows) * args.fraction + 0.5))
    n_data = max(1, int(len(data_rows) * args.fraction + 0.5))

    mc_indices, data_indices, score = optimize_subset(
        mc_values,
        data_values,
        n_mc,
        n_data,
        args.seed,
        args.random_trials,
        args.swap_trials,
    )
    selected_mc = [mc_rows[int(index)] for index in mc_indices]
    selected_data = [data_rows[int(index)] for index in data_indices]
    selected = [*selected_mc, *selected_data]

    relative_lines = [
        f"{'MC' if row['domain'] == 'mc' else 'data'}/{Path(str(row['path'])).name}"
        for row in selected
    ]
    relative_path = args.output_dir / "stage1_subset_seed42_relative.txt"
    relative_path.write_text("\n".join(relative_lines) + "\n")
    local_path = args.output_dir / "stage1_subset_seed42_local.txt"
    local_path.write_text("\n".join(str(row["path"]) for row in selected) + "\n")
    gcs_path = args.output_dir / "stage1_subset_seed42_gcs.txt"
    gcs_path.write_text(
        "\n".join(f"{args.gcs_prefix.rstrip('/')}/{line}" for line in relative_lines) + "\n"
    )

    digest = hashlib.sha256(relative_path.read_bytes()).hexdigest()
    (args.output_dir / "stage1_subset_seed42.sha256").write_text(
        f"{digest}  {relative_path.name}\n"
    )

    mc_totals = np.asarray([sum(int(row[name]) for row in selected_mc) for name in objects])
    data_totals = np.asarray([sum(int(row[name]) for row in selected_data) for name in objects])
    report = {
        "seed": args.seed,
        "fraction": args.fraction,
        "available_files": {"mc": len(mc_rows), "data": len(data_rows)},
        "selected_files": {"mc": len(selected_mc), "data": len(selected_data)},
        "subset_sha256": digest,
        "score": {"worst_abs_log_ratio": score[0], "rms_abs_log_ratio": score[1]},
        "objects": {
            name: {
                "mc": int(mc_totals[index]),
                "data": int(data_totals[index]),
                "mc_data_ratio": float((mc_totals[index] + 1) / (data_totals[index] + 1)),
                "relative_difference": float(
                    abs(mc_totals[index] - data_totals[index])
                    / max((mc_totals[index] + data_totals[index]) / 2, 1)
                ),
            }
            for index, name in enumerate(objects)
        },
    }
    report_path = args.output_dir / "stage1_subset_seed42_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(
        f"Selected files: MC={len(selected_mc)}/{len(mc_rows)}, "
        f"data={len(selected_data)}/{len(data_rows)}"
    )
    print(f"Subset SHA256: {digest}")
    print(f"{'object':12s} {'MC':>14s} {'data':>14s} {'MC/data':>10s} {'rel.diff':>10s}")
    for index, name in enumerate(objects):
        ratio = (mc_totals[index] + 1) / (data_totals[index] + 1)
        relative = report["objects"][name]["relative_difference"]
        print(
            f"{name:12s} {mc_totals[index]:14d} {data_totals[index]:14d} "
            f"{ratio:10.3f} {relative:10.3f}"
        )
    print(f"Wrote manifests and report under {args.output_dir}")


if __name__ == "__main__":
    main()
