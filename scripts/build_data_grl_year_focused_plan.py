#!/usr/bin/env python3
"""Build a deterministic, event-balanced, year-focused H5 conversion plan.

The input H5 files are inspected locally (for example through PNFS), while the
generated group manifests contain the corresponding GCS object URLs.  File
selection is deterministic for a fixed seed and is based on event counts, not
file counts.  This makes it suitable for a controlled paper dataset where one
data-taking year should dominate without changing the downstream event split.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import random
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


RUN_RANGES = {
    2015: (266_904, 284_484),
    2016: (296_939, 311_481),
    2017: (324_320, 341_649),
    2018: (348_197, 364_485),
}


@dataclass(frozen=True)
class FileRecord:
    local_path: str
    source_uri: str
    size_bytes: int
    events: int
    year: int | None
    year_source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--gcs-prefix", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-year", type=int, default=2016)
    parser.add_argument(
        "--primary-fraction",
        type=float,
        default=0.8,
        help="Requested fraction of selected events from the primary year.",
    )
    parser.add_argument(
        "--target-events",
        type=int,
        default=110_000_000,
        help="Target total events before the deterministic 90/10 event split.",
    )
    parser.add_argument("--groups", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Processes used to inspect H5 headers on the source filesystem.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1_000,
        help="Print one scan progress line after this many inspected files.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def python_scalar(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def normalize_year(value) -> int | None:
    value = python_scalar(value)
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        number = int(value)
        if 15 <= number <= 99:
            number += 2000
        return number if 2000 <= number <= 2100 else None
    text = str(value).strip().lower()
    if not text:
        return None
    match = re.search(r"(?:data)?(?:20)?(15|16|17|18)(?:\D|$)", text)
    if match:
        return 2000 + int(match.group(1))
    match = re.search(r"\b(20\d{2})\b", text)
    if match:
        return int(match.group(1))
    return None


def year_from_run_number(run_number: int) -> int | None:
    for year, (minimum, maximum) in RUN_RANGES.items():
        if minimum <= run_number <= maximum:
            return year
    return None


def first_scalar_dataset(handle: h5py.File, candidates: Iterable[str]):
    for name in candidates:
        if name not in handle:
            continue
        dataset = handle[name]
        if not isinstance(dataset, h5py.Dataset) or dataset.size == 0:
            continue
        value = dataset[0]
        if getattr(value, "dtype", None) is not None and value.dtype.fields:
            for field in ("runNumber", "run_number"):
                if field in value.dtype.fields:
                    return python_scalar(value[field])
        array = np.asarray(value).reshape(-1)
        if array.size:
            return python_scalar(array[0])
    return None


def infer_year(handle: h5py.File, path: Path) -> tuple[int | None, str]:
    metadata_attrs = handle["metadata"].attrs if "metadata" in handle else {}
    for label, attrs in (("metadata", metadata_attrs), ("root", handle.attrs)):
        for key in ("data_taking_year", "data_year", "year"):
            year = normalize_year(attrs.get(key))
            if year is not None:
                return year, f"{label}.{key}"
    for label, value in (
        ("root.sample_label", handle.attrs.get("sample_label")),
        ("metadata.sample_label", metadata_attrs.get("sample_label")),
        ("path", str(path)),
    ):
        year = normalize_year(value)
        if year is not None:
            return year, label
    run_number = first_scalar_dataset(
        handle,
        (
            "atlas/event/runNumber",
            "common/event/runNumber",
            "event/runNumber",
            "runNumber",
        ),
    )
    if run_number is not None:
        year = year_from_run_number(int(run_number))
        if year is not None:
            return year, "runNumber"
    return None, "unknown"


def infer_events(handle: h5py.File) -> int:
    metadata_attrs = handle["metadata"].attrs if "metadata" in handle else {}
    for attrs in (metadata_attrs, handle.attrs):
        value = python_scalar(attrs.get("n_events"))
        if value is not None and int(value) > 0:
            return int(value)
    for name in (
        "atlas/event/eventNumber",
        "atlas/event/runNumber",
        "common/event/mu",
        "common/met/pt",
    ):
        if name in handle and isinstance(handle[name], h5py.Dataset):
            if handle[name].shape and handle[name].shape[0] > 0:
                return int(handle[name].shape[0])
    candidates: list[int] = []

    def visitor(_name: str, item) -> None:
        if isinstance(item, h5py.Dataset) and item.shape and item.shape[0] > 0:
            candidates.append(int(item.shape[0]))

    handle.visititems(visitor)
    if not candidates:
        raise ValueError("cannot infer a positive event count")
    return Counter(candidates).most_common(1)[0][0]


def scan_one_file(task: tuple[str, str, str]) -> tuple[FileRecord | None, dict | None]:
    path_text, relative, prefix = task
    path = Path(path_text)
    source_uri = f"{prefix}/{relative}"
    try:
        size = path.stat().st_size
    except Exception as exc:
        return None, {
            "source_uri": source_uri,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    if size == 0:
        return None, {"source_uri": source_uri, "reason": "zero-byte"}
    try:
        with h5py.File(path, "r") as handle:
            events = infer_events(handle)
            year, year_source = infer_year(handle, path)
    except Exception as exc:
        return None, {
            "source_uri": source_uri,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return (
        FileRecord(
            local_path=str(path),
            source_uri=source_uri,
            size_bytes=size,
            events=events,
            year=year,
            year_source=year_source,
        ),
        None,
    )


def scan_files(
    source_dir: Path,
    gcs_prefix: str,
    *,
    workers: int,
    progress_every: int,
) -> tuple[list[FileRecord], list[dict]]:
    if workers <= 0:
        raise ValueError("--workers must be positive")
    if progress_every <= 0:
        raise ValueError("--progress-every must be positive")
    paths = sorted(source_dir.rglob("*.h5"))
    if not paths:
        raise ValueError(f"No H5 files found below {source_dir}")
    prefix = gcs_prefix.rstrip("/")
    tasks = (
        (str(path), path.relative_to(source_dir).as_posix(), prefix) for path in paths
    )
    records: list[FileRecord] = []
    rejected: list[dict] = []
    event_total = 0
    print(
        f"Found {len(paths):,} H5 files; inspecting headers with {workers} workers.",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = executor.map(scan_one_file, tasks, chunksize=16)
        for inspected, (record, rejection) in enumerate(results, start=1):
            if record is not None:
                records.append(record)
                event_total += record.events
            if rejection is not None:
                rejected.append(rejection)
            if inspected % progress_every == 0 or inspected == len(paths):
                print(
                    f"{inspected:,}/{len(paths):,} files | valid={len(records):,} "
                    f"rejected={len(rejected):,} events={event_total:,}",
                    flush=True,
                )
    records.sort(key=lambda record: record.source_uri)
    rejected.sort(key=lambda item: item["source_uri"])
    return records, rejected


def deterministic_take(
    records: list[FileRecord], target_events: int, rng: random.Random
) -> tuple[list[FileRecord], int]:
    candidates = list(records)
    rng.shuffle(candidates)
    selected: list[FileRecord] = []
    events = 0
    for record in candidates:
        if events >= target_events:
            break
        selected.append(record)
        events += record.events
    return selected, events


def select_records(
    records: list[FileRecord],
    *,
    primary_year: int,
    primary_fraction: float,
    target_events: int,
    seed: int,
) -> list[FileRecord]:
    if not 0.5 < primary_fraction <= 1.0:
        raise ValueError("--primary-fraction must be greater than 0.5 and at most 1")
    if target_events <= 0:
        raise ValueError("--target-events must be positive")
    known = [record for record in records if record.year is not None]
    primary = [record for record in known if record.year == primary_year]
    secondary = [record for record in known if record.year != primary_year]
    if not primary:
        raise ValueError(
            f"No files were identified as Data{str(primary_year)[-2:]}; "
            "inspect the printed year inventory before selecting a subset"
        )

    primary_target = round(target_events * primary_fraction)
    secondary_target = target_events - primary_target
    primary_selected, primary_events = deterministic_take(
        primary, primary_target, random.Random(seed)
    )
    secondary_selected, secondary_events = deterministic_take(
        secondary, secondary_target, random.Random(seed + 1)
    )

    selected_uris = {
        record.source_uri for record in primary_selected + secondary_selected
    }
    remaining_primary = [
        record for record in primary if record.source_uri not in selected_uris
    ]
    remaining_secondary = [
        record for record in secondary if record.source_uri not in selected_uris
    ]
    total = primary_events + secondary_events
    if total < target_events:
        fill_pool = remaining_primary + remaining_secondary
        fill_selected, _ = deterministic_take(
            fill_pool, target_events - total, random.Random(seed + 2)
        )
    else:
        fill_selected = []
    selected = primary_selected + secondary_selected + fill_selected
    if sum(record.events for record in selected) < target_events:
        raise ValueError(
            f"Only {sum(record.events for record in selected):,} classified events "
            f"are available, below the {target_events:,} target"
        )
    return sorted(selected, key=lambda record: record.source_uri)


def balance_groups(records: list[FileRecord], group_count: int) -> list[list[FileRecord]]:
    if group_count <= 0:
        raise ValueError("--groups must be positive")
    if len(records) < group_count:
        raise ValueError("Fewer selected files than requested groups")
    groups: list[list[FileRecord]] = [[] for _ in range(group_count)]
    group_events = [0] * group_count
    for record in sorted(records, key=lambda item: (-item.events, item.source_uri)):
        index = min(range(group_count), key=lambda item: (group_events[item], item))
        groups[index].append(record)
        group_events[index] += record.events
    return [sorted(group, key=lambda record: record.source_uri) for group in groups]


def sha256_lines(lines: Iterable[str]) -> str:
    payload = "\n".join(lines) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def write_plan(
    args: argparse.Namespace,
    records: list[FileRecord],
    rejected: list[dict],
    selected: list[FileRecord],
    groups: list[list[FileRecord]],
) -> dict:
    output_dir: Path = args.output_dir
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    groups_dir = output_dir / "groups"
    groups_dir.mkdir(parents=True)

    inventory_by_year = Counter(
        "unknown" if record.year is None else str(record.year) for record in records
    )
    events_by_year = Counter()
    for record in records:
        events_by_year["unknown" if record.year is None else str(record.year)] += record.events
    selected_files_by_year = Counter(str(record.year) for record in selected)
    selected_events_by_year = Counter()
    for record in selected:
        selected_events_by_year[str(record.year)] += record.events

    selected_lines = [record.source_uri for record in selected]
    selected_sha256 = sha256_lines(selected_lines)
    group_records = []
    for index, group in enumerate(groups):
        group_id = f"group-{index:05d}"
        group_events_by_year = Counter()
        for record in group:
            group_events_by_year[str(record.year)] += record.events
        payload = {
            "format_version": 1,
            "group_id": group_id,
            "files": [
                {
                    "source_uri": record.source_uri,
                    "expected_events": record.events,
                    "data_taking_year": record.year,
                }
                for record in group
            ],
        }
        (groups_dir / f"{group_id}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        group_records.append(
            {
                "group_id": group_id,
                "file_count": len(group),
                "expected_events": sum(record.events for record in group),
                "events_by_year": dict(sorted(group_events_by_year.items())),
                "manifest": f"groups/{group_id}.json",
            }
        )

    total_selected_events = sum(record.events for record in selected)
    primary_selected_events = selected_events_by_year[str(args.primary_year)]
    plan = {
        "format_version": 1,
        "selection": {
            "method": "seeded random files stratified by data-taking year",
            "seed": args.seed,
            "primary_year": args.primary_year,
            "primary_fraction_requested": args.primary_fraction,
            "primary_fraction_selected": primary_selected_events
            / total_selected_events,
            "target_events": args.target_events,
            "selected_events": total_selected_events,
            "selected_file_count": len(selected),
            "selected_inventory_sha256": selected_sha256,
            "selected_files_by_year": dict(sorted(selected_files_by_year.items())),
            "selected_events_by_year": dict(sorted(selected_events_by_year.items())),
        },
        "source_inventory": {
            "source_dir": str(args.source_dir.resolve()),
            "gcs_prefix": args.gcs_prefix.rstrip("/"),
            "valid_file_count": len(records),
            "rejected_file_count": len(rejected),
            "valid_files_by_year": dict(sorted(inventory_by_year.items())),
            "valid_events_by_year": dict(sorted(events_by_year.items())),
        },
        "grouping": {
            "method": "greedy expected-event balancing",
            "group_count": len(groups),
            "groups": group_records,
        },
    }
    (output_dir / "campaign_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "inventory.jsonl").write_text(
        "".join(json.dumps(asdict(record), sort_keys=True) + "\n" for record in records)
    )
    (output_dir / "rejected_inputs.json").write_text(
        json.dumps(rejected, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "source_objects.txt").write_text("\n".join(selected_lines) + "\n")
    (output_dir / "group_ids.txt").write_text(
        "\n".join(record["group_id"] for record in group_records) + "\n"
    )
    return plan


def main() -> None:
    args = parse_args()
    records, rejected = scan_files(
        args.source_dir,
        args.gcs_prefix,
        workers=args.workers,
        progress_every=args.progress_every,
    )
    inventory_events = Counter()
    for record in records:
        inventory_events["unknown" if record.year is None else str(record.year)] += (
            record.events
        )
    print(
        "Available events by year: "
        + ", ".join(
            f"{year}={events:,}" for year, events in sorted(inventory_events.items())
        ),
        flush=True,
    )
    selected = select_records(
        records,
        primary_year=args.primary_year,
        primary_fraction=args.primary_fraction,
        target_events=args.target_events,
        seed=args.seed,
    )
    groups = balance_groups(selected, args.groups)
    plan = write_plan(args, records, rejected, selected, groups)
    selection = plan["selection"]
    group_events = [
        group["expected_events"] for group in plan["grouping"]["groups"]
    ]
    print(f"Valid H5 files: {plan['source_inventory']['valid_file_count']:,}")
    print(f"Rejected H5 files: {plan['source_inventory']['rejected_file_count']:,}")
    print(f"Selected files: {selection['selected_file_count']:,}")
    print(f"Selected events: {selection['selected_events']:,}")
    print(
        f"Data{str(args.primary_year)[-2:]} event fraction: "
        f"{selection['primary_fraction_selected']:.3%}"
    )
    print(f"Groups: {plan['grouping']['group_count']}")
    print(f"Expected events/group: {min(group_events):,}--{max(group_events):,}")
    print(f"Subset SHA256: {selection['selected_inventory_sha256']}")


if __name__ == "__main__":
    main()
