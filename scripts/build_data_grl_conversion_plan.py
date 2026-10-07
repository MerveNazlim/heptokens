#!/usr/bin/env python3
"""Build deterministic, retryable H5 groups from a GCS object listing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-list",
        type=Path,
        required=True,
        help="Text produced by: gcloud storage ls --recursive PREFIX",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--files-per-group", type=int, default=200)
    parser.add_argument("--expected-files", type=int, default=55_616)
    parser.add_argument(
        "--assignment",
        choices=("contiguous", "round-robin"),
        default="contiguous",
        help=(
            "Assign adjacent sorted files to each group, or distribute the sorted "
            "inventory round-robin across groups. Round-robin is preferable when "
            "small or empty files may be clustered by filename."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def source_objects(path: Path) -> list[str]:
    objects = sorted(
        {
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip().startswith("gs://") and line.strip().endswith(".h5")
        }
    )
    if not objects:
        raise ValueError(f"No gs://...h5 objects found in {path}")
    return objects


def build_plan(args: argparse.Namespace) -> dict:
    if args.files_per_group <= 0:
        raise ValueError("--files-per-group must be positive")
    objects = source_objects(args.source_list)
    if args.expected_files and len(objects) != args.expected_files:
        raise ValueError(
            f"GCS inventory is incomplete: found {len(objects):,} H5 objects, "
            f"expected {args.expected_files:,}"
        )
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(args.output_dir)
        shutil.rmtree(args.output_dir)
    groups_dir = args.output_dir / "groups"
    groups_dir.mkdir(parents=True)

    inventory_text = "\n".join(objects) + "\n"
    inventory_sha256 = hashlib.sha256(inventory_text.encode()).hexdigest()
    group_count = math.ceil(len(objects) / args.files_per_group)
    if args.assignment == "contiguous":
        assignments = [
            objects[start : start + args.files_per_group]
            for start in range(0, len(objects), args.files_per_group)
        ]
    else:
        assignments = [objects[index::group_count] for index in range(group_count)]

    group_records = []
    for group_index, selected in enumerate(assignments):
        group_id = f"group-{group_index:05d}"
        group_payload = {
            "format_version": 1,
            "group_id": group_id,
            "files": [{"source_uri": uri} for uri in selected],
        }
        group_path = groups_dir / f"{group_id}.json"
        group_path.write_text(json.dumps(group_payload, indent=2) + "\n")
        group_records.append(
            {
                "group_id": group_id,
                "file_count": len(selected),
                "manifest": str(group_path.relative_to(args.output_dir)),
                "first_source": selected[0],
                "last_source": selected[-1],
            }
        )

    plan = {
        "format_version": 1,
        "source_list": str(args.source_list.resolve()),
        "inventory_sha256": inventory_sha256,
        "input_file_count": len(objects),
        "files_per_group": args.files_per_group,
        "assignment": args.assignment,
        "group_count": len(group_records),
        "groups": group_records,
    }
    (args.output_dir / "campaign_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "source_objects.txt").write_text(inventory_text)
    (args.output_dir / "group_ids.txt").write_text(
        "\n".join(record["group_id"] for record in group_records) + "\n"
    )
    return plan


def main() -> None:
    plan = build_plan(parse_args())
    print(f"H5 objects: {plan['input_file_count']:,}")
    print(f"Groups: {plan['group_count']:,}")
    print(f"Files/group: {plan['files_per_group']}")
    print(f"Assignment: {plan['assignment']}")
    print(f"Inventory SHA256: {plan['inventory_sha256']}")


if __name__ == "__main__":
    main()
