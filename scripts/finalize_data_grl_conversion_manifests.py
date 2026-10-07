#!/usr/bin/env python3
"""Validate completed conversion groups and build final pretraining manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPRESENTATIONS = ("q1", "q8", "continuous")
SPLITS = ("train", "val")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-plan", type=Path, required=True)
    parser.add_argument("--status-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def finalize(args: argparse.Namespace) -> dict:
    plan = load_json(args.campaign_plan)
    plan_groups = plan.get("groups")
    if plan_groups is None:
        plan_groups = plan.get("grouping", {}).get("groups", [])
    expected_groups = {
        item["group_id"]: item for item in plan_groups
    }
    if not expected_groups:
        raise ValueError("Campaign plan contains no groups")

    planned_input_count = plan.get("input_file_count")
    if planned_input_count is None:
        planned_input_count = plan.get("selection", {}).get("selected_file_count")
    if planned_input_count is None:
        raise ValueError("Campaign plan does not declare its selected input count")

    inventory_sha256 = plan.get("inventory_sha256")
    if inventory_sha256 is None:
        inventory_sha256 = plan.get("selection", {}).get(
            "selected_inventory_sha256"
        )
    if not inventory_sha256:
        raise ValueError("Campaign plan does not declare its inventory SHA256")

    statuses = {}
    for path in sorted(args.status_dir.glob("group-*.json")):
        status = load_json(path)
        group_id = status.get("group_id")
        if group_id in statuses:
            raise ValueError(f"Duplicate status for {group_id}")
        statuses[group_id] = status
    missing = sorted(set(expected_groups) - set(statuses))
    unexpected = sorted(set(statuses) - set(expected_groups))
    if missing or unexpected:
        raise ValueError(
            f"Incomplete status set: missing={missing[:20]} unexpected={unexpected[:20]}"
        )

    all_sources = set()
    source_counts = {}
    train_rows = 0
    val_rows = 0
    invalid_sources = []
    representation_files = {
        representation: {split: [] for split in SPLITS}
        for representation in REPRESENTATIONS
    }
    representation_bytes = {
        representation: {split: 0 for split in SPLITS}
        for representation in REPRESENTATIONS
    }
    campaign_membership = hashlib.sha256()
    reference_settings = None

    groups_root = args.campaign_plan.parent
    for group_id in sorted(expected_groups):
        plan_group = expected_groups[group_id]
        group_manifest = load_json(
            groups_root
            / plan_group.get("manifest", f"groups/{group_id}.json")
        )
        expected_sources = [entry["source_uri"] for entry in group_manifest["files"]]
        status = statuses[group_id]
        observed_sources = list(status.get("source_counts", {}))
        if len(expected_sources) != len(set(expected_sources)):
            raise ValueError(f"Duplicate source in {group_id} campaign assignment")
        # The converter sorts JSON dictionary keys, independently of the input
        # file order. Validate membership without changing split fingerprints.
        if set(observed_sources) != set(expected_sources):
            raise ValueError(
                f"{group_id} source list differs from its campaign assignment"
            )
        duplicates = all_sources.intersection(observed_sources)
        if duplicates:
            raise ValueError(
                f"Sources assigned to more than one completed group: {sorted(duplicates)[:10]}"
            )
        all_sources.update(observed_sources)
        source_counts.update(status["source_counts"])
        invalid_sources.extend(status.get("invalid_sources", []))

        settings = {
            key: status.get(key)
            for key in (
                "seed",
                "train_fraction_requested",
                "split_method",
                "max_seq_length",
                "q1_vocab_size",
                "q1_max_quantizers",
                "q8_vocab_size",
                "q8_max_quantizers",
                "continuous_max_feature_dim",
            )
        }
        if reference_settings is None:
            reference_settings = settings
        elif settings != reference_settings:
            raise ValueError(
                f"{group_id} conversion settings differ: {settings} != {reference_settings}"
            )

        train_rows += int(status["train_rows"])
        val_rows += int(status["val_rows"])
        campaign_membership.update(group_id.encode())
        campaign_membership.update(status["membership_sha256"].encode())

        for split in SPLITS:
            reference_basenames = None
            reference_rows = None
            for representation in REPRESENTATIONS:
                records = status["outputs"][representation][split]
                basenames = [Path(record["path"]).name for record in records]
                rows = [int(record["rows"]) for record in records]
                if reference_basenames is None:
                    reference_basenames = basenames
                    reference_rows = rows
                elif basenames != reference_basenames or rows != reference_rows:
                    raise ValueError(
                        f"{group_id} {split} output parts are not aligned across representations"
                    )
                representation_files[representation][split].extend(records)
                representation_bytes[representation][split] += sum(
                    int(record["bytes"]) for record in records
                )

    if len(all_sources) != int(planned_input_count):
        raise ValueError(
            f"Completed unique sources={len(all_sources):,}, "
            f"planned={int(planned_input_count):,}"
        )
    if train_rows + val_rows != sum(
        int(item["processed"]) for item in source_counts.values()
    ):
        raise ValueError("Aggregated row totals do not match source counts")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    campaign_sha256 = campaign_membership.hexdigest()
    representation_manifests = {}
    for representation in REPRESENTATIONS:
        manifest = {
            "format_version": 1,
            "representation": representation,
            "data_domain": "real_data_only",
            "seed": reference_settings["seed"],
            "train_fraction_requested": reference_settings[
                "train_fraction_requested"
            ],
            "split_method": reference_settings["split_method"],
            "inventory_sha256": inventory_sha256,
            "membership_sha256": campaign_sha256,
            "input_file_count": len(all_sources),
            "valid_input_file_count": len(all_sources) - len(invalid_sources),
            "invalid_input_file_count": len(invalid_sources),
            "invalid_sources": invalid_sources,
            "group_count": len(statuses),
            "train_rows": train_rows,
            "val_rows": val_rows,
            "total_rows": train_rows + val_rows,
            "train_shards": len(representation_files[representation]["train"]),
            "val_shards": len(representation_files[representation]["val"]),
            "train_bytes": representation_bytes[representation]["train"],
            "val_bytes": representation_bytes[representation]["val"],
            "settings": reference_settings,
            "files": representation_files[representation],
            "source_counts": source_counts,
        }
        path = args.output_dir / f"{representation}_manifest.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        representation_manifests[representation] = str(path)

    summary = {
        "format_version": 1,
        "status": "complete",
        "input_file_count": len(all_sources),
        "valid_input_file_count": len(all_sources) - len(invalid_sources),
        "invalid_input_file_count": len(invalid_sources),
        "invalid_sources": invalid_sources,
        "group_count": len(statuses),
        "inventory_sha256": inventory_sha256,
        "membership_sha256": campaign_sha256,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "total_rows": train_rows + val_rows,
        "manifests": representation_manifests,
    }
    (args.output_dir / "campaign_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    summary = finalize(parse_args())
    print("PASS: all conversion groups are complete and aligned")
    print(f"Input H5 files: {summary['input_file_count']:,}")
    print(f"Invalid H5 files: {summary['invalid_input_file_count']:,}")
    print(f"Groups: {summary['group_count']:,}")
    print(f"Train rows: {summary['train_rows']:,}")
    print(f"Validation rows: {summary['val_rows']:,}")
    print(f"Membership SHA256: {summary['membership_sha256']}")


if __name__ == "__main__":
    main()
