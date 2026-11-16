#!/usr/bin/env python3
"""Check the real stored features and a completed grouping-control checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch

from heptokens.data.continuous_schema import (
    CONTINUOUS_SCHEMA_METADATA_KEY,
    regroup_continuous_schema,
)
from heptokens.models.foundation_continuous import _load_checkpoint_state

PRESET = "kinematics_auxiliary_v1"
FEATURE_COUNTS = dict(jets=14, electrons=8, muons=6, photons=7, taus=7, tracks=8)


def preflight(prepared: Path, output: Path) -> dict:
    schemas = []
    for split in ("train", "val"):
        paths = sorted((prepared / split).glob("*.parquet"))
        if not paths:
            raise ValueError(f"No prepared {split} Parquets")
        metadata = pq.ParquetFile(paths[0]).schema_arrow.metadata or {}
        if CONTINUOUS_SCHEMA_METADATA_KEY not in metadata:
            raise ValueError(f"No continuous schema in {paths[0]}")
        schemas.append(json.loads(metadata[CONTINUOUS_SCHEMA_METADATA_KEY]))
    if schemas[0] != schemas[1]:
        raise ValueError("Prepared train/validation schemas differ")
    source = schemas[0]
    if set(source["objects"]) != set(FEATURE_COUNTS):
        raise ValueError("Unexpected object species in the original dataset")
    resolved = regroup_continuous_schema(source, grouping=PRESET)
    report = {}
    for name, spec in resolved["objects"].items():
        if spec["feature_count"] != FEATURE_COUNTS[name]:
            raise ValueError(f"Unexpected input feature count for {name}")
        groups = {
            group: [spec["feature_names"][i] for i in indices]
            for group, indices in spec["groups"].items()
        }
        expected_kinematics = ["pt", "eta", "phi"] + (["mass"] if name == "jets" else [])
        if groups["kinematics"] != expected_kinematics:
            raise ValueError(f"Unexpected kinematics for {name}: {groups['kinematics']}")
        if len(groups) != (3 if name in {"jets", "tracks"} else 2):
            raise ValueError(f"Wrong number of groups for {name}")
        if spec["feature_names"] != source["objects"][name]["feature_names"]:
            raise ValueError(f"Input feature order changed for {name}")
        report[name] = groups
    output.mkdir(parents=True, exist_ok=True)
    (output / "original_continuous_schema.json").write_text(json.dumps(source, indent=2) + "\n")
    (output / "resolved_grouping_schema.json").write_text(json.dumps(resolved, indent=2) + "\n")
    (output / "grouping_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS original continuous inputs; two groups except jets/tracks (three)")
    print(json.dumps(report, indent=2))
    return resolved


def completed(run_dir: Path, output: Path, epochs: int, mode: str) -> None:
    checkpoint = run_dir / "checkpoints/last.ckpt"
    if not checkpoint.is_file() or not (run_dir / "SUCCESS.txt").is_file():
        raise ValueError("Missing final checkpoint or SUCCESS marker")
    state = _load_checkpoint_state(str(checkpoint))
    hparams = state.get("hyper_parameters", {})
    expected = json.loads((output / "resolved_grouping_schema.json").read_text())
    if hparams.get("feature_grouping") != PRESET or hparams.get("continuous_schema") != expected:
        raise ValueError("Checkpoint does not contain the validated new grouping")
    if state.get("epoch") != epochs - 1:
        raise ValueError(f"Incomplete epochs: checkpoint epoch={state.get('epoch')}")
    steps = int(state.get("global_step", 0))
    if steps <= 0 or (mode == "smoke" and steps != 200):
        raise ValueError(f"Unexpected completed optimizer steps: {steps}")
    for name, tensor in state.get("state_dict", {}).items():
        if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite checkpoint parameter: {name}")
    for key, value in dict(hidden_dim=256, num_heads=8, num_layers=4,
                           inner_num_heads=4, inner_num_layers=1, max_seq_length=256,
                           mask_prob=0.15, learning_rate=1e-4).items():
        if hparams.get(key) != value:
            raise ValueError(f"Unexpected transformer/training setting {key}={hparams.get(key)}")
    log = (output / "training.log").read_text(errors="replace")
    if f"max_epochs={epochs}" not in log or "Trainer.fit" not in log:
        raise ValueError("Training log does not confirm normal max-epoch completion")
    report = dict(passed=True, feature_grouping=PRESET, epochs=epochs, global_step=steps,
                  checkpoint=str(checkpoint), mode=mode)
    (output / "grouping_training_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS completed grouping-control training", json.dumps(report))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--mode", choices=("smoke", "full"))
    args = parser.parse_args()
    if args.prepared_dir is not None:
        preflight(args.prepared_dir, args.output_dir)
    elif args.run_dir is not None and args.epochs is not None and args.mode is not None:
        completed(args.run_dir, args.output_dir, args.epochs, args.mode)
    else:
        parser.error("Provide --prepared-dir or all of --run-dir/--epochs/--mode")


if __name__ == "__main__":
    main()
