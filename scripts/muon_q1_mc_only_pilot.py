#!/usr/bin/env python3
"""Repeat the completed three-epoch muon Q1 data-init pilot using only its MC."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
from joblib import dump, load
import numpy as np
from omegaconf import OmegaConf
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

import muon_q1_init_control as initialization
from heptokens.data.atlas_event_iterable import _resolve_h5_slice
from heptokens.data.atlas_event_mc_only import mc_split_indices
from heptokens.data.transforms import (
    CompositeTransformer,
    FeatureWiseTransformer,
    LogScaleTransform,
)

ARM = "data_init"
TARGET = "heptokens.data.atlas_event_mc_only.AtlasEventMCOnlyMapModule"
NEW_SOURCES = (
    "scripts/muon_q1_mc_only_pilot.py",
    "src/heptokens/data/atlas_event_mc_only.py",
    "src/heptokens/data/atlas_event_iterable.py",
    "src/heptokens/data/transforms.py",
)


def baseline_context(path, root):
    plan = initialization.verify_plan(path)
    if plan["root"] != str(root) or plan.get("pilot_epochs") != 3:
        raise ValueError("Use the completed three-epoch pilot in its original checkout")
    initialization.validate_reference(initialization.load_config(plan["reference"]))
    initialization.verify_pilot_completion(plan, ARM)
    if not (path.parent / ARM / "SUCCESS.txt").is_file():
        raise ValueError("The reference data_init pilot has not completed")
    cfg = initialization.load_config(plan["configs"][ARM]["path"])
    dm = cfg["datamodule"]
    if (
        dm.get("data_domains") is not None
        or dm.get("sampling_domain_fractions")
        or dm.get("sampling_num_samples")
    ):
        raise ValueError(
            "Reference already changes sampling; review before creating an MC-only pilot"
        )
    if (
        cfg["model"]["num_quantizers"],
        cfg["model"]["codebook_size"],
        cfg["model"]["codebook_dim"],
    ) != (1, 16384, 8):
        raise ValueError("Expected the completed Q1/cb16384/dim8 pilot")
    if not cfg["model"]["data_codebook_init"] or cfg["model"]["dead_code_reset"]:
        raise ValueError("Expected data initialization on, unused-code replacement off")
    return plan, cfg


def classify_paths(paths, data_dir):
    data_dir = data_dir.resolve(strict=True)
    data = [p for p in paths if data_dir in Path(p).resolve().parents]
    excluded = set(data)
    mc = [p for p in paths if p not in excluded]
    if not mc or not data:
        raise ValueError(
            "Saved inputs must include MC and files within the explicit data directory"
        )
    return mc, data


def event_counts(dm):
    from heptokens.data.atlas_event_mappable import AtlasEventMapDataset

    counts = []
    for path in dm["data_paths"]:
        with h5py.File(path, "r") as handle:
            counts.append(
                int(
                    AtlasEventMapDataset._infer_dimensions(
                        handle,
                        dm.get("event_inputs") or [],
                        dm["object_collections"],
                        dm.get("num_events"),
                    )[1]
                )
            )
    return counts


def preprocessing_node(dm):
    preprocess = (dm.get("transforms") or {}).get("preprocess") or {}
    node = preprocess.get("cst_fn") or {}
    if (
        preprocess.get("_target_") != "heptokens.data.collation.preprocess_objects_batch"
        or not preprocess.get("_partial_")
        or node.get("_target_") != "joblib.load"
        or initialization.joblib_paths(dm) != [node.get("filename")]
    ):
        raise ValueError("Expected one explicit object-feature preprocessing joblib")
    return node


def preprocessing_spec(source, output):
    """Retain the transform recipe, never its mixed-domain fitted statistics."""
    dm = source["datamodule"]
    node = preprocessing_node(dm)
    original = load(node["filename"])
    collection = next(c for c in dm["object_collections"] if c["object_name"] == "muons")
    if (
        not isinstance(original, CompositeTransformer)
        or not isinstance(original.log_transformer, FeatureWiseTransformer)
        or not isinstance(original.final_transformer, StandardScaler)
        or original.n_features_in_ != len(collection["inputs"])
    ):
        raise ValueError("Expected the saved muon log-standard recipe and matching feature count")
    logs = []
    for config in original.log_transformer.feature_configs:
        transforms = config["transformers"]
        if len(transforms) != 1 or not isinstance(transforms[0], LogScaleTransform):
            raise ValueError("Unexpected log-transform recipe; refusing to guess")
        indices = original.log_transformer._get_indices(config)
        logs.append({"indices": list(indices), "offset": transforms[0].offset})
    return {
        "source_recipe_joblib": node["filename"],
        "joblib": str(output / "preprocessing/muons_log_standard_mc_train.joblib"),
        "metadata": str(output / "preprocessing/muons_log_standard_mc_train.json"),
        "receipt": str(output / "preprocessing/fit_receipt.json"),
        "mode": "log_standard",
        "feature_paths": collection["inputs"],
        "log_transforms": logs,
        "scaler_params": original.final_transformer.get_params(deep=False),
        "fit_domain": "mc",
        "fit_partition": "train",
        "fit_objects": "all valid objects in the original MC training partition",
    }


def mc_config(source, output, mc_paths, counts):
    cfg = initialization.arm_config(source, output, ARM, pilot_epochs=3)
    dm = cfg["datamodule"]
    dm.update(
        _target_=TARGET,
        data_paths=list(mc_paths),
        reference_data_paths=list(source["datamodule"]["data_paths"]),
        reference_event_counts=list(counts),
        split_audit_path=str(output / ARM / "mc_membership.json"),
        preprocessing_audit_path=str(output / "preprocessing/muons_log_standard_mc_train.json"),
    )
    preprocessing_node(dm)["filename"] = str(
        output / "preprocessing/muons_log_standard_mc_train.joblib"
    )
    if cfg["model"] != source["model"]:
        raise ValueError("MC-only pilot must not change any model settings")
    return cfg


def verify_plan(path):
    plan = json.loads(path.read_text())
    if plan.get("version") != 2:
        raise ValueError(
            "Old mixed-joblib plan. Use the new train-fit output; do not submit the old pilot."
        )
    if plan["kind"] != "muon_q1_mc_only_pilot" or Path(plan["output"]) != path.parent.resolve():
        raise ValueError("Wrong or moved MC-only pilot plan")
    baseline, source = baseline_context(Path(plan["baseline_plan"]), Path(plan["root"]))
    for record in plan["inputs"] + plan["sources"] + list(plan["configs"].values()):
        initialization.verify_record(record)
    if plan["runtime"] != initialization.runtime_record() or plan["runtime"] != baseline["runtime"]:
        raise ValueError("Runtime changed since the completed reference pilot")
    mc, data = classify_paths(source["datamodule"]["data_paths"], Path(plan["data_dir"]))
    if mc != plan["mc_paths"] or data != plan["excluded_data_paths"]:
        raise ValueError("Saved MC/data selection changed")
    if plan["preprocessing"] != preprocessing_spec(source, path.parent.resolve()):
        raise ValueError("Frozen MC preprocessing recipe changed")
    expected = mc_config(source, path.parent.resolve(), mc, plan["reference_event_counts"])
    actual = initialization.load_config(plan["configs"][ARM]["path"])
    differences = initialization.diff_paths(expected, actual)
    if differences:
        raise ValueError(f"Unexpected MC-only settings changed: {differences}")
    return plan


def verify_preprocessing(plan):
    spec = plan["preprocessing"]
    receipt = json.loads(Path(spec["receipt"]).read_text())
    if receipt["plan_sha256"] != initialization.digest(Path(plan["output"]) / "plan.json"):
        raise ValueError("MC preprocessing was fitted for a different plan")
    for record in receipt["artifacts"]:
        initialization.verify_record(record)
    if {r["path"] for r in receipt["artifacts"]} != {spec["joblib"], spec["metadata"]}:
        raise ValueError("Wrong preprocessing artifacts")
    metadata = json.loads(Path(spec["metadata"]).read_text())
    if metadata["spec"] != spec or metadata["h5_files"] != plan["mc_paths"]:
        raise ValueError("Preprocessing did not follow the frozen MC-only recipe")
    return metadata


def fit_preprocessing(plan):
    """Stream valid MC training objects into an unfitted copy of the recipe."""
    spec = plan["preprocessing"]
    if Path(spec["receipt"]).exists():
        return verify_preprocessing(plan)
    if Path(spec["joblib"]).exists() or Path(spec["metadata"]).exists():
        raise ValueError("Incomplete preprocessing fit exists; nothing overwritten")
    cfg = initialization.load_config(plan["configs"][ARM]["path"])
    dm = cfg["datamodule"]
    train, val, test = mc_split_indices(
        dm["reference_data_paths"],
        dm["reference_event_counts"],
        dm["data_paths"],
        dm["train_frac"],
        dm["val_frac"],
        dm["seed"],
    )
    mc_count = len(train) + len(val) + len(test)
    selected = np.zeros(mc_count, dtype=bool)
    selected[train] = True
    ordered_hash = hashlib.sha256(train.tobytes()).hexdigest()
    n_train = len(train)
    del train, val, test
    transformer = clone(load(spec["source_recipe_joblib"]))
    if hasattr(transformer.final_transformer, "mean_"):
        raise ValueError("Recipe clone unexpectedly retained fitted statistics")
    collection = next(c for c in dm["object_collections"] if c["object_name"] == "muons")
    cap = (dm.get("max_objects") or {}).get("muons", dm.get("num_objects")) or None
    mask_path = collection.get("mask_input", dm.get("mask_input"))
    counts = dict(zip(dm["reference_data_paths"], dm["reference_event_counts"]))
    offset = objects = 0
    for path in plan["mc_paths"]:
        n_events = counts[path]
        membership = selected[offset : offset + n_events]
        offset += n_events
        if not membership.any():
            continue
        with h5py.File(path, "r") as handle:
            for begin in range(0, n_events, 4096):
                end = min(begin + 4096, n_events)
                keep = membership[begin:end]
                if not keep.any():
                    continue
                arrays = []
                for feature in collection["inputs"]:
                    array = _resolve_h5_slice(handle, feature, slice(begin, end), cap).astype(
                        np.float32
                    )
                    arrays.append(array[:, None] if array.ndim == 1 else array)
                values = np.stack(arrays, axis=-1)
                values = np.clip(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), -1e4, 1e4)
                if mask_path is None:
                    mask = np.ones(values.shape[:2], dtype=bool)
                else:
                    mask = _resolve_h5_slice(handle, mask_path, slice(begin, end), values.shape[1])
                    mask = mask[:, None] if mask.ndim == 1 else mask
                    mask = mask.astype(bool)
                valid = values[keep][mask[keep]]
                if not len(valid):
                    continue
                if not objects:
                    transformer.n_features_in_ = len(collection["inputs"])
                    transformer.log_transformer.fit(valid)
                logged = transformer.log_transformer.transform(valid)
                if not np.isfinite(logged).all():
                    raise ValueError(f"Nonfinite MC training values after saved log recipe: {path}")
                transformer.final_transformer.partial_fit(logged)
                objects += len(valid)
        print(
            f"  MC-train preprocessing: {objects:,} valid objects fitted; {Path(path).name}",
            flush=True,
        )
    if not objects:
        raise ValueError("No valid MC training objects for preprocessing")
    metadata = {
        "object_type": "muons",
        "mode": "log_standard",
        "spec": spec,
        "feature_paths": collection["inputs"],
        "feature_names": [Path(p).name for p in collection["inputs"]],
        "h5_files": plan["mc_paths"],
        "n_objects_fit": objects,
        "fit_domain": "mc",
        "fit_partition": "train",
        "train_events": n_train,
        "ordered_compact_train_indices_sha256": ordered_hash,
        "note": "Only valid MC training objects contribute. No collision-data, validation or test objects. "
        "Streaming StandardScaler.partial_fit retains the original log/scaling recipe, not its fitted state.",
    }
    path = Path(spec["joblib"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".joblib.tmp")
    dump(transformer, temporary)
    temporary.replace(path)
    initialization.write_json(Path(spec["metadata"]), metadata)
    initialization.write_json(
        Path(spec["receipt"]),
        {
            "plan_sha256": initialization.digest(Path(plan["output"]) / "plan.json"),
            "artifacts": [
                initialization.file_record(Path(spec[key])) for key in ("joblib", "metadata")
            ],
        },
    )
    return verify_preprocessing(plan)


def describe(plan):
    cfg = initialization.load_config(plan["configs"][ARM]["path"])
    dm = cfg["datamodule"]
    print("Muon Q1 MC-only CHECK: ONE fresh three-epoch pilot")
    print(f"  completed MC+data reference: {Path(plan['baseline_plan']).parent / ARM}")
    print(f"  original MC files retained: {len(plan['mc_paths'])}")
    mc_files = set(plan["mc_paths"])
    mc_events = sum(
        n
        for p, n in zip(dm["reference_data_paths"], plan["reference_event_counts"])
        if p in mc_files
    )
    print(f"  original MC events retained across partitions: {mc_events:,}")
    print(
        f"  collision-data files excluded from feature loading: {len(plan['excluded_data_paths'])}"
    )
    print("  Q1; K=16384; dim=8; widths=[128,256,512]; batch=1024")
    print("  epochs to run=3; original scheduler horizon=20")
    print("  data initialization=true; pool=65536; unused-code replacement=false")
    print("  retain original global MC train/val/test memberships; no resplit or oversampling")
    print("  same MC population, fewer total training steps than MC+data")
    print(f"  NEW MC-training-only joblib: {initialization.joblib_paths(dm)}")
    print(
        "  fit original log-standard recipe on valid MC TRAINING objects only; no validation/test or data"
    )
    print("  preprocessing fit happens at submission, not during the dry run")
    features = next(c["inputs"] for c in dm["object_collections"] if c["object_name"] == "muons")
    print(f"  features (saved order): {features}")
    print(f"  output: {plan['output']}")


def prepare(args):
    root, output = args.root.resolve(), args.output.resolve()
    baseline_path = args.baseline.resolve() / "plan.json"
    path = output / "plan.json"
    if path.exists():
        plan = verify_plan(path)
        if (
            plan["root"] != str(root)
            or plan["baseline_plan"] != str(baseline_path)
            or plan["data_dir"] != str(args.data_dir.resolve())
        ):
            raise ValueError("Existing plan uses different inputs; use a separate output")
        describe(plan)
        return
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output is nonempty; nothing overwritten")
    baseline, source = baseline_context(baseline_path, root)
    for protected in (baseline_path.parent, Path(baseline["reference"]).parent):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError("Keep MC-only output separate from completed trainings")
    mc, data = classify_paths(source["datamodule"]["data_paths"], args.data_dir)
    counts = event_counts(source["datamodule"])
    preprocessing = preprocessing_spec(source, output)
    cfg = mc_config(source, output, mc, counts)
    sources = [initialization.file_record(root / p) for p in NEW_SOURCES]
    inputs = [
        initialization.file_record(p)
        for p in (
            baseline_path,
            baseline_path.parent / ARM / "SUCCESS.txt",
            baseline_path.parent / ARM / "pilot_completion.json",
        )
    ]
    initialization.preflight()
    (output / "configs").mkdir(parents=True)
    cfg_path = output / "configs" / f"{ARM}.yaml"
    OmegaConf.save(OmegaConf.create(cfg), cfg_path)
    plan = {
        "version": 2,
        "kind": "muon_q1_mc_only_pilot",
        "root": str(root),
        "output": str(output),
        "baseline_plan": str(baseline_path),
        "pilot_epochs": 3,
        "data_dir": str(args.data_dir.resolve()),
        "mc_paths": mc,
        "excluded_data_paths": data,
        "reference_event_counts": counts,
        "inputs": inputs,
        "sources": sources,
        "runtime": baseline["runtime"],
        "configs": {ARM: initialization.file_record(cfg_path)},
        "preprocessing": preprocessing,
        "intervention": "Use only the original MC events; refit log-standard preprocessing on MC training objects.",
        "source_config_changes": initialization.diff_paths(source, cfg),
        "notes": [
            "No additional MC files, repeated sampling, K change or unused-code replacement.",
            "Retain the MC subset of the original mixed-input global event permutation.",
            "Only MC feature arrays are loaded. Original data event counts preserve split membership.",
            "Three MC-only epochs have fewer optimizer steps than three mixed epochs; not matched compute.",
            "MC validation differs from mixed validation; their aggregate losses are not paired metrics.",
            "Original preprocessing recipe retained, fitted statistics replaced using only valid MC training objects.",
            "Both training domain and preprocessing statistics differ from the reference; not a domain-only ablation.",
            "Different scaling makes preprocessed loss values incomparable; compare physical-unit errors on common MC events.",
            "Historical split membership is reconstructed, not checked against a training-time manifest.",
        ],
    }
    initialization.write_json(path, plan)
    describe(plan)
    print(f"Frozen plan: {path}; no training submitted")


def run(args):
    import fcntl
    import torch

    output = args.output.resolve()
    with (output / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = verify_plan(output / "plan.json")
        run_dir = output / ARM
        if (run_dir.exists() and any(run_dir.iterdir())) or (output / f"{ARM}.log").exists():
            raise ValueError("Refusing to overwrite/resume the MC-only pilot")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Expose exactly one GPU using CUDA_VISIBLE_DEVICES")
        initialization.preflight()
        fit_preprocessing(plan)
        cfg_path = Path(plan["configs"][ARM]["path"])
        cmd = [
            sys.executable,
            str(Path(plan["root"]) / "scripts/train.py"),
            "--config-path",
            str(cfg_path.parent),
            "--config-name",
            ARM,
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(plan["root"]) / "src") + os.pathsep + env.get("PYTHONPATH", "")
        cfg = initialization.load_config(cfg_path)
        env["WANDB_MODE"] = "offline" if cfg["logger"].get("offline") else "online"
        initialization.write_json(
            output / "execution.json",
            {
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu": str(torch.cuda.get_device_properties(0)),
                "torch_cuda": torch.version.cuda,
            },
        )
        print(f"START MC-only pilot: {' '.join(cmd)}", flush=True)
        with (output / f"{ARM}.log").open("x") as log:
            subprocess.run(
                cmd, cwd=plan["root"], env=env, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        summarize(args)


def summarize(args):
    output = args.output.resolve()
    plan = verify_plan(output / "plan.json")
    preprocessing = verify_preprocessing(plan)
    initialization.verify_pilot_completion(plan, ARM)
    if not (output / ARM / "SUCCESS.txt").is_file():
        raise ValueError("MC-only pilot did not complete")
    membership = json.loads((output / ARM / "mc_membership.json").read_text())
    if membership["mc_files"] != plan["mc_paths"]:
        raise ValueError("MC dataset differs from the frozen inputs")
    if (
        membership["partitions"]["train"]["ordered_compact_indices_sha256"]
        != preprocessing["ordered_compact_train_indices_sha256"]
    ):
        raise ValueError("Preprocessing and training used different MC train membership")
    rows = [
        json.loads(p.read_text())
        for p in sorted((output / ARM / "validation_usage").glob("*.json"))
    ]
    candidates = [r for r in rows if r["initialization_completed"]]
    if not candidates:
        raise ValueError("No post-initialization MC validation records")
    for row in rows:
        if (
            len(row["counts"]) != 1
            or len(row["counts"][0]) != 16384
            or not row["data_codebook_init"]
        ):
            raise ValueError("Validation did not use the expected Q1/cb16384 data-init model")
    best = min(candidates, key=lambda r: r["validation_metrics"]["val/total_loss"])
    initialization.write_json(
        output / "pilot_summary.json",
        {
            "baseline": str(Path(plan["baseline_plan"]).parent),
            "mc_membership": membership,
            "preprocessing": preprocessing,
            "validation_passes": [{k: v for k, v in r.items() if k != "counts"} for r in rows],
            "best_post_init_by_total_loss": {k: v for k, v in best.items() if k != "counts"},
            "note": "MC-only validation metrics, not directly comparable to the reference mixed-validation loss. "
            "MC-only fitted scaling also changes the loss units. Matched MC physical reconstruction plots are a subsequent step.",
        },
    )
    print(f"MC-only pilot complete: {output / 'pilot_summary.json'}")
    print(
        f"Best MC validation at step {best['step']}: recon={best['validation_metrics']['val/recon_loss']:.6g}; usage={best['codebook_metrics'][0]}"
    )


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "summarize"))
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--baseline", type=Path, default=root / "results/atlas_muon_q1_initialization_pilot_e3_v1"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/zephyr/Data/viviana/bnl-treasure/data_new/h5/realdata"),
    )
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    main()
