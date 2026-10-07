#!/usr/bin/env python3
"""Figure 2: audit unused H5 files, evaluate a frozen stage-1 scan, then plot.

No training is launched. Existing comparison caches are not imported: their
sample lists are not evidence of held-out evaluation. Re-running evaluate uses
only caches belonging to the same audited plan. Plot needs no models or GPUs.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from paper_plot_style import OBJECTS

log = logging.getLogger(__name__)
VERSION = "paper-capacity-v1"
SELECTED_CONFIGURATIONS = {
    "jets": (8, 2048, 8),
    "electrons": (8, 2048, 8),
    "muons": (8, 2048, 8),
    "photons": (8, 2048, 8),
    "taus": (8, 4096, 8),
    "tracks": (8, 4096, 8),
}
Q8_CB2048_LABEL = "stage1 q8 cb2048 dim8"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_record(path, *, content_hash=False):
    path = Path(path).resolve()
    stat = path.stat()
    record = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if content_hash:
        record["sha256"] = sha256(path)
    return record


def verify_file(record):
    actual = file_record(record["path"], content_hash="sha256" in record)
    if actual != record:
        raise ValueError(
            f"File changed since audit: {record['path']}. Create a new plan/output directory."
        )


def h5_names(paths):
    if not isinstance(paths, (list, tuple)) or not paths:
        raise ValueError(
            "Missing explicit H5 input list; cannot establish file-disjoint evaluation."
        )
    names = [Path(str(path)).name for path in paths]
    if any(not name.endswith(".h5") or any(c in name for c in "*?[") for name in names):
        raise ValueError(
            "H5 provenance must contain explicit .h5 filenames, not directories/globs."
        )
    if len(names) != len(set(names)):
        raise ValueError("Ambiguous duplicate H5 basenames in provenance.")
    return names


def select_unused_files(directory, excluded_names, n_files, exclude_pattern=None):
    # Condor scratch prefixes differ from Zephyr. Basename matching is
    # deliberately conservative: exclude a matching name in either domain.
    candidates = sorted(
        path.resolve() for path in Path(directory).glob("*.h5") if path.stat().st_size
    )
    names = [path.name for path in candidates]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate H5 identities under {directory}")
    available = [
        path
        for path in candidates
        if path.name not in excluded_names
        and not (exclude_pattern and fnmatch.fnmatch(path.name, exclude_pattern))
    ]
    if len(available) < n_files:
        raise ValueError(
            f"Only {len(available)} file-disjoint H5 files in {directory}; requested {n_files}. "
            "Do not substitute training files. Supply another evaluation dataset or audit the saved event split."
        )
    return [file_record(path) for path in available[:n_files]]


def plain(cfg, key):
    value = OmegaConf.select(cfg, key)
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def training_controls(cfg, training_names):
    model = plain(cfg, "model")
    for key in ("num_quantizers", "codebook_size", "codebook_dim"):
        model.pop(key, None)
    keys = (
        "seed",
        "datamodule._target_",
        "datamodule.batch_size",
        "datamodule.seed",
        "datamodule.train_frac",
        "datamodule.val_frac",
        "datamodule.test_frac",
        "datamodule.num_events",
        "datamodule.num_objects",
        "datamodule.max_objects",
        "datamodule.object_type",
        "datamodule.object_collections",
        "datamodule.mask_input",
        "datamodule.data_domains",
        "datamodule.sampling_domain_fractions",
        "datamodule.sampling_balance_by",
        "datamodule.sampling_num_samples",
        "trainer.max_epochs",
        "trainer.max_steps",
        "trainer.limit_train_batches",
        "trainer.limit_val_batches",
        "trainer.precision",
        "trainer.accumulate_grad_batches",
        "trainer.gradient_clip_val",
    )
    transforms = plain(cfg, "datamodule.transforms")
    if transforms and "preprocess" in transforms:
        # The absolute scratch path is not a training control. The fitted
        # transform itself is compared independently using its checksum.
        cst_fn = transforms["preprocess"].get("cst_fn", {})
        if "filename" in cst_fn:
            cst_fn["filename"] = Path(cst_fn["filename"]).name
    return {
        "model_except_capacity": model,
        "ordered_training_h5_names": training_names,
        "transforms": transforms,
        **{key: plain(cfg, key) for key in keys},
    }


def audit_run(scan, object_name, label, run_dir, checkpoint_name):
    cfg_path = run_dir / "full_config.yaml"
    cfg = OmegaConf.load(cfg_path)
    q, k, d = (
        int(plain(cfg, f"model.{key}"))
        for key in ("num_quantizers", "codebook_size", "codebook_dim")
    )
    if label != f"stage1 q{q} cb{k} dim{d}" or plain(cfg, "datamodule.object_type") != object_name:
        raise ValueError(f"Run name and saved model/object configuration disagree: {run_dir}")
    checkpoint = run_dir / "checkpoints" / checkpoint_name
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Exact requested checkpoint missing: {checkpoint}; no fallback allowed."
        )
    prep = scan.verified_stage1_preprocessor(argparse.Namespace(object=object_name), run_dir)
    if prep is None:
        raise ValueError(f"Cannot verify copied stage-1 preprocessing provenance: {run_dir}")
    saved_prep = plain(cfg, "datamodule.transforms.preprocess.cst_fn.filename")
    if not saved_prep or Path(saved_prep).name != prep.name:
        raise ValueError(f"Saved preprocessor filename disagrees with verified joblib: {run_dir}")
    metadata_path = prep.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("object_type") != object_name:
        raise ValueError(f"Wrong object in preprocessing metadata: {metadata_path}")
    training_paths = plain(cfg, "datamodule.data_paths")
    if not training_paths and plain(cfg, "datamodule.data_path"):
        training_paths = [plain(cfg, "datamodule.data_path")]
    training_names = h5_names(training_paths)
    preprocessing_names = h5_names(metadata.get("h5_files"))
    return {
        "object": object_name,
        "label": label,
        "q": q,
        "k": k,
        "d": d,
        "run_dir": str(run_dir.resolve()),
        "config": file_record(cfg_path, content_hash=True),
        "checkpoint": file_record(checkpoint, content_hash=True),
        "preprocessor": file_record(prep, content_hash=True),
        "preprocessor_metadata": file_record(metadata_path, content_hash=True),
        "training_h5_names": training_names,
        "preprocessing_h5_names": preprocessing_names,
        "controls": training_controls(cfg, training_names),
    }


def build_plan(args):
    import compare_object_google_stage1_scan as scan

    destination = args.output_dir / "plan.json"
    if destination.exists():
        raise FileExistsError(
            f"Plan already frozen: {destination}. Use evaluate/plot or a new output directory."
        )
    runs, skipped = [], []
    # The draft table has 13 configurations; the historical script has 14.
    specs = {
        label: suffix
        for label, suffix in scan.STAGE1_SPECS.items()
        if args.scan_set == "all" or label != "stage1 q8 cb2048 dim8"
    }
    for obj in OBJECTS:
        object_runs = []
        for label, suffix in specs.items():
            base = args.google_base / obj / f"{obj}_stage1_{suffix}"
            try:
                run_dir = scan.resolve_run_dir(base, label)
                if not (run_dir / "SUCCESS.txt").is_file():
                    raise FileNotFoundError(f"No SUCCESS.txt: {run_dir}")
            except FileNotFoundError as exc:
                if not args.skip_missing_runs:
                    raise
                skipped.append({"object": obj, "label": label, "reason": str(exc)})
                log.warning("Skipping %s / %s: %s", obj, label, exc)
                continue
            # Bad provenance or missing checkpoints are errors, not skippable runs.
            object_runs.append(audit_run(scan, obj, label, run_dir, args.checkpoint_name))
        if not object_runs:
            raise ValueError(f"No completed runs for {obj}")
        reference = object_runs[0]
        for run in object_runs[1:]:
            if run["controls"] != reference["controls"]:
                keys = [
                    key
                    for key in run["controls"]
                    if run["controls"][key] != reference["controls"][key]
                ]
                raise ValueError(
                    f"Non-capacity training controls differ for {obj}: "
                    f"{reference['label']} vs {run['label']}: {keys}"
                )
            if run["preprocessor"]["sha256"] != reference["preprocessor"]["sha256"]:
                raise ValueError(f"Different preprocessing transforms within {obj} scan")
        runs.extend(object_runs)
        log.info(
            "%s: %d completed runs; encoder hidden_dims=%s",
            obj,
            len(object_runs),
            reference["controls"]["model_except_capacity"]
            .get("encoder", {})
            .get("model", {})
            .get("hidden_dims"),
        )
    excluded = {
        name
        for run in runs
        for key in ("training_h5_names", "preprocessing_h5_names")
        for name in run[key]
    }
    samples = {
        "mc": select_unused_files(args.mc_dir, excluded, args.n_files, args.exclude_mc_pattern),
        "data": select_unused_files(args.data_dir, excluded, args.n_files),
    }
    if {r["path"] for r in samples["mc"]} & {r["path"] for r in samples["data"]}:
        raise ValueError("Simulation and collision-data input lists overlap")
    plan = {
        "version": VERSION,
        "scan_set": args.scan_set,
        "checkpoint_name": args.checkpoint_name,
        "runs": runs,
        "skipped": skipped,
        "samples": samples,
        "excluded_h5_names": sorted(excluded),
        "evaluation": {
            "n_bins": args.n_bins,
            "min_bin_count": args.min_bin_count,
            "min_denominator": 1e-8,
            "max_valid_objects": args.max_valid_objects,
            "batch_size": args.batch_size,
            "num_workers": 0,
            "num_events_per_file": None,
            "split": "val",
        },
        "evaluation_note": "All events in file-disjoint evaluation inputs, not the training validation split. "
        "Disjointness is based on recorded H5 basenames, not an event-ID deduplication audit.",
    }
    plan["plan_id"] = digest_json(plan)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(plan, indent=2) + "\n")
    for sample, files in samples.items():
        (args.output_dir / f"{sample}_files.txt").write_text(
            "".join(f"{r['path']}\n" for r in files)
        )
        log.info("%s: %d file-disjoint inputs", sample, len(files))
    log.info("Frozen audit: %s (no inference performed)", destination)


def load_plan(output_dir):
    plan = json.loads((output_dir / "plan.json").read_text())
    stored = plan.pop("plan_id")
    if plan["version"] != VERSION or digest_json(plan) != stored:
        raise ValueError(
            "Unsupported or modified plan. Create a new audit; do not edit a frozen plan."
        )
    plan["plan_id"] = stored
    return plan


def read_completed_metrics(record, plan, obj):
    verify_file(record)
    result = json.loads(Path(record["path"]).read_text())
    if result["plan_id"] != plan["plan_id"]:
        raise ValueError(f"Wrong source plan for {obj} metrics")
    rows = result["rows"]
    expected = {
        (sample, run["label"], i)
        for run in plan["runs"]
        if run["object"] == obj
        for sample in plan["samples"]
        for i in range(plan["evaluation"]["n_bins"])
    }
    actual = [(r["sample"], r["label"], r["bin"]) for r in rows]
    if (
        set(actual) != expected
        or len(actual) != len(expected)
        or any(r["object"] != obj for r in rows)
    ):
        raise ValueError(f"Source metrics are incomplete or duplicated for {obj}")
    return rows


def add_q8_cb2048(args):
    """Extend a completed audit without changing its inputs or copying its caches."""
    import compare_object_google_stage1_scan as scan

    destination = args.output_dir / "plan.json"
    if destination.exists() or args.output_dir.resolve() == args.source_dir.resolve():
        raise FileExistsError(
            "Use a new output directory; the completed source plan stays unchanged."
        )
    source = load_plan(args.source_dir)
    if "parent" in source:
        raise ValueError("This plan is already extended. Resume evaluate/plot on that plan.")
    source_record = file_record(args.source_dir / "plan.json", content_hash=True)
    for records in source["samples"].values():
        for record in records:
            verify_file(record)
    metrics, references, additions, skipped = {}, {}, [], []
    for obj in OBJECTS:
        metrics[obj] = file_record(args.source_dir / f"metrics_{obj}.json", content_hash=True)
        read_completed_metrics(metrics[obj], source, obj)
        old_runs = [r for r in source["runs"] if r["object"] == obj]
        if any(r["label"] == Q8_CB2048_LABEL for r in old_runs):
            continue
        base = args.google_base / obj / f"{obj}_stage1_q8_cb2048_dim8_e20"
        try:
            run_dir = scan.resolve_run_dir(base, Q8_CB2048_LABEL)
            if not (run_dir / "SUCCESS.txt").is_file():
                raise FileNotFoundError(f"No SUCCESS.txt: {run_dir}")
        except FileNotFoundError as exc:
            if not args.skip_missing_runs:
                raise
            skipped.append({"object": obj, "label": Q8_CB2048_LABEL, "reason": str(exc)})
            log.warning("Skipping %s / %s: %s", obj, Q8_CB2048_LABEL, exc)
            continue
        run = audit_run(scan, obj, Q8_CB2048_LABEL, run_dir, source["checkpoint_name"])
        reference = old_runs[0]
        for key in ("config", "preprocessor"):
            verify_file(reference[key])
        controls = training_controls(
            OmegaConf.load(reference["config"]["path"]), reference["training_h5_names"]
        )
        if run["controls"] != controls:
            raise ValueError(f"Non-capacity training controls differ for added {obj} run")
        if run["preprocessor"]["sha256"] != reference["preprocessor"]["sha256"]:
            raise ValueError(f"Preprocessing differs for added {obj} run")
        seen = set(run["training_h5_names"]) | set(run["preprocessing_h5_names"])
        for sample, records in source["samples"].items():
            overlap = seen & {Path(r["path"]).name for r in records}
            if overlap:
                raise ValueError(
                    f"Added {obj} run overlaps frozen {sample} sample: {sorted(overlap)}. "
                    "Cannot reuse this evaluation; inputs have not been changed."
                )
        references[obj] = {}
        cache_args = argparse.Namespace(**source["evaluation"])
        for sample, records in source["samples"].items():
            candidates = scan.cache_path_candidates(
                output_dir=args.source_dir / "cache" / source["plan_id"] / obj,
                object_name=obj,
                sample_label=sample,
                model_label=reference["label"],
                run_dir=Path(reference["run_dir"]),
                checkpoint=Path(reference["checkpoint"]["path"]),
                files=[r["path"] for r in records],
                preprocessor=Path(reference["preprocessor"]["path"]),
                args=cache_args,
            )
            cache = next((p for p in candidates if p.is_file()), None)
            if cache is None:
                raise FileNotFoundError(
                    f"Missing original-object reference cache for {obj}/{sample}: "
                    f"{candidates[0]}. No old model will be re-evaluated automatically."
                )
            references[obj][sample] = file_record(cache, content_hash=True)
        additions.append(run)
        log.info("%s: adding only %s; keeping all previous results", obj, Q8_CB2048_LABEL)
    if not additions:
        raise ValueError("No additional completed q8/cb2048/dim8 runs to include")
    plan = deepcopy(source)
    plan.pop("plan_id")
    plan["scan_set"] = "all"
    plan["runs"].extend(additions)
    added_keys = {(r["object"], r["label"]) for r in additions + skipped}
    plan["skipped"] = [
        r for r in source["skipped"] if (r["object"], r["label"]) not in added_keys
    ] + skipped
    plan["excluded_h5_names"] = sorted(
        set(source["excluded_h5_names"])
        | {
            name
            for r in additions
            for key in ("training_h5_names", "preprocessing_h5_names")
            for name in r[key]
        }
    )
    plan["parent"] = {
        "plan": source_record,
        "plan_id": source["plan_id"],
        "metrics": metrics,
        "reference_caches": references,
    }
    plan["plan_id"] = digest_json(plan)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(plan, indent=2) + "\n")
    for sample, records in plan["samples"].items():
        (args.output_dir / f"{sample}_files.txt").write_text(
            "".join(f"{r['path']}\n" for r in records)
        )
    log.info(
        "Frozen extension: %s; %d new model/sample evaluations; old results reused, "
        "no inference performed and no array caches copied",
        destination,
        len(additions) * len(plan["samples"]),
    )


def inherited_reference(plan, source, rows, obj, sample):
    record = plan["parent"]["reference_caches"][obj][sample]
    verify_file(record)
    old_run = next(r for r in source["runs"] if r["object"] == obj)
    log.info("Reading original-object reference from %s (no inference)", record["path"])
    with np.load(record["path"], allow_pickle=False) as cached:
        if (
            str(cached["checkpoint"]) != old_run["checkpoint"]["path"]
            or int(cached["codebook_size"]) != old_run["k"]
        ):
            raise ValueError(f"Wrong reference cache for {obj}/{sample}")
        reference = {
            "original": cached["original"],
            "feature_names": [str(v) for v in cached["feature_names"]],
        }
    reference["reconstruction"] = reference["original"]
    binned = sorted(
        (r for r in rows if r["sample"] == sample and r["label"] == old_run["label"]),
        key=lambda r: r["bin"],
    )
    if not binned or any(r["evaluated_objects"] != len(reference["original"]) for r in binned):
        raise ValueError(
            f"Reference cache object count disagrees with completed metrics: {obj}/{sample}"
        )
    bins = np.array([r["bin_low"] for r in binned] + [binned[-1]["bin_high"]])
    return reference, bins


def validate_checkpoint(run):
    import torch

    checkpoint = torch.load(run["checkpoint"]["path"], map_location="cpu", weights_only=False)
    hparams = checkpoint["hyper_parameters"]
    actual = tuple(int(hparams[key]) for key in ("num_quantizers", "codebook_size", "codebook_dim"))
    expected = (run["q"], run["k"], run["d"])
    if actual != expected:
        raise ValueError(
            f"Checkpoint capacity {actual} != audited config {expected}: {run['checkpoint']['path']}"
        )


def response_rows(scan, original, decoded, bins, plan, run, sample):
    settings = plan["evaluation"]
    centers, values, counts, _, _ = scan.binned_metric_values(
        original,
        decoded,
        bins,
        metric_kind="ratio_iqr_over_median",
        min_bin_count=settings["min_bin_count"],
        min_denominator=settings["min_denominator"],
    )
    return [
        {
            "object": run["object"],
            "sample": sample,
            "label": run["label"],
            "q": run["q"],
            "k": run["k"],
            "d": run["d"],
            "bin": i,
            "bin_low": float(bins[i]),
            "bin_high": float(bins[i + 1]),
            "pt": float(center),
            "response_width_percent": float(value) * 100 if np.isfinite(value) else None,
            "objects_in_bin": int(count),
            "evaluated_objects": len(original),
        }
        for i, (center, value, count) in enumerate(zip(centers, values, counts))
    ]


def evaluate(args):
    import torch
    import compare_object_google_stage1_scan as scan

    plan = load_plan(args.output_dir)
    source = None
    inherited_keys = set()
    if "parent" in plan:
        verify_file(plan["parent"]["plan"])
        source = load_plan(Path(plan["parent"]["plan"]["path"]).parent)
        if source["plan_id"] != plan["parent"]["plan_id"]:
            raise ValueError("Wrong parent plan")
        inherited_keys = {(r["object"], r["label"]) for r in source["runs"]}
    for files in plan["samples"].values():
        for record in files:
            verify_file(record)
    for obj in args.objects:
        runs = [
            run
            for run in plan["runs"]
            if run["object"] == obj and (obj, run["label"]) not in inherited_keys
        ]
        for run in runs:
            for key in ("config", "checkpoint", "preprocessor", "preprocessor_metadata"):
                verify_file(run[key])
            validate_checkpoint(run)
        rows = read_completed_metrics(plan["parent"]["metrics"][obj], source, obj) if source else []
        collect_args = argparse.Namespace(
            **plan["evaluation"],
            object=obj,
            google_preprocessor=None,
            checkpoint_name=plan["checkpoint_name"],
            force=False,
            cache_only=args.cache_only,
        )
        for sample, records in plan["samples"].items():
            if not runs:
                continue
            reference, bins = None, None
            if source:
                reference, bins = inherited_reference(plan, source, rows, obj, sample)
                pt_index = [name.lower() for name in reference["feature_names"]].index("pt")
                original_pt = reference["original"][:, pt_index]
            for run in runs:
                item = scan.collect_one(
                    sample_label=sample,
                    model_label=run["label"],
                    run_dir=Path(run["run_dir"]),
                    files=[r["path"] for r in records],
                    output_dir=args.output_dir / "cache" / plan["plan_id"] / obj,
                    args=collect_args,
                    device=torch.device(args.device),
                )
                indices = np.asarray(item["indices"])
                nq = indices.shape[-1] if indices.ndim > 1 else 1
                if nq != run["q"] or item["codebook_size"] != run["k"]:
                    raise ValueError(f"Wrong model capacity in arrays for {obj} / {run['label']}")
                if reference is None:
                    # Keep one original array, not every model's full diagnostics.
                    reference = {
                        "original": item["original"],
                        "feature_names": item["feature_names"],
                        "reconstruction": item["original"],
                    }
                    names = [name.lower() for name in item["feature_names"]]
                    pt_index = names.index("pt")
                    original_pt = reference["original"][:, pt_index]
                    if not np.all(np.isfinite(original_pt) & (original_pt > 0)):
                        raise ValueError(
                            f"Nonpositive/nonfinite original pT in {obj}; inspect inputs before plotting."
                        )
                    bins = scan.domain.make_bins(original_pt, plan["evaluation"]["n_bins"])
                if item["feature_names"] != reference["feature_names"]:
                    raise ValueError(f"Feature ordering differs within {obj}")
                scan.domain.verify_shared_inputs(
                    sample,
                    obj,
                    {"reference": reference, run["label"]: item},
                    reference["feature_names"],
                )
                decoded_pt = item["reconstruction"][:, pt_index]
                if not np.all(np.isfinite(decoded_pt) & (decoded_pt > 0)):
                    raise ValueError(
                        f"Nonpositive/nonfinite decoded pT for {obj} / {run['label']}; inspect diagnostics."
                    )
                rows.extend(response_rows(scan, original_pt, decoded_pt, bins, plan, run, sample))
                del item
        output = args.output_dir / f"metrics_{obj}.json"
        temporary = output.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"plan_id": plan["plan_id"], "rows": rows}, indent=2, allow_nan=False) + "\n"
        )
        temporary.replace(output)
        log.info("Saved %s", output)


def response_plot_values(rows, y_scale):
    values = np.array([r["response_width_percent"] for r in rows], dtype=float)
    values[~np.isfinite(values)] = np.nan
    if y_scale == "log":
        # Leave gaps for zero widths; do not invent an epsilon response.
        values[values <= 0] = np.nan
    return values


def response_axis_limits(rows, y_scale):
    values = response_plot_values(rows, y_scale)
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError(f"No finite {'positive ' if y_scale == 'log' else ''}response widths")
    if y_scale == "linear":
        return 0.0, max(float(finite.max()) * 1.12, 0.01)
    low, high = np.log10([finite.min(), finite.max()])
    padding = max(0.08 * (high - low), 0.08)
    return 10 ** (low - padding), 10 ** (high + padding)


def render_figures(rows, output_dir, *, title_suffix="", y_scale="linear"):
    if y_scale not in ("linear", "log"):
        raise ValueError(f"Unsupported y scale: {y_scale}")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter, StrMethodFormatter
    from paper_plot_style import (
        OBJECT_LABELS,
        PT_LABEL,
        RESPONSE_LABEL,
        capacity_curve_style,
        capacity_legends,
        paper_style,
        save_figure,
        style_axis,
    )

    configs = sorted({(r["q"], r["k"], r["d"]) for r in rows})
    suffix = "_log" if y_scale == "log" else ""
    # Use both domains to preserve matched limits between the two figures.
    limits = {
        obj: response_axis_limits([r for r in rows if r["object"] == obj], y_scale)
        for obj in OBJECTS
    }
    with paper_style():
        for sample, sample_label in (("mc", "Simulation"), ("data", "Collision data")):
            fig, axes = plt.subplots(2, 3, figsize=(12.0, 8.2))
            highlighted_objects = {}
            fig.subplots_adjust(
                left=0.085, right=0.985, bottom=0.26, top=0.95, wspace=0.25, hspace=0.31
            )
            if title_suffix:
                fig.text(0.535, 0.99, title_suffix.strip(" |"), ha="center", va="top", fontsize=8)
            for panel, (ax, obj, label) in enumerate(zip(axes.flat, OBJECTS, OBJECT_LABELS)):
                object_rows = [row for row in rows if row["object"] == obj]
                chosen_config = SELECTED_CONFIGURATIONS[obj]
                if not any(
                    r["sample"] == sample and (r["q"], r["k"], r["d"]) == chosen_config
                    for r in object_rows
                ):
                    log.warning(
                        "%s/%s: selected q%d/cb%d/dim%d is absent from saved metrics; "
                        "no substitute will be highlighted",
                        obj,
                        sample,
                        *chosen_config,
                    )
                ax.set_yscale(y_scale)
                for q, k, d in configs:
                    selected = sorted(
                        (
                            r
                            for r in object_rows
                            if r["sample"] == sample and (r["q"], r["k"], r["d"]) == (q, k, d)
                        ),
                        key=lambda r: r["bin"],
                    )
                    if not selected:
                        continue
                    if not any(r["response_width_percent"] is not None for r in selected):
                        raise ValueError(
                            f"No populated pT bins for {obj}/{sample}/q{q}/cb{k}/dim{d}. "
                            "Inspect object counts before making a publication figure."
                        )
                    if y_scale == "log":
                        omitted = sum(
                            r["response_width_percent"] is not None
                            and r["response_width_percent"] <= 0
                            for r in selected
                        )
                        if omitted:
                            log.warning(
                                "%s/%s/q%d/cb%d/dim%d: omitting %d nonpositive widths on log axis",
                                obj,
                                sample,
                                q,
                                k,
                                d,
                                omitted,
                            )
                    highlight = (q, k, d) == chosen_config
                    if highlight:
                        highlighted_objects.setdefault(chosen_config, []).append(label)
                    ax.plot(
                        [r["pt"] for r in selected],
                        response_plot_values(selected, y_scale),
                        **capacity_curve_style(q, k, d, selected=highlight),
                    )
                style_axis(ax)
                ax.set_title(f"({chr(97 + panel)}) {label}", loc="left")
                ax.set_ylim(*limits[obj])
                if panel == 0:
                    ax.text(
                        0.035,
                        0.96,
                        sample_label,
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=9,
                    )
                if y_scale == "log":
                    ax.yaxis.set_major_locator(LogLocator(base=10))
                    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
                    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
                    ax.yaxis.set_minor_formatter(NullFormatter())
                if panel % 3 == 0:
                    ax.set_ylabel(RESPONSE_LABEL)
                if panel >= 3:
                    ax.set_xlabel(PT_LABEL)
            capacity_legends(
                fig,
                configs,
                highlighted_objects,
                [(ax.get_position().x0 + ax.get_position().x1) / 2 for ax in axes[1]],
            )
            save_figure(fig, output_dir / f"figure2_capacity_{sample}{suffix}")
            plt.close(fig)


def plot(args):
    plan = load_plan(args.output_dir)
    rows = []
    for obj in OBJECTS:
        result = json.loads((args.output_dir / f"metrics_{obj}.json").read_text())
        if result["plan_id"] != plan["plan_id"]:
            raise ValueError(f"Wrong audit plan for {obj} metrics")
        rows.extend(result["rows"])
    render_figures(rows, args.output_dir, y_scale=args.y_scale)
    with (args.output_dir / "figure2_capacity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    skipped = (
        "\n".join(f"- {r['object']}: {r['label']} ({r['reason']})" for r in plan["skipped"])
        or "None."
    )
    suffix = "_log" if args.y_scale == "log" else ""
    if args.y_scale == "log":
        omitted = sum(
            r["response_width_percent"] is not None and r["response_width_percent"] <= 0
            for r in rows
        )
        axes_caption = (
            "Y axes are logarithmic and match between domains for each object. "
            f"Nonpositive widths are not displayed ({omitted} bin points); no epsilon is added. "
        )
    else:
        axes_caption = "Linear y axes start at zero and match between domains for each object. "
    caption = (
        "# Figure 2: Tokenizer capacity scan\n\n"
        "Relative response width R_pT = 100 * IQR(r) / median(r), with "
        "r = pT_decoded / pT_original, in bins of original detector-reconstructed pT. "
        "Original does not mean generator-level truth. "
        "Each domain uses the same file list and ordered objects for every configuration of each object type. "
        "Files recorded in any selected run's model or preprocessing inputs are excluded. "
        "This is file-identity disjointness, not a cross-file event-ID deduplication guarantee. "
        f"Checkpoint: {plan['checkpoint_name']}, with no fallback. "
        f"At most {plan['evaluation']['max_valid_objects']:,} valid objects per object/domain; "
        "actual counts are in the CSV. Equal-width bins span the original pT 1st--99th percentiles "
        "(min/max fallback for degenerate percentile ranges). "
        f"Bins with fewer than {plan['evaluation']['min_bin_count']} valid responses are omitted. "
        "Colour encodes N_q, line style encodes K, and marker encodes d_c, except for the "
        "selected per-object configurations, which use thick black lines and stars: "
        "q8/cb2048/dim8 for jets, electrons, muons, and photons; q8/cb4096/dim8 for taus and tracks. "
        "These mark the stage-1 scan configurations matching the current tokenizer choices, "
        "not full-training results. Missing selections are not replaced with another configuration. "
        f"{axes_caption}"
        "No seed or sampling uncertainty is estimated.\n\n"
        "## Missing/incomplete configurations\n" + skipped + "\n"
    )
    (args.output_dir / f"figure2_caption{suffix}.md").write_text(caption)
    log.info("Saved simulation/data PDF and PNG, CSV, and caption under %s", args.output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("plan", help="Audit runs and freeze file-disjoint inputs; no inference")
    audit.add_argument("--google-base", type=Path, default=Path("google_results"))
    audit.add_argument("--mc-dir", type=Path, required=True)
    audit.add_argument("--data-dir", type=Path, required=True)
    audit.add_argument("--exclude-mc-pattern", default="DAOD_PHYSLITE.370016*")
    audit.add_argument(
        "--scan-set",
        choices=("draft", "all"),
        default="all",
        help="all (default): 14 configurations including q8/cb2048/dim8; draft: legacy 13",
    )
    audit.add_argument(
        "--skip-missing-runs", action="store_true", help="Record absent/incomplete runs explicitly"
    )
    audit.add_argument("--checkpoint-name", choices=("best.ckpt", "last.ckpt"), required=True)
    audit.add_argument("--n-files", type=int, default=20)
    audit.add_argument("--max-valid-objects", type=int, default=1_000_000)
    audit.add_argument("--batch-size", type=int, default=256)
    audit.add_argument("--n-bins", type=int, default=14)
    audit.add_argument("--min-bin-count", type=int, default=50)
    extension = sub.add_parser(
        "add-q8-cb2048",
        help="Add q8/cb2048/dim8 to a completed plan, reusing its samples and results",
    )
    extension.add_argument("--source-dir", type=Path, required=True)
    extension.add_argument("--google-base", type=Path, default=Path("google_results"))
    extension.add_argument("--skip-missing-runs", action="store_true")
    evaluation = sub.add_parser(
        "evaluate", help="Evaluate exact audited runs; reuse only this plan's caches"
    )
    evaluation.add_argument("--device", default="cpu")
    evaluation.add_argument("--objects", nargs="+", choices=OBJECTS, default=list(OBJECTS))
    evaluation.add_argument("--cache-only", action="store_true")
    plotting = sub.add_parser(
        "plot", help="Render frozen metrics without inference/checkpoints/GPU"
    )
    plotting.add_argument(
        "--y-scale",
        choices=("linear", "log"),
        default="linear",
        help="Display scale only; log writes separate *_log figures without changing the metrics",
    )
    for cmd in (audit, extension, evaluation, plotting):
        cmd.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan" and any(
        getattr(args, key) <= 0
        for key in ("n_files", "max_valid_objects", "batch_size", "n_bins", "min_bin_count")
    ):
        parser.error("File, object, batch, bin and minimum-count settings must be positive")
    return args


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    {"plan": build_plan, "add-q8-cb2048": add_q8_cb2048, "evaluate": evaluate, "plot": plot}[
        args.command
    ](args)


if __name__ == "__main__":
    main()
