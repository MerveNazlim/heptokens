#!/usr/bin/env python3
"""Full Q8 reconstruction: frozen audit, paired inference, summaries, paper plots.

Uses the existing canonical diagnostic loader, never stage-1 model selection or
legacy cache discovery. Plotting reads small summaries only. No training runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

import paper_tokenizer_capacity as capacity
from paper_plot_style import OBJECTS, OBJECT_LABELS, paper_style, save_figure, style_axis
from paper_tokenizer_features import read_arrays

log = logging.getLogger(__name__)
VERSION = "paper-full-q8-reconstruction-v1"
DOMAINS = {"mc": "Simulation", "data": "Collision data"}
ORIGINAL_COLOR = "#404040"
DECODED_COLOR = "#235789"
ENERGY_FEATURES = {
    "pt",
    "mass",
    "ptvarcone30",
    "topoetcone20",
    "topoetcone40",
    "ptcone20",
    "trk_iso03",
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def seal(value, key):
    return {**value, key: capacity.digest_json(value)}


def read_sealed(path, key):
    value = json.loads(path.read_text())
    identity = value.pop(key)
    if capacity.digest_json(value) != identity:
        raise ValueError(f"Modified {key}: {path}")
    return {**value, key: identity}


def load_plan(directory):
    plan = read_sealed(directory / "plan.json", "plan_id")
    if plan["version"] != VERSION:
        raise ValueError(
            "Not a full Q8 reconstruction plan; stage-1 plans cannot be evaluated here"
        )
    return plan


def audit_run(run_base, obj, checkpoint_name, *, configuration=None):
    q, k, d = configuration if configuration is not None else capacity.SELECTED_CONFIGURATIONS[obj]
    name = f"{obj}_full_dim{d}_cb{k}_q{q}_e20_new_mcdata"
    run_dir = (run_base / name).resolve()
    if not (run_dir / "SUCCESS.txt").is_file():
        raise FileNotFoundError(f"Full training is not marked complete: {run_dir}/SUCCESS.txt")
    cfg = OmegaConf.load(run_dir / "full_config.yaml")
    actual = tuple(
        int(capacity.plain(cfg, f"model.{key}"))
        for key in ("num_quantizers", "codebook_size", "codebook_dim")
    )
    if actual != (q, k, d) or capacity.plain(cfg, "datamodule.object_type") != obj:
        raise ValueError(
            f"Wrong full-training configuration for {obj}: {actual}, expected {(q, k, d)}"
        )
    if capacity.plain(cfg, "datamodule.output_mode") != "object":
        raise ValueError(f"Expected object-mode tokenizer: {run_dir}")
    checkpoint = run_dir / "checkpoints" / checkpoint_name
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing exact checkpoint: {checkpoint}; no fallback allowed")
    saved = capacity.plain(cfg, "datamodule.transforms.preprocess.cst_fn.filename")
    if not saved:
        raise ValueError(f"Missing saved preprocessing path: {run_dir}")
    prep = Path(saved)
    prep = (run_dir / prep).resolve() if not prep.is_absolute() else prep.resolve()
    if not prep.is_file():
        raise FileNotFoundError(
            f"Saved full-training joblib missing: {prep}; no substitution allowed"
        )
    metadata_path = prep.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    collections = capacity.plain(cfg, "datamodule.object_collections") or []
    matches = [c for c in collections if c["object_name"] == obj]
    if len(matches) != 1:
        raise ValueError(f"Missing/ambiguous saved feature configuration for {obj}")
    inputs = matches[0]["inputs"]
    names = [Path(p).name for p in inputs]
    if (
        metadata.get("object_type") != obj
        or metadata.get("feature_paths") != inputs
        or metadata.get("feature_names") != names
    ):
        raise ValueError(
            f"Preprocessor object/feature order differs from saved training: {metadata_path}"
        )
    training = capacity.plain(cfg, "datamodule.data_paths")
    if not training and capacity.plain(cfg, "datamodule.data_path"):
        training = [capacity.plain(cfg, "datamodule.data_path")]
    training_names = capacity.h5_names(training)
    return {
        "object": obj,
        "label": f"full q{q} cb{k} dim{d}",
        "q": q,
        "k": k,
        "d": d,
        "run_dir": str(run_dir),
        "feature_names": names,
        "config": capacity.file_record(run_dir / "full_config.yaml", content_hash=True),
        "completion": capacity.file_record(run_dir / "SUCCESS.txt", content_hash=True),
        "checkpoint": capacity.file_record(checkpoint, content_hash=True),
        "preprocessor": capacity.file_record(prep, content_hash=True),
        "preprocessor_metadata": capacity.file_record(metadata_path, content_hash=True),
        "training_h5_names": training_names,
        "preprocessing_h5_names": capacity.h5_names(metadata.get("h5_files")),
        "controls": capacity.training_controls(cfg, training_names),
    }


def read_file_list(path):
    names = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    capacity.h5_names(names)
    if any(not Path(p).is_absolute() for p in names):
        raise ValueError(f"Use absolute H5 paths in {path}")
    records = [capacity.file_record(p) for p in names]
    if any(not r["size"] for r in records):
        raise ValueError(f"Empty evaluation file in {path}")
    return records


def verify_holdout(runs, samples):
    sample_names = {
        domain: set(capacity.h5_names([r["path"] for r in records]))
        for domain, records in samples.items()
    }
    if sample_names["mc"] & sample_names["data"]:
        raise ValueError("MC and data H5 identities overlap")
    overlaps = []
    for run in runs:
        seen = set(run["training_h5_names"]) | set(run["preprocessing_h5_names"])
        for sample, names in sample_names.items():
            common = sorted(seen & names)
            if common:
                overlaps.append(f"{run['object']}/{sample}: {len(common)} files, e.g. {common[:3]}")
    if overlaps:
        raise ValueError(
            "NOT held out from full training/preprocessing:\n"
            + "\n".join(overlaps)
            + "\nStage-1 holdout is not automatically full-training holdout. "
            "Supply unused files; no inference or sample replacement performed."
        )


def build_plan(args):
    destination = args.output_dir / "plan.json"
    if destination.exists():
        raise FileExistsError(
            f"Audit already frozen: {destination}; use evaluate/plot or a new directory"
        )
    runs = [audit_run(args.run_base, obj, args.checkpoint_name) for obj in OBJECTS]
    if args.sample_plan:
        source = capacity.load_plan(args.sample_plan)
        samples = source["samples"]
        settings = {
            key: source["evaluation"][key]
            for key in (
                "max_valid_objects",
                "batch_size",
                "num_workers",
                "num_events_per_file",
                "split",
            )
        }
        source_record = capacity.file_record(args.sample_plan / "plan.json", content_hash=True)
    else:
        samples = {"mc": read_file_list(args.mc_files), "data": read_file_list(args.data_files)}
        settings = {
            "max_valid_objects": args.max_valid_objects,
            "batch_size": args.batch_size,
            "num_workers": 0,
            "num_events_per_file": None,
            "split": "val",
        }
        source_record = None
    for records in samples.values():
        for record in records:
            capacity.verify_file(record)
    protocol = getattr(args, "evaluation_protocol", "file-holdout")
    test_partition = None
    note = (
        "All events from file-disjoint inputs, excluding model and preprocessor "
        "input basenames. Not a cross-file event-ID deduplication guarantee."
    )
    if protocol == "saved-test":
        from paper_tokenizer_test_split import CAVEAT, freeze_partition

        if not getattr(args, "acknowledge_preprocessing_overlap", False):
            raise ValueError("Saved-test requires explicit --acknowledge-preprocessing-overlap")
        test_partition = freeze_partition(runs, samples, args.output_dir)
        settings.update(split="test", num_workers=0, num_events_per_file=None)
        note = CAVEAT
        log.warning(note)
    else:
        verify_holdout(runs, samples)
    plan = seal(
        {
            "version": VERSION,
            "runs": runs,
            "samples": samples,
            "sample_source": source_record,
            "checkpoint_name": args.checkpoint_name,
            "evaluation": settings,
            "evaluation_protocol": protocol,
            "test_partition": test_partition,
            "input_momentum_unit": args.input_momentum_unit,
            "statistics": {
                "n_bins": args.n_bins,
                "min_ratio_count": args.min_ratio_count,
                "percentiles": [0.5, 99.5],
            },
            "holdout_note": note,
        },
        "plan_id",
    )
    write_json(destination, plan)
    for sample, records in samples.items():
        (args.output_dir / f"{sample}_files.txt").write_text(
            "".join(f"{r['path']}\n" for r in records)
        )
    for run in runs:
        log.info(
            "%s: %s; %s; joblib=%s",
            run["object"],
            run["label"],
            run["checkpoint"]["path"],
            run["preprocessor"]["path"],
        )
    log.info("Frozen full-training audit: %s; no inference performed", destination)


def cache_path(directory, plan, run, sample):
    return directory / "cache" / plan["plan_id"] / run["object"] / f"{sample}.npz"


def read_cache(path, plan, run, sample):
    receipt = read_sealed(path.with_suffix(".json"), "receipt_id")
    if (
        receipt["plan_id"] != plan["plan_id"]
        or receipt["object"] != run["object"]
        or receipt["sample"] != sample
        or receipt["arrays"]["path"] != str(path.resolve())
    ):
        raise ValueError(f"Wrong paired-array provenance: {path}")
    capacity.verify_file(receipt["arrays"])
    arrays = read_arrays(path, run)
    if arrays["feature_names"] != run["feature_names"]:
        raise ValueError(f"Cached feature order differs from full training: {path}")
    return arrays, receipt


def infer(run, plan, sample, device):
    import torch
    import analyze_vqvae_tokenizer as diagnostics
    from compare_object_google_stage1_scan import force_common_eval_split

    capacity.validate_checkpoint(run)
    cfg = OmegaConf.load(run["config"]["path"])
    OmegaConf.update(
        cfg, "datamodule.transforms.preprocess.cst_fn.filename", run["preprocessor"]["path"]
    )
    saved_test = plan.get("evaluation_protocol") == "saved-test"
    if not saved_test:
        force_common_eval_split(cfg)
        cfg.datamodule.num_events = plan["evaluation"]["num_events_per_file"]
    transforms, inverse = diagnostics.transform_list_and_cst_fn_from_cfg(cfg)
    if inverse is None:
        raise ValueError("No inverse preprocessor found; physical-unit plots cannot be produced")
    device = torch.device(device)
    model = diagnostics.load_analysis_model(Path(run["run_dir"]), run["checkpoint"]["path"], device)
    try:
        if saved_test:
            from paper_tokenizer_test_split import test_loader

            loader = test_loader(cfg, plan, sample, transforms)
            original, decoded, indices, _ = diagnostics.collect_diagnostics_from_loader(
                model=model,
                loader=loader,
                cst_inverse_transformer=inverse,
                device=device,
                max_valid_objects=plan["evaluation"]["max_valid_objects"],
            )
        else:
            original, decoded, indices, _ = diagnostics.collect_diagnostics_for_h5_files(
                cfg=cfg,
                model=model,
                h5_files=[r["path"] for r in plan["samples"][sample]],
                cst_inverse_transformer=inverse,
                device=device,
                **plan["evaluation"],
            )
        return {
            "original": original,
            "reconstruction": decoded,
            "indices": indices,
            "feature_names": np.asarray(run["feature_names"]),
            "checkpoint": np.asarray(run["checkpoint"]["path"]),
            "codebook_size": np.asarray(run["k"]),
        }
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def evaluate(args):
    plan = load_plan(args.output_dir)
    if plan.get("evaluation_protocol") == "saved-test":
        from paper_tokenizer_test_split import verify_partition

        verify_partition(plan)
    for records in plan["samples"].values():
        for record in records:
            capacity.verify_file(record)
    for run in plan["runs"]:
        if run["object"] not in args.objects:
            continue
        for key in ("config", "completion", "checkpoint", "preprocessor", "preprocessor_metadata"):
            capacity.verify_file(run[key])
        for sample in DOMAINS:
            path = cache_path(args.output_dir, plan, run, sample)
            if path.exists() or path.with_suffix(".json").exists():
                read_cache(path, plan, run, sample)
                log.info("Reusing verified cache: %s", path)
                continue
            if args.cache_only:
                raise FileNotFoundError(f"Missing full-training cache: {path}; inference forbidden")
            log.info("Evaluating %s/%s with %s", run["object"], sample, run["checkpoint"]["path"])
            arrays = infer(run, plan, sample, args.device)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **arrays)
            read_arrays(temporary, run)
            if list(arrays["feature_names"]) != run["feature_names"]:
                raise ValueError("Inference feature order mismatch")
            temporary.replace(path)
            write_json(
                path.with_suffix(".json"),
                seal(
                    {
                        "plan_id": plan["plan_id"],
                        "object": run["object"],
                        "sample": sample,
                        "arrays": capacity.file_record(path, content_hash=True),
                    },
                    "receipt_id",
                ),
            )
            del arrays
            log.info("Saved paired arrays for every feature: %s", path)


def display_feature(name, unit):
    # Follow the scan's feature units, but require explicit input units instead
    # of deciding GeV/MeV from the magnitude of a physics distribution.
    from analyze_vqvae_tokenizer import feature_label

    energy = name.lower() in ENERGY_FEATURES
    return {
        "name": name,
        "label": feature_label(name),
        "unit": "GeV" if energy else "",
        "scale": 0.001 if energy and unit == "MeV" else 1.0,
    }


def bin_edges(values, n_bins, percentiles, *, discrete=False):
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        return np.linspace(-0.5, 0.5, n_bins + 1)
    if discrete:
        rounded = np.rint(finite)
        if np.allclose(finite, rounded, atol=1e-5, rtol=0) and rounded.max() - rounded.min() <= 32:
            return np.arange(rounded.min() - 0.5, rounded.max() + 1.5)
    low, high = np.percentile(finite, percentiles)
    if low == high:
        low, high = float(finite.min()), float(finite.max())
    if low == high:
        width = max(abs(low) * 0.01, 0.5)
        low, high = low - width, high + width
    return np.linspace(low, high, n_bins + 1)


def feature_summary(original, decoded, edges, residual_edges, min_count, name):
    finite = np.isfinite(original) & np.isfinite(decoded)
    x, y = original[finite], decoded[finite]
    residual = y - x
    n = len(x)
    counts_x = np.histogram(x, edges)[0]
    counts_y = np.histogram(y, edges)[0]
    counts_r = np.histogram(residual, residual_edges)[0]
    paired = np.histogram2d(x, y, bins=(edges, edges))[0].astype(np.int64)
    ratios = [float(b / a) if a >= min_count else None for a, b in zip(counts_x, counts_y)]
    result = {
        "total_objects": len(original),
        "finite_pairs": n,
        "nonfinite_pairs": int((~finite).sum()),
        "original_counts": counts_x.tolist(),
        "decoded_counts": counts_y.tolist(),
        "bin_count_ratio": ratios,
        "residual_counts": counts_r.tolist(),
        "paired_counts": paired.tolist(),
        "outside_original": int(n - counts_x.sum()),
        "outside_decoded": int(n - counts_y.sum()),
        "outside_residual": int(n - counts_r.sum()),
        "outside_paired": int(n - paired.sum()),
        "median_residual": float(np.median(residual)) if n else None,
        "median_absolute_residual": float(np.median(np.abs(residual))) if n else None,
        "residual_iqr": float(np.diff(np.percentile(residual, [25, 75]))[0]) if n else None,
    }
    if name.lower() == "pt":
        valid = (x > 1e-8) & (y > 0)
        response = y[valid] / x[valid]
        result.update(
            {
                "response_pairs": int(valid.sum()),
                "excluded_response_pairs": int(n - valid.sum()),
                "response_width_percent": (
                    float(100 * np.diff(np.percentile(response, [25, 75]))[0] / np.median(response))
                    if len(response)
                    else None
                ),
                "median_relative_residual_percent": (
                    float(100 * np.median(response - 1)) if len(response) else None
                ),
            }
        )
    return result


def summarize_arrays(items, plan, run):
    settings = plan["statistics"]
    features = []
    for i, name in enumerate(run["feature_names"]):
        feature = display_feature(name, plan["input_momentum_unit"])
        pairs = {
            s: (
                np.asarray(a["original"][:, i], dtype=float) * feature["scale"],
                np.asarray(a["reconstruction"][:, i], dtype=float) * feature["scale"],
            )
            for s, a in items.items()
        }
        combined = np.concatenate([x[np.isfinite(x) & np.isfinite(y)] for x, y in pairs.values()])
        residuals = np.concatenate(
            [(y - x)[np.isfinite(x) & np.isfinite(y)] for x, y in pairs.values()]
        )
        edges = bin_edges(combined, settings["n_bins"], settings["percentiles"], discrete=True)
        residual_edges = bin_edges(residuals, settings["n_bins"], settings["percentiles"])
        feature.update(
            {
                "edges": edges.tolist(),
                "residual_edges": residual_edges.tolist(),
                "domains": {
                    s: feature_summary(
                        x, y, edges, residual_edges, settings["min_ratio_count"], name
                    )
                    for s, (x, y) in pairs.items()
                },
            }
        )
        features.append(feature)
    return features


def load_summary(directory, plan, obj, *, verify_sources=False):
    path = directory / "summaries" / f"{obj}.json"
    result = read_sealed(path, "summary_id")
    if result["plan_id"] != plan["plan_id"] or result["object"] != obj:
        raise ValueError(f"Wrong summary provenance: {path}")
    run = next(r for r in plan["runs"] if r["object"] == obj)
    if [f["name"] for f in result["features"]] != run["feature_names"]:
        raise ValueError(f"Wrong summary feature order: {path}")
    if verify_sources:
        for record in result["sources"]:
            capacity.verify_file(record)
    return result


def summarize(args):
    plan = load_plan(args.output_dir)
    for run in plan["runs"]:
        obj = run["object"]
        if obj not in args.objects:
            continue
        destination = args.output_dir / "summaries" / f"{obj}.json"
        if destination.exists():
            load_summary(args.output_dir, plan, obj, verify_sources=True)
            log.info("Reusing summaries: %s", destination)
            continue
        items, sources = {}, []
        for sample in DOMAINS:
            path = cache_path(args.output_dir, plan, run, sample)
            items[sample], receipt = read_cache(path, plan, run, sample)
            sources.extend(
                [
                    receipt["arrays"],
                    capacity.file_record(path.with_suffix(".json"), content_hash=True),
                ]
            )
        features = summarize_arrays(items, plan, run)
        write_json(
            destination,
            seal(
                {
                    "plan_id": plan["plan_id"],
                    "object": obj,
                    "sources": sources,
                    "features": features,
                },
                "summary_id",
            ),
        )
        del items
        log.info("Saved all-feature summaries: %s; no inference", destination)


def axis_label(feature, prefix=""):
    return prefix + feature["label"] + (f" [{feature['unit']}]" if feature["unit"] else "")


def draw_distribution(
    fig,
    cell,
    feature,
    sample,
    *,
    log_y=True,
    count_label="objects",
    reference_label="Original",
    prediction_label="Decoded Q8",
    ratio_label="Decoded Q8 /\nOriginal",
    ratio_ylim=None,
):
    from matplotlib.ticker import LogLocator, NullFormatter, StrMethodFormatter

    if ratio_ylim is not None:
        ratio_low, ratio_high = ratio_ylim
        if not np.isfinite([ratio_low, ratio_high]).all() or not 0 <= ratio_low < 1 < ratio_high:
            raise ValueError("Ratio limits must be finite and satisfy 0 <= lower < 1 < upper")
    inner = cell.subgridspec(2, 1, height_ratios=(3.3, 1), hspace=0.04)
    ax = fig.add_subplot(inner[0])
    ratio_ax = fig.add_subplot(inner[1], sharex=ax)
    row = feature["domains"][sample]
    edges = np.asarray(feature["edges"])
    n = max(row["finite_pairs"], 1)
    has_input = "input_counts" in row
    fill_counts = row["input_counts"] if has_input else row["original_counts"]
    ax.stairs(np.asarray(fill_counts) / n, edges, color="#C5C9CD", fill=True, alpha=0.45)
    if has_input:
        ax.stairs(
            np.asarray(row["input_counts"]) / n, edges, color=ORIGINAL_COLOR, label="Original"
        )
    ax.stairs(
        np.asarray(row["original_counts"]) / n,
        edges,
        color="#008B72" if has_input else ORIGINAL_COLOR,
        linestyle="--" if has_input else "-",
        label=reference_label,
    )
    ax.stairs(
        np.asarray(row["decoded_counts"]) / n, edges, color=DECODED_COLOR, label=prediction_label
    )
    ax.set_ylabel(f"Normalized {count_label}")
    ax.tick_params(labelbottom=False)
    visible = np.concatenate(
        [
            np.asarray(r[key], dtype=float) / max(r["finite_pairs"], 1)
            for r in feature["domains"].values()
            for key in ("original_counts", "decoded_counts", "input_counts")
            if key in r
        ]
    )
    positive = visible[visible > 0]
    if log_y and len(positive):
        ax.set_yscale("log")
        high = float(positive.max()) * 2
        low = min(float(positive.min()) * 0.5, high / 10)
        ax.set_ylim(low, high)
        if high / low < 30:
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.3g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    elif len(positive):
        ax.set_ylim(0, float(positive.max()) * 1.2)
    if not row["finite_pairs"]:
        ax.text(0.5, 0.5, "No finite pairs", transform=ax.transAxes, ha="center")
    ratio = np.array([np.nan if v is None else v for v in row["bin_count_ratio"]])
    centers = (edges[:-1] + edges[1:]) / 2
    ratio_ax.axhline(1, color=ORIGINAL_COLOR, ls="--", lw=0.9)
    if ratio_ylim is None:
        ratio_ax.plot(centers, ratio, ".", color=DECODED_COLOR, ms=3)
        finite = ratio[np.isfinite(ratio)]
        low, high = (
            (min(0.8, float(finite.min())), max(1.2, float(finite.max())))
            if len(finite)
            else (0.8, 1.2)
        )
        pad = 0.1 * (high - low)
        ratio_ax.set_ylim(max(0, low - pad), high + pad)
        ratio_ax.set_yticks(sorted(set([round(max(0, low), 2), 1.0, round(high, 2)])))
    else:
        finite = np.isfinite(ratio)
        inside = finite & (ratio >= ratio_low) & (ratio <= ratio_high)
        ratio_ax.plot(centers[inside], ratio[inside], ".", color=DECODED_COLOR, ms=3)
        # Arrows retain each bin's x position without altering the saved ratio values.
        for outside, head, tail in (
            (finite & (ratio > ratio_high), 0.96, 0.66),
            (finite & (ratio < ratio_low), 0.04, 0.34),
        ):
            for center in centers[outside]:
                ratio_ax.annotate(
                    "",
                    xy=(center, head),
                    xytext=(center, tail),
                    xycoords=ratio_ax.get_xaxis_transform(),
                    textcoords=ratio_ax.get_xaxis_transform(),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": DECODED_COLOR,
                        "lw": 0.9,
                        "mutation_scale": 6,
                        "shrinkA": 0,
                        "shrinkB": 0,
                    },
                )
        ratio_ax.set_ylim(ratio_low, ratio_high)
        ratio_ax.set_yticks([ratio_low, 1.0, ratio_high])
    ratio_ax.set_ylabel(ratio_label, fontsize=8)
    ratio_ax.set_xlabel(axis_label(feature))
    ratio_ax.set_xlim(edges[0], edges[-1])
    for panel in (ax, ratio_ax):
        style_axis(panel)
    return ax, ratio_ax


def render_pt(
    summaries,
    sample,
    directory,
    *,
    log_y=True,
    quantizers=8,
    filename=None,
    prediction_label=None,
    ratio_label=None,
    ratio_ylim=None,
):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 8.2))
    grid = fig.add_gridspec(
        2, 3, left=0.075, right=0.985, bottom=0.08, top=0.93, wspace=0.35, hspace=0.4
    )
    for i, (obj, label) in enumerate(zip(OBJECTS, OBJECT_LABELS)):
        feature = next(f for f in summaries[obj]["features"] if f["name"].lower() == "pt")
        ax, _ = draw_distribution(
            fig,
            grid[i // 3, i % 3],
            feature,
            sample,
            log_y=log_y,
            prediction_label=prediction_label or f"Decoded Q{quantizers}",
            ratio_label=ratio_label or f"Decoded Q{quantizers} /\nOriginal",
            ratio_ylim=ratio_ylim,
        )
        ax.set_title(f"({chr(97+i)}) {label}", loc="left")
        row = feature["domains"][sample]
        width = row.get("response_width_percent")
        bias = row.get("median_relative_residual_percent")
        if width is not None:
            ax.text(
                0.96,
                0.94,
                rf"$R_{{p_T}}={width:.3g}\%$" + "\n" + rf"Median $\delta p_T/p_T={bias:.3g}\%$",
                transform=ax.transAxes,
                va="top",
                ha="right",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
            )
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
    fig.text(0.075, 0.975, DOMAINS[sample], ha="left", va="top", fontsize=10)
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.65, 0.99), ncol=2, frameon=False
    )
    save_figure(
        fig, directory / (filename or f"figure3_pt_{sample}_{'log' if log_y else 'linear'}")
    )
    return fig


def plot_pt_only(args, plan, summaries, destination):
    """Render only the six-object pT grid from sealed summaries, without array reads."""
    import matplotlib.pyplot as plt

    if set(summaries) != set(OBJECTS):
        raise ValueError("The 2 x 3 pT grid requires all six object summaries")
    samples = getattr(args, "samples", list(DOMAINS))
    runs = {run["object"]: run for run in plan["runs"]}
    for obj in OBJECTS:
        run = runs[obj]
        if (run["q"], run["k"], run["d"]) != capacity.SELECTED_CONFIGURATIONS[obj]:
            raise ValueError(f"Wrong selected full-Q8 tokenizer: {obj}")
        features = [f for f in summaries[obj]["features"] if f["name"].lower() == "pt"]
        if len(features) != 1 or features[0]["unit"] != "GeV":
            raise ValueError(f"Need one pT summary in GeV: {obj}")
        for sample in samples:
            if features[0]["domains"].get(sample, {}).get("finite_pairs", 0) <= 0:
                raise ValueError(
                    f"No finite pT pairs for {obj}/{sample}; refusing empty paper panel"
                )
    sources = [capacity.file_record(args.output_dir / "plan.json", content_hash=True)]
    sources.extend(
        capacity.file_record(args.output_dir / "summaries" / f"{obj}.json", content_hash=True)
        for obj in OBJECTS
    )
    with paper_style():
        for sample in samples:
            plt.close(render_pt(summaries, sample, destination, log_y=args.y_scale == "log"))
            stem = destination / f"figure3_pt_{sample}_{args.y_scale}"
            rows = []
            for obj in OBJECTS:
                feature = next(f for f in summaries[obj]["features"] if f["name"].lower() == "pt")
                rows.append(
                    {
                        "object": obj,
                        "sample": sample,
                        "unit": "GeV",
                        **{k: runs[obj][k] for k in ("q", "k", "d")},
                        **{
                            k: v
                            for k, v in feature["domains"][sample].items()
                            if not isinstance(v, list)
                        },
                    }
                )
            with stem.with_suffix(".csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            stem.with_suffix(".md").write_text(
                f"# Object pT reconstruction: {DOMAINS[sample]}\n\n"
                "A 2 x 3 grid, in order: jets, electrons, muons, photons, taus and tracks. "
                "Original detector-reconstructed pT (grey) and reconstructed/decoded full-Q8 "
                "pT (blue) are overlaid using identical bins. Original does not mean generator truth. "
                "Selected full models: q8/dim8/cb2048 for jets, electrons, muons and photons; "
                f"q8/dim8/cb4096 for taus and tracks; {plan['checkpoint_name']}. "
                "Saved preprocessing, masks, caps and paired objects are unchanged. "
                f"{plan['holdout_note']}\n\n"
                "Histograms are normalized by all finite paired objects, including objects "
                "outside the displayed window. Original and decoded retain the common edges "
                f"from the original MC/data {plan['statistics']['percentiles']} percentiles. "
                "The lower panel shows decoded/original bin counts with a dashed unity line; "
                f"bins with fewer than {plan['statistics']['min_ratio_count']} original entries "
                "are omitted. It is not a per-object response ratio. All x axes are in GeV. "
                "R_pT = 100 IQR(r)/median(r), with r = decoded/original pT, original pT > 1e-8 GeV "
                "and decoded pT > 0. The bias annotation is 100 median(r-1). These are integrated "
                "sample statistics, not binned scan widths. Exclusions and out-of-window counts "
                "are reported in the CSV. No uncertainty bands are inferred. "
                "Only sealed summaries are read for plotting; large caches are not re-audited.\n"
            )
            write_json(
                stem.with_suffix(".json"),
                seal(
                    {
                        "plan_id": plan["plan_id"],
                        "sources": sources,
                        "panel_order": list(OBJECTS),
                        "sample": sample,
                        "y_scale": args.y_scale,
                    },
                    "figure_id",
                ),
            )
            log.info(
                "Saved pT-only grid: %s; summaries only, no inference", stem.with_suffix(".png")
            )


def render_triptych_page(
    features,
    obj,
    sample,
    page,
    pages,
    *,
    log_y=True,
    title=None,
    count_label="objects",
    reference_label="Original",
    prediction_label="Decoded Q8",
    ratio_label="Decoded Q8 /\nOriginal",
    residual_label="Decoded - original\n",
    reference_axis="Original ",
    prediction_axis="Decoded ",
    header=None,
    show_absolute_error=False,
):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fig = plt.figure(figsize=(12, 3.1 * len(features) + 0.8))
    grid = fig.add_gridspec(
        len(features),
        3,
        left=0.075,
        right=0.93,
        bottom=0.75 / fig.get_figheight(),
        top=1 - 0.55 / fig.get_figheight(),
        wspace=0.43,
        hspace=0.55,
    )
    for i, feature in enumerate(features):
        ax, _ = draw_distribution(
            fig,
            grid[i, 0],
            feature,
            sample,
            log_y=log_y,
            count_label=count_label,
            reference_label=reference_label,
            prediction_label=prediction_label,
            ratio_label=ratio_label,
        )
        ax.set_title(feature["label"], loc="left")
        if i == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper right")
        row = feature["domains"][sample]
        res = fig.add_subplot(grid[i, 1])
        res.stairs(
            np.asarray(row["residual_counts"]) / max(row["finite_pairs"], 1),
            feature["residual_edges"],
            color=DECODED_COLOR,
            fill=True,
            alpha=0.7,
        )
        res.axvline(0, color=ORIGINAL_COLOR, ls="--", lw=0.9)
        res.set_xlabel(axis_label(feature, residual_label))
        res.set_ylabel(f"Normalized {count_label}")
        res.set_title("Residual", loc="left")
        if row["median_residual"] is not None:
            annotation = f"Median: {row['median_residual']:.3g}\nIQR: {row['residual_iqr']:.3g}"
            if show_absolute_error:
                annotation += f"\nMedian |residual|: {row['median_absolute_residual']:.3g}"
            res.text(
                0.96,
                0.94,
                annotation,
                transform=res.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
            )
        else:
            res.text(0.5, 0.5, "No finite pairs", transform=res.transAxes, ha="center")
        style_axis(res)
        pair_ax = fig.add_subplot(grid[i, 2])
        edges = np.asarray(feature["edges"])
        counts = np.asarray(row["paired_counts"])
        if counts.max() > 0:
            mesh = pair_ax.pcolormesh(
                edges,
                edges,
                np.ma.masked_equal(counts.T, 0),
                cmap="Blues",
                norm=LogNorm(vmin=1, vmax=max(2, counts.max())),
                rasterized=True,
            )
            colorbar = fig.colorbar(mesh, ax=pair_ax, pad=0.025, fraction=0.045)
            colorbar.set_label(f"{count_label.capitalize()} / bin", fontsize=8)
            colorbar.ax.tick_params(labelsize=8)
        else:
            pair_ax.text(0.5, 0.5, "No pairs in range", transform=pair_ax.transAxes, ha="center")
        pair_ax.plot(edges[[0, -1]], edges[[0, -1]], color=ORIGINAL_COLOR, ls="--", lw=0.9)
        pair_ax.set(xlim=(edges[0], edges[-1]), ylim=(edges[0], edges[-1]), aspect="equal")
        pair_ax.set_xlabel(axis_label(feature, reference_axis))
        pair_ax.set_ylabel(axis_label(feature, prediction_axis))
        pair_ax.set_title("Paired density", loc="left")
        style_axis(pair_ax)
    label = title if title is not None else dict(zip(OBJECTS, OBJECT_LABELS))[obj]
    fig.text(
        0.075,
        1 - 0.12 / fig.get_figheight(),
        header if header is not None else f"{label} | {DOMAINS[sample]} | full Q8",
        ha="left",
        va="top",
        fontsize=11,
    )
    fig.text(
        0.97, 1 - 0.12 / fig.get_figheight(), f"{page}/{pages}", ha="right", va="top", fontsize=9
    )
    return fig


def plot(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    plan = load_plan(args.output_dir)
    summaries = {obj: load_summary(args.output_dir, plan, obj) for obj in args.objects}
    destination = args.output_dir / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    if getattr(args, "pt_only", False):
        plot_pt_only(args, plan, summaries, destination)
        return
    mode = args.y_scale
    rows = []
    with paper_style():
        for sample in getattr(args, "samples", list(DOMAINS)):
            if set(summaries) == set(OBJECTS):
                plt.close(render_pt(summaries, sample, destination, log_y=mode == "log"))
            for obj, summary in summaries.items():
                features = summary["features"]
                pages = [
                    features[i : i + args.features_per_page]
                    for i in range(0, len(features), args.features_per_page)
                ]
                stem = f"full_q8_{obj}_triptychs_{sample}_{mode}"
                with PdfPages(destination / f"{stem}.pdf") as pdf:
                    for i, features_page in enumerate(pages, 1):
                        fig = render_triptych_page(
                            features_page, obj, sample, i, len(pages), log_y=mode == "log"
                        )
                        pdf.savefig(fig, facecolor="white")
                        fig.savefig(destination / f"{stem}_page{i:02d}.png", facecolor="white")
                        plt.close(fig)
                for f in features:
                    r = f["domains"][sample]
                    keys = (
                        "total_objects",
                        "finite_pairs",
                        "nonfinite_pairs",
                        "outside_original",
                        "outside_decoded",
                        "outside_residual",
                        "outside_paired",
                        "median_residual",
                        "median_absolute_residual",
                        "residual_iqr",
                        "response_pairs",
                        "excluded_response_pairs",
                        "response_width_percent",
                        "median_relative_residual_percent",
                    )
                    rows.append(
                        {
                            "object": obj,
                            "sample": sample,
                            "feature": f["name"],
                            "unit": f["unit"],
                            **{k: r.get(k) for k in keys},
                        }
                    )
    selection = "all" if set(args.objects) == set(OBJECTS) else "_".join(args.objects)
    with (destination / f"reconstruction_{selection}_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    caption = (
        "# Full Q8 reconstruction\n\n"
        "Selected full trainings: q8/dim8/cb2048 for jets, electrons, muons and photons; "
        "q8/dim8/cb4096 for taus and tracks. "
        f"Exact checkpoint: {plan['checkpoint_name']}. {plan['holdout_note']} "
        "Original refers to detector-reconstructed inputs, not generator-level truth. "
        "Each feature uses the same paired objects in the overlay, residual and paired-density panels. "
        "Nonfinite pairs are excluded from all three and counted in the CSV. "
        "Histograms are divided by the total finite paired-object count, including objects outside "
        "the displayed range; not independently renormalized within each displayed window. "
        "The ratio is decoded/original bin counts, not a per-object response. "
        f"Ratios with fewer than {plan['statistics']['min_ratio_count']} original entries are omitted. "
        "Common edges for both domains use the combined original 0.5--99.5 percentiles "
        "(min/max or constant-range fallback; small integer ranges use unit-width bins). "
        "Residual edges use the pooled residual 0.5--99.5 percentiles. Out-of-range counts are in the CSV. "
        "Statistics use all finite pairs, not only displayed bins. Paired density uses all paired "
        "objects without subsampling; both axes have identical limits and a y=x reference. "
        "Residuals are decoded minus original in the displayed units, including unwrapped phi "
        "to retain the existing diagnostic convention. Decoded discrete features are not rounded; "
        "these panels do not measure classification accuracy. "
        "R_pT = 100 IQR(r)/median(r), r=pT_decoded/pT_original, using positive decoded pT and "
        "original pT > 1e-8 GeV; exclusions are counted. The Figure 3 annotation is integrated over "
        "this evaluation sample, unlike the binned capacity-scan width. "
        "Median relative residual is 100 median(r-1). No uncertainty bands or seed variation are inferred. "
        "MC/data object counts may differ. "
        f"Declared input momentum unit: {plan['input_momentum_unit']}; energy-like features are shown in GeV. "
        "Other features retain their native units. No magnitude-based unit guessing is applied.\n"
    )
    (destination / f"reconstruction_{selection}_caption.md").write_text(caption)
    log.info("Saved full-training distributions and triptychs to %s; no inference", destination)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser(
        "plan", help="Verify full runs and unused evaluation files; no inference"
    )
    audit.add_argument(
        "--run-base", type=Path, default=Path("results/atlas_object_final_tokenizers_new_mcdata")
    )
    audit.add_argument("--checkpoint-name", choices=("best.ckpt", "last.ckpt"), required=True)
    audit.add_argument(
        "--evaluation-protocol", choices=("file-holdout", "saved-test"), default="file-holdout"
    )
    audit.add_argument(
        "--acknowledge-preprocessing-overlap",
        action="store_true",
        help="Required for saved-test: these events are not preprocessing-disjoint",
    )
    audit.add_argument(
        "--sample-plan",
        type=Path,
        help="Reuse Figure 2 files/settings; file-holdout checks disjointness, saved-test filters original test events",
    )
    audit.add_argument("--mc-files", type=Path, help="Explicit absolute H5 file list")
    audit.add_argument("--data-files", type=Path)
    audit.add_argument("--input-momentum-unit", choices=("GeV", "MeV"), required=True)
    audit.add_argument("--max-valid-objects", type=int, default=1_000_000)
    audit.add_argument("--batch-size", type=int, default=256)
    audit.add_argument("--n-bins", type=int, default=60)
    audit.add_argument("--min-ratio-count", type=int, default=20)
    evaluation = sub.add_parser(
        "evaluate", help="Cache paired full-model arrays; reuse verified caches"
    )
    evaluation.add_argument("--cache-only", action="store_true")
    evaluation.add_argument("--device", default="cpu")
    summary = sub.add_parser(
        "summarize", help="Compute all-feature histograms/statistics from caches; no inference"
    )
    plotting = sub.add_parser(
        "plot", help="Plot Figure 3 and feature triptychs from summaries; no inference"
    )
    plotting.add_argument("--features-per-page", type=int, default=3)
    plotting.add_argument("--y-scale", choices=("log", "linear"), default="log")
    plotting.add_argument(
        "--pt-only",
        action="store_true",
        help="Only the 2 x 3 pT overlay/ratio figure; skip feature triptychs",
    )
    plotting.add_argument("--samples", choices=tuple(DOMAINS), nargs="+", default=list(DOMAINS))
    for cmd in (audit, evaluation, summary, plotting):
        cmd.add_argument("--output-dir", type=Path, required=True)
    for cmd in (evaluation, summary, plotting):
        cmd.add_argument("--objects", choices=OBJECTS, nargs="+", default=list(OBJECTS))
    args = parser.parse_args(argv)
    if args.command == "plan":
        if args.evaluation_protocol == "saved-test" and not args.acknowledge_preprocessing_overlap:
            parser.error(
                "saved-test requires --acknowledge-preprocessing-overlap; not file-disjoint holdout"
            )
        if args.evaluation_protocol != "saved-test" and args.acknowledge_preprocessing_overlap:
            parser.error("Preprocessing-overlap acknowledgement only applies to saved-test")
        if bool(args.sample_plan) == bool(args.mc_files or args.data_files):
            parser.error("Choose either --sample-plan or both --mc-files and --data-files")
        if not args.sample_plan and not (args.mc_files and args.data_files):
            parser.error("Both MC and data file lists are required")
        if any(
            getattr(args, key) <= 0
            for key in ("max_valid_objects", "batch_size", "n_bins", "min_ratio_count")
        ):
            parser.error("Counts and sizes must be positive")
    if args.command == "plot" and args.features_per_page <= 0:
        parser.error("--features-per-page must be positive")
    if args.command == "plot" and args.pt_only and set(args.objects) != set(OBJECTS):
        parser.error("--pt-only requires all six objects (omit --objects to include all)")
    return args


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    {"plan": build_plan, "evaluate": evaluate, "summarize": summarize, "plot": plot}[args.command](
        args
    )


if __name__ == "__main__":
    main()
