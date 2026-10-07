#!/usr/bin/env python3
"""Q1 reconstruction and codebook diagnostics on the audited Q8 samples.

Use independently trained full dim8/cb16384/q1 models, not a prefix of Q8.
Reuses the Figure 3 renderer and frozen test membership; never edits Q8 outputs.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import logging
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

import paper_tokenizer_reconstruction as reco
from paper_plot_style import OBJECTS, OBJECT_LABELS, paper_style, save_figure, style_axis
from paper_tokenizer_codebooks import selected_run, stage_metrics

log = logging.getLogger(__name__)
VERSION = "paper-full-q1-pt-reconstruction-v1"
CONFIGURATION = (1, 16384, 8)
DIAGNOSTIC_VERSION = "paper-full-q1-codebook-diagnostic-v1"
FEATURE_VERSION = "paper-full-q1-all-feature-reconstruction-v1"


def input_profile(run):
    reco.capacity.verify_file(run["config"])
    cfg = OmegaConf.load(run["config"]["path"])
    dm = OmegaConf.to_container(cfg.datamodule, resolve=True)
    profile = {
        key: dm.get(key)
        for key in (
            "object_type",
            "output_mode",
            "object_collections",
            "max_objects",
            "num_objects",
            "mask_input",
            "event_inputs",
            "label_key",
            "transforms",
        )
    }
    # Paths may differ, but the exact fitted transform and its other options must agree.
    profile["transforms"]["preprocess"]["cst_fn"]["filename"] = run["preprocessor"]["sha256"]
    return profile


def check_sample_compatibility(run, reference_run, reference):
    if run["feature_names"] != reference_run["feature_names"]:
        raise ValueError(f"Q1/Q8 feature order differs: {run['object']}")
    if input_profile(run) != input_profile(reference_run):
        raise ValueError(f"Q1/Q8 input preparation or saved joblib differs: {run['object']}")
    protocol = reference.get("evaluation_protocol", "file-holdout")
    if protocol == "saved-test":
        from paper_tokenizer_test_split import saved_settings

        _, current = saved_settings(run)
        _, previous = saved_settings(reference_run)
        if current != previous or current != reference["test_partition"]["settings"]:
            raise ValueError(
                f"Q1/Q8 saved training split differs: {run['object']}; cannot reuse membership"
            )
    elif protocol == "file-holdout":
        reco.verify_holdout([run], reference["samples"])
    else:
        raise ValueError(f"Unsupported reference evaluation protocol: {protocol}")


def build_plan(args):
    if args.output_dir.resolve() == args.reference_dir.resolve():
        raise ValueError("Q1 needs its own output directory; Q8 results must not be overwritten")
    destination = args.output_dir / "plan.json"
    if destination.exists():
        raise FileExistsError(f"Q1 audit already frozen: {destination}; use evaluate/plot")
    reference = reco.load_plan(args.reference_dir)
    if reference.get("evaluation_protocol") == "saved-test":
        from paper_tokenizer_test_split import verify_partition

        verify_partition(reference)
    runs, sources = [], {}
    for obj in OBJECTS:
        reference_run = selected_run(reference, obj)
        run = reco.audit_run(args.run_base, obj, args.checkpoint_name, configuration=CONFIGURATION)
        check_sample_compatibility(run, reference_run, reference)
        path = reco.cache_path(args.reference_dir, reference, reference_run, "mc")
        receipt = reco.read_sealed(path.with_suffix(".json"), "receipt_id")
        if (
            receipt["plan_id"],
            receipt["object"],
            receipt["sample"],
            receipt["arrays"]["path"],
        ) != (
            reference["plan_id"],
            obj,
            "mc",
            str(path.resolve()),
        ):
            raise ValueError(f"Wrong original-object cache receipt: {path}")
        summary_path = args.reference_dir / "summaries" / f"{obj}.json"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Missing existing Q8 bin summary: {summary_path}. Run paper_tokenizer_reconstruction.py "
                f"summarize --output-dir {args.reference_dir} --objects {obj}; it reuses the original caches."
            )
        summary = reco.load_summary(args.reference_dir, reference, obj)
        if receipt["arrays"] not in summary["sources"]:
            raise ValueError(f"Q8 summary does not describe the chosen MC cache: {obj}")
        pt = next(f for f in summary["features"] if f["name"].lower() == "pt")
        if pt["unit"] != "GeV" or pt["domains"]["mc"]["finite_pairs"] <= 0:
            raise ValueError(f"Reference pT panel is empty or not in GeV: {obj}")
        sources[obj] = {
            "run": reference_run,
            "cache_receipt": receipt,
            "summary": reco.capacity.file_record(summary_path, content_hash=True),
            "pt": {key: deepcopy(value) for key, value in pt.items() if key != "domains"},
        }
        runs.append(run)
        log.info(
            "%s: %s; checkpoint=%s; joblib=%s",
            obj,
            run["label"],
            run["checkpoint"]["path"],
            run["preprocessor"]["path"],
        )
    plan = reco.seal(
        {
            "version": VERSION,
            "runs": runs,
            "sources": sources,
            "reference_plan": reco.capacity.file_record(
                args.reference_dir / "plan.json", content_hash=True
            ),
            "reference_plan_id": reference["plan_id"],
            "samples": {"mc": reference["samples"]["mc"]},
            "evaluation": reference["evaluation"],
            "evaluation_protocol": reference.get("evaluation_protocol", "file-holdout"),
            "test_partition": reference.get("test_partition"),
            "input_momentum_unit": reference["input_momentum_unit"],
            "statistics": reference["statistics"],
            "checkpoint_name": args.checkpoint_name,
            "holdout_note": reference["holdout_note"],
        },
        "plan_id",
    )
    reco.write_json(destination, plan)
    log.info("Frozen Q1-only simulation audit: %s; no inference", destination)


def load_plan(directory):
    plan = reco.read_sealed(directory / "plan.json", "plan_id")
    if plan["version"] != VERSION or {r["object"] for r in plan["runs"]} != set(OBJECTS):
        raise ValueError("Expected a six-object Q1 simulation plan")
    if len(plan["runs"]) != 6 or any(
        (r["q"], r["k"], r["d"]) != CONFIGURATION for r in plan["runs"]
    ):
        raise ValueError("Q1 plan contains a different tokenizer capacity")
    return plan


def load_summary(directory, plan, obj):
    value = reco.read_sealed(directory / "summaries" / f"{obj}.json", "summary_id")
    if value["plan_id"] != plan["plan_id"] or value["object"] != obj:
        raise ValueError(f"Wrong Q1 summary provenance: {obj}")
    return value


def validate_arrays(arrays, original, run):
    if list(arrays["feature_names"]) != run["feature_names"]:
        raise ValueError("Q1 inference feature order differs from audit")
    if not np.array_equal(arrays["original"], original, equal_nan=True):
        raise ValueError(f"Q1/Q8 original objects differ for {run['object']}; no summary written")
    indices = np.asarray(arrays["indices"])
    if indices.shape != (len(original), 1) or not np.all(
        np.isfinite(indices) & (indices >= 0) & (indices < 16384) & (indices == np.floor(indices))
    ):
        raise ValueError("Expected one valid Q1 code per original object")
    if (
        str(arrays["checkpoint"]) != run["checkpoint"]["path"]
        or int(arrays["codebook_size"]) != 16384
    ):
        raise ValueError("Wrong Q1 checkpoint/codebook returned by inference")
    if arrays["reconstruction"].shape != original.shape:
        raise ValueError("Q1 decoded features have wrong shape")


def summarize_pt(arrays, original, plan, run):
    validate_arrays(arrays, original, run)
    feature = deepcopy(plan["sources"][run["object"]]["pt"])
    index = next(i for i, name in enumerate(run["feature_names"]) if name.lower() == "pt")
    feature["domains"] = {
        "mc": reco.feature_summary(
            original[:, index] * feature["scale"],
            arrays["reconstruction"][:, index] * feature["scale"],
            np.asarray(feature["edges"]),
            np.asarray(feature["residual_edges"]),
            plan["statistics"]["min_ratio_count"],
            "pt",
        )
    }
    return feature


def codebook_diagnostic(arrays, feature, run, min_count):
    """Assignment statistics and pT support in the exact bins used in the figure."""
    k = run["k"]
    metrics = stage_metrics(arrays["indices"], k, quantizers=1)[0]
    codes = np.asarray(arrays["indices"][:, 0], dtype=np.int64)
    counts = np.bincount(codes, minlength=k)
    index = next(i for i, name in enumerate(run["feature_names"]) if name.lower() == "pt")
    x = np.asarray(arrays["original"][:, index] * feature["scale"], dtype=float)
    y = np.asarray(arrays["reconstruction"][:, index] * feature["scale"], dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    edges = np.asarray(feature["edges"])
    n_bins = len(edges) - 1

    # Record observed per-code output spread instead of treating floating-point
    # variations in the same decoder output as distinct reconstruction prototypes.
    low, high = np.full(k, np.inf), np.full(k, -np.inf)
    finite_y = np.isfinite(y)
    np.minimum.at(low, codes[finite_y], y[finite_y])
    np.maximum.at(high, codes[finite_y], y[finite_y])
    observed = np.isfinite(low) & np.isfinite(high)
    spread = high[observed] - low[observed]
    tolerance = 1e-5 + 1e-4 * np.maximum(np.abs(low[observed]), np.abs(high[observed]))

    bin_id = np.searchsorted(edges, y[finite], side="right") - 1
    bin_id[y[finite] == edges[-1]] = n_bins - 1
    in_window = (bin_id >= 0) & (bin_id < n_bins)
    # Count distinct assigned codes per decoded bin and their concentration. A
    # code straddling a boundary due to numerical noise can occur in both bins.
    pairs, multiplicity = np.unique(
        codes[finite][in_window] * n_bins + bin_id[in_window], return_counts=True
    )
    support = np.bincount(pairs % n_bins, minlength=n_bins)
    dominant = np.zeros(n_bins, dtype=np.int64)
    np.maximum.at(dominant, pairs % n_bins, multiplicity)
    row = feature["domains"]["mc"]
    bins = []
    for b in range(n_bins):
        original_n, decoded_n = row["original_counts"][b], row["decoded_counts"][b]
        bins.append(
            {
                "pt_low_gev": float(edges[b]),
                "pt_high_gev": float(edges[b + 1]),
                "original_objects": original_n,
                "decoded_objects": decoded_n,
                "assigned_codes_in_decoded_bin": int(support[b]),
                "largest_code_share": float(dominant[b] / decoded_n) if decoded_n else None,
                "decoded_over_original": float(decoded_n / original_n) if original_n else None,
                "original_meets_ratio_threshold": original_n >= min_count,
                "empty_decoded_with_populated_original": original_n >= min_count and decoded_n == 0,
            }
        )
    ordered = np.sort(counts)[::-1]
    metrics.update(
        rare_max_assignments=5,
        rare_used_codes=int(np.count_nonzero((counts > 0) & (counts <= 5))),
        top_code_assignment_percent=float(100 * ordered[0] / len(codes)),
        top_10_codes_assignment_percent=float(100 * ordered[:10].sum() / len(codes)),
        codes_with_finite_decoded_pt=int(observed.sum()),
        nonfinite_decoded_pt_objects=int((~finite_y).sum()),
        max_within_code_pt_spread_gev=float(spread.max()) if len(spread) else None,
        codes_exceeding_pt_spread_tolerance=int(np.count_nonzero(spread > tolerance)),
        spread_tolerance_absolute_gev=1e-5,
        spread_tolerance_relative=1e-4,
        plotted_original_objects=int(sum(row["original_counts"])),
        plotted_decoded_objects=int(sum(row["decoded_counts"])),
        original_bin_count_threshold=min_count,
        populated_original_bins=sum(b["original_meets_ratio_threshold"] for b in bins),
        empty_decoded_with_populated_original_bins=sum(
            b["empty_decoded_with_populated_original"] for b in bins
        ),
    )
    return {
        "metrics": metrics,
        "assignment_counts": counts.tolist(),
        "decoded_pt_min_gev_by_code": [float(v) if np.isfinite(v) else None for v in low],
        "decoded_pt_max_gev_by_code": [float(v) if np.isfinite(v) else None for v in high],
        "bins": bins,
    }


def save_diagnostic(directory, plan, run, arrays, feature, summary):
    value = reco.seal(
        {
            "version": DIAGNOSTIC_VERSION,
            "plan_id": plan["plan_id"],
            "object": run["object"],
            "summary_id": summary["summary_id"],
            "sample": "mc",
            "configuration": list(CONFIGURATION),
            "histogram_counts_match_figure": True,
            "all_original_features_identical_to_q8": True,
            "note": (
                "Used means assigned at least once in this frozen evaluation sample, not "
                "throughout training. Unused codes may be active in other samples. Coverage "
                "uses only assigned codes, not a decode of all codebook entries. Bin counts "
                "use the same finite-pair selection as the figure; assignment counts include "
                "all evaluated objects. No automatic diagnosis of codebook collapse."
            ),
            **codebook_diagnostic(arrays, feature, run, plan["statistics"]["min_ratio_count"]),
        },
        "diagnostic_id",
    )
    path = directory / "codebook_summaries" / f"{run['object']}.json"
    reco.write_json(path, value)
    return value


def report_diagnostic(directory, value):
    obj, m = value["object"], value["metrics"]
    destination = directory / "codebook_summaries" / f"{obj}_pt_bins.csv"
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(value["bins"][0]))
        writer.writeheader()
        writer.writerows(value["bins"])
    print(
        f"{obj}: N={m['assignments']:,}; used={m['used_codes']:,}/16384 "
        f"({m['used_percent']:.2f}%); perplexity={m['perplexity']:.1f} "
        f"(P/K={m['normalized_perplexity']:.4f}); "
        f"top 10 codes={m['top_10_codes_assignment_percent']:.2f}% of objects; "
        f"empty decoded bins={m['empty_decoded_with_populated_original_bins']}/"
        f"{m['populated_original_bins']} original bins with >= "
        f"{m['original_bin_count_threshold']} objects",
        flush=True,
    )
    if m["codes_exceeding_pt_spread_tolerance"]:
        log.warning(
            "%s: %s codes have appreciable within-code decoded pT variation; "
            "check the decoder before interpreting these as fixed prototypes",
            obj,
            m["codes_exceeding_pt_spread_tolerance"],
        )
    log.info("Saved observed code counts/coverage: %s", destination)


def evaluate(args, *, diagnostics=False):
    plan = load_plan(args.output_dir)
    reco.capacity.verify_file(plan["reference_plan"])
    reference_dir = Path(plan["reference_plan"]["path"]).parent
    reference = reco.load_plan(reference_dir)
    if reference["plan_id"] != plan["reference_plan_id"]:
        raise ValueError("Reference Q8 audit changed")
    if plan["evaluation_protocol"] == "saved-test":
        from paper_tokenizer_test_split import verify_partition

        verify_partition(plan)
    for record in plan["samples"]["mc"]:
        reco.capacity.verify_file(record)
    for run in plan["runs"]:
        obj = run["object"]
        if obj not in args.objects:
            continue
        source = plan["sources"][obj]
        for key in ("config", "completion", "checkpoint", "preprocessor", "preprocessor_metadata"):
            reco.capacity.verify_file(run[key])
        reco.capacity.verify_file(source["summary"])
        path = Path(source["cache_receipt"]["arrays"]["path"])
        receipt = reco.read_sealed(path.with_suffix(".json"), "receipt_id")
        if receipt != source["cache_receipt"]:
            raise ValueError(f"Reference cache receipt changed: {obj}")
        destination = args.output_dir / "summaries" / f"{obj}.json"
        summary = None
        if destination.exists():
            reco.capacity.verify_file(receipt["arrays"])
            summary = load_summary(args.output_dir, plan, obj)
            if not diagnostics:
                log.info("Reusing verified Q1 pT summary: %s", destination)
                continue
        diagnostic_path = args.output_dir / "codebook_summaries" / f"{obj}.json"
        if diagnostics and diagnostic_path.exists():
            value = reco.read_sealed(diagnostic_path, "diagnostic_id")
            if (
                summary is None
                or value["version"] != DIAGNOSTIC_VERSION
                or value["plan_id"] != plan["plan_id"]
                or value["object"] != obj
                or value["summary_id"] != summary["summary_id"]
            ):
                raise ValueError(f"Wrong Q1 diagnostic provenance: {diagnostic_path}")
            report_diagnostic(args.output_dir, value)
            continue
        originals, _ = reco.read_cache(path, reference, source["run"], "mc")
        log.info("Evaluating %s: full Q1, Simulation, same frozen rows/cap as Q8", obj)
        arrays = reco.infer(run, plan, "mc", args.device)
        feature = summarize_pt(arrays, originals["original"], plan, run)
        if summary is not None:
            previous = summary["features"][0]
            if previous["edges"] != feature["edges"] or any(
                previous["domains"]["mc"][key] != feature["domains"]["mc"][key]
                for key in ("total_objects", "finite_pairs", "original_counts", "decoded_counts")
            ):
                raise ValueError(
                    f"Q1 rerun does not reproduce plotted histogram counts: {obj}; "
                    "no diagnostic saved or figure changed"
                )
        else:
            summary = reco.seal(
                {
                    "plan_id": plan["plan_id"],
                    "object": obj,
                    "features": [feature],
                    "matching": {
                        "all_original_features_identical_to_q8": True,
                        "objects": len(originals["original"]),
                    },
                    "source_receipt_id": receipt["receipt_id"],
                },
                "summary_id",
            )
            reco.write_json(destination, summary)
            log.info("Saved Q1 summary: %s; no duplicate large array cache", destination)
        value = save_diagnostic(args.output_dir, plan, run, arrays, feature, summary)
        report_diagnostic(args.output_dir, value)
        del arrays, originals


def diagnose(args):
    evaluate(args, diagnostics=True)


def feature_context(directory, plan):
    reco.capacity.verify_file(plan["q1_plan"])
    previous = load_plan(directory)
    if previous["plan_id"] != plan["q1_plan_id"]:
        raise ValueError("Existing Q1 pT audit changed")
    reco.capacity.verify_file(previous["reference_plan"])
    reference_dir = Path(previous["reference_plan"]["path"]).parent
    reference = reco.load_plan(reference_dir)
    if reference["plan_id"] != previous["reference_plan_id"]:
        raise ValueError("Reference Q8 audit changed")
    return previous, reference


def load_feature_plan(directory):
    plan = reco.read_sealed(directory / "feature_plan.json", "plan_id")
    if (
        plan["version"] != FEATURE_VERSION
        or set(plan["samples"]) != set(reco.DOMAINS)
        or len(plan["runs"]) != len(OBJECTS)
        or {r["object"] for r in plan["runs"]} != set(OBJECTS)
        or any((r["q"], r["k"], r["d"]) != CONFIGURATION for r in plan["runs"])
    ):
        raise ValueError("Expected the six-object, MC/data full-Q1 feature audit")
    return plan


def build_feature_plan(args):
    """Extend the existing pT audit without changing its plan or summaries."""
    destination = args.output_dir / "feature_plan.json"
    if destination.exists():
        plan = load_feature_plan(args.output_dir)
        feature_context(args.output_dir, plan)
        log.info("Reusing frozen Q1 all-feature audit: %s; no inference", destination)
        return
    previous = load_plan(args.output_dir)
    reco.capacity.verify_file(previous["reference_plan"])
    reference_dir = Path(previous["reference_plan"]["path"]).parent
    reference = reco.load_plan(reference_dir)
    if reference["plan_id"] != previous["reference_plan_id"]:
        raise ValueError("Reference Q8 audit changed")
    if reference.get("evaluation_protocol") == "saved-test":
        from paper_tokenizer_test_split import verify_partition

        verify_partition(reference)
    sources = {}
    for run in previous["runs"]:
        obj = run["object"]
        reference_run = previous["sources"][obj]["run"]
        check_sample_compatibility(run, reference_run, reference)
        for key in ("config", "completion", "checkpoint", "preprocessor", "preprocessor_metadata"):
            reco.capacity.verify_file(run[key])
        source_summary = previous["sources"][obj]["summary"]
        reco.capacity.verify_file(source_summary)
        summary = reco.load_summary(reference_dir, reference, obj, verify_sources=True)
        receipts = {}
        for sample in reco.DOMAINS:
            path = reco.cache_path(reference_dir, reference, reference_run, sample)
            receipt = reco.read_sealed(path.with_suffix(".json"), "receipt_id")
            if (
                receipt["plan_id"],
                receipt["object"],
                receipt["sample"],
                receipt["arrays"]["path"],
            ) != (reference["plan_id"], obj, sample, str(path.resolve())):
                raise ValueError(f"Wrong original-object cache receipt: {path}")
            if receipt["arrays"] not in summary["sources"]:
                raise ValueError(f"Q8 feature summary/cache disagree: {obj}/{sample}")
            if any(sample not in feature["domains"] for feature in summary["features"]):
                raise ValueError(f"Missing Q8 feature domain: {obj}/{sample}")
            receipts[sample] = receipt
        if receipts["mc"] != previous["sources"][obj]["cache_receipt"]:
            raise ValueError(f"Q1 pT and all-feature MC samples disagree: {obj}")
        sources[obj] = {
            "run": reference_run,
            "summary": source_summary,
            "cache_receipts": receipts,
            "features": [
                {key: deepcopy(value) for key, value in f.items() if key != "domains"}
                for f in summary["features"]
            ],
        }
    plan = reco.seal(
        {
            "version": FEATURE_VERSION,
            "q1_plan": reco.capacity.file_record(args.output_dir / "plan.json", content_hash=True),
            "q1_plan_id": previous["plan_id"],
            "runs": previous["runs"],
            "sources": sources,
            "samples": reference["samples"],
            **{
                key: previous[key]
                for key in (
                    "evaluation",
                    "evaluation_protocol",
                    "test_partition",
                    "input_momentum_unit",
                    "statistics",
                    "checkpoint_name",
                    "holdout_note",
                )
            },
        },
        "plan_id",
    )
    reco.write_json(destination, plan)
    log.info("Frozen Q1 all-feature MC/data audit: %s; no inference", destination)


def summarize_features(items, plan, run):
    """Use frozen original-feature bins, with Q1-specific pooled residual bins."""
    features = deepcopy(plan["sources"][run["object"]]["features"])
    if [f["name"] for f in features] != run["feature_names"]:
        raise ValueError(f"Q1/Q8 all-feature order differs: {run['object']}")
    settings = plan["statistics"]
    for i, feature in enumerate(features):
        pairs = {
            sample: (
                arrays["original"][:, i] * feature["scale"],
                arrays["reconstruction"][:, i] * feature["scale"],
            )
            for sample, arrays in items.items()
        }
        residuals = np.concatenate(
            [(y - x)[np.isfinite(x) & np.isfinite(y)] for x, y in pairs.values()]
        )
        residual_edges = reco.bin_edges(residuals, settings["n_bins"], settings["percentiles"])
        feature["residual_edges"] = residual_edges.tolist()
        feature["domains"] = {
            sample: reco.feature_summary(
                x,
                y,
                np.asarray(feature["edges"]),
                residual_edges,
                settings["min_ratio_count"],
                feature["name"],
            )
            for sample, (x, y) in pairs.items()
        }
    return features


def load_feature_summary(directory, plan, obj, *, verify_sources=False):
    path = directory / "feature_summaries" / f"{obj}.json"
    value = reco.read_sealed(path, "summary_id")
    run = next(r for r in plan["runs"] if r["object"] == obj)
    if (
        value["plan_id"] != plan["plan_id"]
        or value["object"] != obj
        or [f["name"] for f in value["features"]] != run["feature_names"]
        or any(set(f["domains"]) != set(reco.DOMAINS) for f in value["features"])
        or value["source_receipt_ids"]
        != {
            sample: receipt["receipt_id"]
            for sample, receipt in plan["sources"][obj]["cache_receipts"].items()
        }
    ):
        raise ValueError(f"Wrong Q1 all-feature summary provenance: {path}")
    if verify_sources:
        for receipt in plan["sources"][obj]["cache_receipts"].values():
            reco.capacity.verify_file(receipt["arrays"])
    return value


def evaluate_features(args):
    plan = load_feature_plan(args.output_dir)
    previous, reference = feature_context(args.output_dir, plan)
    if plan["evaluation_protocol"] == "saved-test":
        from paper_tokenizer_test_split import verify_partition

        verify_partition(plan)
    for records in plan["samples"].values():
        for record in records:
            reco.capacity.verify_file(record)
    for run in plan["runs"]:
        obj = run["object"]
        if obj not in args.objects:
            continue
        for key in ("config", "completion", "checkpoint", "preprocessor", "preprocessor_metadata"):
            reco.capacity.verify_file(run[key])
        source = plan["sources"][obj]
        reco.capacity.verify_file(source["summary"])
        destination = args.output_dir / "feature_summaries" / f"{obj}.json"
        if destination.exists():
            load_feature_summary(args.output_dir, plan, obj, verify_sources=True)
            log.info("Reusing verified Q1 all-feature summary: %s; no inference", destination)
            continue
        items = {}
        for sample in reco.DOMAINS:
            receipt = source["cache_receipts"][sample]
            path = Path(receipt["arrays"]["path"])
            originals, actual = reco.read_cache(path, reference, source["run"], sample)
            if actual != receipt:
                raise ValueError(f"Reference cache receipt changed: {obj}/{sample}")
            log.info("Evaluating full Q1 %s/%s: same frozen rows, cap and joblib", obj, sample)
            arrays = reco.infer(run, plan, sample, args.device)
            validate_arrays(arrays, originals["original"], run)
            if sample == "mc" and (args.output_dir / "summaries" / f"{obj}.json").exists():
                legacy = load_summary(args.output_dir, previous, obj)
                feature = summarize_pt(arrays, originals["original"], previous, run)
                row, old_row = feature["domains"]["mc"], legacy["features"][0]["domains"]["mc"]
                if any(
                    row[key] != old_row[key]
                    for key in (
                        "total_objects",
                        "finite_pairs",
                        "original_counts",
                        "decoded_counts",
                    )
                ):
                    raise ValueError(
                        f"Q1 all-feature rerun does not reproduce plotted pT counts: {obj}"
                    )
            items[sample] = arrays
            del originals
        summary = reco.seal(
            {
                "plan_id": plan["plan_id"],
                "object": obj,
                "source_receipt_ids": {
                    s: r["receipt_id"] for s, r in source["cache_receipts"].items()
                },
                "matching": {
                    s: {
                        "all_original_features_identical_to_q8": True,
                        "objects": len(a["original"]),
                    }
                    for s, a in items.items()
                },
                "features": summarize_features(items, plan, run),
            },
            "summary_id",
        )
        reco.write_json(destination, summary)
        del items
        log.info("Saved all Q1 features: %s; no duplicate bulk array cache", destination)


def plot_features(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    plan = load_feature_plan(args.output_dir)
    summaries = {obj: load_feature_summary(args.output_dir, plan, obj) for obj in args.objects}
    destination = args.output_dir / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    with paper_style():
        for obj, summary in summaries.items():
            features = summary["features"]
            page_size = args.features_per_page or len(features)
            pages = [features[i : i + page_size] for i in range(0, len(features), page_size)]
            for sample in args.samples:
                stem = f"full_q1_{obj}_triptychs_{sample}_{args.y_scale}"
                with PdfPages(destination / f"{stem}.pdf") as pdf:
                    for page, subset in enumerate(pages, 1):
                        fig = reco.render_triptych_page(
                            subset,
                            obj,
                            sample,
                            page,
                            len(pages),
                            log_y=args.y_scale == "log",
                            prediction_label="Reconstructed",
                            ratio_label="Reconstructed /\nOriginal",
                            residual_label="Reconstructed - original\n",
                            prediction_axis="Reconstructed ",
                            header=f"{dict(zip(OBJECTS, OBJECT_LABELS))[obj]} | {reco.DOMAINS[sample]}",
                            show_absolute_error=True,
                        )
                        pdf.savefig(fig, facecolor="white")
                        fig.savefig(destination / f"{stem}_page{page:02d}.png", facecolor="white")
                        plt.close(fig)
                reco.write_json(
                    destination / f"{stem}.json",
                    {
                        "plan_id": plan["plan_id"],
                        "summary_id": summary["summary_id"],
                        "object": obj,
                        "sample": sample,
                        "features_per_page": page_size,
                        "feature_names": [f["name"] for f in features],
                        "y_scale": args.y_scale,
                    },
                )
                for feature in features:
                    rows.append(
                        {
                            "object": obj,
                            "sample": sample,
                            "feature": feature["name"],
                            "unit": feature["unit"],
                            "q": 1,
                            "k": 16384,
                            "d": 8,
                            **{
                                k: v
                                for k, v in feature["domains"][sample].items()
                                if not isinstance(v, list)
                            },
                        }
                    )
                log.info("Saved %s.pdf and page PNGs; summaries only, no inference", stem)
    selection = "all" if set(args.objects) == set(OBJECTS) else "_".join(args.objects)
    name = f"q1_features_{selection}_{'_'.join(args.samples)}_{args.y_scale}"
    with (destination / f"{name}_metrics.csv").open("w", newline="") as handle:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (destination / f"{name}_caption.md").write_text(
        "# Full Q1 object-feature reconstruction\n\n"
        "Separately trained single-codebook VQ-VAEs (Q1, K=16,384, latent dimension 8), "
        "not Q8 prefixes or event-transformer predictions. "
        f"Exact checkpoint: {plan['checkpoint_name']}. Original means detector-reconstructed inputs, not truth. "
        "Separate simulation and collision-data figures show every saved feature: original/reconstructed distributions "
        "with bin-count ratios, reconstructed-minus-original residuals, and paired density with a y=x line. "
        "All original arrays match the corresponding frozen Q8 samples exactly, including object order. "
        "Original-feature edges are copied from the existing Q8 summaries; residual edges use the pooled Q1 "
        "MC/data residual percentiles, so the narrower Q8 residual range is not imposed on Q1. "
        f"Percentiles: {plan['statistics']['percentiles']}; minimum original ratio-bin count: "
        f"{plan['statistics']['min_ratio_count']}. Histograms normalize to all finite pairs, including "
        "out-of-window objects. Nonfinite and out-of-range counts are reported in the CSV. "
        "Residual annotations give signed median, IQR and median absolute residual, not mean absolute error. "
        "Phi residuals are unwrapped, matching the existing tokenizer triptychs; discrete decoded features are "
        "not rounded. Paired density uses all finite pairs with common x/y limits, without subsampling. "
        f"Input momentum unit: {plan['input_momentum_unit']}; energy-like features are displayed in GeV. "
        "No uncertainty bands are inferred. The earlier Q1 pT summaries and figures remain unchanged.\n\n"
        + plan["holdout_note"]
        + "\n"
    )


def load_codebook_plot_data(directory, plan, obj):
    path = directory / "codebook_summaries" / f"{obj}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}; run diagnose --objects {obj} first. Plotting never runs inference."
        )
    value = reco.read_sealed(path, "diagnostic_id")
    summary = load_summary(directory, plan, obj)
    if (
        value["version"] != DIAGNOSTIC_VERSION
        or value["plan_id"] != plan["plan_id"]
        or value["object"] != obj
        or value["summary_id"] != summary["summary_id"]
        or value["configuration"] != list(CONFIGURATION)
        or value["sample"] != "mc"
    ):
        raise ValueError(f"Wrong Q1 diagnostic provenance: {path}")
    counts = np.asarray(value["assignment_counts"])
    metrics = value["metrics"]
    if (
        counts.shape != (CONFIGURATION[1],)
        or counts.dtype.kind not in "iu"
        or np.any(counts < 0)
        or int(counts.sum()) != metrics["assignments"]
        or int(np.count_nonzero(counts)) != metrics["used_codes"]
    ):
        raise ValueError(f"Inconsistent Q1 assignment counts: {path}")
    feature = summary["features"][0]
    bins, edges = value["bins"], feature["edges"]
    if len(bins) != len(edges) - 1 or any(
        b["pt_low_gev"] != edges[i]
        or b["pt_high_gev"] != edges[i + 1]
        or b["original_objects"] != feature["domains"]["mc"]["original_counts"][i]
        or b["decoded_objects"] != feature["domains"]["mc"]["decoded_counts"][i]
        for i, b in enumerate(bins)
    ):
        raise ValueError(f"Q1 diagnostic bins do not match the pT figure: {path}")
    return value


def render_codebook_utilization(obj, metrics, *, reported=False):
    """Overview can also display explicitly reported aggregate diagnostic numbers."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    m = metrics
    k, used = CONFIGURATION[1], m["used_codes"]
    populated = m["populated_original_bins"]
    empty = m["empty_decoded_with_populated_original_bins"]
    blue, grey, green, orange = "#235789", "#C9CDD2", "#009E73", "#D55E00"
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.3))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.25, top=0.80, wspace=0.38)
    panels = (
        (
            [used, k - used],
            ["Used", "Unused"],
            [blue, grey],
            k,
            "(a) Codebook utilization",
            "Codebook entries",
        ),
        (
            [used, m["perplexity"]],
            ["Used codes", "Effective codes\n(perplexity)"],
            [blue, green],
            max(used, m["perplexity"], 1),
            "(b) Within the used subset",
            "Number of codes",
        ),
        (
            [populated - empty, empty],
            ["With decoded\nentries", "No decoded\nentries"],
            [blue, orange],
            max(populated, 1),
            r"(c) $p_T$ bin coverage",
            "Populated original bins",
        ),
    )
    for panel, (ax, (values, labels, colors, scale, title, ylabel)) in enumerate(zip(axes, panels)):
        ax.bar([0, 1], values, width=0.58, color=colors, edgecolor="#444444", linewidth=0.5)
        ax.set_xticks([0, 1], labels)
        ax.set_ylim(0, scale * 1.24)
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(title, loc="left", fontsize=10.5, pad=12)
        ax.set_ylabel(ylabel)
        style_axis(ax)
        ax.xaxis.set_tick_params(which="minor", bottom=False, top=False)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        for i, count in enumerate(values):
            text = f"{count:,.0f}" if panel != 1 or i == 0 else f"{count:,.1f}"
            if panel == 0:
                text += f"\n({100 * count / k:.2f}%)"
            elif panel == 2 and populated:
                text += f"\n({100 * count / populated:.1f}%)"
            ax.text(i, count + scale * 0.025, text, ha="center", va="bottom", fontsize=9)
    axes[1].text(
        0.98,
        0.97,
        f"P/K = {m['normalized_perplexity']:.4f}",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )
    label = OBJECT_LABELS[OBJECTS.index(obj)]
    source = " | Reported summary" if reported else ""
    fig.text(
        0.065, 0.945, f"{label} | Q1 | Simulation | N = {m['assignments']:,}{source}", fontsize=12
    )
    fig.text(
        0.065,
        0.085,
        "Unused means not assigned in this evaluation sample, not necessarily during training.",
        fontsize=9,
    )
    fig.text(
        0.065,
        0.035,
        f"Panel (c): {populated} bins with at least {m['original_bin_count_threshold']} original objects per bin.",
        fontsize=9,
    )
    return fig


def render_codebook_coverage(value):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    m, bins, obj = value["metrics"], value["bins"], value["object"]
    counts = np.sort(np.asarray(value["assignment_counts"]))[::-1]
    counts = counts[counts > 0]
    edges = np.array([b["pt_low_gev"] for b in bins] + [bins[-1]["pt_high_gev"]])
    centers = (edges[:-1] + edges[1:]) / 2
    empty = np.array([b["empty_decoded_with_populated_original"] for b in bins])
    fig = plt.figure(figsize=(11.8, 5.5))
    grid = fig.add_gridspec(1, 2, left=0.075, right=0.985, top=0.82, bottom=0.18, wspace=0.29)
    rank = fig.add_subplot(grid[0, 0])
    right = grid[0, 1].subgridspec(2, 1, height_ratios=[2.4, 1], hspace=0.06)
    hist = fig.add_subplot(right[0])
    support = fig.add_subplot(right[1], sharex=hist)
    rank.plot(
        np.arange(1, len(counts) + 1),
        100 * counts / m["assignments"],
        color="#235789",
        marker=".",
        markersize=3,
    )
    rank.set_yscale("log")
    rank.set_xlim(0.5, max(len(counts) + 0.5, 2))
    rank.set_xlabel("Used code rank (most frequent first)")
    rank.set_ylabel("Objects assigned to code [%]")
    rank.set_title("(a) Assignment frequency", loc="left", pad=12)
    rank.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    rank.text(
        0.03,
        0.04,
        f"{len(counts):,} used codes shown\n"
        f"Top 10 codes: {m['top_10_codes_assignment_percent']:.2f}% of objects",
        transform=rank.transAxes,
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
    )
    hist.stairs([b["original_objects"] for b in bins], edges, color="#555555", label="Original")
    hist.stairs([b["decoded_objects"] for b in bins], edges, color="#235789", label="Reconstructed")
    hist.set_yscale("log")
    hist.set_ylim(bottom=0.8)
    hist.set_ylabel("Objects / bin")
    hist.set_title(r"(b) $p_T$ counts and code coverage", loc="left", pad=12)
    hist.legend(frameon=False, loc="upper right")
    hist.tick_params(labelbottom=False)
    support.bar(
        centers,
        [b["assigned_codes_in_decoded_bin"] for b in bins],
        width=np.diff(edges) * 0.85,
        color="#235789",
    )
    support.scatter(
        centers[empty],
        np.zeros(int(empty.sum())),
        marker="x",
        s=22,
        color="#D55E00",
        zorder=5,
        clip_on=False,
    )
    support.set_ylim(bottom=0)
    support.set_xlim(edges[0], edges[-1])
    support.set_xlabel(r"$p_T$ [GeV]")
    support.set_ylabel("Assigned\ncodes / bin")
    support.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
    for ax in (rank, hist, support):
        style_axis(ax)
    label = OBJECT_LABELS[OBJECTS.index(obj)]
    fig.text(0.075, 0.945, f"{label} | Q1 | Simulation | N = {m['assignments']:,}", fontsize=12)
    fig.text(
        0.075,
        0.07,
        "Orange crosses: no decoded entries despite "
        f">= {m['original_bin_count_threshold']} original objects in that bin.",
        fontsize=9,
    )
    warning = "Observed assigned codes only; no inference or rebinning."
    if m["codes_exceeding_pt_spread_tolerance"]:
        warning += " Warning: fixed-code pT variation exceeds tolerance."
    fig.text(0.075, 0.025, warning, fontsize=9)
    return fig


def plot_codebooks(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plan = load_plan(args.output_dir)
    # Preflight every requested object before writing any figures. No model,
    # source H5, Q8 cache or checkpoint is opened by this summary-only command.
    values = {obj: load_codebook_plot_data(args.output_dir, plan, obj) for obj in args.objects}
    destination = args.output_dir / "figures"
    with paper_style():
        for obj, value in values.items():
            for suffix, fig in (
                ("utilization", render_codebook_utilization(obj, value["metrics"])),
                ("coverage", render_codebook_coverage(value)),
            ):
                stem = destination / f"q1_{obj}_codebook_{suffix}"
                save_figure(fig, stem)
                plt.close(fig)
                reco.write_json(
                    stem.with_suffix(".json"),
                    {
                        "plan_id": plan["plan_id"],
                        "diagnostic_id": value["diagnostic_id"],
                        "summary_id": value["summary_id"],
                        "object": obj,
                    },
                )
                log.info("Saved %s.png and .pdf; no inference", stem)


def plot(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plan = load_plan(args.output_dir)
    summaries = {obj: load_summary(args.output_dir, plan, obj) for obj in OBJECTS}
    for obj, summary in summaries.items():
        if (
            len(summary["features"]) != 1
            or summary["features"][0]["domains"]["mc"]["finite_pairs"] <= 0
        ):
            raise ValueError(f"Missing/nonfinite Q1 pT panel: {obj}")
    destination = args.output_dir / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    name = f"q1_pt_mc_{args.y_scale}"
    ratio_ylim = tuple(getattr(args, "ratio_ylim", (0.5, 1.5)))
    with paper_style():
        fig = reco.render_pt(
            summaries,
            "mc",
            destination,
            log_y=args.y_scale == "log",
            quantizers=1,
            filename=name,
            prediction_label="Reconstructed",
            ratio_label="Reconstructed /\nOriginal",
            ratio_ylim=ratio_ylim,
        )
        plt.close(fig)
    rows = [
        {
            "object": obj,
            "sample": "mc",
            "q": 1,
            "k": 16384,
            "d": 8,
            **{
                k: v
                for k, v in summaries[obj]["features"][0]["domains"]["mc"].items()
                if not isinstance(v, list)
            },
        }
        for obj in OBJECTS
    ]
    with (destination / f"{name}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    reco.write_json(
        destination / f"{name}.json",
        {
            "plan_id": plan["plan_id"],
            "summary_ids": {obj: summary["summary_id"] for obj, summary in summaries.items()},
            "y_scale": args.y_scale,
            "ratio_ylim": list(ratio_ylim),
            "out_of_range_ratio_marker": "up/down arrows at original bin centers",
        },
    )
    (destination / f"{name}.md").write_text(
        "# Q1 object pT reconstruction in simulation\n\n"
        r"Original and reconstructed $p_T$ distributions for the object-specific VQ-VAEs in simulation. "
        "The panels show jets, electrons, muons, photons, taus and tracks, using identical "
        "histogram bins for each original/reconstructed pair. The lower panels show the ratio "
        "of reconstructed to original histogram bin counts, with a dashed unity line. "
        f"Their displayed range is {ratio_ylim[0]:g}--{ratio_ylim[1]:g}; upward/downward arrows "
        "mark values above/below this range at their bin centers. "
        r"The annotations give the integrated object-by-object relative response width "
        r"$R_{p_T}=\mathrm{IQR}(r)/\mathrm{median}(r)$, with "
        r"$r=p_T^{\mathrm{reconstructed}}/p_T^{\mathrm{original}}$, "
        r"and the median relative residual $(p_T^{\mathrm{reconstructed}}-p_T^{\mathrm{original}})"
        r"/p_T^{\mathrm{original}}$, both displayed as percentages. "
        "IQR denotes the 75th minus the 25th percentile. The response width uses paired objects; "
        "it is not the histogram-bin ratio displayed in the lower panels.\n\n"
        "## Evaluation Details\n\n"
        "Original means detector-reconstructed input, not generator truth. Reconstructed values "
        "come from six independently trained single-codebook VQ-VAEs "
        "(Q1, K=16,384, latent dimension 8). "
        "Not Q0-only decoding of Q8 and not masked event-model prediction. "
        f"Exact checkpoints: {plan['checkpoint_name']}. "
        "The simulation rows, object cap and feature preprocessing match the existing Q8 audit; "
        "original arrays were checked for exact agreement in all features. Bin edges are copied "
        "from its frozen pT summaries. Q8 curves are not drawn.\n\n"
        "Histograms use the number of finite original/decoded pairs, including out-of-window "
        "objects, as denominator. Ratio bins below the saved minimum original count are omitted. "
        "The width annotation is 100 R_pT; bias = 100 median(r - 1). "
        "As in the reference figure, these use original "
        "pT > 1e-8 GeV and decoded pT > 0; exclusions are reported in the CSV. "
        "These are integrated metrics, not binned response widths. No uncertainty bands are inferred.\n\n"
        + plan["holdout_note"]
        + "\n"
    )
    log.info("Saved Q1-only six-panel figure: %s", destination / f"{name}.png")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser(
        "plan", help="Audit Q1 models against the existing Q8 sample; no inference"
    )
    audit.add_argument("--reference-dir", type=Path, required=True)
    audit.add_argument(
        "--run-base", type=Path, default=Path("results/atlas_object_final_tokenizers_new_mcdata")
    )
    audit.add_argument("--checkpoint-name", choices=("best.ckpt", "last.ckpt"), default="best.ckpt")
    evaluation = sub.add_parser(
        "evaluate", help="Run Q1 on the frozen simulation rows; save small pT summaries"
    )
    diagnostic = sub.add_parser(
        "diagnose",
        help="Check Q1 occupancy/pT coverage; one frozen-sample inference pass if missing",
    )
    for p in (evaluation, diagnostic):
        p.add_argument("--device", default="cpu")
        p.add_argument("--objects", choices=OBJECTS, nargs="+", default=list(OBJECTS))
    plotting = sub.add_parser(
        "plot", help="Draw the Q1-only figure from small summaries, no inference"
    )
    plotting.add_argument("--y-scale", choices=("log", "linear"), default="log")
    plotting.add_argument(
        "--ratio-ylim",
        type=float,
        nargs=2,
        default=(0.5, 1.5),
        metavar=("LOW", "HIGH"),
        help="Histogram-ratio display limits; out-of-range bins are marked with arrows",
    )
    codebook_plotting = sub.add_parser(
        "plot-codebooks", help="Plot saved Q1 usage/coverage, no inference"
    )
    codebook_plotting.add_argument("--objects", choices=OBJECTS, nargs="+", default=list(OBJECTS))
    feature_audit = sub.add_parser(
        "plan-features",
        help="Extend the existing Q1 pT audit to all MC/data features; no inference",
    )
    feature_evaluation = sub.add_parser(
        "evaluate-features", help="Evaluate the full Q1 tokenizers on frozen MC/data samples once"
    )
    feature_evaluation.add_argument("--device", default="cpu")
    feature_evaluation.add_argument("--objects", choices=OBJECTS, nargs="+", default=list(OBJECTS))
    feature_plotting = sub.add_parser(
        "plot-features", help="Draw all-feature Q1 triptychs from summaries; no inference"
    )
    feature_plotting.add_argument("--objects", choices=OBJECTS, nargs="+", default=list(OBJECTS))
    feature_plotting.add_argument(
        "--samples", choices=tuple(reco.DOMAINS), nargs="+", default=list(reco.DOMAINS)
    )
    feature_plotting.add_argument("--y-scale", choices=("log", "linear"), default="log")
    feature_plotting.add_argument(
        "--features-per-page",
        type=int,
        default=0,
        help="0: all features on one page per object/domain; positive values split pages",
    )
    for p in (
        audit,
        evaluation,
        diagnostic,
        plotting,
        codebook_plotting,
        feature_audit,
        feature_evaluation,
        feature_plotting,
    ):
        p.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "plot-features" and args.features_per_page < 0:
        parser.error("--features-per-page must be zero or positive")
    if args.command == "plot" and (
        not np.isfinite(args.ratio_ylim).all()
        or not 0 <= args.ratio_ylim[0] < 1 < args.ratio_ylim[1]
    ):
        parser.error("--ratio-ylim must be finite and satisfy 0 <= LOW < 1 < HIGH")
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    {
        "plan": build_plan,
        "evaluate": evaluate,
        "diagnose": diagnose,
        "plot": plot,
        "plot-codebooks": plot_codebooks,
        "plan-features": build_feature_plan,
        "evaluate-features": evaluate_features,
        "plot-features": plot_features,
    }[args.command](args)
