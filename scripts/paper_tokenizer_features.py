#!/usr/bin/env python3
"""Cache-only feature scan summaries and one object per page for the paper.

Uses the frozen capacity audit and its parent caches in place. There is no
inference path, checkpoint loading, new file selection, or preprocessor fitting.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from contextlib import ExitStack
from pathlib import Path

import numpy as np

import paper_tokenizer_capacity as capacity
from paper_plot_style import (
    OBJECTS,
    OBJECT_LABELS,
    PT_LABEL,
    RESPONSE_LABEL,
    capacity_curve_style,
    capacity_legends,
    paper_style,
    save_figure,
    style_axis,
)

log = logging.getLogger(__name__)
METRICS_VERSION = "paper-all-features-v1"
CATEGORICAL_FEATURES = {
    "charge",
    "lhloose",
    "lhmedium",
    "lhtight",
    "isloose",
    "ismedium",
    "istight",
    "nndecaymode",
}


def cache_owners(source_dir, plan):
    owners = {(r["object"], r["label"]): (source_dir, plan) for r in plan["runs"]}
    if "parent" in plan:
        record = plan["parent"]["plan"]
        capacity.verify_file(record)
        directory = Path(record["path"]).parent
        parent = capacity.load_plan(directory)
        if parent["plan_id"] != plan["parent"]["plan_id"] or any(
            parent[key] != plan[key] for key in ("samples", "evaluation", "checkpoint_name")
        ):
            raise ValueError("Parent evaluation does not match the frozen feature audit")
        current_runs = {(r["object"], r["label"]): r for r in plan["runs"]}
        for run in parent["runs"]:
            key = (run["object"], run["label"])
            if current_runs.get(key) != run:
                raise ValueError(f"Inherited run changed: {key}")
        owners.update(cache_owners(directory, parent))
    return owners


def resolve_cache(scan, directory, plan, run, sample):
    for key in ("config", "checkpoint", "preprocessor", "preprocessor_metadata"):
        capacity.verify_file(run[key])
    candidates = scan.cache_path_candidates(
        output_dir=directory / "cache" / plan["plan_id"] / run["object"],
        object_name=run["object"],
        sample_label=sample,
        model_label=run["label"],
        run_dir=Path(run["run_dir"]),
        checkpoint=Path(run["checkpoint"]["path"]),
        files=[r["path"] for r in plan["samples"][sample]],
        preprocessor=Path(run["preprocessor"]["path"]),
        args=argparse.Namespace(**plan["evaluation"]),
    )
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            f"Missing audited cache for {run['object']}/{sample}/{run['label']}: "
            f"{candidates[0]}. No inference or legacy-cache fallback is allowed."
        )
    return path


def read_arrays(path, run):
    with np.load(path, allow_pickle=False) as cached:
        required = {
            "original",
            "reconstruction",
            "indices",
            "feature_names",
            "checkpoint",
            "codebook_size",
        }
        if not required.issubset(cached.files):
            raise ValueError(
                f"Incomplete diagnostic arrays in {path}: {sorted(required - set(cached.files))}"
            )
        if (
            str(cached["checkpoint"]) != run["checkpoint"]["path"]
            or int(cached["codebook_size"]) != run["k"]
        ):
            raise ValueError(f"Wrong checkpoint/codebook in {path}")
        original, decoded = cached["original"], cached["reconstruction"]
        names = [str(name) for name in cached["feature_names"]]
        indices = cached["indices"]
        if indices.ndim == 1:
            indices = indices[:, None]
        if (
            original.ndim != 2
            or decoded.shape != original.shape
            or not len(original)
            or original.shape[1] != len(names)
            or len(set(n.lower() for n in names)) != len(names)
            or indices.shape != (len(original), run["q"])
        ):
            raise ValueError(f"Inconsistent feature/object/quantizer shapes in {path}")
        if not np.all(
            np.isfinite(indices)
            & (indices >= 0)
            & (indices < run["k"])
            & (indices == np.floor(indices))
        ):
            raise ValueError(f"Invalid code indices in {path}")
    return {"original": original, "reconstruction": decoded, "feature_names": names}


def feature_bins(scan, name, original, n_bins):
    finite = np.asarray(original)[np.isfinite(original)]
    if not len(finite):
        return np.array([]), np.array([]), "no finite original values"
    if name.lower() in CATEGORICAL_FEATURES:
        rounded = np.rint(finite)
        if not np.allclose(finite, rounded, rtol=0, atol=1e-5):
            raise ValueError(f"Non-integral original category values for {name}")
        centers = np.unique(rounded)
        if len(centers) > 64:
            raise ValueError(f"Unexpected category count for {name}: {len(centers)}")
        if len(centers) == 1:
            bins = np.array([centers[0] - 0.5, centers[0] + 0.5])
        else:
            bins = np.r_[
                centers[0] - (centers[1] - centers[0]) / 2,
                (centers[:-1] + centers[1:]) / 2,
                centers[-1] + (centers[-1] - centers[-2]) / 2,
            ]
        return bins, centers, "observed original categories"
    if finite.min() == finite.max():
        value = float(finite[0])
        delta = max(abs(value) * 0.01, 0.5)
        return (
            np.array([value - delta, value + delta]),
            np.array([value]),
            "constant original feature",
        )
    bins = scan.domain.make_bins(original, n_bins)
    return bins, 0.5 * (bins[:-1] + bins[1:]), "original 1st--99th percentiles (min/max fallback)"


def summarize_feature(scan, original, decoded, bins, centers, settings, run, sample, feature):
    kind = scan.binned_metric_kind(feature)
    _, values, counts, _, _ = scan.binned_metric_values(
        original,
        decoded,
        bins,
        metric_kind=kind,
        min_bin_count=settings["min_bin_count"],
        min_denominator=settings["min_denominator"],
    )
    if kind == "ratio_iqr_over_median":
        values *= 100
    return [
        {
            "object": run["object"],
            "sample": sample,
            "label": run["label"],
            "q": run["q"],
            "k": run["k"],
            "d": run["d"],
            "feature": feature,
            "metric": kind,
            "bin": i,
            "bin_low": float(bins[i]),
            "bin_high": float(bins[i + 1]),
            "x": float(center),
            "value": float(value) if np.isfinite(value) else None,
            "objects_in_bin": int(count),
            "evaluated_objects": len(original),
        }
        for i, (center, value, count) in enumerate(zip(centers, values, counts))
    ]


def load_metrics(path, plan, obj, *, verify_sources=False):
    result = json.loads(path.read_text())
    identity = result.pop("metrics_id")
    if (
        result["version"] != METRICS_VERSION
        or result["plan_id"] != plan["plan_id"]
        or result["object"] != obj
        or capacity.digest_json(result) != identity
    ):
        raise ValueError(f"Wrong, changed, or outdated feature metrics: {path}")
    if verify_sources:
        for record in [result["pt_metrics"], *result["cache_files"]]:
            capacity.verify_file(record)
    result["metrics_id"] = identity
    return result


def cache_metrics(args):
    import compare_object_google_stage1_scan as scan

    plan = capacity.load_plan(args.source_dir)
    owners = cache_owners(args.source_dir, plan)
    output_dir = args.source_dir / "all_features"
    for obj in args.objects:
        destination = output_dir / f"metrics_{obj}.json"
        if destination.exists():
            load_metrics(destination, plan, obj, verify_sources=True)
            log.info("Reusing %s; no arrays re-read", destination)
            continue
        pt_record = capacity.file_record(args.source_dir / f"metrics_{obj}.json", content_hash=True)
        pt_rows = capacity.read_completed_metrics(pt_record, plan, obj)
        runs = [r for r in plan["runs"] if r["object"] == obj]
        if not runs:
            raise ValueError(f"No audited runs for {obj}")
        paths = {}
        for sample in plan["samples"]:
            for run in runs:
                owner_dir, owner_plan = owners[(obj, run["label"])]
                paths[(sample, run["label"])] = resolve_cache(
                    scan, owner_dir, owner_plan, run, sample
                )
        records = [capacity.file_record(path) for path in paths.values()]
        rows, features, binning = [], None, {}
        for sample in plan["samples"]:
            reference = None
            for run in runs:
                path = paths[(sample, run["label"])]
                log.info(
                    "%s/%s/%s: summarizing cached features from %s", obj, sample, run["label"], path
                )
                item = read_arrays(path, run)
                previous_pt = sorted(
                    (r for r in pt_rows if r["sample"] == sample and r["label"] == run["label"]),
                    key=lambda r: r["bin"],
                )
                if any(r["evaluated_objects"] != len(item["original"]) for r in previous_pt):
                    raise ValueError(f"Cached object count differs from frozen pT metrics: {path}")
                if reference is None:
                    reference = {
                        "original": item["original"],
                        "reconstruction": item["original"],
                        "feature_names": item["feature_names"],
                    }
                    if features is None:
                        features = [
                            {
                                "name": name,
                                "label": scan.feature_label(name),
                                "unit": scan.feature_unit(name),
                                "metric": scan.binned_metric_kind(name),
                            }
                            for name in item["feature_names"]
                        ]
                    if item["feature_names"] != [f["name"] for f in features]:
                        raise ValueError(f"Feature ordering differs across domains for {obj}")
                    bins_by_feature = {}
                    binning[sample] = {}
                    for i, feature in enumerate(features):
                        name = feature["name"]
                        if name.lower() == "pt":
                            bins = np.array(
                                [r["bin_low"] for r in previous_pt] + [previous_pt[-1]["bin_high"]]
                            )
                            centers = np.array([r["pt"] for r in previous_pt])
                            note = "exact frozen Figure 2 pT bins"
                        else:
                            bins, centers, note = feature_bins(
                                scan,
                                name,
                                reference["original"][:, i],
                                plan["evaluation"]["n_bins"],
                            )
                        bins_by_feature[name] = bins, centers
                        binning[sample][name] = {
                            "edges": bins.tolist(),
                            "centers": centers.tolist(),
                            "note": note,
                        }
                if item["feature_names"] != reference["feature_names"]:
                    raise ValueError(f"Feature ordering differs within {obj}: {path}")
                scan.domain.verify_shared_inputs(
                    sample,
                    obj,
                    {"reference": reference, run["label"]: item},
                    reference["feature_names"],
                )
                for i, feature in enumerate(features):
                    name = feature["name"]
                    bins, centers = bins_by_feature[name]
                    if not len(bins):
                        log.warning(
                            "%s/%s/%s: no finite original values; retaining empty panel",
                            obj,
                            sample,
                            name,
                        )
                        continue
                    feature_rows = summarize_feature(
                        scan,
                        reference["original"][:, i],
                        item["reconstruction"][:, i],
                        bins,
                        centers,
                        plan["evaluation"],
                        run,
                        sample,
                        name,
                    )
                    if name.lower() == "pt":
                        old_values = np.array(
                            [r["response_width_percent"] for r in previous_pt], dtype=float
                        )
                        new_values = np.array([r["value"] for r in feature_rows], dtype=float)
                        if not np.allclose(
                            old_values, new_values, rtol=1e-10, atol=1e-12, equal_nan=True
                        ):
                            raise ValueError(
                                f"Cache pT widths differ from completed Figure 2 metrics: {path}"
                            )
                        if [r["objects_in_bin"] for r in feature_rows] != [
                            r["objects_in_bin"] for r in previous_pt
                        ]:
                            raise ValueError(
                                f"Cache pT bin counts differ from completed Figure 2 metrics: {path}"
                            )
                    rows.extend(feature_rows)
                del item
            del reference
        for record in [pt_record, *records]:
            capacity.verify_file(record)
        result = {
            "version": METRICS_VERSION,
            "plan_id": plan["plan_id"],
            "object": obj,
            "features": features,
            "binning": binning,
            "pt_metrics": pt_record,
            "cache_files": records,
            "rows": rows,
        }
        result["metrics_id"] = capacity.digest_json(result)
        output_dir.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.replace(destination)
        log.info("Saved %s (%d features); no inference", destination, len(features))


def feature_scale(rows, requested):
    values = np.array([r["value"] for r in rows], dtype=float)
    finite = values[np.isfinite(values)]
    # Zero IQR is common for discrete features; keep it visible, never add epsilon.
    scale = "log" if requested == "log" and len(finite) and np.all(finite > 0) else "linear"
    if not len(finite):
        return scale, (0, 1)
    return scale, capacity.response_axis_limits(
        [{"response_width_percent": value} for value in finite], scale
    )


def render_object(result, output_dir, *, y_scale="log", pdfs=None, title_suffix=""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter, StrMethodFormatter

    obj, features, rows = result["object"], result["features"], result["rows"]
    configs = sorted({(r["q"], r["k"], r["d"]) for r in rows})
    label = OBJECT_LABELS[OBJECTS.index(obj)]
    nrows = (len(features) + 2) // 3
    height = 2.5 * nrows + 2.9
    chosen = capacity.SELECTED_CONFIGURATIONS[obj]
    suffix = "_log" if y_scale == "log" else ""
    display = {
        f["name"]: feature_scale([r for r in rows if r["feature"] == f["name"]], y_scale)
        for f in features
    }
    with paper_style():
        for sample, domain in (("mc", "Simulation"), ("data", "Collision data")):
            fig, axes = plt.subplots(nrows, 3, figsize=(12, height), squeeze=False)
            fig.subplots_adjust(
                left=0.085,
                right=0.985,
                bottom=2.35 / height,
                top=1 - 0.65 / height,
                wspace=0.34,
                hspace=0.47,
            )
            fig.text(
                0.085, 1 - 0.13 / height, f"{label} | {domain}{title_suffix}", fontsize=11, va="top"
            )
            highlighted = {}
            for panel, (ax, feature) in enumerate(zip(axes.flat, features)):
                name = feature["name"]
                scale, limits = display[name]
                feature_rows = [r for r in rows if r["sample"] == sample and r["feature"] == name]
                ax.set_yscale(scale)
                for q, k, d in configs:
                    curve = sorted(
                        (r for r in feature_rows if (r["q"], r["k"], r["d"]) == (q, k, d)),
                        key=lambda r: r["bin"],
                    )
                    if not curve:
                        continue
                    selected = (q, k, d) == chosen
                    if selected and any(r["value"] is not None for r in curve):
                        highlighted[chosen] = [label]
                    ax.plot(
                        [r["x"] for r in curve],
                        np.array([r["value"] for r in curve], dtype=float),
                        **capacity_curve_style(q, k, d, selected=selected),
                    )
                style_axis(ax)
                ax.set_ylim(*limits)
                if scale == "log":
                    # Narrow log ranges still need readable numeric ticks.
                    decades = np.floor(np.log10(limits[1])) - np.ceil(np.log10(limits[0]))
                    ax.yaxis.set_major_locator(
                        LogLocator(base=10) if decades >= 1 else MaxNLocator(nbins=4, min_n_ticks=3)
                    )
                    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
                    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
                    ax.yaxis.set_minor_formatter(NullFormatter())
                ax.set_title(f"({chr(97 + panel)}) {feature['label']}", loc="left", fontsize=10)
                unit = f" [{feature['unit']}]" if feature["unit"] else ""
                ax.set_xlabel(
                    PT_LABEL if name.lower() == "pt" else f"Original {feature['label']}{unit}",
                    fontsize=9,
                )
                ax.set_ylabel(
                    (
                        RESPONSE_LABEL
                        if name.lower() == "pt"
                        else (
                            "Relative response width [%]"
                            if feature["metric"] == "ratio_iqr_over_median"
                            else f"Residual IQR{unit}"
                        )
                    ),
                    fontsize=9,
                )
                note = None
                if not any(r["value"] is not None for r in feature_rows):
                    note = "No populated finite bins"
                elif y_scale == "log" and scale == "linear":
                    note = "Linear scale: zero width present"
                if note:
                    ax.text(0.03, 0.96, note, transform=ax.transAxes, va="top", fontsize=7)
                if name.lower() in CATEGORICAL_FEATURES:
                    centers = result["binning"][sample][name]["centers"]
                    if len(centers) <= 12:
                        ax.set_xticks(centers)
            for ax in list(axes.flat)[len(features) :]:
                fig.delaxes(ax)
            if not highlighted:
                log.warning(
                    "%s/%s: selected configuration has no finite feature curves; no substitute highlight",
                    obj,
                    sample,
                )
            capacity_legends(
                fig,
                configs,
                highlighted,
                (0.214, 0.535, 0.856),
                top=1.52 / height,
                bottom=0.041 / height,
            )
            stem = output_dir / f"capacity_{obj}_all_features_{sample}{suffix}"
            save_figure(fig, stem)
            if pdfs is not None:
                pdfs[sample].savefig(fig, facecolor="white")
            plt.close(fig)
    return {
        name: {"scale": scale, "limits": list(limits)} for name, (scale, limits) in display.items()
    }


def plot(args):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages

    plan = capacity.load_plan(args.source_dir)
    output_dir = args.source_dir / "all_features"
    results = [load_metrics(output_dir / f"metrics_{obj}.json", plan, obj) for obj in args.objects]
    suffix = "_log" if args.y_scale == "log" else ""
    # A subset gets its own bundle name, so it cannot overwrite the six-object PDF.
    bundle = "all_objects" if tuple(args.objects) == OBJECTS else "_".join(args.objects)
    display = {}
    with ExitStack() as stack:
        pdfs = {
            sample: stack.enter_context(
                PdfPages(output_dir / f"capacity_{bundle}_feature_pages_{sample}{suffix}.pdf")
            )
            for sample in ("mc", "data")
        }
        for result in results:
            obj = result["object"]
            display[obj] = render_object(result, output_dir, y_scale=args.y_scale, pdfs=pdfs)
            with (output_dir / f"capacity_{obj}_all_features.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0]))
                writer.writeheader()
                writer.writerows(result["rows"])
    notes = {
        "plan_id": plan["plan_id"],
        "objects": args.objects,
        "metrics_ids": {r["object"]: r["metrics_id"] for r in results},
        "selected_configurations": {
            obj: capacity.SELECTED_CONFIGURATIONS[obj] for obj in args.objects
        },
        "display": display,
        "skipped_runs": [r for r in plan["skipped"] if r["object"] in args.objects],
    }
    (output_dir / f"capacity_{bundle}_display{suffix}.json").write_text(
        json.dumps(notes, indent=2) + "\n"
    )
    (output_dir / f"capacity_{bundle}_caption{suffix}.md").write_text(
        "# All-feature tokenizer capacity scans\n\n"
        "One page per object and evaluation domain; each panel overlays the completed stage-1 "
        "configurations in the frozen Figure 2 audit. Thick black stars identify the current "
        "per-object choice, without substituting full-training results.\n\n"
        "pT and mass: 100 * IQR(decoded/original) / abs(median(decoded/original)) [%]. "
        "All other features: IQR(decoded - original), in the cached input's native units. "
        "This preserves the existing scan metric convention, including unwrapped phi differences. "
        "Residual IQR for charge/ID/decay-mode features is NOT classification accuracy: "
        "zero width does not prove exact reconstruction or zero bias.\n\n"
        "pT bin edges, counts, and widths are checked against the completed Figure 2 metrics. "
        "Other continuous features use common original-feature 1st--99th percentile bins "
        "(min/max fallback). Discrete charge/ID/decay-mode features use observed-category bins; "
        "constant features use one bin. No feature is silently omitted. "
        f"Minimum bin population: {plan['evaluation']['min_bin_count']}. "
        "Missing/nonfinite metrics are gaps, not zeros. With --y-scale log, a panel containing "
        "zero widths switches to linear in BOTH domains; zeros are not replaced with epsilon. "
        "Each feature has matched simulation/data y-limits, in its own units. "
        "No uncertainty bands are inferred.\n\n"
        "All summaries come from exact plan-scoped diagnostic arrays (including parent-plan "
        "caches in place). No new models, files, preprocessing transforms, or inference are used. "
        "See metrics JSON for cache identities, feature names and bin edges; the display JSON "
        "records scale choices and missing runs. Original means detector-reconstructed, not "
        "generator truth. File disjointness is not an event-ID deduplication guarantee.\n"
    )
    log.info(
        "Saved %d object pages per domain, individual PDF/PNG and PDF bundles under %s",
        len(results),
        output_dir,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    summarize = sub.add_parser(
        "cache-metrics", help="Summarize ALL features from audited caches only; never infer"
    )
    plotting = sub.add_parser(
        "plot", help="Plot saved feature summaries only; no cache arrays or GPU"
    )
    plotting.add_argument("--y-scale", choices=("linear", "log"), default="log")
    for command in (summarize, plotting):
        command.add_argument(
            "--source-dir", type=Path, required=True, help="Completed Figure 2 audit directory"
        )
        command.add_argument("--objects", nargs="+", choices=OBJECTS, default=list(OBJECTS))
    args = parser.parse_args()
    if len(set(args.objects)) != len(args.objects):
        parser.error("Each object may be requested only once")
    return args


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    {"cache-metrics": cache_metrics, "plot": plot}[args.command](args)


if __name__ == "__main__":
    main()
