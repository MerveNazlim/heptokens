#!/usr/bin/env python3
"""Scalar sum of object pT on the existing full-Q8 event-closure sample.

The collection list is explicit. Existing event caches are read by reference;
only additional collections are inferred, without changing HT or HZZ results.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np

import paper_event_reconstruction as event
import paper_tokenizer_test_split as split
from paper_plot_style import paper_style, save_figure

reco = event.reco
log = logging.getLogger(__name__)
VERSION = "paper-full-q8-st-v1"
OBJECTS = ("jets", "electrons", "muons", "photons", "taus", "tracks")
RUN_RECORDS = ("config", "completion", "checkpoint", "preprocessor", "preprocessor_metadata")


def scalar_pt_sums(arrays, ids):
    """Sum each packed event independently; empty events are zero, bad pT is NaN."""
    event.validate_arrays(arrays, ids)
    offsets = arrays["offsets"]
    nonempty = np.flatnonzero(np.diff(offsets))
    sums = np.zeros((len(ids), 2), dtype=np.float64)
    for column, key in enumerate(("original", "reconstruction")):
        pts = np.asarray(arrays[key][:, 0], dtype=np.float64)
        pts = np.where(np.isfinite(pts) & (pts >= 0), pts, np.nan)
        if len(nonempty):
            sums[nonempty, column] = np.add.reduceat(pts, offsets[nonempty])
    return sums


def load_plan(directory):
    plan = reco.read_sealed(directory / "plan.json", "plan_id")
    if plan["version"] != VERSION:
        raise ValueError("Not an ST plan; use a separate output directory")
    return plan


def build_plan(args):
    if (args.output_dir / "plan.json").exists():
        raise FileExistsError("ST plan already frozen; use evaluate/summarize/plot")
    base = event.load_plan(args.event_dir)
    parent = event.source_plan(base)
    if parent.get("evaluation_protocol") != "saved-test":
        raise ValueError("ST requires the saved-test full-Q8 event workflow")
    split.verify_partition(parent)
    for record in base["implementation"]:
        reco.capacity.verify_file(record)
    ids = {}
    for sample in reco.DOMAINS:
        ids[sample] = event.membership_ids(base, sample)
        expected = event.event_selection(parent, sample, base["settings"]["max_events_per_domain"])
        if not np.array_equal(ids[sample], expected):
            raise ValueError(f"Source event membership differs from saved-test selection: {sample}")
    runs = []
    for obj in args.objects:
        matches = [r for r in parent["runs"] if r["object"] == obj]
        if len(matches) != 1:
            raise ValueError(f"Missing/ambiguous full tokenizer: {obj}")
        run = matches[0]
        if (run["q"], run["k"], run["d"]) != reco.capacity.SELECTED_CONFIGURATIONS[obj]:
            raise ValueError(f"Wrong selected full tokenizer: {obj}")
        for key in RUN_RECORDS:
            reco.capacity.verify_file(run[key])
        if any(run["feature_names"].count(k) != 1 for k in ("pt", "eta", "phi")):
            raise ValueError(f"Need exactly one pt/eta/phi in saved {obj} features")
        _, settings = split.saved_settings(run)
        if settings != parent["test_partition"]["settings"]:
            raise ValueError(f"Saved training/test split differs for {obj}")
        runs.append(run)
    shared = {}
    base_runs = {r["object"]: r for r in base["runs"]}
    for sample in reco.DOMAINS:
        shared[sample] = {}
        for run in runs:
            obj = run["object"]
            if obj not in base_runs:
                continue
            if run != base_runs[obj]:
                raise ValueError(f"Source event tokenizer differs from parent: {obj}")
            path = event.cache_path(args.event_dir, base, obj, sample)
            if not path.is_file() or not path.with_suffix(".json").is_file():
                raise FileNotFoundError(
                    f"Complete the existing event evaluation first: {path}. "
                    "ST will reuse it rather than duplicate inference."
                )
            _, receipt = event.read_cache(path, base, run, sample, ids[sample])
            shared[sample][obj] = {
                "arrays": receipt["arrays"],
                "receipt": reco.capacity.file_record(path.with_suffix(".json"), content_hash=True),
            }
            log.info("Reuse existing aligned cache: %s/%s", obj, sample)
    plan = reco.seal(
        {
            "version": VERSION,
            "event_plan": reco.capacity.file_record(
                args.event_dir / "plan.json", content_hash=True
            ),
            "source_plan": base["source_plan"],
            "membership": base["membership"],
            "counts": base["counts"],
            "runs": runs,
            "reused_caches": shared,
            "settings": {
                "objects": args.objects,
                "include_met": False,
                "pt_threshold_gev": None,
                "batch_size": base["settings"]["batch_size"],
                "input_momentum_unit": base["settings"]["input_momentum_unit"],
                "checkpoint_name": base["settings"]["checkpoint_name"],
            },
            "statistics": base["statistics"],
            "holdout_note": base["holdout_note"],
            "implementation": [
                reco.capacity.file_record(p, content_hash=True)
                for p in (__file__, event.__file__, event.physics.__file__)
            ],
        },
        "plan_id",
    )
    reco.write_json(args.output_dir / "plan.json", plan)
    log.info("ST = sum pT over %s; no additional pT cut, MET excluded", ", ".join(args.objects))
    log.info("MC/data events: %s; original saved masks/caps retained; no inference", plan["counts"])
    if "tracks" in args.objects:
        log.warning("Tracks are included explicitly; momentum may also appear in jets/leptons")


def read_object(directory, plan, run, sample, ids):
    shared = plan["reused_caches"][sample].get(run["object"])
    if shared is not None:
        reco.capacity.verify_file(plan["event_plan"])
        reco.capacity.verify_file(shared["receipt"])
        reco.capacity.verify_file(shared["arrays"])
        base = event.load_plan(Path(plan["event_plan"]["path"]).parent)
        path, owner = Path(shared["arrays"]["path"]), base
    else:
        path, owner = event.cache_path(directory, plan, run["object"], sample), plan
    arrays, receipt = event.read_cache(path, owner, run, sample, ids)
    sources = [
        receipt["arrays"],
        reco.capacity.file_record(path.with_suffix(".json"), content_hash=True),
    ]
    return arrays, sources


def evaluate(args):
    plan = load_plan(args.output_dir)
    reco.capacity.verify_file(plan["source_plan"])
    reco.capacity.verify_file(plan["event_plan"])
    parent = reco.load_plan(Path(plan["source_plan"]["path"]).parent)
    split.verify_partition(parent)
    for record in plan["implementation"]:
        reco.capacity.verify_file(record)
    for sample in args.samples:
        ids = event.membership_ids(plan, sample)
        for run in plan["runs"]:
            for key in RUN_RECORDS:
                reco.capacity.verify_file(run[key])
            obj = run["object"]
            path = event.cache_path(args.output_dir, plan, obj, sample)
            if (
                obj in plan["reused_caches"][sample]
                or path.exists()
                or path.with_suffix(".json").exists()
            ):
                read_object(args.output_dir, plan, run, sample, ids)
                log.info("Reusing verified event cache: %s/%s", obj, sample)
                continue
            if args.cache_only:
                raise FileNotFoundError(f"Missing ST event cache: {path}; inference forbidden")
            arrays = event.infer(run, parent, plan, sample, ids, args.device)
            event.save_npz(path, **arrays)
            reco.write_json(
                path.with_suffix(".json"),
                reco.seal(
                    {
                        "plan_id": plan["plan_id"],
                        "object": obj,
                        "sample": sample,
                        "checkpoint": run["checkpoint"],
                        "arrays": reco.capacity.file_record(path, content_hash=True),
                    },
                    "receipt_id",
                ),
            )


def load_summary(directory, plan):
    summary = reco.read_sealed(directory / "st_summary.json", "summary_id")
    if summary["plan_id"] != plan["plan_id"]:
        raise ValueError("Wrong ST summary provenance")
    return summary


def summarize(args):
    plan = load_plan(args.output_dir)
    destination = args.output_dir / "st_summary.json"
    if destination.exists():
        for source in load_summary(args.output_dir, plan)["sources"]:
            reco.capacity.verify_file(source)
        log.info("Reusing ST summary")
        return
    pairs, sources = {}, []
    for sample in reco.DOMAINS:
        ids = event.membership_ids(plan, sample)
        pairs[sample] = np.zeros((len(ids), 2), dtype=np.float64)
        for run in plan["runs"]:
            arrays, records = read_object(args.output_dir, plan, run, sample, ids)
            pairs[sample] += scalar_pt_sums(arrays, ids)
            sources.extend(records)
            del arrays
    settings = plan["statistics"]
    pooled = np.concatenate([p[:, 0] for p in pairs.values()])
    residuals = np.concatenate([p[:, 1] - p[:, 0] for p in pairs.values()])
    edges = reco.bin_edges(pooled, settings["n_bins"], settings["percentiles"])
    residual_edges = reco.bin_edges(residuals, settings["n_bins"], settings["percentiles"])
    domains = {}
    for sample, values in pairs.items():
        x, y = values.T
        row = reco.feature_summary(x, y, edges, residual_edges, settings["min_ratio_count"], "ST")
        row.update(
            {
                "total_events": len(x),
                "original_eligible": int(np.isfinite(x).sum()),
                "decoded_invalid": int((np.isfinite(x) & ~np.isfinite(y)).sum()),
                "original_empty_or_zero": int(np.count_nonzero(x == 0)),
            }
        )
        log.info(
            "ST/%s: %d eligible, %d paired, %d decoded-invalid events",
            sample,
            row["original_eligible"],
            row["finite_pairs"],
            row["decoded_invalid"],
        )
        if row["finite_pairs"] == 0:
            raise ValueError(f"No finite ST pairs for {sample}; refusing empty plot")
        domains[sample] = row
    reco.write_json(
        destination,
        reco.seal(
            {
                "plan_id": plan["plan_id"],
                "sources": sources,
                "features": [
                    {
                        "name": "ST",
                        "label": r"$S_T$",
                        "unit": "GeV",
                        "edges": edges.tolist(),
                        "residual_edges": residual_edges.tolist(),
                        "domains": domains,
                    }
                ],
            },
            "summary_id",
        ),
    )


def caption(plan):
    objects = ", ".join(plan["settings"]["objects"])
    tracks = (
        "Tracks are included; their momentum can also be represented by jets or leptons. "
        if "tracks" in plan["settings"]["objects"]
        else "Tracks are excluded. "
    )
    return (
        "# Scalar sum of object transverse momenta\n\n"
        f"ST = sum of scalar pT over these collections: {objects}. "
        "MET and event/structural tokens are excluded. "
        + tracks
        + "No additional pT cut is applied, including to jets; this is distinct from HT's "
        "20 GeV jet threshold. The same original object slots, saved masks, cleaning and "
        "per-collection object caps are used before and after decoding. No new overlap-removal "
        "or deduplication is performed; this is the scalar sum of the stored selected "
        "collections, not a unique visible-energy sum. Empty events have ST=0. Nonfinite or "
        "negative pT in any included object invalidates that side of the event pair rather "
        "than silently dropping or clipping that object's contribution.\n\n"
        f"Full Q8 tokenizers with their saved joblibs and {plan['settings']['checkpoint_name']}; "
        "exact object run/configuration records are in plan.json. The same MC/data event "
        "identities as the inclusive event-observable plan are retained; no HZZ signal "
        "selection is applied. Existing caches are reused by verified reference. "
        f"Counts: {plan['counts']}. {plan['holdout_note']}\n\n"
        "Each triptych contains the original/decoded distributions and decoded/original "
        "bin-count ratio, decoded-minus-original residual in GeV, and paired density with "
        "y=x and equal axes. Original means detector-reconstructed, not generator truth. "
        f"Common MC/data display edges use {plan['statistics']['percentiles']} percentiles "
        "of original ST; residual ranges use paired residual percentiles. Histograms "
        "normalize by all finite paired events, including out-of-window values. "
        f"Ratio bins require {plan['statistics']['min_ratio_count']} original entries. "
        "Counts, exclusions, median residual, median absolute residual and IQR are exported. "
        "No uncertainty bands are inferred. Inclusive HT/Delta R and dedicated HZZ mass "
        "results are unchanged; this is object-tokenizer closure before sequence assembly.\n"
    )


def plot(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plan = load_plan(args.output_dir)
    summary = load_summary(args.output_dir, plan)
    destination = args.output_dir / "figures"
    with paper_style():
        for sample in reco.DOMAINS:
            fig = reco.render_triptych_page(
                summary["features"],
                "event",
                sample,
                1,
                1,
                log_y=args.y_scale == "log",
                title="Scalar object pT sum",
                count_label="events",
            )
            save_figure(fig, destination / f"st_{sample}_{args.y_scale}")
            plt.close(fig)
    rows = [
        {
            "sample": sample,
            **{k: v for k, v in values.items() if not isinstance(v, list) and k != "total_objects"},
        }
        for sample, values in summary["features"][0]["domains"].items()
    ]
    with (destination / "st_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (destination / "st_caption.md").write_text(caption(plan))
    log.info("Saved ST triptychs: %s; summaries only, no inference", destination)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("plan")
    audit.add_argument("--event-dir", type=Path, required=True)
    audit.add_argument(
        "--objects",
        nargs="+",
        choices=OBJECTS,
        required=True,
        help="Explicit collections in the scalar sum; MET is not included",
    )
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument(
        "--samples", choices=tuple(reco.DOMAINS), nargs="+", default=list(reco.DOMAINS)
    )
    evaluation.add_argument("--device", default="cpu")
    evaluation.add_argument("--cache-only", action="store_true")
    summary = commands.add_parser("summarize")
    plotting = commands.add_parser("plot")
    plotting.add_argument("--y-scale", choices=("log", "linear"), default="log")
    for command in (audit, evaluation, summary, plotting):
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "plan" and len(set(args.objects)) != len(args.objects):
        parser.error("Each object collection must appear exactly once")
    return args


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    {"plan": build_plan, "evaluate": evaluate, "summarize": summarize, "plot": plot}[args.command](
        args
    )


if __name__ == "__main__":
    main()
