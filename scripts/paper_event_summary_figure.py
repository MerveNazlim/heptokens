#!/usr/bin/env python3
"""Three paper panels: HZZ m4l, inclusive ST (no tracks), and leading lepton-jet Delta R.

Reads sealed plans and small summaries only. No inference, H5 loading, NPZ
loading, new selections, or histogram rebinning are performed.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np

import paper_event_reconstruction as event
import paper_event_st as st
import paper_hzz_mass_reconstruction as hzz
from paper_plot_style import paper_style, save_figure

reco = event.reco
log = logging.getLogger(__name__)
OBJECTS = {"jets", "electrons", "muons", "photons", "taus"}


def same_content(a, b):
    return bool(a.get("sha256")) and (a["sha256"], a["size"]) == (b.get("sha256"), b["size"])


def verify_compatibility(plans, event_record):
    base, scalar, mass = plans["event"], plans["st"], plans["hzz"]
    selected = scalar["settings"]["objects"]
    if len(selected) != 5 or set(selected) != OBJECTS:
        raise ValueError(
            "ST must contain exactly jets, electrons, muons, photons and taus; no tracks"
        )
    if scalar["settings"]["include_met"] or scalar["settings"]["pt_threshold_gev"] is not None:
        raise ValueError("Expected ST without MET or an additional pT threshold")
    if not same_content(scalar["event_plan"], event_record):
        raise ValueError("ST was not built from the supplied event-observable plan")
    if scalar["membership"] != base["membership"] or scalar["counts"] != base["counts"]:
        raise ValueError("ST and Delta R must use the same frozen event sample")
    if base["settings"]["delta_r_pairing"] != "leading_original_lepton_and_jet":
        raise ValueError("Expected leading original lepton-leading original jet Delta R")
    for name, plan in plans.items():
        if not same_content(plan["source_plan"], base["source_plan"]):
            raise ValueError(f"{name} has a different full-tokenizer reconstruction audit")
        for key in ("input_momentum_unit", "checkpoint_name"):
            if plan["settings"][key] != base["settings"][key]:
                raise ValueError(f"{name} differs in {key}")
        if plan["statistics"] != base["statistics"]:
            raise ValueError(f"{name} has different binning/ratio statistics settings")
    runs = {}
    for name, plan in plans.items():
        for run in plan["runs"]:
            obj = run["object"]
            if (run["q"], run["k"], run["d"]) != reco.capacity.SELECTED_CONFIGURATIONS[obj]:
                raise ValueError(f"Wrong selected full-Q8 model: {name}/{obj}")
            if obj in runs and runs[obj] != run:
                raise ValueError(f"Different model/preprocessing records for {obj}")
            runs[obj] = run


def select_feature(summary, name, label, unit):
    matches = [f for f in summary["features"] if f["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"Missing/ambiguous {name} summary")
    feature = matches[0]
    if feature["unit"] != unit or "mc" not in feature["domains"]:
        raise ValueError(f"Wrong units or missing simulation summary for {name}")
    row = feature["domains"]["mc"]
    edges = np.asarray(feature["edges"], dtype=float)
    if (
        edges.ndim != 1
        or len(edges) < 2
        or not np.isfinite(edges).all()
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError(f"Invalid bin edges for {name}")
    if row["finite_pairs"] <= 0:
        raise ValueError(f"No finite {name} pairs; refusing an empty paper panel")
    for key in ("original_counts", "decoded_counts", "bin_count_ratio"):
        if len(row[key]) != len(edges) - 1:
            raise ValueError(f"Invalid {key} shape for {name}")
    for key in ("original_counts", "decoded_counts"):
        counts = np.asarray(row[key])
        if (
            not np.isfinite(counts).all()
            or np.any(counts < 0)
            or counts.sum() > row["finite_pairs"]
        ):
            raise ValueError(f"Invalid histogram counts for {name}")
    return {**feature, "label": label, "domains": {"mc": row}}


def load_inputs(args):
    directories = {"event": args.event_dir, "st": args.st_dir, "hzz": args.hzz_dir}
    modules = {"event": event, "st": st, "hzz": hzz}
    plans, summaries, sources = {}, {}, []
    filenames = {"event": "event_summary.json", "st": "st_summary.json", "hzz": "hzz_summary.json"}
    for name, directory in directories.items():
        plans[name] = modules[name].load_plan(directory)
        summaries[name] = modules[name].load_summary(directory, plans[name])
        sources.extend(
            reco.capacity.file_record(directory / filename, content_hash=True)
            for filename in ("plan.json", filenames[name])
        )
    event_record = reco.capacity.file_record(args.event_dir / "plan.json", content_hash=True)
    verify_compatibility(plans, event_record)
    features = [
        select_feature(summaries["hzz"], "m4l", r"$m_{4\ell}$", "GeV"),
        select_feature(summaries["st"], "ST", r"$S_T$", "GeV"),
        select_feature(summaries["event"], "delta_r_lj", r"$\Delta R(\ell_1,j_1)$", ""),
    ]
    return plans, features, sources


def render(features, y_scales):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 4.4))
    grid = fig.add_gridspec(1, 3, left=0.075, right=0.985, bottom=0.16, top=0.79, wspace=0.32)
    samples = (r"$H\to ZZ^{(*)}\to4\ell$ signal", "Inclusive MC", "Inclusive MC")
    handles = labels = None
    for index, (feature, scale, sample_label) in enumerate(
        zip(features, y_scales, samples, strict=True)
    ):
        ax, ratio = reco.draw_distribution(
            fig, grid[index], feature, "mc", log_y=scale == "log", count_label="events"
        )
        ax.set_title(
            f"({chr(97 + index)}) {feature['label']}\n{sample_label}",
            loc="left",
            fontsize=10,
            linespacing=1.5,
        )
        if index == 0:
            handles, labels = ax.get_legend_handles_labels()
        else:
            ax.set_ylabel("")
            ratio.set_ylabel("")
    fig.text(0.075, 0.965, "Simulation", ha="left", va="top", fontsize=10)
    fig.legend(
        handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.99), ncol=2, frameon=False
    )
    return fig


def caption(plans, features):
    base, mass = plans["event"], plans["hzz"]
    rows = [f["domains"]["mc"] for f in features]
    return (
        "# Reconstruction of derived event observables\n\n"
        "Original (detector-reconstructed input) and decoded full-Q8 distributions, "
        "with decoded/original bin-count ratios underneath. The three panels deliberately "
        "use different physics selections: (a) m4l uses HZZ signal simulation, DSIDs "
        f"{mass['settings']['signal_dsids']}; (b) ST and (c) Delta R use the same inclusive "
        "MC event subset. The HZZ signal sample is NOT the common sample for all panels. "
        f"Finite paired events (m4l, ST, Delta R): {[r['finite_pairs'] for r in rows]}.\n\n"
        "m4l uses the four highest-original-pT electrons/muons and the same identities "
        "after decoding, with fixed lepton rest masses; no SFOS pairing, ID/isolation "
        "cuts or imposed mass window are added. ST sums pT over jets, electrons, muons, "
        "photons and taus; tracks and MET are excluded. No extra pT threshold or new "
        "overlap removal is applied to ST. Saved masks, object caps and cleaning are retained. "
        "Delta R uses the leading original electron/muon and leading original jet with "
        f"pT > {base['settings']['jet_pt_threshold_gev']:g} GeV, retaining the same pair "
        "after decoding and wrapping Delta phi. Mass and ST are in GeV; Delta R is "
        "dimensionless. Invalid decoded selections are not replaced.\n\n"
        "All panels use the same audited selected full-Q8 models and joblibs; no training "
        "or inference is performed by this plotting command. It reads sealed summary "
        "statistics without rebinning or changing samples. Histogram bin edges are retained "
        "from the individual plots; each histogram is normalized by the full finite "
        "paired-event count, including events outside the displayed window. "
        f"Ratio bins require {base['statistics']['min_ratio_count']} original entries. "
        "No seed or sampling uncertainty is inferred. CSV includes counts, exclusions "
        "and residual statistics. Model and source identities are recorded in the figure "
        "manifest; this small-summary plotting step does not re-audit the underlying "
        f"large caches or H5 files. {base['holdout_note']}\n"
    )


def plot(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plans, features, sources = load_inputs(args)
    with paper_style():
        fig = render(features, args.y_scales)
        save_figure(fig, args.output)
        plt.close(fig)
    rows = []
    for feature, sample in zip(
        features, ("HZZ signal MC", "inclusive MC", "inclusive MC"), strict=True
    ):
        rows.append(
            {
                "observable": feature["name"],
                "sample": sample,
                "unit": feature["unit"],
                **{
                    key: value
                    for key, value in feature["domains"]["mc"].items()
                    if not isinstance(value, list) and key != "total_objects"
                },
            }
        )
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    args.output.with_suffix(".md").write_text(caption(plans, features))
    reco.write_json(
        args.output.with_suffix(".json"),
        reco.seal(
            {
                "version": "paper-event-three-distributions-v1",
                "sources": sources,
                "plans": {name: plan["plan_id"] for name, plan in plans.items()},
                "panel_order": [f["name"] for f in features],
                "y_scales": args.y_scales,
                "implementation": reco.capacity.file_record(__file__, content_hash=True),
            },
            "figure_id",
        ),
    )
    log.info("Saved %s and PDF; summaries only, no inference", args.output.with_suffix(".png"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hzz-dir", type=Path, required=True)
    parser.add_argument("--st-dir", type=Path, required=True)
    parser.add_argument("--event-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output stem or .png/.pdf path; both PNG and PDF are produced",
    )
    parser.add_argument(
        "--y-scales",
        nargs=3,
        choices=("linear", "log"),
        default=["linear", "log", "linear"],
        metavar=("M4L", "ST", "DR"),
    )
    args = parser.parse_args(argv)
    if args.output.suffix and args.output.suffix not in (".png", ".pdf"):
        parser.error("--output must be a stem or .png/.pdf path")
    return args


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    plot(parse_args())


if __name__ == "__main__":
    main()
