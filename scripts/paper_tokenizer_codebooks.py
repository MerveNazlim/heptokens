#!/usr/bin/env python3
"""Figure 4: full-Q8 codebook grouped bars from verified reconstruction caches.

No training, inference, legacy cache discovery, or sample replacement. Summarize
once with matched MC/data object counts; plotting reads only small summaries.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
from pathlib import Path

import numpy as np

import paper_tokenizer_reconstruction as reconstruction
from paper_plot_style import OBJECTS, OBJECT_LABELS, paper_style, save_figure, style_axis

log = logging.getLogger(__name__)
VERSION = "paper-full-q8-codebooks-v1"
DOMAINS = reconstruction.DOMAINS
STAGE_COLORS = (
    "#C6DBEF",
    "#9ECAE1",
    "#6BAED6",
    "#4292C6",
    "#2171B5",
    "#08519C",
    "#083D7C",
    "#082B58",
)


def selected_run(plan, obj):
    matches = [run for run in plan["runs"] if run["object"] == obj]
    if len(matches) != 1:
        raise ValueError(f"Missing/ambiguous full-training run for {obj}")
    run = matches[0]
    actual = (run["q"], run["k"], run["d"])
    expected = reconstruction.capacity.SELECTED_CONFIGURATIONS[obj]
    if actual != expected:
        raise ValueError(
            f"Wrong selected full Q8 configuration for {obj}: {actual}, expected {expected}"
        )
    return run


def validate_indices(indices, k, *, quantizers=8):
    indices = np.asarray(indices)
    if indices.ndim != 2 or indices.shape[1] != quantizers or len(indices) == 0:
        raise ValueError(
            f"Expected nonempty (objects, {quantizers}) code indices, got {indices.shape}"
        )
    if indices.dtype.kind not in "iuf" or not np.all(
        np.isfinite(indices) & (indices >= 0) & (indices < k) & (indices == np.floor(indices))
    ):
        raise ValueError("Invalid code indices; every stage needs one assignment in [0, K)")


def stage_metrics(indices, k, *, quantizers=8):
    validate_indices(indices, k, quantizers=quantizers)
    rows = []
    for q in range(quantizers):
        counts = np.bincount(indices[:, q].astype(np.int64), minlength=k)
        probabilities = counts[counts > 0].astype(float) / len(indices)
        entropy = float(-np.sum(probabilities * np.log(probabilities)))
        perplexity = float(np.clip(np.exp(entropy), 1, k))
        used = int(np.count_nonzero(counts))
        rows.append(
            {
                "quantizer": q,
                "assignments": len(indices),
                "used_codes": used,
                "unused_codes": k - used,
                "used_percent": 100.0 * used / k,
                "unused_percent": 100.0 * (k - used) / k,
                "entropy_nats": entropy,
                "perplexity": perplexity,
                "normalized_perplexity": perplexity / k,
            }
        )
    return rows


def matched_metrics(items, obj, k, seed):
    for indices in items.values():
        validate_indices(indices, k)
    n = min(len(items[sample]) for sample in DOMAINS)
    result = {}
    for domain_index, sample in enumerate(DOMAINS):
        indices = items[sample]
        # Stable per object/domain even when summaries are made in separate jobs.
        rng = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence([seed, OBJECTS.index(obj), domain_index]))
        )
        rows = (
            np.arange(n, dtype=np.int64)
            if len(indices) == n
            else np.sort(rng.choice(len(indices), size=n, replace=False))
        )
        result[sample] = {
            "available_objects": len(indices),
            "matched_objects": n,
            "selection": "all" if len(indices) == n else "uniform_without_replacement",
            "row_indices_sha256": hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest(),
            "metrics": stage_metrics(indices[rows], k),
        }
    return result


def summary_path(directory, obj):
    return directory / "codebook_summaries" / f"{obj}.json"


def load_summary(directory, plan, obj, *, verify_sources=False):
    path = summary_path(directory, obj)
    result = reconstruction.read_sealed(path, "summary_id")
    run = selected_run(plan, obj)
    if (
        result["version"] != VERSION
        or result["plan_id"] != plan["plan_id"]
        or result["object"] != obj
        or (result["q"], result["k"], result["d"]) != (run["q"], run["k"], run["d"])
    ):
        raise ValueError(f"Wrong codebook summary provenance: {path}")
    if verify_sources:
        for record in result["sources"]:
            reconstruction.capacity.verify_file(record)
    return result


def summarize(args):
    plan = reconstruction.load_plan(args.output_dir)
    runs = [selected_run(plan, obj) for obj in args.objects]
    missing = []
    for run in runs:
        for sample in DOMAINS:
            path = reconstruction.cache_path(args.output_dir, plan, run, sample)
            missing.extend(str(p) for p in (path, path.with_suffix(".json")) if not p.is_file())
    if missing:
        raise FileNotFoundError(
            "Missing full-Q8 reconstruction caches/receipts:\n"
            + "\n".join(missing)
            + "\nFinish reconstruction evaluate for these objects first. No inference performed. "
            "Use --objects jets for an explicitly jets-only preview."
        )
    for run in runs:
        obj = run["object"]
        destination = summary_path(args.output_dir, obj)
        if destination.exists():
            existing = load_summary(args.output_dir, plan, obj, verify_sources=True)
            if existing["matching_seed"] != args.seed:
                raise ValueError(
                    f"Matching seed already frozen to {existing['matching_seed']}: {destination}"
                )
            log.info("Reusing matched codebook summary: %s", destination)
            continue
        items, sources = {}, []
        for sample in DOMAINS:
            path = reconstruction.cache_path(args.output_dir, plan, run, sample)
            arrays, receipt = reconstruction.read_cache(path, plan, run, sample)
            del arrays
            with np.load(path, allow_pickle=False) as archive:
                items[sample] = archive["indices"]
            sources.extend(
                [
                    receipt["arrays"],
                    reconstruction.capacity.file_record(
                        path.with_suffix(".json"), content_hash=True
                    ),
                ]
            )
        domains = matched_metrics(items, obj, run["k"], args.seed)
        reconstruction.write_json(
            destination,
            reconstruction.seal(
                {
                    "version": VERSION,
                    "plan_id": plan["plan_id"],
                    "object": obj,
                    "q": run["q"],
                    "k": run["k"],
                    "d": run["d"],
                    "matching_seed": args.seed,
                    "matching_rng": "NumPy PCG64/SeedSequence(seed, object_index, domain_index)",
                    "numpy_version": np.__version__,
                    "domains": domains,
                    "sources": sources,
                },
                "summary_id",
            ),
        )
        log.info(
            "%s: %s matched objects/domain (MC %s, data %s available); no inference",
            obj,
            f"{domains['mc']['matched_objects']:,}",
            f"{len(items['mc']):,}",
            f"{len(items['data']):,}",
        )
        del items


def render(summaries):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import NullLocator

    objects = [obj for obj in OBJECTS if obj in summaries]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.7), sharex=True, sharey="row")
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.22, top=0.93, wspace=0.12, hspace=0.18)
    x = np.arange(len(objects))
    width = 0.095
    for column, sample in enumerate(DOMAINS):
        for row, (metric, high) in enumerate((("used_percent", 100), ("normalized_perplexity", 1))):
            ax = axes[row, column]
            for q, color in enumerate(STAGE_COLORS):
                heights = [
                    summaries[obj]["domains"][sample]["metrics"][q][metric] for obj in objects
                ]
                ax.bar(
                    x + (q - 3.5) * width,
                    heights,
                    width=width,
                    color=color,
                    edgecolor="white",
                    linewidth=0.25,
                    zorder=2,
                )
            ax.set_ylim(0, high)
            ax.set_yticks(np.linspace(0, high, 5))
            ax.set_xlim(-0.6, len(objects) - 0.4)
            ax.set_xticks(x, [OBJECT_LABELS[OBJECTS.index(obj)] for obj in objects])
            style_axis(ax)
            ax.xaxis.set_minor_locator(NullLocator())
            if row == 0:
                ax.set_title(f"({chr(97 + column)}) {DOMAINS[sample]}", loc="left")
            else:
                ax.set_xlabel("Object tokenizer")
                ax.text(0.015, 0.95, f"({chr(99 + column)})", transform=ax.transAxes, va="top")
    axes[0, 0].set_ylabel("Used codebook entries [%]")
    axes[1, 0].set_ylabel(r"Normalized perplexity $\mathcal{P}/K$")
    fig.legend(
        handles=[Patch(facecolor=color, label=f"Q{q}") for q, color in enumerate(STAGE_COLORS)],
        title="Residual quantizer stage",
        title_fontsize=10,
        ncol=8,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.035),
        frameon=False,
        columnspacing=1.7,
        handlelength=1.4,
    )
    return fig


def caption(plan, summaries):
    counts = "; ".join(
        f"{OBJECT_LABELS[OBJECTS.index(obj)]}: {summary['domains']['mc']['matched_objects']:,}"
        for obj, summary in summaries.items()
    )
    seed = next(iter(summaries.values()))["matching_seed"]
    return (
        "# Codebook utilization and normalized perplexity\n\n"
        "Selected full Q8/dim8 trainings: K=2048 for jets, electrons, muons and photons; "
        "K=4096 for taus and tracks (K is the number of entries per quantizer). "
        f"Checkpoint: {plan['checkpoint_name']}. {plan['holdout_note']}\n\n"
        "Columns show simulation and collision data. Bars Q0--Q7 denote successive residual "
        "quantizers, with fixed stage colours across all panels. A code is used if assigned "
        "at least once in the matched evaluation sample. Utilization = 100 times the number "
        "of used entries divided by K. For assignment counts n_k, p_k = n_k/N and "
        "normalized perplexity = exp(-sum_{p_k>0} p_k ln p_k)/K, including all K entries "
        "in the normalization, not only the used entries.\n\n"
        f"Equal object counts per domain within each object type: {counts}. "
        "These are object counts, not event counts; they need not match across object types. "
        "For each object type N=min(N_MC,N_data). All rows of the smaller cache are used, "
        f"and the larger cache is sampled uniformly without replacement with matching seed {seed} "
        "(PCG64, independent per-object/domain streams). The same selected rows are used for "
        "every Q stage and both metrics. No kinematic or process reweighting is applied. "
        "All cached valid-object assignments are eligible; no distribution-plot range or "
        "finite-reconstruction cut is applied. Unused means unused in this evaluation sample, "
        "not dead throughout training. No uncertainty bands or repeated-training variation "
        "are inferred. Statistics are computed from cached assignments without new inference.\n"
    )


def plot(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plan = reconstruction.load_plan(args.output_dir)
    summaries = {obj: load_summary(args.output_dir, plan, obj) for obj in args.objects}
    if len({s["matching_seed"] for s in summaries.values()}) != 1:
        raise ValueError("Object summaries use different matching seeds; do not mix them")
    selection = "" if set(summaries) == set(OBJECTS) else "_" + "_".join(summaries)
    stem = args.output_dir / "figures" / f"figure4_codebook_bars{selection}"
    with paper_style():
        fig = render(summaries)
        save_figure(fig, stem)
        plt.close(fig)
    rows = []
    for obj, summary in summaries.items():
        for sample, domain in summary["domains"].items():
            rows.extend(
                {
                    "object": obj,
                    "sample": sample,
                    "codebook_size": summary["k"],
                    "available_objects": domain["available_objects"],
                    "matched_objects": domain["matched_objects"],
                    **metrics,
                }
                for metrics in domain["metrics"]
            )
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    stem.with_suffix(".md").write_text(caption(plan, summaries))
    log.info("Saved %s.[pdf,png,csv,md]; summaries only, no inference", stem)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    summary = commands.add_parser(
        "summarize", help="Verify existing caches and freeze matched counts"
    )
    plotting = commands.add_parser("plot", help="Render grouped bars from summaries only")
    summary.add_argument(
        "--seed", type=int, default=42, help="Matching subsample seed, not a training seed"
    )
    for cmd in (summary, plotting):
        cmd.add_argument(
            "--output-dir",
            type=Path,
            required=True,
            help="Existing full-Q8 reconstruction plan directory",
        )
        cmd.add_argument("--objects", nargs="+", choices=OBJECTS, default=list(OBJECTS))
    args = parser.parse_args(argv)
    if len(set(args.objects)) != len(args.objects):
        parser.error("Duplicate objects are not allowed")
    args.objects = [obj for obj in OBJECTS if obj in args.objects]
    if args.command == "summarize" and args.seed < 0:
        parser.error("Matching seed must be nonnegative")
    return args


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    {"summarize": summarize, "plot": plot}[args.command](args)


if __name__ == "__main__":
    main()
