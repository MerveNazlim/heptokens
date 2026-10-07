#!/usr/bin/env python3
"""Compare completed reset OFF/ON pilots on identical saved MC validation muons."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf

import analyze_vqvae_tokenizer as diagnostics
import muon_q1_init_control as initialization
import muon_q1_mc_only_pilot as mc
import muon_q1_mc_reset_pilot as reset
import paper_tokenizer_reconstruction as reco
from paper_plot_style import paper_style, save_figure, style_axis

log = logging.getLogger(__name__)
ARMS = ("reset_off", "reset_on")
LABELS = {"reset_off": "Reset OFF", "reset_on": "Reset ON"}
COLORS = {"reset_off": "#D55E00", "reset_on": "#235789"}


def audit(args):
    reset_plan, baseline = reset.verify_plan(args.reset_dir.resolve() / "plan.json")
    reset.summarize(argparse.Namespace(output=args.reset_dir))
    roots = {"reset_off": Path(baseline["output"]), "reset_on": Path(reset_plan["output"])}
    plans = {"reset_off": baseline, "reset_on": reset_plan}
    records, configs = [], {}
    for arm in ARMS:
        root, plan = roots[arm], plans[arm]
        initialization.verify_pilot_completion(plan, "data_init")
        configs[arm] = initialization.load_config(plan["configs"]["data_init"]["path"])
        paths = (
            root / "plan.json",
            Path(plan["configs"]["data_init"]["path"]),
            root / "data_init/SUCCESS.txt",
            root / "data_init/pilot_completion.json",
            root / "data_init/checkpoints/pilot_end.ckpt",
            root / "data_init/mc_membership.json",
        )
        records.extend(initialization.file_record(path) for path in paths)
    model_off, model_on = configs["reset_off"]["model"], configs["reset_on"]["model"]
    if initialization.diff_paths(model_off, model_on) != ["dead_code_reset"]:
        raise ValueError("Model settings differ beyond reset OFF/ON")
    if initialization.diff_paths(
        configs["reset_off"]["datamodule"], configs["reset_on"]["datamodule"]
    ) != ["split_audit_path"]:
        raise ValueError("Reset OFF/ON saved input preparation differs")
    fit = mc.verify_preprocessing(baseline)
    for key in ("joblib", "metadata", "receipt"):
        records.append(initialization.file_record(Path(baseline["preprocessing"][key])))
    if not args.input_momentum_unit:
        raise ValueError("Declare --input-momentum-unit GeV or MeV; no automatic unit guessing")
    return {
        "version": "muon-mc-reset-validation-end-e3-v1",
        "sources": records,
        "roots": {k: str(v) for k, v in roots.items()},
        "max_valid_objects": args.max_valid_objects,
        "batch_size": args.batch_size,
        "input_momentum_unit": args.input_momentum_unit,
        "checkpoint": "pilot_end.ckpt",
        "preprocessing": fit,
        "comparison": json.loads((args.reset_dir / "comparison.json").read_text()),
        "sample": "First N valid objects in original MC validation permutation order; no resplit",
        "note": "Identical saved MC validation membership and training-fitted MC joblib. "
        "Not file-disjoint holdout, not independent test evaluation; three-epoch pilot endpoints.",
    }, configs


def check_membership(actual, reference):
    for key in ("mc_files", "seed", "split_fractions"):
        if actual[key] != reference[key]:
            raise ValueError(f"Evaluation membership differs: {key}")
    if actual["partitions"] != reference["partitions"]:
        raise ValueError("Evaluation train/val/test membership differs from training audit")


def validate_pairs(original, items, names):
    if original.ndim != 2 or original.shape[1] != len(names) or not len(original):
        raise ValueError("Empty or invalid original-object arrays")
    for arm in ARMS:
        x, y, codes = items[arm]
        if not np.array_equal(x, original, equal_nan=True) or y.shape != original.shape:
            raise ValueError(f"Original objects/order differ for {arm}")
        if (
            codes.shape != (len(original), 1)
            or codes.dtype.kind not in "iu"
            or np.any((codes < 0) | (codes >= 16384))
        ):
            raise ValueError(f"Expected valid Q1/cb16384 indices for {arm}")


def evaluate(args):
    if args.max_valid_objects <= 0 or args.batch_size <= 0:
        raise ValueError("Object cap and evaluation batch size must be positive")
    manifest, configs = audit(args)
    out = args.output_dir
    roots = [Path(p) for p in manifest["roots"].values()]
    if any(out.resolve() == p or out.resolve() in p.parents for p in roots):
        raise ValueError("Keep plotting output below figures/, not over a training directory")
    manifest_path = out / "evaluation.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(
                "Evaluation settings or inputs changed; existing cache not overwritten"
            )
    elif out.exists() and any(out.iterdir()):
        raise ValueError("Nonempty output without an evaluation audit; nothing overwritten")
    else:
        reco.write_json(manifest_path, manifest)
    arrays_path = out / "paired_muons.npz"
    if arrays_path.exists() or arrays_path.with_suffix(".json").exists():
        read_pairs(out)
        log.info("Reusing paired cache; no inference")
        return
    import torch

    cfg = OmegaConf.create(configs["reset_off"])
    _, inverse = diagnostics.transform_list_and_cst_fn_from_cfg(cfg)
    if inverse is None:
        raise ValueError("Missing inverse MC preprocessor")
    cfg.datamodule.split_audit_path = str(out / "evaluation_membership.json")
    cfg.datamodule.batch_size = args.batch_size
    cfg.datamodule.num_workers = 0
    cfg.datamodule.pin_memory = False
    if "persistent_workers" in cfg.datamodule:
        cfg.datamodule.persistent_workers = False
    if "multiprocessing_context" in cfg.datamodule:
        cfg.datamodule.multiprocessing_context = None
    log.info("Loading saved MC datamodule once; original global split, no file selection/refit")
    dm = hydra.utils.instantiate(cfg.datamodule)
    actual = json.loads((out / "evaluation_membership.json").read_text())
    for root in roots:
        check_membership(actual, json.loads((root / "data_init/mc_membership.json").read_text()))
    loader = diagnostics.dataloader_from_datamodule(dm, "val")
    names = [
        Path(p).name
        for p in next(
            c for c in cfg.datamodule.object_collections if c.object_name == "muons"
        ).inputs
    ]
    device = diagnostics.choose_device(args.device)
    items = {}
    for arm in ARMS:
        run = Path(manifest["roots"][arm]) / "data_init"
        checkpoint = run / "checkpoints/pilot_end.ckpt"
        log.info("Decoding %s endpoint on the same MC validation loader", LABELS[arm])
        model = diagnostics.load_analysis_model(run, str(checkpoint), device)
        if tuple(
            int(getattr(model.hparams, k))
            for k in ("num_quantizers", "codebook_size", "codebook_dim")
        ) != (1, 16384, 8) or bool(model.hparams.dead_code_reset) != (arm == "reset_on"):
            raise ValueError(f"Wrong capacity/reset setting in checkpoint: {checkpoint}")
        if not model.hparams.data_codebook_init:
            raise ValueError(f"Wrong initialization setting in checkpoint: {checkpoint}")
        x, y, indices, _ = diagnostics.collect_diagnostics_from_loader(
            model=model,
            loader=loader,
            cst_inverse_transformer=inverse,
            device=device,
            max_valid_objects=args.max_valid_objects,
        )
        items[arm] = (x, y, indices)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    original = items["reset_off"][0]
    validate_pairs(original, items, names)
    temporary = arrays_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        original=original,
        feature_names=np.asarray(names),
        **{arm: items[arm][1] for arm in ARMS},
        **{f"{arm}_indices": items[arm][2] for arm in ARMS},
    )
    temporary.replace(arrays_path)
    reco.write_json(
        arrays_path.with_suffix(".json"),
        {
            "arrays": initialization.file_record(arrays_path),
            "evaluation": initialization.file_record(manifest_path),
            "membership": initialization.file_record(out / "evaluation_membership.json"),
            "objects": len(original),
        },
    )
    log.info(
        "VERIFIED: %s identical paired MC validation muons; cache=%s",
        f"{len(original):,}",
        arrays_path,
    )


def read_pairs(out):
    receipt = json.loads((out / "paired_muons.json").read_text())
    for key in ("arrays", "evaluation", "membership"):
        initialization.verify_record(receipt[key])
    manifest = json.loads((out / "evaluation.json").read_text())
    for record in manifest["sources"]:
        initialization.verify_record(record)
    with np.load(out / "paired_muons.npz", allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    names = arrays["feature_names"].tolist()
    validate_pairs(
        arrays["original"],
        {arm: (arrays["original"], arrays[arm], arrays[f"{arm}_indices"]) for arm in ARMS},
        names,
    )
    if len(arrays["original"]) != receipt["objects"]:
        raise ValueError("Cache object count differs from receipt")
    return manifest, arrays


def residual(x, y, name):
    delta = y - x
    return (delta + np.pi) % (2 * np.pi) - np.pi if name.lower() == "phi" else delta


def binned_errors(pt, x, y, edges, name, min_count):
    delta = residual(x, y, name)
    result = []
    for i in range(len(edges) - 1):
        keep = (pt >= edges[i]) & (pt < edges[i + 1]) & np.isfinite(x) & np.isfinite(y)
        if i == len(edges) - 2:
            keep |= (pt == edges[i + 1]) & np.isfinite(x) & np.isfinite(y)
        n = int(keep.sum())
        values = {
            "objects": n,
            "median_absolute_residual": None,
            "median_residual": None,
            "residual_iqr": None,
            "response_width_percent": None,
        }
        if n >= min_count:
            r = delta[keep]
            values.update(
                median_absolute_residual=float(np.median(np.abs(r))),
                median_residual=float(np.median(r)),
                residual_iqr=float(np.diff(np.percentile(r, [25, 75]))[0]),
            )
            if name.lower() == "pt":
                valid = keep & (x > 1e-8) & (y > 0)
                ratio = y[valid] / x[valid]
                if len(ratio) >= min_count:
                    values["response_width_percent"] = float(
                        100 * np.diff(np.percentile(ratio, [25, 75]))[0] / np.median(ratio)
                    )
        result.append(values)
    return result


def feature_summaries(manifest, arrays, bins, min_count):
    names = arrays["feature_names"].tolist()
    pt_index = next(i for i, n in enumerate(names) if n.lower() == "pt")
    unit = manifest["input_momentum_unit"]
    pt = arrays["original"][:, pt_index] * reco.display_feature("pt", unit)["scale"]
    pt_edges = reco.bin_edges(pt, 15, (0.5, 99.5))
    features = []
    for i, name in enumerate(names):
        feature = reco.display_feature(name, unit)
        x = arrays["original"][:, i] * feature["scale"]
        ys = {arm: arrays[arm][:, i] * feature["scale"] for arm in ARMS}
        edges = reco.bin_edges(x, bins, (0.5, 99.5), discrete=True)
        deltas = np.concatenate([residual(x, y, name) for y in ys.values()])
        r_edges = reco.bin_edges(deltas, bins, (0.5, 99.5))
        rows = {}
        for arm, y in ys.items():
            row = reco.feature_summary(x, y, edges, r_edges, min_count, name)
            d = residual(x, y, name)
            d = d[np.isfinite(x) & np.isfinite(y)]
            if not len(d):
                raise ValueError(f"No finite reconstructed {name} pairs for {arm}")
            row.update(
                residual_counts=np.histogram(d, r_edges)[0].tolist(),
                median_residual=float(np.median(d)),
                median_absolute_residual=float(np.median(np.abs(d))),
                residual_iqr=float(np.diff(np.percentile(d, [25, 75]))[0]),
                outside_residual=int(len(d) - np.histogram(d, r_edges)[0].sum()),
                binned=binned_errors(pt, x, y, pt_edges, name, min_count),
                empty_decoded_bins=int(
                    np.count_nonzero(
                        (np.asarray(row["original_counts"]) >= min_count)
                        & (np.asarray(row["decoded_counts"]) == 0)
                    )
                ),
            )
            rows[arm] = row
        feature.update(edges=edges.tolist(), residual_edges=r_edges.tolist(), rows=rows)
        features.append(feature)
    return {
        "objects": len(pt),
        "features": features,
        "pt_edges": pt_edges.tolist(),
        "min_count": min_count,
        "histogram_bins": bins,
    }


def summaries(manifest, arrays, bins, min_count):
    summary = feature_summaries(manifest, arrays, bins, min_count)
    usage = {}
    for arm in ARMS:
        counts = diagnostics.codebook_counts(arrays[f"{arm}_indices"], 16384)[0]
        p = counts[counts > 0] / counts.sum()
        perplexity = float(np.exp(-np.sum(p * np.log(p))))
        usage[arm] = {
            **diagnostics.codebook_summary(counts[None])["quantizer_0"],
            "perplexity": perplexity,
            "normalized_perplexity": perplexity / 16384,
            "top10_percent": float(100 * np.sort(counts)[-10:].sum() / counts.sum()),
            "sorted_counts": np.sort(counts)[::-1].tolist(),
        }
    return {**summary, "codebooks": usage}


def draw_distribution(fig, cell, feature):
    inner = cell.subgridspec(2, 1, height_ratios=(3.3, 1), hspace=0.04)
    ax = fig.add_subplot(inner[0])
    ratio_ax = fig.add_subplot(inner[1], sharex=ax)
    edges = np.asarray(feature["edges"])
    rows = feature["rows"]
    first = rows[ARMS[0]]
    n = max(first["finite_pairs"], 1)
    ax.stairs(
        np.asarray(first["original_counts"]) / n, edges, fill=True, color="#C5C9CD", alpha=0.45
    )
    ax.stairs(np.asarray(first["original_counts"]) / n, edges, color="#404040", label="Original")
    for arm in ARMS:
        row = rows[arm]
        ax.stairs(
            np.asarray(row["decoded_counts"]) / max(row["finite_pairs"], 1),
            edges,
            color=COLORS[arm],
            label=LABELS[arm],
        )
        ratio_ax.plot(
            (edges[1:] + edges[:-1]) / 2,
            [np.nan if r is None else r for r in row["bin_count_ratio"]],
            ".-",
            color=COLORS[arm],
            ms=2.5,
            lw=0.8,
        )
    ax.set_yscale("log")
    ax.set_ylabel("Normalized objects")
    ax.tick_params(labelbottom=False)
    ratio_ax.axhline(1, color="#404040", ls="--", lw=0.9)
    ratio_ax.set_ylabel("Decoded /\nOriginal", fontsize=8)
    ratio_ax.set_xlabel(reco.axis_label(feature))
    ratio_ax.set_xlim(edges[0], edges[-1])
    style_axis(ax)
    style_axis(ratio_ax)
    return ax


def draw_residual(ax, feature):
    for arm in ARMS:
        row = feature["rows"][arm]
        ax.stairs(
            np.asarray(row["residual_counts"]) / max(row["finite_pairs"], 1),
            feature["residual_edges"],
            color=COLORS[arm],
            label=LABELS[arm],
        )
    ax.axvline(0, color="#666666", ls="--", lw=0.8)
    ax.set_xlabel(reco.axis_label(feature, "Decoded - original "))
    ax.set_ylabel("Normalized objects")
    ax.set_title("Residual", loc="left")
    annotations = []
    for arm in ARMS:
        row = feature["rows"][arm]
        annotations.append(
            f"{LABELS[arm]}: median={row['median_residual']:.3g}; "
            f"median |res.|={row['median_absolute_residual']:.3g}"
        )
    ax.text(
        0.98,
        0.97,
        "\n".join(annotations),
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    style_axis(ax)


def draw_binned(ax, feature, summary, metric="median_absolute_residual"):
    edges = np.asarray(summary["pt_edges"])
    for arm in ARMS:
        values = [r[metric] for r in feature["rows"][arm]["binned"]]
        ax.plot(
            (edges[1:] + edges[:-1]) / 2,
            [np.nan if v is None else v for v in values],
            "o-",
            ms=3,
            color=COLORS[arm],
            label=LABELS[arm],
        )
    ax.set_xlabel(r"Original $p_T$ [GeV]")
    if metric == "response_width_percent":
        ax.set_ylabel(r"Response width $R_{p_T}$ [%]")
    else:
        ax.set_ylabel(
            "Median |decoded - original|" + (f" [{feature['unit']}]" if feature["unit"] else "")
        )
    ax.set_title("Binned reconstruction error", loc="left")
    style_axis(ax)


def render_reconstruction(summary, out, heading=None):
    import matplotlib.pyplot as plt

    with paper_style():
        fig = plt.figure(figsize=(15, 3.3 * len(summary["features"])))
        grid = fig.add_gridspec(
            len(summary["features"]),
            3,
            left=0.065,
            right=0.985,
            top=0.96,
            bottom=0.04,
            hspace=0.6,
            wspace=0.32,
        )
        for i, feature in enumerate(summary["features"]):
            ax = draw_distribution(fig, grid[i, 0], feature)
            ax.set_title(feature["label"], loc="left")
            draw_residual(fig.add_subplot(grid[i, 1]), feature)
            draw_binned(fig.add_subplot(grid[i, 2]), feature, summary)
            if i == 0:
                handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False)
        fig.text(
            0.065,
            0.995,
            heading or f"Muons | MC validation | end of epoch 3 | N={summary['objects']:,}",
            va="top",
        )
        save_figure(fig, out / "muons_all_features")
        plt.close(fig)
        pt = next(f for f in summary["features"] if f["name"].lower() == "pt")
        fig = plt.figure(figsize=(11, 8))
        grid = fig.add_gridspec(
            2, 2, left=0.085, right=0.98, top=0.94, bottom=0.085, hspace=0.48, wspace=0.32
        )
        draw_distribution(fig, grid[0, 0], pt).set_title(r"Muon $p_T$", loc="left")
        draw_residual(fig.add_subplot(grid[0, 1]), pt)
        draw_binned(fig.add_subplot(grid[1, 0]), pt, summary, "response_width_percent")
        draw_binned(fig.add_subplot(grid[1, 1]), pt, summary)
        fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False)
        save_figure(fig, out / "muons_pt_comparison")
        plt.close(fig)


def render(summary, comparison, out):
    import matplotlib.pyplot as plt

    render_reconstruction(summary, out)
    with paper_style():
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout="constrained")
        for ax, key, label, limit in (
            (axes[0, 0], "percent_used", "Used entries [%]", 100),
            (axes[0, 1], "normalized_perplexity", r"Normalized perplexity $\mathcal{P}/K$", 1),
        ):
            values = [summary["codebooks"][a][key] for a in ARMS]
            ax.bar([LABELS[a] for a in ARMS], values, color=[COLORS[a] for a in ARMS])
            ax.set_ylim(0, limit)
            ax.set_ylabel(label)
            for i, v in enumerate(values):
                ax.text(i, min(limit * 0.96, v + limit * 0.02), f"{v:.4g}", ha="center")
        for arm in ARMS:
            counts = np.asarray(summary["codebooks"][arm]["sorted_counts"])
            positive = counts[counts > 0]
            axes[1, 0].plot(
                np.arange(1, len(positive) + 1),
                positive / counts.sum(),
                color=COLORS[arm],
                label=LABELS[arm],
            )
            rows = comparison["paired_validation_passes"]
            axes[1, 1].plot(
                [r["step"] for r in rows],
                [r[arm]["validation_metrics"]["val/recon_loss"] for r in rows],
                color=COLORS[arm],
                label=LABELS[arm],
            )
        axes[1, 0].set(
            xscale="log",
            yscale="log",
            xlabel="Code rank (sorted independently)",
            ylabel="Assignment fraction",
        )
        axes[1, 1].set(xlabel="Training step", ylabel="Validation reconstruction loss")
        axes[1, 0].legend(frameon=False)
        axes[1, 1].legend(frameon=False)
        for ax in axes.flat:
            style_axis(ax)
        save_figure(fig, out / "muons_codebooks")
        plt.close(fig)


def plot(args):
    manifest, arrays = read_pairs(args.output_dir)
    summary = summaries(manifest, arrays, args.bins, args.min_bin_count)
    reco.write_json(args.output_dir / "summary.json", summary)
    render(summary, manifest["comparison"], args.output_dir)
    pt = next(f for f in summary["features"] if f["name"].lower() == "pt")
    for arm in ARMS:
        usage, row = summary["codebooks"][arm], pt["rows"][arm]
        log.info(
            "%s: N=%s; used=%s/16384 (%.3f%%); P/K=%.5f; pT R=%s%%; empty decoded pT bins=%s",
            LABELS[arm],
            f"{summary['objects']:,}",
            usage["used_codes"],
            usage["percent_used"],
            usage["normalized_perplexity"],
            row.get("response_width_percent"),
            row["empty_decoded_bins"],
        )
    (args.output_dir / "README.md").write_text(
        "# Muon Q1 MC-only reset comparison\n\n" + manifest["note"] + "\n\n"
        f"Both models: Q1/cb16384/dim8, data initialization ON; {manifest['checkpoint']}. "
        "The saved MC datamodule is constructed once. Both models decode the same ordered "
        f"{summary['objects']:,} validation muons, with the exact shared MC-training-fitted joblib. "
        "No refitting, training, sample replacement or new event splitting.\n\n"
        "Original means detector-reconstructed input, inverse-transformed from the canonical "
        "preprocessed batch; not generator truth. Histograms share original-derived bins and "
        "normalize by all finite pairs, including out-of-window objects. Ratios are decoded/original "
        f"bin counts; bins below {args.min_bin_count} original entries are omitted. "
        "Phi residuals are wrapped to [-pi,pi). Other residuals are decoded minus original. "
        "Binned median absolute errors are plotted against original pT. "
        "R_pT = 100 IQR(decoded/original pT)/median(decoded/original pT), using positive pT pairs. "
        "Summary JSON reports counts, exclusions and out-of-range objects.\n\n"
        "Used means assigned at least once in this evaluation sample, not alive throughout training. "
        "P/K = exp(-sum(p log p))/16384. Occupancy alone is not reconstruction quality. "
        "These are early three-epoch results; no seed or uncertainty bands are implied.\n"
    )
    log.info(
        "Plots: %s/{muons_pt_comparison,muons_all_features,muons_codebooks}.{png,pdf}",
        args.output_dir,
    )


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "plot"))
    parser.add_argument(
        "--reset-dir", type=Path, default=root / "results/atlas_muon_q1_mc_only_reset_pilot_e3_v1"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-valid-objects", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--input-momentum-unit", choices=("GeV", "MeV"))
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument("--min-bin-count", type=int, default=20)
    args = parser.parse_args()
    args.reset_dir = args.reset_dir.resolve()
    args.output_dir = (args.output_dir or args.reset_dir / "figures/mc_validation_end_e3").resolve()
    if args.bins < 2 or args.min_bin_count < 1:
        parser.error("Use at least two bins and a positive minimum bin count")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.command == "run":
        evaluate(args)
    plot(args)


if __name__ == "__main__":
    main()
