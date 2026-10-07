#!/usr/bin/env python3
"""Partial decodes of the frozen full-Q8 reconstruction sample, without re-encoding.

Read the existing audited code indices and decode prefixes of length 1/2/4/8.
Only small summaries are saved alongside the original reconstruction outputs.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
from pathlib import Path

import numpy as np

import paper_tokenizer_reconstruction as reco
from paper_plot_style import (
    OBJECTS,
    OBJECT_LABELS,
    OBJECT_COLORS,
    PT_LABEL,
    RESPONSE_LABEL,
    paper_style,
    save_figure,
    style_axis,
)
from paper_tokenizer_codebooks import selected_run, validate_indices

log = logging.getLogger(__name__)
VERSION = "paper-full-q8-stagewise-v1"
DEPTHS = (1, 2, 4, 8)
STAGE_LABELS = ("Q0", "Q0-Q1", "Q0-Q3", "Q0-Q7")
STAGE_COLORS = ("#9ECAE1", "#4292C6", "#2171B5", "#08306B")
DOMAINS = reco.DOMAINS
FULL_RTOL, FULL_ATOL = 1e-4, 1e-5


def settings(args):
    return {
        "depths": list(DEPTHS),
        "pt_bins": args.pt_bins,
        "min_bin_count": args.min_bin_count,
        "pt_percentiles": [1, 99],
        "min_original_pt_gev": 1e-8,
        "full_decode_rtol": FULL_RTOL,
        "full_decode_atol_native": FULL_ATOL,
    }


def summary_path(directory, obj, sample):
    return directory / "stagewise_summaries" / obj / f"{sample}.json"


def load_summary(directory, plan, obj, sample, *, verify_source=False):
    path = summary_path(directory, obj, sample)
    result = reco.read_sealed(path, "summary_id")
    if (result["version"], result["plan_id"], result["object"], result["sample"]) != (
        VERSION,
        plan["plan_id"],
        obj,
        sample,
    ):
        raise ValueError(f"Wrong stagewise summary provenance: {path}")
    if verify_source:
        source = result["source"]
        expected = reco.cache_path(directory, plan, selected_run(plan, obj), sample)
        if source["arrays"]["path"] != str(expected.resolve()):
            raise ValueError(f"Wrong source cache: {path}")
        receipt = reco.read_sealed(expected.with_suffix(".json"), "receipt_id")
        if receipt != source:
            raise ValueError(f"Source receipt changed: {path}")
        reco.capacity.verify_file(source["arrays"])
    return result


def load_bundle(run, device):
    import torch
    from omegaconf import OmegaConf
    from analyze_vqvae_tokenizer import load_analysis_model, transform_list_and_cst_fn_from_cfg

    cfg = OmegaConf.load(run["config"]["path"])
    OmegaConf.update(
        cfg, "datamodule.transforms.preprocess.cst_fn.filename", run["preprocessor"]["path"]
    )
    _, inverse = transform_list_and_cst_fn_from_cfg(cfg)
    if inverse is None:
        raise ValueError("Missing inverse preprocessor; no substitution or refitting allowed")
    model = load_analysis_model(
        Path(run["run_dir"]), run["checkpoint"]["path"], torch.device(device)
    )
    actual = tuple(
        int(getattr(model.hparams, key))
        for key in (
            "num_quantizers",
            "codebook_size",
            "codebook_dim",
        )
    )
    if actual != (run["q"], run["k"], run["d"]):
        raise ValueError(f"Checkpoint capacity {actual} differs from audited full Q8 run")
    return model.eval(), inverse


def decode_prefixes(model, indices, inverse, n_features, batch_size, device):
    """Omitted levels contribute zero vectors, never codebook entry zero."""
    import torch

    validate_indices(indices, int(model.hparams.codebook_size))
    if batch_size < 1 or model.training:
        raise ValueError("Positive decode batch size and eval-mode model required")
    vq = model.vector_quantization
    layers = getattr(vq, "layers", ())
    if len(layers) != 8:
        raise ValueError("Expected the eight layers of the audited ResidualVQ")
    # The production model sums unprojected code vectors. Refuse a different VQ architecture.
    for module in (vq, *layers):
        for key in ("project_in", "project_out"):
            projection = getattr(module, key, None)
            if projection is not None and not isinstance(projection, torch.nn.Identity):
                raise ValueError(f"Unsupported quantizer {key}; cannot assume a raw latent sum")
    books = [getattr(layer, "codebook", None) for layer in layers]
    expected = (int(model.hparams.codebook_size), int(model.hparams.codebook_dim))
    if any(not torch.is_tensor(book) or tuple(book.shape) != expected for book in books):
        raise ValueError(f"Expected one (K, d_c) codebook per level: {expected}")
    result = None
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            codes = torch.as_tensor(
                indices[start : start + batch_size], dtype=torch.long, device=device
            )
            mask = torch.ones((len(codes), 1), dtype=torch.bool, device=device)
            batch = {"mask": mask}
            latent = torch.zeros((len(codes), 1, expected[1]), dtype=books[0].dtype, device=device)
            for level, book in enumerate(books):
                latent = latent + book[codes[:, level]][:, None, :]
                depth = level + 1
                if depth not in DEPTHS:
                    continue
                decoded = model.decode(latent, batch)[mask].cpu().float().numpy()
                if decoded.shape != (len(codes), n_features):
                    raise ValueError("Decoder output does not match saved feature order/shape")
                with np.errstate(over="ignore", invalid="ignore"):
                    physical = np.asarray(inverse.inverse_transform(decoded))
                if result is None:
                    result = np.empty((4, len(indices), n_features), dtype=physical.dtype)
                result[DEPTHS.index(depth), start : start + len(codes)] = physical
            end = start + len(codes)
            if end // 100000 > start // 100000 or end == len(indices):
                log.info(
                    "Decoded all four prefixes: %s / %s objects", f"{end:,}", f"{len(indices):,}"
                )
    return result


def verify_full_decode(decoded, cached, names):
    if decoded.shape != cached.shape or decoded.shape[1] != len(names):
        raise ValueError("Full-decode check has inconsistent shapes")
    matching = np.isclose(decoded, cached, rtol=FULL_RTOL, atol=FULL_ATOL, equal_nan=True)
    failures = {
        name: int((~matching[:, i]).sum())
        for i, name in enumerate(names)
        if not matching[:, i].all()
    }
    if failures:
        raise ValueError(
            f"Eight-code decode does not reproduce the existing cache: {failures}. "
            "No stagewise summary written; check checkpoint, preprocessing and decoder."
        )
    differences = []
    for i in range(len(names)):
        finite = np.isfinite(decoded[:, i]) & np.isfinite(cached[:, i])
        differences.append(
            float(np.max(np.abs(decoded[finite, i] - cached[finite, i]))) if finite.any() else None
        )
    return {
        "objects_checked": len(cached),
        "rtol": FULL_RTOL,
        "atol_native": FULL_ATOL,
        "max_absolute_difference_native": dict(zip(names, differences)),
    }


def metrics(x, y, *, pt, phi, min_count):
    n = len(x)
    result = dict.fromkeys(
        (
            "median_residual",
            "median_absolute_residual",
            "mean_absolute_residual",
            "residual_iqr",
            "response_width_percent",
            "median_relative_residual_percent",
        )
    )
    result.update(
        objects=n, response_objects=0, excluded_response_objects=0, nonpositive_decoded_pt=0
    )
    if n >= min_count:
        delta = y - x
        if phi:
            delta = (delta + np.pi) % (2 * np.pi) - np.pi
        result.update(
            median_residual=float(np.median(delta)),
            median_absolute_residual=float(np.median(np.abs(delta))),
            mean_absolute_residual=float(np.mean(np.abs(delta))),
            residual_iqr=float(np.subtract(*np.percentile(delta, [75, 25]))),
        )
    if pt:
        valid = x > 1e-8
        result.update(
            response_objects=int(valid.sum()),
            excluded_response_objects=int((~valid).sum()),
            nonpositive_decoded_pt=int((y[valid] <= 0).sum()),
        )
        if valid.sum() >= min_count:
            response = y[valid] / x[valid]
            median = float(np.median(response))
            result["median_relative_residual_percent"] = float(100 * (median - 1))
            if median > 0:
                result["response_width_percent"] = float(
                    100 * np.subtract(*np.percentile(response, [75, 25])) / median
                )
    return result


def summarize_arrays(original, stages, plan, run, options):
    if stages.shape != (4, *original.shape) or original.shape[1] != len(run["feature_names"]):
        raise ValueError("Partial decodes must preserve every source row and feature")
    features = []
    for i, name in enumerate(run["feature_names"]):
        feature = reco.display_feature(name, plan["input_momentum_unit"])
        if name.lower() == "phi":
            feature["unit"] = "rad"
        x = np.asarray(original[:, i], dtype=float) * feature["scale"]
        ys = np.asarray(stages[:, :, i], dtype=float) * feature["scale"]
        finite = np.isfinite(x) & np.isfinite(ys).all(axis=0)
        feature.update(
            source_objects=len(x),
            common_finite_objects=int(finite.sum()),
            excluded_nonfinite_objects=int((~finite).sum()),
            nonfinite_original_objects=int((~np.isfinite(x)).sum()),
            nonfinite_decoded_objects=[int((~np.isfinite(y)).sum()) for y in ys],
        )
        pt, phi = name.lower() == "pt", name.lower() == "phi"
        feature["stages"] = [
            {"depth": depth, **metrics(x[finite], y[finite], pt=pt, phi=phi, min_count=1)}
            for depth, y in zip(DEPTHS, ys)
        ]
        if pt:
            usable = finite & (x > options["min_original_pt_gev"])
            edges = reco.bin_edges(x[usable], options["pt_bins"], options["pt_percentiles"])
            feature["pt_bin_edges"] = edges.tolist()
            feature["pt_outside_bins"] = int((usable & ((x < edges[0]) | (x > edges[-1]))).sum())
            rows = []
            for b, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
                inside = usable & (x >= low) & ((x <= high) if b == len(edges) - 2 else (x < high))
                for depth, y in zip(DEPTHS, ys):
                    rows.append(
                        {
                            "bin": b,
                            "bin_low": float(low),
                            "bin_high": float(high),
                            "x": float((low + high) / 2),
                            "depth": depth,
                            **metrics(
                                x[inside],
                                y[inside],
                                pt=True,
                                phi=False,
                                min_count=options["min_bin_count"],
                            ),
                        }
                    )
            feature["pt_bins"] = rows
        features.append(feature)
    return features


def evaluate(args):
    plan = reco.load_plan(args.output_dir)
    options = settings(args)
    for obj in args.objects:
        run = selected_run(plan, obj)
        bundle = None
        try:
            for sample in args.samples:
                destination = summary_path(args.output_dir, obj, sample)
                if destination.exists():
                    existing = load_summary(args.output_dir, plan, obj, sample, verify_source=True)
                    if existing["settings"] != options:
                        raise ValueError(
                            f"Frozen stagewise settings differ: {destination}; refusing overwrite"
                        )
                    log.info("Reusing verified stagewise summary: %s", destination)
                    continue
                path = reco.cache_path(args.output_dir, plan, run, sample)
                if not path.exists() or not path.with_suffix(".json").exists():
                    raise FileNotFoundError(
                        f"Missing audited reconstruction cache: {path}; no sample substitution"
                    )
                arrays, receipt = reco.read_cache(path, plan, run, sample)
                with np.load(path, allow_pickle=False) as loaded:
                    indices = loaded["indices"]
                validate_indices(indices, run["k"])
                if len(indices) != len(arrays["original"]):
                    raise ValueError("Code indices and original rows differ")
                if bundle is None:
                    for key in (
                        "config",
                        "completion",
                        "checkpoint",
                        "preprocessor",
                        "preprocessor_metadata",
                    ):
                        reco.capacity.verify_file(run[key])
                    bundle = load_bundle(run, args.device)
                log.info(
                    "%s/%s: decoding 1/2/4/8 codes for %s cached objects; no H5 or encoder",
                    obj,
                    sample,
                    f"{len(indices):,}",
                )
                decoded = decode_prefixes(
                    bundle[0],
                    indices,
                    bundle[1],
                    len(run["feature_names"]),
                    args.decode_batch_size,
                    args.device,
                )
                verification = verify_full_decode(
                    decoded[-1], arrays["reconstruction"], run["feature_names"]
                )
                features = summarize_arrays(arrays["original"], decoded, plan, run, options)
                reco.write_json(
                    destination,
                    reco.seal(
                        {
                            "version": VERSION,
                            "plan_id": plan["plan_id"],
                            "object": obj,
                            "sample": sample,
                            "settings": options,
                            "source": receipt,
                            "features": features,
                            "full_decode_check": verification,
                            "holdout_note": plan["holdout_note"],
                            "execution": {
                                "device": args.device,
                                "decode_batch_size": args.decode_batch_size,
                            },
                        },
                        "summary_id",
                    ),
                )
                log.info("All-eight-code check passed; saved %s", destination)
                del arrays, indices, decoded
        finally:
            if bundle is not None:
                del bundle
                if args.device.startswith("cuda"):
                    import torch

                    torch.cuda.empty_cache()


def number_values(rows, key):
    return [row[key] if row[key] is not None else np.nan for row in rows]


def logarithmic_width_axis(ax):
    positive = any(
        np.any(np.isfinite(line.get_ydata()) & (np.asarray(line.get_ydata()) > 0))
        for line in ax.lines
    )
    if not positive:
        ax.set_ylim(0.1, 1)
        ax.text(
            0.5,
            0.5,
            "No positive finite values",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#666666",
            fontsize=8,
        )
    ax.set_yscale("log", nonpositive="mask")


def stage_axis(ax, *, log_y=False):
    style_axis(ax)
    ax.set_xticks(range(4), STAGE_LABELS)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlim(-0.15, 3.15)
    if log_y:
        logarithmic_width_axis(ax)


def plot_summary(summaries, objects, samples, destination, log_y):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(samples), figsize=(6 * len(samples), 7.2), squeeze=False)
    for col, sample in enumerate(samples):
        for obj in objects:
            feature = next(
                f for f in summaries[obj, sample]["features"] if f["name"].lower() == "pt"
            )
            for row, key in enumerate(("response_width_percent", "median_absolute_residual")):
                axes[row, col].plot(
                    range(4),
                    number_values(feature["stages"], key),
                    color=OBJECT_COLORS[obj],
                    marker="o",
                    markersize=4,
                    label=OBJECT_LABELS[OBJECTS.index(obj)],
                )
        axes[0, col].set_title(DOMAINS[sample], loc="left")
        for row in range(2):
            stage_axis(axes[row, col], log_y=log_y)
        axes[1, col].set_xlabel("Cumulative codes decoded")
    axes[0, 0].set_ylabel(RESPONSE_LABEL)
    axes[1, 0].set_ylabel(r"Median $|p_T^{\mathrm{decoded}}-p_T^{\mathrm{original}}|$ [GeV]")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.15, top=0.94, hspace=0.30, wspace=0.25)
    save_figure(fig, destination / "stagewise_pt_summary")
    plt.close(fig)


def plot_binned(summaries, objects, samples, destination, log_y):
    import matplotlib.pyplot as plt

    for sample in samples:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7.4))
        for ax, obj in zip(axes.flat, OBJECTS):
            if obj not in objects:
                ax.set_visible(False)
                continue
            feature = next(
                f for f in summaries[obj, sample]["features"] if f["name"].lower() == "pt"
            )
            for depth, label, color in zip(DEPTHS, STAGE_LABELS, STAGE_COLORS):
                rows = [r for r in feature["pt_bins"] if r["depth"] == depth]
                ax.plot(
                    [r["x"] for r in rows],
                    number_values(rows, "response_width_percent"),
                    color=color,
                    marker="o",
                    markersize=3,
                    label=label,
                )
            ax.set_title(OBJECT_LABELS[OBJECTS.index(obj)], loc="left")
            style_axis(ax)
            if log_y:
                logarithmic_width_axis(ax)
        for ax in axes[-1]:
            ax.set_xlabel(PT_LABEL)
        for ax in axes[:, 0]:
            ax.set_ylabel(RESPONSE_LABEL)
        handles, labels = next(
            ax for ax in axes.flat if ax.get_visible()
        ).get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            title=DOMAINS[sample] + " | Cumulative codes decoded",
            loc="lower center",
            ncol=4,
            frameon=False,
        )
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.95, wspace=0.32, hspace=0.32)
        save_figure(fig, destination / f"stagewise_pt_binned_{sample}")
        plt.close(fig)


def plot_features(summaries, objects, samples, destination, log_y):
    import matplotlib.pyplot as plt

    for obj in objects:
        names = [f["name"] for f in summaries[obj, samples[0]]["features"]]
        for page in range(math.ceil(len(names) / 5)):
            subset = names[page * 5 : (page + 1) * 5]
            fig, axes = plt.subplots(
                len(subset), 3, figsize=(12, 2.8 * len(subset) + 0.8), squeeze=False
            )
            for row, name in enumerate(subset):
                for sample in samples:
                    feature = next(
                        f for f in summaries[obj, sample]["features"] if f["name"] == name
                    )
                    keys = (
                        "response_width_percent" if name.lower() == "pt" else "residual_iqr",
                        "median_absolute_residual",
                        "median_residual",
                    )
                    unit = f" [{feature['unit']}]" if feature["unit"] else ""
                    labels = (
                        RESPONSE_LABEL if name.lower() == "pt" else "Residual IQR" + unit,
                        "Median absolute residual" + unit,
                        "Median residual" + unit,
                    )
                    for col, (key, ylabel) in enumerate(zip(keys, labels)):
                        axes[row, col].plot(
                            range(4),
                            number_values(feature["stages"], key),
                            color=OBJECT_COLORS[obj],
                            linestyle="-" if sample == "mc" else "--",
                            marker="o" if sample == "mc" else "s",
                            markersize=4,
                            label=DOMAINS[sample],
                        )
                        axes[row, col].set_ylabel(ylabel)
                        axes[row, col].set_title(feature["label"], loc="left")
                for col, ax in enumerate(axes[row]):
                    stage_axis(ax, log_y=log_y and col != 2)
                    if col == 2:
                        ax.axhline(0, color="#777777", ls=":", lw=0.8)
            fig.suptitle(
                OBJECT_LABELS[OBJECTS.index(obj)]
                + " | Partial decoding of the same full Q8 tokenizer",
                fontsize=11,
            )
            fig.legend(
                *axes[0, 0].get_legend_handles_labels(),
                loc="lower center",
                ncol=len(samples),
                frameon=False,
            )
            fig.tight_layout(rect=(0, 0.04, 1, 0.97), h_pad=1.4, w_pad=2)
            save_figure(fig, destination / f"stagewise_{obj}_features_page{page + 1}")
            plt.close(fig)


def plot(args):
    import matplotlib

    matplotlib.use("Agg")

    plan = reco.load_plan(args.output_dir)
    summaries = {}
    for obj in args.objects:
        run = selected_run(plan, obj)
        for sample in args.samples:
            summary = load_summary(args.output_dir, plan, obj, sample)
            if [f["name"] for f in summary["features"]] != run["feature_names"]:
                raise ValueError("Stagewise feature order differs from frozen run")
            summaries[obj, sample] = summary
    if len({reco.capacity.digest_json(s["settings"]) for s in summaries.values()}) != 1:
        raise ValueError("Mixed stagewise statistical settings; cannot combine summaries")
    selection = "" if tuple(args.objects) == OBJECTS else "_" + "_".join(args.objects)
    domains = "" if tuple(args.samples) == tuple(DOMAINS) else "_" + "_".join(args.samples)
    destination = (
        args.output_dir / "figures" / ("stagewise" + selection + domains + "_" + args.yscale)
    )
    destination.mkdir(parents=True, exist_ok=True)
    with paper_style():
        plot_summary(summaries, args.objects, args.samples, destination, args.yscale == "log")
        plot_binned(summaries, args.objects, args.samples, destination, args.yscale == "log")
        plot_features(summaries, args.objects, args.samples, destination, args.yscale == "log")
    rows = []
    for (obj, sample), summary in summaries.items():
        for feature in summary["features"]:
            for row in feature["stages"] + feature.get("pt_bins", []):
                rows.append(
                    {
                        "object": obj,
                        "sample": sample,
                        "feature": feature["name"],
                        "unit": feature["unit"],
                        "source_objects": feature["source_objects"],
                        "common_finite_objects": feature["common_finite_objects"],
                        "excluded_nonfinite_objects": feature["excluded_nonfinite_objects"],
                        "bin": -1,
                        **row,
                    }
                )
    with (destination / "stagewise_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    reco.write_json(
        destination / "sources.json",
        {
            "plan_id": plan["plan_id"],
            "summaries": {f"{o}/{s}": value["summary_id"] for (o, s), value in summaries.items()},
        },
    )
    (destination / "definitions.md").write_text(
        "# Stagewise full-Q8 reconstruction\n\n"
        "One fixed production tokenizer per object: sum its first 1, 2, 4 or 8 code vectors, "
        "then apply the same decoder and saved inverse preprocessor. Omitted code vectors are zero, "
        "not entry 0. No new encoding, training, H5 selection or event-model prediction. "
        "The full eight-code decode was checked against every cached reconstruction row.\n\n"
        "For each object/domain/feature, the finite-row intersection across all four depths is used. "
        "Nonfinite counts at each depth and common exclusions are recorded in stagewise_summaries. "
        "MC and data retain their own cached populations, not equalized or reweighted samples. "
        "No pT window cuts the pooled metrics. pT bin edges use original 1st--99th percentiles; "
        "the same edges and object rows apply at every depth. Counts and low-count gaps are recorded.\n\n"
        "R_pT [%] = 100 * IQR(decoded/original) / median(decoded/original). Only original pT > "
        "1e-8 GeV is eligible; nonpositive decoded values are counted, not removed. Width is "
        "undefined for a nonpositive median response. Other widths are residual IQR in native/physical "
        "units. Bias is median(decoded-original); typical error is median(abs(residual)), not mean "
        "absolute error (also exported in the CSV). Phi residuals wrap to [-pi, pi). "
        "Categorical values are not rounded and numeric errors are not classification accuracy.\n\n"
        "These are partial decodes of one Q8 model, NOT separately trained Q1/Q2/Q4 models or "
        "event-transformer input ablations. The decoder was trained on the full latent sum; "
        "partial-decoding errors need not improve monotonically. No uncertainty bands or seed variation "
        "are inferred. On logarithmic axes zero/undefined metrics are not drawn; the CSV preserves them.\n\n"
        + "Evaluation caveat: "
        + plan["holdout_note"]
        + "\n"
    )
    log.info("Wrote stagewise figures, metrics and definitions: %s", destination)


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("evaluate", "plot"):
        p = sub.add_parser(command)
        p.add_argument(
            "--output-dir",
            type=Path,
            required=True,
            help="Existing full-Q8 reconstruction root containing plan.json",
        )
        p.add_argument("--objects", nargs="+", choices=OBJECTS, default=list(OBJECTS))
        p.add_argument("--samples", nargs="+", choices=DOMAINS, default=list(DOMAINS))
        if command == "evaluate":
            p.add_argument(
                "--device", default="cpu", help="cpu, cuda, or cuda:0 in the visible GPU set"
            )
            p.add_argument("--decode-batch-size", type=positive_int, default=4096)
            p.add_argument("--pt-bins", type=positive_int, default=14)
            p.add_argument("--min-bin-count", type=positive_int, default=50)
        else:
            p.add_argument("--yscale", choices=("linear", "log"), default="log")
    args = parser.parse_args()
    args.objects = [o for o in OBJECTS if o in args.objects]
    args.samples = [s for s in DOMAINS if s in args.samples]
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    {"evaluate": evaluate, "plot": plot}[args.command](args)
