"""Paired Q1/Q8 physical-feature errors from existing masked prediction caches."""

import csv
import inspect
import math
from pathlib import Path

import numpy as np

import paper_masked_object_reconstruction as paper
from paper_masked_originals import verify_original
from paper_plot_style import QUANTIZER_COLORS, RESPONSE_LABEL, style_axis

VERSION = "masked-original-comparison-v1"
METRICS = (
    "bias",
    "residual_iqr",
    "median_absolute_error",
    "mean_absolute_error",
    "p90_absolute_error",
    "response_width_percent",
)


def residual(original, predicted, name):
    difference = predicted - original
    if name.lower() == "phi":
        difference = (difference + np.pi) % (2 * np.pi) - np.pi
    return difference


def metrics(original, predicted, name):
    delta = residual(original, predicted, name)
    result = dict.fromkeys(METRICS)
    result.update(count=len(delta), response_count=0, nonpositive_predictions=0)
    if not len(delta):
        return result
    absolute = np.abs(delta)
    result.update(
        bias=float(np.median(delta)),
        residual_iqr=float(np.diff(np.percentile(delta, [25, 75]))[0]),
        median_absolute_error=float(np.median(absolute)),
        mean_absolute_error=float(np.mean(absolute)),
        p90_absolute_error=float(np.percentile(absolute, 90)),
    )
    if name.lower() == "pt":
        # Selection must not depend on which model produced a positive prediction.
        valid = original > 1e-8
        response = predicted[valid] / original[valid]
        result["response_count"] = int(valid.sum())
        result["nonpositive_predictions"] = int((predicted <= 0).sum())
        median = float(np.median(response)) if len(response) else 0.0
        if median > 0:
            result["response_width_percent"] = float(
                100 * np.diff(np.percentile(response, [25, 75]))[0] / median
            )
    return result


def discrete_input(values):
    return bool(
        len(values)
        and np.allclose(values, np.rint(values), atol=1e-5, rtol=0)
        and np.ptp(np.rint(values)) <= 32
    )


def comparison_edges(values, n_bins, binning):
    if discrete_input(values) or binning == "linear":
        return paper.reco.bin_edges(values, n_bins, [0.5, 99.5], discrete=True)
    edges = np.unique(np.percentile(values, np.linspace(0.5, 99.5, n_bins + 1)))
    if len(edges) < 2:
        edges = paper.reco.bin_edges(values, n_bins, [0.5, 99.5])
    return edges


def summarize_feature(original, decoded, name, unit, settings):
    feature = paper.reco.display_feature(name, unit)
    finite = np.isfinite(original)
    for values in decoded.values():
        finite &= np.isfinite(values)
    x = original[finite] * feature["scale"]
    ys = {key: value[finite] * feature["scale"] for key, value in decoded.items()}
    if not len(x):
        raise ValueError(f"No common finite objects for {name}")
    edges = comparison_edges(x, settings["n_bins"], settings["binning"])
    bin_ids = np.searchsorted(edges, x, side="right") - 1
    bin_ids[x == edges[-1]] = len(edges) - 2
    selections = [bin_ids == index for index in range(len(edges) - 1)]
    counts = [int(selected.sum()) for selected in selections]
    centers = [
        float(np.median(x[selected])) if selected.any() else float((a + b) / 2)
        for selected, a, b in zip(selections, edges[:-1], edges[1:])
    ]
    differences = {key: residual(x, y, name) for key, y in ys.items()}
    residual_edges = paper.reco.bin_edges(
        np.concatenate([differences[key] for key in ("q1_prediction", "q8_prediction")]),
        60,
        [0.5, 99.5],
    )
    series = {}
    for key, y in ys.items():
        rows = []
        for selected in selections:
            row = metrics(x[selected], y[selected], name)
            if row["count"] < settings["min_bin_count"]:
                row.update(dict.fromkeys(METRICS))
            elif row["response_count"] < settings["min_bin_count"]:
                row["response_width_percent"] = None
            rows.append(row)
        hist = np.histogram(differences[key], residual_edges)[0]
        series[key] = dict(
            pooled=metrics(x, y, name),
            bins=rows,
            residual_counts=hist.tolist(),
            outside_residual=int(len(x) - hist.sum()),
        )
    if not any(count >= settings["min_bin_count"] for count in counts):
        paper.log.warning(
            "%s: all bins below %s objects; binned curves will be blank",
            name,
            settings["min_bin_count"],
        )
    feature.update(
        unit="rad" if name.lower() == "phi" else feature["unit"],
        common_objects=len(x),
        excluded_nonfinite=int((~finite).sum()),
        discrete_input=discrete_input(x),
        edges=edges.tolist(),
        centers=centers,
        counts=counts,
        outside_bins=int(len(x) - sum(counts)),
        residual_edges=residual_edges.tolist(),
        series=series,
    )
    return feature


def build_summary(args):
    directories = {q: getattr(args, f"q{q}_dir").resolve() for q in (8, 1)}
    for directory in directories.values():
        for obj in paper.OBJECTS:
            if not (directory / "original_arrays" / f"{obj}.json").is_file():
                raise FileNotFoundError(
                    f"Original inputs missing in {directory}; run --postprocess-only --with-original first. No inference started"
                )
    paper.audit_pair(args)
    plans = {
        q: paper.reco.read_sealed(directory / paper.PLAN, "plan_id")
        for q, directory in directories.items()
    }
    settings = {"n_bins": args.n_bins, "min_bin_count": args.min_bin_count, "binning": args.binning}
    objects, sources = {}, {}
    for obj in paper.OBJECTS:
        originals, decodes, names = {}, {}, None
        sources[obj] = {}
        for q, directory in directories.items():
            originals[q], original_receipt = verify_original(directory, plans[q], obj)
            path, decoded_receipt = paper.verify_receipt(directory, plans[q], obj)
            with np.load(path, allow_pickle=False) as data:
                current_names = data["feature_names"].tolist()
                if names is not None and current_names != names:
                    raise ValueError(f"Q1/Q8 feature order differs: {obj}")
                names = current_names
                decodes[f"q{q}_prediction"] = data["prediction"]
                decodes[f"q{q}_tokenizer"] = data["reference"]
            sources[obj][f"q{q}"] = {"original": original_receipt, "decoded": decoded_receipt}
        if not np.array_equal(originals[8], originals[1]):
            raise ValueError(f"Q1/Q8 original features differ: {obj}; no comparison written")
        features = [
            summarize_feature(
                originals[8][:, i],
                {key: values[:, i] for key, values in decodes.items()},
                name,
                plans[8]["input_momentum_unit"],
                settings,
            )
            for i, name in enumerate(names)
        ]
        objects[obj] = {"features": features}
    return paper.reco.seal(
        {
            "version": VERSION,
            "plans": {f"q{q}": plan for q, plan in plans.items()},
            "sources": sources,
            "settings": settings,
            "source_digest": paper.capacity.digest_json(
                inspect.getsource(residual)
                + inspect.getsource(metrics)
                + inspect.getsource(discrete_input)
                + inspect.getsource(comparison_edges)
                + inspect.getsource(summarize_feature)
            ),
            "objects": objects,
        },
        "summary_id",
    )


def series_to_plot(show_tokenizer):
    for q in (1, 8):
        yield f"q{q}_prediction", f"Q{q} prediction", QUANTIZER_COLORS[q], "-", (
            "o" if q == 1 else "s"
        )
        if show_tokenizer:
            yield f"q{q}_tokenizer", f"Q{q} tokenizer only", QUANTIZER_COLORS[q], "--", None


def render_page(summary, obj, metric, *, show_tokenizer=False):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    features = summary["objects"][obj]["features"]
    nrows = math.ceil(len(features) / 3)
    height = 3.05 * nrows + 0.8
    fig, axes = plt.subplots(nrows, 3, figsize=(12.8, height), squeeze=False)
    fig.subplots_adjust(
        left=0.07,
        right=0.985,
        bottom=0.55 / height,
        top=1 - 0.85 / height,
        hspace=0.62,
        wspace=0.42,
    )
    label = dict(zip(paper.OBJECTS, paper.OBJECT_LABELS))[obj]
    title = {
        "width": "Binned width",
        "bias": "Binned bias",
        "mae": "Binned mean absolute error",
        "residuals": "Residual distributions",
    }[metric]
    fig.text(
        0.07,
        1 - 0.12 / height,
        f"{label} | {title} | Masked prediction vs original",
        va="top",
        fontsize=11,
    )
    handles = None
    for ax, feature in zip(axes.flat, features):
        name, unit = feature["name"], feature["unit"]
        suffix = f" [{unit}]" if unit else ""
        row_metric = (
            ("response_width_percent" if name.lower() == "pt" else "residual_iqr")
            if metric == "width"
            else {"bias": "bias", "mae": "mean_absolute_error"}.get(metric)
        )
        for key, legend, color, linestyle, marker in series_to_plot(show_tokenizer):
            series = feature["series"][key]
            if metric == "residuals":
                ax.stairs(
                    np.asarray(series["residual_counts"]) / feature["common_objects"],
                    feature["residual_edges"],
                    color=color,
                    linestyle=linestyle,
                    label=legend,
                )
            else:
                values = [
                    np.nan if row[row_metric] is None else row[row_metric] for row in series["bins"]
                ]
                ax.plot(
                    feature["centers"],
                    values,
                    color=color,
                    ls=linestyle,
                    marker=marker,
                    ms=3.5,
                    label=legend,
                )
                if key.endswith("prediction"):
                    centers = np.asarray(feature["centers"])
                    edges = np.asarray(feature["edges"])
                    values = np.asarray(values)
                    valid = np.isfinite(values)
                    if valid.any():
                        ax.errorbar(
                            centers[valid],
                            values[valid],
                            xerr=np.stack((centers - edges[:-1], edges[1:] - centers))[:, valid],
                            fmt="none",
                            ecolor=color,
                            elinewidth=0.7,
                            alpha=0.4,
                        )
        if metric == "residuals":
            ax.axvline(0, color="#444444", ls=":", lw=0.9)
            ax.set_xlabel("Predicted - original " + feature["label"] + suffix)
            ax.set_ylabel("Normalized masked objects")
            lines = []
            for q in (1, 8):
                stats = feature["series"][f"q{q}_prediction"]["pooled"]
                lines.append(
                    f"Q{q}: median {stats['bias']:.3g}; median |res.| {stats['median_absolute_error']:.3g}"
                )
            ax.text(
                0.97,
                0.95,
                "\n".join(lines),
                transform=ax.transAxes,
                va="top",
                ha="right",
                fontsize=7,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
            )
            ax.set_ylim(bottom=0)
        else:
            ax.set_xlabel("Original " + feature["label"] + suffix)
            ax.set_ylabel(
                {
                    "bias": "Median residual" + suffix,
                    "mae": "Mean |residual|" + suffix,
                    "width": RESPONSE_LABEL if name.lower() == "pt" else "IQR(residual)" + suffix,
                }[metric]
            )
            ax.set_xlim(feature["edges"][0], feature["edges"][-1])
            if metric == "bias":
                ax.axhline(0, color="#444444", ls=":", lw=0.9)
            else:
                ax.set_ylim(bottom=0)
            missing = sum(
                row[row_metric] is None for row in feature["series"]["q1_prediction"]["bins"]
            )
            missing = max(
                missing,
                sum(row[row_metric] is None for row in feature["series"]["q8_prediction"]["bins"]),
            )
            if missing:
                ax.text(
                    0.04,
                    0.95,
                    f"{missing}/{len(feature['counts'])} bins unavailable",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=7,
                    color="#555555",
                )
        ax.set_title(
            f"{feature['label']}  (N = {feature['common_objects']:,})", loc="left", fontsize=10
        )
        ax.xaxis.set_major_locator(
            MaxNLocator(nbins=5, integer=metric != "residuals" and feature["discrete_input"])
        )
        style_axis(ax)
        if handles is None:
            handles = ax.get_legend_handles_labels()
    for ax in list(axes.flat)[len(features) :]:
        ax.set_visible(False)
    fig.legend(
        *handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 1 - 0.36 / height),
        ncol=4 if show_tokenizer else 2,
        frameon=False,
        fontsize=9,
    )
    return fig


def compare(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = build_summary(args)
    destination = args.output_dir
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "paired_error_summary.json"
    if summary_path.exists():
        saved = paper.reco.read_sealed(summary_path, "summary_id")
        if saved != summary:
            raise ValueError(
                "Comparison inputs/settings changed; use an explicit different output directory"
            )
    else:
        paper.reco.write_json(summary_path, summary)
    rows = []
    with paper.paper_style():
        for obj in paper.OBJECTS:
            for metric in ("width", "bias", "mae", "residuals"):
                fig = render_page(
                    summary, obj, metric, show_tokenizer=args.show_tokenizer_reference
                )
                paper.save_figure(fig, destination / f"q1_vs_q8_{obj}_{metric}")
                plt.close(fig)
            for feature in summary["objects"][obj]["features"]:
                for key, series in feature["series"].items():
                    for index, row in enumerate([series["pooled"], *series["bins"]], start=-1):
                        rows.append(
                            dict(
                                object=obj,
                                feature=feature["name"],
                                unit=feature["unit"],
                                series=key,
                                bin=index,
                                low=feature["edges"][index] if index >= 0 else None,
                                high=feature["edges"][index + 1] if index >= 0 else None,
                                **row,
                            )
                        )
    with (destination / "paired_error_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (destination / "definitions.md").write_text(
        "# Paired Q1/Q8 masked-prediction errors\n\n"
        "Both predictions are compared with identical original physical inputs at the same saved masked-object positions. "
        "Originals were recovered from aligned continuous inputs with the saved export preprocessors; numerical/preprocessing precision applies. "
        "This is an end-to-end physical-feature comparison, not the earlier prediction-minus-true-code-decode residual.\n\n"
        "Bins depend only on original values and are shared across Q1/Q8. Default quantile bins have roughly equal counts "
        "within the original 0.5--99.5 percentile range; repeated edges are collapsed. Small integer inputs use unit-width bins. "
        f"Points are drawn at the median original value in each bin; horizontal bars show bin extents, not statistical uncertainty. Bins below {args.min_bin_count} common objects are suppressed, not interpolated. "
        "The pT width is 100 * IQR(predicted/original) / median(predicted/original). Only original > 1e-8 GeV is required; "
        "nonpositive predictions remain in the response sample and their counts are recorded. Width is undefined if its median is nonpositive. "
        "Other widths are IQR(predicted - original). Bias is the median signed residual; MAE is the mean absolute residual. "
        "Median absolute residual and its 90th percentile are also reported in the CSV. Phi residuals are wrapped to [-pi, pi), "
        "unlike the unchanged older triptychs. No dimensionless relative residual is used for features crossing zero.\n\n"
        "Residual histograms share edges for both predictions and are normalized by the full common finite count, including "
        "out-of-range residuals. Histogram windows use pooled prediction-residual 0.5--99.5 percentiles; clipped counts are in the summary. "
        "Pooled metrics include tails outside both plotted windows. All plotted metrics use a common finite intersection of both "
        "predictions and tokenizer references; excluded counts are recorded. The strict paired-cache audit runs first.\n\n"
        "The CSV additionally records tokenizer-only (true-code decoded minus original) errors to separate tokenizer compression "
        "from event prediction. --show-tokenizer-reference adds dashed reference curves; they are not a mathematical error floor. "
        "Discrete/ID outputs are not rounded or thresholded: numeric errors here are not classification accuracy, and zero IQR or "
        "zero median absolute residual can coexist with rare mistakes. Inspect mean absolute error and the residual histogram.\n\n"
        "These are single-checkpoint point estimates, without uncertainty bands or a significance claim. Sparse-bin fluctuations "
        "do not establish superiority. A significance analysis would require paired resampling clustered by source event. "
        "The comparison assesses the two recorded pipelines, not an isolated causal effect of quantizer count. "
        "Models, predictions, samples, masks and preprocessing are not changed. No new inference or downloads occur. "
        "Source plans, receipts and settings are frozen in paired_error_summary.json.\n"
    )
    paper.reco.write_json(
        destination / "plot_options.json",
        {
            "summary": paper.capacity.file_record(summary_path, content_hash=True),
            "show_tokenizer_reference": args.show_tokenizer_reference,
            "colors": {str(q): QUANTIZER_COLORS[q] for q in (1, 8)},
        },
    )
    paper.log.info(
        "Saved paired Q1/Q8 width, bias, MAE and residual pages: %s; cached arrays only",
        destination,
    )
