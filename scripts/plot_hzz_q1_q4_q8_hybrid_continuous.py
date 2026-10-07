#!/usr/bin/env python3
"""Redraw saved HZZ binary/multiclass results on CPU, without model imports."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from evaluate_q1_hierarchical_flat_hzz import write_plots


MODEL_ORDER = ("q1", "q4", "q8", "hybrid", "continuous")
MODEL_NAMES = {"q1": "Q1", "q4": "Q4", "q8": "Q8", "hybrid": "Hybrid", "continuous": "Continuous"}
COLORS = {
    "q1": "#D55E00",
    "q4": "#CC79A7",
    "q8": "#0072B2",
    "hybrid": "#8064A2",
    "continuous": "#009E73",
}
DISPLAY_NAMES = {
    "background": r"$ZZ^{(*)}$ continuum",
    "signal": "ggF",
    "ggf": "ggF",
    "vbf": "VBF",
    "vh": "VH",
    "zh": "ZH",
    "ggzh": "ggZH",
    "wh": "WH",
    "tth": r"$t\bar{t}H + tH$",
    "th": "tH",
    "zz_continuum": r"$ZZ^{(*)}$ continuum",
}
PREFIX = "pretrained_fine_tuned"
Q8_RUNS = ("parallel", "autoregressive")


def file_record(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


@dataclass
class Evaluation:
    key: str
    task: str
    directory: Path
    class_names: tuple[str, ...]
    labels: np.ndarray
    probabilities: np.ndarray
    report: dict
    sources: list[dict]
    membership_sha256: str | None
    source_kind: str
    metric_result: dict | None = None

    @property
    def counts(self) -> dict:
        return {
            name: int(np.count_nonzero(self.labels == i)) for i, name in enumerate(self.class_names)
        }

    @property
    def name(self) -> str:
        name = MODEL_NAMES[self.key]
        if self.source_kind.startswith("paired_q8_"):
            return name + " " + self.report["selected_q8_run"]
        if self.key in ("q1", "q8") and self.source_kind == "legacy_roc_scores":
            return name + " (previous baseline)"
        if self.report.get("object_projection_mode") == "identity":
            return name + " (direct)"
        return name

    def metrics(self) -> dict:
        if self.metric_result is not None:
            return self.metric_result
        predicted = self.probabilities.argmax(axis=1)
        raw = confusion_matrix(self.labels, predicted, labels=range(len(self.class_names)))
        per_class = {
            name: float(roc_auc_score(self.labels == i, self.probabilities[:, i]))
            for i, name in enumerate(self.class_names)
        }
        result = {
            "task": self.task,
            "model": self.key,
            "name": self.name,
            "events": int(self.labels.size),
            "class_names": list(self.class_names),
            "class_counts": self.counts,
            "accuracy": float(accuracy_score(self.labels, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(self.labels, predicted)),
            "per_class_auc": per_class,
            "raw_confusion_matrix": raw.tolist(),
            "normalized_confusion_matrix": (raw / raw.sum(axis=1, keepdims=True)).tolist(),
        }
        if self.task == "binary":
            result["auc"] = per_class["signal"]
            fpr, tpr, _ = roc_curve(self.labels, self.probabilities[:, 1])
            for efficiency in (50, 70, 80):
                result[f"rejection_at_{efficiency}pct"] = float(
                    1 / max(np.interp(efficiency / 100, tpr, fpr), 1 / self.counts["background"])
                )
        else:
            result["macro_ovr_auc"] = float(np.mean(list(per_class.values())))
            result["weighted_ovr_auc"] = float(
                np.average(list(per_class.values()), weights=list(self.counts.values()))
            )
        self.metric_result = result
        return result


def read_report(directory: Path, key: str) -> tuple[dict, list[dict]]:
    path = directory / "evaluation_summary.json"
    if not path.is_file():
        path = directory / "summary.json"
    value = json.loads(path.read_text())
    if isinstance(value, list):
        rows = [row for row in value if row.get("setup") == "Pretrained, fine-tuned"]
        if len(rows) != 1:
            raise ValueError(f"Need exactly one pretrained/fine-tuned result in {path}")
        value = rows[0]
    if not isinstance(value, dict):
        raise ValueError(f"Expected an evaluation report in {path}")
    if (
        value.get("representation") is not None
        and key != "hybrid"
        and value["representation"] != key
    ):
        raise ValueError(f"Wrong representation for {key}: {path}")
    return value, [file_record(path)]


def read_paired_q8_report(directory: Path, task: str, run: str) -> tuple[dict, list[dict]]:
    if run not in Q8_RUNS:
        raise ValueError(f"Unknown Q8 prediction branch: {run}")
    metadata_path = directory / "evaluation_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("task") != task:
        raise ValueError(f"Q8 metadata task does not match {task}: {metadata_path}")
    if not isinstance(metadata.get("class_names"), list) or not metadata["class_names"]:
        raise ValueError(f"Missing Q8 class definitions: {metadata_path}")
    checkpoint = metadata.get("checkpoints", {}).get(run)
    digest = metadata.get("ordered_input_sha256", {}).get(run)
    if not checkpoint or not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"Missing Q8 {run} checkpoint/input provenance: {metadata_path}")
    digests = metadata["ordered_input_sha256"]
    if (
        all(branch in digests for branch in Q8_RUNS)
        and digests["parallel"] != digests["autoregressive"]
    ):
        raise ValueError(f"Q8 branches have different ordered input digests: {metadata_path}")
    summary_path = directory / "summary.json"
    rows = json.loads(summary_path.read_text())
    if not isinstance(rows, list):
        raise ValueError(f"Expected paired Q8 run summaries: {summary_path}")
    selected = [row for row in rows if row.get("run") == f"Q8 {run}"]
    if len(selected) != 1 or selected[0].get("checkpoint") != checkpoint:
        raise ValueError(f"Need exactly one matching Q8 {run} summary/checkpoint: {summary_path}")
    report = {
        **selected[0],
        "class_names": metadata["class_names"],
        "selected_q8_run": run,
        "paired_evaluation_metadata": metadata,
    }
    return report, [file_record(summary_path), file_record(metadata_path)]


def identity_digest(arrays: dict, labels: np.ndarray, classes: tuple[str, ...]) -> str | None:
    if "source_file" not in arrays and "event_index" not in arrays:
        return None
    sources, indices = arrays.get("source_file"), arrays.get("event_index")
    if (
        sources is None
        or indices is None
        or sources.shape != labels.shape
        or indices.shape != labels.shape
    ):
        raise ValueError("Event identities must be aligned with labels")
    if not np.issubdtype(indices.dtype, np.integer) or np.any(indices < 0):
        raise ValueError("Event indices must be nonnegative integers")
    identities = [(str(source), int(index)) for source, index in zip(sources, indices, strict=True)]
    if any(not source for source, _ in identities) or len(set(identities)) != labels.size:
        raise ValueError("Duplicate or missing test event identities")
    members = sorted(
        (source, index, classes[int(label)])
        for (source, index), label in zip(identities, labels, strict=True)
    )
    digest = hashlib.sha256()
    for member in members:
        digest.update(json.dumps(member, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def load_evaluation(
    key: str, task: str, directory: Path, *, q8_run: str = "parallel"
) -> Evaluation:
    directory = directory.resolve()
    path = directory / "test_predictions.npz"
    source_kind = "test_predictions"
    paired_q8 = False
    if path.is_file():
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        paired_q8 = any(f"{run}_probabilities" in arrays for run in Q8_RUNS)
        if paired_q8:
            if key != "q8":
                raise ValueError(f"Cannot relabel paired Q8 predictions as {key}: {path}")
            report, sources = read_paired_q8_report(directory, task, q8_run)
            probabilities = arrays.get(f"{q8_run}_probabilities")
            if probabilities is None:
                raise ValueError(f"Missing selected Q8 {q8_run} probabilities: {path}")
            source_kind = f"paired_q8_{q8_run}"
        else:
            report, sources = read_report(directory, key)
            probabilities = arrays.get("probabilities")
        labels = arrays.get("labels", arrays.get("label"))
        if (
            "labels" in arrays
            and "label" in arrays
            and not np.array_equal(arrays["labels"], arrays["label"])
        ):
            raise ValueError(f"Conflicting saved label arrays: {path}")
        if probabilities is None:
            scores = arrays.get("signal_score")
            if task != "binary" or scores is None or scores.ndim != 1:
                raise ValueError(f"Missing classifier probabilities: {path}")
            probabilities = np.column_stack((1 - scores, scores))
        classes = arrays.get("class_names", report.get("class_names"))
        if classes is None:
            if task != "binary":
                raise ValueError(f"Missing multiclass label/column definitions: {path}")
            classes = ("background", "signal")
    else:
        if task != "binary":
            raise FileNotFoundError(f"Need saved multiclass probabilities: {path}")
        report, sources = read_report(directory, key)
        path = directory / "roc_arrays.npz"
        with np.load(path, allow_pickle=False) as archive:
            labels = archive[f"{PREFIX}_labels"]
            scores = archive[f"{PREFIX}_scores"]
            arrays = {"labels": labels}
            for name in ("source_file", "event_index"):
                if f"{PREFIX}_{name}" in archive:
                    arrays[name] = archive[f"{PREFIX}_{name}"]
            arrays["saved_fpr"] = archive[f"{PREFIX}_fpr"]
            arrays["saved_tpr"] = archive[f"{PREFIX}_tpr"]
        probabilities = np.column_stack((1 - scores, scores))
        classes = ("background", "signal")
        source_kind = "legacy_roc_scores"
    sources.append(file_record(path))
    classes = tuple(str(value) for value in classes)
    if len(set(classes)) != len(classes) or not all(classes):
        raise ValueError(f"Duplicate/missing class names: {path}")
    if report.get("class_names") is not None and tuple(report["class_names"]) != classes:
        raise ValueError(f"Class definitions disagree with the report: {path}")
    if paired_q8 and task == "binary" and classes == ("zz_continuum", "ggf"):
        classes = ("background", "signal")
    if task == "binary" and classes != ("background", "signal"):
        raise ValueError(f"Binary columns must be background=0, signal=1: {path}")
    if task == "multiclass" and len(classes) < 3:
        raise ValueError(f"Cannot label a binary export as multiclass: {path}")
    if labels is None or labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"Need one integer true label per event: {path}")
    if set(np.unique(labels).tolist()) != set(range(len(classes))):
        raise ValueError(f"Missing/out-of-range true classes: {path}")
    if probabilities.shape != (labels.size, len(classes)) or not np.isfinite(probabilities).all():
        raise ValueError(f"Invalid probability dimensions/values: {path}")
    if (
        np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.allclose(probabilities.sum(1), 1, atol=1e-5, rtol=0)
    ):
        raise ValueError(f"Invalid softmax probabilities: {path}")
    if report.get("signal_label", 1) != 1 and task == "binary":
        raise ValueError(f"Wrong signal-score convention: {path}")
    if (
        task == "binary"
        and report.get("score_definition", "softmax(logits)[:,1]") != "softmax(logits)[:,1]"
    ):
        raise ValueError(f"Wrong signal-score definition: {path}")
    if "predictions" in arrays and not np.array_equal(
        arrays["predictions"], probabilities.argmax(1)
    ):
        raise ValueError(f"Saved predicted labels disagree with argmax probabilities: {path}")
    count = report.get("test_events", report.get("events"))
    if count != labels.size:
        raise ValueError(f"Report event count does not match saved predictions: {path}")
    value = Evaluation(
        key,
        task,
        directory,
        classes,
        labels,
        probabilities,
        report,
        sources,
        identity_digest(arrays, labels, classes),
        source_kind,
    )
    metrics = value.metrics()
    expected_auc = metrics["auc"] if task == "binary" else metrics["macro_ovr_auc"]
    for field in ("auc", "test_auc") if task == "binary" else ("macro_ovr_auc", "test_auc"):
        if field in report and not np.isclose(report[field], expected_auc, atol=1e-8, rtol=0):
            raise ValueError(f"Saved predictions do not reproduce {field}: {path}")
    for field in ("accuracy", "balanced_accuracy", "weighted_ovr_auc"):
        if (
            field in report
            and field in metrics
            and not np.isclose(report[field], metrics[field], atol=1e-8, rtol=0)
        ):
            raise ValueError(f"Saved predictions do not reproduce {field}: {path}")
    if "per_class_auc" in report:
        for name, number in metrics["per_class_auc"].items():
            if name not in report["per_class_auc"] or not np.isclose(
                report["per_class_auc"][name], number, atol=1e-8, rtol=0
            ):
                raise ValueError(f"Saved predictions do not reproduce {name} AUC: {path}")
    for field in ("raw_confusion_matrix", "confusion_matrix", "normalized_confusion_matrix"):
        if field in report:
            expected_matrix = metrics[
                (
                    "normalized_confusion_matrix"
                    if field.startswith("normalized")
                    else "raw_confusion_matrix"
                )
            ]
            observed = np.asarray(report[field])
            if observed.shape != np.shape(expected_matrix) or not np.allclose(
                observed, expected_matrix, atol=1e-8, rtol=0
            ):
                raise ValueError(f"Saved predictions do not reproduce {field}: {path}")
    if source_kind == "legacy_roc_scores":
        fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
        for field, expected_array in (("saved_fpr", fpr), ("saved_tpr", tpr)):
            if arrays[field].shape != expected_array.shape or not np.allclose(
                arrays[field], expected_array, atol=1e-12, rtol=0
            ):
                raise ValueError(f"Saved scores do not reproduce ROC arrays: {path}")
    return value


def validate_group(values: list[Evaluation]) -> dict:
    reference = values[0]
    reference_classes = set(reference.class_names)
    audited = [value for value in values if value.membership_sha256 is not None]
    for value in values:
        if set(value.class_names) != reference_classes or value.counts != reference.counts:
            raise ValueError(
                f"Different true classes/test counts: {reference.key} versus {value.key}"
            )
    if audited and any(
        value.membership_sha256 != audited[0].membership_sha256 for value in audited[1:]
    ):
        raise ValueError(
            "Test event identities or semantic labels differ between "
            + ", ".join(value.key for value in audited)
        )
    unavailable = [value.key for value in values if value.membership_sha256 is None]
    return {
        "models": [value.key for value in values],
        "classes": list(reference.class_names),
        "events": int(reference.labels.size),
        "class_counts": reference.counts,
        "identity_verified_models": [value.key for value in audited],
        "membership_sha256": audited[0].membership_sha256 if audited else None,
        "identity_unavailable_models": unavailable,
        "note": (
            "Saved event IDs and semantic labels match, independent of row/column order. "
            "Older exports without IDs can only be checked for matching class counts; "
            "their event membership is not independently verified."
            if unavailable
            else "Saved event IDs and semantic labels match, independent of row/column order."
        ),
    }


def save_plot(figure, stem: Path):
    for suffix in ("png", "pdf"):
        figure.savefig(stem.with_suffix("." + suffix), dpi=240)
    plt.close(figure)


def draw_matrix(axis, value: Evaluation, matrix, classes=None):
    classes = classes or value.class_names
    order = [value.class_names.index(name) for name in classes]
    matrix = matrix[np.ix_(order, order)]
    names = [DISPLAY_NAMES.get(name, name) for name in classes]
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    axis.set_xticks(range(len(names)), labels=names, rotation=35, ha="right", fontsize=9)
    axis.set_yticks(range(len(names)), labels=names, fontsize=9)
    axis.set_xlabel("Predicted class", fontsize=11, loc="center")
    axis.set_ylabel("True class", fontsize=11, loc="center")
    axis.set_title(f"{value.name} | N = {value.labels.size:,}", fontsize=12)
    axis.tick_params(which="both", length=0, top=False, right=False)
    for row in range(len(names)):
        for column in range(len(names)):
            number = matrix[row, column]
            axis.text(
                column,
                row,
                f"{100 * number:.1f}%",
                ha="center",
                va="center",
                fontsize=10 if len(names) <= 5 else 8,
                color="white" if number >= 0.5 else "black",
            )
    return image


def plot_confusions(values: list[Evaluation], output: Path, group_name: str):
    for value in values:
        metrics = value.metrics()
        raw = np.asarray(metrics["raw_confusion_matrix"])
        matrix = np.asarray(metrics["normalized_confusion_matrix"])
        stem = output / f"{value.task}_{value.key}_confusion_matrix"
        figure, axis = plt.subplots(figsize=(7.5, 6.8), constrained_layout=True)
        image = draw_matrix(axis, value, matrix)
        colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label("Fraction of true class", fontsize=11)
        colorbar.ax.tick_params(labelsize=10)
        save_plot(figure, stem)
        with stem.with_suffix(".csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["true_class", "predicted_class", "count", "row_fraction"])
            for i, true_class in enumerate(value.class_names):
                for j, predicted_class in enumerate(value.class_names):
                    writer.writerow(
                        [true_class, predicted_class, int(raw[i, j]), float(matrix[i, j])]
                    )
    if len(values) < 2:
        return
    columns = min(3, len(values))
    rows = math.ceil(len(values) / columns)
    side = max(4.0, 0.7 * max(len(value.class_names) for value in values) + 1.3)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * side + 0.6, rows * (side + 0.6)),
        squeeze=False,
        constrained_layout=True,
    )
    active = []
    for axis, value in zip(axes.flat, values):
        image = draw_matrix(
            axis,
            value,
            np.asarray(value.metrics()["normalized_confusion_matrix"]),
            values[0].class_names,
        )
        active.append(axis)
    for axis in list(axes.flat)[len(values) :]:
        axis.set_visible(False)
    colorbar = figure.colorbar(image, ax=active, fraction=0.025, pad=0.03)
    colorbar.set_label("Fraction of true class", fontsize=11)
    colorbar.ax.tick_params(labelsize=10)
    save_plot(figure, output / f"{group_name}_confusion_matrices")


def plot_multiclass_rocs(values: list[Evaluation], output: Path, group_name: str):
    classes = values[0].class_names
    columns = min(3, len(classes))
    rows = math.ceil(len(classes) / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(columns * 4.6, rows * 4.2), squeeze=False, constrained_layout=True
    )
    for axis, name in zip(axes.flat, classes):
        for value in values:
            index = value.class_names.index(name)
            labels, scores = value.labels == index, value.probabilities[:, index]
            fpr, tpr, _ = roc_curve(labels, scores)
            auc = roc_auc_score(labels, scores)
            axis.plot(
                fpr, tpr, lw=2, color=COLORS[value.key], label=f"{value.name} (AUC={auc:.3f})"
            )
        axis.plot([0, 1], [0, 1], "--", color="0.5", lw=0.9)
        axis.set(xlim=(0, 1), ylim=(0, 1.02))
        axis.tick_params(axis="both", which="both", labelsize=10)
        axis.set_xlabel("Background efficiency (rest)", fontsize=11)
        axis.set_ylabel("Class efficiency", fontsize=11)
        axis.set_title(DISPLAY_NAMES.get(name, name) + " vs rest", fontsize=12)
        axis.grid(True, alpha=0.2)
        axis.legend(loc="lower right", fontsize=8, frameon=False)
    for axis in list(axes.flat)[len(classes) :]:
        axis.set_visible(False)
    save_plot(figure, output / f"{group_name}_roc_ovr_comparison")


def parse_args(argv=None):
    root = Path("results")
    previous = root / "atlas_hzz_200m_data_pretrain_comparison"
    parser = argparse.ArgumentParser(description=__doc__)
    binary_defaults = {
        "q1": root / "google_q1_direct256_cluster198/test_evaluation",
        "q4": root / "google_q4_svsb_cluster206/test_evaluation",
        "q8": root / "hzz_q8_direct_google/cluster_166_proc_0/test_predictions",
        "hybrid": root / "google_hybrid_svsb_cluster203/test_evaluation",
        "continuous": previous / "four_setup_continuous",
    }
    for key, default in binary_defaults.items():
        parser.add_argument(f"--binary-{key}", type=Path, default=default)
    parser.add_argument(
        "--multiclass-q4",
        type=Path,
        default=root / "google_q4_multiclass_cluster209/test_evaluation",
    )
    parser.add_argument(
        "--multiclass-hybrid",
        type=Path,
        default=root / "google_hybrid_multiclass_cluster204/test_evaluation",
    )
    parser.add_argument(
        "--multiclass-q8",
        type=Path,
        required=True,
        help="Exact existing Q8 multiclass evaluation directory; no automatic replacement",
    )
    for key in ("q1", "continuous"):
        parser.add_argument(
            f"--multiclass-{key}",
            type=Path,
            help="Include only if this multiclass evaluation actually exists",
        )
    for task in ("binary", "multiclass"):
        parser.add_argument(
            f"--{task}-q8-run",
            choices=Q8_RUNS,
            default="parallel",
            help="Select one branch if the Q8 export contains both parallel/autoregressive results",
        )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "paper/hzz_q1_q4_q8_hybrid_continuous_roc_confusion_v1",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check all saved arrays and memberships; write no outputs",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    binary = [
        load_evaluation(key, "binary", getattr(args, f"binary_{key}"), q8_run=args.binary_q8_run)
        for key in MODEL_ORDER
    ]
    multiclass = [
        load_evaluation(key, "multiclass", path, q8_run=args.multiclass_q8_run)
        for key in MODEL_ORDER
        if (path := getattr(args, f"multiclass_{key}")) is not None
    ]
    binary_audit = validate_group(binary)
    grouped = {}
    for value in multiclass:
        grouped.setdefault(tuple(sorted(value.class_names)), []).append(value)
    group_audits = [validate_group(values) for values in grouped.values()]
    metadata = {
        "binary": binary_audit,
        "multiclass_groups": group_audits,
        "multiclass_not_supplied": [
            key for key in MODEL_ORDER if getattr(args, f"multiclass_{key}") is None
        ],
        "sources": [
            {
                "task": value.task,
                "model": value.key,
                "directory": str(value.directory),
                "files": value.sources,
                "source_kind": value.source_kind,
                "evaluation_report": value.report,
            }
            for value in binary + multiclass
        ],
        "conventions": {
            "confusion": "Rows=true class; columns=argmax predicted class; rows normalized to 100%",
            "binary_roc": "signal=1; own paired labels/scores for each model",
            "rejection": "1/background efficiency; floor=1/N_background",
            "multiclass_roc": "One class versus all remaining classes; compare only matching class sets and test membership",
            "inference": "None; saved probabilities only",
        },
    }
    signature = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    metadata["input_signature"] = signature
    for audit in [binary_audit, *group_audits]:
        print(
            f"PASS {','.join(audit['models'])}: {audit['events']:,} events, {len(audit['classes'])} classes"
        )
        if audit["identity_unavailable_models"]:
            print(
                "WARNING: event IDs unavailable for "
                + ", ".join(audit["identity_unavailable_models"])
                + "; matching counts do not prove membership"
            )
    if len(grouped) > 1:
        print(
            "Different multiclass tasks kept separate; no ROCs overlaid across different class sets."
        )
    if args.validate_only:
        print("Validation only; no figures or directories written.")
        return
    output = args.output_dir.resolve()
    if any(
        output == value.directory or output.is_relative_to(value.directory)
        for value in binary + multiclass
    ):
        raise ValueError("Output directory must be separate from every saved input evaluation")
    existing = output / "comparison_metadata.json"
    if output.exists() and any(output.iterdir()):
        if (
            not existing.is_file()
            or json.loads(existing.read_text()).get("input_signature") != signature
        ):
            raise FileExistsError(
                "Output directory contains different/unknown results; keep it and choose a new named directory"
            )
    output.mkdir(parents=True, exist_ok=True)
    plt.style.use(hep.style.ROOT)
    curves = []
    for value in binary:
        fpr, tpr, _ = roc_curve(value.labels, value.probabilities[:, 1])
        curves.append(
            {
                "name": value.name,
                "fpr": fpr,
                "tpr": tpr,
                "auc": value.metrics()["auc"],
                "background_events": value.counts["background"],
                "color": COLORS[value.key],
            }
        )
    binary_name = "binary_" + "_".join(value.key for value in binary)
    write_plots(output, curves)
    for name in ("background_efficiency", "background_rejection"):
        for suffix in ("png", "pdf"):
            (output / f"{name}.{suffix}").replace(output / f"{binary_name}_{name}.{suffix}")
    plot_confusions(binary, output, binary_name)
    for values in grouped.values():
        group_name = (
            "multiclass_"
            + "_".join(value.key for value in values)
            + f"_{len(values[0].class_names)}class"
        )
        plot_confusions(values, output, group_name)
        plot_multiclass_rocs(values, output, group_name)
    rows = [value.metrics() for value in binary + multiclass]
    (output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (output / "metrics.csv").open("w", newline="") as stream:
        fields = [
            "task",
            "model",
            "name",
            "events",
            "accuracy",
            "balanced_accuracy",
            "auc",
            "macro_ovr_auc",
            "weighted_ovr_auc",
            "rejection_at_50pct",
            "rejection_at_70pct",
            "rejection_at_80pct",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with (output / "per_class_auc.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["task", "model", "class", "events", "auc"])
        for row in rows:
            for name, auc in row["per_class_auc"].items():
                writer.writerow([row["task"], row["model"], name, row["class_counts"][name], auc])
    (output / "comparison_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "captions.md").write_text(
        "# HZZ saved-evaluation comparisons\n\n"
        "Binary curves compare Q1, Q4, Q8, hybrid and continuous fine-tuned classifiers on their "
        "saved test predictions. Background rejection is the inverse background efficiency with a "
        "one-background-event floor. Input paths, architectures where exported, counts and membership "
        "checks are recorded in comparison_metadata.json. A '(previous baseline)' label denotes the "
        "older cached classifier, not a substituted direct-backbone run. Architectures are not "
        "assumed identical; this is not automatically a controlled quantizer-depth ablation.\n\n"
        "Confusion matrices use true classes as rows and argmax predicted classes as columns. "
        "Each row sums to 100%; all figures share the 0--100% color range. Multiclass ROC curves "
        "are one-versus-rest. Different class-set tasks are shown separately, never overlaid. "
        "Exports without event IDs are checked for class counts only; their membership cannot be "
        "independently verified. Paired Q8 exports use only the explicitly selected prediction branch; "
        "their ordered-input digest compares the two Q8 passes, not Q8 membership against other models. "
        "Unsupplied Q1/continuous multiclass evaluations are not invented. "
        "No models, H5 files, Parquet shards, Google credentials or GPUs are used.\n"
    )
    for row in rows:
        print(
            f"{row['task']:10s} {row['name']:25s} AUC={row.get('auc', row.get('macro_ovr_auc')):.6f}"
        )
    print(f"Saved all PNG/PDF figures and tables in {output}; no inference")


if __name__ == "__main__":
    main()
