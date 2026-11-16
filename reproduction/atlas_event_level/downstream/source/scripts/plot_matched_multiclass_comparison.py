#!/usr/bin/env python3
"""Plot per-class FPR/TPR ROC curves only after proving identical test events."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve

NAMES = ("ggf", "vbf", "vh", "tth", "zz_continuum")
DISPLAY = ("ggF", "VBF", "VH", "ttH + tH", r"ZZ$^{(*)}$ continuum")
SOURCE_SHA = "19cdf0a7afe372d9269df1213474e5945196c63c34b8b864b447b60197f8ba3a"


def load_predictions(path):
    path = Path(path)
    if path.is_dir():
        path = path / "test_predictions.npz"
    with np.load(path, allow_pickle=False) as saved:
        label_key = "labels" if "labels" in saved else "label"
        labels = np.asarray(saved[label_key])
        scores = np.asarray(saved["probabilities"])
        sources = np.asarray(saved["source_file"], dtype=str)
        events = np.asarray(saved["event_index"])
        names = tuple(saved["class_names"].astype(str))
    if names != NAMES or scores.shape != (len(labels), 5):
        raise ValueError("Require the same five classes in the same score-column order")
    if labels.ndim != 1 or sources.shape != labels.shape or events.shape != labels.shape:
        raise ValueError("Incomplete event identities or labels")
    if set(labels.tolist()) != set(range(5)) or not np.isfinite(scores).all() or not np.allclose(scores.sum(1), 1, atol=1e-5):
        raise ValueError("Invalid class probabilities/labels")
    if np.any(scores < 0) or np.any(scores > 1) or np.any(events < 0) or np.any(sources == ""):
        raise ValueError("Invalid scores or event identities")
    order = np.lexsort((events, sources))
    labels, scores, sources, events = labels[order], scores[order], sources[order], events[order]
    if np.any((sources[1:] == sources[:-1]) & (events[1:] == events[:-1])):
        raise ValueError("Repeated test event")
    return dict(labels=labels, probabilities=scores, source_file=sources, event_index=events)


def load_q8_reference(directory, reference):
    """Reuse old paired-Q8 scores without pretending they contain event IDs.

    Old exports carry the input-manifest checksum and ordered tensor digests,
    but not per-event IDs. Require the exact original canonical manifest and
    matching class counts; record this narrower provenance check explicitly.
    """
    directory = Path(directory)
    metadata = json.loads((directory / 'evaluation_metadata.json').read_text())
    if metadata.get('task') != 'multiclass' or tuple(metadata.get('class_names', [])) != NAMES:
        raise ValueError('Not the original five-class Q8 reference')
    if metadata.get('manifest_sha256') != SOURCE_SHA:
        raise ValueError('Q8 reference belongs to a different/unverified input manifest')
    digests = metadata.get('ordered_input_sha256', {})
    if len(digests.get('parallel', '')) != 64 or digests.get('parallel') != digests.get('autoregressive'):
        raise ValueError('Missing/mismatched original Q8 input provenance')
    with np.load(directory / 'test_predictions.npz', allow_pickle=False) as saved:
        labels, scores = saved['labels'], saved['parallel_probabilities']
    if labels.ndim != 1 or scores.shape != (len(labels), 5) or not np.isfinite(scores).all() \
            or np.any(scores < 0) or np.any(scores > 1) or not np.allclose(scores.sum(1), 1, atol=1e-5):
        raise ValueError('Invalid original Q8 scores')
    if not np.array_equal(np.bincount(labels, minlength=5), np.bincount(reference['labels'], minlength=5)):
        raise ValueError('Q8 reference classes/test counts differ')
    return dict(labels=labels, probabilities=scores), dict(source_manifest_sha256=SOURCE_SHA,
        ordered_input_sha256=digests['parallel'], event_ids_available=False,
        note='Canonical source manifest and class counts verified; old scores have no per-event IDs')


def compare(runs, output, *, expected_events=364441, q8_reference=None):
    if not runs or len({name for name, _ in runs}) != len(runs):
        raise ValueError("Supply distinct named runs")
    data = [(name, load_predictions(path)) for name, path in runs]
    reference = data[0][1]
    if len(reference["labels"]) != expected_events:
        raise ValueError(f"Expected {expected_events} test events; found {len(reference['labels'])}")
    for name, arrays in data[1:]:
        for key in ("source_file", "event_index", "labels"):
            if not np.array_equal(reference[key], arrays[key]):
                raise ValueError(f"{name}: test events or labels differ; refusing an unmatched comparison")
    identity_models = [name for name, _ in data]
    q8_provenance = None
    if q8_reference is not None:
        if 'Q8 parallel' in identity_models:
            raise ValueError('Q8 reference was supplied twice')
        arrays, q8_provenance = load_q8_reference(q8_reference, reference)
        data.append(('Q8 parallel', arrays))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    figure, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    rows, numeric = [], {}
    palette = plt.get_cmap("tab10").colors
    for index, (name, arrays) in enumerate(data):
        for label, (class_name, display) in enumerate(zip(NAMES, DISPLAY)):
            fpr, tpr, threshold = roc_curve(arrays["labels"] == label, arrays["probabilities"][:, label])
            area = float(auc(fpr, tpr))
            axes.flat[label].plot(fpr, tpr, color=palette[index % len(palette)], lw=2,
                                 label=f"{name} (AUC={area:.4f})")
            rows.append(dict(model=name, class_name=class_name, auc=area, test_events=expected_events))
            numeric.update({f"run{index}_{class_name}_{key}": value
                            for key, value in (("fpr", fpr), ("tpr", tpr), ("thresholds", threshold))})
    for axis, display in zip(axes.flat, DISPLAY):
        axis.plot([0, 1], [0, 1], "--", color="0.5", lw=1)
        axis.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False Positive Rate",
                 ylabel="True Positive Rate", title=f"{display} vs rest")
        axis.grid(alpha=0.2)
        axis.legend(loc="lower right", fontsize=9)
    axes.flat[-1].set_visible(False)
    for extension in ("png", "pdf"):
        figure.savefig(output / f"multiclass_roc_comparison.{extension}", dpi=200)
    plt.close(figure)
    columns = min(3, len(data))
    figure, grid = plt.subplots((len(data) + columns - 1) // columns, columns,
        figsize=(6 * columns, 5.7 * ((len(data) + columns - 1) // columns)), squeeze=False,
        constrained_layout=True)
    matrices = {}
    for axis, (name, arrays) in zip(grid.flat, data):
        matrix = confusion_matrix(arrays["labels"], arrays["probabilities"].argmax(1), labels=range(5), normalize="true")
        matrices[name] = matrix.tolist()
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        axis.set(xticks=range(5), yticks=range(5), xticklabels=DISPLAY, yticklabels=DISPLAY,
                 xlabel="Predicted class", ylabel="True class", title=f"{name} | N = {expected_events:,}")
        plt.setp(axis.get_xticklabels(), rotation=40, ha="right")
        for i in range(5):
            for j in range(5):
                axis.text(j, i, f"{100 * matrix[i, j]:.1f}%", ha="center", va="center",
                          color="white" if matrix[i, j] > 0.5 else "black", fontsize=10)
    for axis in list(grid.flat)[len(data):]:
        axis.set_visible(False)
    figure.colorbar(image, ax=list(grid.flat)[:len(data)], label="Fraction of true class", shrink=0.85)
    for extension in ("png", "pdf"):
        figure.savefig(output / f"multiclass_confusion_comparison.{extension}", dpi=200)
    plt.close(figure)
    with (output / "per_class_auc.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(output / "roc_arrays.npz", **numeric)
    with (output / "roc_points.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "class", "false_positive_rate", "true_positive_rate", "threshold"])
        for index, (name, _) in enumerate(data):
            for class_name in NAMES:
                arrays = [numeric[f"run{index}_{class_name}_{key}"] for key in ("fpr", "tpr", "thresholds")]
                writer.writerows((name, class_name, x, y, threshold) for x, y, threshold in zip(*arrays, strict=True))
    (output / "comparison_summary.json").write_text(json.dumps(dict(
        matched_test_events=q8_reference is None, test_events=expected_events, classes=NAMES,
        event_identity_verified_models=identity_models, legacy_q8_provenance=q8_provenance,
        inputs={name: str(Path(path).resolve()) for name, path in runs}, per_class_auc=rows,
        normalized_confusion_matrices=matrices), indent=2) + "\n")
    print(f"PASS identical test identities/labels; FPR/TPR ROC and confusion plots: {output}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=TEST_EVALUATION_DIRECTORY")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument('--q8-reference-dir', type=Path,
        help='Optional older paired Q8 evaluation; requires the exact canonical manifest checksum')
    args = parser.parse_args()
    if any("=" not in value for value in args.run):
        parser.error("Every --run must be NAME=PATH")
    compare([tuple(value.split("=", 1)) for value in args.run], args.output_dir, q8_reference=args.q8_reference_dir)


if __name__ == "__main__":
    main()
