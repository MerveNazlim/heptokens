#!/usr/bin/env python3
"""Compare parallel- and autoregressive-pretrained Q8 downstream classifiers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score, roc_curve,
)

from evaluate_q1_hierarchical_flat_hzz import (
    install_hydra_checkpoint_compatibility, write_plots,
)


RUNS = {
    "parallel": ("Q8 parallel", "#0072B2", "-"),
    "autoregressive": ("Q8 autoregressive", "#D55E00", "--"),
}
DISPLAY_NAMES = {
    "ggf": "ggF", "vbf": "VBF", "vh": "VH",
    "tth": r"$t\bar{t}H + tH$", "zz_continuum": r"$ZZ^{(*)}$ continuum",
}


def read_manifest(prepared_dir: Path, task: str) -> tuple[dict, list[str]]:
    manifest = json.loads((prepared_dir / "manifest.json").read_text())
    shape = manifest.get("token_shape")
    if not shape or len(shape) != 2 or shape[1] != 8:
        raise ValueError(f"Expected grouped Q8 token_shape, found {shape}")
    if task == "multiclass":
        classes = sorted(manifest["classes"], key=lambda item: int(item["label"]))
        if [int(item["label"]) for item in classes] != list(range(len(classes))):
            raise ValueError("Class labels must be contiguous from zero")
        names = [item["name"] for item in classes]
        if names != ["ggf", "vbf", "vh", "tth", "zz_continuum"]:
            raise ValueError(f"Expected the five production classes, found {names}")
    else:
        if manifest["background"]["label"] != 0 or manifest["signal"]["label"] != 1:
            raise ValueError("Binary labels must be background=0, signal=1")
        names = ["zz_continuum", "ggf"]
    if int(manifest["split_counts"]["test"]["total"]) <= 0:
        raise ValueError("Empty test split")
    return manifest, names


def validate_predictions(labels: np.ndarray, probabilities: np.ndarray, n_classes: int) -> None:
    if labels.ndim != 1 or probabilities.shape != (labels.size, n_classes):
        raise ValueError("Prediction shape does not match labels/classes")
    if set(np.unique(labels)) != set(range(n_classes)):
        raise ValueError("Test labels must include every expected class and no other labels")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Invalid classifier probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Softmax probabilities do not sum to one")


def predict_with_digest(model, dataloader, device) -> tuple[np.ndarray, np.ndarray, str]:
    import torch
    from heptokens.data.sequence import sequence_batch_to_device

    digest = hashlib.sha256()
    probabilities, labels = [], []
    with torch.inference_mode():
        for index, batch in enumerate(dataloader, 1):
            # Verify both passes see identical ordered inputs, not only matching labels.
            for key in ("tokens", "mask", "type_ids", "labels"):
                value = batch[key].detach().cpu().numpy()
                digest.update(f"{key}:{value.dtype}:{value.shape}".encode())
                digest.update(value.tobytes())
            prediction = model.predict_step(sequence_batch_to_device(batch, str(device)))
            probabilities.append(prediction["probabilities"].detach().cpu().numpy())
            labels.append(prediction["label"].detach().cpu().numpy())
            if index % 100 == 0:
                print(f"Evaluated {index:,} test batches", flush=True)
    if not labels:
        raise ValueError("No test batches were loaded")
    return np.concatenate(probabilities), np.concatenate(labels), digest.hexdigest()


def save_figure(figure, directory: Path, name: str) -> None:
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(directory / f"{name}.{suffix}", dpi=240)
    plt.close(figure)


def plot_roc(curves: list[dict], output_dir: Path, name: str, title: str = "") -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    for curve in curves:
        axis.plot(
            curve["fpr"], curve["tpr"], color=curve["color"],
            linestyle=curve["linestyle"], linewidth=2.2,
            label=f"{curve['name']} (AUC={curve['auc']:.3f})",
        )
    axis.plot([0, 1], [0, 1], "--", color="0.5", linewidth=1)
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate", ylabel="True positive rate")
    if title:
        axis.set_title(title, fontsize=18)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="lower right", fontsize=12, frameon=False)
    save_figure(figure, output_dir, name)


def write_report(output_dir: Path, metadata: dict, arrays: dict[str, np.ndarray]) -> list[dict]:
    plt.style.use(hep.style.ROOT)
    task, names = metadata["task"], metadata["class_names"]
    labels = arrays["labels"]
    curves, rows = [], []
    for run, (display, color, linestyle) in RUNS.items():
        probabilities = arrays[f"{run}_probabilities"]
        validate_predictions(labels, probabilities, len(names))
        predictions = probabilities.argmax(axis=1)
        row = {
            "run": display, "checkpoint": metadata["checkpoints"][run],
            "test_events": int(labels.size),
            "accuracy": float(accuracy_score(labels, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        }
        if task == "binary":
            scores = probabilities[:, 1]
            fpr, tpr, _ = roc_curve(labels, scores)
            row["auc"] = float(roc_auc_score(labels, scores))
            n_background = int(np.count_nonzero(labels == 0))
            for efficiency in (50, 70, 80):
                row[f"rejection_at_{efficiency}pct"] = float(
                    1.0 / max(float(np.interp(efficiency / 100, tpr, fpr)), 1.0 / n_background)
                )
            curves.append({
                "name": display, "color": color, "linestyle": linestyle,
                "fpr": fpr, "tpr": tpr, "auc": row["auc"],
                "background_events": n_background,
            })
        else:
            from evaluate_grouped_hzz_multiclass import plot_confusion_matrix, plot_roc_curves

            row["macro_ovr_auc"] = float(roc_auc_score(labels, probabilities, multi_class="ovr", average="macro"))
            row["weighted_ovr_auc"] = float(roc_auc_score(labels, probabilities, multi_class="ovr", average="weighted"))
            raw = confusion_matrix(labels, predictions, labels=range(len(names)))
            normalized = confusion_matrix(labels, predictions, labels=range(len(names)), normalize="true")
            row["raw_confusion_matrix"] = raw.tolist()
            row["normalized_confusion_matrix"] = normalized.tolist()
            directory = output_dir / run
            directory.mkdir(exist_ok=True)
            with plt.rc_context({"font.size": 13, "axes.labelsize": 16, "axes.titlesize": 16,
                                 "xtick.labelsize": 12, "ytick.labelsize": 12}):
                plot_confusion_matrix(normalized, [DISPLAY_NAMES[n] for n in names], directory)
                aucs = plot_roc_curves(labels, probabilities, [DISPLAY_NAMES[n] for n in names], directory)
            row["per_class_auc"] = {n: aucs[DISPLAY_NAMES[n]] for n in names}
        rows.append(row)
    if task == "binary":
        write_plots(output_dir, curves)
        plot_roc(curves, output_dir, "roc")
    else:
        for class_index, class_name in enumerate(names):
            class_curves = []
            for run, (display, color, linestyle) in RUNS.items():
                binary_labels = labels == class_index
                scores = arrays[f"{run}_probabilities"][:, class_index]
                fpr, tpr, _ = roc_curve(binary_labels, scores)
                class_curves.append({
                    "name": display, "color": color, "linestyle": linestyle,
                    "fpr": fpr, "tpr": tpr, "auc": float(roc_auc_score(binary_labels, scores)),
                })
            plot_roc(class_curves, output_dir, f"roc_{class_name}", f"{DISPLAY_NAMES[class_name]} vs rest")
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    scalar_fields = [key for key in rows[0] if key not in (
        "raw_confusion_matrix", "normalized_confusion_matrix", "per_class_auc",
    )]
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, scalar_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        metric = "auc" if task == "binary" else "macro_ovr_auc"
        print(f"{row['run']:20s} accuracy={row['accuracy']:.4f} {metric}={row[metric]:.4f}")
    print(f"Plots and metrics: {output_dir.resolve()}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("binary", "multiclass"))
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--parallel-checkpoint", type=Path)
    parser.add_argument("--autoregressive-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--replot-dir", type=Path, help="Use saved predictions; no model inference")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.replot_dir:
        metadata = json.loads((args.replot_dir / "evaluation_metadata.json").read_text())
        with np.load(args.replot_dir / "test_predictions.npz", allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        output_dir = args.output_dir or args.replot_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        write_report(output_dir, metadata, arrays)
        return
    for name in ("task", "prepared_dir", "parallel_checkpoint", "autoregressive_checkpoint", "output_dir"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required for inference")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Output directory is not empty; choose a new folder or --replot-dir")
    manifest, names = read_manifest(args.prepared_dir, args.task)
    checkpoints = {"parallel": args.parallel_checkpoint.resolve(), "autoregressive": args.autoregressive_checkpoint.resolve()}
    for path in checkpoints.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if checkpoints["parallel"] == checkpoints["autoregressive"]:
        raise ValueError("The two checkpoints must differ")

    import torch
    from heptokens.data.token_parquet import GroupedTokenParquetClassificationModule
    from heptokens.models.foundation_grouped_cls_classifier import LitGroupedCLSClassifier

    install_hydra_checkpoint_compatibility()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    datamodule = GroupedTokenParquetClassificationModule(
        prepared_dir=str(args.prepared_dir.resolve()), n_classes=len(names), seed=42,
        batch_size=args.batch_size, num_workers=0, stream_batch_size=4096,
        shuffle_buffer_size=8192, pin_memory=device.type == "cuda", persistent_workers=False,
        label_column="label",
    )
    vocabulary = datamodule.get_token_vocabulary()
    if vocabulary is None:
        raise ValueError("Classification shards have no token-vocabulary metadata")
    arrays, signatures, digests = {}, {}, {}
    for run, checkpoint in checkpoints.items():
        print(f"Loading {run}: {checkpoint}", flush=True)
        model = LitGroupedCLSClassifier.load_from_checkpoint(
            str(checkpoint), map_location="cpu", backbone_ckpt_path=None,
        )
        print("Full fine-tuned classifier weights restored (strict load).", flush=True)
        fields = ("n_classes", "max_quantizers", "vocab_size", "max_seq_length", "hidden_dim",
                  "num_layers", "num_heads", "quantizer_embedding_dim", "classifier_hidden_dim",
                  "freeze_backbone", "use_type_embedding", "use_position_embedding")
        signatures[run] = {key: model.hparams.get(key) for key in fields}
        signatures[run]["object_projection_mode"] = model.hparams.get("object_projection_mode", "linear")
        signatures[run]["class_weights"] = model.class_weights.tolist()
        if model.n_classes != len(names) or model.hparams.max_quantizers != 8:
            raise ValueError(f"Wrong class count or quantizer count in {checkpoint}")
        if model.hparams.max_seq_length != manifest["token_shape"][0]:
            raise ValueError("Checkpoint sequence length differs from the dataset")
        if model.hparams.vocab_size != vocabulary["vocab_size"]:
            raise ValueError("Checkpoint vocabulary size differs from the dataset")
        if len(signatures) == 2 and signatures["parallel"] != signatures["autoregressive"]:
            raise ValueError(f"Classifier architectures/settings differ: {signatures}")
        model.eval().to(device)
        probabilities, labels, digest = predict_with_digest(model, datamodule.test_dataloader(), device)
        validate_predictions(labels, probabilities, len(names))
        if labels.size != int(manifest["split_counts"]["test"]["total"]):
            raise ValueError("Not all test events were evaluated")
        if "labels" in arrays and (not np.array_equal(arrays["labels"], labels) or digest != digests["parallel"]):
            raise ValueError("The two passes saw different ordered test inputs")
        arrays["labels"] = labels
        arrays[f"{run}_probabilities"] = probabilities
        digests[run] = digest
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"PASS: identical ordered inputs for all {arrays['labels'].size:,} test events", flush=True)
    metadata = {
        "task": args.task, "prepared_dir": str(args.prepared_dir.resolve()),
        "class_names": names, "checkpoints": {key: str(path) for key, path in checkpoints.items()},
        "model_settings": signatures, "ordered_input_sha256": digests,
        "batch_size": args.batch_size, "num_workers": 0, "seed": 42,
        "manifest_sha256": hashlib.sha256((args.prepared_dir / "manifest.json").read_bytes()).hexdigest(),
        "background_rejection_floor": "1 / number of test background events",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "test_predictions.npz", **arrays)
    (args.output_dir / "evaluation_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_report(args.output_dir, metadata, arrays)


if __name__ == "__main__":
    main()
