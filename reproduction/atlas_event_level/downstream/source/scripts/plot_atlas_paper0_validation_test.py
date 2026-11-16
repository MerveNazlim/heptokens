#!/usr/bin/env python3
"""Inference-only validation+test diagnostic for the three existing job-520 fits.

Validation selected these checkpoints: this is NOT held-out test performance.
Original splits, checkpoints, saved test scores and test-only figures are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import torch
from torch.utils.data import DataLoader, Subset

from evaluate_q1_hierarchical_flat_hzz import install_hydra_checkpoint_compatibility
from train_atlas_paper0_q1_4l import make_model, read_dataset, _load_checkpoint_state

ARMS = {
    "scratch": ("Tokenized, from scratch", "#0072B2", "--"),
    "finetune": ("Tokenized, fine-tuned", "#D55E00", "-"),
    "continuous_finetune": ("Continuous, fine-tuned", "#009E73", "-"),
}
WARNING = "Validation + test diagnostic; validation was used to select checkpoints."


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict(model, dataset, label):
    model.eval()
    chunks, start = [], time.monotonic()
    with torch.inference_mode():
        for i, batch in enumerate(DataLoader(dataset, batch_size=256, shuffle=False)):
            chunks.append(model(batch).softmax(dim=1)[:, 1].numpy())
            if (i + 1) % 10 == 0:
                print(f"{label}: {min((i+1)*256, len(dataset)):,}/{len(dataset):,} "
                      f"events, {time.monotonic()-start:.0f}s", flush=True)
    scores = np.concatenate(chunks)
    if scores.shape != (len(dataset),) or not np.isfinite(scores).all():
        raise ValueError("Incomplete or invalid inference")
    return scores


def verify_test(data, saved, report):
    for key, expected in (("source_file", data.sources), ("event_index", data.indices),
                          ("label", data.features["labels"].numpy())):
        np.testing.assert_array_equal(saved[key], expected)
    if sha_saved := report.get("prediction_sha256"):
        if sha_saved != saved["_file_sha256"]:
            raise ValueError("Original saved test predictions changed")
    if not np.isclose(roc_auc_score(saved["label"], saved["signal_score"]),
                      report["auc"], rtol=0, atol=1e-12):
        raise ValueError("Original test AUC mismatch")


def merge(val, test, test_saved, val_scores):
    val_ids = set(zip(val.sources.tolist(), val.indices.tolist()))
    test_ids = set(zip(test.sources.tolist(), test.indices.tolist()))
    if len(val_ids) != len(val) or len(test_ids) != len(test) or val_ids & test_ids:
        raise ValueError("Repeated event identities in validation/test")
    payload = dict(
        source_file=np.concatenate((val.sources, test.sources)),
        event_index=np.concatenate((val.indices, test.indices)),
        label=np.concatenate((val.features["labels"].numpy(), test_saved["label"])),
        signal_score=np.concatenate((val_scores, test_saved["signal_score"])),
        split=np.concatenate((np.full(len(val), 1, dtype=np.int8),
                              np.full(len(test), 2, dtype=np.int8))),
    )
    if np.bincount(payload["label"], minlength=2).tolist() != [2543, 93056]:
        raise ValueError("Unexpected combined cohort")
    return payload


def figures(out, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    plt.style.use(hep.style.ROOT)
    for kind in ("background_rejection", "roc"):
        fig, ax = plt.subplots(figsize=(8, 6))
        for arm, curve in curves.items():
            name, color, style = ARMS[arm]
            fpr, tpr = curve["fpr"], curve["tpr"]
            if kind == "background_rejection":
                positive = fpr > 0
                x, y = tpr[positive], 1 / fpr[positive]
            else:
                x, y = fpr, tpr
            ax.plot(x, y, color=color, linestyle=style, linewidth=2.2,
                    label=f"{name} (AUC={curve['auc']:.3f})")
        ax.set_xlim(0, 1)
        if kind == "background_rejection":
            ax.set_yscale("log")
            ax.set_ylim(1, None)
            ax.set_xlabel("Signal efficiency")
            ax.set_ylabel("Background rejection")
            loc = "upper right"
        else:
            ax.plot([0, 1], [0, 1], "--", color="0.55", linewidth=1.2, label="Random")
            ax.set_ylim(0, 1)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            loc = "lower right"
        ax.set_title("Validation + test (diagnostic)", fontsize=14, pad=12)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc=loc, fontsize=12, frameon=False, labelspacing=0.4)
        fig.tight_layout()
        for extension in ("pdf", "png"):
            fig.savefig(out / f"{kind}.{extension}", dpi=240)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--q1-data", type=Path, required=True)
    parser.add_argument("--continuous-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Refusing to overwrite any existing output")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    install_hydra_checkpoint_compatibility()
    curves, reports, canonical = {}, {}, None
    datasets = {}
    for arm in ARMS:
        continuous = arm == "continuous_finetune"
        root = args.results_root / arm
        report = json.loads((root / "evaluation_summary.json").read_text())
        checkpoint = root / "best.ckpt"
        if report["arm"] != arm or sha(checkpoint) != report["best_checkpoint_sha256"]:
            raise ValueError(f"Wrong selected checkpoint: {arm}")
        data_root = args.continuous_data if continuous else args.q1_data
        if sha(data_root / "manifest.json") != report["manifest_sha256"]:
            raise ValueError(f"Wrong input manifest: {arm}")
        if continuous not in datasets:
            datasets[continuous] = read_dataset(data_root, report["manifest_sha256"], continuous=continuous)
        manifest, sets, input_spec, weights = datasets[continuous]
        if manifest["membership_sha256"] != report["membership_sha256"]:
            raise ValueError("Wrong selected membership")
        with np.load(root / "test_predictions.npz", allow_pickle=False) as data:
            saved = {key: data[key] for key in data.files}
        saved["_file_sha256"] = sha(root / "test_predictions.npz")
        verify_test(sets["test"], saved, report)
        selected = _load_checkpoint_state(checkpoint)
        if selected["epoch"] != report["best_epoch"]:
            raise ValueError("Wrong selected epoch")
        model = make_model(input_spec, weights, None, continuous=continuous)
        model.load_state_dict(selected["state_dict"], strict=True)
        # CUDA used medium matmul precision; CPU FP32 is not bit-identical.
        # Check the selected model and then rescore BOTH splits on the same backend.
        check = np.concatenate([np.flatnonzero(saved["label"] == label)[
            np.linspace(0, (saved["label"] == label).sum()-1, 128, dtype=int)] for label in (0, 1)])
        check_scores = predict(model, Subset(sets["test"], check), f"{arm} reproducibility")
        np.testing.assert_allclose(check_scores, saved["signal_score"][check], rtol=2e-3, atol=2e-3)
        val_scores = predict(model, sets["val"], f"{arm} validation")
        test_scores = predict(model, sets["test"], f"{arm} test")
        rescore_auc = float(roc_auc_score(saved["label"], test_scores))
        if abs(rescore_auc-report["auc"]) > 1e-4:
            raise ValueError("CPU rescoring changed test AUC by more than numerical tolerance")
        combined = merge(sets["val"], sets["test"], saved | {"signal_score": test_scores}, val_scores)
        identities = tuple(combined[k] for k in ("source_file", "event_index", "label", "split"))
        if canonical is None:
            canonical = identities
        else:
            for got, expected in zip(identities, canonical):
                np.testing.assert_array_equal(got, expected)
        np.savez_compressed(out / f"{arm}_validation_test_predictions.npz", **combined)
        fpr, tpr, thresholds = roc_curve(combined["label"], combined["signal_score"], drop_intermediate=False)
        curves[arm] = dict(fpr=fpr, tpr=tpr, auc=float(roc_auc_score(combined["label"], combined["signal_score"])))
        np.savez_compressed(out / f"{arm}_roc_arrays.npz", fpr=fpr, tpr=tpr, thresholds=thresholds)
        reports[arm] = dict(auc=curves[arm]["auc"], original_gpu_test_auc=report["auc"],
            rescored_cpu_test_auc=rescore_auc,
            full_test_max_abs_score_difference=float(np.max(np.abs(test_scores-saved["signal_score"]))),
            validation_auc=float(roc_auc_score(sets["val"].features["labels"].numpy(), val_scores)),
            checkpoint_sha256=report["best_checkpoint_sha256"], selected_epoch=report["best_epoch"],
            test_predictions_sha256=saved["_file_sha256"], manifest_sha256=report["manifest_sha256"],
            membership_sha256=report["membership_sha256"],
            check_max_abs_score_difference=float(np.max(np.abs(check_scores-saved["signal_score"][check]))))
        print(f"PASS {arm}: combined AUC={curves[arm]['auc']:.9f}", flush=True)
        del model, selected
    figures(out, curves)
    (out / "plot_provenance.json").write_text(json.dumps(dict(
        warning=WARNING, sample="validation+test", held_out=False, no_retraining=True,
        inference_note="Both splits rescored on CPU FP32; original GPU test artifacts unchanged.",
        training_events_included=False, evaluation_weighting="unweighted", background_events=2543,
        signal_events=93056, total_events=95599, zero_fpr_policy="omit; at least one passing background",
        torch_version=torch.__version__, device="cpu", runs=reports,
    ), indent=2) + "\n")
    print(out.resolve(), flush=True)


if __name__ == "__main__":
    main()
