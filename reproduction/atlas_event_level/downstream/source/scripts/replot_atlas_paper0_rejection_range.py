#!/usr/bin/env python3
"""Axis-only copy of the saved job-520 validation+test rejection plot."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np


def main():
    root = (Path(__file__).resolve().parents[1] / "output/pdf"
            / "atlas_hzz_paper0_4l_job520_validation_test")
    stem = root / "background_rejection_ymax60000"
    outputs = [stem.with_suffix(ext) for ext in (".pdf", ".png", ".json")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Refusing to replace an existing plot copy")
    report = json.loads((root / "plot_provenance.json").read_text())
    sources = [root / "plot_provenance.json"]
    arms = {
        "scratch": ("Tokenized, from scratch", "#0072B2", "--"),
        "finetune": ("Tokenized, fine-tuned", "#D55E00", "-"),
        "continuous_finetune": ("Continuous, fine-tuned", "#009E73", "-"),
    }
    plt.style.use(hep.style.ROOT)
    fig, ax = plt.subplots(figsize=(8, 6))
    for arm, (name, color, style) in arms.items():
        path = root / f"{arm}_roc_arrays.npz"
        sources.append(path)
        with np.load(path, allow_pickle=False) as curve:
            fpr, tpr = curve["fpr"], curve["tpr"]
        auc = float(np.trapz(tpr, fpr))
        np.testing.assert_allclose(auc, report["runs"][arm]["auc"], atol=1e-12, rtol=0)
        positive = fpr > 0
        ax.plot(tpr[positive], 1 / fpr[positive], color=color, linestyle=style,
                linewidth=2.2, label=f"{name} (AUC={auc:.3f})")
    ax.set_xlim(0, 1)
    ax.set_yscale("log")
    ax.set_ylim(1, 60000)
    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Background rejection")
    ax.set_title("Validation + test (diagnostic)", fontsize=14, pad=12)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", fontsize=12, frameon=False, labelspacing=0.4)
    fig.tight_layout()
    fig.savefig(outputs[0], dpi=240)
    fig.savefig(outputs[1], dpi=240)
    plt.close(fig)
    provenance = dict(report, axis_only_replot=True, no_new_inference=True,
                      xlim=[0, 1], ylim=[1, 60000], yscale="log",
                      sources_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in sources})
    outputs[2].write_text(json.dumps(provenance, indent=2) + "\n")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
