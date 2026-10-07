"""Shared, publication-sized styling for TREASURE paper figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

OBJECTS = ("jets", "electrons", "muons", "photons", "taus", "tracks")
OBJECT_LABELS = ("Jets", "Electrons", "Muons", "Photons", "Taus", "Tracks")
OBJECT_COLORS = dict(
    zip(OBJECTS, ("#235789", "#D55E00", "#009E73", "#CC79A7", "#7851A9", "#666666"))
)
QUANTIZER_COLORS = {1: "#666666", 2: "#B45289", 4: "#D55E00", 6: "#009E73", 8: "#235789"}
CODEBOOK_STYLES = {
    2048: (0, (1, 1)),
    4096: "-",
    8192: "--",
    16384: (0, (6, 2, 1, 2)),
    32768: (0, (10, 3)),
}
DIMENSION_MARKERS = {8: "o", 16: "s"}
PT_LABEL = r"Original reconstructed $p_T$ [GeV]"
RESPONSE_LABEL = r"Relative response width $R_{p_T}$ [%]"


def paper_style():
    return mpl.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 1.0,
            "lines.linewidth": 1.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def style_axis(ax):
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.tick_params(which="major", length=5)
    ax.tick_params(which="minor", length=2.5)
    ax.grid(axis="y", which="major", color="#E4E7EA", linewidth=0.7)
    ax.set_axisbelow(True)


def capacity_curve_style(q, k, d, *, selected):
    return {
        "color": "#000000" if selected else QUANTIZER_COLORS[q],
        "linestyle": CODEBOOK_STYLES[k],
        "marker": "*" if selected else DIMENSION_MARKERS[d],
        "markersize": 6 if selected else 3.5,
        "linewidth": 2.6 if selected else 1.2,
        "alpha": 1 if selected else 0.7,
        "zorder": 4 if selected else 2,
    }


def capacity_legends(fig, configs, highlighted_objects, x_positions, *, top=0.185, bottom=0.005):
    from matplotlib.lines import Line2D

    blocks = (
        (
            [
                Line2D([], [], color=QUANTIZER_COLORS[q], lw=2, label=str(q))
                for q in sorted({c[0] for c in configs})
            ],
            r"Residual depth $N_q$",
            2.2,
        ),
        (
            [
                Line2D([], [], color="#444444", ls=CODEBOOK_STYLES[k], label=f"{k:,}")
                for k in sorted({c[1] for c in configs})
            ],
            r"Codebook size $K$",
            4.2,
        ),
        (
            [
                Line2D(
                    [], [], color="#444444", ls="none", marker=DIMENSION_MARKERS[d], label=str(d)
                )
                for d in sorted({c[2] for c in configs})
            ],
            r"Latent dimension $d_c$",
            1.2,
        ),
    )
    for x, (handles, title, handlelength) in zip(x_positions, blocks):
        fig.legend(
            handles=handles,
            title=title,
            title_fontsize=10,
            loc="upper center",
            bbox_to_anchor=(x, top),
            ncol=2,
            frameon=False,
            handlelength=handlelength,
            columnspacing=1.7,
            borderaxespad=0,
            labelspacing=0.5,
        )
    if highlighted_objects:
        fig.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    color="#000000",
                    ls=CODEBOOK_STYLES[k],
                    marker="*",
                    lw=2.6,
                    label=", ".join(labels) + rf": $N_q={q},\ K={k},\ d_c={d}$",
                )
                for (q, k, d), labels in highlighted_objects.items()
            ],
            title="Selected configuration",
            title_fontsize=9,
            fontsize=8,
            ncol=len(highlighted_objects),
            loc="lower center",
            bbox_to_anchor=(sum(x_positions) / len(x_positions), bottom),
            frameon=False,
            handlelength=3.2,
        )


def save_figure(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(stem.with_suffix(suffix), facecolor="white")
