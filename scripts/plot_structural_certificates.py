"""Plot the frozen Conv-attention and rational-normalizer certificates."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "athena" / "results"
OUTPUT = ROOT / "paper_figures" / "appendix_structural_certificates"

COLORS = {
    "blue": "#326B8C",
    "green": "#2E7D6B",
    "orange": "#C97835",
    "red": "#B54B4B",
    "ink": "#18252D",
    "gray": "#66747C",
    "light": "#DCE4E8",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.5,
        "axes.edgecolor": COLORS["gray"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": COLORS["ink"],
        "ytick.color": COLORS["ink"],
        "text.color": COLORS["ink"],
        "axes.labelcolor": COLORS["ink"],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def direct_labels(ax, x, values, formatter, offset_points=6) -> None:
    for xpos, value in zip(x, values):
        ax.annotate(
            formatter(value),
            (xpos, value),
            xytext=(0, offset_points),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color=COLORS["ink"],
        )


def main() -> None:
    attention = load("conv_joint_attention_certificate_v1_summary.json")
    normalizer = load("rational_norm_safety_v1_summary.json")
    seeds = np.arange(3)
    labels = ["seed 0", "seed 1", "seed 2"]

    attention_error = np.array(
        [row["max_relative_l2_error"] for row in attention["checkpoints"]]
    )
    action_nrmse = np.array(
        [row["action_nrmse"] for row in normalizer["checkpoints"]]
    )
    range_coverage = 100 * np.array(
        [
            row["minimum_site_fraction_rows_v_in_0p1_10"]
            for row in normalizer["checkpoints"]
        ]
    )
    scale_coverage = 100 * np.array(
        [
            row["minimum_site_fraction_rows_local_scale_error_at_or_below_threshold"]
            for row in normalizer["checkpoints"]
        ]
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.15))

    ax = axes[0]
    ax.scatter(seeds, attention_error, s=48, color=COLORS["blue"], zorder=3)
    ax.axhline(1e-6, color=COLORS["red"], lw=1.2, ls="--", label="gate $10^{-6}$")
    ax.set_yscale("log")
    ax.set_ylim(1e-17, 1e-5)
    ax.set_xticks(seeds, labels)
    ax.set_ylabel("maximum relative $L_2$ error")
    ax.set_title("a  Exact joint attention")
    ax.grid(axis="y", color=COLORS["light"], lw=0.7)
    ax.legend(frameon=False, loc="upper left", fontsize=7.2)
    direct_labels(ax, seeds, attention_error, lambda x: f"{x:.2g}", 5)
    ax.text(
        0.03,
        0.05,
        "4,608 head-input cases, all finite",
        transform=ax.transAxes,
        fontsize=7.2,
        color=COLORS["gray"],
    )

    ax = axes[1]
    ax.scatter(seeds, action_nrmse, s=48, color=COLORS["green"], zorder=3)
    ax.axhline(1e-3, color=COLORS["red"], lw=1.2, ls="--", label="gate $10^{-3}$")
    ax.set_yscale("log")
    ax.set_ylim(1e-7, 3e-3)
    ax.set_xticks(seeds, labels)
    ax.set_ylabel("action NRMSE, deployed vs fp64 RNorm")
    ax.set_title("b  Deployed rational fidelity")
    ax.grid(axis="y", color=COLORS["light"], lw=0.7)
    ax.legend(frameon=False, loc="upper left", fontsize=7.2)
    direct_labels(ax, seeds, action_nrmse, lambda x: f"{x:.2g}", 5)
    ax.text(
        0.03,
        0.05,
        "544.5M norm rows, all finite",
        transform=ax.transAxes,
        fontsize=7.2,
        color=COLORS["gray"],
    )

    ax = axes[2]
    width = 0.34
    ax.bar(
        seeds - width / 2,
        range_coverage,
        width,
        color=COLORS["orange"],
        label="$v\\in[0.1,10]$",
        zorder=3,
    )
    ax.bar(
        seeds + width / 2,
        scale_coverage,
        width,
        color=COLORS["blue"],
        label="scale error $\\leq3.4\\%$",
        zorder=3,
    )
    ax.axhline(99, color=COLORS["red"], lw=1.2, ls="--", label="secondary gate 99%")
    ax.set_ylim(50, 102)
    ax.set_xticks(seeds, labels)
    ax.set_ylabel("minimum site coverage (%)")
    ax.set_title("c  Exact-RMS approximation scope")
    ax.grid(axis="y", color=COLORS["light"], lw=0.7, zorder=0)
    ax.legend(frameon=False, loc="center right", fontsize=6.8)
    for xpos, value in zip(seeds - width / 2, range_coverage):
        ax.text(xpos, value + 1.2, f"{value:.2f}", ha="center", fontsize=7.0)
    for xpos, value in zip(seeds + width / 2, scale_coverage):
        ax.text(xpos, value + 1.2, f"{value:.2f}", ha="center", fontsize=7.0)

    fig.tight_layout(w_pad=2.4)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=260, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
