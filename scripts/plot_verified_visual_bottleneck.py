#!/usr/bin/env python3
"""Generate the paper figure from the verified visual-bottleneck artifact."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "artifacts" / "visual_bottleneck" / "verify.py"
SUMMARY = ROOT / "athena" / "results" / "visual_taskmap_blind_rank96_summary.json"
OUTPUT = ROOT / "paper_figures" / "visual_bottleneck_verified"

COLORS = {
    "blue": "#326B8C",
    "green": "#2E7D6B",
    "gray": "#8B989F",
    "pale": "#DCE4E8",
    "ink": "#18252D",
    "red": "#B54B4B",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.2,
        "axes.titlesize": 9.2,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.2,
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


def verified_values() -> tuple[dict, dict]:
    process = subprocess.run(
        [sys.executable, str(VERIFIER)],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    verified = json.loads(process.stdout)
    if verified.get("artifact_verified") is not True:
        raise RuntimeError("Visual-bottleneck artifact did not verify")
    summary = json.loads(SUMMARY.read_text())
    if summary.get("schema") != "xvla-corrected-blind-subspace-summary-v2":
        raise RuntimeError("Refusing an uncorrected visual-bottleneck summary")
    return verified, summary


def main() -> None:
    verified, summary = verified_values()
    point = verified["point"]
    intervals = verified["bootstrap"]["intervals_95"]

    learned = np.array([point["activation_retention"], point["jacobian_retention"]])
    random_retention = np.array(point["random_retention"])
    values = np.concatenate([learned, random_retention])
    labels = ["energy", "Jac.", "rnd 1", "rnd 2", "rnd 3"]
    colors = [COLORS["blue"], COLORS["green"], COLORS["gray"], COLORS["gray"], COLORS["gray"]]

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.35), gridspec_kw={"width_ratios": [1.15, 1.0]})

    ax = axes[0]
    x = np.arange(len(values))
    ax.bar(x, 100 * values, color=colors, width=0.68, zorder=3)
    interval_names = ("activation_retention", "jacobian_retention")
    for index, interval_name in enumerate(interval_names):
        low, high = intervals[interval_name]
        ax.errorbar(
            index,
            100 * values[index],
            yerr=[[100 * (values[index] - low)], [100 * (high - values[index])]],
            fmt="none",
            ecolor=COLORS["ink"],
            elinewidth=0.9,
            capsize=2.5,
            zorder=4,
        )
    for index, value in enumerate(values):
        ax.text(index, 100 * value + 2.2, f"{100*value:.1f}", ha="center", fontsize=7.0)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 108)
    ax.set_ylabel("baseline successes retained (%)")
    ax.set_title("a  Structured vs random")
    ax.grid(axis="y", color=COLORS["pale"], lw=0.7, zorder=0)

    ax = axes[1]
    categories = np.arange(4)
    labels = ["seed 0", "seed 1", "seed 2", "pooled"]
    activation = np.array(
        [summary["by_checkpoint"][str(seed)]["activation_retention"] for seed in range(3)]
        + [summary["pooled"]["activation_retention"]]
    )
    jacobian = np.array(
        [summary["by_checkpoint"][str(seed)]["jacobian_retention"] for seed in range(3)]
        + [summary["pooled"]["jacobian_retention"]]
    )
    ax.plot(categories, 100 * activation, marker="o", lw=1.7, color=COLORS["blue"], label="activation energy")
    ax.plot(categories, 100 * jacobian, marker="o", lw=1.7, color=COLORS["green"], label="action Jacobian")
    ax.axhline(85, color=COLORS["red"], lw=1.0, ls="--")
    ax.text(2.95, 86.1, "frozen selectivity gate", ha="right", color=COLORS["red"], fontsize=6.6)
    ax.set_xticks(categories, labels)
    ax.set_ylim(82, 102)
    ax.set_ylabel("baseline successes retained (%)")
    ax.set_title("b  Across checkpoints")
    ax.grid(axis="y", color=COLORS["pale"], lw=0.7, zorder=0)
    ax.legend(frameon=False, fontsize=6.6, loc="lower left")

    fig.text(
        0.5,
        0.008,
        "rank 96 fixed before corrected outcomes, tasks 8-9 absent from corrected projector construction",
        ha="center",
        fontsize=6.5,
        color=COLORS["gray"],
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1), w_pad=2.4)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=260, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
