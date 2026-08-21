#!/usr/bin/env python3
"""Plot the verified instruction-embedding-path ablation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "artifacts/instruction_embedding_ablation_v1/verify.py"
PDF = ROOT / "paper_figures/instruction_embedding_ablation.pdf"
PNG = ROOT / "paper_figures/instruction_embedding_ablation.png"


def verified_result() -> dict:
    completed = subprocess.run(
        [sys.executable, str(VERIFY), "--root", str(ROOT), "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def main() -> None:
    result = verified_result()
    checkpoints = result["by_checkpoint"]
    pooled = result["pooled"]
    tasks = result["by_task"]
    bootstrap = result["bootstrap_interval"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.3,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "legend.fontsize": 7.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    full_color = "#2166AC"
    zero_color = "#B2182B"
    gap_color = "#1B7837"
    pooled_color = "#762A83"
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.15, 2.48), gridspec_kw={"width_ratios": [0.93, 1.45]})

    labels = ["Ckpt 0", "Ckpt 1", "Ckpt 2", "Pooled"]
    full = [100 * row["full_rate"] for row in checkpoints] + [100 * pooled["full_rate"]]
    zeroed = [100 * row["zeroed_rate"] for row in checkpoints] + [100 * pooled["zeroed_rate"]]
    x = np.arange(len(labels))
    width = 0.34
    bars_full = ax0.bar(x - width / 2, full, width, color=full_color, label="Full lexical path")
    bars_zero = ax0.bar(x + width / 2, zeroed, width, color=zero_color, hatch="///", label="Post-lookup vectors zeroed")
    for bars in (bars_full, bars_zero):
        for bar in bars:
            ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.0, f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=7.0)
    ax0.set_ylim(0, 106)
    ax0.set_ylabel("Closed-loop success (%)")
    ax0.set_xticks(x, labels)
    ax0.grid(axis="y", color="#D9D9D9", linewidth=0.55, zorder=0)
    ax0.set_axisbelow(True)
    ax0.set_title("(a) Identical correct token IDs", loc="left", fontweight="bold")
    ax0.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.48, -0.18), ncol=1)

    task_gaps = np.asarray([100 * row["gap"] for row in tasks])
    tx = np.arange(10)
    ax1.bar(tx, task_gaps, width=0.68, color=gap_color, alpha=0.9)
    for index, value in enumerate(task_gaps):
        ax1.text(index, value + 2.0, f"{value:.0f}", ha="center", va="bottom", fontsize=6.7)
    pooled_x = 10.6
    pooled_gap = 100 * pooled["gap"]
    low, high = [100 * value for value in bootstrap]
    ax1.errorbar(
        [pooled_x],
        [pooled_gap],
        yerr=[[pooled_gap - low], [high - pooled_gap]],
        fmt="D",
        color=pooled_color,
        ecolor=pooled_color,
        elinewidth=1.5,
        capsize=3,
        markersize=5,
        label="Pooled gap, state-cluster 95% CI",
    )
    ax1.text(pooled_x, high + 3.0, f"{pooled_gap:.1f}", ha="center", va="bottom", color=pooled_color, fontsize=7.0, fontweight="bold")
    ax1.axhline(20, color="#555555", linewidth=0.9, linestyle="--", label="Frozen 20-point gate")
    ax1.set_xlim(-0.65, 11.15)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Full minus zeroed success (points)")
    ax1.set_xticks(list(tx) + [pooled_x], [f"T{i}" for i in tx] + ["Pooled"])
    ax1.grid(axis="y", color="#D9D9D9", linewidth=0.55, zorder=0)
    ax1.set_axisbelow(True)
    ax1.set_title("(b) Positive gap for every task", loc="left", fontweight="bold")
    ax1.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.52, -0.18), ncol=2)

    fig.subplots_adjust(left=0.075, right=0.995, top=0.88, bottom=0.27, wspace=0.28)
    PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF, bbox_inches="tight")
    fig.savefig(PNG, dpi=240, bbox_inches="tight")
    print(PDF)
    print(PNG)


if __name__ == "__main__":
    main()
