"""Plot the prospectively frozen closed-loop instruction-necessity result."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "athena/results/instruction_necessity_v1_summary.json"
OUTPUT = ROOT / "paper_figures/instruction_necessity_closed_loop"

COLORS = {
    "correct": "#214761",
    "copresent": "#D66B3D",
    "empty": "#9AA6AD",
    "empty_gap": "#2E8B78",
    "grid": "#DDE4E8",
    "ink": "#17242D",
}


def main() -> None:
    result = json.loads(SUMMARY.read_text())
    if not (
        result["identity_validated"]
        and result["overall_pass"]
        and result["claim_eligible"]
        and all(result["gates"].values())
    ):
        raise RuntimeError("Refusing to plot a nonpassing instruction-necessity summary")

    conditions = (
        "correct_prompt",
        "copresent_distractor_prompt",
        "empty_instruction",
    )
    condition_labels = ("Correct", "Co-present\ncontrol", "Empty")
    condition_colors = (
        COLORS["correct"],
        COLORS["copresent"],
        COLORS["empty"],
    )
    counts = np.array(
        [
            [
                result["by_checkpoint"][str(seed)]["copresent_distractor_prompt"][
                    "correct_successes"
                ],
                result["by_checkpoint"][str(seed)]["copresent_distractor_prompt"][
                    "control_successes"
                ],
                result["by_checkpoint"][str(seed)]["empty_instruction"][
                    "control_successes"
                ],
            ]
            for seed in range(3)
        ],
        dtype=int,
    )
    gaps = {
        control: np.array(
            [result["by_task"][str(task)][control]["paired_gap"] * 100 for task in range(10)]
        )
        for control in conditions[1:]
    }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.6,
            "axes.titlesize": 8.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 7.8,
            "axes.edgecolor": "#78858D",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "text.color": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 2.45),
        gridspec_kw={"width_ratios": [0.92, 1.18], "wspace": 0.27},
    )

    ax = axes[0]
    x = np.arange(3)
    width = 0.23
    for index, (label, color) in enumerate(zip(condition_labels, condition_colors)):
        bars = ax.bar(
            x + (index - 1) * width,
            counts[:, index],
            width=width,
            color=color,
            label=label.replace("\n", " "),
            zorder=3,
        )
        for bar, value in zip(bars, counts[:, index]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 2.0,
                f"{value}/100",
                ha="center",
                va="bottom",
                fontsize=6.1,
                rotation=90,
            )
    ax.set_xticks(x, ["Checkpoint 0", "Checkpoint 1", "Checkpoint 2"])
    ax.set_ylim(0, 108)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylabel("Closed-loop successes")
    ax.set_title("(a) Checkpoint success")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6, zorder=0)
    ax.legend(frameon=False, fontsize=6.4, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))

    ax = axes[1]
    tasks = np.arange(10)
    ax.axhline(0, color="#66727A", linewidth=0.7, zorder=1)
    ax.plot(
        tasks,
        gaps["copresent_distractor_prompt"],
        color=COLORS["copresent"],
        marker="o",
        markersize=4.0,
        linewidth=1.45,
        label="Correct minus co-present control",
        zorder=3,
    )
    ax.plot(
        tasks,
        gaps["empty_instruction"],
        color=COLORS["empty_gap"],
        marker="s",
        markersize=3.7,
        linewidth=1.35,
        label="Correct minus empty",
        zorder=3,
    )
    ax.set_xlim(-0.35, 9.35)
    ax.set_ylim(0, 90)
    ax.set_xticks(tasks)
    ax.set_yticks(np.arange(0, 91, 20))
    ax.set_xlabel("LIBERO-Object task index")
    ax.set_ylabel("Paired success gap (points)")
    ax.set_title("(b) Per-task paired gaps")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6, zorder=0)
    ax.legend(frameon=False, fontsize=6.4, loc="upper center", bbox_to_anchor=(0.5, -0.18))

    fig.subplots_adjust(left=0.075, right=0.995, top=0.90, bottom=0.29)
    for suffix, options in (
        (".pdf", {}),
        (".png", {"dpi": 240}),
    ):
        fig.savefig(OUTPUT.with_suffix(suffix), bbox_inches="tight", facecolor="white", **options)
    plt.close(fig)


if __name__ == "__main__":
    main()
