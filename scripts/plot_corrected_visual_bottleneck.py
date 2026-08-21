"""Plot the corrected blind visual-bottleneck and action-specificity program.

The input is the hardened twelve-shard summary. This script intentionally plots
every frozen comparison whether the corresponding gate passes or fails.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "athena" / "results" / "visual_taskmap_blind_rank96_summary.json"
OUTPUT = ROOT / "paper_figures" / "main_corrected_visual_bottleneck"
EXPECTED_SCHEMA = "xvla-corrected-blind-subspace-summary-v2"

COLORS = {
    "blue": "#326B8C",
    "green": "#2E7D6B",
    "orange": "#C97835",
    "red": "#B54B4B",
    "ink": "#18252D",
    "gray": "#66747C",
    "light": "#DCE4E8",
    "pale": "#B8C2C8",
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


def load_summary() -> dict:
    summary = json.loads(SUMMARY.read_text())
    if summary.get("schema") != EXPECTED_SCHEMA:
        raise RuntimeError("Refusing a stale or uncorrected visual-subspace summary")
    scope = summary.get("scope", {})
    if scope.get("seeds") != [0, 1, 2] or scope.get("tasks") != [8, 9]:
        raise RuntimeError("Summary does not cover the frozen checkpoint-task matrix")
    if scope.get("rank") != 96 or len(summary.get("input_results", {})) != 12:
        raise RuntimeError("Summary does not cover the frozen rank or twelve inputs")
    required_gates = {
        "blind_selectivity_pass",
        "action_specificity_pass",
        "jacobian_beats_every_random_control",
        "activation_mse_no_greater_than_jacobian",
    }
    if not required_gates.issubset(summary.get("gates", {})):
        raise RuntimeError("Summary is missing frozen gate outcomes")
    return summary


def percentage(value: float) -> str:
    return f"{100 * value:.1f}"


def main() -> None:
    summary = load_summary()
    seeds = np.arange(3)
    seed_labels = ["seed 0", "seed 1", "seed 2"]

    blind = summary["blind_by_checkpoint"]
    jacobian_blind = np.array(
        [blind[str(seed)]["jacobian_retention"] for seed in seeds]
    )
    random_controls = np.array(
        [
            [
                summary["random_controls_by_checkpoint"][str(seed)][str(control)][
                    "random_retained"
                ]
                / summary["random_controls_by_checkpoint"][str(seed)][str(control)][
                    "full_successes"
                ]
                for control in range(3)
            ]
            for seed in seeds
        ],
        dtype=float,
    )
    specificity = summary["by_checkpoint"]
    jacobian_specificity = np.array(
        [specificity[str(seed)]["jacobian_retention"] for seed in seeds]
    )
    activation_specificity = np.array(
        [specificity[str(seed)]["activation_retention"] for seed in seeds]
    )

    activation_mse = summary["mean_offline_activation_reconstruction_mse"]
    action_mse = summary["mean_offline_action_mse"]
    ratio_labels = ["visual\ntokens", "translation", "rotation", "gripper"]
    ratios = np.array(
        [
            activation_mse["activation_energy_topk"]
            / activation_mse["causal_topk"],
            *[
                action_mse["activation_energy_topk"][group]
                / action_mse["causal_topk"][group]
                for group in ("translation", "rotation", "gripper")
            ],
        ]
    )
    if not np.isfinite(ratios).all() or np.any(ratios <= 0):
        raise RuntimeError("Offline MSE ratios must be finite and positive")

    fig, axes = plt.subplots(1, 3, figsize=(10.7, 3.15))

    ax = axes[0]
    ax.plot(
        seeds,
        100 * jacobian_blind,
        marker="o",
        lw=1.8,
        color=COLORS["green"],
        label="action-Jacobian",
        zorder=4,
    )
    offsets = (-0.08, 0.0, 0.08)
    for control, offset in enumerate(offsets):
        ax.scatter(
            seeds + offset,
            100 * random_controls[:, control],
            s=30,
            color=COLORS["pale"],
            edgecolor=COLORS["gray"],
            linewidth=0.5,
            label="fixed random controls" if control == 0 else None,
            zorder=3,
        )
    ax.axhline(85, color=COLORS["red"], lw=1.1, ls="--", label="retention gate 85%")
    ax.set_xticks(seeds, seed_labels)
    ax.set_ylim(0, 104)
    ax.set_ylabel("baseline successes retained (%)")
    ax.set_title("a  Rank-selection-blind control")
    ax.grid(axis="y", color=COLORS["light"], lw=0.7)
    ax.legend(frameon=False, fontsize=6.8, loc="lower left")
    for x, value in zip(seeds, jacobian_blind):
        ax.text(x, 100 * value + 2.2, percentage(value), ha="center", fontsize=7.0)

    ax = axes[1]
    width = 0.34
    ax.bar(
        seeds - width / 2,
        100 * jacobian_specificity,
        width,
        color=COLORS["green"],
        label="action-Jacobian",
        zorder=3,
    )
    ax.bar(
        seeds + width / 2,
        100 * activation_specificity,
        width,
        color=COLORS["blue"],
        label="activation energy",
        zorder=3,
    )
    ax.axhline(90, color=COLORS["red"], lw=1.1, ls="--", label="Jacobian gate 90%")
    ax.set_xticks(seeds, seed_labels)
    ax.set_ylim(0, 104)
    ax.set_ylabel("baseline successes retained (%)")
    ax.set_title("b  Action-specificity control")
    ax.grid(axis="y", color=COLORS["light"], lw=0.7, zorder=0)
    ax.legend(frameon=False, fontsize=6.8, loc="lower left")

    ax = axes[2]
    x = np.arange(len(ratios))
    bar_colors = [COLORS["blue"]] + [COLORS["orange"]] * 3
    ax.bar(x, ratios, color=bar_colors, width=0.65, zorder=3)
    ax.axhline(1.0, color=COLORS["gray"], lw=1.0, ls="--")
    ax.set_yscale("log")
    lower = min(0.5, float(ratios.min()) / 1.6)
    upper = max(2.0, float(ratios.max()) * 1.6)
    ax.set_ylim(lower, upper)
    ax.set_xticks(x, ratio_labels)
    ax.set_ylabel("activation-energy / Jacobian MSE")
    ax.set_title("c  Generic reconstruction control")
    ax.grid(axis="y", color=COLORS["light"], lw=0.7, which="both", zorder=0)
    for xpos, value in zip(x, ratios):
        ax.annotate(
            f"{value:.2g}$\\times$",
            (xpos, value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.0,
        )

    status = (
        f"Blind gate: {'PASS' if summary['gates']['blind_selectivity_pass'] else 'FAIL'}   "
        f"Action-specificity gate: {'PASS' if summary['gates']['action_specificity_pass'] else 'FAIL'}"
    )
    fig.text(0.5, -0.015, status, ha="center", fontsize=7.5, color=COLORS["gray"])
    fig.tight_layout(w_pad=2.2)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=260, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
