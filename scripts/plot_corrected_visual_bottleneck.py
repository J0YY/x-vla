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
EXPECTED_CACHE_SHA256 = "053cf7e392054c4bc1ac0ea280828c3baf7f02a43e2feee22f27734956575662"
EXPECTED_CHECKPOINTS = {
    "0": "96f11093701d6b52deefb50b7921b46e2c987e5b9dbce947997882f7c59da6c9",
    "1": "cc0780b989165a80a449c03cbb3574e2f56b2e5747a64cf32d1fde418df8ec3c",
    "2": "413a770071bd8f16b924c5604f7c7a58567d2eb6c40c66aab202bcf046fec910",
}
EXPECTED_MAPPING = {
    "0": 9,
    "1": 4,
    "2": 1,
    "3": 3,
    "4": 0,
    "5": 7,
    "6": 2,
    "7": 6,
    "8": 5,
    "9": 8,
}

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
    if scope.get("conditions_same_allocation") is not True:
        raise RuntimeError("Conditions were not evaluated in matched allocations")
    if scope.get("checkpoint_sha256") != EXPECTED_CHECKPOINTS:
        raise RuntimeError("Checkpoint identities do not match the frozen program")
    if scope.get("cache_sha256") != EXPECTED_CACHE_SHA256:
        raise RuntimeError("Cache identity does not match the frozen program")
    metadata = scope.get("cache_metadata", {})
    if (
        metadata.get("revision")
        != "e1e080d7df1d0a359dff5c86c222e047549f447f"
        or metadata.get("metadata_sha256")
        != "34caee9641ae50bb4e077de306a7d0031753757882da8b1f117e7ea36a486b42"
        or metadata.get("dataset_to_official_task") != EXPECTED_MAPPING
        or metadata.get("ordering_matches_official") is not False
    ):
        raise RuntimeError("Summary does not carry the corrected task mapping")
    if scope.get("evaluation_protocol") != {
        "res": 64,
        "horizon": 8,
        "num_steps_wait": 10,
        "exec_h": 8,
        "eps_per_task": 50,
        "max_steps": 400,
    }:
        raise RuntimeError("Evaluation protocol differs from the frozen program")
    expected_inputs = {
        f"results/visual_taskmap_{family}_s{seed}_r96_{suffix}{task}.json"
        for seed in range(3)
        for task in (8, 9)
        for family, suffix in (("activation", "t"), ("vit", "blind_t"))
    }
    if set(summary["input_results"]) != expected_inputs:
        raise RuntimeError("Summary input paths do not match the exact twelve-file matrix")
    if any(
        len(identity.get("sha256", "")) != 64
        or any(character not in "0123456789abcdef" for character in identity["sha256"])
        for identity in summary["input_results"].values()
    ):
        raise RuntimeError("Summary input identities contain a malformed SHA-256")
    required_gates = {
        "blind_selectivity_pass",
        "action_specificity_pass",
        "jacobian_beats_every_random_control",
        "activation_mse_no_greater_than_jacobian",
    }
    if not required_gates.issubset(summary.get("gates", {})):
        raise RuntimeError("Summary is missing frozen gate outcomes")
    random_pass = all(
        row["jacobian_minus_random"] > 0
        for controls in summary["random_controls_by_checkpoint"].values()
        for row in controls.values()
    )
    blind_pass = (
        summary["blind_pooled"]["jacobian_retention"] >= 0.85 and random_pass
    )
    specificity_pass = (
        summary["pooled"]["jacobian_retention"] >= 0.90
        and summary["pooled"]["jacobian_minus_activation"] >= 0.10
        and all(
            row["jacobian_minus_activation"] > 0
            for row in summary["by_checkpoint"].values()
        )
        and summary["bootstrap"]["jacobian_minus_activation_ci95"][0] > 0
        and summary["mean_offline_activation_reconstruction_mse"][
            "activation_energy_topk"
        ]
        <= summary["mean_offline_activation_reconstruction_mse"]["causal_topk"]
    )
    if summary["gates"]["blind_selectivity_pass"] is not blind_pass:
        raise RuntimeError("Blind gate does not recompute")
    if summary["gates"]["action_specificity_pass"] is not specificity_pass:
        raise RuntimeError("Action-specificity gate does not recompute")
    return summary


def percentage(value: float) -> str:
    return f"{100 * value:.1f}"


def mse_ratio(numerator: float, denominator: float) -> float:
    if denominator < 0 or numerator < 0:
        raise RuntimeError("MSE values must be nonnegative")
    if denominator == 0:
        if numerator == 0:
            return 1.0
        raise RuntimeError("Cannot plot a positive MSE relative to an exact-zero denominator")
    return numerator / denominator


def main() -> None:
    summary = load_summary()
    seeds = np.arange(3)
    categories = np.arange(4)
    category_labels = ["seed 0", "seed 1", "seed 2", "pooled"]

    blind = summary["blind_by_checkpoint"]
    jacobian_blind = np.array(
        [blind[str(seed)]["jacobian_retention"] for seed in seeds]
        + [summary["blind_pooled"]["jacobian_retention"]]
    )
    random_by_seed = np.array(
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
    random_pooled = np.array(
        [
            sum(
                summary["random_controls_by_checkpoint"][str(seed)][str(control)][
                    "random_retained"
                ]
                for seed in seeds
            )
            / sum(
                summary["random_controls_by_checkpoint"][str(seed)][str(control)][
                    "full_successes"
                ]
                for seed in seeds
            )
            for control in range(3)
        ]
    )
    random_controls = np.vstack([random_by_seed, random_pooled])
    specificity = summary["by_checkpoint"]
    jacobian_specificity = np.array(
        [specificity[str(seed)]["jacobian_retention"] for seed in seeds]
        + [summary["pooled"]["jacobian_retention"]]
    )
    activation_specificity = np.array(
        [specificity[str(seed)]["activation_retention"] for seed in seeds]
        + [summary["pooled"]["activation_retention"]]
    )

    activation_mse = summary["mean_offline_activation_reconstruction_mse"]
    action_mse = summary["mean_offline_action_mse"]
    activation_ratio = mse_ratio(
        activation_mse["activation_energy_topk"],
        activation_mse["causal_topk"],
    )
    action_ratios = np.array(
        [
            mse_ratio(
                action_mse["activation_energy_topk"][group],
                action_mse["causal_topk"][group],
            )
            for group in ("translation", "rotation", "gripper")
        ]
    )
    if not np.isfinite([activation_ratio, *action_ratios]).all() or any(
        value < 0 for value in [activation_ratio, *action_ratios]
    ):
        raise RuntimeError("Offline MSE ratios must be finite and nonnegative")

    fig, axes = plt.subplots(2, 2, figsize=(5.5, 5.25))
    axes = axes.ravel()

    ax = axes[0]
    ax.plot(
        categories,
        100 * jacobian_blind,
        marker="o",
        lw=1.8,
        color=COLORS["green"],
        zorder=4,
    )
    offsets = (-0.08, 0.0, 0.08)
    for control, offset in enumerate(offsets):
        ax.scatter(
            categories + offset,
            100 * random_controls[:, control],
            s=30,
            color=COLORS["pale"],
            edgecolor=COLORS["gray"],
            linewidth=0.5,
            zorder=3,
        )
    ax.axhline(85, color=COLORS["red"], lw=1.1, ls="--")
    ax.text(0.02, 86.7, "pooled gate 85%", ha="left", color=COLORS["red"], fontsize=6.8)
    ax.text(3.0, 106.0, "Jacobian", ha="right", color=COLORS["green"], fontsize=6.8)
    ax.text(3.0, 100 * random_controls[-1].max() + 2.0, "3 fixed random", ha="right", color=COLORS["gray"], fontsize=6.8)
    ax.set_xticks(categories, category_labels)
    ax.set_ylim(0, 110)
    ax.set_ylabel("baseline successes retained (%)")
    ax.set_title("a  Blind-task selected vs random")
    ax.grid(axis="y", color=COLORS["light"], lw=0.7)
    ax.text(3.0, 100 * jacobian_blind[-1] + 2.0, percentage(jacobian_blind[-1]), ha="center", fontsize=6.8)

    ax = axes[1]
    width = 0.34
    ax.bar(
        categories - width / 2,
        100 * jacobian_specificity,
        width,
        color=COLORS["green"],
        label="action-Jacobian",
        zorder=3,
    )
    ax.bar(
        categories + width / 2,
        100 * activation_specificity,
        width,
        color=COLORS["blue"],
        label="activation energy",
        zorder=3,
    )
    ax.axhline(90, color=COLORS["red"], lw=1.1, ls="--")
    ax.text(2.98, 91.8, "pooled J gate 90%", ha="right", color=COLORS["red"], fontsize=6.8)
    ax.set_xticks(categories, category_labels)
    ax.set_ylim(0, 112)
    ax.set_ylabel("baseline successes retained (%)")
    ax.set_title("b  Action-specificity control")
    ax.grid(axis="y", color=COLORS["light"], lw=0.7, zorder=0)
    ax.legend(frameon=False, fontsize=6.6, loc="upper left", ncol=2)
    margin = 100 * summary["pooled"]["jacobian_minus_activation"]
    ci_low, ci_high = summary["bootstrap"]["jacobian_minus_activation_ci95"]
    ax.text(
        0.98,
        0.05,
        f"pooled margin {margin:.1f} pp\n95% interval [{100*ci_low:.1f}, {100*ci_high:.1f}] pp",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.8,
        color=COLORS["gray"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.2},
    )

    ax = axes[2]
    ax.bar([0], [activation_ratio], color=COLORS["blue"], width=0.58, zorder=3)
    ax.axhline(1.0, color=COLORS["gray"], lw=1.0, ls="--")
    ax.set_ylim(0, max(1.3, activation_ratio * 1.25))
    ax.set_xticks([0], ["activation-energy basis"])
    ax.set_ylabel("activation-energy / Jacobian MSE")
    ax.set_title("c  Raw-activation control, gated")
    ax.grid(axis="y", color=COLORS["light"], lw=0.7, zorder=0)
    ax.text(0, activation_ratio + 0.035 * ax.get_ylim()[1], f"{activation_ratio:.2g}$\\times$", ha="center", fontsize=7.0)
    ax.text(0.97, 1.02, "gate $\\leq1$", transform=ax.get_yaxis_transform(), ha="right", fontsize=6.8, color=COLORS["gray"])

    ax = axes[3]
    x = np.arange(3)
    ax.bar(x, action_ratios, color=COLORS["orange"], width=0.62, zorder=3)
    ax.axhline(1.0, color=COLORS["gray"], lw=1.0, ls="--")
    ax.set_ylim(0, max(2.0, float(action_ratios.max()) * 1.22))
    ax.set_xticks(x, ["translation", "rotation", "gripper"])
    ax.set_ylabel("activation-energy / Jacobian MSE")
    ax.set_title("d  Action distortion, descriptive")
    ax.grid(axis="y", color=COLORS["light"], lw=0.7, zorder=0)
    for xpos, value in zip(x, action_ratios):
        ax.text(xpos, value + 0.035 * ax.get_ylim()[1], f"{value:.2g}$\\times$", ha="center", fontsize=7.0)

    status = (
        f"Blind gate: {'PASS' if summary['gates']['blind_selectivity_pass'] else 'FAIL'}   "
        f"Action-specificity gate: {'PASS' if summary['gates']['action_specificity_pass'] else 'FAIL'}"
    )
    fig.text(0.5, 0.006, status, ha="center", fontsize=7.2, color=COLORS["gray"])
    fig.tight_layout(rect=(0, 0.025, 1, 1), h_pad=2.0, w_pad=1.7)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=260, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
