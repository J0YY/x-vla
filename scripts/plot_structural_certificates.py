"""Plot the frozen joint-block, modality, and rational-normalizer certificates."""

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


def main() -> None:
    attention = load("conv_joint_attention_certificate_v1_summary.json")
    ffn = load("conv_joint_ffn_certificate_v1_summary.json")
    block = load("conv_joint_block_certificate_v1_summary.json")
    modality = load("vit_modality_contribution_v1_summary.json")
    full_forward_v2 = RESULTS / "vit_full_learned_forward_v2_expanded_summary.json"
    if not full_forward_v2.is_file():
        raise FileNotFoundError(
            "The figure requires the expanded v2 learned-forward summary and must not "
            "silently fall back to the smaller v1 study."
        )
    full_forward = load(full_forward_v2.name)
    normalizer = load("rational_norm_safety_v1_summary.json")
    seeds = np.arange(3)

    certificate_summaries = (attention, ffn, block, modality)
    certificate_labels = (
        "Conv attn.",
        "Conv FFN",
        "Conv block",
        "ViT\nsources",
        "ViT full\nforward",
        "RNorm\naction",
    )
    reconstruction_ratios = np.array(
        [
            [
                row["max_relative_l2_error"]
                / summary["aggregate"]["relative_l2_gate"]
                for row in summary["checkpoints"]
            ]
            for summary in certificate_summaries
        ]
    )
    rational_action_ratios = np.array(
        [row["action_nrmse"] / 1e-3 for row in normalizer["checkpoints"]]
    )
    full_forward_rows = full_forward.get("results", full_forward.get("checkpoints"))
    full_forward_gate = full_forward["aggregate"].get(
        "gate",
        full_forward["aggregate"].get("end_to_end_action_relative_l2_gate"),
    )
    full_forward_ratios = np.array(
        [
            row["max_action_relative_l2_error"] / full_forward_gate
            for row in full_forward_rows
        ]
    )
    gate_normalized_errors = np.vstack(
        [reconstruction_ratios, full_forward_ratios, rational_action_ratios]
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.15))

    ax = axes[0]
    offsets = (-0.16, 0.0, 0.16)
    categories = np.arange(6)
    for seed, offset in zip(seeds, offsets):
        ax.scatter(
            categories + offset,
            gate_normalized_errors[:, seed],
            s=40,
            label=f"seed {seed}",
            zorder=3,
        )
    ax.axhline(1.0, color=COLORS["red"], lw=1.2, ls="--", label="frozen gate")
    ax.set_yscale("log")
    ax.set_ylim(1e-10, 4)
    ax.set_xticks(categories, certificate_labels)
    ax.tick_params(axis="x", labelsize=7.7)
    ax.set_ylabel("certificate metric / frozen gate")
    ax.set_title("a  Independent reconstruction certificates")
    ax.grid(axis="y", color=COLORS["light"], lw=0.7)
    ax.legend(frameon=False, loc="lower right", fontsize=6.9, ncol=2)

    ax = axes[1]
    sources = ("vision", "robot_state", "action_query", "instruction")
    source_labels = ("Vision", "Robot state", "Action query", "Instruction")
    source_colors = (COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["gray"])
    blocks = np.arange(8)
    bottom = np.zeros(8)
    for source, label, color in zip(sources, source_labels, source_colors):
        values = 100 * np.array(
            [
                modality["module_contribution_distributions"]["by_layer"][
                    f"block_{block_index}"
                ][source]["coherent_energy_fraction"]["mean"]
                for block_index in blocks
            ]
        )
        ax.bar(blocks, values, bottom=bottom, color=color, label=label, zorder=3)
        bottom += values
    ax.set_xticks(blocks)
    ax.set_ylim(0, 100)
    ax.set_xlabel("joint block")
    ax.set_ylabel("mean coherent-energy share (%)")
    ax.set_title("b  ViT attention source partition by depth", pad=31)
    ax.grid(axis="y", color=COLORS["light"], lw=0.7, zorder=0)
    ax.legend(
        frameon=True,
        facecolor="white",
        framealpha=0.88,
        edgecolor="none",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        fontsize=6.8,
        ncol=4,
    )

    fig.tight_layout(w_pad=2.0)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=260, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
