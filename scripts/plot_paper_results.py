"""Generate publication-oriented χ-VLA result figures.

The constants below are copied from verified result JSONs on the `xvla-data`
Modal volume and from the matched-protocol table in DEVLOG.md:

  odt_libero_action_vit_rational.json
  libero_gram_ablation_closedloop_vit_rational.json
  odt_attention_real_weights_ckpt_linear_rat_vit_s0_v2.json
  decomposability_audit_vit_rational.json
  DEVLOG.md entries 67 and 69

The script writes both PNG previews and vector PDF files.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_figures"
OUT.mkdir(exist_ok=True)

COLORS = {
    "ink": "#18252E",
    "muted": "#60717C",
    "grid": "#DCE3E7",
    "gray": "#9DA8AE",
    "blue": "#356FA8",
    "cyan": "#66AFC4",
    "green": "#2D8C78",
    "gold": "#D89B2B",
    "red": "#B94747",
    "purple": "#7657A6",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
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


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )


def save_both(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return center - half, center + half


def main_capability() -> None:
    raise RuntimeError(
        "The legacy capability plot is retired. Run scripts/plot_verified_vit_capability.py "
        "after artifacts/vit_capability/verify.py instead."
    )


def appendix_matched_architecture_cost() -> None:
    labels = ["Conventional twin", "Rational χ-VLA"]
    hardware_labels = ["A30", "A6000"]
    successes = {
        "Conventional twin": np.array([448, 451]),
        "Rational χ-VLA": np.array([426, 422]),
    }
    trials = 500
    eager_ratio = np.array([5.3278, 3.4892])
    compiled_ratio = np.array([1.7381, 1.7305])
    colors = [COLORS["gray"], COLORS["blue"]]

    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.15))
    ax = axes[0]
    positions = np.arange(2)
    width = 0.34
    for model_index, (label, color) in enumerate(zip(labels, colors)):
        values = 100 * successes[label] / trials
        intervals = np.array(
            [wilson_interval(int(value), trials) for value in successes[label]]
        )
        error = np.vstack(
            [values - 100 * intervals[:, 0], 100 * intervals[:, 1] - values]
        )
        x_values = positions + (model_index - 0.5) * width
        ax.bar(x_values, values, width, color=color, label=label)
        ax.errorbar(
            x_values,
            values,
            yerr=error,
            fmt="none",
            ecolor=COLORS["ink"],
            elinewidth=1.1,
            capsize=3,
        )
        for position, value in zip(x_values, values):
            ax.text(position, value + 1.0, f"{value:.1f}%", ha="center", fontweight="bold")
    ax.set_xticks(positions, hardware_labels)
    ax.set_ylim(78, 95)
    ax.set_ylabel("Closed-loop success (%)")
    ax.set_title("Complete same-hardware controls")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.legend(
        frameon=True,
        facecolor="white",
        framealpha=0.92,
        edgecolor="none",
        fontsize=7.2,
        loc="lower right",
    )
    panel_label(ax, "a")

    ax = axes[1]
    width = 0.34
    ax.bar(
        positions - width / 2,
        eager_ratio,
        width,
        color=COLORS["gold"],
        label="Eager",
    )
    ax.bar(
        positions + width / 2,
        compiled_ratio,
        width,
        color=COLORS["green"],
        label="Compiled",
    )
    for position, value in zip(positions - width / 2, eager_ratio):
        ax.text(position, value + 0.15, f"{value:.2f}×", ha="center", fontweight="bold")
    for position, value in zip(positions + width / 2, compiled_ratio):
        ax.text(position, value + 0.15, f"{value:.2f}×", ha="center", fontweight="bold")
    ax.set_xticks(positions, hardware_labels)
    ax.set_ylim(0, 6.3)
    ax.set_ylabel("χ / conventional latency ratio")
    ax.set_title("Compiler narrows the runtime gap")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.legend(frameon=False, loc="upper right")
    panel_label(ax, "b")

    fig.suptitle("Conversion cost is nonzero, while compilation narrows the systems gap", y=1.03)
    fig.text(
        0.5,
        -0.02,
        "Each capability bar uses the same 500 canonical trials on one GPU class. The χ deficit "
        "is 4.4 points on A30 and 5.8 on A6000. Compilation narrows the latency ratio to 1.73–1.74× on both GPUs.",
        ha="center",
        fontsize=7.7,
        color=COLORS["muted"],
    )
    fig.tight_layout()
    save_both(fig, "appendix_matched_architecture_cost")


def appendix_counterfactual_grounding() -> None:
    records = []
    labels = []
    for seed in range(3):
        path = ROOT / "athena" / "results" / f"causal_s{seed}.json"
        with path.open() as handle:
            records.append(json.load(handle)["causal"])
        labels.append(f"χ-{seed}")
    conventional_path = ROOT / "athena" / "results" / "causal_conventional_s0.json"
    with conventional_path.open() as handle:
        records.append(json.load(handle)["causal"])
    labels.append("Conventional")

    shifts = np.array([record["mean_paired_preference_shift_m"] for record in records])
    intervals = np.array(
        [record["mean_paired_preference_shift_95pct_bootstrap_ci_m"] for record in records]
    )
    toward = np.array(
        [record["fraction_shift_toward_counterfactual_named_object"] for record in records]
    ) * 100
    ended = np.array(
        [record["fraction_counterfactual_ended_closer_to_named_object"] for record in records]
    ) * 100
    positions = np.arange(len(records))

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.25))
    ax = axes[0]
    error = np.vstack([shifts - intervals[:, 0], intervals[:, 1] - shifts])
    for index, position in enumerate(positions):
        color = COLORS["blue"] if index < 3 else COLORS["gray"]
        ax.errorbar(
            position,
            shifts[index],
            yerr=error[:, index : index + 1],
            fmt="o",
            color=color,
            ecolor=color,
            markersize=7,
            elinewidth=1.6,
            capsize=4,
        )
    ax.axhline(0, color=COLORS["gray"], linestyle="--", linewidth=1)
    ax.axvline(2.5, color=COLORS["grid"], linewidth=1)
    for position, value in zip(positions, shifts):
        ax.text(position, value + 0.011, f"{value:.3f} m", ha="center", fontweight="bold")
    ax.set_xticks(positions, labels)
    ax.set_ylim(-0.01, max(intervals[:, 1]) + 0.055)
    ax.set_ylabel("Paired preference shift (m)")
    ax.set_title("Trajectory response to renamed object")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    panel_label(ax, "a")

    ax = axes[1]
    width = 0.34
    ax.bar(
        positions - width / 2,
        toward,
        width,
        color=COLORS["green"],
        label="Shift toward renamed object",
    )
    ax.bar(
        positions + width / 2,
        ended,
        width,
        color=COLORS["gold"],
        label="End closer to renamed object",
    )
    for x, value in zip(positions - width / 2, toward):
        ax.text(x, value + 2, f"{value:.0f}%", ha="center", fontsize=8, fontweight="bold")
    for x, value in zip(positions + width / 2, ended):
        ax.text(x, value + 2, f"{value:.0f}%", ha="center", fontsize=8, fontweight="bold")
    ax.axvline(2.5, color=COLORS["grid"], linewidth=1)
    ax.set_xticks(positions, labels)
    ax.set_ylim(0, 120)
    ax.set_ylabel("Paired rollouts (%)")
    ax.set_title("Direction changes more often than outcome")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.legend(frameon=False, fontsize=7.4, loc="upper center")
    panel_label(ax, "b")

    fig.suptitle("Language reliably changes motion, but often does not overcome the scene prior", y=1.03)
    fig.text(
        0.5,
        -0.025,
        "Each model uses 100 paired canonical states and 80-step rollouts. Error bars are "
        "trial-bootstrap 95% intervals. Proximity is a behavioral grounding outcome, not "
        "counterfactual task success.",
        ha="center",
        fontsize=7.5,
        color=COLORS["muted"],
    )
    fig.tight_layout()
    save_both(fig, "appendix_counterfactual_grounding")


def main_causal_subspace() -> None:
    raise RuntimeError(
        "Retired: this figure used the pre-correction cache task map. "
        "Use scripts/plot_corrected_visual_bottleneck.py after the hardened summary passes."
    )
    ks = np.array([4, 8, 16, 32, 64])
    global_mse = {
        "Translation": np.array([1.62466, 1.72915, 0.96152, 0.36517, 0.05780]),
        "Rotation": np.array([1.65144, 1.73598, 1.33522, 0.84604, 0.12104]),
        "Gripper": np.array([1.04071, 0.93834, 0.71632, 0.07590, 0.00670]),
    }
    random_mse = {
        "Translation": np.array([1.03016, 1.49419, 0.92472, 0.70817, 0.27626]),
        "Rotation": np.array([0.79328, 0.93897, 0.91378, 0.71070, 0.56058]),
        "Gripper": np.array([0.80667, 0.71972, 0.88879, 0.56080, 0.28005]),
    }
    line_colors = {
        "Translation": COLORS["blue"],
        "Rotation": COLORS["purple"],
        "Gripper": COLORS["green"],
    }

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.55), gridspec_kw={"width_ratios": [1.35, 0.9]})

    ax = axes[0]
    for group in global_mse:
        ratio = random_mse[group] / global_mse[group]
        ax.plot(ks, ratio, marker="o", linewidth=2, color=line_colors[group], label=group)
        ax.text(66, ratio[-1], f"{ratio[-1]:.1f}×", color=line_colors[group], va="center", fontsize=8)
    ax.axhline(1, color=COLORS["gray"], linestyle="--", linewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_ylim(0.4, 65)
    ax.set_xlim(3.5, 82)
    ax.set_xlabel("Retained visual-bond dimensions")
    ax.set_ylabel("Random MSE / causal-subspace MSE")
    ax.grid(which="major", color=COLORS["grid"], linewidth=0.8)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Offline action faithfulness")
    ax.text(
        0.02,
        0.03,
        "Above 1 means the causal subspace is better",
        transform=ax.transAxes,
        fontsize=7.5,
        color=COLORS["muted"],
    )
    panel_label(ax, "a")

    ax = axes[1]
    conditions = ["Full\n384 dims", "Causal top 64\nof 384", "Random 64\nof 384"]
    successes = np.array([19, 18, 0])
    total = 20
    rates = successes / total * 100
    intervals = np.array([wilson_interval(int(s), total) for s in successes]) * 100
    yerr = np.vstack([rates - intervals[:, 0], intervals[:, 1] - rates])
    bars = ax.bar(
        np.arange(3),
        rates,
        color=[COLORS["ink"], COLORS["green"], COLORS["red"]],
        width=0.65,
        zorder=3,
    )
    ax.errorbar(
        np.arange(3),
        rates,
        yerr=yerr,
        fmt="none",
        ecolor=COLORS["ink"],
        elinewidth=1,
        capsize=3,
        zorder=4,
    )
    for bar, s in zip(bars, successes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(bar.get_height() + 3, 2),
            f"{s}/{total}",
            ha="center",
            fontweight="bold",
        )
    ax.set_ylim(0, 108)
    ax.set_xticks(np.arange(3), conditions)
    ax.set_ylabel("Closed-loop success (%)")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8, zorder=0)
    ax.set_title("Behavioral intervention")
    ax.text(
        0.5,
        0.96,
        "4 tasks × 5 trials, matched seeds",
        transform=ax.transAxes,
        ha="center",
        fontsize=7.5,
        color=COLORS["muted"],
    )
    panel_label(ax, "b")

    fig.suptitle("Causally selected visual directions preserve policy behavior", fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save_both(fig, "main_causal_subspace")


def main_causal_subspace_replication() -> None:
    raise RuntimeError(
        "Retired: this figure used the pre-correction cache task map. "
        "Use scripts/plot_corrected_visual_bottleneck.py after the hardened summary passes."
    )
    checkpoint_labels = ["Checkpoint 0", "Checkpoint 1", "Checkpoint 2"]
    conditions = ["Full 384", "Selected 96", "Median random 96"]
    successes = np.array(
        [
            [17, 17, 2],
            [15, 14, 0],
            [20, 19, 0],
        ]
    )
    total_per_checkpoint = 20
    colors = [COLORS["ink"], COLORS["green"], COLORS["red"]]

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.55), gridspec_kw={"width_ratios": [1.45, 0.9]})

    ax = axes[0]
    positions = np.arange(len(checkpoint_labels))
    width = 0.24
    for condition_index, (condition, color) in enumerate(zip(conditions, colors)):
        rates = 100 * successes[:, condition_index] / total_per_checkpoint
        x_values = positions + (condition_index - 1) * width
        bars = ax.bar(x_values, rates, width, color=color, label=condition, zorder=3)
        for bar, count in zip(bars, successes[:, condition_index]):
            inside = bar.get_height() > 15
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() - 3.0 if inside else max(bar.get_height() + 2.0, 1.5),
                f"{count}/20",
                ha="center",
                va="top" if inside else "bottom",
                fontsize=7.3,
                fontweight="bold",
                color="white" if inside else COLORS["ink"],
            )
    ax.set_xticks(positions, checkpoint_labels)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Closed-loop success (%)")
    ax.set_title("Exploratory screen across checkpoints")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=7.4, loc="upper center", ncol=3)
    panel_label(ax, "a")

    ax = axes[1]
    pooled = successes.sum(axis=0)
    pooled_total = 3 * total_per_checkpoint
    rates = 100 * pooled / pooled_total
    intervals = np.array(
        [wilson_interval(int(count), pooled_total) for count in pooled]
    ) * 100
    yerr = np.vstack([rates - intervals[:, 0], intervals[:, 1] - rates])
    bars = ax.bar(np.arange(3), rates, color=colors, width=0.65, zorder=3)
    ax.errorbar(
        np.arange(3),
        rates,
        yerr=yerr,
        fmt="none",
        ecolor=COLORS["ink"],
        elinewidth=1,
        capsize=3,
        zorder=4,
    )
    for bar, count in zip(bars, pooled):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(bar.get_height() + 3, 2),
            f"{count}/{pooled_total}",
            ha="center",
            fontweight="bold",
        )
    ax.set_xticks(
        np.arange(3),
        ["Full\n384 dims", "Selected\n96 dims", "Median random\n96 dims"],
    )
    ax.set_ylim(0, 108)
    ax.set_ylabel("Closed-loop success (%)")
    ax.set_title("Pooled descriptive result")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8, zorder=0)
    panel_label(ax, "b")

    fig.suptitle(
        "One quarter of the visual bond preserves policy behavior",
        fontsize=12,
        fontweight="bold",
        y=1.03,
    )
    fig.text(
        0.5,
        -0.025,
        "Exploratory tasks 4 to 7, five canonical trials per task and checkpoint. Rank 96 retains "
        "49 of 52 full-policy successes. Three random controls total 1/60, 3/60, and 3/60. "
        "Pooled intervals are descriptive.",
        ha="center",
        fontsize=7.5,
        color=COLORS["muted"],
    )
    fig.tight_layout()
    save_both(fig, "main_causal_subspace_replication")


def appendix_causal_rank_curve() -> None:
    raise RuntimeError(
        "Retired: this figure used the pre-correction cache task map. "
        "Use scripts/plot_corrected_visual_bottleneck.py after the hardened summary passes."
    )
    ranks = np.array([64, 96, 128, 192])
    selected_success = np.array([30, 50, 50, 51])
    selected_trials = 60
    random_success = np.array([0, 7, 50, 99])
    random_trials = 180
    retained_success = np.array([28, 49, 48, 50])
    full_success = np.array([51, 52, 50, 52])

    selected_rate = 100 * selected_success / selected_trials
    random_rate = 100 * random_success / random_trials
    conditional_retention = 100 * retained_success / full_success

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.45))

    ax = axes[0]
    ax.plot(
        ranks,
        selected_rate,
        marker="o",
        linewidth=2.2,
        color=COLORS["green"],
        label="Selected subspace",
    )
    ax.plot(
        ranks,
        random_rate,
        marker="o",
        linewidth=2.2,
        color=COLORS["red"],
        label="Three random controls",
    )
    ax.axvline(96, color=COLORS["gold"], linewidth=1.4, linestyle="--")
    ax.set_xticks(ranks)
    ax.set_ylim(-3, 103)
    ax.set_xlabel("Retained visual dimensions")
    ax.set_ylabel("Closed-loop success (%)")
    ax.set_title("Selectivity across ranks")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    panel_label(ax, "a")

    ax = axes[1]
    bars = ax.bar(
        ranks,
        conditional_retention,
        width=20,
        color=[COLORS["muted"], COLORS["gold"], COLORS["muted"], COLORS["muted"]],
        zorder=3,
    )
    ax.axhline(90, color=COLORS["ink"], linewidth=1, linestyle=":")
    for bar, retained, full in zip(bars, retained_success, full_success):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() - 4,
            f"{retained}/{full}",
            ha="center",
            va="top",
            fontsize=8,
            fontweight="bold",
            color="white",
        )
    ax.set_xticks(ranks)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Retained visual dimensions")
    ax.set_ylabel("Full-success retention (%)")
    ax.set_title("Rank 96 is the smallest near-lossless width")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8, zorder=0)
    panel_label(ax, "b")

    fig.suptitle(
        "Rank 96 is the empirical selectivity knee",
        fontsize=12,
        fontweight="bold",
        y=1.03,
    )
    fig.text(
        0.5,
        -0.025,
        "Three checkpoints, tasks 4 to 7. Selected conditions use 60 trials per rank. "
        "Random conditions pool 180 equal-rank trials. Retention is conditional on full-policy success.",
        ha="center",
        fontsize=7.5,
        color=COLORS["muted"],
    )
    fig.tight_layout()
    save_both(fig, "appendix_causal_rank_curve")


BLOCK_IDS = [0, 6, 7]
RATIOS_BY_RANK = {
    8: [
        [1.029899, 1.034935, 1.000332, 0.967630, 1.025700, 1.044299, 1.017759, 1.006163, 1.016473, 1.029717, 0.987108, 1.006097],
        [1.235524, 1.318516, 0.894439, 1.369967, 1.204505, 0.821915, 1.105911, 2.842439, 1.104515, 1.005638, 0.917139, 1.771876],
        [2.068100, 1.016185, 1.162671, 1.020562, 0.976551, 1.003895, 1.033514, 1.077219, 1.041281, 0.946057, 1.011031, 1.226104],
    ],
    32: [
        [1.178920, 1.036237, 1.198454, 1.074969, 1.206168, 1.163979, 1.054716, 1.129669, 1.049042, 1.001247, 1.030287, 1.014755],
        [1.136727, 3.288572, 1.208132, 2.298111, 1.220563, 0.809650, 1.442885, 3.735143, 1.830604, 1.263818, 1.123355, 1.890609],
        [1.930893, 1.144411, 1.063623, 1.220404, 0.960104, 1.081003, 1.001907, 2.080034, 1.203653, 1.179349, 0.952055, 1.224754],
    ],
    128: [
        [4.193511, 1.066657, 10.135155, 2.026207, 4.136808, 2.756692, 2.902461, 2.127159, 1.761359, 1.492354, 1.589449, 2.775945],
        [1.798980, 5.809215, 2.288232, 3.798864, 3.265137, 1.487944, 3.697289, 7.458778, 3.336510, 3.124641, 2.087070, 3.303461],
        [5.456413, 2.518422, 2.364107, 1.928278, 1.699979, 1.862740, 2.240150, 3.715455, 2.043569, 1.890129, 1.986659, 3.054431],
    ],
}


def main_exact_attention_odt() -> None:
    ks = np.array([8, 32, 128])
    flattened = {k: np.concatenate(RATIOS_BY_RANK[k]) for k in ks}
    medians = np.array([np.median(flattened[k]) for k in ks])
    q25 = np.array([np.quantile(flattened[k], 0.25) for k in ks])
    q75 = np.array([np.quantile(flattened[k], 0.75) for k in ks])
    favor = np.array([np.mean(flattened[k] > 1) * 100 for k in ks])

    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.55), gridspec_kw={"width_ratios": [0.85, 1.45]})

    ax = axes[0]
    ax.fill_between(ks, q25, q75, color=COLORS["cyan"], alpha=0.25, label="Interquartile range")
    ax.plot(ks, medians, marker="o", color=COLORS["blue"], linewidth=2, label="Median over 36 block-head pairs")
    ax.axhline(1, color=COLORS["gray"], linestyle="--", linewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_ylim(0.75, 3.15)
    ax.set_xlabel("Retained dimensions")
    ax.set_ylabel("Top-subspace / random faithfulness ratio")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.set_title("Rank dependence")
    for x, y, pct in zip(ks, medians, favor):
        ax.text(x, y + 0.15, f"{pct:.0f}% > 1", ha="center", fontsize=7.5, color=COLORS["muted"])
    ax.legend(frameon=False, loc="upper left", fontsize=7.5)
    panel_label(ax, "a")

    ax = axes[1]
    rng = np.random.default_rng(7)
    k128_by_block = RATIOS_BY_RANK[128]
    positions = np.arange(len(BLOCK_IDS))
    for position, values in enumerate(k128_by_block):
        jitter = rng.uniform(-0.16, 0.16, len(values))
        ax.scatter(
            np.full(len(values), position) + jitter,
            values,
            s=18,
            color=COLORS["blue"],
            alpha=0.45,
            edgecolor="none",
            zorder=2,
        )
    means = np.array([np.mean(v) for v in k128_by_block])
    med_block = np.array([np.median(v) for v in k128_by_block])
    ax.plot(positions, med_block, color=COLORS["ink"], marker="o", linewidth=1.8, label="Block median", zorder=4)
    ax.plot(positions, means, color=COLORS["gold"], marker="D", linewidth=1.2, label="Block mean", zorder=3)
    ax.axhline(1, color=COLORS["gray"], linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_ylim(0.7, 13)
    ax.set_xticks(positions, [str(block) for block in BLOCK_IDS])
    ax.set_xlabel("Transformer block")
    ax.set_ylabel("Faithfulness ratio at rank 128")
    ax.grid(axis="y", which="major", color=COLORS["grid"], linewidth=0.8)
    ax.set_title("Polynomial-core structure by depth")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")
    ax.annotate(
        f"Block 6 mean {means[1]:.2f}×",
        xy=(1, means[1]),
        xytext=(0.95, 7.2),
        arrowprops={"arrowstyle": "->", "color": COLORS["gold"], "lw": 1},
        color="#7A5612",
        fontsize=8,
    )
    ax.annotate(
        f"Block 7 median {med_block[2]:.2f}×",
        xy=(2, med_block[2]),
        xytext=(1.45, 1.02),
        arrowprops={"arrowstyle": "->", "color": COLORS["green"], "lw": 1},
        color=COLORS["green"],
        fontsize=8,
    )
    panel_label(ax, "b")

    fig.suptitle("Weight-derived polynomial cores expose learned attention structure", fontsize=12, fontweight="bold", y=1.03)
    fig.text(
        0.01,
        -0.015,
        "Real trained weights and real LIBERO activations. Each point in panel b is one attention head. "
        "Ratios above 1 favor the core-derived subspace over a random subspace of the same rank.",
        fontsize=7.5,
        color=COLORS["muted"],
    )
    fig.tight_layout()
    save_both(fig, "main_exact_attention_odt")


def appendix_decomposability_audit() -> None:
    module_labels = ["Linear", "RationalNorm", "Embedding", "Patch projection"]
    module_counts = np.array([112, 74, 2, 1])
    errors = np.array([1.6362595e-6, 1.3954528e-7])

    fig, axes = plt.subplots(1, 2, figsize=(8.1, 3.2), gridspec_kw={"width_ratios": [1.15, 0.85]})

    ax = axes[0]
    y = np.arange(len(module_labels))[::-1]
    ax.barh(y, module_counts, color=[COLORS["blue"], COLORS["green"], COLORS["gray"], COLORS["cyan"]])
    ax.set_yticks(y, module_labels)
    ax.set_xlabel("Leaf modules invoked")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    for yi, count in zip(y, module_counts):
        ax.text(count + 2, yi, str(count), va="center", fontweight="bold")
    ax.set_xlim(0, 125)
    ax.set_title("Deployed operator inventory")
    ax.text(
        0.98,
        0.05,
        "Forbidden normalization 0\nSoftmax / GELU / ReLU 0\nAUDIT PASS",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=COLORS["green"],
        fontsize=8.2,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#EAF5F1", "edgecolor": COLORS["green"], "lw": 0.8},
    )
    panel_label(ax, "a")

    ax = axes[1]
    labels = ["Norm reconstruction\nmax absolute error", "Bilinear FFN\nmedian relative error"]
    bars = ax.bar(np.arange(2), errors, color=[COLORS["cyan"], COLORS["green"]], width=0.58, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(5e-9, 5e-5)
    ax.set_xticks(np.arange(2), labels)
    ax.set_ylabel("Error")
    ax.grid(axis="y", which="major", color=COLORS["grid"], linewidth=0.8, zorder=0)
    ax.set_title("Explicit P(x) / Q(x) verification")
    for bar, error in zip(bars, errors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            error * 1.6,
            f"{error:.2e}",
            ha="center",
            fontweight="bold",
            fontsize=8,
        )
    ax.text(
        0.5,
        0.84,
        "54,784 real activation rows\nResidual is bounded by deployed float32 statistics",
        transform=ax.transAxes,
        ha="center",
        fontsize=6.8,
        color=COLORS["muted"],
    )
    panel_label(ax, "b")

    fig.suptitle(
        "Mechanical operator audit of the seed-0 20.1M ViT checkpoint",
        fontsize=12,
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout()
    save_both(fig, "appendix_decomposability_audit")


def appendix_ensemble_per_task() -> None:
    raise RuntimeError(
        "Retired with the historical capability aggregates. "
        "Use scripts/plot_verified_vit_capability.py after artifact verification."
    )
    tasks = ["Soup", "Cream\ncheese", "Salad", "BBQ", "Ketchup", "Tomato", "Butter", "Milk", "Pudding", "Orange"]
    seeds = np.array(
        [
            [94, 94, 88],
            [98, 98, 98],
            [98, 98, 100],
            [88, 88, 80],
            [90, 96, 96],
            [82, 80, 80],
            [86, 100, 96],
            [100, 98, 100],
            [100, 100, 98],
            [100, 98, 96],
        ],
        dtype=float,
    )
    ensemble = np.array([92, 100, 98, 88, 98, 88, 100, 98, 100, 100], dtype=float)
    single_mean = seeds.mean(axis=1)
    delta = ensemble - single_mean

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 4.8), gridspec_kw={"height_ratios": [2.1, 0.8]}, sharex=True)
    ax = axes[0]
    x = np.arange(len(tasks))
    for seed_idx in range(3):
        ax.scatter(x, seeds[:, seed_idx], color=COLORS["gray"], s=18, alpha=0.55, zorder=2)
    ax.plot(x, single_mean, color=COLORS["blue"], marker="o", linewidth=1.6, label="Mean of single checkpoints")
    ax.plot(x, ensemble, color=COLORS["gold"], marker="D", linewidth=1.8, label="Prediction ensemble")
    ax.set_ylim(76, 102)
    ax.set_ylabel("Success (%)")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.set_title("Full matched-protocol ensemble by task")
    panel_label(ax, "a")

    ax = axes[1]
    ax.bar(x, delta, color=[COLORS["green"] if d >= 0 else COLORS["red"] for d in delta], width=0.65)
    ax.axhline(0, color=COLORS["ink"], linewidth=0.8)
    ax.set_ylabel("Gain\n(points)")
    ax.set_xticks(x, tasks)
    ax.set_ylim(-2, 8)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    panel_label(ax, "b")

    fig.text(
        0.01,
        -0.005,
        "Overall: mean of single checkpoints 93.9%, ensemble 96.2%. Gray dots are individual checkpoints.",
        fontsize=7.5,
        color=COLORS["muted"],
    )
    fig.tight_layout()
    save_both(fig, "appendix_ensemble_per_task")


def appendix_surgery_sweep() -> None:
    summary_path = ROOT / "athena" / "results" / "surgery_discovery_summary.json"
    summary = json.loads(summary_path.read_text())
    blocks = [0, 2, 4, 6, 7]
    ranks = [64, 128, 256]
    rows = {(int(row["block_index"]), int(row["rank"])): row for row in summary["rows"]}
    keep = np.array(
        [[100 * rows[(block, rank)]["keep_advantage"] for rank in ranks] for block in blocks]
    )
    removal = np.array(
        [[100 * rows[(block, rank)]["removal_advantage"] for rank in ranks] for block in blocks]
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), constrained_layout=True)
    panels = [
        (keep, "Keep selected minus matched random", TwoSlopeNorm(vmin=-15, vcenter=0, vmax=15)),
        (
            removal,
            "Matched random removal minus selected removal",
            TwoSlopeNorm(vmin=-15, vcenter=0, vmax=40),
        ),
    ]
    for ax, (values, title, norm) in zip(axes, panels):
        image = ax.imshow(values, cmap="RdYlGn", norm=norm, aspect="auto")
        ax.set_xticks(range(len(ranks)), [str(rank) for rank in ranks])
        ax.set_yticks(range(len(blocks)), [str(block) for block in blocks])
        ax.set_xlabel("Edited subspace rank")
        ax.set_ylabel("Joint block")
        ax.set_title(title)
        for row_index in range(len(blocks)):
            for column_index in range(len(ranks)):
                value = values[row_index, column_index]
                color = "white" if abs(value) >= 20 else COLORS["ink"]
                ax.text(
                    column_index,
                    row_index,
                    f"{value:+.0f}",
                    ha="center",
                    va="center",
                    fontweight="bold",
                    color=color,
                )
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("Success advantage (points)")

    fig.suptitle(
        "No coefficient-surgery configuration passes both pre-specified gates",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.025,
        "Positive is favorable in both panels. Each cell uses 20 paired task-episode states per condition.",
        ha="center",
        fontsize=7.5,
        color=COLORS["muted"],
    )
    save_both(fig, "appendix_surgery_sweep")


if __name__ == "__main__":
    appendix_matched_architecture_cost()
    appendix_counterfactual_grounding()
    main_exact_attention_odt()
    appendix_decomposability_audit()
    appendix_surgery_sweep()
    print(f"Wrote publication figures to {OUT}")
