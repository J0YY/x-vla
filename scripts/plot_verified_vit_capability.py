#!/usr/bin/env python3
"""Plot the verified ViT member and arithmetic-mean ensemble capability."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from artifacts.vit_capability.verify import verify_artifact  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "paper_figures" / "vit_capability_verified.pdf",
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    summary = verify_artifact(ROOT)
    members = summary["members"]
    ensemble = summary["ensemble"]
    figure_data = summary["figure_data"]

    plt.rcParams.update(
        {
            "axes.linewidth": 0.7,
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )
    member_color = "#356B9A"
    ensemble_color = "#D9772D"
    grid_color = "#D7DCE2"

    figure, (overall_axis, task_axis) = plt.subplots(
        1,
        2,
        figsize=(5.5, 3.15),
        gridspec_kw={"width_ratios": [0.82, 1.65], "wspace": 0.38},
    )

    labels = ["Seed 0", "Seed 1", "Seed 2", "Mean ens."]
    rates = [member["success_rate"] for member in members] + [ensemble["success_rate"]]
    positions = np.arange(len(labels))[::-1]
    colors = [member_color] * 3 + [ensemble_color]
    overall_axis.barh(positions, rates, color=colors, height=0.58, edgecolor="white", linewidth=0.4)
    overall_axis.axvline(
        summary["member_success_rate_mean"],
        color="#4C566A",
        linewidth=0.9,
        linestyle=(0, (3, 2)),
        zorder=0,
    )
    for position, rate in zip(positions, rates):
        overall_axis.text(rate + 0.006, position, f"{100 * rate:.1f}", va="center", fontsize=7)
    overall_axis.text(
        0.605,
        3.48,
        "member mean 85.3",
        ha="left",
        va="bottom",
        color="#4C566A",
        fontsize=6.5,
    )
    overall_axis.set_yticks(positions, labels)
    overall_axis.set_xlim(0.6, 1.02)
    overall_axis.set_ylim(-1.15, 3.58)
    overall_axis.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0], ["60", "70", "80", "90", "100"])
    overall_axis.set_xlabel("Success (%)")
    overall_axis.set_title("a  Overall", loc="left", fontweight="bold", fontsize=8)
    overall_axis.grid(axis="x", color=grid_color, linewidth=0.55)
    overall_axis.set_axisbelow(True)
    overall_axis.spines[["top", "right"]].set_visible(False)
    overall_axis.text(
        0.605,
        -0.72,
        "+8.1 vs. member mean\n+3.4 vs. best, 3 passes",
        ha="left",
        va="top",
        color=ensemble_color,
        fontsize=6.6,
        fontweight="bold",
    )

    task_positions = np.arange(10)[::-1]
    height = 0.34
    member_task_rates = np.asarray(figure_data["member_per_task_mean_success_rates"])
    ensemble_task_rates = np.asarray(figure_data["ensemble_per_task_success_rates"])
    task_axis.barh(
        task_positions + height / 2,
        member_task_rates,
        height=height,
        color=member_color,
        label="3-seed mean",
    )
    task_axis.barh(
        task_positions - height / 2,
        ensemble_task_rates,
        height=height,
        color=ensemble_color,
        label="Ensemble",
    )
    task_axis.set_yticks(task_positions, figure_data["task_labels"])
    task_axis.set_xlim(0.3, 1.01)
    task_axis.set_xticks([0.4, 0.6, 0.8, 1.0], ["40", "60", "80", "100"])
    task_axis.set_xlabel("Success (%)")
    task_axis.set_title("b  Per task", loc="left", fontweight="bold", fontsize=8)
    task_axis.grid(axis="x", color=grid_color, linewidth=0.55)
    task_axis.set_axisbelow(True)
    task_axis.spines[["top", "right"]].set_visible(False)
    task_axis.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.01),
        frameon=False,
        ncol=2,
        borderaxespad=0,
        columnspacing=1.0,
        handlelength=1.4,
        fontsize=6.4,
    )

    figure.text(
        0.5,
        0.032,
        "LIBERO-Object, 50 canonical trials per task, 280-step cap",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="#4C566A",
    )
    figure.subplots_adjust(left=0.14, right=0.99, top=0.91, bottom=0.17)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fixed_time = dt.datetime(2026, 8, 21, 0, 0, 0, tzinfo=dt.timezone.utc)
    figure.savefig(
        args.output,
        format="pdf",
        metadata={
            "Title": "Verified ViT capability",
            "Author": "Anonymous",
            "Subject": "LIBERO-Object capability under the 280-step protocol",
            "Keywords": "robot learning, capability, ensemble",
            "Creator": "plot_verified_vit_capability.py",
            "CreationDate": fixed_time,
            "ModDate": fixed_time,
        },
    )
    plt.close(figure)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
