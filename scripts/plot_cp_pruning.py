#!/usr/bin/env python3
"""Plot closed-loop retention under weight-only CP-term pruning."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("athena/results"))
    parser.add_argument(
        "--output", type=Path, default=Path("paper_figures/appendix_cp_pruning.pdf")
    )
    return parser.parse_args()


def load_result(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def pruning_point(path: Path) -> tuple[float, float]:
    result = load_result(path)
    pruning = result["cp_pruning"]
    capability = result["capability"]
    compression = 100.0 * float(pruning["whole_model_parameter_equivalent_fraction"])
    success = 100.0 * float(capability["overall"])
    return compression, success


def fraction_label(path: Path) -> str:
    match = re.search(r"_f(\d+p\d+)", path.stem)
    if match is None:
        raise ValueError(f"Cannot identify pruning fraction in {path}")
    return match.group(1)


def panel(
    axis,
    results: Path,
    magnitude_glob: str,
    random_glob: str,
    baseline_path: Path | None,
    fallback_baseline: tuple[int, int] | None,
    title: str,
) -> None:
    magnitude_paths = sorted(results.glob(magnitude_glob))
    magnitude = [pruning_point(path) for path in magnitude_paths]
    if baseline_path is not None and baseline_path.exists():
        baseline_result = load_result(baseline_path)["capability"]
        baseline = 100.0 * float(baseline_result["overall"])
    elif fallback_baseline is not None:
        baseline = 100.0 * fallback_baseline[0] / fallback_baseline[1]
    else:
        baseline = None

    if baseline is not None:
        axis.scatter([0], [baseline], marker="*", s=150, color="#1f4e79", zorder=5)
        axis.axhline(baseline, color="#1f4e79", alpha=0.22, linewidth=1.2)
        axis.text(0.8, baseline + 1.6, f"unpruned {baseline:.0f}%", color="#1f4e79")
    if magnitude:
        x, y = map(np.asarray, zip(*magnitude))
        order = np.argsort(x)
        axis.plot(
            x[order],
            y[order],
            marker="o",
            linewidth=2.2,
            markersize=6,
            color="#d1495b",
            label="lowest-norm CP terms",
            zorder=4,
        )

    random_by_fraction: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for path in results.glob(random_glob):
        random_by_fraction[fraction_label(path)].append(pruning_point(path))
    random_summary = []
    for points in random_by_fraction.values():
        compressions = [point[0] for point in points]
        successes = [point[1] for point in points]
        random_summary.append(
            (
                float(np.median(compressions)),
                float(np.median(successes)),
                float(np.min(successes)),
                float(np.max(successes)),
            )
        )
    if random_summary:
        random_summary.sort()
        x = np.asarray([row[0] for row in random_summary])
        y = np.asarray([row[1] for row in random_summary])
        low = np.asarray([row[2] for row in random_summary])
        high = np.asarray([row[3] for row in random_summary])
        axis.errorbar(
            x,
            y,
            yerr=np.stack([y - low, high - y]),
            marker="s",
            linestyle="--",
            linewidth=1.5,
            capsize=3,
            color="#6c757d",
            label="matched random, median and range",
            zorder=3,
        )

    axis.set_title(title, fontweight="bold", pad=10)
    axis.set_xlabel("Equivalent whole-model parameters removed (%)")
    axis.set_xlim(-2, 48)
    axis.set_ylim(-3, 103)
    axis.grid(axis="y", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)


def main() -> None:
    args = parse_args()
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), sharey=True)
    panel(
        axes[0],
        args.results,
        "cp_pruning_magnitude_f*.json",
        "cp_pruning_random_f*_s*.json",
        None,
        (44, 50),
        "Rational ViT checkpoint",
    )
    panel(
        axes[1],
        args.results,
        "conv_cp_pruning_magnitude_f*.json",
        "conv_cp_pruning_random_f*_s*.json",
        args.results / "conv_s1_baseline_first5.json",
        None,
        "Rational convolutional checkpoint",
    )
    axes[0].set_ylabel("Closed-loop success (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.035),
        ncol=max(1, len(labels)),
        frameon=False,
    )
    figure.suptitle(
        "Weight-only CP-term removal reveals a closed-loop compression boundary",
        y=1.11,
        fontsize=12,
        fontweight="bold",
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()

