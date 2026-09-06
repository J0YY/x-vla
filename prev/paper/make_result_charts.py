#!/usr/bin/env python3
"""Turn the three 'RESULTS SO FAR' tables into simple bar charts."""
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os

# --- validated dataviz palette (light mode) ---
BLUE   = "#2a78d6"   # "ours" / global ODT  -> the method we want to look good
GRAY   = "#b8b6ad"   # baseline / local
GREEN  = "#008300"   # strong baseline
ORANGE = "#eb6834"
RED    = "#e34948"   # random (the "bad" control)
INK    = "#0b0b0b"
INK2   = "#52514e"
GRID   = "#e6e4dc"
SURF   = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF,
    "font.size": 12, "font.family": "DejaVu Sans",
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK,
})

OUT = os.path.dirname(os.path.abspath(__file__))

def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, lw=1)

def label_bars(ax, bars, fmt="{:.3f}", dy=0):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, h+dy, fmt.format(h),
                ha="center", va="bottom", fontsize=10, color=INK)

# ============================================================
# CHART 1 — accuracy retained vs. #directions kept (higher=better)
# ============================================================
k       = ["4", "6", "8", "12", "16"]
g_odt   = [0.571, 0.777, 0.797, 0.809, 0.811]
l_svd   = [0.523, 0.686, 0.798, 0.807, 0.811]

fig, ax = plt.subplots(figsize=(8, 4.6))
x = range(len(k)); w = 0.38
b1 = ax.bar([i-w/2 for i in x], g_odt, w, label="global ODT (ours)", color=BLUE)
b2 = ax.bar([i+w/2 for i in x], l_svd, w, label="local SVD", color=GRAY)
label_bars(ax, b1, dy=0.004); label_bars(ax, b2, dy=0.004)
style(ax)
ax.set_xticks(list(x)); ax.set_xticklabels(k)
ax.set_xlabel("directions kept  (out of 32)")
ax.set_ylabel("accuracy retained")
ax.set_ylim(0.5, 0.86)
ax.set_title("Accuracy kept as we shrink to k directions  (higher = better)",
             fontsize=13, weight="bold", loc="left", pad=12)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(f"{OUT}/chart1_directions.png", dpi=200)
plt.close(fig)

# ============================================================
# CHART 2 — faithfulness: deletion (lower=better) & insertion (higher=better)
# ============================================================
data = {
    "χ-ViT (attention)": {
        "methods": ["ODT (ours)", "PCA", "random"],
        "colors":  [BLUE, GRAY, RED],
        "deletion":  [0.150, 0.151, 0.498],
        "insertion": [0.792, 0.788, 0.491],
    },
    "classifier": {
        "methods": ["ODT (ours)", "int. grad", "PCA", "random"],
        "colors":  [BLUE, GREEN, GRAY, RED],
        "deletion":  [0.222, 0.191, 0.204, 0.548],
        "insertion": [0.786, 0.784, 0.800, 0.530],
    },
}

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for row, (sub, d) in enumerate(data.items()):
    for col, (metric, arrow, better) in enumerate(
            [("deletion", "↓", "lower = better"),
             ("insertion", "↑", "higher = better")]):
        ax = axes[row][col]
        vals = d[metric]
        bars = ax.bar(d["methods"], vals, color=d["colors"], width=0.62)
        label_bars(ax, bars, dy=0.008)
        style(ax)
        ax.set_ylim(0, 0.9)
        ax.set_title(f"{sub}  —  {metric} AUC {arrow}  ({better})",
                     fontsize=12, weight="bold", loc="left", pad=8)
        ax.tick_params(axis="x", labelsize=10)
fig.suptitle("Are the directions we call important actually important?",
             fontsize=14, weight="bold", x=0.02, ha="left")
fig.text(0.02, 0.945,
         "delete them → accuracy should collapse (low deletion AUC)   •   "
         "add them back → accuracy should recover (high insertion AUC)",
         fontsize=10.5, color=INK2, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f"{OUT}/chart2_faithfulness.png", dpi=200)
plt.close(fig)

# ============================================================
# CHART 3 — attention compression error (lower=better)
# ============================================================
comp = {
    "1 attention layer": {"full": 5.23,
        "k": ["32", "64", "96"],
        "global": [6.11, 5.47, 5.27], "local": [7.88, 6.27, 5.55]},
    "2 attention layers": {"full": 5.98,
        "k": ["32", "64", "96"],
        "global": [6.18, 6.02, 5.99], "local": [7.78, 6.52, 6.18]},
}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
for ax, (name, d) in zip(axes, comp.items()):
    x = range(len(d["k"])); w = 0.38
    b1 = ax.bar([i-w/2 for i in x], d["global"], w,
                label="global (output-aware, ours)", color=BLUE)
    b2 = ax.bar([i+w/2 for i in x], d["local"], w,
                label="local (weight SVD)", color=GRAY)
    label_bars(ax, b1, fmt="{:.2f}", dy=0.03)
    label_bars(ax, b2, fmt="{:.2f}", dy=0.03)
    ax.axhline(d["full"], ls="--", lw=1.6, color=INK2)
    ax.text(0.015, 0.965, f"– – –  full uncompressed model = {d['full']}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9.5, color=INK2)
    style(ax)
    ax.set_xticks(list(x)); ax.set_xticklabels(d["k"])
    ax.set_xlabel("ranks retained  k")
    ax.set_ylim(5.0, 8.6)
    ax.set_title(name, fontsize=12, weight="bold", loc="left", pad=8)
axes[0].set_ylabel("action error  (lower = better)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, loc="upper right", ncol=2,
           fontsize=10, bbox_to_anchor=(0.99, 0.995))
fig.suptitle("Does it still work through attention?",
             fontsize=13.5, weight="bold", x=0.02, y=0.98, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f"{OUT}/chart3_attention.png", dpi=200)
plt.close(fig)

print("wrote:", os.listdir(OUT))
