"""
generate_fm_figure.py
---------------------
Generates Figure: Failure Mode Distribution — Baseline vs SEP.

Layout: two pie charts side by side, one per condition.
Each pie shows the percentage contribution of FM1, FM2a, FM2b (non-FM),
and tasks with no failure (discovered ≥1 blocker without abandonment).

Output: paper/figures/fm_distribution.pdf  (and .png for preview)

Usage:
    python generate_fm_figure.py

Requirements:
    pip install matplotlib
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data (averaged across 3 passes, 100 tasks each) ───────────────────────────
# All values are % of 100 tasks

DATA = {
    "Baseline": {
        "FM1: Silent Omission":         55.0,
        "FM2a: Abandonment":            19.0,
        "FM2b: Continued after reject":  0.3,
        "No failure":                   25.7,   # 100 - 55 - 19 - 0.3
    },
    "SEP (ours)": {
        "FM1: Silent Omission":          7.3,
        "FM2a: Abandonment":            15.7,
        "FM2b: Continued after reject": 13.3,
        "No failure":                   63.7,   # 100 - 7.3 - 15.7 - 13.3
    },
}

# ── Colors — vibrant, accessible ─────────────────────────────────────────────
COLORS = {
    "FM1: Silent Omission":          "#E05A2B",   # deep orange
    "FM2a: Abandonment":             "#F5A623",   # amber
    "FM2b: Continued after reject":  "#4A90D9",   # steel blue
    "No failure":                    "#5DBB63",   # medium green
}

LABELS_SHORT = {
    "FM1: Silent Omission":          "FM1\nSilent Omission",
    "FM2a: Abandonment":             "FM2a\nAbandonment",
    "FM2b: Continued after reject":  "FM2b\nContinued",
    "No failure":                    "No failure\n(blocker found)",
}

# ── Figure ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
fig.subplots_adjust(top=0.82, bottom=0.05, left=0.02, right=0.98, wspace=0.35)

for ax, (condition, data) in zip(axes, DATA.items()):
    labels  = list(data.keys())
    sizes   = list(data.values())
    colors  = [COLORS[l] for l in labels]

    # Explode FM1 slice slightly to highlight it
    explode = [0.04 if "FM1" in l else 0.0 for l in labels]

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        colors=colors,
        explode=explode,
        autopct=lambda p: f"{p:.1f}%" if p > 2 else "",
        pctdistance=0.70,
        startangle=90,
        counterclock=False,
        wedgeprops={"linewidth": 1.2, "edgecolor": "white"},
        textprops={"fontsize": 9},
    )

    for at in autotexts:
        at.set_fontsize(8.5)
        at.set_fontweight("bold")
        at.set_color("white")

    ax.set_title(condition, fontsize=13, fontweight="bold", pad=14)

# ── Shared legend below both pies ─────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(color=COLORS[l], label=LABELS_SHORT[l].replace("\n", " "))
    for l in COLORS
]
fig.legend(
    handles=legend_patches,
    loc="lower center",
    ncol=4,
    fontsize=8.5,
    frameon=True,
    framealpha=0.9,
    edgecolor="#cccccc",
    bbox_to_anchor=(0.5, -0.02),
)

fig.suptitle(
    "Failure Mode Distribution — Baseline vs SEP\n"
    r"\small{(averaged across 3 passes, 100 tasks each)}",
    fontsize=12,
    fontweight="bold",
    y=0.97,
)

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs("paper/figures", exist_ok=True)
fig.savefig("paper/figures/fm_distribution.pdf", bbox_inches="tight", dpi=300)
fig.savefig("paper/figures/fm_distribution.png", bbox_inches="tight", dpi=200)
print("Saved: paper/figures/fm_distribution.pdf")
print("Saved: paper/figures/fm_distribution.png")
plt.close()
