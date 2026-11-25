"""
pipeline_figure.py  ->  outputs/paper/pipeline.pdf / .png
Fig. 1: experimental pipeline (two-column width).
Run: python -m src.pipeline_figure
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42})

BLUE_F, BLUE_E = "#e3ecf7", "#1f5fa8"
ORG_F, ORG_E = "#fbe9d8", "#d9730d"
GRY_F, GRY_E = "#efefef", "#555555"

fig, ax = plt.subplots(figsize=(7.2, 2.8))
ax.set_xlim(0, 100)
ax.set_ylim(2.5, 40)
ax.axis("off")

BW, BH = 17.2, 11.5          # box size
XS = [1.5, 21.1, 40.7, 60.3, 79.9]   # x of five columns
Y1, Y2 = 24.0, 5.5           # y of the two rows


def box(x, y, text, fc, ec, bold_first=True, w=BW, h=BH, fs=6.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                fc=fc, ec=ec, lw=1.0))
    lines = text.split("\n")
    n = len(lines)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h / 2 + (n / 2 - i - 0.5) * 3.0, ln,
                ha="center", va="center", fontsize=fs,
                fontweight="bold" if (i == 0 and bold_first) else "normal")


def arrow(x1, y1, x2, y2, color="#333333", style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=7,
                                 lw=0.9, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))


# Row labels
ax.text(0.2, 38.6, "Development phase (80% of data): fit models and tune thresholds",
        fontsize=7, fontweight="bold", color=BLUE_E, va="center")
ax.text(0.2, 21.0, "Final evaluation (test set)",
        fontsize=7, fontweight="bold", color=ORG_E, va="center")

# Row 1
r1 = [
    "Data\n283,726 transactions\n(duplicates removed)\n0.17% fraud",
    "Stratified split\n80% development\n20% test (held out)",
    "5-fold CV\nSMOTE on training\nfolds only",
    "Three tree models\nOut-of-fold p\nand disagreement u",
    r"Tune thresholds" + "\n" + r"($\ell$, $h$, $\tau$)" + "\nto minimise cost",
]
for x, t in zip(XS, r1):
    box(x, Y1, t, BLUE_F, BLUE_E, fs=6.2)
for i in range(4):
    arrow(XS[i] + BW + 0.3, Y1 + BH / 2, XS[i + 1] - 0.3, Y1 + BH / 2)

# Row 2
r2 = [
    "Final models\ntrained on whole\ndevelopment set\n(SMOTE applied)",
    "Score test set\nonce: ensemble p\nand disagreement u",
    "Frozen thresholds\nTriage rule\n(methods B and D)",
]
for x, t in zip(XS[:3], r2):
    box(x, Y2, t, ORG_F, ORG_E, fs=6.2)
arrow(XS[0] + BW + 0.3, Y2 + BH / 2, XS[1] - 0.3, Y2 + BH / 2)
arrow(XS[1] + BW + 0.3, Y2 + BH / 2, XS[2] - 0.3, Y2 + BH / 2)

# Decision container with three outcomes
cx, cw = 60.3, 38.4
ax.add_patch(FancyBboxPatch((cx, Y2 + 0.8), cw, BH - 1.6, boxstyle="round,pad=0.25,rounding_size=1.2",
                            fc="#f7f7f7", ec=GRY_E, lw=0.8, ls="--"))
ax.text(cx + cw / 2, Y2 + BH - 1.9, "Triage decision", ha="center", va="center",
        fontsize=6.2, fontweight="bold")
ow = 10.4
ox = [cx + 1.2, cx + 1.2 + ow + 1.2, cx + 1.2 + 2 * (ow + 1.2)]
for x, lab, fc, ec in zip(ox, ["Approve", "Review", "Block"],
                          ["#dff0e0", "#fff3cd", "#f8d7da"], ["#2e7d32", "#b8860b", "#b02a37"]):
    box(x, Y2 + 2.0, lab, fc, ec, w=ow, h=4.2, fs=6.4)
ax.text(ox[1] + ow / 2, Y2 - 1.2, "SHAP explanation for reviewed cases",
        ha="center", va="center", fontsize=5.8, style="italic")
arrow(XS[2] + BW + 0.3, Y2 + BH / 2, cx - 0.3, Y2 + BH / 2)

# Vertical links
tx = XS[1] + BW * 0.9
arrow(tx, Y1 - 0.3, tx, Y2 + BH + 0.4, color=ORG_E)
ax.text(tx + 0.8, (Y1 + Y2 + BH) / 2 + 1.2, "test data", fontsize=5.8,
        style="italic", color=ORG_E, va="center")
# thresholds -> frozen thresholds (elbow)
fx = XS[4] + BW / 2
yE = 19.2
ax.plot([fx, fx], [Y1 - 0.3, yE], color=BLUE_E, lw=0.9)
ax.plot([fx, XS[2] + BW / 2], [yE, yE], color=BLUE_E, lw=0.9)
arrow(XS[2] + BW / 2, yE, XS[2] + BW / 2, Y2 + BH + 0.4, color=BLUE_E)
ax.text(fx - 1.0, yE + 1.1, "frozen thresholds", fontsize=5.8, style="italic",
        color=BLUE_E, va="center", ha="right")

out = os.environ.get("PIPE_OUT", os.path.join("outputs", "paper"))
os.makedirs(out, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out, f"pipeline.{ext}"), dpi=300, bbox_inches="tight")
print("saved", os.path.join(out, "pipeline.png"))