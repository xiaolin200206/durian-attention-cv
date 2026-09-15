#!/usr/bin/env python3
"""make_graphical_abstract.py — Elsevier graphical abstract.

Single landscape panel, legible at thumbnail size. Elsevier asks for at least
560 x 1100 px; this renders at 1400 x 560 equivalent and is exported as PDF
(vector), PNG and TIFF at 400 dpi.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path(__file__).parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
})

C_DARK = "#1b1b1b"
C_MID = "#6e6e6e"
C_LIGHT = "#d9d6d1"
C_ACC = "#c1440e"
C_BLUE = "#2f6690"
MM = 1 / 25.4

fig, ax = plt.subplots(figsize=(180 * MM, 95 * MM))
ax.set_xlim(0, 100)
ax.set_ylim(0, 52)
ax.axis("off")

ax.text(50, 49.4, "The sampling unit, not the image count, governs reported accuracy",
        ha="center", va="center", fontsize=11.5, fontweight="bold", color=C_DARK)

BOX_TOP, BOX_BOT = 44.0, 4.0
H = BOX_TOP - BOX_BOT

# ---------------------------------------------------------------- left block
ax.add_patch(FancyBboxPatch((1.5, BOX_BOT), 30, H,
                            boxstyle="round,pad=0.5,rounding_size=1.0",
                            facecolor="#f4f2ef", edgecolor=C_LIGHT, linewidth=1.0))
ax.text(16.5, 41.4, "Field images are not\nindependent observations",
        ha="center", va="top", fontsize=8.4, color=C_DARK, linespacing=1.4)

rows = [("Durian, this study", "560 images", "73 capture sessions"),
        ("PlantVillage", "40,490 images", "7,524 physical leaves")]
y = 32.0
for name, imgs, units in rows:
    ax.text(16.5, y, name, ha="center", va="center", fontsize=7.6,
            fontweight="bold", color=C_DARK)
    ax.text(16.5, y - 3.4, imgs, ha="center", va="center", fontsize=7.8, color=C_MID)
    ax.add_patch(FancyArrowPatch((16.5, y - 5.2), (16.5, y - 7.0),
                                 arrowstyle="-|>", mutation_scale=8,
                                 color=C_ACC, linewidth=1.0))
    ax.text(16.5, y - 8.6, units, ha="center", va="center", fontsize=7.8,
            color=C_ACC, fontweight="bold")
    y -= 15.5

# ---------------------------------------------------------------- centre
ax.add_patch(FancyBboxPatch((34.5, BOX_BOT), 30, H,
                            boxstyle="round,pad=0.5,rounding_size=1.0",
                            facecolor="#ffffff", edgecolor=C_ACC, linewidth=1.4))
ax.text(49.5, 41.4, "Split by image, and the test\nset is not held out",
        ha="center", va="top", fontsize=8.4, color=C_DARK, linespacing=1.4)

ax.text(49.5, 32.5, "96.7%", ha="center", va="center", fontsize=15,
        fontweight="bold", color=C_ACC)
ax.text(49.5, 27.0, "99.6%", ha="center", va="center", fontsize=15,
        fontweight="bold", color=C_ACC)
ax.text(49.5, 22.2, "of test images share a specimen\nwith the training set",
        ha="center", va="center", fontsize=7.0, color=C_DARK, linespacing=1.4)

ax.plot([38.5, 60.5], [17.8, 17.8], color=C_LIGHT, lw=0.9)
ax.text(49.5, 14.8, "A near-duplicate audit of the\nsame partition reports 4.6%",
        ha="center", va="center", fontsize=7.2, color=C_BLUE, linespacing=1.4,
        fontweight="bold")
ax.text(49.5, 9.4, "Hash and DINOv2 features do not\nrecover the grouping after release",
        ha="center", va="center", fontsize=6.3, color=C_MID, linespacing=1.4)

# ---------------------------------------------------------------- right
ax.add_patch(FancyBboxPatch((67.5, BOX_BOT), 31, H,
                            boxstyle="round,pad=0.5,rounding_size=1.0",
                            facecolor="#f4f2ef", edgecolor=C_LIGHT, linewidth=1.0))
ax.text(83.0, 41.4, "Group by session and every\nnumber changes",
        ha="center", va="top", fontsize=8.4, color=C_DARK, linespacing=1.4)

models = ["EfficientNetV2-S", "ConvNeXt-Tiny", "MobileNetV3-Large"]
img = [93.4, 97.3, 92.4]
ses = [88.5, 81.8, 74.0]
yy = 32.6
for m, a, b in zip(models, img, ses):
    ax.text(69.3, yy, m, ha="left", va="center", fontsize=6.5, color=C_MID)
    x0 = 69.3 + (b - 72) * 0.60
    x1 = 69.3 + (a - 72) * 0.60
    ax.plot([x0, x1], [yy - 2.6, yy - 2.6], color="#c9c5c0", lw=2.4,
            solid_capstyle="round")
    ax.plot([x0], [yy - 2.6], "o", ms=4.4, color=C_DARK)
    ax.plot([x1], [yy - 2.6], "o", ms=4.4, color="white",
            markeredgecolor=C_ACC, markeredgewidth=1.3)
    ax.text(x1 + 1.2, yy - 2.6, f"+{a - b:.1f}", ha="left", va="center",
            fontsize=6.6, color=C_ACC, fontweight="bold")
    yy -= 7.6

ax.plot([70.2], [11.6], "o", ms=4.4, color=C_DARK)
ax.text(71.4, 11.6, "session-level", ha="left", va="center", fontsize=6.4, color=C_DARK)
ax.plot([84.0], [11.6], "o", ms=4.4, color="white",
        markeredgecolor=C_ACC, markeredgewidth=1.3)
ax.text(85.2, 11.6, "image-level", ha="left", va="center", fontsize=6.4, color=C_ACC)

ax.text(83.0, 7.4, "12.2 pp on average across nine\narchitectures, and the ranking changes",
        ha="center", va="center", fontsize=6.4, color=C_DARK,
        fontweight="bold", linespacing=1.4)

# arrows between blocks
for x in (32.3, 65.3):
    ax.add_patch(FancyArrowPatch((x, 25.0), (x + 1.8, 25.0), arrowstyle="-|>",
                                 mutation_scale=11, color=C_MID, linewidth=1.2))

ax.text(50, 1.2, "Record the sampling unit. It cannot be reconstructed once the data are released.",
        ha="center", va="center", fontsize=7.8, color=C_DARK, style="italic")

for ext in ("pdf", "png", "tiff"):
    fig.savefig(OUT / f"Graphical_abstract.{ext}")
plt.close(fig)
print("wrote Graphical_abstract.pdf / .png / .tiff")
