"""Schematic: where the attention module sits, and what the single-channel spatial gate (SG; lfa in the code) computes."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Liberation Sans', 'DejaVu Sans']
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, INK2, BLUE, LIGHT, GREY = '#0b0b0b', '#52514e', '#2a78d6', '#e8f0fb', '#f1f0ec'


def box(ax, x, y, w, h, text, fc=GREY, ec=INK2, bold=False, fs=7.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.06', fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=INK,
            fontweight='bold' if bold else 'normal', linespacing=1.3)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=10, lw=1, color=INK2))


fig, ax = plt.subplots(figsize=(6.85, 3.7))
ax.set_xlim(0, 10.3)
ax.set_ylim(0, 5.6)
ax.axis('off')
# (a) pipeline
ax.text(0.05, 5.35, '(a) Classifier', fontsize=9, fontweight='bold', color=INK)
y = 3.95
box(ax, 0.05, y, 1.5, 1.0, 'Image\n224 × 224 × 3')
box(ax, 1.9, y, 1.95, 1.0, 'EfficientNet-B0\nfeatures\n(ImageNet weights)')
box(ax, 4.2, y, 2.15, 1.0, 'Attention module\nnone | SG |\nSE | CBAM', fc=LIGHT, ec=BLUE, bold=True)
box(ax, 6.7, y, 1.5, 1.0, 'Global\naverage\npooling')
box(ax, 8.55, y, 1.7, 1.0, 'Dropout 0.3 +\nlinear layer\n(4 classes)')
for x0, x1 in [(1.55, 1.9), (3.85, 4.2), (6.35, 6.7), (8.2, 8.55)]:
    arrow(ax, x0, y + 0.5, x1, y + 0.5)
ax.text(4.02, y - 0.25, '1280 × 7 × 7', ha='center', fontsize=7, color=INK2)
# (b) LFA
ax.text(0.05, 2.65, '(b) Single-channel spatial gate (SG)', fontsize=9, fontweight='bold', color=INK)
y = 0.35
box(ax, 0.05, y, 1.5, 1.0, 'X\nC × H × W')
box(ax, 1.95, y, 2.0, 1.0, '1 × 1 convolution\nC → 1\n(C + 1 parameters)')
box(ax, 4.35, y, 1.5, 1.0, 'Sigmoid\nA ∈ (0, 1)\n1 × H × W')
box(ax, 6.25, y, 0.9, 1.0, '⊙', fs=12)
box(ax, 7.55, y, 2.7, 1.0, "X′ = X ⊙ A\none weight per position,\nshared by all channels")
for x0, x1 in [(1.55, 1.95), (3.95, 4.35), (5.85, 6.25), (7.15, 7.55)]:
    arrow(ax, x0, y + 0.5, x1, y + 0.5)
ax.add_patch(FancyArrowPatch((0.8, y + 1.0), (6.7, y + 1.0), connectionstyle='arc3,rad=-0.18',
                             arrowstyle='-|>', mutation_scale=10, lw=1, color=INK2))
ax.text(3.75, 1.95, 'X', fontsize=8, color=INK2, ha='center', style='italic')
fig.savefig('figures/fig1_architecture.png', dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig('figures/fig1_architecture.eps', bbox_inches='tight', facecolor='white')
print('ok')
