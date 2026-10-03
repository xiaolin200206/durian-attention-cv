"""Figures for the Pattern Recognition Letters version, from the released result files.

    python paper/prl/make_figs_prl.py          # run from the repository root

Writes paper/prl/figs/fig1.pdf (schematic, via make_fig1.py), fig2.pdf (paired differences and
seed-only differences, two panels) and graphical_abstract.png / .pdf.
"""
import json
import runpy
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'paper' / 'prl' / 'figs'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

CONFIGS = ['none', 'lfa', 'se', 'cbam']
LABEL = {'none': 'no attention', 'lfa': '+ LFA', 'se': '+ SE', 'cbam': '+ CBAM'}
INK, INK2, GRID, SURF, GREY = '#0b0b0b', '#52514e', '#e4e3df', '#ffffff', '#9a9a94'
BLUE, ORANGE = '#2a78d6', '#eb6834'


def style(fs=8):
    plt.rcParams.update({
        'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
        'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID,
        'grid.linewidth': 0.7, 'axes.spines.top': False, 'axes.spines.right': False,
        'font.size': fs, 'axes.titlesize': fs + 1, 'axes.titleweight': 'bold',
        'font.family': ['Liberation Sans', 'DejaVu Sans'], 'pdf.fonttype': 42})


def load():
    N = json.load(open(ROOT / 'results' / 'numbers.json'))
    runs = pd.read_csv(ROOT / 'results' / 'run_metrics.csv')
    pf = runs.groupby(['config', 'repeat', 'fold']).macro_f1.mean().unstack('config')[CONFIGS]
    sg = runs.pivot_table(index=['config', 'repeat', 'fold'], columns='seed', values='macro_f1')
    gap = (sg[sg.columns[0]] - sg[sg.columns[1]]).abs()
    return N, pf, gap


def panel_a(ax, N, pf, rng, fs=8):
    comps = ['lfa', 'se', 'cbam']
    lim = 0
    for yi, c in enumerate(comps):
        d = (pf[c] - pf['none']).values
        st = N['paired_macro_f1'][c]
        lim = max(lim, np.abs(d).max())
        ax.scatter(d, yi + rng.uniform(-0.13, 0.13, len(d)), s=14, color=GREY, edgecolor=SURF, lw=0.6, zorder=2)
        ax.plot([st['ci_low'], st['ci_high']], [yi, yi], color=BLUE, lw=2.2, zorder=3, solid_capstyle='round')
        ax.scatter([st['mean']], [yi], s=46, color=BLUE, edgecolor=SURF, lw=1.5, zorder=4)
        ax.text(1.02, yi, f"{st['mean']:+.2f} [{st['ci_low']:.1f}, {st['ci_high']:.1f}]", va='center',
                color=INK2, fontsize=fs - 0.5, transform=ax.get_yaxis_transform())
    ax.text(1.02, -0.62, 'mean [95% CI]', color=INK2, fontsize=fs - 0.5, fontweight='bold',
            transform=ax.get_yaxis_transform())
    ax.axvline(0, color=INK2, lw=0.8, zorder=1)
    ax.set_yticks(range(3), [LABEL[c] for c in comps])
    ax.set_xlabel('Macro F1, module minus plain network (pp)')
    ax.set_xlim(-np.ceil(lim + 1), np.ceil(lim + 1))
    ax.set_ylim(2.5, -0.8)
    ax.grid(axis='y', visible=False)


def panel_b(ax, N, gap, rng, fs=8):
    eff = abs(N['paired_macro_f1']['lfa']['mean'])
    for yi, c in enumerate(CONFIGS):
        v = gap.loc[c].values
        ax.scatter(v, yi + rng.uniform(-0.13, 0.13, len(v)), s=14, color=GREY, edgecolor=SURF, lw=0.6, zorder=2)
        ax.plot([np.median(v)] * 2, [yi - 0.27, yi + 0.27], color=BLUE, lw=2.2, zorder=3)
        ax.text(1.02, yi, f'median {np.median(v):.1f}', va='center', color=INK2, fontsize=fs - 0.5,
                transform=ax.get_yaxis_transform())
    ax.axvline(eff, color=ORANGE, lw=1.2, ls='--', zorder=1)
    ax.text(eff + 0.3, 3.62, f'|mean LFA effect| = {eff:.2f}', color=ORANGE, fontsize=fs - 0.5, va='center')
    ax.set_yticks(range(4), [LABEL[c] for c in CONFIGS])
    ax.set_xlabel('|seed 0 − seed 1|, same fold and configuration (pp)')
    ax.set_xlim(left=0)
    ax.set_ylim(3.85, -0.5)
    ax.grid(axis='y', visible=False)


def fig2(N, pf, gap):
    style(8)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.45), gridspec_kw={'width_ratios': [1, 1]})
    panel_a(axes[0], N, pf, rng)
    panel_b(axes[1], N, gap, rng)
    for ax, lab in zip(axes, ['(a)', '(b)']):
        ax.text(-0.02, 1.06, lab, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='right')
    fig.tight_layout(w_pad=4.5)
    fig.savefig(OUT / 'fig2.pdf')
    fig.savefig(OUT / 'fig2.png', dpi=600)
    plt.close(fig)


def graphical_abstract(N, pf, gap):
    style(10)
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(8.85, 3.54))      # 2655 x 1062 px at 300 dpi (minimum 1328 x 531)
    gs = fig.add_gridspec(1, 2, left=0.11, right=0.86, top=0.66, bottom=0.17, wspace=0.95)
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    panel_a(ax1, N, pf, rng, fs=10)
    panel_b(ax2, N, gap, rng, fs=10)
    ax1.set_title('Effect of a module: about 0, CI ±4–7 pp', loc='left', fontsize=10.5)
    ax2.set_title(f"Changing only the seed: {N['seed_gap']['mean']:.1f} pp on average", loc='left', fontsize=10.5)
    fig.text(0.02, 0.94, 'Lightweight attention on 550 field durian disease images: 20 grouped folds × 2 seeds, '
             'corrected paired tests', fontsize=12, fontweight='bold', color=INK, va='center')
    fig.text(0.02, 0.86, 'EfficientNet-B0 with LFA (1,281 parameters), SE or CBAM against the plain network. '
             'No module met the pre-registered decision rule;', fontsize=9.5, color=INK2, va='center')
    fig.text(0.02, 0.80, 'the seed alone moved macro F1 by more than the 2-pp threshold. '
             'Single-run comparisons of a few points on data of this size are not informative.',
             fontsize=9.5, color=INK2, va='center')
    fig.savefig(OUT / 'graphical_abstract.png', dpi=300)
    fig.savefig(OUT / 'graphical_abstract.pdf')
    plt.close(fig)


def fig1():
    """Compact two-panel schematic for the double-column page (same content as make_fig1.py)."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    style(7)
    LIGHT, BOX = '#e8f0fb', '#f1f0ec'

    def box(ax, x, y, w, h, text, fc=BOX, ec=INK2, bold=False, fs=7):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.06', fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=INK,
                fontweight='bold' if bold else 'normal', linespacing=1.25)

    def arrow(ax, x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=8, lw=0.9, color=INK2))

    fig, ax = plt.subplots(figsize=(7.1, 2.35))
    ax.set_xlim(0, 10.3)
    ax.set_ylim(0, 3.55)
    ax.axis('off')
    ax.text(0.05, 3.35, '(a) Classifier', fontsize=8, fontweight='bold', color=INK)
    y = 2.1
    box(ax, 0.05, y, 1.5, 1.0, 'Image\n224 × 224 × 3')
    box(ax, 1.9, y, 1.95, 1.0, 'EfficientNet-B0\nfeatures\n(ImageNet weights)')
    box(ax, 4.2, y, 2.15, 1.0, 'Attention module\nnone | LFA |\nSE | CBAM', fc=LIGHT, ec=BLUE, bold=True)
    box(ax, 6.7, y, 1.5, 1.0, 'Global\naverage\npooling')
    box(ax, 8.55, y, 1.7, 1.0, 'Dropout 0.3 +\nlinear layer\n(4 classes)')
    for x0, x1 in [(1.55, 1.9), (3.85, 4.2), (6.35, 6.7), (8.2, 8.55)]:
        arrow(ax, x0, y + 0.5, x1, y + 0.5)
    ax.text(4.02, y - 0.22, '1280 × 7 × 7', ha='center', fontsize=6.5, color=INK2)
    ax.text(0.05, 1.62, '(b) Lesion-focus attention (LFA)', fontsize=8, fontweight='bold', color=INK)
    y = 0.08
    box(ax, 0.05, y, 1.5, 1.0, 'X\nC × H × W')
    box(ax, 1.95, y, 2.0, 1.0, '1 × 1 convolution\nC → 1\n(C + 1 parameters)')
    box(ax, 4.35, y, 1.5, 1.0, 'Sigmoid\nA ∈ (0, 1)\n1 × H × W')
    box(ax, 6.25, y, 0.9, 1.0, '⊙', fs=11)
    box(ax, 7.55, y, 2.7, 1.0, "X′ = X ⊙ A\none weight per position,\nshared by all channels")
    for x0, x1 in [(1.55, 1.95), (3.95, 4.35), (5.85, 6.25), (7.15, 7.55)]:
        arrow(ax, x0, y + 0.5, x1, y + 0.5)
    ax.add_patch(FancyArrowPatch((0.8, y + 1.0), (6.7, y + 1.0), connectionstyle='arc3,rad=-0.12',
                                 arrowstyle='-|>', mutation_scale=8, lw=0.9, color=INK2))
    ax.text(4.7, 1.5, 'X', fontsize=7, color=INK2, ha='center', style='italic')
    fig.savefig(OUT / 'fig1.pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(OUT / 'fig1.png', dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    N, pf, gap = load()
    fig1()
    fig2(N, pf, gap)
    graphical_abstract(N, pf, gap)
    print('wrote', sorted(p.name for p in OUT.iterdir()))
