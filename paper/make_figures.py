"""Figures of the paper.

    python paper/make_figures.py             # run from the repository root, after compute_numbers.py

fig1_examples.pdf    two example photographs per released class, from figures/examples/
fig2_schematic.pdf   classifier and the single-channel spatial gate (SG; 'lfa' in the code)
fig3_performance.pdf (a) confusion matrix of the plain network, (b) macro F1 on each test fold
fig4_comparison.pdf  (a) paired differences from the plain network, (b) seed-only differences
All result panels use the primary scoring (video frames excluded).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / 'figures'
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
from compute_numbers import run_metrics  # noqa: E402

CONFIGS = ['none', 'lfa', 'se', 'cbam']
LABEL = {'none': 'Plain network', 'lfa': '+ SG', 'se': '+ SE', 'cbam': '+ CBAM'}
CLS_LABEL = ['Algal leaf\nspot', 'Leaf blight\ncomplex', 'Phomopsis\nleaf spot', 'Root and\ncollar rot']
INK, INK2, GRID, SURF, GREY = '#0b0b0b', '#52514e', '#e4e3df', '#ffffff', '#9a9a94'
BLUE, ORANGE = '#2a78d6', '#eb6834'


def style(fs=8):
    plt.rcParams.update({
        'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
        'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID,
        'grid.linewidth': 0.7, 'axes.spines.top': False, 'axes.spines.right': False,
        'font.size': fs, 'axes.titlesize': fs + 1, 'font.family': ['Liberation Sans', 'DejaVu Sans'],
        'pdf.fonttype': 42})


def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(OUT / f'{name}.png', dpi=400, bbox_inches='tight')
    plt.close(fig)


def fig1_examples():
    """Two example photographs per released class (pink disease was excluded from the experiments)."""
    style(8)
    cols = [('algal', 'Algal leaf spot'), ('leafblight', 'Leaf blight complex'),
            ('phomopsis', r'$\it{Phomopsis}$ leaf spot'), ('pink', 'Pink disease*'), ('root', 'Root and collar rot')]
    fig, axes = plt.subplots(2, 5, figsize=(7.1, 3.9))
    for j, (key, title) in enumerate(cols):
        for i in range(2):
            ax = axes[i, j]
            ax.imshow(plt.imread(OUT / 'examples' / f'{key}_{i + 1}.jpg'))
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s_ in ax.spines.values():
                s_.set_visible(False)
            ax.text(0.04, 0.97, f'({"abcdefghij"[i * 5 + j]})', ha='left', va='top', transform=ax.transAxes,
                    fontsize=8, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.45, lw=0))
        axes[0, j].set_title(title, fontsize=7.5)
    fig.tight_layout(h_pad=0.3, w_pad=0.3)
    save(fig, 'fig1_examples')


def fig2_schematic():
    style(7)
    LIGHT, BOX = '#e8f0fb', '#f1f0ec'

    def box(ax, x, y, w, h, text, fc=BOX, ec=INK2, bold=False, fs=7):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.06', fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=INK,
                fontweight='bold' if bold else 'normal', linespacing=1.25)

    def arrow(ax, x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=8, lw=0.9, color=INK2))

    fig, ax = plt.subplots(figsize=(7.1, 2.35))
    ax.set_xlim(0, 10.3); ax.set_ylim(0, 3.55); ax.axis('off')
    ax.text(0.05, 3.35, '(a) Classifier', fontsize=8, fontweight='bold', color=INK)
    y = 2.1
    box(ax, 0.05, y, 1.5, 1.0, 'Image\n224 × 224 × 3')
    box(ax, 1.9, y, 1.95, 1.0, 'EfficientNet-B0\nfeatures\n(ImageNet weights)')
    box(ax, 4.2, y, 2.15, 1.0, 'Attention module\nnone | SG |\nSE | CBAM', fc=LIGHT, ec=BLUE, bold=True)
    box(ax, 6.7, y, 1.5, 1.0, 'Global\naverage\npooling')
    box(ax, 8.55, y, 1.7, 1.0, 'Dropout 0.3 +\nlinear layer\n(4 classes)')
    for x0, x1 in [(1.55, 1.9), (3.85, 4.2), (6.35, 6.7), (8.2, 8.55)]:
        arrow(ax, x0, y + 0.5, x1, y + 0.5)
    ax.text(4.02, y - 0.22, '1280 × 7 × 7', ha='center', fontsize=6.5, color=INK2)
    ax.text(0.05, 1.62, '(b) Single-channel spatial gate (SG)', fontsize=8, fontweight='bold', color=INK)
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
    save(fig, 'fig2_schematic')


def load_novid():
    sess = pd.read_csv(ROOT / 'data' / 'sessions.csv')
    sess['video'] = sess.session.astype(str).str.startswith('video:')
    p = pd.read_csv(ROOT / 'results' / 'preds.csv').merge(sess[['cls', 'file', 'video']], on=['cls', 'file'])
    return p[~p.video]


def fig3_performance(N, p, r):
    style(8)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), gridspec_kw={'width_ratios': [1, 1.25]})
    ax = axes[0]
    cm = np.array(N['main_novid']['confusion_row_pct']['none'])
    ax.imshow(cm, cmap='Blues', vmin=0, vmax=100)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{cm[i, j]:.1f}', ha='center', va='center', fontsize=7.5,
                    color='white' if cm[i, j] > 55 else INK)
    short = ['Algal', 'Blight', 'Phomopsis', 'Root/collar']
    ax.set_xticks(range(4), short, rotation=30, ha='right'); ax.set_yticks(range(4), short)
    ax.set_xlabel('Predicted class'); ax.set_ylabel('True class'); ax.grid(False)
    ax = axes[1]
    pf = r.groupby(['config', 'repeat', 'fold']).macro_f1.mean().unstack('config')[CONFIGS]
    order = pf.mean(axis=1).sort_values().index
    x = np.arange(len(order))
    for c in CONFIGS:
        ax.plot(x, pf.loc[order, c].values, color=GREY, lw=0.8, alpha=0.8, zorder=2)
    ax.plot(x, pf.loc[order].mean(axis=1).values, color=BLUE, lw=2, marker='o', ms=3.5, zorder=3,
            label='mean of the four configurations')
    ax.set_xticks(x, [f'{a}.{b}' for a, b in order], rotation=90, fontsize=6.5)
    ax.set_xlabel('Test fold (repeat.fold), sorted by mean score')
    ax.set_ylabel('Macro F1 (%)')
    ax.set_ylim(20, 100)
    ax.legend(frameon=False, loc='lower right', fontsize=7)
    for a, lab in zip(axes, ['(a)', '(b)']):
        a.text(-0.02, 1.04, lab, transform=a.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='right')
    fig.tight_layout(w_pad=2.5)
    save(fig, 'fig3_performance')


def fig4_comparison(N, r):
    style(8)
    rng = np.random.default_rng(0)
    pf = r.groupby(['config', 'repeat', 'fold']).macro_f1.mean().unstack('config')[CONFIGS]
    sg = r.pivot_table(index=['config', 'repeat', 'fold'], columns='seed', values='macro_f1')
    gap = (sg[0] - sg[1]).abs()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6))
    ax = axes[0]
    comps = ['lfa', 'se', 'cbam']
    lim = 0
    for yi, c in enumerate(comps):
        d = (pf[c] - pf['none']).values
        st = N['main_novid']['paired'][c]
        lim = max(lim, np.abs(d).max())
        ax.axvspan(-2, 2, color='#f1f0ec', zorder=0) if yi == 0 else None
        ax.scatter(d, yi + rng.uniform(-0.13, 0.13, len(d)), s=14, color=GREY, edgecolor=SURF, lw=0.6, zorder=2)
        ax.plot([st['ci_low'], st['ci_high']], [yi, yi], color=BLUE, lw=2.2, zorder=3, solid_capstyle='round')
        ax.scatter([st['mean']], [yi], s=46, color=BLUE, edgecolor=SURF, lw=1.5, zorder=4)
        ax.text(1.02, yi, f"{st['mean']:+.2f}\n[{st['ci_low']:.1f}, {st['ci_high']:.1f}]", va='center',
                color=INK2, fontsize=7, transform=ax.get_yaxis_transform())
    ax.axvline(0, color=INK2, lw=0.8, zorder=1)
    ax.set_yticks(range(3), [LABEL[c] for c in comps])
    ax.set_xlabel('Macro F1, module minus plain network (pp)')
    ax.set_xlim(-np.ceil(lim + 1), np.ceil(lim + 1)); ax.set_ylim(2.5, -0.6); ax.grid(axis='y', visible=False)
    ax = axes[1]
    for yi, c in enumerate(CONFIGS):
        v = gap.loc[c].values
        ax.scatter(v, yi + rng.uniform(-0.13, 0.13, len(v)), s=14, color=GREY, edgecolor=SURF, lw=0.6, zorder=2)
        ax.plot([np.median(v)] * 2, [yi - 0.27, yi + 0.27], color=BLUE, lw=2.2, zorder=3)
        ax.text(1.02, yi, f'median\n{np.median(v):.1f}', va='center', color=INK2, fontsize=7,
                transform=ax.get_yaxis_transform())
    ax.axvline(2, color=ORANGE, lw=1.2, ls='--', zorder=1)
    ax.text(2.4, 3.6, '2-pp threshold', color=ORANGE, fontsize=7, va='center')
    ax.set_yticks(range(4), [LABEL[c] for c in CONFIGS])
    ax.set_xlabel('|seed 0 − seed 1|, same fold and configuration (pp)')
    ax.set_xlim(left=0); ax.set_ylim(3.85, -0.5); ax.grid(axis='y', visible=False)
    for a, lab in zip(axes, ['(a)', '(b)']):
        a.text(-0.02, 1.04, lab, transform=a.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='right')
    fig.tight_layout(w_pad=5)
    save(fig, 'fig4_comparison')


if __name__ == '__main__':
    N = json.load(open(HERE / 'numbers.json'))
    p = load_novid()
    r = run_metrics(p)
    fig1_examples()
    fig2_schematic()
    fig3_performance(N, p, r)
    fig4_comparison(N, r)
    print('wrote', sorted(x.name for x in OUT.iterdir()))
