"""Figures for the backbone and decision-support experiment.

    python paper/make_figures_v3.py          # run from the repository root, after compute_numbers_v3.py

fig5_backbones.pdf  (a) paired differences from EfficientNet-B0, (b) macro F1 against CPU latency
fig6_decision.pdf   (a) accuracy against coverage when the most confident images are decided first,
                    (b) share of each class deferred to the grower at the 90 % target
All panels use the primary scoring (video frames excluded).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import make_figures as mf  # noqa: E402
from compute_numbers import CLASSES, run_metrics  # noqa: E402
from compute_numbers_v3 import CFG, PCOLS, PRIMARY, REF  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

LABEL = {'effb0': 'EfficientNet-B0', 'mnv3l': 'MobileNetV3-L', 'mnv4m': 'MobileNetV4-M',
         'cnxv2t': 'ConvNeXt V2-T', 'vits': 'ViT-S/16'}
COL = {'effb0': mf.INK2, 'mnv3l': mf.GREY, 'mnv4m': '#c8c7c0', 'cnxv2t': mf.BLUE, 'vits': mf.ORANGE}
CLS_SHORT = ['Algal', 'Leaf blight', 'Phomopsis', 'Root rot']


def load():
    sess = pd.read_csv(ROOT / 'data' / 'sessions.csv')
    sess['video'] = sess.session.astype(str).str.startswith('video:')
    p = pd.read_csv(ROOT / 'results' / 'v3' / 'preds.csv').merge(sess[['cls', 'file', 'video']], on=['cls', 'file'])
    return p[(p.role == 'test') & ~p.video]


def fig5(N, p):
    mf.style(8)
    rng = np.random.default_rng(0)
    r = run_metrics(p)
    pf = r.groupby(['config', 'repeat', 'fold']).macro_f1.mean().unstack('config')
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), gridspec_kw={'width_ratios': [1.15, 1]})
    ax = axes[0]
    lim = 0
    for yi, c in enumerate(PRIMARY):
        d = (pf[c] - pf[REF]).values
        st = N['novid']['paired'][c]
        lim = max(lim, np.abs(d).max())
        if yi == 0:
            ax.axvspan(-2, 2, color='#f1f0ec', zorder=0)
        ax.scatter(d, yi + rng.uniform(-0.13, 0.13, len(d)), s=13, color=mf.GREY, edgecolor=mf.SURF, lw=0.6, zorder=2)
        ax.plot([st['ci_low'], st['ci_high']], [yi, yi], color=mf.BLUE, lw=2.2, zorder=3, solid_capstyle='round')
        ax.scatter([st['mean']], [yi], s=44, color=mf.BLUE, edgecolor=mf.SURF, lw=1.5, zorder=4)
        ax.text(1.02, yi, f"{st['mean']:+.1f} [{st['ci_low']:.1f}, {st['ci_high']:.1f}]\nHolm p = {st['p_holm']:.3f}",
                va='center', color=mf.INK2, fontsize=6.5, transform=ax.get_yaxis_transform())
    ax.axvline(0, color=mf.INK2, lw=0.8, zorder=1)
    ax.set_yticks(range(4), [LABEL[c] for c in PRIMARY])
    ax.set_xlabel('Macro F1, backbone minus EfficientNet-B0 (pp)')
    ax.set_xlim(-np.ceil(lim + 1), np.ceil(lim + 1)); ax.set_ylim(3.5, -0.6); ax.grid(axis='y', visible=False)
    ax = axes[1]
    for c in CFG:
        x, y = N['cost'][c]['cpu_ms_1thr_median'], N['novid']['summary'][c]['mean']
        ax.scatter([x], [y], s=34 + 3 * N['cost'][c]['params_M'], color=COL[c], edgecolor=mf.SURF, lw=1.0, zorder=3)
        dx, dy, ha = (6, 0.9, 'left')
        if c == 'mnv4m': dx, dy = (6, -1.9)
        if c == 'effb0': dx, dy = (6, 0.7)
        if c == 'mnv3l': dx, dy = (6, -2.0)
        ax.text(x + dx, y + dy, LABEL[c], fontsize=6.8, color=mf.INK2, ha=ha)
    ax.set_xlabel('CPU latency, one thread, batch 1 (ms)')
    ax.set_ylabel('Macro F1 (%)')
    ax.set_xlim(0, 215); ax.set_ylim(54, 80)
    for a, lab in zip(axes, ['(a)', '(b)']):
        a.text(-0.02, 1.04, lab, transform=a.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='right')
    fig.tight_layout(w_pad=5.5)
    mf.save(fig, 'fig5_backbones')


def fig6(N, p):
    mf.style(8)
    grid = np.linspace(0.05, 1.0, 20)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), gridspec_kw={'width_ratios': [1, 1.1]})
    ax = axes[0]
    for c in CFG:
        accs = []
        for _, g in p[p.config == c].groupby(['repeat', 'fold', 'seed']):
            conf, corr = g[PCOLS].values.max(axis=1), (g.y_true == g.y_pred).values
            o = np.argsort(-conf, kind='stable')
            cum = np.cumsum(corr[o]) / np.arange(1, len(o) + 1)
            accs.append([cum[max(0, int(round(len(o) * q)) - 1)] * 100 for q in grid])
        ax.plot(grid * 100, np.mean(accs, 0), color=COL[c], lw=1.8 if c in ('cnxv2t', 'vits', 'effb0') else 1.1,
                label=LABEL[c], zorder=3)
    ax.axhline(90, color=mf.ORANGE, lw=0.9, ls='--', zorder=1)
    ax.text(6, 90.6, '90 % target', color=mf.ORANGE, fontsize=6.8)
    ax.set_xlabel('Coverage: share of images decided by the model (%)')
    ax.set_ylabel('Accuracy on decided images (%)')
    ax.set_xlim(5, 100); ax.set_ylim(60, 101)
    ax.legend(frameon=False, fontsize=6.5, loc='lower left')
    ax = axes[1]
    show = ['effb0', 'cnxv2t', 'vits']
    w = 0.26
    for k, c in enumerate(show):
        v = [N['novid']['selective']['90'][c]['defer_pct_run_mean'][cl] for cl in CLASSES]
        ax.bar(np.arange(4) + (k - 1) * w, v, w * 0.92, color=COL[c], label=LABEL[c], zorder=3)
        for i, x in enumerate(v):
            ax.text(i + (k - 1) * w, x + 1.2, f'{x:.0f}', ha='center', fontsize=6, color=mf.INK2)
    ax.set_xticks(range(4), CLS_SHORT)
    ax.set_ylabel('Images referred to the grower (%)')
    ax.set_ylim(0, 62); ax.grid(axis='x', visible=False)
    ax.legend(frameon=False, fontsize=6.5, loc='upper right')
    for a, lab in zip(axes, ['(a)', '(b)']):
        a.text(-0.02, 1.04, lab, transform=a.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='right')
    fig.tight_layout(w_pad=3)
    mf.save(fig, 'fig6_decision')


if __name__ == '__main__':
    N = json.load(open(HERE / 'numbers_v3.json'))
    p = load()
    fig5(N, p)
    fig6(N, p)
    print('wrote fig5_backbones, fig6_decision')
