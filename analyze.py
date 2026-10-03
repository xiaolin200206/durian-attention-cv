"""Recompute every number, table and figure in the paper from the saved predictions.

    python analyze.py --results results

Reads results/preds.csv, results/results.csv, results/latency_params.csv,
results/external_vietnam.csv and data/sessions.csv. Writes results/numbers.json, the tables and figures/*.png.
Needs no GPU and no images.

Macro F1 is computed over the classes present in each test fold. Three of the 20 folds have
no Root_disease image in the test set; scoring that class as zero in those folds would lower
their macro F1 by about a quarter for every configuration alike (see README).
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

CLASSES = ['Algal', 'Leaf_rot', 'Phomopsis', 'Root_disease']
CONFIGS = ['none', 'lfa', 'se', 'cbam']
LABEL = {'none': 'B0 (no attention)', 'lfa': 'B0 + LFA', 'se': 'B0 + SE', 'cbam': 'B0 + CBAM'}
INK, INK2, GRID, SURF, GREY = '#0b0b0b', '#52514e', '#e4e3df', '#ffffff', '#9a9a94'
BLUE, ORANGE = '#2a78d6', '#eb6834'


def style():
    plt.rcParams.update({
        'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
        'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID,
        'grid.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
        'font.size': 9, 'axes.titlesize': 10, 'axes.titleweight': 'bold', 'font.family': ['Liberation Sans', 'DejaVu Sans']})


def corrected_ttest(d, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test over J paired fold differences."""
    d = np.asarray(d, dtype=float)
    J = len(d)
    se = np.sqrt((1 / J + n_test / n_train) * d.var(ddof=1))
    t = d.mean() / se
    h = stats.t.ppf(0.975, J - 1) * se
    return {'mean': d.mean(), 'ci_low': d.mean() - h, 'ci_high': d.mean() + h,
            'p_corrected': 2 * stats.t.sf(abs(t), J - 1),
            'p_uncorrected': stats.ttest_1samp(d, 0).pvalue,
            'p_wilcoxon': stats.wilcoxon(d).pvalue,
            'wins': int((d > 0).sum()), 'losses': int((d < 0).sum()), 'J': J}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results')
    ap.add_argument('--figures', default='figures')
    ap.add_argument('--data', default='data')
    a = ap.parse_args()
    R, F = Path(a.results), Path(a.figures)
    F.mkdir(exist_ok=True)
    style()
    out = {}

    res = pd.read_csv(R / 'results.csv')
    preds = pd.read_csv(R / 'preds.csv')

    # ---- per-run metrics, recomputed from predictions
    rows = []
    for (cfg, rep, k, s), g in preds.groupby(['config', 'repeat', 'fold', 'seed']):
        present = sorted(set(g.y_true))
        rows.append({'config': cfg, 'repeat': rep, 'fold': k, 'seed': s, 'n_test': len(g),
                     'classes_in_test': len(present),
                     'macro_f1': f1_score(g.y_true, g.y_pred, labels=present, average='macro', zero_division=0) * 100,
                     'accuracy': accuracy_score(g.y_true, g.y_pred) * 100})
    runs = pd.DataFrame(rows)
    runs.to_csv(R / 'run_metrics.csv', index=False)
    sizes = res.groupby(['repeat', 'fold'])[['n_train', 'n_val', 'n_test']].first()
    n_train = float((sizes.n_train + sizes.n_val).mean())
    n_test = float(sizes.n_test.mean())
    out['design'] = {
        'runs': int(len(runs)), 'folds': int(len(sizes)),
        'train_mean': round(float(sizes.n_train.mean()), 1), 'train_min': int(sizes.n_train.min()), 'train_max': int(sizes.n_train.max()),
        'val_mean': round(float(sizes.n_val.mean()), 1), 'val_min': int(sizes.n_val.min()), 'val_max': int(sizes.n_val.max()),
        'test_mean': round(n_test, 1), 'test_min': int(sizes.n_test.min()), 'test_max': int(sizes.n_test.max()),
        'folds_missing_a_class': int((runs.groupby(['repeat', 'fold']).classes_in_test.first() < 4).sum()),
        'test_over_train_ratio': round(n_test / n_train, 4)}

    summ = runs.groupby('config')[['macro_f1', 'accuracy']].agg(['mean', 'std', 'min', 'max']).reindex(CONFIGS)
    out['summary'] = {c: {'macro_f1_mean': round(summ.loc[c, ('macro_f1', 'mean')], 2),
                          'macro_f1_sd': round(summ.loc[c, ('macro_f1', 'std')], 2),
                          'macro_f1_min': round(summ.loc[c, ('macro_f1', 'min')], 2),
                          'macro_f1_max': round(summ.loc[c, ('macro_f1', 'max')], 2),
                          'acc_mean': round(summ.loc[c, ('accuracy', 'mean')], 2),
                          'acc_sd': round(summ.loc[c, ('accuracy', 'std')], 2)} for c in CONFIGS}

    # ---- paired comparisons (seeds averaged within fold)
    pf = runs.groupby(['config', 'repeat', 'fold']).macro_f1.mean().unstack('config')[CONFIGS]
    pa = runs.groupby(['config', 'repeat', 'fold']).accuracy.mean().unstack('config')[CONFIGS]
    out['paired_macro_f1'], out['paired_accuracy'] = {}, {}
    for c in ['lfa', 'se', 'cbam']:
        out['paired_macro_f1'][c] = {k: round(float(v), 3) for k, v in corrected_ttest(pf[c] - pf['none'], n_train, n_test).items()}
        out['paired_accuracy'][c] = {k: round(float(v), 3) for k, v in corrected_ttest(pa[c] - pa['none'], n_train, n_test).items()}
    fm = pf.mean(axis=1)
    out['fold_spread'] = {'fold_mean_min': round(float(fm.min()), 1), 'fold_mean_max': round(float(fm.max()), 1),
                          'sd_of_fold_means': round(float(fm.std(ddof=1)), 2),
                          'sd_of_config_means': round(float(pf.mean().std(ddof=1)), 2)}

    # ---- run-to-run variability
    base = runs[runs.config == 'none'].set_index(['repeat', 'fold', 'seed']).macro_f1
    out['single_run'] = {}
    for c in ['lfa', 'se', 'cbam']:
        d = runs[runs.config == c].set_index(['repeat', 'fold', 'seed']).macro_f1 - base
        out['single_run'][c] = {'share_ahead_3pp': round(float((d >= 3).mean()), 3),
                                'share_behind_3pp': round(float((d <= -3).mean()), 3),
                                'min': round(float(d.min()), 2), 'max': round(float(d.max()), 2), 'pairs': int(len(d))}
    sg = runs.pivot_table(index=['config', 'repeat', 'fold'], columns='seed', values='macro_f1')
    gap = (sg[sg.columns[0]] - sg[sg.columns[1]]).abs()
    out['seed_gap'] = {'mean': round(float(gap.mean()), 2), 'median': round(float(gap.median()), 2),
                       'max': round(float(gap.max()), 2),
                       'by_config_median': {c: round(float(gap.loc[c].median()), 2) for c in CONFIGS}}

    # ---- per-class metrics (only runs in which the class is in the test set)
    pc = []
    for (cfg, rep, k, s), g in preds.groupby(['config', 'repeat', 'fold', 'seed']):
        p, r, f1, n = precision_recall_fscore_support(g.y_true, g.y_pred, labels=list(range(4)), zero_division=0)
        for i, cl in enumerate(CLASSES):
            if n[i]:
                pc.append({'config': cfg, 'class': cl, 'precision': p[i] * 100, 'recall': r[i] * 100, 'f1': f1[i] * 100})
    pc = pd.DataFrame(pc)
    t = pc.groupby(['config', 'class'])[['precision', 'recall', 'f1']].agg(['mean', 'std'])
    t = t.reindex(pd.MultiIndex.from_product([CONFIGS, CLASSES])).round(1)
    t.to_csv(R / 'per_class_metrics.csv')
    out['per_class'] = {f'{c}|{cl}': {m: f"{t.loc[(c, cl), (m, 'mean')]:.1f} ± {t.loc[(c, cl), (m, 'std')]:.1f}"
                                      for m in ['precision', 'recall', 'f1']} for c in CONFIGS for cl in CLASSES}
    out['per_class_runs'] = pc.groupby(['config', 'class']).size().unstack().loc['none'].to_dict()
    cms = {}
    for cfg in CONFIGS:
        g = preds[preds.config == cfg]
        cm = confusion_matrix(g.y_true, g.y_pred, labels=list(range(4)))
        cms[cfg] = (cm / cm.sum(1, keepdims=True) * 100).round(1).tolist()
    out['confusion_row_pct'] = cms

    # ---- cost
    lat = pd.read_csv(R / 'latency_params.csv')
    out['cost'] = lat.set_index('config').to_dict(orient='index')

    # ---- external
    ext = pd.read_csv(R / 'external_vietnam.csv')
    out['external'] = {'n_images': int(ext.n_vietnam.iloc[0]), 'seeds': int(ext.seed.nunique())}
    for col in ['vn_macro_f1_full', 'vn_macro_f1_3class', 'vn_pred_as_root_pct']:
        g = ext.groupby('config')[col].agg(['mean', 'std']).reindex(CONFIGS)
        out['external'][col] = {c: [round(g.loc[c, 'mean'], 2), round(g.loc[c, 'std'], 2)] for c in CONFIGS}
    piv = ext.pivot_table(index='seed', columns='config', values='vn_macro_f1_full')
    out['external']['paired'] = {}
    for c in ['lfa', 'se', 'cbam']:
        d = piv[c] - piv['none']
        out['external']['paired'][c] = {'mean': round(float(d.mean()), 2),
                                        'p_paired_t': round(float(stats.ttest_rel(piv[c], piv['none']).pvalue), 3),
                                        'wins': int((d > 0).sum()), 'n': int(len(d))}
    out['external']['val_macro_f1_mean'] = round(float(ext.val_macro_f1.mean()), 1)

    # ---- grouping sensitivity: video frames were not linked to adjacent stills (see README)
    import common as Cm
    sess = pd.read_csv(Path(a.data) / 'sessions.csv')
    sess = sess[sess.cls.isin(CLASSES)].reset_index(drop=True)
    sess['group_linked'] = Cm.build_groups(sess, link_videos=True).values
    sess['video'] = sess.session.astype(str).str.startswith('video:')
    size_linked = sess.group_linked.value_counts()
    pv = preds.merge(sess[['cls', 'file', 'group_linked', 'video']], on=['cls', 'file'], how='left')
    assert pv.group_linked.notna().all(), 'prediction file not found in sessions.csv'
    ph = sess[sess.cls == 'Phomopsis']
    out['grouping'] = {'video_frames': int(sess.video.sum()), 'videos': int(sess[sess.video].session.nunique()),
                       'phomopsis_groups_linked': int(ph.group_linked.nunique()),
                       'largest_group_linked': int(size_linked.iloc[0]),
                       'largest_group_linked_phomopsis': int((ph.group_linked == size_linked.index[0]).sum())}
    g0 = pv[(pv.config == 'none') & (pv.cls == 'Phomopsis')]
    out['grouping']['phomopsis_recall_video'] = round(float((g0[g0.video].y_true == g0[g0.video].y_pred).mean() * 100), 1)
    out['grouping']['phomopsis_recall_stills'] = round(float((g0[~g0.video].y_true == g0[~g0.video].y_pred).mean() * 100), 1)
    sens_rows = []
    for variant in ['all', 'no_video', 'unsplit_linked']:
        rr = []
        for (cfg, rep, k, s), g in pv.groupby(['config', 'repeat', 'fold', 'seed']):
            if variant == 'no_video':
                g = g[~g.video]
            elif variant == 'unsplit_linked':
                cnt = g.group_linked.value_counts()
                split = cnt[cnt < size_linked[cnt.index].values].index
                g = g[~g.group_linked.isin(split)]
            present = sorted(set(g.y_true))
            rr.append({'config': cfg, 'repeat': rep, 'fold': k, 'seed': s, 'n': len(g),
                       'macro_f1': f1_score(g.y_true, g.y_pred, labels=present, average='macro', zero_division=0) * 100})
        rr = pd.DataFrame(rr)
        if variant == 'no_video':
            sgv = rr.pivot_table(index=['config', 'repeat', 'fold'], columns='seed', values='macro_f1')
            gv = (sgv[sgv.columns[0]] - sgv[sgv.columns[1]]).abs()
            bv = rr[rr.config == 'none'].set_index(['repeat', 'fold', 'seed']).macro_f1
            dv = rr[rr.config == 'lfa'].set_index(['repeat', 'fold', 'seed']).macro_f1 - bv
            out['grouping']['no_video_seed_gap_mean'] = round(float(gv.mean()), 2)
            out['grouping']['no_video_lfa_share_ahead_3pp'] = round(float((dv >= 3).mean()), 3)
            out['grouping']['no_video_lfa_share_behind_3pp'] = round(float((dv <= -3).mean()), 3)
        fm_v = rr.groupby(['config', 'repeat', 'fold']).macro_f1.mean().unstack('config')[CONFIGS]
        for c in CONFIGS:
            row = {'variant': variant, 'config': c, 'test_n_mean': round(float(rr[rr.config == c].n.mean()), 1),
                   'macro_f1_mean': round(float(rr[rr.config == c].macro_f1.mean()), 2),
                   'macro_f1_sd': round(float(rr[rr.config == c].macro_f1.std(ddof=1)), 2)}
            if c != 'none':
                t = corrected_ttest(fm_v[c] - fm_v['none'], n_train, n_test)
                row.update({k2: round(float(t[k2]), 3) for k2 in ['mean', 'ci_low', 'ci_high', 'p_corrected']})
            sens_rows.append(row)
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(R / 'sensitivity_grouping.csv', index=False)
    out['grouping']['sensitivity'] = {f'{r.variant}|{r.config}': {k2: v for k2, v in r._asdict().items()
                                      if k2 not in ('Index', 'variant', 'config') and pd.notna(v)}
                                      for r in sens.itertuples()}

    with open(R / 'numbers.json', 'w') as f:
        json.dump(out, f, indent=1, default=float)

    # ================================================================ figures
    # Fig. 2: paired differences per fold
    fig, ax = plt.subplots(figsize=(6.85, 2.9))
    rng = np.random.default_rng(0)
    comps = ['lfa', 'se', 'cbam']
    lim = 0
    for yi, c in enumerate(comps):
        d = (pf[c] - pf['none']).values
        st = out['paired_macro_f1'][c]
        lim = max(lim, np.abs(d).max())
        ax.scatter(d, yi + rng.uniform(-0.13, 0.13, len(d)), s=20, color=GREY, edgecolor=SURF, lw=0.7, zorder=2)
        ax.plot([st['ci_low'], st['ci_high']], [yi, yi], color=BLUE, lw=2.4, zorder=3, solid_capstyle='round')
        ax.scatter([st['mean']], [yi], s=60, color=BLUE, edgecolor=SURF, lw=1.8, zorder=4)
        ax.text(1.02, yi, f"{st['mean']:+.2f} [{st['ci_low']:.1f}, {st['ci_high']:.1f}]", va='center',
                color=INK2, fontsize=8.5, transform=ax.get_yaxis_transform())
    ax.text(1.02, -0.62, 'Mean [95% CI]', color=INK2, fontsize=8.5, fontweight='bold', transform=ax.get_yaxis_transform())
    ax.axvline(0, color=INK2, lw=0.9, zorder=1)
    ax.set_yticks(range(3), [f'{LABEL[c]}' for c in comps])
    ax.set_xlabel('Macro F1 difference from B0 without attention (percentage points)')
    ax.set_xlim(-np.ceil(lim + 1), np.ceil(lim + 1))
    ax.set_ylim(2.5, -0.8)
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    fig.savefig(F / 'fig2_paired_differences.png', dpi=600)
    fig.savefig(F / 'fig2_paired_differences.eps')
    plt.close(fig)

    # Fig. 3: seed-only differences
    fig, ax = plt.subplots(figsize=(6.85, 3.0))
    eff = abs(out['paired_macro_f1']['lfa']['mean'])
    for yi, c in enumerate(CONFIGS):
        v = gap.loc[c].values
        ax.scatter(v, yi + rng.uniform(-0.13, 0.13, len(v)), s=20, color=GREY, edgecolor=SURF, lw=0.7, zorder=2)
        ax.plot([np.median(v)] * 2, [yi - 0.27, yi + 0.27], color=BLUE, lw=2.4, zorder=3)
        ax.text(1.02, yi, f'median {np.median(v):.1f}', va='center', color=INK2, fontsize=8.5,
                transform=ax.get_yaxis_transform())
    ax.axvline(eff, color=ORANGE, lw=1.3, ls='--', zorder=1)
    ax.text(eff + 0.3, 3.62, f'|mean LFA effect| = {eff:.2f}', color=ORANGE, fontsize=8.5, va='center')
    ax.set_yticks(range(4), [LABEL[c] for c in CONFIGS])
    ax.set_xlabel('|seed 0 − seed 1| in macro F1, same fold and configuration (percentage points)')
    ax.set_xlim(left=0)
    ax.set_ylim(3.85, -0.5)
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    fig.savefig(F / 'fig3_seed_differences.png', dpi=600)
    fig.savefig(F / 'fig3_seed_differences.eps')
    plt.close(fig)

    # Fig. 4: fold-to-fold spread
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = np.arange(4)
    for _, row in pf.iterrows():
        ax.plot(x, row.values, color='#c4c3bd', lw=0.9, zorder=1)
    ax.plot(x, pf.mean().values, color=BLUE, lw=2.4, marker='o', ms=7, markeredgecolor=SURF,
            markeredgewidth=1.8, zorder=3, label='Mean of 20 folds')
    ax.set_xticks(x, [LABEL[c] for c in CONFIGS])
    ax.tick_params(axis='x', labelsize=8)
    ax.set_ylabel('Macro F1 (%), mean of two seeds')
    ax.legend(frameon=False, loc='lower left', bbox_to_anchor=(0, 1.0), fontsize=8.5, ncol=1)
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    fig.savefig(F / 'fig4_fold_spread.png', dpi=600)
    fig.savefig(F / 'fig4_fold_spread.eps')
    plt.close(fig)

    # Fig. 5: confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(6.85, 3.3))
    names = ['Algal leaf spot', 'Leaf rot', 'Phomopsis', 'Root & collar rot']
    for ax, cfg in zip(axes, ['none', 'lfa']):
        cm = np.array(cms[cfg]) / 100
        ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
        ax.grid(False)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{cm[i, j] * 100:.0f}', ha='center', va='center', fontsize=8.5,
                        color='white' if cm[i, j] > 0.55 else INK)
        ax.set_xticks(range(4), names, rotation=30, ha='right', fontsize=8)
        ax.set_yticks(range(4), names if cfg == 'none' else [''] * 4, fontsize=8)
        ax.set_xlabel('Predicted')
        if cfg == 'none':
            ax.set_ylabel('True')
        ax.set_title(LABEL[cfg], loc='left')
    fig.tight_layout()
    fig.savefig(F / 'fig5_confusion.png', dpi=600)
    fig.savefig(F / 'fig5_confusion.eps')
    plt.close(fig)
    print(json.dumps(out['paired_macro_f1'], indent=1))
    print(f'Wrote {R / "numbers.json"} and figures to {F}/')


if __name__ == '__main__':
    main()
