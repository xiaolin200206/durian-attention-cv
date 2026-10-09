"""Every number reported in the paper, recomputed from the released per-image predictions.

    python paper/compute_numbers.py          # run from the repository root

Every quantity is computed under two scorings of the same runs:
  all   - every test image (the scoring of the pre-specified analyses)
  novid - the 63 Phomopsis video frames removed from scoring (primary estimate in this version,
          because the frames were not linked to the still photographs taken with them)
Writes paper/numbers.json. Needs no GPU, no torch and no images.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
CLASSES = ['Algal', 'Leaf_rot', 'Phomopsis', 'Root_disease']
CONFIGS = ['none', 'lfa', 'se', 'cbam']
V2 = ['none_224', 'lfa14_224', 'lfa28_224', 'none_320', 'lfa14_320']
J = 20


def corrected(d, ratio):
    d = np.asarray(d, float)
    s = d.std(ddof=1)
    se = np.sqrt(1 / J + ratio) * s
    tq = stats.t.ppf(0.975, J - 1)
    t = d.mean() / se
    # two one-sided tests against a +/-2 pp margin, same corrected variance
    p_tost = max(stats.t.sf((d.mean() + 2) / se, J - 1), stats.t.cdf((d.mean() - 2) / se, J - 1))
    post = stats.t(J - 1, loc=d.mean(), scale=se)  # correlated Bayesian t-test, flat prior
    return {'mean': d.mean(), 'ci_low': d.mean() - tq * se, 'ci_high': d.mean() + tq * se,
            'half_width': tq * se, 'p_corrected': 2 * stats.t.sf(abs(t), J - 1),
            'p_uncorrected': stats.ttest_1samp(d, 0).pvalue, 'p_wilcoxon': stats.wilcoxon(d).pvalue,
            'wins': int((d > 0).sum()), 'sd_d': s, 'p_tost_2pp': p_tost,
            'post_within_2pp': post.cdf(2) - post.cdf(-2),
            'min_half_width_inf_repeats': 1.96 * np.sqrt(ratio) * s}


def run_metrics(p):
    rows = []
    for (cfg, rep, k, s), g in p.groupby(['config', 'repeat', 'fold', 'seed']):
        present = sorted(set(g.y_true))
        rows.append({'config': cfg, 'repeat': rep, 'fold': k, 'seed': s, 'n': len(g),
                     'macro_f1': f1_score(g.y_true, g.y_pred, labels=present, average='macro', zero_division=0) * 100,
                     'acc': accuracy_score(g.y_true, g.y_pred) * 100})
    return pd.DataFrame(rows)


def block(p, configs, ref, ratio):
    r = run_metrics(p)
    out = {'summary': {}, 'paired': {}, 'paired_acc': {}}
    for c in configs:
        x = r[r.config == c]
        out['summary'][c] = {'mean': x.macro_f1.mean(), 'sd': x.macro_f1.std(ddof=1),
                             'acc_mean': x.acc.mean(), 'acc_sd': x.acc.std(ddof=1), 'n_test_mean': x.n.mean()}
    pf = r.groupby(['config', 'repeat', 'fold']).macro_f1.mean().unstack('config')
    pa = r.groupby(['config', 'repeat', 'fold']).acc.mean().unstack('config')
    for c in configs:
        if c != ref:
            out['paired'][c] = corrected(pf[c] - pf[ref], ratio)
            out['paired_acc'][c] = corrected(pa[c] - pa[ref], ratio)
    sg = r.pivot_table(index=['config', 'repeat', 'fold'], columns='seed', values='macro_f1')
    diff = sg[0] - sg[1]
    out['seed'] = {'mean_abs': diff.abs().mean(), 'median_abs': diff.abs().median(), 'max_abs': diff.abs().max(),
                   'sd_single_run': float(np.sqrt((diff ** 2).mean() / 2)),
                   'by_config_median_abs': diff.abs().groupby(level='config').median().to_dict()}
    base = r[r.config == ref].set_index(['repeat', 'fold', 'seed']).macro_f1
    out['single_run'] = {}
    for c in configs:
        if c != ref:
            dd = r[r.config == c].set_index(['repeat', 'fold', 'seed']).macro_f1 - base
            out['single_run'][c] = {'ahead3': float((dd >= 3).mean()), 'behind3': float((dd <= -3).mean()),
                                    'min': dd.min(), 'max': dd.max(), 'pairs': int(len(dd))}
    fm = pf[configs].mean(axis=1)
    out['fold'] = {'min': fm.min(), 'max': fm.max(), 'sd': fm.std(ddof=1),
                   'sd_config_means': pf[configs].mean().std(ddof=1)}
    return out, r


def per_class(p, configs):
    out, cms = {}, {}
    for c in configs:
        rows = []
        for _, g in p[p.config == c].groupby(['repeat', 'fold', 'seed']):
            pr, rc, f1, sup = precision_recall_fscore_support(g.y_true, g.y_pred, labels=range(4), zero_division=0)
            for i in range(4):
                if sup[i] > 0:
                    rows.append({'cls': CLASSES[i], 'p': pr[i] * 100, 'r': rc[i] * 100, 'f1': f1[i] * 100})
        df = pd.DataFrame(rows)
        out[c] = {k: {m: [g[m].mean(), g[m].std(ddof=1)] for m in ['p', 'r', 'f1']} | {'runs': len(g)}
                  for k, g in df.groupby('cls')}
        cm = confusion_matrix(p[p.config == c].y_true, p[p.config == c].y_pred, labels=range(4)).astype(float)
        cms[c] = (cm / cm.sum(1, keepdims=True) * 100).tolist()
    return out, cms


def main():
    sess = pd.read_csv(ROOT / 'data' / 'sessions.csv')
    sess['video'] = sess.session.astype(str).str.startswith('video:')
    res = pd.read_csv(ROOT / 'results' / 'results.csv')
    sizes = res.groupby(['repeat', 'fold'])[['n_train', 'n_val', 'n_test']].first()
    ratio = sizes.n_test.mean() / (sizes.n_train.mean() + sizes.n_val.mean())
    ratio_train_only = sizes.n_test.mean() / sizes.n_train.mean()
    fold_ratio = sizes.n_test / (sizes.n_train + sizes.n_val)

    N = {'design': {'ratio': ratio, 'factor': 1 / J + ratio, 'factor_train_only': 1 / J + ratio_train_only,
                    'fold_ratio_min': fold_ratio.min(), 'fold_ratio_max': fold_ratio.max()}}

    for name, path, configs, ref in [('main', 'results/preds.csv', CONFIGS, 'none'),
                                     ('v2', 'results/v2/preds.csv', V2, 'none_224')]:
        p = pd.read_csv(ROOT / path)
        if 'config' not in p.columns:
            raise SystemExit(f'{path}: no config column')
        p = p.merge(sess[['cls', 'file', 'video']], on=['cls', 'file'], how='left')
        assert p.video.notna().all()
        for scoring, q in [('all', p), ('novid', p[~p.video])]:
            b, r = block(q, configs, ref, ratio)
            N[f'{name}_{scoring}'] = b
            if name == 'main':
                pc, cm = per_class(q, configs)
                N[f'main_{scoring}']['per_class'] = pc
                N[f'main_{scoring}']['confusion_row_pct'] = cm
            if name == 'v2':
                ps = [b['paired'][c]['p_corrected'] for c in V2[1:]]
                order = np.argsort(ps)
                holm, run = [0] * 4, 0
                for rank, i in enumerate(order):
                    run = max(run, min(1, (4 - rank) * ps[i]))
                    holm[i] = run
                for c, h in zip(V2[1:], holm):
                    b['paired'][c]['p_holm'] = h
                # secondary: lfa14_320 vs none_320
                pf = r.groupby(['config', 'repeat', 'fold']).macro_f1.mean().unstack('config')
                b['secondary_lfa320_vs_none320'] = corrected(pf['lfa14_320'] - pf['none_320'], ratio)
        if name == 'main':
            g0 = p[(p.config == 'none') & (p.cls == 'Phomopsis')]
            N['phomopsis_recall'] = {'video': (g0[g0.video].y_true == g0[g0.video].y_pred).mean() * 100,
                                     'stills': (g0[~g0.video].y_true == g0[~g0.video].y_pred).mean() * 100}
            N['test_size_novid'] = {'min': int(p[~p.video & (p.config == 'none') & (p.seed == 0)]
                                               .groupby(['repeat', 'fold']).size().min()),
                                    'max': int(p[~p.video & (p.config == 'none') & (p.seed == 0)]
                                               .groupby(['repeat', 'fold']).size().max())}

    ext = pd.read_csv(ROOT / 'results' / 'external_vietnam.csv')
    # four-class output scored over the three shared classes, as in the paper
    piv = ext.pivot_table(index='seed', columns='config', values='vn_macro_f1_full')
    N['external'] = {c: {'mean': piv[c].mean(), 'sd': piv[c].std(ddof=1)} for c in CONFIGS}
    for c in CONFIGS[1:]:
        N['external'][c].update({'diff': (piv[c] - piv['none']).mean(),
                                 'p': stats.ttest_rel(piv[c], piv['none']).pvalue,
                                 'wins': int((piv[c] > piv['none']).sum())})

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return round(float(o), 8)
        if isinstance(o, np.integer):
            return int(o)
        return o

    json.dump(clean(N), open(OUT / 'numbers.json', 'w'), indent=1)
    print('wrote', OUT / 'numbers.json')


if __name__ == '__main__':
    main()
