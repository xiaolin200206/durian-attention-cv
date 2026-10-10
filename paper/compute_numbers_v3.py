"""Numbers for the backbone and decision-support experiment (results/v3), recomputed from the per-image predictions.

    python paper/compute_numbers_v3.py          # run from the repository root

Writes paper/numbers_v3.json. Needs no GPU, no torch and no images. Scorings as in compute_numbers.py:
  novid - the 63 Phomopsis video frames removed (primary)
  all   - every test image
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_numbers import CLASSES, J, block, corrected, per_class, run_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
CFG = ['effb0', 'mnv3l', 'mnv4m', 'cnxv2t', 'vits']
REF, PRIMARY = 'effb0', ['mnv3l', 'mnv4m', 'cnxv2t', 'vits']
PCOLS = ['p_Algal', 'p_Leaf_rot', 'p_Phomopsis', 'p_Root_disease']
TARGETS = [0.90, 0.95]
GRID = np.arange(10, 101, 10)  # coverage grid (%) for matched-coverage accuracy


def holm(ps):
    order = np.argsort(ps)
    out, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = run
    return out


def pick_threshold(conf, correct, target):
    for t in np.sort(np.unique(conf)):
        a = correct[conf >= t]
        if len(a) and a.mean() >= target:
            return t
    return np.inf


def per_run_decision(val, test, target):
    rows = []
    for key, gt in test.groupby(['config', 'repeat', 'fold', 'seed']):
        gv = val[(val.config == key[0]) & (val.repeat == key[1]) & (val.fold == key[2]) & (val.seed == key[3])]
        t = pick_threshold(gv[PCOLS].max(axis=1).values, (gv.y_true == gv.y_pred).values, target)
        conf, corr = gt[PCOLS].max(axis=1).values, (gt.y_true == gt.y_pred).values
        acc = conf >= t
        r = {'config': key[0], 'repeat': key[1], 'fold': key[2], 'seed': key[3], 'n': len(gt),
             'n_acc': int(acc.sum()), 'n_acc_correct': int(corr[acc].sum()), 'n_correct': int(corr.sum()),
             'coverage': acc.mean() * 100, 'sel_acc': corr[acc].mean() * 100 if acc.any() else np.nan,
             'all_deferred': bool(not acc.any()), 'all_accepted': bool(acc.all()), 'val_n': len(gv),
             'val_acc': (gv.y_true == gv.y_pred).mean() * 100, 'val_meets': bool((gv.y_true == gv.y_pred).mean() >= target)}
        for j, c in enumerate(CLASSES):
            m = gt.y_true.values == j
            r[f'n_{c}'] = int(m.sum())
            r[f'def_{c}'] = int((~acc[m]).sum())
        rows.append(r)
    return pd.DataFrame(rows)


def curves(test):
    rows = []
    for key, gt in test.groupby(['config', 'repeat', 'fold', 'seed']):
        pr = gt[PCOLS].values
        conf, corr = pr.max(axis=1), (gt.y_true == gt.y_pred).values
        o = np.argsort(-conf, kind='stable')
        risk = np.cumsum(~corr[o]) / np.arange(1, len(o) + 1)
        top2 = np.argsort(-pr, 1)[:, :2]
        r = {'config': key[0], 'repeat': key[1], 'fold': key[2], 'seed': key[3],
             'aurc': risk.mean() * 100, 'top2': (top2 == gt.y_true.values[:, None]).any(1).mean() * 100,
             'acc': corr.mean() * 100}
        n = len(o)
        for g in GRID:
            k = max(1, int(round(n * g / 100)))
            r[f'acc_at_{g}'] = (1 - risk[k - 1]) * 100
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    sess = pd.read_csv(ROOT / 'data' / 'sessions.csv')
    sess['video'] = sess.session.astype(str).str.startswith('video:')
    res = pd.read_csv(ROOT / 'results' / 'v3' / 'results.csv')
    assert res.groupby('config').size().eq(40).all()
    sizes = res.groupby(['repeat', 'fold'])[['n_train', 'n_val', 'n_test']].first()
    ratio = sizes.n_test.mean() / (sizes.n_train.mean() + sizes.n_val.mean())
    N = {'ratio': ratio, 'factor': 1 / J + ratio}

    p = pd.read_csv(ROOT / 'results' / 'v3' / 'preds.csv')
    p = p.merge(sess[['cls', 'file', 'video']], on=['cls', 'file'], how='left')
    assert p.video.notna().all() and len(p) == 35710
    test, val = p[p.role == 'test'], p[p.role == 'val']
    assert test.groupby(['config', 'repeat', 'fold', 'seed']).size().groupby('config').count().eq(40).all()

    # consistency: macro F1 recomputed from predictions equals the value logged during training
    rm = run_metrics(test).merge(res, on=['config', 'repeat', 'fold', 'seed'])
    N['consistency_max_abs_diff'] = float((rm.macro_f1 - rm.test_macro_f1_present).abs().max())
    N['consistency_acc_max_abs_diff'] = float((rm.acc - rm.test_acc).abs().max())

    for scoring, q, qv in [('all', test, val), ('novid', test[~test.video], val[~val.video])]:
        b, r = block(q, CFG, REF, ratio)
        ps = [b['paired'][c]['p_corrected'] for c in PRIMARY]
        for c, h in zip(PRIMARY, holm(ps)):
            b['paired'][c]['p_holm'] = h
            b['paired'][c]['improvement_rule'] = bool(b['paired'][c]['mean'] >= 2 and h < 0.05)
        pf = r.groupby(['config', 'repeat', 'fold']).macro_f1.mean().unstack('config')
        b['cnxv2t_vs_vits'] = corrected(pf['cnxv2t'] - pf['vits'], ratio)
        b['seed_abs_by_config'] = {c: float(v) for c, v in
                                   r.pivot_table(index=['config', 'repeat', 'fold'], columns='seed', values='macro_f1')
                                   .pipe(lambda s: (s[0] - s[1]).abs().groupby(level='config').mean()).items()}
        pc, cm = per_class(q, CFG)
        b['per_class'], b['confusion_row_pct'] = pc, cm
        # fold-level macro F1 (mean of two seeds) for the figure
        b['fold_macro_f1'] = {c: pf[c].tolist() for c in CFG}
        # Phomopsis recall: still photographs vs video frames, all runs
        if scoring == 'all':
            g = test[test.cls == 'Phomopsis']
            b['phomopsis_recall'] = {c: {'video': float((g[(g.config == c) & g.video].y_true ==
                                                      g[(g.config == c) & g.video].y_pred).mean() * 100),
                                         'stills': float((g[(g.config == c) & ~g.video].y_true ==
                                                          g[(g.config == c) & ~g.video].y_pred).mean() * 100)}
                                     for c in CFG}
        # decision support
        cv = curves(q)
        b['aurc_top2'] = {c: {'aurc': [cv[cv.config == c].aurc.mean(), cv[cv.config == c].aurc.std(ddof=1)],
                              'top2': [cv[cv.config == c].top2.mean(), cv[cv.config == c].top2.std(ddof=1)],
                              'acc': cv[cv.config == c].acc.mean(),
                              **{f'acc_at_{g}': cv[cv.config == c][f'acc_at_{g}'].mean() for g in GRID}}
                          for c in CFG}
        pa = cv.groupby(['config', 'repeat', 'fold']).aurc.mean().unstack('config')
        pt = cv.groupby(['config', 'repeat', 'fold']).top2.mean().unstack('config')
        b['aurc_paired'] = {c: corrected(pa[c] - pa[REF], ratio) for c in PRIMARY}   # exploratory
        b['top2_paired'] = {c: corrected(pt[c] - pt[REF], ratio) for c in PRIMARY}   # exploratory
        for key in ('aurc_paired', 'top2_paired'):
            for c, h in zip(PRIMARY, holm([b[key][c]['p_corrected'] for c in PRIMARY])):
                b[key][c]['p_holm'] = h
        sel = {}
        for tgt in TARGETS:
            d = per_run_decision(qv, q, tgt)
            s = {}
            for c in CFG:
                x = d[d.config == c]
                s[c] = {'coverage': [x.coverage.mean(), x.coverage.std(ddof=1)],
                        'sel_acc_mean_of_runs': [x.sel_acc.mean(), x.sel_acc.std(ddof=1)],
                        'sel_acc_pooled': x.n_acc_correct.sum() / x.n_acc.sum() * 100,
                        'acc_all_pooled': x.n_correct.sum() / x.n.sum() * 100,
                        'runs_all_deferred': int(x.all_deferred.sum()), 'runs_all_accepted': int(x.all_accepted.sum()),
                        'runs_below_target': int((x.sel_acc < tgt * 100).sum()),
                        'runs_with_accepted': int(x.sel_acc.notna().sum()),
                        'val_n_mean': float(x.val_n.mean()),
                        'val_acc_mean': float(x.val_acc.mean()), 'test_acc_mean': float(x.n_correct.div(x.n).mul(100).mean()),
                        'runs_val_meets_target': int(x.val_meets.sum()),
                        'defer_pct': {k: float(x[f'def_{k}'].sum() / x[f'n_{k}'].sum() * 100) for k in CLASSES},
                        'defer_pct_run_mean': {k: float((x[f'def_{k}'] / x[f'n_{k}'])[x[f'n_{k}'] > 0].mean() * 100)
                                               for k in CLASSES}}
            sel[f'{int(tgt * 100)}'] = s
        b['selective'] = sel
        N[scoring] = b

    # sensitivity: score only test images whose video-linked group was not split (needs common.py, hence torch)
    try:
        sys.path.insert(0, str(ROOT))
        import common as Cm
        s2 = sess[sess.cls.isin(CLASSES)].reset_index(drop=True)
        s2['group_linked'] = Cm.build_groups(s2, link_videos=True).values
        size_linked = s2.group_linked.value_counts()
        pv = test.merge(s2[['cls', 'file', 'group_linked']], on=['cls', 'file'], how='left')
        assert pv.group_linked.notna().all()
        rows = []
        for (cfg, rep_, k, sd), g in pv.groupby(['config', 'repeat', 'fold', 'seed']):
            cnt = g.group_linked.value_counts()
            split = cnt[cnt < size_linked[cnt.index].values].index
            g = g[~g.group_linked.isin(split)]
            rows.append({'config': cfg, 'repeat': rep_, 'fold': k, 'seed': sd, 'n': len(g),
                         'macro_f1': f1_score(g.y_true, g.y_pred, labels=sorted(set(g.y_true)), average='macro', zero_division=0) * 100})
        rr = pd.DataFrame(rows)
        pfu = rr.groupby(['config', 'repeat', 'fold']).macro_f1.mean().unstack('config')
        un = {'test_n_mean': float(rr[rr.config == REF].n.mean()),
              'summary': {c: {'mean': float(rr[rr.config == c].macro_f1.mean()), 'sd': float(rr[rr.config == c].macro_f1.std(ddof=1))} for c in CFG},
              'paired': {c: corrected(pfu[c] - pfu[REF], ratio) for c in PRIMARY}}
        for c, h in zip(PRIMARY, holm([un['paired'][c]['p_corrected'] for c in PRIMARY])):
            un['paired'][c]['p_holm'] = h
        N['unsplit'] = un
    except ImportError:
        print('torch not available: sensitivity to video-linked groups skipped')

    cost = pd.read_csv(ROOT / 'results' / 'v3' / 'cost.csv').set_index('config')
    N['cost'] = {c: cost.loc[c].to_dict() for c in CFG}

    # rerun agreement: effb0 here vs 'none' in the first experiment (same folds, seeds, schedule except drop_last)
    pm = pd.read_csv(ROOT / 'results' / 'preds.csv').merge(sess[['cls', 'file', 'video']], on=['cls', 'file'])
    agree = {}
    for scoring, q1, q0 in [('all', test, pm), ('novid', test[~test.video], pm[~pm.video])]:
        a = run_metrics(q1[q1.config == REF]).set_index(['repeat', 'fold', 'seed']).macro_f1
        o = run_metrics(q0[q0.config == 'none']).set_index(['repeat', 'fold', 'seed']).macro_f1
        d = (a - o)
        agree[scoring] = {'mean_diff': d.mean(), 'mean_abs_diff': d.abs().mean(), 'sd_diff': d.std(ddof=1),
                          'here': a.mean(), 'first': o.mean(), 'corr': float(np.corrcoef(a, o)[0, 1])}
    N['rerun_agreement'] = agree
    N['minutes_mean'] = res.groupby('config').minutes.mean().to_dict()

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if np.isnan(o) else round(float(o), 8)
        if isinstance(o, np.integer):
            return int(o)
        return o

    json.dump(clean(N), open(OUT / 'numbers_v3.json', 'w'), indent=1)
    print('wrote', OUT / 'numbers_v3.json')


if __name__ == '__main__':
    main()
