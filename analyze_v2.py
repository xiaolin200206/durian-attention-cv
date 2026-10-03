"""Table 6 of the paper (insertion point and input resolution) from results/v2/results.csv.

    python analyze_v2.py --results results/v2

Writes results/v2/table6.md and results/v2/numbers_v2.json. Paired comparisons use the corrected
resampled t-test with the same correction factor as analyze.py; the four primary comparisons
(each configuration against none_224) are Holm-adjusted, as pre-specified in results/v2/PREREG.md.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analyze import corrected_ttest

CONFIGS_V2 = ['none_224', 'lfa14_224', 'lfa28_224', 'none_320', 'lfa14_320']
PRIMARY = ['lfa14_224', 'lfa28_224', 'none_320', 'lfa14_320']
LABEL = {'none_224': 'None, 224 px', 'lfa14_224': 'LFA after block 5 (14 × 14), 224 px',
         'lfa28_224': 'LFA after block 3 (28 × 28), 224 px', 'none_320': 'None, 320 px',
         'lfa14_320': 'LFA after block 5 (20 × 20), 320 px'}


def holm(p):
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results/v2')
    a = ap.parse_args()
    R = Path(a.results)
    r = pd.read_csv(R / 'results.csv')
    sz = r.groupby(['repeat', 'fold'])[['n_train', 'n_val', 'n_test']].first()
    n_train, n_test = float((sz.n_train + sz.n_val).mean()), float(sz.n_test.mean())
    pf = r.groupby(['config', 'repeat', 'fold']).test_macro_f1_present.mean().unstack('config')[CONFIGS_V2]
    out = {'runs': int(len(r)), 'factor': round(1 / len(sz) + n_test / n_train, 3), 'summary': {}, 'paired': {}}
    for c in CONFIGS_V2:
        v = r[r.config == c].test_macro_f1_present
        out['summary'][c] = {'mean': round(float(v.mean()), 2), 'sd': round(float(v.std(ddof=1)), 2), 'n': int(len(v))}
    tests = {c: corrected_ttest(pf[c] - pf['none_224'], n_train, n_test) for c in PRIMARY}
    adj = holm(np.array([tests[c]['p_corrected'] for c in PRIMARY]))
    for c, pa in zip(PRIMARY, adj):
        t = tests[c]
        out['paired'][c] = {k: round(float(t[k]), 3) for k in ['mean', 'ci_low', 'ci_high', 'p_corrected', 'wins', 'losses']}
        out['paired'][c]['p_holm'] = round(float(pa), 3)
        out['paired'][c]['improvement'] = bool(t['mean'] >= 2 and pa < 0.05)
    t = corrected_ttest(pf['lfa14_320'] - pf['none_320'], n_train, n_test)
    out['secondary_lfa14_320_vs_none_320'] = {k: round(float(t[k]), 3) for k in ['mean', 'ci_low', 'ci_high', 'p_corrected', 'wins']}
    sg = r.pivot_table(index=['config', 'repeat', 'fold'], columns='seed', values='test_macro_f1_present')
    out['seed_gap_mean'] = round(float((sg[sg.columns[0]] - sg[sg.columns[1]]).abs().mean()), 2)
    old = Path('results/run_metrics.csv')   # present-class macro F1 of the main experiment (from analyze.py)
    if old.exists():
        o = pd.read_csv(old)
        out['none_main_experiment_mean'] = round(float(o[o.config == 'none'].macro_f1.mean()), 2)
    with open(R / 'numbers_v2.json', 'w') as f:
        json.dump(out, f, indent=1)

    lines = ['| Configuration | Macro F1 (%) | Difference from none, 224 px (pp) [95% CI] | *p*, corrected | *p*, Holm | Folds higher |',
             '|---|---|---|---|---|---|']
    for c in CONFIGS_V2:
        s = out['summary'][c]
        if c == 'none_224':
            lines.append(f"| {LABEL[c]} | {s['mean']:.2f} ± {s['sd']:.2f} | – | – | – | – |")
        else:
            p = out['paired'][c]
            lines.append(f"| {LABEL[c]} | {s['mean']:.2f} ± {s['sd']:.2f} | {p['mean']:+.2f} [{p['ci_low']:.2f}, {p['ci_high']:.2f}] "
                         f"| {p['p_corrected']:.3f} | {p['p_holm']:.3f} | {int(p['wins'])}/20 |")
    (R / 'table6.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(json.dumps({k: v for k, v in out.items() if k != 'summary'}, indent=1))


if __name__ == '__main__':
    main()
