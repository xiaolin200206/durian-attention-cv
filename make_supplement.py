"""Build the supplementary tables (Markdown) from the released result files.

    python analyze.py && python make_supplement.py > supplementary_tables.md
"""
import json
import re

import numpy as np
import pandas as pd
from scipy import stats

R = 'results/'
d = json.load(open(R + 'numbers.json'))
C = ['none', 'lfa', 'se', 'cbam']
L = {'none': 'None', 'lfa': 'LFA', 'se': 'SE', 'cbam': 'CBAM'}
CL = {'Algal': 'Algal leaf spot', 'Leaf_rot': 'Leaf rot', 'Phomopsis': '*Phomopsis* leaf spot',
      'Root_disease': 'Root and collar rot'}
out = ['## Table S1. Per-class precision, recall and F1 (mean ± SD over runs, %)\n',
       '| Class | Configuration | Precision | Recall | F1 |', '|---|---|---|---|---|']
for cl in CL:
    for i, c in enumerate(C):
        v = d['per_class'][f'{c}|{cl}']
        out.append(f"| {CL[cl] if i == 0 else ''} | {L[c]} | {v['precision']} | {v['recall']} | {v['f1']} |")

runs = pd.read_csv(R + 'run_metrics.csv')
res = pd.read_csv(R + 'results.csv')
pf = runs.groupby(['repeat', 'fold', 'config']).macro_f1.mean().unstack('config')[C]
sz = res.groupby(['repeat', 'fold'])[['n_train', 'n_val', 'n_test']].first()
cls = runs.groupby(['repeat', 'fold']).classes_in_test.first()
out += ['\n## Table S2. Fold sizes and macro F1 per fold (mean of two seeds; † no root and collar rot in test)\n',
        '| Repeat | Fold | Train | Validation | Test | None | LFA | SE | CBAM |', '|---|---|---|---|---|---|---|---|---|']
for (rep, k), row in pf.iterrows():
    s = sz.loc[(rep, k)]
    out.append(f"| {rep} | {k}{'†' if cls[(rep, k)] < 4 else ''} | {s.n_train} | {s.n_val} | {s.n_test} | "
               + ' | '.join(f'{row[c]:.2f}' for c in C) + ' |')

a = pd.read_csv(R + 'ablation_replicates_preliminary.csv')
# The earlier code did not control training randomness with the seed, so runs are unpaired replicates.
y = a[a.attention == 'none'].macro_f1
out += ['\n## Table S3. Preliminary experiment on one fixed five-class split (unpaired)\n',
        '| Configuration | Macro F1, mean ± SD | Difference from none (pp) | p, Welch t | p, Mann-Whitney |',
        '|---|---|---|---|---|']
for c in C:
    x = a[a.attention == c].macro_f1
    if c == 'none':
        out.append(f'| {L[c]} | {x.mean():.2f} ± {x.std(ddof=1):.2f} | – | – | – |')
        continue
    out.append(f"| {L[c]} | {x.mean():.2f} ± {x.std(ddof=1):.2f} | {x.mean() - y.mean():+.2f} | "
               f"{stats.ttest_ind(x, y, equal_var=False).pvalue:.3f} | {stats.mannwhitneyu(x, y).pvalue:.3f} |")

pf0 = res.groupby(['repeat', 'fold', 'config']).test_macro_f1.mean().unstack('config')[C]
ntr = (sz.n_train + sz.n_val).mean()
nte = sz.n_test.mean()
out += ['\n## Table S4. Absent class scored as zero (macro F1 over all four classes)\n',
        '| Configuration | Macro F1 (%), mean ± SD | Paired difference (pp) [95% CI] | p, corrected |', '|---|---|---|---|']
for c in C:
    v = res[res.config == c].test_macro_f1
    if c == 'none':
        out.append(f'| {L[c]} | {v.mean():.2f} ± {v.std(ddof=1):.2f} | – | – |')
        continue
    dd = (pf0[c] - pf0['none']).values
    J = len(dd)
    se = np.sqrt((1 / J + nte / ntr) * dd.var(ddof=1))
    h = stats.t.ppf(0.975, J - 1) * se
    p = 2 * stats.t.sf(abs(dd.mean() / se), J - 1)
    out.append(f'| {L[c]} | {v.mean():.2f} ± {v.std(ddof=1):.2f} | {dd.mean():+.2f} [{dd.mean() - h:.2f}, {dd.mean() + h:.2f}] | {p:.3f} |')

sg = pd.read_csv(R + 'sensitivity_grouping.csv')
VN = {'all': 'All test images (as reported)', 'no_video': 'Video frames excluded',
      'unsplit_linked': 'Images in split linked groups excluded'}
out += ['\n## Table S7. Sensitivity to the grouping of video frames\n',
        '| Test images scored | Mean test images per fold | Configuration | Macro F1 (%), mean ± SD | Paired difference (pp) [95% CI] | p, corrected |',
        '|---|---|---|---|---|---|']
for v in VN:
    for i, c in enumerate(C):
        r = sg[(sg.variant == v) & (sg.config == c)].iloc[0]
        lead = f'| {VN[v]} | {r.test_n_mean:.1f} ' if i == 0 else '| | '
        diff = '– | –' if c == 'none' else f'{r["mean"]:+.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}] | {r.p_corrected:.3f}'
        out.append(f'{lead}| {L[c]} | {r.macro_f1_mean:.2f} ± {r.macro_f1_sd:.2f} | {diff} |')

print(re.sub(r'(?<=[\s\[(,|])-(?=\d)', '−', '\n'.join(out)))
