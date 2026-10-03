"""Verify the Pattern Recognition Letters manuscript against the released result files.

    python paper/prl/verify_prl.py          # run from the repository root; exit code 1 on any failure

A. Independent recomputation. Every headline quantity is recomputed here from results/preds.csv,
   results/results.csv, results/v2/results.csv, results/external_vietnam.csv, results/latency_params.csv
   and data/sessions.csv, without importing analyze.py, and compared with results/numbers.json and
   results/v2/numbers_v2.json (the files the manuscript is rendered from).
B. Render match. prl_template.tex is re-rendered and must equal the committed manuscript_prl.tex;
   the supplement is rebuilt and must equal the committed supplementary_prl.md.
C. Hard-coded facts. Numbers typed into the template prose (dataset counts, group sizes, ranges such as
   "9--14 points wide") are checked against the data.
D. Journal rules. PDF <= 7 pages, abstract <= 200 words (PRL authorship & formats template), 1-7 keywords, highlights 3-5 x <= 85 characters,
   every \\cite key in refs.bib, no unresolved placeholders, no '??' in the PDF, every supplementary
   table/figure cited in the text exists in the supplement.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import torch_stub  # noqa: E402,F401
import common as Cm  # noqa: E402

CONFIGS = ['none', 'lfa', 'se', 'cbam']
CLASSES = ['Algal', 'Leaf_rot', 'Phomopsis', 'Root_disease']
FAILS, PASSES = [], []


def check(name, ok, detail=''):
    (PASSES if ok else FAILS).append(name)
    print(('PASS ' if ok else 'FAIL ') + name + (f'  [{detail}]' if detail and not ok else ''))


def close(a, b, tol=0.0051):
    return abs(float(a) - float(b)) <= tol


def corrected(d, ntr, nte):
    d = np.asarray(d, float)
    J = len(d)
    se = np.sqrt((1 / J + nte / ntr) * d.var(ddof=1))
    h = stats.t.ppf(0.975, J - 1) * se
    return d.mean(), d.mean() - h, d.mean() + h, 2 * stats.t.sf(abs(d.mean() / se), J - 1)


# ============================================================ A. independent recomputation
def section_a():
    N = json.load(open(ROOT / 'results' / 'numbers.json'))
    V = json.load(open(ROOT / 'results' / 'v2' / 'numbers_v2.json'))
    preds = pd.read_csv(ROOT / 'results' / 'preds.csv')
    res = pd.read_csv(ROOT / 'results' / 'results.csv')
    check('A: 160 runs x (preds rows per run = n_test)',
          len(res) == 160 and preds.groupby(['config', 'repeat', 'fold', 'seed']).size().eq(
              res.set_index(['config', 'repeat', 'fold', 'seed']).n_test).all())
    rows = []
    for (cfg, rep, k, s), g in preds.groupby(['config', 'repeat', 'fold', 'seed']):
        present = sorted(set(g.y_true))
        rows.append({'config': cfg, 'repeat': rep, 'fold': k, 'seed': s,
                     'f1': 100 * f1_score(g.y_true, g.y_pred, labels=present, average='macro', zero_division=0),
                     'acc': 100 * accuracy_score(g.y_true, g.y_pred)})
    runs = pd.DataFrame(rows)
    sz = res.groupby(['repeat', 'fold'])[['n_train', 'n_val', 'n_test']].first()
    ntr, nte = (sz.n_train + sz.n_val).mean(), sz.n_test.mean()
    d = N['design']
    check('A: fold sizes', d['train_min'] == sz.n_train.min() and d['train_max'] == sz.n_train.max()
          and close(d['train_mean'], sz.n_train.mean(), 0.051) and d['test_min'] == sz.n_test.min()
          and d['test_max'] == sz.n_test.max() and close(d['test_mean'], nte, 0.051)
          and d['val_min'] == sz.n_val.min() and d['val_max'] == sz.n_val.max())
    check('A: correction factor 0.383', close(1 / 20 + nte / ntr, 0.383, 0.0006))
    for c in CONFIGS:
        v = runs[runs.config == c]
        check(f'A: summary {c}', close(N['summary'][c]['macro_f1_mean'], v.f1.mean()) and close(N['summary'][c]['macro_f1_sd'], v.f1.std(ddof=1))
              and close(N['summary'][c]['acc_mean'], v.acc.mean()) and close(N['summary'][c]['acc_sd'], v.acc.std(ddof=1)))
    pf = runs.groupby(['config', 'repeat', 'fold']).f1.mean().unstack('config')
    pa = runs.groupby(['config', 'repeat', 'fold']).acc.mean().unstack('config')
    for c in ['lfa', 'se', 'cbam']:
        m, lo, hi, p = corrected(pf[c] - pf['none'], ntr, nte)
        P = N['paired_macro_f1'][c]
        check(f'A: paired macro F1 {c}', close(P['mean'], m, 0.0006) and close(P['ci_low'], lo, 0.0006) and close(P['ci_high'], hi, 0.0006)
              and close(P['p_corrected'], p, 0.0006) and P['wins'] == (pf[c] > pf['none']).sum(),
              f'{m:.3f} [{lo:.3f},{hi:.3f}] p={p:.3f}')
        check(f'A: uncorrected / Wilcoxon {c}', close(P['p_uncorrected'], stats.ttest_1samp(pf[c] - pf['none'], 0).pvalue, 0.0006)
              and close(P['p_wilcoxon'], stats.wilcoxon(pf[c] - pf['none']).pvalue, 0.0006))
        m, lo, hi, p = corrected(pa[c] - pa['none'], ntr, nte)
        check(f'A: paired accuracy {c}', close(N['paired_accuracy'][c]['mean'], m, 0.0006))
    fm = pf[CONFIGS].mean(axis=1)
    check('A: fold spread', close(N['fold_spread']['fold_mean_min'], fm.min(), 0.051) and close(N['fold_spread']['fold_mean_max'], fm.max(), 0.051)
          and close(N['fold_spread']['sd_of_fold_means'], fm.std(ddof=1)) and close(N['fold_spread']['sd_of_config_means'], pf[CONFIGS].mean().std(ddof=1)))
    sg = runs.pivot_table(index=['config', 'repeat', 'fold'], columns='seed', values='f1')
    gap = (sg[0] - sg[1]).abs()
    check('A: seed gap', close(N['seed_gap']['mean'], gap.mean()) and close(N['seed_gap']['median'], gap.median()) and close(N['seed_gap']['max'], gap.max())
          and all(close(N['seed_gap']['by_config_median'][c], gap.loc[c].median()) for c in CONFIGS))
    base = runs[runs.config == 'none'].set_index(['repeat', 'fold', 'seed']).f1
    for c in ['lfa', 'se', 'cbam']:
        dd = runs[runs.config == c].set_index(['repeat', 'fold', 'seed']).f1 - base
        S = N['single_run'][c]
        check(f'A: single-run shares {c}', close(S['share_ahead_3pp'], (dd >= 3).mean(), 1e-6) and close(S['share_behind_3pp'], (dd <= -3).mean(), 1e-6)
              and close(S['min'], dd.min()) and close(S['max'], dd.max()) and S['pairs'] == 40)
    # per class
    pc = []
    for (cfg, rep, k, s), g in preds.groupby(['config', 'repeat', 'fold', 'seed']):
        p, r, f, n = precision_recall_fscore_support(g.y_true, g.y_pred, labels=list(range(4)), zero_division=0)
        for i, cl in enumerate(CLASSES):
            if n[i]:
                pc.append({'config': cfg, 'class': cl, 'precision': 100 * p[i], 'recall': 100 * r[i], 'f1': 100 * f[i]})
    pc = pd.DataFrame(pc)
    ok = True
    for c in CONFIGS:
        for cl in CLASSES:
            v = pc[(pc.config == c) & (pc['class'] == cl)]
            for m in ['precision', 'recall', 'f1']:
                mean, sd = N['per_class'][f'{c}|{cl}'][m].split(' ± ')
                ok &= close(mean, v[m].mean(), 0.051) and close(sd, v[m].std(ddof=1), 0.051)
    check('A: per-class metrics (16 x 3)', ok)
    check('A: root class present in 34 runs', N['per_class_runs']['Root_disease'] == 34 == len(pc[(pc.config == 'none') & (pc['class'] == 'Root_disease')]))
    # grouping sensitivity
    sess = pd.read_csv(ROOT / 'data' / 'sessions.csv')
    sess = sess[sess.cls.isin(CLASSES)].reset_index(drop=True)
    sess['video'] = sess.session.str.startswith('video:')
    pv = preds.merge(sess[['cls', 'file', 'video']], on=['cls', 'file'], how='left')
    check('A: every prediction maps to sessions.csv', pv.video.notna().all())
    nv = []
    for (cfg, rep, k, s), g in pv[~pv.video].groupby(['config', 'repeat', 'fold', 'seed']):
        present = sorted(set(g.y_true))
        nv.append({'config': cfg, 'repeat': rep, 'fold': k, 'seed': s, 'n': len(g),
                   'f1': 100 * f1_score(g.y_true, g.y_pred, labels=present, average='macro', zero_division=0)})
    nv = pd.DataFrame(nv)
    pfv = nv.groupby(['config', 'repeat', 'fold']).f1.mean().unstack('config')
    for c in CONFIGS:
        S = N['grouping']['sensitivity'][f'no_video|{c}']
        ok = close(S['macro_f1_mean'], nv[nv.config == c].f1.mean()) and close(S['macro_f1_sd'], nv[nv.config == c].f1.std(ddof=1)) and close(S['test_n_mean'], nv[nv.config == c].n.mean(), 0.051)
        if c != 'none':
            m, lo, hi, p = corrected(pfv[c] - pfv['none'], ntr, nte)
            ok &= close(S['mean'], m, 0.0006) and close(S['ci_low'], lo, 0.0006) and close(S['ci_high'], hi, 0.0006) and close(S['p_corrected'], p, 0.0006)
        check(f'A: video frames excluded {c}', ok)
    g0 = pv[(pv.config == 'none') & (pv.cls == 'Phomopsis')]
    check('A: Phomopsis recall video vs stills', close(N['grouping']['phomopsis_recall_video'], 100 * (g0[g0.video].y_true == g0[g0.video].y_pred).mean(), 0.051)
          and close(N['grouping']['phomopsis_recall_stills'], 100 * (g0[~g0.video].y_true == g0[~g0.video].y_pred).mean(), 0.051))
    cm = pd.crosstab(pv[pv.config == 'none'].y_true, pv[pv.config == 'none'].y_pred, normalize='index') * 100
    check('A: confusion none row 0/2 -> leaf rot', close(N['confusion_row_pct']['none'][0][1], cm.loc[0, 1], 0.051) and close(N['confusion_row_pct']['none'][2][1], cm.loc[2, 1], 0.051))
    # external
    ext = pd.read_csv(ROOT / 'results' / 'external_vietnam.csv')
    E = N['external']
    ok = E['n_images'] == 1701 and E['seeds'] == 5
    piv = ext.pivot_table(index='seed', columns='config', values='vn_macro_f1_full')
    for c in CONFIGS:
        ok &= close(E['vn_macro_f1_full'][c][0], piv[c].mean()) and close(E['vn_macro_f1_full'][c][1], piv[c].std(ddof=1))
        if c != 'none':
            ok &= close(E['paired'][c]['mean'], (piv[c] - piv['none']).mean()) and close(E['paired'][c]['p_paired_t'], stats.ttest_rel(piv[c], piv['none']).pvalue, 0.0006)
            ok &= E['paired'][c]['wins'] == int(((piv[c] - piv['none']) > 0).sum())
    check('A: external test', ok)
    check('A: external masking changes means <= 0.2 pp', all(abs(E['vn_macro_f1_full'][c][0] - E['vn_macro_f1_3class'][c][0]) <= 0.2 for c in CONFIGS))
    check('A: external root predictions < 1 %', all(E['vn_pred_as_root_pct'][c][0] < 1 for c in CONFIGS))
    # cost
    lat = pd.read_csv(ROOT / 'results' / 'latency_params.csv').set_index('config')
    check('A: cost table matches latency_params.csv', all(close(N['cost'][c][k], lat.loc[c, k], 1e-6) for c in CONFIGS for k in ['params_M', 'extra_params', 'GMACs', 'cpu_ms_1thr_median']))
    check('A: LFA parameters = 1280 + 1', N['cost']['lfa']['extra_params'] == 1281)
    # v2
    r2 = pd.read_csv(ROOT / 'results' / 'v2' / 'results.csv')
    sz2 = r2.groupby(['repeat', 'fold'])[['n_train', 'n_val', 'n_test']].first()
    ntr2, nte2 = (sz2.n_train + sz2.n_val).mean(), sz2.n_test.mean()
    check('A: v2 uses the same folds', len(r2) == 200 and sz2.equals(sz))
    pf2 = r2.groupby(['config', 'repeat', 'fold']).test_macro_f1_present.mean().unstack('config')
    ps = {}
    for c in ['lfa14_224', 'lfa28_224', 'none_320', 'lfa14_320']:
        m, lo, hi, p = corrected(pf2[c] - pf2['none_224'], ntr2, nte2)
        ps[c] = p
        P = V['paired'][c]
        check(f'A: v2 paired {c}', close(P['mean'], m, 0.0006) and close(P['ci_low'], lo, 0.0006) and close(P['ci_high'], hi, 0.0006)
              and P['wins'] == (pf2[c] > pf2['none_224']).sum() and close(V['summary'][c]['mean'], r2[r2.config == c].test_macro_f1_present.mean()))
    order = sorted(ps, key=ps.get)
    holm = {}
    run = 0
    for i, c in enumerate(order):
        run = max(run, min(1.0, (4 - i) * ps[c]))
        holm[c] = run
    check('A: v2 Holm-adjusted p all 1.0', all(close(V['paired'][c]['p_holm'], holm[c], 0.0006) for c in holm) and all(v == 1.0 for v in holm.values()))
    m, lo, hi, p = corrected(pf2['lfa14_320'] - pf2['none_320'], ntr2, nte2)
    check('A: v2 secondary comparison', close(V['secondary_lfa14_320_vs_none_320']['mean'], m, 0.0006) and close(V['secondary_lfa14_320_vs_none_320']['ci_low'], lo, 0.0006))
    sg2 = r2.pivot_table(index=['config', 'repeat', 'fold'], columns='seed', values='test_macro_f1_present')
    check('A: v2 seed gap', close(V['seed_gap_mean'], (sg2[0] - sg2[1]).abs().mean()))
    check('A: v2 plain rerun vs main', close(V['none_main_experiment_mean'], N['summary']['none']['macro_f1_mean']))
    return N, V


# ============================================================ B. render match
def section_b():
    import render_prl
    rendered = render_prl.render(pdf=False)
    committed = (HERE / 'manuscript_prl.tex').read_text()
    check('B: template re-renders to the committed manuscript_prl.tex', rendered == committed)
    check('B: no unresolved placeholders', not re.search(r'\{\{[^{}]*\}\}|\[\[TABLE', rendered))
    md_before = (HERE / 'supplementary_prl.md').read_text()
    subprocess.run([sys.executable, str(HERE / 'make_supplement_prl.py')], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    check('B: supplement rebuilds identically', md_before == (HERE / 'supplementary_prl.md').read_text())
    return rendered


# ============================================================ C. hard-coded facts
def section_c(tex, N):
    s = pd.read_csv(ROOT / 'data' / 'sessions.csv')
    check('C: 560 released images, 526 IMG_, 20 grower, 14 other', len(s) == 560 and s.file.str.startswith('IMG_').sum() == 526
          and (s.session.str.startswith('whatsapp')).sum() == 20 and (s.session.str.startswith('single')).sum() == 14)
    pink = s[s.cls == 'Pink_disease']
    check('C: pink disease 10 images from 3 sessions', len(pink) == 10 and pink.session.nunique() == 3)
    check('C: 73 sessions within classes', s.session.nunique() == 73)
    s['g'] = Cm.build_groups(s).values
    check('C: 61 groups across classes', s.g.nunique() == 61)
    d = s[s.cls != 'Pink_disease'].reset_index(drop=True)
    d['g'] = Cm.build_groups(d).values
    gc = d.g.value_counts()
    big = d[d.g == gc.index[0]].cls.value_counts()
    check('C: 550 images, 60 groups, median 4.5, largest 128 (78/42/8)', len(d) == 550 and d.g.nunique() == 60 and gc.median() == 4.5
          and gc.iloc[0] == 128 and big.get('Phomopsis') == 78 and big.get('Leaf_rot') == 42 and big.get('Algal') == 8)
    counts = d.cls.value_counts()
    check('C: class counts 162/153/157/78', counts['Algal'] == 162 and counts['Leaf_rot'] == 153 and counts['Phomopsis'] == 157 and counts['Root_disease'] == 78)
    check('C: five multi-class groups', (d.groupby('g').cls.nunique() > 1).sum() == 5)
    img = d[d.file.str.startswith('IMG_')]
    dup = img.groupby('file').cls.nunique()
    dup = dup[dup > 1]
    check('C: 32 duplicated IMG_ names (9885-9972), 64 photographs in one group', len(dup) == 32 and dup.index.min() == 'IMG_9885.jpg'
          and dup.index.max() == 'IMG_9972.jpg' and img[img.file.isin(dup.index)].shape[0] == 64 and img[img.file.isin(dup.index)].g.nunique() == 1)
    v = d[d.session.str.startswith('video:')]
    ph = d[d.cls == 'Phomopsis']
    check('C: 7 videos, 63 frames, all Phomopsis; 7 of 14 Phomopsis groups are single videos',
          v.session.nunique() == 7 and len(v) == 63 and set(v.cls) == {'Phomopsis'} and ph.g.nunique() == 14
          and sum(1 for _, x in ph.groupby('g') if x.session.str.startswith('video:').all()) == 7)
    d['gl'] = Cm.build_groups(d, link_videos=True).values
    gl = d.gl.value_counts()
    check('C: linked largest group 202 incl. 152 Phomopsis', gl.iloc[0] == 202 and (d[d.gl == gl.index[0]].cls == 'Phomopsis').sum() == 152)
    root = d[d.cls == 'Root_disease'].g.value_counts()
    check('C: largest root group 49 of 78 (63 %)', root.iloc[0] == 49 and round(100 * 49 / 78) == 63)
    P = N['paired_macro_f1']
    widths = [P[c]['ci_high'] - P[c]['ci_low'] for c in ['lfa', 'se', 'cbam']]
    check('C: "9--14 points wide"', 8.5 <= min(widths) < 9.5 and 13.5 <= max(widths) < 14.5)
    check('C: "gains larger than about 4--6 points are excluded"', round(min(P[c]['ci_high'] for c in P)) == 4 and round(max(P[c]['ci_high'] for c in P)) == 6)
    check('C: "about 370 training images"', 365 <= N['design']['train_mean'] <= 375)
    e = N['external']['vn_macro_f1_full']
    check('C: external "39--41 %"', 39 <= min(v[0] for v in e.values()) and max(v[0] for v in e.values()) < 41.5)
    check('C: "SE/CBAM 5.1 % of the network"', close(100 * N['cost']['se']['extra_params'] / 1e6 / N['cost']['none']['params_M'], 5.1, 0.05))
    check('C: "about 1.8 points" at 320 px', close(json.load(open(ROOT / 'results' / 'v2' / 'numbers_v2.json'))['paired']['none_320']['mean'], 1.8, 0.05))
    folds = pd.read_csv(ROOT / 'data' / 'folds.csv')
    check('C: folds.csv has 20 folds and no group crosses a boundary',
          folds.groupby(['repeat', 'fold']).ngroups == 20 and all(
              folds[(folds.repeat == r) & (folds.fold == k)].merge(d[['g']], left_on='idx', right_index=True).groupby('g').role.nunique().eq(1).all()
              for r in range(5) for k in range(4)))
    # text claims checked against the tex
    check('C: tex states 160 and 200 runs', '160 runs' in tex and '200 runs' in tex)
    check('C: affiliation consistent with other submissions', 'Institute of Computer Science and Digital Innovation' in tex)


# ============================================================ D. journal rules
def section_d(tex):
    pdf = HERE / 'manuscript_prl.pdf'
    info = subprocess.run(['pdfinfo', str(pdf)], capture_output=True, text=True).stdout
    pages = int(re.search(r'Pages:\s+(\d+)', info).group(1))
    check(f'D: PDF pages {pages} <= 7', pages <= 7)
    text = subprocess.run(['pdftotext', '-layout', str(pdf), '-'], capture_output=True, text=True).stdout
    check('D: no unresolved references (??) in PDF', '??' not in text)
    abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', tex, re.S).group(1)
    words = len(re.sub(r'\\[a-zA-Z]+|[{}$\\]', ' ', abstract).split())
    check(f'D: abstract {words} words <= 200 (PRL template; guide allows 250)', words <= 200)
    kws = re.search(r'\\begin\{keyword\}(.*?)\\end\{keyword\}', tex, re.S).group(1).split(r'\sep')
    check(f'D: {len(kws)} keywords in 1-7', 1 <= len(kws) <= 7)
    bib = (HERE / 'refs.bib').read_text()
    keys = set(re.findall(r'@\w+\{([^,]+),', bib))
    cited = set(k.strip() for m in re.findall(r'\\cite\{([^}]+)\}', tex) for k in m.split(','))
    check('D: every cited key is in refs.bib', cited <= keys, str(cited - keys))
    check('D: every bib entry is cited', keys <= cited, str(keys - cited))
    hl = (HERE / 'highlights.txt').read_text().strip().splitlines()
    hl = [h.strip() for h in hl if h.strip()]
    check(f'D: {len(hl)} highlights, each <= 85 characters', 3 <= len(hl) <= 5 and all(len(h) <= 85 for h in hl), str([len(h) for h in hl]))
    for f in ['figs/fig1.pdf', 'figs/fig2.pdf', 'figs/graphical_abstract.png', 'supplementary_prl.pdf', 'supplementary_prl.docx', 'cover_letter_prl.docx', 'declaration_of_interest_prl.docx', 'authorship_confirmation_prl.pdf']:
        check(f'D: {f} exists', (HERE / f).exists())
    from PIL import Image
    w, h = Image.open(HERE / 'figs' / 'graphical_abstract.png').size
    check(f'D: graphical abstract {w}x{h} >= 1328x531', w >= 1328 and h >= 531)
    supp = (HERE / 'supplementary_prl.md').read_text()
    refs = set(re.findall(r'Table~S(\d)', tex)) | set(re.findall(r'Fig\.~S(\d)', tex))
    tables = set(re.findall(r'## Table S(\d)', supp))
    figs = set(re.findall(r'Figures S(\d) and S(\d)', supp).pop()) if 'Figures S' in supp else set()
    cited_tables = set(re.findall(r'Table~S(\d)', tex))
    cited_figs = set(re.findall(r'Fig\.~S(\d)', tex))
    check('D: supplementary tables cited in text exist', cited_tables <= tables, str(cited_tables - tables))
    check('D: supplementary figures cited in text exist', cited_figs <= figs, str(cited_figs - figs))
    check('D: generative-AI declaration before references', tex.index('Declaration of generative AI') < tex.index(r'\bibliography{refs}'))
    check('D: CRediT, competing interest, funding, data availability present',
          all(k in tex for k in ['CRediT authorship', 'Declaration of competing interest', 'Funding', 'Data availability']))
    check('D: equation (1) present and referenced', r'\label{eq:lfa}' in tex and r'\ref{eq:lfa}' in tex)


if __name__ == '__main__':
    N, V = section_a()
    tex = section_b()
    section_c(tex, N)
    section_d(tex)
    print(f'\n{len(PASSES)} passed, {len(FAILS)} failed')
    if FAILS:
        print('FAILED:', *FAILS, sep='\n  ')
        sys.exit(1)
