"""Render the Pattern Recognition Letters manuscript from prl_template.tex and the result files.

    python paper/prl/render_prl.py [--no-pdf]        # run from the repository root

Every ``{{expression}}`` in the template is evaluated against the numbers in results/numbers.json
(``N``, also exposed at top level), results/v2/numbers_v2.json (``V``) and a few derived quantities;
every ``[[TABLE:name]]`` is built from the same files. The output is paper/prl/manuscript_prl.tex,
compiled with latexmk unless --no-pdf is given. paper/prl/verify_prl.py re-renders and checks.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import torch_stub  # noqa: E402,F401  (lets common.py import without torch)
import common as Cm  # noqa: E402

CONFIGS = ['none', 'lfa', 'se', 'cbam']
CLASSES = ['Algal', 'Leaf_rot', 'Phomopsis', 'Root_disease']
LABEL = {'none': 'None', 'lfa': 'LFA', 'se': 'SE', 'cbam': 'CBAM'}
CLASS_LABEL = {'Algal': 'Algal leaf spot', 'Leaf_rot': 'Leaf rot', 'Phomopsis': r'\emph{Phomopsis} leaf spot',
               'Root_disease': 'Root and collar rot'}
ORGAN = {'Algal': 'Leaf', 'Leaf_rot': 'Leaf', 'Phomopsis': 'Leaf', 'Root_disease': 'Trunk base'}
V2_LABEL = {'none_224': 'None, 224 px', 'lfa14_224': r'LFA after block 5 ($14{\times}14$), 224 px',
            'lfa28_224': r'LFA after block 3 ($28{\times}28$), 224 px', 'none_320': 'None, 320 px',
            'lfa14_320': r'LFA after block 5 ($20{\times}20$), 320 px'}


class A(dict):
    """dict with attribute access, recursively."""
    def __getattr__(self, k):
        v = self[k]
        return A(v) if isinstance(v, dict) else v


# ------------------------------------------------------------------ formatting
def tex_num(s):
    return s.replace('-', r'$-$', 1) if s.startswith('-') else s


def f1(x): return tex_num(f'{x:.1f}')
def f2(x): return tex_num(f'{x:.2f}')
def f3(x): return tex_num(f'{x:.3f}')
def f4(x): return tex_num(f'{x:.4f}')
def pf2(x): return tex_num(f'{x:+.2f}')
def n(x): return f'{int(round(x)):,}'


def pct(x):
    v = round(x * 100, 1)
    return str(int(v)) if v == int(v) else f'{v:.1f}'


def pm(mean, sd, d=2):
    return f'{mean:.{d}f} $\\pm$ {sd:.{d}f}'


def ci(st):
    """'+0.23 [$-$7.39, 7.86]' from a dict with mean / ci_low / ci_high."""
    lo = tex_num('%.2f' % st['ci_low'])
    hi = tex_num('%.2f' % st['ci_high'])
    return '%s [%s, %s]' % (pf2(st['mean']), lo, hi)


# ------------------------------------------------------------------ data
def load():
    N = json.load(open(ROOT / 'results' / 'numbers.json'))
    V = json.load(open(ROOT / 'results' / 'v2' / 'numbers_v2.json'))
    sens = {}
    for k, v in N['grouping']['sensitivity'].items():
        var, cfg = k.split('|')
        sens.setdefault(var, {})[cfg] = v
    pc = {}
    for k, v in N['per_class'].items():
        cfg, cl = k.split('|')
        for m in ['precision', 'recall', 'f1']:
            mean, sd = v[m].split(' ± ')
            pc.setdefault(cfg, {}).setdefault(cl, {})[m + '_mean'] = mean
            pc[cfg][cl][m + '_sd'] = sd
    hw = {c: (N['paired_macro_f1'][c]['ci_high'] - N['paired_macro_f1'][c]['ci_low']) / 2 for c in ['lfa', 'se', 'cbam']}
    ps = [N['paired_macro_f1'][c][k] for c in ['lfa', 'se', 'cbam'] for k in ['p_uncorrected', 'p_wilcoxon']]
    min_uncorr_p = np.floor(min(ps) * 100) / 100
    # exact per-class change LFA vs none from the predictions (unrounded)
    preds = pd.read_csv(ROOT / 'results' / 'preds.csv')
    rows = []
    for (cfg, rep, k, s), g in preds.groupby(['config', 'repeat', 'fold', 'seed']):
        p, r, f, cnt = precision_recall_fscore_support(g.y_true, g.y_pred, labels=list(range(4)), zero_division=0)
        for i, cl in enumerate(CLASSES):
            if cnt[i]:
                rows.append({'config': cfg, 'class': cl, 'recall': r[i] * 100, 'f1': f[i] * 100})
    pcm = pd.DataFrame(rows).groupby(['config', 'class'])[['recall', 'f1']].mean()
    max_pc_f1_change = float((pcm.loc['lfa', 'f1'] - pcm.loc['none', 'f1']).abs().max())
    max_pc_recall_change = float((pcm.loc['lfa', 'recall'] - pcm.loc['none', 'recall']).abs().max())
    ctx = dict(N)
    ctx.update({'N': N, 'V': V, 'sens': sens, 'pc': pc, 'hw': hw, 'min_uncorr_p': min_uncorr_p,
                'cm': N['confusion_row_pct'], 'max_pc_f1_change': max_pc_f1_change,
                'max_pc_recall_change': max_pc_recall_change})
    return {k: (A(v) if isinstance(v, dict) else v) for k, v in ctx.items()}


def class_table():
    s = pd.read_csv(ROOT / 'data' / 'sessions.csv')
    d = s[s.cls.isin(CLASSES)].reset_index(drop=True)
    d['group'] = Cm.build_groups(d).values
    rows = []
    for cl in CLASSES:
        g = d[d.cls == cl]
        gc = g.group.value_counts()
        rows.append((CLASS_LABEL[cl], ORGAN[cl], len(g), g.group.nunique(), 100 * gc.iloc[0] / len(g)))
    total_groups = d.group.nunique()
    out = [r'\begin{table}[t]', r'\centering', r'\footnotesize',
           r'\caption{The four classes. A capture group is a set of photographs kept together in every split (Section~\ref{sec:data}). '
           r'Five groups contain images of more than one class, so the class rows add up to more than the '
           f'{total_groups} groups in total. The last column is the share of each class held by its largest group; '
           r'seven of the 14 \emph{Phomopsis} groups are single videos.}',
           r'\label{tab:classes}',
           r'\begin{tabular}{@{}llrrr@{}}', r'\toprule',
           r'Class & Organ & Images & Groups & Largest group (\%) \\', r'\midrule']
    for name, organ, ni, ng, share in rows:
        out.append(f'{name} & {organ} & {ni} & {ng} & {share:.1f} \\\\')
    out += [r'\midrule', f'Total & & {len(d)} & {total_groups} & \\\\', r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(out), {'classes': rows, 'total_groups': total_groups, 'n_images': len(d)}


def main_table(ctx):
    N, sens = ctx['N'], ctx['sens']
    out = [r'\begin{table*}[t]', r'\centering', r'\footnotesize',
           r'\caption{Macro F1 over ' + str(N['design']['runs']) + r' runs (40 per configuration; mean $\pm$ SD) and paired difference from the plain network over 20 folds, '
           r'scored on all test images and with the ' + str(N['grouping']['video_frames']) + r' video frames excluded. CI and $p$ from the corrected resampled $t$-test; '
           r"``Folds higher'' counts the folds on which the module beat the plain network (all images). Accuracy is in supplementary Table~S2.}",
           r'\label{tab:main}',
           r'\begin{tabular}{@{}lrlrrlrr@{}}', r'\toprule',
           r' & \multicolumn{3}{c}{All test images} & \multicolumn{3}{c}{Video frames excluded} & \\',
           r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
           r'Configuration & Macro F1 (\%) & Difference (pp) [95\% CI] & $p$ & Macro F1 (\%) & Difference (pp) [95\% CI] & $p$ & Folds higher \\',
           r'\midrule']
    for c in CONFIGS:
        a, b = sens['all'][c], sens['no_video'][c]
        if c == 'none':
            out.append(f'{LABEL[c]} & {pm(a["macro_f1_mean"], a["macro_f1_sd"])} & -- & -- & {pm(b["macro_f1_mean"], b["macro_f1_sd"])} & -- & -- & -- \\\\')
        else:
            w = int(N['paired_macro_f1'][c]['wins'])
            out.append(f'{LABEL[c]} & {pm(a["macro_f1_mean"], a["macro_f1_sd"])} & {ci(a)} & {a["p_corrected"]:.3f} & '
                       f'{pm(b["macro_f1_mean"], b["macro_f1_sd"])} & {ci(b)} & {b["p_corrected"]:.3f} & {w}/20 \\\\')
    out += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    return '\n'.join(out)


def followup_table(ctx):
    V = ctx['V']
    out = [r'\begin{table}[t]', r'\centering', r'\footnotesize', r'\setlength{\tabcolsep}{3.5pt}',
           r'\caption{Pre-registered follow-up (' + str(V['runs']) + r' runs; 40 per configuration): LFA at earlier insertion points and at a higher input resolution. '
           r'Mean $\pm$ SD of macro F1 and paired difference from the 224-pixel plain network over 20 folds, corrected test, Holm adjustment over the four comparisons.}',
           r'\label{tab:followup}',
           r'\resizebox{\columnwidth}{!}{%', r'\begin{tabular}{@{}lrlrr@{}}', r'\toprule',
           r'Configuration & Macro F1 (\%) & Difference (pp) [95\% CI] & $p_{\mathrm{Holm}}$ & Folds higher \\', r'\midrule']
    for c in ['none_224', 'lfa14_224', 'lfa28_224', 'none_320', 'lfa14_320']:
        s = V['summary'][c]
        if c == 'none_224':
            out.append(f'{V2_LABEL[c]} & {pm(s["mean"], s["sd"])} & -- & -- & -- \\\\')
        else:
            p = V['paired'][c]
            out.append(f'{V2_LABEL[c]} & {pm(s["mean"], s["sd"])} & {ci(p)} & {p["p_holm"]:.3f} & {int(p["wins"])}/20 \\\\')
    out += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    return '\n'.join(out)


def perclass_table(ctx):
    pc, runs = ctx['N']['per_class'], ctx['N']['per_class_runs']
    out = [r'\begin{table}[t]', r'\centering', r'\footnotesize', r'\setlength{\tabcolsep}{4pt}',
           r'\caption{Per-class precision, recall and F1 (\%, mean $\pm$ SD over runs) for the plain network and LFA, all test images. '
           r'Root and collar rot was present in the test set in ' + str(runs['Root_disease']) + r' of 40 runs per configuration, the other classes in all 40. '
           r'\emph{Phomopsis} values include the video frames (Section~\ref{sec:perclass}). SE and CBAM are in supplementary Table~S1.}',
           r'\label{tab:perclass}',
           r'\begin{tabular}{@{}llrrr@{}}', r'\toprule',
           r'Class & Config. & Precision & Recall & F1 \\', r'\midrule']
    for cl in CLASSES:
        for i, c in enumerate(['none', 'lfa']):
            v = pc[f'{c}|{cl}']
            cells = ' & '.join(v[m].replace(' ± ', ' $\\pm$ ') for m in ['precision', 'recall', 'f1'])
            out.append(f'{CLASS_LABEL[cl] if i == 0 else ""} & {LABEL[c]} & {cells} \\\\')
    out += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(out)


def external_table(ctx):
    E = ctx['N']['external']
    out = [r'\begin{table}[t]', r'\centering', r'\footnotesize',
           r'\caption{Zero-shot macro F1 on ' + f'{E["n_images"]:,}' + r' images of the Vietnamese dataset (three shared classes, four-class output; mean $\pm$ SD over five seeds) '
           r'and paired difference from the plain network. Random guessing would score about 27.9\%.}',
           r'\label{tab:external}',
           r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule',
           r'Configuration & Macro F1 (\%) & Diff.\ (pp) & $p$, paired $t$ & Seeds higher \\', r'\midrule']
    for c in CONFIGS:
        m, sd = E['vn_macro_f1_full'][c]
        if c == 'none':
            out.append(f'{LABEL[c]} & {pm(m, sd)} & -- & -- & -- \\\\')
        else:
            p = E['paired'][c]
            out.append(f'{LABEL[c]} & {pm(m, sd)} & {pf2(p["mean"])} & {p["p_paired_t"]:.3f} & {p["wins"]}/{p["n"]} \\\\')
    out += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(out)


# ------------------------------------------------------------------ render
def render(pdf=True):
    ctx = load()
    tables = {'main': main_table(ctx), 'followup': followup_table(ctx), 'external': external_table(ctx),
              'perclass': perclass_table(ctx)}
    tables['classes'], _ = class_table()
    tpl = (HERE / 'prl_template.tex').read_text()
    env = dict(ctx)
    env.update({'f1': f1, 'f2': f2, 'f3': f3, 'f4': f4, 'pf2': pf2, 'n': n, 'pct': pct, 'int': int, 'round': round, 'min': min, 'max': max, 'abs': abs})

    def sub(m):
        expr = m.group(1).strip()
        try:
            return str(eval(expr, {'__builtins__': {}}, env))
        except Exception as e:  # pragma: no cover
            raise SystemExit(f'cannot evaluate {{{{{expr}}}}}: {e}')
    body = re.sub(r'\{\{(.+?)\}\}', sub, tpl)
    body = re.sub(r'\[\[TABLE:(\w+)\]\]', lambda m: tables[m.group(1)], body)
    out = HERE / 'manuscript_prl.tex'
    out.write_text(body)
    print('wrote', out)
    if pdf:
        subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error', '-quiet', 'manuscript_prl.tex'],
                       cwd=HERE, check=True, stdout=subprocess.DEVNULL)
        print('compiled', HERE / 'manuscript_prl.pdf')
    return body


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-pdf', action='store_true')
    a = ap.parse_args()
    render(pdf=not a.no_pdf)
