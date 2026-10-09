"""Check that every computed number quoted in the manuscript matches the result files.

    python paper/check_numbers.py path/to/manuscript.tex     # after compute_numbers.py

Each check formats a value from paper/numbers.json / results/numbers.json / results/v2/numbers_v2.json
and requires that exact string to occur in the manuscript. Exits non-zero, naming the item, on a miss.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if len(sys.argv) != 2:
    sys.exit('usage: python paper/check_numbers.py path/to/manuscript.tex')
TEX = Path(sys.argv[1]).read_text()
J = json.load(open(HERE / 'numbers.json'))
O = json.load(open(ROOT / 'results' / 'numbers.json'))
V = json.load(open(ROOT / 'results' / 'v2' / 'numbers_v2.json'))


def f(x, d=2):
    s = f'{x:.{d}f}'
    return s.replace('-', '$-$', 1) if s.startswith('-') else s


def m(x, d=2):  # math-mode minus inside $...$
    s = f'{x:.{d}f}'
    return s.replace('-', '-', 1)


checks = []
A, NV = J['main_all'], J['main_novid']
NAME = {'lfa': 'SG', 'se': 'SE', 'cbam': 'CBAM', 'none': 'Plain'}

# Table 4 rows (primary and all-images columns)
for c in ['none', 'lfa', 'se', 'cbam']:
    row = f"{NAME[c]} & {NV['summary'][c]['mean']:.2f} $\\pm$ {NV['summary'][c]['sd']:.2f}"
    checks.append((f'Table 4 {c} primary', row))
    checks.append((f'Table 4 {c} all', f"{A['summary'][c]['mean']:.2f} $\\pm$ {A['summary'][c]['sd']:.2f}"))
for c in ['lfa', 'se', 'cbam']:
    for sc, B in [('novid', NV), ('all', A)]:
        p = B['paired'][c]
        checks.append((f'Table 4 {c} {sc} CI',
                       f"${m(p['mean'])}$ [${m(p['ci_low'])}$, {p['ci_high']:.2f}]"))
    p = NV['paired'][c]
    checks.append((f'Table 4 {c} p/TOST/post', f"{p['p_corrected']:.3f} & {p['p_tost_2pp']:.2f} & {p['post_within_2pp']:.2f}"))
    checks.append((f'min half-width {c}', f"{p['min_half_width_inf_repeats']:.2f}"))
    checks.append((f'accuracy diff {c}', f"${m(NV['paired_acc'][c]['mean'])}$"))
checks.append(('sd_d SG', f"{NV['paired']['lfa']['sd_d']:.2f} points for the SG"))
checks.append(('wins SG', f"higher on {NV['paired']['lfa']['wins']} of 20 folds"))
import math
fl = lambda x: f'{math.floor(x * 100) / 100:.2f}'  # a lower bound must be rounded down
checks.append(('min uncorrected p novid', f"p \\geq {fl(min(min(NV['paired'][c]['p_uncorrected'], NV['paired'][c]['p_wilcoxon']) for c in ['lfa','se','cbam']))}"))
checks.append(('min uncorrected p all', f"p \\geq {fl(min(min(A['paired'][c]['p_uncorrected'], A['paired'][c]['p_wilcoxon']) for c in ['lfa','se','cbam']))}"))
checks.append(('plain accuracy novid', f"{NV['summary']['none']['acc_mean']:.2f}\\%"))
# per-class table
CL = {'Algal': 'Algal leaf spot', 'Leaf_rot': 'Leaf blight complex', 'Phomopsis': '\\emph{Phomopsis} leaf spot', 'Root_disease': 'Root and collar rot'}
for c, lab in [('none', 'Plain'), ('lfa', 'SG')]:
    for k in CL:
        v = NV['per_class'][c][k]
        checks.append((f'per-class {c} {k}', f"{lab} & {v['p'][0]:.1f} $\\pm$ {v['p'][1]:.1f} & {v['r'][0]:.1f} $\\pm$ {v['r'][1]:.1f} & {v['f1'][0]:.1f} $\\pm$ {v['f1'][1]:.1f}"))
checks.append(('Phomopsis runs', f"\\emph{{Phomopsis}} in {NV['per_class']['none']['Phomopsis']['runs']}"))
cm = NV['confusion_row_pct']['none']
checks += [('algal->blight', f'{cm[0][1]:.1f}\\% of algal'), ('phom->blight', f'{cm[2][1]:.1f}\\% of its still'),
           ('phom->algal', f'{cm[2][0]:.1f}\\% as algal'), ('phom stills', f"{J['phomopsis_recall']['stills']:.1f}\\%"),
           ('phom video', f"{J['phomopsis_recall']['video']:.1f}\\%")]
# variation
checks += [('fold range', f"{NV['fold']['min']:.1f}\\% to {NV['fold']['max']:.1f}\\% (SD {NV['fold']['sd']:.2f}"),
           ('config SD', f"(SD {NV['fold']['sd_config_means']:.2f})"),
           ('seed mean', f"{NV['seed']['mean_abs']:.2f} points on average (median {NV['seed']['median_abs']:.2f}, maximum {NV['seed']['max_abs']:.2f}"),
           ('seed all', f"{A['seed']['mean_abs']:.2f} points, so")]
for c, lab in [('none', 'plain network'), ('lfa', 'SG'), ('se', 'SE'), ('cbam', 'CBAM')]:
    checks.append((f'seed median {c}', f"{NV['seed']['by_config_median_abs'][c]:.2f} for" if c != 'none' else f"{NV['seed']['by_config_median_abs'][c]:.2f} points for the plain"))
s = NV['single_run']
checks += [('single SG', f"{round(s['lfa']['ahead3']*40)} of 40 seed-matched pairs ({s['lfa']['ahead3']*100:.0f}\\%) and at least 3 points behind in {round(s['lfa']['behind3']*40)} ({s['lfa']['behind3']*100:.1f}\\%)"),
           ('single SG range', f"$-{abs(s['lfa']['min']):.2f}$ to $+{s['lfa']['max']:.2f}$"),
           ('single SE', f"{s['se']['ahead3']*100:.1f}\\% ahead and {s['se']['behind3']*100:.1f}\\% behind"),
           ('single CBAM', f"CBAM {s['cbam']['ahead3']*100:.0f}\\% and {s['cbam']['behind3']*100:.0f}\\%")]
# design
D = J['design']
checks += [('factor', f"{D['factor']:.3f} against"), ('factor train', f"{D['factor_train_only']:.3f}"),
           ('ratio range', f"{D['fold_ratio_min']:.2f} to {D['fold_ratio_max']:.2f}"),
           ('test novid', f"{J['test_size_novid']['min']}--{J['test_size_novid']['max']}, mean {NV['summary']['none']['n_test_mean']:.1f}")]
# follow-up
VA, VN = J['v2_all'], J['v2_novid']
for c in ['lfa14_224', 'lfa28_224', 'none_320', 'lfa14_320']:
    for B in (VA, VN):
        p = B['paired'][c]
        sign = '+' if p['mean'] > 0 else '-'
        checks.append((f'v2 {c}', f"${sign}{abs(p['mean']):.2f}$ [${m(p['ci_low'])}$, {p['ci_high']:.2f}]"))
    checks.append((f'v2 holm {c}', f"{VA['paired'][c]['p_holm']:.3f}"))
    assert abs(VA['paired'][c]['mean'] - V['paired'][c]['mean']) < 0.01, c
for c in ['none_224', 'lfa14_224', 'lfa28_224', 'none_320', 'lfa14_320']:
    checks.append((f'v2 mean {c}', f"{VA['summary'][c]['mean']:.2f} &"))
    checks.append((f'v2 novid mean {c}', f"& {VN['summary'][c]['mean']:.2f} &"))
sec = VA['secondary_lfa320_vs_none320']
checks += [('v2 secondary', f"${m(sec['mean'])}$ points against the 320-pixel plain network, 95\\% CI ${m(sec['ci_low'])}$ to ${sec['ci_high']:.2f}$)"),
           ('v2 uncorr 28', f"$p = {VA['paired']['lfa28_224']['p_uncorrected']:.3f}$ on all images and {VN['paired']['lfa28_224']['p_uncorrected']:.3f}"),
           ('v2 seed', f"{VA['seed']['mean_abs']:.2f} points, in line")]
# external and cost (from the original numbers.json)
E = O['external']
for c in ['none', 'lfa', 'se', 'cbam']:
    mu, sd = E['vn_macro_f1_full'][c]
    checks.append((f'ext {c}', f"{NAME[c]} & {mu:.2f} $\\pm$ {sd:.2f}"))
for c in ['lfa', 'se', 'cbam']:
    checks.append((f'ext p {c}', f"{E['paired'][c]['p_paired_t']:.3f} & {E['paired'][c]['wins']}/5"))
C = O['cost']
checks += [('latency', f"{C['none']['cpu_ms_1thr_median']:.1f}\\,ms per image"),
           ('lat SG', f"${m(C['lfa']['cpu_ms_1thr_median'] - C['none']['cpu_ms_1thr_median'])}$\\,ms (SG)"),
           ('lat CBAM', f"$+{C['cbam']['cpu_ms_1thr_median'] - C['none']['cpu_ms_1thr_median']:.2f}$\\,ms (CBAM)"),
           ('grouping', f"{O['grouping']['largest_group_linked']} images, including {O['grouping']['largest_group_linked_phomopsis']} of the 157")]
S = O['grouping']['sensitivity']['unsplit_linked|lfa']
checks.append(('unsplit', f"${m(S['mean'])}$ points for the SG (95\\% CI ${m(S['ci_low'])}$ to ${S['ci_high']:.2f}$)"))

bad = [(k, v) for k, v in checks if v not in TEX]
for k, v in bad:
    print(f'MISSING  {k}: {v}')
print(f'{len(checks) - len(bad)}/{len(checks)} checks passed')
sys.exit(1 if bad else 0)
