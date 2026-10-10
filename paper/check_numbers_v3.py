"""Check that every number of the backbone and referral experiment quoted in a manuscript matches the result files.

    python paper/check_numbers_v3.py path/to/manuscript.tex     # after compute_numbers_v3.py

Complements check_numbers.py (which covers the attention-module experiments). Numbers are formatted here
independently of the script that writes the manuscript tables. Also recomputes the headline means directly from
results/v3/results.csv. Exits non-zero, naming the item, on a miss.
"""
import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if len(sys.argv) != 2:
    sys.exit('usage: python paper/check_numbers_v3.py path/to/manuscript.tex')
TEX = Path(sys.argv[1]).read_text()
V = json.load(open(HERE / 'numbers_v3.json'))
NV, AL = V['novid'], V['all']
checks = []


def r(x, d=2):  # round half up
    return str(Decimal(str(x)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def sm(x, d=2):  # signed, math minus
    return f'$-{r(abs(x), d)}$' if x < 0 else f'$+{r(x, d)}$'


def um(x, d=2):
    return f'$-{r(abs(x), d)}$' if x < 0 else r(x, d)


NAME = {'effb0': 'EfficientNet-B0', 'mnv3l': 'MobileNetV3-Large', 'mnv4m': 'MobileNetV4-Conv-Medium',
        'cnxv2t': 'ConvNeXt V2-Tiny', 'vits': 'ViT-S/16'}
BB = list(NAME)

# 0. headline means recomputed from the raw per-run file (all-image scoring)
res = pd.read_csv(ROOT / 'results' / 'v3' / 'results.csv')
for c in BB:
    raw = res[res.config == c].test_macro_f1_present.mean()
    assert abs(raw - AL['summary'][c]['mean']) < 0.01, (c, raw, AL['summary'][c]['mean'])
assert res.groupby('config').size().eq(40).all()
checks.append(('factor', '0.383'))

# 1. Table 7 rows
for c in BB:
    s, a = NV['summary'][c], AL['summary'][c]
    row = f"{NAME[c]} & {r(s['mean'])} $\\pm$ {r(s['sd'])}"
    if c != 'effb0':
        p, q = NV['paired'][c], AL['paired'][c]
        row += (f" & {sm(p['mean'])} [{um(p['ci_low'])}, {um(p['ci_high'])}] & {r(p['p_corrected'], 3)} & {r(p['p_holm'], 3)} & {p['wins']}/20"
                f" & {r(a['mean'])} $\\pm$ {r(a['sd'])} & {r(q['p_holm'], 3)} & {r(V['cost'][c]['cpu_ms_1thr_median'], 1)}")
    else:
        row += f" & -- & -- & -- & -- & {r(a['mean'])} $\\pm$ {r(a['sd'])} & -- & {r(V['cost'][c]['cpu_ms_1thr_median'], 1)}"
    checks.append((f'Table 7 {c}', row))

# 2. Table 8 (per class) and Phomopsis recall
for c in BB:
    P = NV['per_class'][c]
    cells = ' & '.join(f"{r(P[k]['f1'][0], 1)} $\\pm$ {r(P[k]['f1'][1], 1)}" for k in ['Algal', 'Leaf_rot', 'Phomopsis', 'Root_disease'])
    rec = AL['phomopsis_recall'][c]
    checks.append((f'Table 8 {c}', f"{NAME[c]} & {cells} & {r(rec['stills'], 1)} & {r(rec['video'], 1)}"))

# 3. Table 9 (referral)
for c in BB:
    x = NV['selective']['90'][c]
    a = NV['aurc_top2'][c]
    checks.append((f'Table 9 {c}', f"{NAME[c]} & {r(x['coverage'][0], 1)} $\\pm$ {r(x['coverage'][1], 1)} & {r(x['sel_acc_mean_of_runs'][0], 1)} $\\pm$ {r(x['sel_acc_mean_of_runs'][1], 1)} & "
                  f"{r(NV['summary'][c]['acc_mean'], 1)} & {x['runs_below_target']}/{x['runs_with_accepted']} & {x['runs_all_accepted']}/40 & {r(a['acc_at_70'], 1)} & {r(a['aurc'][0])} & {r(a['top2'][0], 1)}"))

# 4. prose
cn, vt, ef = NV['paired']['cnxv2t'], NV['paired']['vits'], 'effb0'
checks += [
    ('abstract f1', f"{r(NV['summary']['cnxv2t']['mean'], 1)}\\% and {r(NV['summary']['vits']['mean'], 1)}\\% against {r(NV['summary']['effb0']['mean'], 1)}\\%"),
    ('abstract holm', f"p = {r(cn['p_holm'], 3)} and {r(vt['p_holm'], 3)}"),
    ('abstract time', f"{r(V['cost']['vits']['cpu_ms_1thr_median'] / V['cost']['effb0']['cpu_ms_1thr_median'], 1)} to {r(V['cost']['cnxv2t']['cpu_ms_1thr_median'] / V['cost']['effb0']['cpu_ms_1thr_median'], 1)} times the CPU time"),
    ('abstract 70', f"{r(NV['aurc_top2']['cnxv2t']['acc_at_70'], 1)}\\% accurate on its 70\\% most confident images"),
    ('abstract top2', f"correct class for {r(NV['aurc_top2']['cnxv2t']['top2'][0], 1)}\\%"),
    ('abstract cov', f"decided {r(NV['selective']['90']['cnxv2t']['coverage'][0], 0)}\\% of images with {r(NV['selective']['90']['cnxv2t']['sel_acc_mean_of_runs'][0], 0)}\\% accuracy"),
    ('cnx diff', f"{sm(cn['mean'])} points for ConvNeXt V2-Tiny (95\\% CI {r(cn['ci_low'])} to {r(cn['ci_high'])})"),
    ('vit diff', f"{sm(vt['mean'])} points for ViT-S/16 ({um(vt['ci_low'])} to {r(vt['ci_high'])})"),
    ('wins', f"ahead on {cn['wins']} of the 20 folds"),
    ('p raw', f"{r(cn['p_corrected'], 3)} and {r(vt['p_corrected'], 3)}, and after Holm adjustment over the four comparisons {r(cn['p_holm'], 3)} and {r(vt['p_holm'], 3)}"),
    ('all images', f"{sm(AL['paired']['cnxv2t']['mean'])} and {sm(AL['paired']['vits']['mean'])} points; Holm $p = {r(AL['paired']['cnxv2t']['p_holm'], 3)}$ and ${r(AL['paired']['vits']['p_holm'], 3)}$; ahead on {AL['paired']['cnxv2t']['wins']} of 20 folds"),
    ('vs each other', f"({um(NV['cnxv2t_vs_vits']['mean'])} points, corrected $p = {r(NV['cnxv2t_vs_vits']['p_corrected'], 3)}$"),
    ('mnv4m', f"scored {r(abs(NV['paired']['mnv4m']['mean']))} points lower than EfficientNet-B0 (95\\% CI {um(NV['paired']['mnv4m']['ci_low'])} to {um(NV['paired']['mnv4m']['ci_high'])}; Holm $p = {r(NV['paired']['mnv4m']['p_holm'], 3)}$; ahead on {NV['paired']['mnv4m']['wins']} of 20 folds)"),
    ('mnv3l', f"MobileNetV3-Large {r(abs(NV['paired']['mnv3l']['mean']))} points lower ({um(NV['paired']['mnv3l']['ci_low'])} to {r(NV['paired']['mnv3l']['ci_high'])}; Holm $p = {r(NV['paired']['mnv3l']['p_holm'], 3)}$)"),
    ('seeds', f"{r(NV['seed_abs_by_config']['cnxv2t'])} points for ConvNeXt V2-Tiny and {r(NV['seed_abs_by_config']['vits'])} for ViT-S/16, against {r(NV['seed_abs_by_config']['effb0'])} for EfficientNet-B0, {r(NV['seed_abs_by_config']['mnv3l'])} for MobileNetV3-Large and {r(NV['seed_abs_by_config']['mnv4m'])} for MobileNetV4-Conv-Medium"),
    ('rerun', f"({r(V['rerun_agreement']['novid']['here'])}\\% against {r(V['rerun_agreement']['novid']['first'])}\\%), but matching runs differed by {r(V['rerun_agreement']['novid']['mean_abs_diff'])} points on average (correlation {r(V['rerun_agreement']['novid']['corr'])})"),
    ('fold range cnx', f"{r(min(NV['fold_macro_f1']['cnxv2t']), 1)}\\% to {r(max(NV['fold_macro_f1']['cnxv2t']), 1)}\\% for ConvNeXt V2-Tiny and from {r(min(NV['fold_macro_f1']['vits']), 1)}\\% to {r(max(NV['fold_macro_f1']['vits']), 1)}\\% for ViT-S/16"),
    ('min hw', f"smaller than {r(cn['min_half_width_inf_repeats'])} and {r(vt['min_half_width_inf_repeats'])} points"),
    ('params', f"{r(V['cost']['cnxv2t']['params_M'])} million parameters and ViT-S/16 {r(V['cost']['vits']['params_M'])} million"),
    ('lat', f"was {r(V['cost']['cnxv2t']['cpu_ms_1thr_median'], 1)} and {r(V['cost']['vits']['cpu_ms_1thr_median'], 1)}\\,ms"),
    ('lat p90', f"(90th percentile {r(V['cost']['cnxv2t']['cpu_ms_1thr_p90'], 1)} and {r(V['cost']['vits']['cpu_ms_1thr_p90'], 1)}\\,ms)"),
    ('lat mnv', f"MobileNetV3-Large ({r(V['cost']['mnv3l']['cpu_ms_1thr_median'], 1)}\\,ms) and MobileNetV4-Conv-Medium ({r(V['cost']['mnv4m']['cpu_ms_1thr_median'], 1)}\\,ms)"),
]
a = NV['aurc_top2']
ap, tp = NV['aurc_paired']['cnxv2t'], NV['top2_paired']['cnxv2t']
checks += [
    ('aurc', f"The AURC was {r(a['cnxv2t']['aurc'][0])} and {r(a['vits']['aurc'][0])} for these two backbones and {r(a['effb0']['aurc'][0])} for EfficientNet-B0"),
    ('aurc diff', f"{sm(ap['mean'])} (95\\% CI {um(ap['ci_low'])} to {um(ap['ci_high'])}, corrected $p = {r(ap['p_corrected'], 3)}$, Holm {r(ap['p_holm'], 3)})"),
    ('top2', f"{r(a['cnxv2t']['top2'][0])}\\% and {r(a['vits']['top2'][0])}\\% of images against {r(a['effb0']['top2'][0])}\\%"),
    ('top2 diff', f"{sm(tp['mean'])} points for ConvNeXt V2-Tiny ({um(tp['ci_low'])} to {r(tp['ci_high'])})"),
    ('acc70', f"{r(a['cnxv2t']['acc_at_70'], 1)}\\% for ConvNeXt V2-Tiny and {r(a['vits']['acc_at_70'], 1)}\\% for ViT-S/16, against {r(a['effb0']['acc_at_70'], 1)}\\% for EfficientNet-B0"),
]
for tg in ['90', '95']:
    for c in ['effb0', 'cnxv2t', 'vits']:
        x = NV['selective'][tg][c]
        checks.append((f'sel {tg} {c}', f"{r(x['coverage'][0], 1)}\\%"))
x90 = {c: NV['selective']['90'][c] for c in BB}
checks += [
    ('below', f"below 90\\% in {x90['cnxv2t']['runs_below_target']} of {x90['cnxv2t']['runs_with_accepted']} ConvNeXt V2-Tiny runs, {x90['vits']['runs_below_target']} of {x90['vits']['runs_with_accepted']} ViT-S/16 runs and {x90['effb0']['runs_below_target']} of {x90['effb0']['runs_with_accepted']} EfficientNet-B0 runs"),
    ('no referral', f"in {x90['cnxv2t']['runs_all_accepted']} of 40 ConvNeXt V2-Tiny runs and {x90['vits']['runs_all_accepted']} of 40 ViT-S/16 runs"),
    ('95 line', f"decided {r(NV['selective']['95']['effb0']['coverage'][0], 1)}\\%, {r(NV['selective']['95']['cnxv2t']['coverage'][0], 1)}\\% and {r(NV['selective']['95']['vits']['coverage'][0], 1)}\\% of the images with an accuracy of {r(NV['selective']['95']['effb0']['sel_acc_mean_of_runs'][0], 1)}\\%, {r(NV['selective']['95']['cnxv2t']['sel_acc_mean_of_runs'][0], 1)}\\% and {r(NV['selective']['95']['vits']['sel_acc_mean_of_runs'][0], 1)}\\%"),
    ('gain cnx', f"Referring {r(100 - x90['cnxv2t']['coverage'][0], 0)}\\% of images raised the accuracy of ConvNeXt V2-Tiny on the decided images from {r(NV['summary']['cnxv2t']['acc_mean'], 1)}\\% to {r(x90['cnxv2t']['sel_acc_mean_of_runs'][0], 1)}\\%"),
    ('gain eff', f"referring {r(100 - x90['effb0']['coverage'][0], 0)}\\% raised it from {r(NV['summary']['effb0']['acc_mean'], 1)}\\% to {r(x90['effb0']['sel_acc_mean_of_runs'][0], 1)}\\%"),
]
d = x90['effb0']['defer_pct_run_mean']
checks.append(('defer eff', f"referred {r(d['Phomopsis'], 1)}\\% of \\emph{{Phomopsis}} images, {r(d['Algal'], 1)}\\% of algal leaf spot, {r(d['Leaf_rot'], 1)}\\% of leaf blight and {r(d['Root_disease'], 1)}\\% of root and collar rot"))
fol = [x90[c]['defer_pct_run_mean'][k] for c in ('cnxv2t', 'vits') for k in ('Algal', 'Leaf_rot', 'Phomopsis')]
checks.append(('defer range', f"between {r(min(fol), 1)}\\% and {r(max(fol), 1)}\\%"))
checks.append(('defer root', f"({r(x90['cnxv2t']['defer_pct_run_mean']['Root_disease'], 1)}\\% and {r(x90['vits']['defer_pct_run_mean']['Root_disease'], 1)}\\%)"))
checks.append(('val size', '34.8 images on average'))
assert abs(NV['selective']['90']['effb0']['val_n_mean'] - 34.8) < 0.05
# validation optimism, pooled values and the video-linked sensitivity analysis
un = V['unsplit']
for c in ['cnxv2t', 'vits']:
    p = un['paired'][c]
    checks.append((f'unsplit {c}', f"{sm(p['mean'])} points" if c == 'cnxv2t' else f"{sm(p['mean'])} for ViT-S/16 ({um(p['ci_low'])} to {um(p['ci_high'])})"))
checks.append(('unsplit n', f"{r(un['test_n_mean'], 1)} test images per fold"))
checks.append(('unsplit holm', f"Holm $p = {r(un['paired']['cnxv2t']['p_holm'], 3)}$ and ${r(un['paired']['vits']['p_holm'], 3)}$"))
s90 = V['novid']['selective']['90']
checks.append(('val acc', f"{r(s90['cnxv2t']['val_acc_mean'], 1)}\\% for ConvNeXt V2-Tiny, {r(s90['vits']['val_acc_mean'], 1)}\\% for ViT-S/16 and {r(s90['effb0']['val_acc_mean'], 1)}\\% for EfficientNet-B0, against {r(s90['cnxv2t']['test_acc_mean'], 1)}\\%, {r(s90['vits']['test_acc_mean'], 1)}\\% and {r(s90['effb0']['test_acc_mean'], 1)}\\% on the test folds"))
checks.append(('val meets', f"in {s90['cnxv2t']['runs_val_meets_target']}, {s90['vits']['runs_val_meets_target']} and {s90['effb0']['runs_val_meets_target']} of 40 runs the validation accuracy"))
checks.append(('pooled', f"{r(s90['cnxv2t']['sel_acc_pooled'], 1)}\\% for ConvNeXt V2-Tiny, {r(s90['vits']['sel_acc_pooled'], 1)}\\% for ViT-S/16 and {r(s90['effb0']['sel_acc_pooled'], 1)}\\% for EfficientNet-B0, because"))
# hand-typed claims in the discussion and checklist
lat = {c: V['cost'][c]['cpu_ms_1thr_median'] for c in BB}
checks += [
    ('disc time', f"{r(lat['cnxv2t'] / lat['effb0'], 1)} times and ViT-S/16 {r(lat['vits'] / lat['effb0'], 1)} times"),
    ('disc range', f"in {r(min(lat['mnv3l'], lat['effb0'], lat['mnv4m']), 0)} to {r(max(lat['mnv3l'], lat['effb0'], lat['mnv4m']), 0)}\\,ms per image"),
    ('disc 163', f"the largest backbone in {r(lat['cnxv2t'], 0)}\\,ms"),
    ('checklist', f"gave 86\\% for ConvNeXt V2-Tiny and ViT-S/16, and fell below 90\\% in {min(x90['cnxv2t']['runs_below_target'], x90['vits']['runs_below_target'])}--{max(x90['cnxv2t']['runs_below_target'], x90['vits']['runs_below_target'])} of 40 runs"),
    ('checklist time', f"the two backbones needed up to {r(lat['cnxv2t'] / lat['effb0'], 1)} times the CPU time"),
    ('conc time', f"{r(lat['vits'] / lat['effb0'], 1)} to {r(lat['cnxv2t'] / lat['effb0'], 1)} times the CPU time"),
    ('disc 8or9', f"in {min(x90['cnxv2t']['runs_all_accepted'], x90['vits']['runs_all_accepted'])} or {max(x90['cnxv2t']['runs_all_accepted'], x90['vits']['runs_all_accepted'])} runs of 40"),
]
assert round(x90['cnxv2t']['sel_acc_mean_of_runs'][0]) == 86 and round(x90['vits']['sel_acc_mean_of_runs'][0]) == 86
assert V['novid']['aurc_top2']['cnxv2t']['acc_at_70'] > 90 and V['novid']['aurc_top2']['vits']['acc_at_70'] > 90

bad = [(k, v) for k, v in checks if v not in TEX]
for k, v in bad:
    print(f'MISSING  {k}: {v}')
print(f'{len(checks) - len(bad)}/{len(checks)} checks passed')
sys.exit(1 if bad else 0)
