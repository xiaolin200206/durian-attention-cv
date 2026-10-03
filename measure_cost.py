"""Parameters, multiply-accumulate operations and CPU latency (batch 1, 224 x 224).

    python measure_cost.py --out results

The four models are timed in interleaved rounds (5 rounds x 40 forward passes each, after
30 warm-up passes) so that drift in machine load does not fall on one model only.
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results')
    ap.add_argument('--no-pretrained', action='store_true', help='skip downloading ImageNet weights (timing is unaffected)')
    a = ap.parse_args()
    x = torch.randn(1, 3, 224, 224)
    models = {c: C.build_model(c, 4, pretrained=not a.no_pretrained).eval() for c in C.CONFIGS}
    base = sum(p.numel() for p in models['none'].parameters())
    rows = []
    for c, m in models.items():
        n = sum(p.numel() for p in m.parameters())
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            m(x)
        rows.append({'config': c, 'params_M': round(n / 1e6, 4), 'extra_params': n - base,
                     'GMACs': round(fc.get_total_flops() / 2 / 1e9, 4)})
    for threads in [1, os.cpu_count()]:
        torch.set_num_threads(threads)
        times = {c: [] for c in models}
        with torch.no_grad():
            for m in models.values():
                for _ in range(30):
                    m(x)
            for _ in range(5):
                for c, m in models.items():
                    for _ in range(40):
                        t0 = time.perf_counter()
                        m(x)
                        times[c].append((time.perf_counter() - t0) * 1000)
        for r in rows:
            r[f'cpu_ms_{threads}thr_median'] = round(float(np.median(times[r['config']])), 2)
            r[f'cpu_ms_{threads}thr_p90'] = round(float(np.percentile(times[r['config']], 90)), 2)
    df = pd.DataFrame(rows)
    Path(a.out).mkdir(exist_ok=True)
    df.to_csv(Path(a.out) / 'latency_params.csv', index=False)
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
