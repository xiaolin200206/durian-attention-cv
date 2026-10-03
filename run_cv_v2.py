"""Follow-up experiment: LFA at earlier insertion points and a higher input resolution (200 runs).

    python run_cv_v2.py --data path/to/data_original.zip --out results/v2 [--quick]

Same 20 grouped folds (data/folds.csv), same two seeds and the same training schedule as run_cv.py.
Five configurations: the plain network at 224 px (rerun), LFA after features[5] (14 x 14) and after
features[3] (28 x 28) at 224 px, and the plain network and LFA after features[5] at 320 px.
The decision rule was written to results/v2/PREREG.md before the runs (see notebooks/lfa_v2_experiment.ipynb).
"""
import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import common as C

# name: (attention, insertion point, input size)
CONFIGS_V2 = {
    'none_224': ('none', None, 224),
    'lfa14_224': ('lfa', 5, 224),
    'lfa28_224': ('lfa', 3, 224),
    'none_320': ('none', None, 320),
    'lfa14_320': ('lfa', 5, 320),
}
PHASES = [['none_224', 'lfa14_224', 'lfa28_224'], ['none_320', 'lfa14_320']]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default='results/v2')
    ap.add_argument('--cache', default='results/cache_512.pkl')
    ap.add_argument('--folds', default='data/folds.csv')
    ap.add_argument('--quick', action='store_true', help='1+1 epochs, 2 folds, no pretrained weights (smoke test)')
    ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cache = C.load_images(a.data, a.cache)
    meta, classes = C.build_meta(cache)
    folds = C.make_folds(meta, a.folds)
    C.check_folds(meta, folds)

    res_csv, pred_csv = out / 'results.csv', out / 'preds.csv'
    done = set()
    if res_csv.exists():
        r0 = pd.read_csv(res_csv)
        done = set(zip(r0.config, r0.repeat, r0.fold, r0.seed))
    keys = sorted(folds.groupby(['repeat', 'fold']).groups)[: 2 if a.quick else None]
    jobs = [(c, rep, k, s) for ph in PHASES for (rep, k) in keys for s in C.SEEDS for c in ph]
    todo = [j for j in jobs if j not in done]
    epochs = (1, 1) if a.quick else (C.EPOCHS_S1, C.EPOCHS_S2)
    print(f'{len(jobs)} runs, {len(jobs) - len(todo)} done, {len(todo)} to go')
    for cfg, rep, k, s in todo:
        att, pos, size = CONFIGS_V2[cfg]
        f = folds[(folds.repeat == rep) & (folds.fold == k)]
        tr, va, te = (f[f.role == r].idx.values for r in ['train', 'val', 'test'])
        t0 = time.time()
        model, vf1 = C.train_model(att, C.to_pil(cache, meta, tr), meta.y.values[tr],
                                   C.to_pil(cache, meta, va), meta.y.values[va], len(classes), s,
                                   epochs=epochs, pretrained=not a.quick, workers=a.workers,
                                   position=pos, size=size)
        _, te_tf = C.make_transforms(size)
        loader = C.DataLoader(C.ImageSet(C.to_pil(cache, meta, te), meta.y.values[te], te_tf),
                              batch_size=64, shuffle=False, num_workers=a.workers)
        logits, yt = C.predict_logits(model, loader)
        yp = logits.argmax(1)
        present = sorted(set(yt))
        row = {'config': cfg, 'repeat': rep, 'fold': k, 'seed': s, 'n_train': len(tr), 'n_val': len(va),
               'n_test': len(te), 'val_macro_f1': round(vf1 * 100, 2),
               'test_macro_f1_present': round(f1_score(yt, yp, labels=present, average='macro', zero_division=0) * 100, 2),
               'test_acc': round(accuracy_score(yt, yp) * 100, 2), 'minutes': round((time.time() - t0) / 60, 2)}
        pd.DataFrame([row]).to_csv(res_csv, mode='a', header=not res_csv.exists(), index=False)
        pd.DataFrame({'config': cfg, 'repeat': rep, 'fold': k, 'seed': s, 'file': meta.file.values[te],
                      'cls': meta.cls.values[te], 'y_true': yt, 'y_pred': yp}).to_csv(
            pred_csv, mode='a', header=not pred_csv.exists(), index=False)
        print(f"{cfg:<10} rep{rep} fold{k} seed{s}  macro F1 {row['test_macro_f1_present']:5.1f}  ({row['minutes']:.1f} min)", flush=True)


if __name__ == '__main__':
    main()
