"""Grouped, repeated cross-validation of EfficientNet-B0 with and without attention (160 runs).

    python run_cv.py --data path/to/data_original.zip --out results [--quick]

The data argument is either a folder or an archive containing the original images in class
folders together with sessions.csv (the Zenodo release has this layout). Results are written
after every run, so an interrupted job resumes where it stopped.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default='results')
    ap.add_argument('--quick', action='store_true', help='1+1 epochs, 2 folds, no pretrained weights (smoke test)')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--folds', default='data/folds.csv', help='fold assignment; generated if missing')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cache = C.load_images(a.data, out / 'cache_512.pkl')
    meta, classes = C.build_meta(cache)
    folds = C.make_folds(meta, a.folds)
    C.check_folds(meta, folds)
    print(f'{len(meta)} images, {meta.group.nunique()} groups, classes {classes}')

    res_csv, pred_csv = out / 'results.csv', out / 'preds.csv'
    done = set()
    if res_csv.exists():
        r0 = pd.read_csv(res_csv)
        done = set(zip(r0.config, r0.repeat, r0.fold, r0.seed))
    keys = sorted(folds.groupby(['repeat', 'fold']).groups)[: 2 if a.quick else None]
    phases = [['none', 'lfa'], ['se', 'cbam']]
    jobs = [(c, rep, k, s) for ph in phases for (rep, k) in keys for s in C.SEEDS for c in ph]
    todo = [j for j in jobs if j not in done]
    epochs = (1, 1) if a.quick else (C.EPOCHS_S1, C.EPOCHS_S2)
    for cfg, rep, k, s in todo:
        f = folds[(folds.repeat == rep) & (folds.fold == k)]
        tr, va, te = (f[f.role == r].idx.values for r in ['train', 'val', 'test'])
        t0 = time.time()
        model, vf1 = C.train_model(cfg, C.to_pil(cache, meta, tr), meta.y.values[tr],
                                   C.to_pil(cache, meta, va), meta.y.values[va], len(classes), s,
                                   epochs=epochs, pretrained=not a.quick, workers=a.workers)
        test_loader = C.DataLoader(C.ImageSet(C.to_pil(cache, meta, te), meta.y.values[te], C.test_tf),
                                   batch_size=64, shuffle=False, num_workers=a.workers)
        logits, yt = C.predict_logits(model, test_loader)
        yp = logits.argmax(1)
        present = sorted(set(yt))
        row = {'config': cfg, 'repeat': rep, 'fold': k, 'seed': s, 'n_train': len(tr), 'n_val': len(va),
               'n_test': len(te), 'val_macro_f1': round(vf1 * 100, 2),
               # as stored by the notebooks: averaged over all four classes (an absent class scores 0)
               'test_macro_f1': round(f1_score(yt, yp, labels=list(range(len(classes))), average='macro', zero_division=0) * 100, 2),
               # primary metric of the paper: averaged over the classes present in the test fold
               'test_macro_f1_present': round(f1_score(yt, yp, labels=present, average='macro', zero_division=0) * 100, 2),
               'test_acc': round(accuracy_score(yt, yp) * 100, 2), 'minutes': round((time.time() - t0) / 60, 2)}
        pd.DataFrame([row]).to_csv(res_csv, mode='a', header=not res_csv.exists(), index=False)
        pd.DataFrame({'config': cfg, 'repeat': rep, 'fold': k, 'seed': s, 'file': meta.file.values[te],
                      'cls': meta.cls.values[te], 'y_true': yt, 'y_pred': yp}).to_csv(
            pred_csv, mode='a', header=not pred_csv.exists(), index=False)
        print(f"{cfg:<5} rep{rep} fold{k} seed{s}  macro F1 {row['test_macro_f1']:5.1f}  ({row['minutes']:.1f} min)", flush=True)


if __name__ == '__main__':
    main()
