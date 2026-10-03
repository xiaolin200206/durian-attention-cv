"""Zero-shot test on the Vietnamese durian leaf dataset (Nguyen Thanh et al., 2025).

    python run_external.py --data data_original.zip --vietnam "Vietnam dataset.zip" --out results

Each configuration is trained on all 550 Malaysian images (five seeds) and scored, without
adaptation, on the three foliar classes that correspond: Leaf_Algal -> Algal,
Leaf_Blight and Leaf_Colletotrichum -> Leaf_rot, Leaf_Phomopsis -> Phomopsis.
"""
import argparse
import hashlib
import io
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score

import common as C

VN_MAP = {'leaf_algal': 'Algal', 'leaf_blight': 'Leaf_rot', 'leaf_colletotrichum': 'Leaf_rot',
          'leaf_phomopsis': 'Phomopsis'}


def load_vietnam(archive):
    ims, ys, seen = [], [], set()
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(archive)
        if src.is_file():
            C._extract(src, tmp)
            root = Path(tmp)
        else:
            root = src
        for p in sorted(root.rglob('*')):
            if not p.is_file() or p.suffix.lower() not in C.IMG_EXT:
                continue
            folder = next((x for x in reversed(p.relative_to(root).parts[:-1]) if x.lower() in VN_MAP), None)
            if folder is None:
                continue
            raw = p.read_bytes()
            h = hashlib.md5(raw).hexdigest()
            if h in seen:          # identical files duplicated across the dataset's own splits
                continue
            seen.add(h)
            ims.append(Image.open(io.BytesIO(C.shrink_like_paper(raw))).convert('RGB'))
            ys.append(VN_MAP[folder.lower()])
    return ims, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--vietnam', required=True)
    ap.add_argument('--out', default='results')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = C.load_images(a.data, out / 'cache_512.pkl')
    meta, classes = C.build_meta(cache)
    vn_ims, vn_cls = load_vietnam(a.vietnam)
    vn_y = np.array([classes.index(c) for c in vn_cls])
    print(f'Vietnam: {len(vn_y)} images', pd.Series(vn_cls).value_counts().to_dict())
    itr, iva = C.pick_inner_val(meta, 2026)
    mapped = [classes.index(c) for c in ['Algal', 'Leaf_rot', 'Phomopsis']]
    loader = C.DataLoader(C.ImageSet(vn_ims, vn_y, C.test_tf), batch_size=64, shuffle=False, num_workers=2)
    csv = out / 'external_vietnam.csv'
    done = set(zip(*pd.read_csv(csv)[['config', 'seed']].values.T)) if csv.exists() else set()
    for s in a.seeds:
        for cfg in C.CONFIGS:
            if (cfg, s) in done:
                continue
            t0 = time.time()
            model, vf1 = C.train_model(cfg, C.to_pil(cache, meta, itr), meta.y.values[itr],
                                       C.to_pil(cache, meta, iva), meta.y.values[iva], len(classes), s,
                                       epochs=(1, 1) if a.quick else (C.EPOCHS_S1, C.EPOCHS_S2), pretrained=not a.quick)
            logits, _ = C.predict_logits(model, loader)
            yp_full = logits.argmax(1)
            masked = logits.copy()
            masked[:, [i for i in range(len(classes)) if i not in mapped]] = -1e9
            row = {'config': cfg, 'seed': s, 'val_macro_f1': round(vf1 * 100, 2), 'n_vietnam': len(vn_y),
                   'vn_macro_f1_full': round(f1_score(vn_y, yp_full, labels=mapped, average='macro', zero_division=0) * 100, 2),
                   'vn_macro_f1_3class': round(f1_score(vn_y, masked.argmax(1), labels=mapped, average='macro', zero_division=0) * 100, 2),
                   'vn_pred_as_root_pct': round(float((yp_full == classes.index('Root_disease')).mean() * 100), 1),
                   'minutes': round((time.time() - t0) / 60, 2)}
            pd.DataFrame([row]).to_csv(csv, mode='a', header=not csv.exists(), index=False)
            print(row, flush=True)


if __name__ == '__main__':
    main()
