"""Shared code: data loading, capture-session grouping, fold construction, models and training.

The experiments reported in the paper were run with the two notebooks in notebooks/.
This module contains the same code in importable form; the scripts in the repository root
call it. Changing anything here changes the protocol, so the defaults are the paper's settings.
"""

import copy
import io
import pickle
import random
import re
import shutil
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights

CLASSES_ALL = ['Algal', 'Leaf_rot', 'Phomopsis', 'Pink_disease', 'Root_disease']
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff', '.heic'}
CONFIGS = ['none', 'lfa', 'se', 'cbam']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Protocol settings used in the paper
N_FOLDS, N_REPEATS = 4, 5
SEEDS = [0, 1]
EPOCHS_S1, EPOCHS_S2 = 15, 15
LR_S1, LR_S2 = 1e-3, 1e-5
BATCH = 16
INNER_VAL_FRAC = 0.1
GROUP_GAP = 3


# ----------------------------------------------------------------------------- data
def norm_class(part):
    part = str(part).strip()
    if part.startswith('Leaf_rot'):
        return 'Leaf_rot'
    for c in CLASSES_ALL:
        if part.lower() == c.lower():
            return c
    return None


def shrink_like_paper(raw_bytes, max_side=512, quality=92):
    """Longest edge 512 px, Lanczos, re-encoded as JPEG q92."""
    with Image.open(io.BytesIO(raw_bytes)) as im:
        im.load()
        if im.mode in ('RGBA', 'P', 'LA'):
            im = im.convert('RGB')
        w, h = im.size
        s = max_side / max(w, h)
        if s < 1:
            im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, 'JPEG', quality=quality, optimize=True)
        return buf.getvalue()


def _extract(archive, dst):
    if zipfile.is_zipfile(archive):
        zipfile.ZipFile(archive).extractall(dst)
    elif tarfile.is_tarfile(archive):
        tarfile.open(archive).extractall(dst)
    else:
        raise SystemExit(f'{archive} is neither a zip nor a tar archive')


def load_images(source, cache_file):
    """Read the 560 original images (a folder or an archive), resize them and cache the result.

    Returns {'images': {(class, filename): jpeg bytes}, 'sessions_csv': text}.
    """
    cache_file = Path(cache_file)
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    source = Path(source)
    if source.is_file():
        work = cache_file.parent / '_extracted'
        work.mkdir(parents=True, exist_ok=True)
        _extract(source, work)
        root = work
    else:
        root = source
    sess = sorted(root.rglob('sessions.csv'), key=lambda p: len(p.parts))
    if not sess:
        raise SystemExit('sessions.csv not found next to the images')
    images = {}
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.suffix.lower() not in IMG_EXT:
            continue
        cls = next((norm_class(x) for x in p.relative_to(root).parts[:-1] if norm_class(x)), None)
        if cls:
            images[(cls, p.name)] = shrink_like_paper(p.read_bytes())
    cache = {'images': images, 'sessions_csv': sess[0].read_text(encoding='utf-8-sig')}
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)
    if source.is_file():
        shutil.rmtree(work, ignore_errors=True)
    return cache


RE_SEQ = re.compile(r'^(?P<prefix>[A-Za-z_]*?)(?P<num>\d{3,6})\b')


def build_groups(d, gap=GROUP_GAP, cross_class=True, link_videos=False):
    """Merge camera-counter bursts (counter gap <= gap) into one group, across classes.

    As run for the paper (link_videos=False), frames extracted from a video keep their own
    group (the video session label), even though the video file carries a camera counter
    number (e.g. IMG_9937) next to still photographs. With link_videos=True the video counter
    number takes part in the burst rule, which joins all seven Phomopsis videos to the largest
    group; analyze.py uses this to score only test images whose linked group was not split.
    Messaging batches and unmatched files keep the session label in sessions.csv.
    """
    if link_videos:
        d = d.copy()
        v = d['session'].astype(str).str.startswith('video:')
        num = d.loc[v, 'session'].str.split(':').str[-1]
        d.loc[v, 'session'] = 'single:' + d.loc[v, 'cls'].astype(str) + ':' + num
        d.loc[v, 'file'] = num + '.jpg'
    d = d.copy()
    d['kind'] = d['session'].astype(str).str.split(':').str[0]
    parsed = d['file'].apply(lambda f: RE_SEQ.match(str(f)))
    d['prefix'] = [(m.group('prefix') or 'IMG') if m else None for m in parsed]
    d['num'] = [int(m.group('num')) if m else np.nan for m in parsed]
    seq = d['kind'].isin(['burst', 'single']) & d['num'].notna()
    lab = {}
    key = ['prefix'] if cross_class else ['cls', 'prefix']
    for k, grp in d[seq].groupby(key):
        run, prev = 0, None
        for i, r in grp.sort_values('num').iterrows():
            if prev is not None and r['num'] - prev > gap:
                run += 1
            key_str = k[0] if isinstance(k, tuple) else k
            lab[i] = f'seq:{key_str}:{run}'
            prev = r['num']
    for i, r in d[~seq].iterrows():
        lab[i] = r['session']
    return pd.Series(lab).reindex(d.index)


def build_meta(cache, drop_pink=True):
    meta = pd.read_csv(io.StringIO(cache['sessions_csv']))
    meta['cls'] = meta['cls'].map(norm_class)
    meta['group'] = build_groups(meta)
    if drop_pink:
        meta = meta[meta['cls'] != 'Pink_disease'].reset_index(drop=True)
    classes = sorted(meta['cls'].unique(), key=CLASSES_ALL.index)
    meta['y'] = meta['cls'].map({c: i for i, c in enumerate(classes)})
    missing = [(c, f) for c, f in zip(meta['cls'], meta['file']) if (c, f) not in cache['images']]
    if missing:
        raise SystemExit(f'{len(missing)} files in sessions.csv have no image, e.g. {missing[:3]}')
    return meta, classes


def pick_inner_val(sub, seed):
    """Group-wise inner validation set: of 10 candidate splits, the one closest to 10 % with >= 3 classes."""
    inner = StratifiedGroupKFold(n_splits=round(1 / INNER_VAL_FRAC), shuffle=True, random_state=seed)
    target = INNER_VAL_FRAC * len(sub)
    cands = [(abs(len(v) - target) + (1e6 if sub.iloc[v]['y'].nunique() < 3 else 0), t, v)
             for t, v in inner.split(sub, sub['y'], sub['group'])]
    _, itr, iva = min(cands, key=lambda c: c[0])
    return itr, iva


def make_folds(meta, folds_csv):
    """Return the fold assignment. If folds_csv exists it is used as is.

    data/folds.csv is the partition used for the paper. Regenerating it with other versions of
    scikit-learn or pandas can give a different (equally valid) partition, so the shipped file is
    the reference.
    """
    folds_csv = Path(folds_csv)
    if folds_csv.exists():
        return pd.read_csv(folds_csv)
    rows = []
    for rep in range(N_REPEATS):
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=rep)
        for k, (tr, te) in enumerate(sgkf.split(meta, meta['y'], meta['group'])):
            itr, iva = pick_inner_val(meta.iloc[tr], 1000 + rep * 10 + k)
            role = np.array(['test'] * len(meta), dtype=object)
            role[tr[itr]] = 'train'
            role[tr[iva]] = 'val'
            rows += [(rep, k, i, role[i]) for i in range(len(meta))]
    folds = pd.DataFrame(rows, columns=['repeat', 'fold', 'idx', 'role'])
    folds_csv.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(folds_csv, index=False)
    return folds


def check_folds(meta, folds):
    for (rep, k), f in folds.groupby(['repeat', 'fold']):
        g = {r: set(meta.loc[f[f.role == r].idx, 'group']) for r in ['train', 'val', 'test']}
        assert not (g['train'] & g['test'] or g['val'] & g['test'] or g['train'] & g['val']), (rep, k)


# ----------------------------------------------------------------------------- models
class LesionFocusAttention(nn.Module):
    """LFA: a 1x1 convolution to one channel, a sigmoid, and element-wise re-weighting."""

    def __init__(self, in_channels):
        super().__init__()
        self.attention_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.attention_conv(x))


class SEModule(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super().__init__()
        mid = max(in_ch // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(in_ch, mid, bias=False), nn.ReLU(inplace=True),
                                nn.Linear(mid, in_ch, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        return x * self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)


class CBAMModule(nn.Module):
    def __init__(self, in_ch, reduction=16, ks=7):
        super().__init__()
        mid = max(in_ch // reduction, 1)
        self.avg, self.mx = nn.AdaptiveAvgPool2d(1), nn.AdaptiveMaxPool2d(1)
        self.cfc = nn.Sequential(nn.Linear(in_ch, mid, bias=False), nn.ReLU(inplace=True),
                                 nn.Linear(mid, in_ch, bias=False))
        self.sconv = nn.Conv2d(2, 1, kernel_size=ks, padding=ks // 2, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        ch = self.sig(self.cfc(self.avg(x).view(b, c)) + self.cfc(self.mx(x).view(b, c))).view(b, c, 1, 1)
        x = x * ch
        sp = self.sig(self.sconv(torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], 1)))
        return x * sp


def build_model(attention, n_cls, pretrained=True, position=None):
    """EfficientNet-B0 with an optional attention module.

    position=None inserts the module after the last feature block (1280 x 7 x 7 at 224 px), as in
    the main experiment. An integer k inserts it after m.features[k] (the follow-up experiment used
    k=3, 40 channels at stride 8, and k=5, 112 channels at stride 16).
    """
    m = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    for p in m.parameters():
        p.requires_grad = False
    att = {'lfa': LesionFocusAttention, 'se': SEModule, 'cbam': CBAMModule}.get(attention)
    mod = None
    if att and position is None:
        mod = att(m.features[-1][0].out_channels)
        m.features = nn.Sequential(m.features, mod)
    elif att:
        blocks = list(m.features.children())
        mod = att(blocks[position][-1].out_channels)
        m.features = nn.Sequential(*blocks[:position + 1], mod, *blocks[position + 1:])
    m.classifier = nn.Sequential(nn.Dropout(p=0.3, inplace=True), nn.Linear(m.classifier[1].in_features, n_cls))
    for p in m.classifier.parameters():
        p.requires_grad = True
    if mod is not None:
        for p in mod.parameters():
            p.requires_grad = True
    return m


# ----------------------------------------------------------------------------- training
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
def make_transforms(size=224):
    """Training and evaluation transforms for a square input of `size` pixels (224 in the paper)."""
    train = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomAffine(degrees=20, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.10)),
    ])
    test = transforms.Compose([transforms.Resize(round(size * 256 / 224)), transforms.CenterCrop(size),
                               transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return train, test


train_tf, test_tf = make_transforms(224)


class ImageSet(Dataset):
    def __init__(self, pil_images, labels, tf):
        self.ims, self.ys, self.tf = list(pil_images), list(labels), tf

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, i):
        return self.tf(self.ims[i]), int(self.ys[i])


def seed_everything(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init(_):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


@torch.no_grad()
def predict_logits(model, loader):
    model.eval()
    out, ys = [], []
    for X, y in loader:
        with torch.autocast(device_type='cuda', enabled=(DEVICE == 'cuda')):
            out.append(model(X.to(DEVICE)).float().cpu())
        ys += y.tolist()
    return torch.cat(out).numpy(), np.array(ys)


def macro_f1_present(yt, yp):
    """Macro F1 over the classes present in yt (a small validation set can miss a class)."""
    return f1_score(yt, yp, labels=sorted(set(yt)), average='macro', zero_division=0)


def train_model(attention, pil_train, y_train, pil_val, y_val, n_cls, seed,
                epochs=(EPOCHS_S1, EPOCHS_S2), pretrained=True, workers=2, position=None, size=224):
    """Two-stage transfer learning; the checkpoint with the best inner-validation macro F1 is kept."""
    seed_everything(seed)
    tr_tf, te_tf = make_transforms(size)
    y_train = np.asarray(y_train)
    counts = np.bincount(y_train, minlength=n_cls)
    cw = torch.tensor(len(y_train) / (n_cls * np.maximum(counts, 1)), dtype=torch.float, device=DEVICE)
    sw = (1.0 / np.maximum(counts, 1))[y_train]
    g = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(torch.as_tensor(sw, dtype=torch.double), len(sw), replacement=True, generator=g)
    kw = dict(num_workers=workers, worker_init_fn=_worker_init, pin_memory=(DEVICE == 'cuda'))
    tl = DataLoader(ImageSet(pil_train, y_train, tr_tf), batch_size=BATCH, sampler=sampler, generator=g, **kw)
    vl = DataLoader(ImageSet(pil_val, y_val, te_tf), batch_size=64, shuffle=False, **kw)

    model = build_model(attention, n_cls, pretrained, position).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=cw)
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE == 'cuda'))
    best = [-1.0, None]
    for n_ep, lr, unfreeze in [(epochs[0], LR_S1, False), (epochs[1], LR_S2, True)]:
        if unfreeze:
            model.load_state_dict(best[1])
            for p in model.parameters():
                p.requires_grad = True
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
        for _ in range(n_ep):
            model.train()
            for X, y in tl:
                X, y = X.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', enabled=(DEVICE == 'cuda')):
                    loss = crit(model(X), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            logits, yv = predict_logits(model, vl)
            f1v = macro_f1_present(yv, logits.argmax(1))
            if f1v > best[0]:
                best = [f1v, copy.deepcopy(model.state_dict())]
    model.load_state_dict(best[1])
    return model, best[0]


def to_pil(cache, meta, idx):
    return [Image.open(io.BytesIO(cache['images'][(meta.at[i, 'cls'], meta.at[i, 'file'])])).convert('RGB')
            for i in idx]
