#!/usr/bin/env python3
"""
build_supplement.py — inventory results/ and generate the supplementary tables
=============================================================================
Stage 1 of filling Supplementary Tables S1-S13.

Run it from the repository root:

    python build_supplement.py --results results --out supplement_generated

It does two things:

  INVENTORY   For every CSV under results/, prints the shape and the column
              names. This is the part to send back if a table comes out wrong:
              the generator below matches columns by name, and the names in
              this repository were not written with that in mind.

  GENERATE    Builds every supplementary table it can from what it finds, as
              Markdown, into --out. Tables it cannot build are listed with the
              script that produces the missing input, so the gap report is a
              to-do list rather than an error log.

Nothing is overwritten in results/. The script only reads.
"""

import argparse
import re
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    raise SystemExit("pandas missing.  pip install pandas")

SEEDS = ["42", "1", "2", "3"]
CONDS = ["group", "image"]

ARCH_ORDER = ["EfficientNetV2-S", "ResNet-101", "ConvNeXt-Tiny", "EfficientNet-B0 + LFA",
              "VGG-16", "ResNet-50", "MobileNetV3-Large", "EfficientNet-B0", "MobileNetV2",
              "ShuffleNetV2-1.0x"]


# --------------------------------------------------------------------- utils
def md_table(df, floatfmt="{:.1f}", index=False):
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    cols = ([d.index.name or ""] if index else []) + [str(c) for c in d.columns]
    out = ["| " + " | ".join(cols) + " |",
           "| " + " | ".join("---" for _ in cols) + " |"]
    for idx, row in d.iterrows():
        cells = ([str(idx)] if index else []) + [str(v) for v in row]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def find_col(df, *keywords, exclude=()):
    """First column whose name contains all keywords and none of exclude."""
    for c in df.columns:
        lc = str(c).lower()
        if all(k in lc for k in keywords) and not any(x in lc for x in exclude):
            return c
    return None


def read(p):
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(f"    !! could not read {p.name}: {e}")
        return None


# ----------------------------------------------------------------- inventory
def inventory(root: Path):
    print("=" * 78)
    print("INVENTORY")
    print("=" * 78)
    seen = {}
    for p in sorted(root.rglob("*.csv")):
        df = read(p)
        if df is None:
            continue
        rel = p.relative_to(root)
        seen.setdefault(p.name, []).append(rel)
        print(f"\n  {rel}")
        print(f"    shape {df.shape[0]} x {df.shape[1]}")
        print(f"    columns: {list(df.columns)}")
        if len(df) <= 4:
            print(df.to_string(index=False, max_colwidth=28))
        else:
            print(df.head(2).to_string(index=False, max_colwidth=28))

    print("\n" + "-" * 78)
    print("  files by name, and which run directories have them:")
    for name, paths in sorted(seen.items()):
        dirs = sorted({str(p.parent) for p in paths})
        print(f"    {name:<28} {len(paths):>2} copies   {', '.join(dirs[:8])}"
              + (" ..." if len(dirs) > 8 else ""))
    print()
    return seen


# ------------------------------------------------------------------ builders
def collect(root: Path, filename: str):
    """{(cond, seed): DataFrame} for every run directory holding `filename`."""
    got = {}
    for cond in CONDS:
        for sd in SEEDS:
            p = root / f"{cond}_s{sd}" / filename
            if p.is_file():
                df = read(p)
                if df is not None:
                    got[(cond, sd)] = df
    return got


def table_s4(root):
    """Per-seed macro F1, both conditions, every architecture."""
    got = collect(root, "comparison_table.csv")
    if not got:
        return None, "comparison_table.csv not found in any run directory"

    sample = next(iter(got.values()))
    mcol = find_col(sample, "macro") or find_col(sample, "f1")
    ncol = find_col(sample, "model") or find_col(sample, "arch") or sample.columns[0]
    if mcol is None:
        return None, f"no macro-F1 column in comparison_table.csv; columns are {list(sample.columns)}"

    rows = {}
    for (cond, sd), df in got.items():
        for _, r in df.iterrows():
            rows.setdefault(str(r[ncol]), {})[f"{cond} s{sd}"] = float(r[mcol])
    out = pd.DataFrame(rows).T
    order = [f"{c} s{s}" for c in CONDS for s in SEEDS]
    out = out[[c for c in order if c in out.columns]]
    for c in CONDS:
        cols = [f"{c} s{s}" for s in SEEDS if f"{c} s{s}" in out.columns]
        if cols:
            out[f"{c} mean"] = out[cols].mean(axis=1)
            out[f"{c} s.d."] = out[cols].std(axis=1, ddof=1)
    if "group mean" in out and "image mean" in out:
        out["inflation (pp)"] = out["image mean"] - out["group mean"]
        out = out.sort_values("group mean", ascending=False)
    out.index.name = "Model"
    return out, None


def table_s5(root):
    """Grouped and image-level cross-validation, per seed and per fold."""
    got = collect(root, "cv_results.csv")
    if not got:
        return None, "cv_results.csv not found in any run directory"
    frames = []
    for (cond, sd), df in got.items():
        d = df.copy()
        d = d[~d.iloc[:, 0].astype(str).str.contains("Mean", case=False, na=False)]
        d["seed"] = sd
        d["condition"] = "session-level" if cond == "group" else "image-level"
        cols = ["condition", "seed"] + [c for c in d.columns if c not in ("condition", "seed")]
        frames.append(d[cols])
    out = pd.concat(frames, ignore_index=True)

    mcol = find_col(out, "macro")
    if mcol:
        f1 = pd.to_numeric(out[mcol], errors="coerce")
        for cond in ("session-level", "image-level"):
            per_seed = f1[out.condition == cond].groupby(out.seed[out.condition == cond]).mean()
            if len(per_seed):
                print(f"      {cond:<14} {per_seed.mean():5.1f} +/- {per_seed.std(ddof=1):.1f} "
                      f"over {len(per_seed)} seeds; fold range "
                      f"{f1[out.condition == cond].min():.1f}-{f1[out.condition == cond].max():.1f}")
    return out, None


def table_s6(root):
    """Attention ablation, per run."""
    got = collect(root, "ablation_results.csv")
    if not got:
        return None, "ablation_results.csv not found in any run directory"
    frames = []
    for (cond, sd), df in got.items():
        if cond != "group":
            continue
        d = df.copy()
        d.insert(0, "seed", sd)
        frames.append(d)
    if not frames:
        return None, "ablation_results.csv found only for the image-level condition"
    return pd.concat(frames, ignore_index=True), None


def table_s6b(root):
    got = collect(root, "mcnemar_results.csv")
    if not got:
        return None, "mcnemar_results.csv not found"
    frames = []
    for (cond, sd), df in got.items():
        d = df.copy()
        d.insert(0, "seed", sd)
        d.insert(0, "condition", cond)
        frames.append(d)
    return pd.concat(frames, ignore_index=True), None


def table_s13(root):
    """Per-class metrics, every run -- the input for the four-class recomputation."""
    got = collect(root, "per_class_metrics.csv")
    if not got:
        return None, "per_class_metrics.csv not found"

    sample = next(iter(got.values()))
    ccol = find_col(sample, "class") or sample.columns[0]
    fcol = find_col(sample, "f1")
    if fcol is None:
        return None, f"no F1 column in per_class_metrics.csv; columns are {list(sample.columns)}"

    rows = []
    for (cond, sd), df in got.items():
        d = df[~df[ccol].astype(str).str.contains("avg|macro|accuracy|weighted", case=False)]
        f1 = pd.to_numeric(d[fcol], errors="coerce")
        if f1.max() <= 1.0:
            f1 = f1 * 100
        five = f1.mean()
        mask = ~d[ccol].astype(str).str.contains("pink", case=False)
        four = f1[mask].mean()
        rows.append({"condition": "session-level" if cond == "group" else "image-level",
                     "seed": sd,
                     "macro F1, five classes": five,
                     "macro F1, four classes (pink excluded)": four,
                     "difference (pp)": four - five})
    out = pd.DataFrame(rows).sort_values(["condition", "seed"])
    return out, None


def table_vietnam(root):
    """WARNING. vietnam_validation.csv holds the PRE-CORRECTION cross-region
    numbers: macro-averaged over the union of true and predicted labels, which
    is the scikit-learn default. Sect. 3.7 of the manuscript explains why that
    convention is wrong here -- every architecture predicts pink disease on the
    Vietnamese data, so the denominator becomes five instead of four and every
    value falls by about 20%. Tables 10 and 11 report the corrected figures,
    recomputed by fix_vietnam_metrics.py over an explicit label set.

    Example: this file gives ConvNeXt-Tiny 29.7 at seed 42; Table 11 (four
    classes) reports 36.7 and Table 10 (three foliar classes) reports 48.9.

    DO NOT paste this table into the supplement as it stands.
    """
    got = collect(root, "vietnam_validation.csv")
    if not got:
        return None, "vietnam_validation.csv not found"
    frames = []
    for (cond, sd), df in got.items():
        d = df.copy()
        d.insert(0, "seed", sd)
        d.insert(0, "condition", cond)
        frames.append(d)
    return pd.concat(frames, ignore_index=True), None


def table_robustness(root):
    got = collect(root, "robustness_results.csv")
    if not got:
        return None, "robustness_results.csv not found"
    frames = []
    for (cond, sd), df in got.items():
        d = df.copy()
        d.insert(0, "seed", sd)
        d.insert(0, "condition", cond)
        frames.append(d)
    return pd.concat(frames, ignore_index=True), None


# ----------------------------------------------------------------------- main
BUILDERS = [
    ("S4",  "Per-seed macro F1, both partition conditions, nine architectures", table_s4,  True),
    ("S5",  "Cross-validation, per seed and per fold", table_s5, False),
    ("S6",  "Attention ablation, per-run values", table_s6, False),
    ("S6b", "Statistics for the ablation (descriptive only; see Sect. S10.1)", table_s6b, False),
    ("S13", "Five-class against four-class macro averaging, every run", table_s13, False),
    ("Sx-vietnam-UNCORRECTED", "Cross-region validation, per run -- PRE-CORRECTION, see the warning in the file", table_vietnam, False),
    ("Sx-robust", "Perturbation results, per run (input for Table 9)", table_robustness, False),
]

NEEDS_A_RUN = {
    "S1":  ("Duplicate and near-duplicate audit, Malaysian dataset",
            "python audit_dataset.py --root <images_512> --out audit   "
            "(then python fill_table_s1.py)"),
    "S2":  ("Audit of the external Vietnamese datasets",
            "python audit_public.py ... ; python classify_redundancy.py --manifest ..."),
    "S3":  ("Matched centre-crop control",
            "python make_centrecrop.py ... ; retrain 3 architectures ; "
            "python fix_crop_control.py"),
    "S7":  ("Pairwise session overlap between every pair of splits",
            "python session_split.py --root <images_512> --manifest sessions.csv "
            "(report the overlap block for both partitions)"),
    "S8":  ("EXIF survey of the 1,120 released image files",
            "python check_provenance.py --root <release_root>"),
    "S9":  ("Camera sequence numbers appearing under more than one class",
            "python label_conflicts.py --root <images_512> --sessions sessions.csv"),
    "S11": ("Full threshold sweeps for all content-based groupings",
            "python validate_proxy.py ... ; python embed_proxy.py --sims 0.70,...,0.95 ..."),
    "S12": ("Per-test-image contingency between reference and proxy flags",
            "python leak_detection.py --sessions sessions.csv ..."),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="supplement_generated")
    ap.add_argument("--no-inventory", action="store_true")
    a = ap.parse_args()

    root = Path(a.results)
    if not root.is_dir():
        raise SystemExit(f"no such directory: {root}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if not a.no_inventory:
        inventory(root)

    print("=" * 78)
    print("GENERATE")
    print("=" * 78)

    built, failed = [], []
    for tag, title, fn, use_index in BUILDERS:
        try:
            df, err = fn(root)
        except Exception as e:
            df, err = None, f"{type(e).__name__}: {e}"
        if df is None or len(df) == 0:
            print(f"  {tag:<12} SKIPPED   {err}")
            failed.append((tag, title, err))
            continue
        head = f"## Table {tag}. {title}\n\n"
        if "UNCORRECTED" in tag:
            head += ("> **Do not use as it stands.** These are the pre-correction cross-region\n"
                     "> values, macro-averaged over the union of true and predicted labels\n"
                     "> (the scikit-learn default). Because every architecture predicts pink\n"
                     "> disease on the Vietnamese data, the denominator becomes five instead\n"
                     "> of four and every value falls by roughly 20%. Tables 10 and 11 of the\n"
                     "> manuscript report the corrected figures from `fix_vietnam_metrics.py`.\n"
                     "> Example: ConvNeXt-Tiny is 29.7 here, 36.7 in Table 11, 48.9 in Table 10.\n\n")
        body = head + md_table(df, index=use_index) + "\n"
        (out / f"Table_{tag}.md").write_text(body, encoding="utf-8")
        print(f"  {tag:<12} built     {df.shape[0]} rows x {df.shape[1]} cols "
              f"-> {out / f'Table_{tag}.md'}")
        built.append((tag, title))

    gaps = out / "GAPS.md"
    lines = ["# Supplementary tables still to generate", ""]
    if failed:
        lines += ["## Builders that ran but found nothing", ""]
        for tag, title, err in failed:
            lines += [f"**Table {tag} — {title}**", "", f"    {err}", ""]
    lines += ["## Tables that need a script run, not an aggregation", ""]
    for tag, (title, cmd) in NEEDS_A_RUN.items():
        lines += [f"**Table {tag} — {title}**", "", f"    {cmd}", ""]
    gaps.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n  gap report -> {gaps}")
    print(f"\n  {len(built)} table(s) built, {len(failed)} builder(s) found nothing, "
          f"{len(NEEDS_A_RUN)} table(s) need a script run.")
    print("\n  If a generated table looks wrong, send back the INVENTORY block above:")
    print("  the builders match columns by name and the names were not written for that.")


if __name__ == "__main__":
    main()
