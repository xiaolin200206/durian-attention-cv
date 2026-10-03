# Lightweight attention for field durian disease classification: grouped, repeated cross-validation

Code and results for the paper *Seed-to-seed variation outweighs lightweight attention gains: a grouped, repeated cross-validation study on field durian disease images* (Lin Ding Shan, submitted to *Pattern Recognition Letters*).

Every number, table and figure in the manuscript is generated from the files in `results/`, not typed: `paper/prl/render_prl.py` fills the manuscript template from `results/numbers.json`, and `paper/prl/verify_prl.py` recomputes every reported quantity independently from the raw per-image predictions and fails, naming the item, on any disagreement (87 checks).

The study compares EfficientNet-B0 with and without three attention modules (LFA, a 1,281-parameter spatial gate; squeeze-and-excitation; CBAM) on 550 field images of four durian disease classes. Images are split by capture group (camera-counter bursts), and no group is split between training and test data. Each configuration is trained on 20 folds (4 folds × 5 repeats) with two fixed seeds, and configurations are compared in pairs with the corrected resampled *t*-test of Nadeau and Bengio (2003).

**Result in one line:** no module measurably improved the plain network (LFA − none = −0.13 percentage points of macro F1, 95% CI −4.56 to 4.31), while changing only the random seed moved macro F1 by 5.34 points on average. A pre-registered follow-up (200 runs) moved LFA to earlier layers (14 × 14 and 28 × 28 feature maps) and raised the input to 320 px; none of these met the decision rule either (`results/v2/`).

## Data

- Images and capture-session identifiers: Zenodo, https://doi.org/10.5281/zenodo.22177133. Use the full-resolution images in class folders together with `sessions.csv`; the scripts resize them exactly as in the paper.
- `data/sessions.csv`: class, file name and capture session for each of the 560 images.
- `data/folds.csv`: the fold assignment used in the paper (`repeat`, `fold`, row index into the 550-image table, `role` = train / val / test). Regenerating it with other versions of scikit-learn or pandas can give a different partition, so this file is the reference.
- External test: the Vietnamese durian leaf dataset of Nguyen Thanh et al. (2025), *Data in Brief* 61, 111845, available from its authors.

## Files

| File | Purpose |
|---|---|
| `common.py` | Data loading, capture-group construction, folds, models, seeded training |
| `run_cv.py` | The 160 cross-validation runs (resumable) |
| `run_cv_v2.py` | The 200-run follow-up: LFA at earlier insertion points and 320 px input (resumable) |
| `run_external.py` | Training on all 550 images and zero-shot test on the Vietnamese dataset |
| `measure_cost.py` | Parameters, MACs and CPU latency |
| `analyze.py` | Every number, table and figure in the paper, from the saved predictions (no GPU) |
| `analyze_v2.py` | Table 6 (follow-up experiment) from `results/v2/results.csv` |
| `make_supplement.py` | Supplementary tables |
| `make_fig1.py` | Fig. 1 (schematic) |
| `notebooks/` | The Colab notebooks that produced the reported results |
| `results/` | Raw outputs: `results.csv`, `preds.csv` (every test prediction), `latency_params.csv`, `external_vietnam.csv`, `ablation_replicates_preliminary.csv`, and the derived `numbers.json`, `run_metrics.csv`, `per_class_metrics.csv`, `sensitivity_grouping.csv`; `results/v2/` holds the follow-up experiment with its pre-registered decision rule |
| `figures/` | Figures of the analysis scripts |
| `paper/prl/` | The manuscript: `prl_template.tex` + `render_prl.py` → `manuscript_prl.tex/.pdf`; `make_figs_prl.py` (figures and graphical abstract), `make_supplement_prl.py` (supplementary tables S1–S9), `verify_prl.py` (87 checks), `refs.bib`, highlights, cover letter |

## Reproduce

```bash
pip install -r requirements.txt

# numbers, tables and figures from the saved predictions (minutes, CPU only)
python analyze.py
python make_supplement.py > supplementary_tables.md
python analyze_v2.py

# full experiments (GPU recommended; about 4-5 h for run_cv.py on one GPU)
python run_cv.py --data path/to/images_or_archive --out results --folds data/folds.csv
python run_external.py --data path/to/images_or_archive --vietnam path/to/vietnam_archive --out results
python measure_cost.py --out results
python run_cv_v2.py --data path/to/images_or_archive --out results/v2   # about 10 h
```

`run_cv.py --quick` runs a two-fold smoke test with one epoch per stage and no pretrained weights.

## Note on the video frames

The 63 video frames (seven *Phomopsis* videos) were grouped per video and were not passed through the burst rule, although the video files carry camera counter numbers next to still photographs of the largest burst. A video can therefore fall in a different fold from stills taken just before or after it, which inflates the absolute scores, mainly for *Phomopsis*. `analyze.py` reports two sensitivity analyses (`results/sensitivity_grouping.csv`, Supplementary Table S7): scoring without the video frames, and scoring only test images whose group would not have been split if videos had been linked (`common.build_groups(..., link_videos=True)`). The paired comparisons between configurations change little; the absolute macro F1 falls from about 75% to about 66%. The released folds and results are those of the runs as performed (`link_videos=False`).

## Notes on the metric

Macro F1 is computed over the classes present in each test fold. In 3 of the 20 folds no root and collar rot image reaches the test set. The notebooks in `notebooks/` stored a macro F1 averaged over all four classes, which scores the absent class as zero in those folds; `analyze.py` recomputes the metric from `preds.csv`. Both versions are reported in the paper (main text and Supplementary Table S4); the paired comparisons are almost identical.

## Licence and citation

Code: MIT licence (see `LICENSE`). Please cite the paper and the dataset (see `CITATION.cff`).
