# Field durian disease images: grouped, repeated cross-validation of an EfficientNet-B0 classifier

Code, fold assignment, per-image predictions and analysis scripts for a study of how reliably durian (*Durio zibethinus*) diseases can be recognised from field photographs when photographs of one capture event are never split between training and test data.

Author: Lin Ding Shan, independent researcher, Malaysia (ORCID [0009-0009-6031-8479](https://orcid.org/0009-0009-6031-8479)). The accompanying paper is under review; please cite the dataset and this repository (see `CITATION.cff`) until it is published.

## What the study does

- **Data.** 550 field photographs of four durian disease classes (algal leaf spot, leaf blight complex, *Phomopsis* leaf spot, root and collar rot) taken in commercial orchards in Negeri Sembilan, Peninsular Malaysia, between July 2025 and June 2026. Every image is assigned to a capture group reconstructed from camera counters, so near-identical photographs of one specimen stay together.
- **Evaluation.** EfficientNet-B0 (ImageNet weights) evaluated with stratified group k-fold cross-validation: 4 folds × 5 repeats = 20 test folds, two fixed seeds per fold, configurations compared in pairs with the corrected resampled *t*-test of Nadeau and Bengio (2003), plus equivalence tests against ±2 percentage points.
- **Comparisons.** The plain network against three lightweight attention modules: a single-channel spatial gate (**SG**, 1,281 parameters; called `lfa` / LFA in the code and result files), squeeze-and-excitation (SE) and CBAM. A follow-up experiment (200 runs) moves the SG to earlier layers and raises the input to 320 px.
- **External test.** Models trained on the Malaysian images are tested without adaptation on the Vietnamese durian leaf dataset of Nguyen Thanh et al. (2025).

## Main results

Primary scoring: the 63 *Phomopsis* video frames excluded (see *Note on the video frames*).

| Quantity | Value |
|---|---|
| Macro F1 of the plain network, 20 grouped folds × 2 seeds | 66.6 % (75.1 % on all test images) |
| Mean macro F1 of a single test fold | 42.8 % to 90.6 % |
| Mean change in macro F1 from changing only the random seed | 5.7 points |
| *Phomopsis* recall, still photographs vs. video frames | 41.5 % vs. 95.7 % |
| SG − plain network (95 % CI) | −0.13 (−4.55 to 4.28) points |
| SE − plain network | −1.26 (−10.28 to 7.76) points |
| CBAM − plain network | −0.23 (−5.83 to 5.37) points |
| Smallest difference resolvable with any number of repeated folds (SG) | about ±3.9 points |
| Macro F1 on the Vietnamese dataset (four configurations) | 39–41 % |

No attention module, insertion point or input size met the decision rule (gain ≥ 2 points with the corrected 95 % CI above zero).

## Repository layout

| Path | Contents |
|---|---|
| `common.py` | Data loading, capture-group construction, folds, models, seeded training |
| `run_cv.py` | The 160 cross-validation runs (resumable) |
| `run_cv_v2.py` | The 200-run follow-up: SG at earlier insertion points and 320 px input (resumable) |
| `run_external.py` | Training on the Malaysian images and the test on the Vietnamese dataset |
| `measure_cost.py` | Parameters, multiply–accumulate operations and CPU latency |
| `analyze.py` | Derived result files (`results/numbers.json`, `run_metrics.csv`, `per_class_metrics.csv`, `sensitivity_grouping.csv`) and `figures/` from the saved predictions |
| `analyze_v2.py` | Follow-up experiment summary (`results/v2/numbers_v2.json`, `table6.md`) |
| `make_fig1.py` | Classifier and SG schematic in `figures/` |
| `paper/compute_numbers.py` | Every number reported in the paper, under both scorings (all test images; video frames excluded) → `paper/numbers.json` |
| `paper/make_figures.py` | Figures of the paper → `paper/figures/` (example photographs in `paper/figures/examples/`) |
| `paper/make_supplement.py` | Supplementary Tables S1–S10 and Figs. S1–S2 → `paper/supplementary.md` / `.docx` / `.pdf` |
| `paper/check_numbers.py` | Checks every computed number quoted in a manuscript `.tex` file against `paper/numbers.json` and the result files; fails, naming the item, on any disagreement |
| `data/` | `sessions.csv` (class, file name and capture session for all 560 released images) and `folds.csv` (the fold assignment used) |
| `notebooks/` | The Colab notebooks that produced the reported runs |
| `results/` | Raw outputs: `results.csv` (one row per run), `preds.csv` (every test prediction), `latency_params.csv`, `external_vietnam.csv`, `ablation_replicates_preliminary.csv`, and derived files; `results/v2/` holds the follow-up experiment, including `PREREG.md`, the analysis written before its runs |
| `figures/` | Figures written by `analyze.py` and `make_fig1.py` |

## Data

- **Images and capture-session identifiers:** Zenodo, https://doi.org/10.5281/zenodo.22177133 (560 images in five classes; pink disease, ten images, is released but was not used in the experiments). Use the full-resolution images in class folders together with `sessions.csv`; the scripts resize them exactly as in the study.
- **Fold assignment:** `data/folds.csv` (`repeat`, `fold`, row index into the 550-image table, `role` = train / val / test). Regenerating the folds with other versions of scikit-learn or pandas can give a different partition, so this file is the reference.
- **External test:** the Vietnamese durian leaf dataset of Nguyen Thanh et al. (2025), *Data in Brief* 61, 111845, available from its authors.

## Reproduce

```bash
pip install -r requirements.txt

# everything reported, from the saved predictions (minutes, CPU only, no images needed)
python analyze.py
python analyze_v2.py --results results/v2
python make_fig1.py
python paper/compute_numbers.py
python paper/make_figures.py
python paper/make_supplement.py
python paper/check_numbers.py path/to/manuscript.tex     # optional, with the manuscript source

# full experiments (GPU recommended; about 4-5 h for run_cv.py on one GPU)
python run_cv.py --data path/to/images_or_archive --out results --folds data/folds.csv
python run_external.py --data path/to/images_or_archive --vietnam path/to/vietnam_archive --out results
python measure_cost.py --out results
python run_cv_v2.py --data path/to/images_or_archive --out results/v2   # about 10 h
```

`run_cv.py --quick` runs a two-fold smoke test with one epoch per stage and no pretrained weights. Some GPU operations are non-deterministic even with fixed seeds, so a rerun may differ slightly from the released predictions.

## Note on the video frames

The 63 video frames (seven *Phomopsis* videos) were grouped per video but were not passed through the burst rule, although the video files carry camera counter numbers next to still photographs of the largest burst. A video can therefore fall in a different fold from stills taken just before or after it, which inflates the absolute scores, mainly for *Phomopsis*. The paper therefore uses the scoring with the video frames excluded as its primary estimate and reports every comparison on all test images as well. `analyze.py` also scores only test images whose group would not have been split if the videos had been linked (`common.build_groups(..., link_videos=True)`). The released folds and results are those of the runs as performed (`link_videos=False`).

## Notes on the metric

Macro F1 is computed over the classes present in each test fold: in 3 of the 20 folds no root and collar rot image reaches the test set, and with the video frames excluded no *Phomopsis* image does in 4 folds. The notebooks stored a macro F1 averaged over all four classes, which scores an absent class as zero; both versions are reported (Supplementary Table S4), and the paired comparisons are almost identical.

## Licence and citation

Code: MIT licence (see `LICENSE`). Please cite the dataset (doi:10.5281/zenodo.22177133) and this repository (see `CITATION.cff`).
