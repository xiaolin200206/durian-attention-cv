# Pre-specified analysis, written before any v2 run

This is the text the notebook (`notebooks/lfa_v2_experiment.ipynb`, cell 3) writes to the run folder before the first run; the timestamp line is omitted here.

Folds: runs_phase0/folds.csv (20 grouped folds, as in the first experiment). Seeds 0 and 1.
Training protocol identical to the first experiment except for the two factors below.

Configurations: none_224 (no attention, 224 px); lfa14_224 (LFA after features[5], 224 px);
lfa28_224 (LFA after features[3], 224 px); none_320 (no attention, 320 px); lfa14_320 (LFA after features[5], 320 px).
LFA = 1x1 conv to one channel, sigmoid, element-wise multiplication (default PyTorch init).
features[3] output: 40 ch, stride 8; features[5] output: 112 ch, stride 16.
Resolution s: train RandomResizedCrop(s, scale 0.7-1.0); test Resize(round(s*256/224)) + CenterCrop(s).

Primary metric: macro F1 over classes present in each test fold; seeds averaged within fold.
Primary comparisons (each vs none_224): lfa14_224, lfa28_224, none_320, lfa14_320.
Test: Nadeau-Bengio corrected resampled t-test over 20 folds, factor 1/20 + mean n_test / mean (n_train + n_val).
Multiplicity: Holm over the 4 primary comparisons.
Counted as an improvement only if mean paired difference >= 2 pp AND Holm-adjusted p < 0.05.
Secondary (descriptive): lfa14_320 vs none_320; none_224 here vs 'none' in runs_phase0 (rerun agreement).
All results are reported whatever their direction.
