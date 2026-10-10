# Pre-specified analysis, written 2026-10-09T10:05:51 before any v3 run

Folds: runs_phase0/folds.csv (20 grouped folds, as in the first experiment). Seeds [0, 1].
Training protocol identical to the first experiment (two stages, 15+15 epochs, Adam 0.001 then 1e-05,
batch 16, class-weighted loss and sampler, same augmentation, checkpoint by inner-validation macro F1), except:
- backbone (below); classification head Dropout(0.3) + Linear for every backbone; input 224 px for every backbone
  (MobileNetV4-Conv-Medium was pretrained at 256 px but is fine-tuned and tested at 224 px like the others);
- per-backbone normalisation from timm's pretrained config (torchvision ImageNet values for effb0);
- training DataLoader with drop_last=True (MobileNetV4 has a BatchNorm in its head); effb0 is rerun under the same setting.

Configurations: {'effb0': ('torchvision', 'efficientnet_b0'), 'mnv3l': ('timm', 'mobilenetv3_large_100.ra_in1k'), 'mnv4m': ('timm', 'mobilenetv4_conv_medium.e500_r256_in1k'), 'cnxv2t': ('timm', 'convnextv2_tiny.fcmae_ft_in22k_in1k'), 'vits': ('timm', 'vit_small_patch16_224.augreg_in21k_ft_in1k')}

Primary metric: macro F1 over classes present in each test fold, seeds averaged within fold.
Primary scoring: the 63 Phomopsis video frames excluded; all test images reported as secondary.
Primary comparisons (each vs effb0): ['mnv3l', 'mnv4m', 'cnxv2t', 'vits'].
Test: Nadeau-Bengio corrected resampled t-test over 20 folds, factor 1/20 + mean n_test / mean (n_train + n_val).
Multiplicity: Holm over the 4 primary comparisons.
Counted as an improvement only if mean paired difference >= 2 pp AND Holm-adjusted p < 0.05.

Decision-support (selective prediction) analysis, per run:
- confidence = maximum softmax probability;
- the threshold is chosen on the inner-validation set only: the smallest confidence threshold at which the accepted
  validation images reach the target accuracy ([0.9, 0.95]); if no threshold reaches it, every image is deferred;
- on the test fold we report coverage (share decided by the model), accuracy on the accepted images, and the share of each
  class deferred to the grower; also the area under the risk-coverage curve (AURC) and top-2 accuracy;
- summaries are means and SDs over the 40 runs of each backbone; no significance test is planned for these.
Secondary (descriptive): effb0 here vs 'none' in runs_phase0 (rerun agreement); parameters, MACs-free CPU latency.
All results are reported whatever their direction.
