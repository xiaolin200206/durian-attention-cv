# Backbone and referral experiment

Produced by `notebooks/backbone_decision_v3.ipynb` on a Colab Tesla T4 (library versions in `versions.txt`), with the same 20 grouped folds
(`data/folds.csv`), seeds 0 and 1 and training schedule as the main experiment. Files:

- `PREREG.md`: the analysis (backbones, decision rule, referral procedure), written before the first run.
- `results.csv`: one row per run (200 rows: 5 backbones x 20 folds x 2 seeds), including the macro F1 stored during training.
- `preds.csv`: every validation (`role = val`) and test (`role = test`) prediction with the four softmax probabilities.
- `cost.csv`: parameters and single-thread CPU latency (median and 90th percentile of 50 passes) in the same Colab session.
- `versions.txt`: library and GPU versions.

Derived numbers: `python paper/compute_numbers_v3.py` -> `paper/numbers_v3.json`.
Backbones (config names): `effb0` EfficientNet-B0 (torchvision); `mnv3l` MobileNetV3-Large, `mnv4m` MobileNetV4-Conv-Medium, `cnxv2t` ConvNeXt V2-Tiny, `vits` ViT-S/16 (timm).
