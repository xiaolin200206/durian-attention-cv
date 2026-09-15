# Manuscript

`manuscript.md` and `supplementary_material.md` are the current version of the accompanying paper.

**The title and framing have changed between submissions and may change again.** The repository is
deliberately not named after the paper. Nothing in `results/`, `figures/` or the scripts depends on
which journal the manuscript is with; the numbers are properties of the data.

`EDITS_APPLIED.md` records the prose edits made in one such reframing, with the originals, so that
any of them can be reverted.

## What has not changed across versions

- 560 images, 73 reconstructed capture sessions, five disease classes
- 96.7% of test images share a session with training under an image-level partition
- 12.2 pp mean macro-F1 difference between image-level and session-level partitioning, nine architectures, four seeds
- PlantVillage: 40,490 images, 7,524 recorded leaves, 99.6% leakage under an image-level split
- Content-based recovery: best pair-F1 0.402, against 0.298 for merging each class entirely

If a version of the manuscript disagrees with any of these, the repository is right and the
manuscript is stale; `CHANGELOG.md` says when each figure was last regenerated.
