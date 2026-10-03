Lin Ding Shan
Institute of Computer Science and Digital Innovation, UCSI University
Jalan Menara Gading, UCSI Heights, Cheras 56000, Kuala Lumpur, Malaysia
1002475487@ucsiuniversity.edu.my

3 October 2026

Editor-in-Chief
*Pattern Recognition Letters*

Dear Editor,

Please consider the enclosed research report, *Seed-to-seed variation outweighs lightweight attention gains: a grouped, repeated cross-validation study on field durian disease images*, for publication in *Pattern Recognition Letters*.

**What the paper reports.** Attention modules added to pretrained convolutional networks are routinely reported to raise plant disease classification accuracy by a few points. We tested three lightweight modules (a 1,281-parameter spatial gate, squeeze-and-excitation and CBAM) on an EfficientNet-B0 classifier of four durian disease classes, using 550 field images from Malaysian orchards, under an evaluation that controls the three things that usually go uncontrolled on datasets of this size: images were split by capture group so that near-duplicate photographs never straddle a split, every configuration was trained on 20 grouped folds with two fixed seeds (160 runs), and configurations were compared in pairs with the corrected resampled *t*-test. A decision rule was written before the runs, and a 200-run follow-up at earlier insertion points and higher resolution was pre-registered. No module met the rule (LFA: −0.13 percentage points, 95% CI −4.56 to 4.31). Changing only the random seed moved macro F1 by 5.3 points on average, and in one seed-matched single-run pair in five the module appeared to be 3 points ahead. The models transferred poorly to an independent Vietnamese dataset.

**Why it fits the journal.** The report is a concise methodological finding of wide application rather than a new architecture: on small field datasets, the uncontrolled variation from the split, the seed and the test is of the same size as the gains attributed to lightweight attention, so single-run comparisons of a few points carry little information about the architecture. The paper shows how large that variation is, with a design that other small-data pattern-recognition studies can reuse. It is 7 pages in the double-column format including all figures, tables and references.

**Reproducibility.** The images and capture-session identifiers are deposited on Zenodo (doi:10.5281/zenodo.22177133). The code, the exact fold assignment, every per-image test prediction, the pre-registered analysis of the follow-up and the scripts that regenerate every number, table and figure in the paper are released in a public repository (https://github.com/xiaolin200206/durian-attention-cv), together with a verification script that recomputes every reported quantity from the raw predictions.

**Declarations.** The work is original, has not been published previously (including as a preprint) and is not under consideration elsewhere. The study is single-authored; the author collected and labelled the images, designed and ran all experiments and wrote the paper. The author is developing a handheld decision-support device for durian disease and pest scouting and therefore has a potential financial interest in the field; this is stated in the manuscript. No funding was received. A generative AI tool (Claude, Anthropic) was used to help write and check the experiment and analysis code and to edit the text; the author verified every result and takes full responsibility for the content, as declared in the manuscript. No human participants or animals were involved; access to the orchards was granted by their owners.

Thank you for considering the manuscript.

Yours sincerely,

Lin Ding Shan
