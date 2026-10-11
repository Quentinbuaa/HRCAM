# UECFoodPixComplete: frozen external test with a mixed outcome

The [protocol](UECFOODPIX_EXTERNAL_PROTOCOL.md) was fixed before seeing
UECFoodPixComplete classifier or metric results. The creators provide
[manual pixel-level food masks and an official 9000/1000 train/test split](https://mm.cs.uec.ac.jp/uecfoodpix/).
The original dataset is a segmentation task, so we prespecified a derived
single-food-class subset. It retained 5766 training and 586 test images;
3234 train and 414 test images contained zero or multiple food classes,
and no retained test class was absent from training. The official author
masks were inspected in an outcome-blind contact sheet. Two test sources
failed the fixed 5--95% foreground-area rule, leaving **584 sources / 1752
pairs**. Among these, 1010 (57.65%) changed predicted label. The frozen
ResNet-18 plus separately fitted logistic head achieved **66.55%** clean
test accuracy on all 586 single-class test sources before the mask-area
exclusion.

| Measure | AUROC | AP |
|---|---:|---:|
| H-RCAM | 0.7808 | 0.8351 |
| JSD | 0.7807 | 0.8306 |
| NCD | 0.7440 | 0.8078 |
| CD | 0.6968 | 0.7723 |
| KLD | 0.7320 | 0.7641 |
| SNO | 0.7308 | 0.7967 |
| BAT | 0.7148 | 0.7712 |
| SpatialW1 | 0.7424 | 0.8031 |
| FAS | 0.6368 | 0.7224 |
| BAI | 0.5379 | 0.6277 |

The H-RCAM AUROC source-cluster bootstrap 95% interval was **[0.7531,
0.8115]**. Its paired AUROC-difference interval versus NCD was **[0.0223,
0.0529]**, but versus JSD it was **[-0.0112, 0.0112]**. In particular,
H-RCAM's 0.0001 point AUROC edge over JSD is not convincing evidence of an
advantage. Across black, gray and white replacements, H-RCAM AUROC was
0.7294, 0.7999 and 0.8081, respectively; JSD was marginally higher on
gray (0.8021 versus 0.7999).

The prespecified fusion comparison is similarly mixed:

| Fixed JSD/NCD fusion | AUROC | AP |
|---|---:|---:|
| Minimum | **0.7828** | 0.8335 |
| Maximum | 0.7480 | 0.8094 |
| Arithmetic | 0.7623 | 0.8215 |
| Geometric | 0.7730 | 0.8303 |
| Harmonic (H-RCAM) | 0.7808 | **0.8351** |

Thus H-RCAM remained a useful *association with label inconsistency* on
this food-image subset, but it was not the best rule by AUROC and had no
clear AUROC advantage over JSD. The high AP must be read alongside the
57.65% positive prevalence. This single-label filtered food task, one
classifier and one perturbation family do not establish transfer to all
food-image classification systems. The official source permits
non-commercial research use only; do not redistribute the raw data.

## Audit trail

- [Frozen protocol](UECFOODPIX_EXTERNAL_PROTOCOL.md)
- [Per-pair scores and outcomes](results/uecfoodpix_external_v1/external_records.jsonl)
- [Transport candidates](results/uecfoodpix_external_v1/external_transport.jsonl)
- [Full summary](results/uecfoodpix_external_v1/external_summary.json)
- [Generated compact report](results/uecfoodpix_external_v1/external_report.md)
- [Filtered source IDs](results/uecfoodpix_external_v1/uecfoodpix_selected_ids.json)
