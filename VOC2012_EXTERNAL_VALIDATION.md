# PASCAL VOC 2012: frozen external test

The [protocol](VOC2012_EXTERNAL_PROTOCOL.md) was fixed before seeing any
VOC classifier or H-RCAM outcomes. [PASCAL VOC 2012](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/)
has an official segmentation train/validation split but does not publicly
release masks for its hidden test set. We therefore used the official
segmentation **validation** set as our untouched evaluation set. The
original archive was obtained partly from the official host and resumed
from a mirror; its MD5 matches the published torchvision value
`6cd6e144f989b92b3379bac3b3de84fd` exactly.

Because the original task allows multiple object classes per image, we
prespecified a single-object-*class* subset. Of 1464 train and 1449
validation images, **933 train and 927 validation images** had exactly one
non-background object class in their official masks; there were no
evaluation classes absent from training. We preserve all pixels labelled
with that object class or marked void/uncertain and replace only pixels
labelled background. An outcome-blind mask QA sheet was inspected but is not
redistributed because it contains original dataset images. Ninety-three
validation images failed the fixed 5--95% preserved-pixel-area rule,
leaving **834 sources / 2502 pairs**. Of these, **588 (23.50%)** changed
predicted label. The frozen ImageNet ResNet-18 plus independently fitted
logistic head had **85.87%** clean accuracy on all 927 filtered validation
sources before the mask-area exclusion.

| Measure | AUROC | AP |
|---|---:|---:|
| H-RCAM | 0.8438 | 0.6661 |
| JSD | 0.8410 | 0.6399 |
| NCD | 0.8197 | 0.6482 |
| CD | 0.7552 | 0.5588 |
| KLD | 0.7778 | 0.5163 |
| SNO | 0.7961 | 0.6136 |
| BAT | 0.7557 | 0.4999 |
| SpatialW1 | 0.8001 | 0.5947 |
| FAS | 0.6567 | 0.4047 |
| BAI | 0.5473 | 0.3181 |

H-RCAM's source-cluster bootstrap 95% AUROC interval was **[0.8169,
0.8705]**. The paired AUROC difference was positive versus NCD
([0.0147, 0.0336]), but its interval versus JSD **crossed zero**
([-0.0060, 0.0124]). H-RCAM AUROC was 0.8230, 0.8646 and 0.8450 for
black, gray and white backgrounds, respectively. Thus the overall
0.8438 AUROC supports discrimination on this derived VOC task but does not
establish an advantage over JSD.

| Fixed JSD/NCD fusion | AUROC | AP |
|---|---:|---:|
| Minimum | 0.8426 | 0.6546 |
| Maximum | 0.8292 | 0.6522 |
| Arithmetic | 0.8384 | 0.6642 |
| Geometric | 0.8419 | **0.6669** |
| Harmonic (H-RCAM) | **0.8438** | 0.6661 |

Harmonic fusion had the highest observed AUROC among these fixed rules,
but its margin over geometric was small (about 0.0019), the paired interval
crossed zero ([-0.0003, 0.0039]), and geometric had slightly higher AP.
The original VOC task is multi-label and often multi-object; our selected
single-class subset, one classifier, official validation rather than hidden
test split, and one background-perturbation family limit transfer claims.
AP should be read alongside the 23.50% prediction-inconsistency prevalence.

## Audit trail

- [Frozen protocol](VOC2012_EXTERNAL_PROTOCOL.md)
- [Per-pair scores and outcomes](results/voc2012_external_v1/external_records.jsonl)
- [Transport candidates](results/voc2012_external_v1/external_transport.jsonl)
- [Full summary](results/voc2012_external_v1/external_summary.json)
- [Generated compact report](results/voc2012_external_v1/external_report.md)
- [Filtered source IDs](results/voc2012_external_v1/voc2012_selected_ids.json)
