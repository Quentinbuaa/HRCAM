# R2C3: fixed-ground-truth diagnostic

**Completed and independently audited. Current manuscript: RQ3.3 / Table 9.**

Original fitted ResNet-18 models, original Pet/CUB source identities and masks, and the same black/gray/white replacement probes. No training, component selection, recalibration, natural-background extension, or R1C2/R1C3 additional study was performed in this run.

All rows use the same three-way matched valid pairs within each dataset: 2979 Pet pairs and 2887 CUB pairs.

| Attribution target | Pet AUROC | Pet AP | CUB AUROC | CUB AP |
|---|---:|---:|---:|---:|
| Own-predicted class | 0.8953 | 0.6888 | 0.8067 | 0.8684 |
| Fixed source-predicted class | 0.7973 | 0.5095 | 0.7658 | 0.8484 |
| Fixed ground-truth class | 0.7650 | 0.4111 | 0.7288 | 0.8183 |

## Target definitions

- Own-class explains f(I) on the source and f(I') on the follow-up.
- Fixed-source explains f(I) on both images.
- Fixed ground-truth explains the original annotated class y on both images, including source-incorrect cases.

## Coverage and the original Table 8

| Dataset | Convention | Mask-eligible pairs | Convention-valid pairs | Invalid pairs | Valid sources |
|---|---|---:|---:|---:|---:|
| Oxford-IIIT Pet | Own-predicted class | 2979 | 2979 | 0 | 993 |
| Oxford-IIIT Pet | Fixed source-predicted class | 2979 | 2979 | 0 | 993 |
| Oxford-IIIT Pet | Fixed ground-truth class | 2979 | 2979 | 0 | 993 |
| CUB-200-2011 | Own-predicted class | 2889 | 2889 | 0 | 963 |
| CUB-200-2011 | Fixed source-predicted class | 2889 | 2887 | 2 | 963 |
| CUB-200-2011 | Fixed ground-truth class | 2889 | 2889 | 0 | 963 |

Oxford-IIIT Pet: the original own/fixed-source matched set has 2979 pairs. The three-way comparison excludes another 0 pairs (0 consistent, 0 inconsistent).

CUB-200-2011: the original own/fixed-source matched set has 2887 pairs. The three-way comparison excludes another 0 pairs (0 consistent, 0 inconsistent).

Current Table 9 recomputes every convention on the three-way intersection. Table 3 retains its own-class valid set; the matched CUB comparison excludes two invalid fixed-source pairs. Convention-specific all-valid estimates and the original two-way Table 8 reproduction are preserved in [summary.json](analysis/summary.json).

## H-RCAM confidence intervals on the three-way common set

| Dataset | Convention | AUROC (95% CI) | AP (95% CI) |
|---|---|---:|---:|
| Oxford-IIIT Pet | Own-predicted class | 0.8953 [0.8753, 0.9155] | 0.6888 [0.6349, 0.7423] |
| Oxford-IIIT Pet | Fixed source-predicted class | 0.7973 [0.7673, 0.8248] | 0.5095 [0.4437, 0.5696] |
| Oxford-IIIT Pet | Fixed ground-truth class | 0.7650 [0.7326, 0.7962] | 0.4111 [0.3472, 0.4742] |
| CUB-200-2011 | Own-predicted class | 0.8067 [0.7843, 0.8260] | 0.8684 [0.8474, 0.8873] |
| CUB-200-2011 | Fixed source-predicted class | 0.7658 [0.7425, 0.7854] | 0.8484 [0.8266, 0.8681] |
| CUB-200-2011 | Fixed ground-truth class | 0.7288 [0.7023, 0.7527] | 0.8183 [0.7911, 0.8417] |

Intervals use 1000 paired source-cluster bootstrap resamples, seed 20261011, retaining the available black/gray/white pairs together. Pair outcomes and the source draw are shared across conventions and components. They describe source-sampling uncertainty conditional on these classifiers/probes and the valid-map rules. No pair-level independence p-values are reported.

## Source-correctness diagnostic

| Dataset | Source stratum | Common sources | Common pairs | Own AUROC | Fixed-source AUROC | GT AUROC |
|---|---|---:|---:|---:|---:|---:|
| Oxford-IIIT Pet | source_correct | 869 | 2607 | 0.8912 | 0.7906 | 0.7906 |
| Oxford-IIIT Pet | source_incorrect | 124 | 372 | 0.8319 | 0.7100 | 0.4769 |
| CUB-200-2011 | source_correct | 541 | 1623 | 0.8072 | 0.7582 | 0.7582 |
| CUB-200-2011 | source_incorrect | 422 | 1264 | 0.7674 | 0.7360 | 0.6085 |

When the source prediction is correct, the GT target equals the fixed-source target. Their maps/scores therefore agree; the third convention adds information on source-incorrect images. Ground-truth maps can be invalid for some images, so the source-correctness coverage breakdown is also retained in summary.json. Own/fixed-source scores agree on prediction-consistent pairs. Ground-truth scores need not agree with them on prediction-consistent but source-incorrect pairs.

## Component diagnostics on the same common set

| Convention | Measure | Pet AUROC | Pet AP | CUB AUROC | CUB AP |
|---|---|---:|---:|---:|---:|
| Own-predicted class | JSD | 0.8778 | 0.6354 | 0.7797 | 0.8492 |
| Own-predicted class | NCD | 0.8816 | 0.6777 | 0.7940 | 0.8585 |
| Own-predicted class | HRCAM | 0.8953 | 0.6888 | 0.8067 | 0.8684 |
| Fixed source-predicted class | JSD | 0.8278 | 0.5643 | 0.7894 | 0.8712 |
| Fixed source-predicted class | NCD | 0.7457 | 0.4380 | 0.7293 | 0.8164 |
| Fixed source-predicted class | HRCAM | 0.7973 | 0.5095 | 0.7658 | 0.8484 |
| Fixed ground-truth class | JSD | 0.7984 | 0.4461 | 0.7470 | 0.8333 |
| Fixed ground-truth class | NCD | 0.7156 | 0.3633 | 0.7002 | 0.7984 |
| Fixed ground-truth class | HRCAM | 0.7650 | 0.4111 | 0.7288 | 0.8183 |

## Interpretation for the revision

The practical argument is lack of reference labels at assessment time. This supports prediction-based targeting; both own-class and fixed-source conventions are label-free. The additional reason for own-class is that it explains the actual decision for each image. This is a different question from evidence stability for a fixed class. Its target choice depends on predictions and, when predictions differ, the comparison also changes the explained class.

The GT experiment is a labelled evaluation diagnostic, not a requirement for using H-RCAM on unlabelled inputs. It does not remove the need to acknowledge prediction-dependent targeting, and its observed results should not be used to select a preferred target convention after the fact. Further target-switch diagnostics are released in `../r1c2_target_switch_20261105_v1/`.

## Verification and released files

Original records, splits and cached maps were hash-checked during the run.
The [independent audit](INDEPENDENT_AUDIT.json) uses closed-form Grad-CAM,
SciPy component formulas, pure-Python ranks and bootstrap reconstruction.
The full [summary](analysis/summary.json), matched score JSONLs, saved
bootstrap metric arrays and executed `code/` are included. The CSV filename
`table8_three_conventions_with_ci.csv` retains the previous table numbering;
it is the input to current Table 9. Images, masks, checkpoints and native
heatmaps are not redistributed.
