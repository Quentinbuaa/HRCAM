# Fixed-source-class sensitivity check: two development datasets

The original classifier checkpoints, image subsets, masks, three background replacements,
Grad-CAM layer, normalization, and H-RCAM formula are unchanged. No weights or thresholds
were fitted. Original own-class and fixed-source-class maps were already saved together
during the original extraction; this analysis verifies and reuses those caches.

The source map explains the source image's predicted class in both conventions. The
follow-up map explains its own prediction (primary convention) or the source prediction
(sensitivity convention). The evaluation outcome remains actual prediction inconsistency.

Results below use the intersection of valid pairs across both conventions. Percentile
95% intervals use 1000 paired source-image-cluster bootstrap resamples; all available
backgrounds of each sampled source are retained together. These are retrospective
sensitivity results on the development datasets, not new independent verification.

## Main comparison

| Dataset | Convention | Matched pairs (inconsistent) | H-RCAM AUROC (95% CI) | H-RCAM AP (95% CI) |
|---|---|---:|---:|---:|
| Oxford-IIIT Pet | Each image's own predicted class | 2979 (464) | 0.8953 [0.8753, 0.9155] | 0.6888 [0.6349, 0.7423] |
| Oxford-IIIT Pet | Fixed source-predicted class | 2979 (464) | 0.7973 [0.7673, 0.8248] | 0.5095 [0.4437, 0.5696] |
| CUB-200-2011 | Each image's own predicted class | 2887 (1826) | 0.8067 [0.7843, 0.8260] | 0.8684 [0.8474, 0.8873] |
| CUB-200-2011 | Fixed source-predicted class | 2887 (1826) | 0.7658 [0.7425, 0.7854] | 0.8484 [0.8266, 0.8681] |

## Paired change: fixed-source minus own-class H-RCAM

| Dataset | AUROC difference (95% CI) | AP difference (95% CI) |
|---|---:|---:|
| Oxford-IIIT Pet | -0.0980 [-0.1270, -0.0738] | -0.1792 [-0.2308, -0.1313] |
| CUB-200-2011 | -0.0409 [-0.0536, -0.0274] | -0.0200 [-0.0295, -0.0107] |

## Component comparison under fixed-source targeting

| Dataset | Measure | AUROC | AP |
|---|---|---:|---:|
| Oxford-IIIT Pet | JSD | 0.8278 | 0.5643 |
| Oxford-IIIT Pet | NCD | 0.7457 | 0.4380 |
| Oxford-IIIT Pet | HRCAM | 0.7973 | 0.5095 |
| CUB-200-2011 | JSD | 0.7894 | 0.8712 |
| CUB-200-2011 | NCD | 0.7293 | 0.8164 |
| CUB-200-2011 | HRCAM | 0.7658 | 0.8484 |

## Checks and coverage

### Oxford-IIIT Pet

- Reproduced manuscript own-class H-RCAM: AUROC 0.8953, AP 0.6888.
- Eligible valid pairs: own 2979; fixed 2979; matched 2979.
- Own-valid pairs excluded by fixed-map invalidity: 0.
- Matched sources: 993; consistent pairs: 2515; inconsistent pairs: 464.
- Maximum H-RCAM difference for consistent pairs: 0 (same target expected).
- Maximum independently recomputed JSD/NCD error: 1.33e-15.
- Fresh gradient replay: 48 pairs for both conventions; maximum map error 0.
- Model, records, splits, and cached maps retain their original SHA256 hashes.

### CUB-200-2011

- Reproduced manuscript own-class H-RCAM: AUROC 0.8066, AP 0.8684.
- Eligible valid pairs: own 2889; fixed 2887; matched 2887.
- Own-valid pairs excluded by fixed-map invalidity: 2.
- Matched sources: 963; consistent pairs: 1061; inconsistent pairs: 1826.
- Maximum H-RCAM difference for consistent pairs: 0 (same target expected).
- Maximum independently recomputed JSD/NCD error: 1.11e-15.
- Fresh gradient replay: 48 pairs for both conventions; maximum map error 0.
- Model, records, splits, and cached maps retain their original SHA256 hashes.

## Interpretation boundaries

A drop under fixed-source targeting indicates sensitivity of the observed discrimination
to the explanation-target convention. It is not a causal decomposition of the score.
Remaining AUROC above 0.5 supports an association with label inconsistency even when
the target class is held fixed, but does not establish prediction correctness, universal
robustness, or superiority over all candidates. This check compares H-RCAM and its two
components only; transport and other individual candidates were not reranked.

The original manuscript, original extraction files, and GitHub release are not changed.
