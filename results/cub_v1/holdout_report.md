# CUB-200-2011 holdout results

Exploratory independent reproduction; not a publication-ready confirmation.

## own-class CAM

Analyzed: 2889/3000 pairs. Positive prevalence: 0.633.

| Score | AUROC (95% cluster CI) | Average precision |
|---|---:|---:|
| d1 | 0.7012 (0.6747, 0.7243) | 0.8061 |
| d2 | 0.6568 (0.6305, 0.6818) | 0.7411 |
| d3 | 0.7603 (0.7366, 0.7829) | 0.8311 |
| d4 | 0.7939 (0.7727, 0.8153) | 0.8586 |
| RCAM | 0.7580 (0.7354, 0.7792) | 0.8380 |
| BRCAM | 0.7745 (0.7516, 0.7966) | 0.8453 |
| jsd | 0.7796 (0.7541, 0.8034) | 0.8492 |
| foreground_shift | 0.5906 (0.5577, 0.6195) | 0.7191 |
| background_increase | 0.5443 (0.5271, 0.5619) | 0.6880 |
| learned_bounded | 0.7796 (0.7541, 0.8034) | 0.8492 |
| confidence_drop | 0.8790 (0.8622, 0.8940) | 0.9157 |
| prediction_jsd | 0.9580 (0.9495, 0.9637) | 0.9766 |

Clean accuracy: 0.5520; transformed accuracy: 0.3420; label consistency: 0.3583.

## fixed-class CAM

Analyzed: 2887/3000 pairs. Positive prevalence: 0.632.

| Score | AUROC (95% cluster CI) | Average precision |
|---|---:|---:|
| d1 | 0.6742 (0.6527, 0.6984) | 0.7887 |
| d2 | 0.7703 (0.7523, 0.7920) | 0.8613 |
| d3 | 0.6814 (0.6552, 0.7044) | 0.7748 |
| d4 | 0.7293 (0.7042, 0.7503) | 0.8164 |
| RCAM | 0.7185 (0.6949, 0.7409) | 0.8104 |
| BRCAM | 0.7177 (0.6926, 0.7401) | 0.8073 |
| jsd | 0.7894 (0.7690, 0.8109) | 0.8712 |
| foreground_shift | 0.6614 (0.6321, 0.6847) | 0.7782 |
| background_increase | 0.4892 (0.4719, 0.5039) | 0.6533 |
| learned_bounded | 0.7894 (0.7690, 0.8109) | 0.8712 |
| confidence_drop | 0.8789 (0.8621, 0.8939) | 0.9155 |
| prediction_jsd | 0.9579 (0.9494, 0.9637) | 0.9766 |

Clean accuracy: 0.5520; transformed accuracy: 0.3420; label consistency: 0.3583.

## Limitations

- Review the run metadata and protocol for mask provenance and model-selection exposure.
- One dataset and one classifier checkpoint; results do not establish general superiority.
- Label inconsistency is not synonymous with actual classification failure.
- Learned bounded score is supervised and must not be described as a label-free metric.
- Confidence intervals reflect sampled source images, not training-seed variation.
