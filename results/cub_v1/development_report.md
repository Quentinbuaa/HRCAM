# CUB-200-2011 development results

Exploratory independent reproduction; not a publication-ready confirmation.

## own-class CAM

Analyzed: 3477/3618 pairs. Positive prevalence: 0.635.

| Score | AUROC (95% cluster CI) | Average precision |
|---|---:|---:|
| d1 | 0.6933 (0.6684, 0.7124) | 0.8058 |
| d2 | 0.6667 (0.6423, 0.6923) | 0.7528 |
| d3 | 0.7594 (0.7378, 0.7777) | 0.8404 |
| d4 | 0.7911 (0.7692, 0.8082) | 0.8641 |
| RCAM | 0.7576 (0.7330, 0.7764) | 0.8448 |
| BRCAM | 0.7733 (0.7510, 0.7911) | 0.8531 |
| jsd | 0.7727 (0.7546, 0.7928) | 0.8529 |
| foreground_shift | 0.5707 (0.5458, 0.5936) | 0.7228 |
| background_increase | 0.5508 (0.5371, 0.5666) | 0.6846 |
| learned_bounded | 0.7727 (0.7546, 0.7928) | 0.8529 |
| confidence_drop | 0.8633 (0.8497, 0.8805) | 0.9048 |
| prediction_jsd | 0.9587 (0.9520, 0.9659) | 0.9774 |

Clean accuracy: 0.5697; transformed accuracy: 0.3574; label consistency: 0.3588.

## fixed-class CAM

Analyzed: 3470/3618 pairs. Positive prevalence: 0.634.

| Score | AUROC (95% cluster CI) | Average precision |
|---|---:|---:|
| d1 | 0.6617 (0.6369, 0.6828) | 0.7876 |
| d2 | 0.7723 (0.7503, 0.7908) | 0.8659 |
| d3 | 0.6583 (0.6346, 0.6817) | 0.7767 |
| d4 | 0.7019 (0.6802, 0.7250) | 0.8106 |
| RCAM | 0.6972 (0.6756, 0.7198) | 0.8086 |
| BRCAM | 0.6944 (0.6723, 0.7180) | 0.8055 |
| jsd | 0.7737 (0.7523, 0.7933) | 0.8666 |
| foreground_shift | 0.6392 (0.6191, 0.6620) | 0.7730 |
| background_increase | 0.5059 (0.4912, 0.5174) | 0.6593 |
| learned_bounded | 0.7737 (0.7523, 0.7933) | 0.8666 |
| confidence_drop | 0.8632 (0.8497, 0.8804) | 0.9045 |
| prediction_jsd | 0.9585 (0.9518, 0.9658) | 0.9772 |

Clean accuracy: 0.5697; transformed accuracy: 0.3574; label consistency: 0.3588.

## Limitations

- Review the run metadata and protocol for mask provenance and model-selection exposure.
- One dataset and one classifier checkpoint; results do not establish general superiority.
- Label inconsistency is not synonymous with actual classification failure.
- Learned bounded score is supervised and must not be described as a label-free metric.
- Confidence intervals reflect sampled source images, not training-seed variation.
