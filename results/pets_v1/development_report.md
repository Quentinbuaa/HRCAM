# Oxford-IIIT Pet development results

Exploratory independent reproduction; not a publication-ready confirmation.

## own-class CAM

Analyzed: 2211/2220 pairs. Positive prevalence: 0.144.

| Score | AUROC (95% cluster CI) | Average precision |
|---|---:|---:|
| d1 | 0.7852 (0.7432, 0.8266) | 0.4960 |
| d2 | 0.7913 (0.7624, 0.8191) | 0.3608 |
| d3 | 0.8421 (0.8136, 0.8704) | 0.5730 |
| d4 | 0.8714 (0.8435, 0.8978) | 0.6153 |
| RCAM | 0.8551 (0.8240, 0.8831) | 0.5920 |
| BRCAM | 0.8600 (0.8309, 0.8849) | 0.6013 |
| jsd | 0.8706 (0.8405, 0.8961) | 0.5734 |
| foreground_shift | 0.6475 (0.6065, 0.6921) | 0.2966 |
| background_increase | 0.5497 (0.5187, 0.5835) | 0.2446 |
| learned_bounded | 0.8706 (0.8405, 0.8961) | 0.5734 |
| confidence_drop | 0.9755 (0.9643, 0.9836) | 0.8969 |
| prediction_jsd | 0.9766 (0.9679, 0.9843) | 0.9048 |

Clean accuracy: 0.9095; transformed accuracy: 0.8423; label consistency: 0.8559.

## fixed-class CAM

Analyzed: 2211/2220 pairs. Positive prevalence: 0.144.

| Score | AUROC (95% cluster CI) | Average precision |
|---|---:|---:|
| d1 | 0.6722 (0.6305, 0.7140) | 0.3256 |
| d2 | 0.7956 (0.7581, 0.8362) | 0.4887 |
| d3 | 0.6778 (0.6395, 0.7173) | 0.2996 |
| d4 | 0.7317 (0.6967, 0.7708) | 0.3767 |
| RCAM | 0.7156 (0.6801, 0.7511) | 0.3639 |
| BRCAM | 0.7123 (0.6773, 0.7493) | 0.3529 |
| jsd | 0.8040 (0.7656, 0.8406) | 0.5013 |
| foreground_shift | 0.6840 (0.6421, 0.7255) | 0.3427 |
| background_increase | 0.4622 (0.4380, 0.4847) | 0.1463 |
| learned_bounded | 0.8040 (0.7656, 0.8406) | 0.5013 |
| confidence_drop | 0.9755 (0.9643, 0.9836) | 0.8969 |
| prediction_jsd | 0.9766 (0.9679, 0.9843) | 0.9048 |

Clean accuracy: 0.9095; transformed accuracy: 0.8423; label consistency: 0.8559.

## Limitations

- Review the run metadata and protocol for mask provenance and model-selection exposure.
- One dataset and one classifier checkpoint; results do not establish general superiority.
- Label inconsistency is not synonymous with actual classification failure.
- Learned bounded score is supervised and must not be described as a label-free metric.
- Confidence intervals reflect sampled source images, not training-seed variation.
