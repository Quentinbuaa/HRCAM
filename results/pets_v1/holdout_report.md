# Oxford-IIIT Pet holdout results

Exploratory independent reproduction; not a publication-ready confirmation.

## own-class CAM

Analyzed: 2979/2997 pairs. Positive prevalence: 0.156.

| Score | AUROC (95% cluster CI) | Average precision |
|---|---:|---:|
| d1 | 0.7998 (0.7698, 0.8342) | 0.5596 |
| d2 | 0.8149 (0.7892, 0.8409) | 0.4453 |
| d3 | 0.8529 (0.8242, 0.8797) | 0.6281 |
| d4 | 0.8816 (0.8579, 0.9055) | 0.6777 |
| RCAM | 0.8661 (0.8413, 0.8921) | 0.6493 |
| BRCAM | 0.8687 (0.8422, 0.8937) | 0.6599 |
| jsd | 0.8778 (0.8546, 0.8998) | 0.6354 |
| foreground_shift | 0.6367 (0.5965, 0.6775) | 0.3514 |
| background_increase | 0.5790 (0.5451, 0.6104) | 0.2932 |
| learned_bounded | 0.8778 (0.8546, 0.8998) | 0.6354 |
| confidence_drop | 0.9850 (0.9809, 0.9891) | 0.9285 |
| prediction_jsd | 0.9799 (0.9735, 0.9852) | 0.9247 |

Clean accuracy: 0.8759; transformed accuracy: 0.8318; label consistency: 0.8415.

## fixed-class CAM

Analyzed: 2979/2997 pairs. Positive prevalence: 0.156.

| Score | AUROC (95% cluster CI) | Average precision |
|---|---:|---:|
| d1 | 0.7432 (0.7117, 0.7758) | 0.4511 |
| d2 | 0.8261 (0.8013, 0.8503) | 0.5516 |
| d3 | 0.7060 (0.6701, 0.7427) | 0.3675 |
| d4 | 0.7457 (0.7107, 0.7800) | 0.4380 |
| RCAM | 0.7593 (0.7264, 0.7902) | 0.4568 |
| BRCAM | 0.7466 (0.7125, 0.7785) | 0.4326 |
| jsd | 0.8278 (0.8006, 0.8534) | 0.5643 |
| foreground_shift | 0.7007 (0.6638, 0.7341) | 0.3899 |
| background_increase | 0.4736 (0.4473, 0.5005) | 0.1800 |
| learned_bounded | 0.8278 (0.8006, 0.8534) | 0.5643 |
| confidence_drop | 0.9850 (0.9809, 0.9891) | 0.9285 |
| prediction_jsd | 0.9799 (0.9735, 0.9852) | 0.9247 |

Clean accuracy: 0.8759; transformed accuracy: 0.8318; label consistency: 0.8415.

## Limitations

- Review the run metadata and protocol for mask provenance and model-selection exposure.
- One dataset and one classifier checkpoint; results do not establish general superiority.
- Label inconsistency is not synonymous with actual classification failure.
- Learned bounded score is supervised and must not be described as a label-free metric.
- Confidence intervals reflect sampled source images, not training-seed variation.
