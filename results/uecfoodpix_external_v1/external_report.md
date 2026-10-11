# UECFoodPixComplete untouched external test

Run: `uecfoodpix_external_v1`; protocol: `UECFOODPIX_EXTERNAL_PROTOCOL.md`.

Analyzed 1752/1758 pairs from 584/586 sources. Inconsistent: 1010 (57.65%).
Clean source accuracy: 66.55%. Transport candidates: complete.

| Measure | AUROC | AP | Median unchanged | Median changed |
|---|---:|---:|---:|---:|
| H-RCAM | 0.7808 | 0.8351 | 0.0318 | 0.0656 |
| NCD | 0.7440 | 0.8078 | 0.0453 | 0.1111 |
| JSD | 0.7807 | 0.8306 | 0.0243 | 0.0511 |
| D1 | 0.6968 | 0.7723 | 0.0316 | 0.0492 |
| D2 | 0.7320 | 0.7641 | 0.0904 | 0.1950 |
| D3 | 0.7308 | 0.7967 | 0.3497 | 0.5542 |
| Foreground shift | 0.6368 | 0.7224 | 0.0400 | 0.0618 |
| Background increase | 0.5379 | 0.6277 | 0.0000 | 0.0000 |
| BAT | 0.7148 | 0.7712 | 0.0390 | 0.0622 |
| SpatialW1 | 0.7424 | 0.8031 | 0.0382 | 0.0578 |

Fixed JSD/NCD fusion rules:

| Rule | AUROC | AP |
|---|---:|---:|
| Minimum | 0.7828 | 0.8335 |
| Maximum | 0.7480 | 0.8094 |
| Arithmetic | 0.7623 | 0.8215 |
| Geometric | 0.7730 | 0.8303 |

Source-cluster bootstrap AUROC difference intervals (2.5%, median, 97.5%):
- H-RCAM minus NCD: [0.022251155520545365, 0.03678215245176125, 0.052911969602573515]
- H-RCAM minus JSD: [-0.011244127519018262, 5.811847993009156e-05, 0.011207498905437761]
- H-RCAM minus Arithmetic: [0.00955655414679247, 0.01850007983479468, 0.028503364338183977]
- H-RCAM minus Geometric: [0.0033239539614753565, 0.007757373737359796, 0.012931204966662379]

This is an external validation of the frozen metric, not proof of causal robustness.
