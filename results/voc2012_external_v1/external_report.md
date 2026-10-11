# PASCAL-VOC-2012 untouched external test

Run: `voc2012_external_v1`; protocol: `VOC2012_EXTERNAL_PROTOCOL.md`.

Analyzed 2502/2781 pairs from 834/927 sources. Inconsistent: 588 (23.50%).
Clean source accuracy: 85.87%. Transport candidates: complete.

| Measure | AUROC | AP | Median unchanged | Median changed |
|---|---:|---:|---:|---:|
| H-RCAM | 0.8438 | 0.6661 | 0.0266 | 0.1039 |
| NCD | 0.8197 | 0.6482 | 0.0341 | 0.1406 |
| JSD | 0.8410 | 0.6399 | 0.0258 | 0.0846 |
| D1 | 0.7552 | 0.5588 | 0.0324 | 0.0688 |
| D2 | 0.7778 | 0.5163 | 0.1150 | 0.3957 |
| D3 | 0.7961 | 0.6136 | 0.3082 | 0.6171 |
| Foreground shift | 0.6567 | 0.4047 | 0.0363 | 0.0691 |
| Background increase | 0.5473 | 0.3181 | 0.0000 | 0.0000 |
| BAT | 0.7557 | 0.4999 | 0.0382 | 0.0769 |
| SpatialW1 | 0.8001 | 0.5947 | 0.0383 | 0.0804 |

Fixed JSD/NCD fusion rules:

| Rule | AUROC | AP |
|---|---:|---:|
| Minimum | 0.8426 | 0.6546 |
| Maximum | 0.8292 | 0.6522 |
| Arithmetic | 0.8384 | 0.6642 |
| Geometric | 0.8419 | 0.6669 |

Source-cluster bootstrap AUROC difference intervals (2.5%, median, 97.5%):
- H-RCAM minus NCD: [0.014670162405541018, 0.023930903603926634, 0.03357983183812497]
- H-RCAM minus JSD: [-0.005975986708177686, 0.003125600015165042, 0.012401347368258018]
- H-RCAM minus Arithmetic: [0.0011898388288809625, 0.005469502738124243, 0.009680795483697274]
- H-RCAM minus Geometric: [-0.0002957680751943442, 0.0018419927188781937, 0.003938608829842927]

This is an external validation of the frozen metric, not proof of causal robustness.
