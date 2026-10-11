# Flowers-102 untouched external test

Run: `flowers102_external_v1`; protocol: `FLOWERS102_EXTERNAL_PROTOCOL.md`.

Analyzed 2961/3060 pairs from 987/1020 sources. Inconsistent: 1382 (46.67%).
Clean source accuracy: 87.55%. Transport candidates: complete.

| Measure | AUROC | AP | Median unchanged | Median changed |
|---|---:|---:|---:|---:|
| H-RCAM | 0.8361 | 0.8078 | 0.0218 | 0.0697 |
| NCD | 0.8173 | 0.7933 | 0.0324 | 0.1183 |
| JSD | 0.8230 | 0.7828 | 0.0179 | 0.0510 |
| D1 | 0.7450 | 0.7223 | 0.0271 | 0.0520 |
| D2 | 0.7719 | 0.6857 | 0.0645 | 0.1896 |
| D3 | 0.7837 | 0.7537 | 0.3013 | 0.5499 |
| Foreground shift | 0.6441 | 0.6267 | 0.0305 | 0.0541 |
| Background increase | 0.5082 | 0.5203 | 0.0000 | 0.0000 |
| BAT | 0.7395 | 0.6977 | 0.0332 | 0.0582 |
| SpatialW1 | 0.7969 | 0.7616 | 0.0321 | 0.0601 |

Fixed JSD/NCD fusion rules:

| Rule | AUROC | AP |
|---|---:|---:|
| Minimum | 0.8304 | 0.7942 |
| Maximum | 0.8188 | 0.7934 |
| Arithmetic | 0.8281 | 0.8028 |
| Geometric | 0.8339 | 0.8082 |

Source-cluster bootstrap AUROC difference intervals (2.5%, median, 97.5%):
- H-RCAM minus NCD: [0.009747594420692025, 0.01888957787802914, 0.02787695198768128]
- H-RCAM minus JSD: [0.0068393048845853425, 0.013084916580632766, 0.019254043732549672]
- H-RCAM minus Arithmetic: [0.002718891044201685, 0.008044294147581177, 0.013265682546385453]
- H-RCAM minus Geometric: [-0.00044773954868297937, 0.0022647507302929193, 0.004846042186311721]

This is an external validation of the frozen metric, not proof of causal robustness.
