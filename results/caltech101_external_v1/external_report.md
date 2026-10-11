# Caltech-101 untouched external test

Run: `caltech101_external_v1`; protocol: `CALTECH101_EXTERNAL_PROTOCOL.md`.

Analyzed 2997/3030 pairs from 999/1010 sources. Inconsistent: 555 (18.52%).
Clean source accuracy: 90.99%. Transport candidates: complete.

| Measure | AUROC | AP | Median unchanged | Median changed |
|---|---:|---:|---:|---:|
| H-RCAM | 0.8739 | 0.6367 | 0.0162 | 0.0744 |
| NCD | 0.8556 | 0.6188 | 0.0211 | 0.1052 |
| JSD | 0.8663 | 0.6165 | 0.0142 | 0.0593 |
| D1 | 0.8000 | 0.5544 | 0.0244 | 0.0573 |
| D2 | 0.8160 | 0.4743 | 0.0558 | 0.2339 |
| D3 | 0.8172 | 0.5633 | 0.2496 | 0.5159 |
| Foreground shift | 0.6770 | 0.3798 | 0.0260 | 0.0548 |
| Background increase | 0.5187 | 0.2520 | 0.0000 | 0.0000 |
| BAT | 0.7889 | 0.4824 | 0.0287 | 0.0645 |
| SpatialW1 | 0.8460 | 0.5991 | 0.0293 | 0.0668 |

Fixed JSD/NCD fusion rules:

| Rule | AUROC | AP |
|---|---:|---:|
| Minimum | 0.8732 | 0.6295 |
| Maximum | 0.8590 | 0.6218 |
| Arithmetic | 0.8682 | 0.6348 |
| Geometric | 0.8719 | 0.6371 |

Source-cluster bootstrap AUROC difference intervals (2.5%, median, 97.5%):
- H-RCAM minus NCD: [0.01085632816432991, 0.01807827271245782, 0.026215729981068404]
- H-RCAM minus JSD: [0.0015576027524967007, 0.007628598367000916, 0.013846519608540647]
- H-RCAM minus Arithmetic: [0.0024447282335017364, 0.005564173767320879, 0.008942890044145235]
- H-RCAM minus Geometric: [0.0003782876899434046, 0.0019597748426321004, 0.0036975810032339185]

This is an external validation of the frozen metric, not proof of causal robustness.
