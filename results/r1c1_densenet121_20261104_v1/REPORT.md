# R1C1: fixed DenseNet-121 architecture-transfer results

Both prespecified datasets are reported. No architecture or metric was selected by these results.

| Dataset | Architecture | Target | Valid pairs | Inconsistency | H-RCAM AUROC (95% CI) | AP (95% CI) |
|---|---|---|---:|---:|---:|---:|
| Oxford-IIIT Pet | ResNet-18 (existing) | own | 2979 | 15.58% | 0.8953 [0.8750, 0.9144] | 0.6888 [0.6325, 0.7444] |
| Oxford-IIIT Pet | ResNet-18 (existing) | fixed | 2979 | 15.58% | 0.7973 [0.7668, 0.8258] | 0.5095 [0.4488, 0.5765] |
| Oxford-IIIT Pet | DenseNet-121 | own | 2979 | 10.51% | 0.9473 [0.9287, 0.9629] | 0.7718 [0.7110, 0.8270] |
| Oxford-IIIT Pet | DenseNet-121 | fixed | 2979 | 10.51% | 0.8272 [0.7946, 0.8578] | 0.4789 [0.4027, 0.5504] |
| CUB-200-2011 | ResNet-18 (existing) | own | 2889 | 63.27% | 0.8066 [0.7862, 0.8256] | 0.8684 [0.8476, 0.8873] |
| CUB-200-2011 | ResNet-18 (existing) | fixed | 2887 | 63.25% | 0.7658 [0.7445, 0.7859] | 0.8484 [0.8262, 0.8686] |
| CUB-200-2011 | DenseNet-121 | own | 2889 | 56.21% | 0.8610 [0.8444, 0.8775] | 0.8880 [0.8664, 0.9070] |
| CUB-200-2011 | DenseNet-121 | fixed | 2886 | 56.17% | 0.7627 [0.7420, 0.7835] | 0.8118 [0.7852, 0.8369] |

## Component comparisons

| Dataset | Architecture | Target | Measure | AUROC (95% CI) | AP (95% CI) |
|---|---|---|---|---:|---:|
| Oxford-IIIT Pet | ResNet-18 (existing) | own | HRCAM | 0.8953 [0.8750, 0.9144] | 0.6888 [0.6325, 0.7444] |
| Oxford-IIIT Pet | ResNet-18 (existing) | own | JSD | 0.8778 [0.8571, 0.8987] | 0.6354 [0.5804, 0.6899] |
| Oxford-IIIT Pet | ResNet-18 (existing) | own | NCD | 0.8816 [0.8597, 0.9032] | 0.6777 [0.6216, 0.7321] |
| Oxford-IIIT Pet | ResNet-18 (existing) | fixed | HRCAM | 0.7973 [0.7668, 0.8258] | 0.5095 [0.4488, 0.5765] |
| Oxford-IIIT Pet | ResNet-18 (existing) | fixed | JSD | 0.8278 [0.8015, 0.8556] | 0.5643 [0.5033, 0.6278] |
| Oxford-IIIT Pet | ResNet-18 (existing) | fixed | NCD | 0.7457 [0.7117, 0.7790] | 0.4380 [0.3788, 0.5030] |
| Oxford-IIIT Pet | DenseNet-121 | own | HRCAM | 0.9473 [0.9287, 0.9629] | 0.7718 [0.7110, 0.8270] |
| Oxford-IIIT Pet | DenseNet-121 | own | JSD | 0.9363 [0.9195, 0.9515] | 0.7024 [0.6286, 0.7667] |
| Oxford-IIIT Pet | DenseNet-121 | own | NCD | 0.9358 [0.9134, 0.9540] | 0.7392 [0.6749, 0.8000] |
| Oxford-IIIT Pet | DenseNet-121 | fixed | HRCAM | 0.8272 [0.7946, 0.8578] | 0.4789 [0.4027, 0.5504] |
| Oxford-IIIT Pet | DenseNet-121 | fixed | JSD | 0.8384 [0.8077, 0.8657] | 0.5048 [0.4247, 0.5747] |
| Oxford-IIIT Pet | DenseNet-121 | fixed | NCD | 0.7960 [0.7605, 0.8293] | 0.4081 [0.3362, 0.4855] |
| CUB-200-2011 | ResNet-18 (existing) | own | HRCAM | 0.8066 [0.7862, 0.8256] | 0.8684 [0.8476, 0.8873] |
| CUB-200-2011 | ResNet-18 (existing) | own | JSD | 0.7796 [0.7580, 0.8000] | 0.8492 [0.8268, 0.8696] |
| CUB-200-2011 | ResNet-18 (existing) | own | NCD | 0.7939 [0.7723, 0.8142] | 0.8586 [0.8368, 0.8785] |
| CUB-200-2011 | ResNet-18 (existing) | fixed | HRCAM | 0.7658 [0.7445, 0.7859] | 0.8484 [0.8262, 0.8686] |
| CUB-200-2011 | ResNet-18 (existing) | fixed | JSD | 0.7894 [0.7683, 0.8093] | 0.8712 [0.8523, 0.8893] |
| CUB-200-2011 | ResNet-18 (existing) | fixed | NCD | 0.7293 [0.7051, 0.7518] | 0.8164 [0.7911, 0.8399] |
| CUB-200-2011 | DenseNet-121 | own | HRCAM | 0.8610 [0.8444, 0.8775] | 0.8880 [0.8664, 0.9070] |
| CUB-200-2011 | DenseNet-121 | own | JSD | 0.8249 [0.8052, 0.8442] | 0.8522 [0.8256, 0.8774] |
| CUB-200-2011 | DenseNet-121 | own | NCD | 0.8511 [0.8333, 0.8678] | 0.8822 [0.8628, 0.9006] |
| CUB-200-2011 | DenseNet-121 | fixed | HRCAM | 0.7627 [0.7420, 0.7835] | 0.8118 [0.7852, 0.8369] |
| CUB-200-2011 | DenseNet-121 | fixed | JSD | 0.8014 [0.7814, 0.8210] | 0.8478 [0.8245, 0.8706] |
| CUB-200-2011 | DenseNet-121 | fixed | NCD | 0.7265 [0.7043, 0.7492] | 0.7779 [0.7502, 0.8048] |

## Paired H-RCAM-minus-component differences

Positive differences favor H-RCAM. Intervals use the same source-cluster resamples.

| Dataset | Architecture | Target | Contrast | AUROC difference (95% CI) | AP difference (95% CI) |
|---|---|---|---|---:|---:|
| Oxford-IIIT Pet | ResNet-18 (existing) | own | HRCAM_minus_JSD | +0.0175 [+0.0086, +0.0263] | +0.0534 [+0.0347, +0.0738] |
| Oxford-IIIT Pet | ResNet-18 (existing) | own | HRCAM_minus_NCD | +0.0137 [+0.0058, +0.0216] | +0.0111 [-0.0070, +0.0308] |
| Oxford-IIIT Pet | ResNet-18 (existing) | fixed | HRCAM_minus_JSD | -0.0305 [-0.0460, -0.0150] | -0.0548 [-0.0823, -0.0276] |
| Oxford-IIIT Pet | ResNet-18 (existing) | fixed | HRCAM_minus_NCD | +0.0516 [+0.0406, +0.0623] | +0.0715 [+0.0547, +0.0880] |
| Oxford-IIIT Pet | DenseNet-121 | own | HRCAM_minus_JSD | +0.0109 [-0.0020, +0.0219] | +0.0694 [+0.0392, +0.1042] |
| Oxford-IIIT Pet | DenseNet-121 | own | HRCAM_minus_NCD | +0.0115 [+0.0063, +0.0167] | +0.0326 [+0.0056, +0.0605] |
| Oxford-IIIT Pet | DenseNet-121 | fixed | HRCAM_minus_JSD | -0.0111 [-0.0307, +0.0075] | -0.0259 [-0.0560, +0.0074] |
| Oxford-IIIT Pet | DenseNet-121 | fixed | HRCAM_minus_NCD | +0.0312 [+0.0215, +0.0405] | +0.0708 [+0.0401, +0.1015] |
| CUB-200-2011 | ResNet-18 (existing) | own | HRCAM_minus_JSD | +0.0270 [+0.0152, +0.0396] | +0.0192 [+0.0102, +0.0292] |
| CUB-200-2011 | ResNet-18 (existing) | own | HRCAM_minus_NCD | +0.0127 [+0.0059, +0.0190] | +0.0099 [+0.0038, +0.0162] |
| CUB-200-2011 | ResNet-18 (existing) | fixed | HRCAM_minus_JSD | -0.0237 [-0.0357, -0.0113] | -0.0228 [-0.0318, -0.0141] |
| CUB-200-2011 | ResNet-18 (existing) | fixed | HRCAM_minus_NCD | +0.0365 [+0.0285, +0.0441] | +0.0320 [+0.0255, +0.0397] |
| CUB-200-2011 | DenseNet-121 | own | HRCAM_minus_JSD | +0.0361 [+0.0232, +0.0484] | +0.0358 [+0.0236, +0.0472] |
| CUB-200-2011 | DenseNet-121 | own | HRCAM_minus_NCD | +0.0099 [+0.0050, +0.0148] | +0.0058 [-0.0000, +0.0117] |
| CUB-200-2011 | DenseNet-121 | fixed | HRCAM_minus_JSD | -0.0387 [-0.0530, -0.0240] | -0.0360 [-0.0484, -0.0240] |
| CUB-200-2011 | DenseNet-121 | fixed | HRCAM_minus_NCD | +0.0362 [+0.0299, +0.0426] | +0.0339 [+0.0278, +0.0410] |

## Coverage and classifier accuracy

| Dataset | Architecture | Target | Valid pairs | Invalid-map pairs | Mask-ineligible pairs | Clean accuracy (all sources) | Transformed accuracy (all pairs) |
|---|---|---|---:|---:|---:|---:|---:|
| Oxford-IIIT Pet | ResNet-18 (existing) | own | 2979 | 0 | 18 | 87.59% | 83.18% |
| Oxford-IIIT Pet | ResNet-18 (existing) | fixed | 2979 | 0 | 18 | 87.59% | 83.18% |
| Oxford-IIIT Pet | DenseNet-121 | own | 2979 | 0 | 18 | 90.19% | 87.19% |
| Oxford-IIIT Pet | DenseNet-121 | fixed | 2979 | 0 | 18 | 90.19% | 87.19% |
| CUB-200-2011 | ResNet-18 (existing) | own | 2889 | 0 | 111 | 55.20% | 34.20% |
| CUB-200-2011 | ResNet-18 (existing) | fixed | 2887 | 3 | 111 | 55.20% | 34.20% |
| CUB-200-2011 | DenseNet-121 | own | 2889 | 0 | 111 | 62.80% | 42.07% |
| CUB-200-2011 | DenseNet-121 | fixed | 2886 | 3 | 111 | 62.80% | 42.07% |

## Common valid pairs across backbones

The pair identities are identical within each row; each backbone retains its own prediction outcomes.

| Dataset | Target | Common pairs | DenseNet H-RCAM AUROC | ResNet H-RCAM AUROC | DenseNet valid pairs excluded | ResNet valid pairs excluded |
|---|---|---:|---:|---:|---:|---:|
| Oxford-IIIT Pet | own | 2979 | 0.9473 | 0.8953 | 0 | 0 |
| Oxford-IIIT Pet | fixed | 2979 | 0.8272 | 0.7973 | 0 | 0 |
| CUB-200-2011 | own | 2889 | 0.8610 | 0.8066 | 0 | 0 |
| CUB-200-2011 | fixed | 2884 | 0.7624 | 0.7654 | 2 | 3 |

Full uncertainty, paired differences, validity coverage, per-background results,
accuracy, and provenance are in the JSON summaries. Different backbones produce different
prediction-inconsistency outcomes; their AUROCs are not classifier-accuracy comparisons.

These are architecture-transfer results on the original development datasets, not new
independent-dataset validation. One fitted head per dataset was evaluated; source-cluster
intervals do not quantify training variability. The manuscript has not been changed.
