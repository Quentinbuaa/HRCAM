# RQ3.4: Natural-background replacement

Existing ResNet-18 models; original masks; one distinct photographic donor per source.

| Dataset | Target | Valid pairs | Inconsistency | H-RCAM AUROC (95% CI) | AP (95% CI) |
|---|---|---:|---:|---:|---:|
| pets_v1 | own | 993 | 16.01% | 0.9020 [0.8753, 0.9253] | 0.7150 [0.6474, 0.7782] |
| pets_v1 | fixed | 993 | 16.01% | 0.7899 [0.7494, 0.8289] | 0.5054 [0.4276, 0.5798] |
| cub_v1 | own | 963 | 73.21% | 0.8186 [0.7894, 0.8460] | 0.9224 [0.9027, 0.9381] |
| cub_v1 | fixed | 962 | 73.18% | 0.7623 [0.7315, 0.7955] | 0.8990 [0.8785, 0.9193] |

## All original baselines: own-class targets

| Method | Pet AUROC | Pet AP | CUB AUROC | CUB AP |
|---|---:|---:|---:|---:|
| JSD | 0.8889 | 0.6552 | 0.8038 | 0.9172 |
| NCD | 0.8870 | 0.7041 | 0.8080 | 0.9147 |
| CD | 0.8032 | 0.5548 | 0.7062 | 0.8707 |
| KLD | 0.8094 | 0.4151 | 0.6938 | 0.8442 |
| SNO | 0.8603 | 0.6507 | 0.7683 | 0.8913 |
| BAT | 0.8303 | 0.5510 | 0.7403 | 0.8915 |
| SpatialW1 | 0.8612 | 0.6149 | 0.7645 | 0.8960 |
| FAS | 0.6788 | 0.4079 | 0.6467 | 0.8468 |
| BAI | 0.6134 | 0.3511 | 0.6382 | 0.8385 |
| H-RCAM | 0.9020 | 0.7150 | 0.8186 | 0.9224 |

## Original constant-color context (descriptive)

| Dataset | Target | Background | n | Inconsistency | H-RCAM AUROC | AP |
|---|---|---|---:|---:|---:|---:|
| pets_v1 | own | black | 993 | 18.43% | 0.8779 | 0.7040 |
| pets_v1 | own | gray | 993 | 13.80% | 0.9126 | 0.6898 |
| pets_v1 | own | white | 993 | 14.50% | 0.8989 | 0.6768 |
| pets_v1 | own | pooled_constant | 2979 | 15.58% | 0.8953 | 0.6888 |
| pets_v1 | fixed | black | 993 | 18.43% | 0.7920 | 0.5829 |
| pets_v1 | fixed | gray | 993 | 13.80% | 0.8127 | 0.4737 |
| pets_v1 | fixed | white | 993 | 14.50% | 0.7880 | 0.4581 |
| pets_v1 | fixed | pooled_constant | 2979 | 15.58% | 0.7973 | 0.5095 |
| cub_v1 | own | black | 963 | 72.07% | 0.7850 | 0.8912 |
| cub_v1 | own | gray | 963 | 57.32% | 0.8241 | 0.8535 |
| cub_v1 | own | white | 963 | 60.44% | 0.8031 | 0.8585 |
| cub_v1 | own | pooled_constant | 2889 | 63.27% | 0.8066 | 0.8684 |
| cub_v1 | fixed | black | 962 | 72.04% | 0.7680 | 0.8921 |
| cub_v1 | fixed | gray | 963 | 57.32% | 0.7604 | 0.8075 |
| cub_v1 | fixed | white | 962 | 60.40% | 0.7576 | 0.8313 |
| cub_v1 | fixed | pooled_constant | 2887 | 63.25% | 0.7658 | 0.8484 |

CD/KLD/SNO use the original frozen calibration transform, matching the original table policy.
Raw unclipped estimates and clipping counts are retained in summary.json. No new calibration was fitted.
CIs use 1000 paired source resamples. Full differences and median-gap intervals are in summary.json.
Different backgrounds can produce different inconsistency labels and prevalence; AP is not directly prevalence invariant.
These composites test photographic replacement under supplied masks, not imperfect segmentation.
