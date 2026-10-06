# Transport candidate development results

Exploratory development only; no fresh confirmatory result

## Controlled examples

| Case | D1 | D4 | JSD | RCAM | Spatial W1 | BAT |
|---|---:|---:|---:|---:|---:|---:|
| identity | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| near_shift | 0.0505 | 1.0003 | 1.0000 | 0.4986 | 0.0505 | 0.0253 |
| far_shift | 0.1515 | 1.0003 | 1.0000 | 0.6164 | 0.1515 | 0.0758 |
| within_foreground | 0.0505 | 1.0003 | 1.0000 | 0.4986 | 0.0505 | 0.0253 |
| cross_into_background | 0.0505 | 1.0003 | 1.0000 | 0.4986 | 0.0505 | 0.5253 |
| fixed_centroid_redistribution | 0.0000 | 1.0006 | 1.0000 | 0.4406 | 0.0714 | 0.0357 |

RCAM uses frozen calibration scales; synthetic extrapolation is clipped and flagged in JSON.

## own-class CAM

2211 pairs, 737 sources.

| Score | Development AUROC | AP |
|---|---:|---:|
| d1 | 0.7852 | 0.4960 |
| d2 | 0.7913 | 0.3608 |
| d3 | 0.8421 | 0.5730 |
| d4 | 0.8714 | 0.6153 |
| RCAM | 0.8551 | 0.5920 |
| jsd | 0.8706 | 0.5734 |
| spatial_w1 | 0.8409 | 0.5330 |
| BAT | 0.7886 | 0.4341 |

- spatial_w1_minus_RCAM: -0.0142, paired 95% interval [-0.0260, -0.0029].
- spatial_w1_minus_d4: -0.0305, paired 95% interval [-0.0441, -0.0164].
- BAT_minus_RCAM: -0.0665, paired 95% interval [-0.0994, -0.0376].
- BAT_minus_d4: -0.0829, paired 95% interval [-0.1166, -0.0536].

Median time for both candidate scores: 7.82 ms/pair.

## fixed-class CAM

2211 pairs, 737 sources.

| Score | Development AUROC | AP |
|---|---:|---:|
| d1 | 0.6722 | 0.3256 |
| d2 | 0.7956 | 0.4887 |
| d3 | 0.6778 | 0.2996 |
| d4 | 0.7317 | 0.3767 |
| RCAM | 0.7156 | 0.3639 |
| jsd | 0.8040 | 0.5013 |
| spatial_w1 | 0.7323 | 0.3740 |
| BAT | 0.7409 | 0.3926 |

- spatial_w1_minus_RCAM: 0.0167, paired 95% interval [-0.0008, 0.0357].
- spatial_w1_minus_d4: 0.0005, paired 95% interval [-0.0222, 0.0256].
- BAT_minus_RCAM: 0.0254, paired 95% interval [-0.0082, 0.0577].
- BAT_minus_d4: 0.0092, paired 95% interval [-0.0261, 0.0476].

Median time for both candidate scores: 7.76 ms/pair.

## Limitations

- Candidate designed after earlier RCAM findings; development performance is not confirmation.
- Transport-based saliency comparison is established prior art; BAT novelty is unverified.
- Synthetic cases isolate specified properties and do not prove classifier vulnerability.
- The boundary penalty is an explicit fixed semantic assumption, not empirically optimal.
- Attribution moving onto the foreground is scored as change too; consult the signed mass change.
- A stable background-dependent or wrong classifier can still have zero distance.
- Pooling loses within-cell detail; the formal metric applies to represented distributions.
- Both candidates add mask/transport costs; a larger AUROC alone is not the sole acceptance criterion.
