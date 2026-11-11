# Standalone revised Table 3

Author requested a new Table 3 in Markdown on 2026-11-05. Preserve the
original ResNet-18, layer4 Grad-CAM, own-predicted-class targets, Pet/CUB
source pools, original masks and black/gray/white probes. Use 2979 Pet pairs
from 993 sources and 2889 CUB pairs from 963 sources, not the smaller
intersections from target or method/layer sensitivity experiments.

The draft includes the original ten methods with 95% source-cluster CIs and
the three additional distances requested in R2C4: L1, L2, SSIM. The user was
offered an original-ten-only display as an optional preference; if chosen,
the same saved estimates can be displayed with those rows filtered out.
No classifier training, new inference, new CAM extraction, component/fusion
selection, or manuscript editing. No changes to Table 2, 4, 7 or 8.

## Original methods

Reproduce the exact original Table 3 point estimates (40 entries) from the
original own-class eligible valid records, archived transport scores, and
frozen calibration. JSD/NCD/H-RCAM are unchanged. CD/KLD/SNO retain the
original frozen min-max clipping policy, not a newly fitted normalization.
Keep all rows in their original order; add L1/L2/SSIM before the final H row.

## Additional distances: fixed before examining their results

Use the archived full-resolution 224x224 independently [0,1]-scaled maps,
with no new mass normalization, smoothing, or map-validity selection.

- L1 = mean(abs(A-B)), the usual L1 norm divided by the fixed pixel count.
- L2 = sqrt(mean((A-B)^2)), the usual Euclidean norm divided by sqrt(N).
  These constant rescalings do not alter AUROC/AP rankings.
- SSIM distance = 1 - mean SSIM, so larger means more dissimilar.
  Use Gaussian 11x11 weights (sigma 1.5), data_range=1, K1=.01, K2=.03,
  population covariance, and average the valid interior after cropping
  five pixels from each border. Implement using existing SciPy Gaussian
  filters; independently verify with an explicit 11x11 valid convolution.
  No parameter selection based on AUROC. Do not clip negative SSIM values.

Settings follow the Wang-compatible SSIM guidance in the primary scikit-image
documentation, inspected before calculation:
https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity

## Uncertainty and display

1000 source-image-cluster bootstrap draws, seed 20261011. Sample source IDs,
retaining all three available background pairs together. Use identical
draws across every method and compute percentile 95% intervals for AUROC/AP,
and paired H-minus-baseline differences. Skip single-outcome draws and
report effective resamples. These are individual, not simultaneous intervals.

Use grouped-score weighted rank calculations for speed; cross-check against
sklearn on actual replicated-source draws and independently reconstruct
table metrics and intervals. Preserve all results regardless of rank.

Main Markdown: the original five-column layout, each numeric cell showing
estimate [95% CI], expanded with the three requested rows. Bold only the
largest unrounded observed point estimates, not a claim of significance.
Footnotes identify targeting, source counts, selection-influenced development
comparisons, bootstrap unit, and added-distance definitions. Preserve the
original caption wording where suitable. Retain full precision in CSV/JSON
and paired-comparison details in a companion file.

Run saved-map analysis only on remote-5080 via Tailscale 100.74.145.98.
Every remote Python starts with taskset -c 24-31; four numerical threads.
No dependencies need to be installed. Freeze code/protocol and hashes of
original data before analysis. Copy verified compact artifacts locally and
deliver Reviews/Table3_revised.md. Submitted manuscript must stay unchanged.
