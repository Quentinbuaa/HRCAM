# Oxford Flowers-102 independent external evaluation

Protocol fixed 2026-10-11 before reading any Flowers-102 classifier or
H-RCAM outcomes. This is an external-dataset test of a metric selected using
Oxford-IIIT Pet and CUB-200-2011. It is not another opportunity to tune the
metric or choose the best-looking dataset.

## Data and foreground QA

- Use the original Oxford Flowers-102 images, labels, official `setid.mat`
  training/validation/test split, and original `102segmentations.tgz`.
- Call these *provided segmentations*, not manually verified ground truth.
  Determine their color encoding without consulting model outcomes. Sample a
  deterministic, class-spread contact sheet and inspect mask failures before
  inference. Report the quality issue and coverage even if results are poor.
- Outcome-blind inspection on 24 classes found that provided JPEGs encode
  background as near-pure blue. Decode a foreground pixel when its RGB
  Euclidean distance from (0, 0, 255) is at least 110. This radius is fixed
  before any classifier or H-RCAM outcome is observed.
- Resize RGB to 224x224 bilinear, binary mask to 224x224 nearest-neighbor.
  Preserve the mask-designated foreground bit-for-bit, replacing only the
  designated background with RGB 0, 128, or 255.
- Exclude mask areas below 5% or above 95% from the primary paired-score
  analysis, as in the existing Pet/CUB scripts. Report exclusions. No
  outcome-dependent mask filtering or manual mask correction.

## Classifier and evaluation sample

- Freeze the same ImageNet-pretrained ResNet-18 backbone. Fit a 102-class
  multinomial logistic-regression head (C=1, max_iter=1000) using the union
  of official Flowers training and validation images. The official test set
  is never used for classifier fitting or metric selection.
- Before seeing test results, choose 10 official-test images per class with
  NumPy seed 20261011 (1020 source images total). This mirrors the roughly
  1000-source Pet/CUB main comparisons. Retain the selected IDs in a manifest.
  Evaluate all three backgrounds for every selected source. Do not replace
  unfavorable images with other images.
- Report clean classification accuracy for the selected test sources and the
  label-inconsistency prevalence across pairs/backgrounds. The classifier is
  a measurement instrument, not a contribution to benchmark accuracy.

## Frozen metric and comparisons

- Use the existing `experiment.collect` pipeline: Grad-CAM at ResNet-18
  `layer4`, own-predicted-class convention, nonnegative maps normalized by
  their map maximum. Flag zero/constant maps invalid and report counts.
- H-RCAM is fixed as the harmonic mean of normalized attribution-map JSD
  and NCD: J = JSD/log(2), C = (1-Pearson r)/2,
  H-RCAM = 2JC/(J+C+1e-12). No weights, thresholds, or dataset min-max
  normalization are learned from Flowers-102.
- The primary discrimination endpoint is the observed indicator that the
  source and background-replaced image receive different predicted labels.
  It is not ground-truth misclassification or a causal robustness guarantee.
- Primary comparisons: H-RCAM versus NCD and JSD, all pairs pooled, AUROC
  and AP. Also report baseline D1/D2/D3, foreground-mass change and signed
  background increase. Compute BAT and SpatialW1 if time and compute allow,
  retaining any omitted comparison explicitly. Report all backgrounds, not
  only whichever looks best.
- As a secondary check of the retrospectively selected harmonic rule, report
  fixed minimum, maximum, arithmetic-mean, and geometric-mean JSD/NCD fusion
  rules on the same new pairs. Do not change H-RCAM if another rule wins.
- Report source-cluster bootstrap percentile 95% CIs (1000 resamples, seed
  20261012) for H-RCAM AUROC and paired AUROC differences versus NCD, JSD,
  arithmetic fusion, and geometric fusion.
  Cluster all three background pairs from each sampled source. Include AP,
  medians for consistent/inconsistent pairs, source-level coverage and a
  by-background sensitivity breakdown. Do not claim superiority when its
  paired-difference CI overlaps zero.

No manuscript results are to be changed until extraction and QA are audited.
