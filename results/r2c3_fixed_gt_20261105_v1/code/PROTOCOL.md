# R2C3: Fixed-ground-truth attribution targets

Author explicitly authorized execution on 2026-11-04 before going to rest.
Purpose: add a fixed-ground-truth diagnostic to the own-class/fixed-source
comparison in RQ3.3 / Table 8. No manuscript edits are authorized by this run.
This is not authorization for the separately discussed R1C2 root-cause study,
R1C3 predictive validation, mask-error studies, or another architecture run.

## Fixed scientific settings

- Use the exact original `pets_v1` and `cub_v1` holdout identities (999/1000
  source images before mask exclusions). Preserve original ordering and
  batches of 16, including the final shorter batch, for cuDNN reproducibility.
- Reuse original fitted ResNet-18 classifiers, original prepared images,
  supplied masks, ImageNet normalization and final `layer4` Grad-CAM.
- Use the existing black/gray/white replacements. No new backgrounds,
  classifier training, component selection, or metric recalibration.
- Own-class: source target f(I), follow-up target f(I').
- Fixed-source: both targets f(I).
- Fixed-ground-truth: both targets the original annotated class y. Keep y
  fixed across all three foreground-preserving follow-ups, even if wrong
  predictions occur. Source-correct and source-incorrect cases are retained.
- Reuse archived own/fixed-source maps and scores only after checking hashes,
  source IDs, labels, original-order replay, and applicable map identities.
- Extract GT maps for every original source and follow-up. Preserve all
  records, including invalid/zero maps and mask-ineligible cases.
- JSD uses base 2 and epsilon 1e-8; NCD=(1-Pearson)/2; H=2*J*C/(J+C+1e-12).
  Map validity and mask eligibility (foreground fraction .05-.95) follow the
  original rules. Do not assign numerical scores to invalid map pairs.

## Comparison and uncertainty

The proposed three-row Table 8 extension compares H-RCAM AUROC/AP on the
intersection of eligible pairs valid under **all three** conventions. Compute
own/fixed-source scores anew on that common set; do not reuse the original
Table 8 numbers if GT invalidity changes the set. Disclose common pair/source
counts, removed outcomes, and invalid source/follow-up maps by convention.

Also retain each convention's all-valid estimates and the original two-way
matched comparison, so common-set exclusions and reproduction of the existing
Table 8 remain visible. JSD/NCD diagnostics use the same paired subsets.

Use 1000 paired source-cluster bootstrap resamples with seed 20261011,
matching the existing target-sensitivity study. Retain all available
backgrounds and all compared methods/targets for each sampled source. Report
95% percentile intervals for AUROC/AP and paired differences between target
conventions and between H-RCAM and its two components. No pair-level p-values.
Source-correct/incorrect and per-background breakdowns are descriptive;
single-outcome subsets are reported as not estimable for discrimination.

GT and fixed-source maps/scores must agree wherever the original prediction
equals y. Own and fixed-source maps/scores must agree on prediction-consistent
pairs. GT need not equal those conventions on source-incorrect consistent
pairs. Report these identities and the available source-incorrect coverage.

The labelled GT analysis is a diagnostic, not a requirement for applying
H-RCAM to unlabelled images. It does not establish independent prediction
robustness or identify an additive causal contribution from target switching.

## Execution and audit

Run over Tailscale `100.74.145.98`, SSH alias remote-5080 with
HostKeyAlias=192.168.3.133. Every remote Python process must start with
`taskset -c 24-31`; BLAS/PyTorch threads must be four. Use the existing
`.venv/bin/python` environment; no dependency/environment changes are needed.

Write to a fresh run only. Snapshot protocol/scripts and original-file hashes
before extraction. Preserve failed attempts. Independently verify new maps
with closed-form Grad-CAM for the GAP+linear head, scores with SciPy, table
entries with pure-Python rank counts, and core intervals with a separate
implementation. Copy compact artifacts locally; retain large maps remotely
with hashes. Verify the submitted manuscript hash remains unchanged.
