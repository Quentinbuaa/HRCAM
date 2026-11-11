# Agreed table layouts and saved-map baseline extension

Author approved adding L1/L2/SSIM to DenseNet-121 on all six datasets and
matching the revised Table 3 layout. Table 4 also uses the same baseline
comparison and layout. Only saved-map scoring/statistics are authorized;
no model training, inference, attribution extraction, outcome filtering,
parameter tuning or canonical manuscript edits.

Ten cases: DenseNet-121 Pet/CUB plus all four independent datasets, and
original ResNet-18 on the four independent datasets. Preserve original
valid own-class pair pools, classifiers, targets, masks and black/gray/white
probes. Keep every dataset regardless of results. Reuse all original ten
scores; no new transport computations or changed calibrations.

New distances match the frozen revised Table 3 exactly on independently
[0,1]-scaled 224x224 maps: L1=mean absolute difference; L2=root mean squared
difference; SSIM distance=1-SSIM, Gaussian11x11 sigma1.5, data range1,
K1=.01/K2=.03, population covariance, valid-interior crop5. Four workers
maximum. No selected parameters or favorable variants.

1000 source-image-cluster draws per case, all background pairs retained
together, same draws for all methods. DenseNet uses its frozen seed20261104;
ResNet external uses archived seed20261012, preserving existing AUROC CIs.
Weighted exact-rank bootstrap for all 13 methods; independently verify
metrics using scalar tied-rank counts and materialized source draws.
Reuse the DenseNet original estimates/intervals bit-for-bit after checking
the same resamples reproduce them. Verify all 80 published ResNet Table 4
point entries to four decimals. Keep paired H-RCAM-minus-baseline intervals.

Independent audit: all 260 point entries; all new CIs for ResNet external
(104 marginal+96 paired) and DenseNet new baselines (36 marginal+36 paired).
Old DenseNet intervals are reused, not claimed as newly independently
re-estimated. Explicit valid-window 2D convolution verifies L1/L2/SSIM on
eight prespecified source positions times three backgrounds in every case;
also check map-to-cached-score correspondence with JSD/NCD recomputation.
Input hashes must remain unchanged. Use existing Python/SciPy/sklearn only.

remote-5080 over Tailscale: CPUs24–31, four numerical threads, GPU disabled,
durable job.py launcher. Freeze code/protocol before execution; preserve all
old results. Local display changes retain snapshots of previous drafts.

Agreed displays: Table2 separate columns Δmedian, medianCI, Δmean, meanCI,
Δmin, Δmax. Tables3/4/DenseNet: estimate on first line and CI underneath.
Table4 uses two dataset-pair blocks, DenseNet three, within one numbered
table each. No tiny font by default; render and inspect standalone PDF.
