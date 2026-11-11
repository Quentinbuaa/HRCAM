# R1C1 Table 3 extension: saved-map analysis

Requested after the initial DenseNet-121 component experiment. Freeze these
choices before inspecting the seven additional baseline rankings.

- Reuse the completed DenseNet-121 run, both original datasets and all valid
  own-predicted-class pairs. No training, new forward passes, new Grad-CAM
  extraction, target changes, or outcome-based exclusions.
- Evaluate all nine original candidates and H-RCAM, in Table 3 order:
  JSD, NCD, CD, KLD, SNO, BAT, SpatialW1, FAS, BAI, H-RCAM.
- Reuse unmodified `metrics.py` and `transport_metrics.py`. Preserve epsilon,
  top-20% stable tie handling, foreground mask, and grid=14 exact transport.
  Keep saved JSD/NCD authoritative after checking independent recomputation.
- Use raw candidate distances as defined, without fitted min-max scaling or
  clipping. AUROC/AP are invariant to strictly increasing affine scaling;
  clipping can introduce ties. Do not transplant ResNet-specific bounds onto
  DenseNet. Check that raw ResNet CD/KLD/SNO reproduce Table 3 to four decimals
  and document any differences. No favorable preprocessing variant selection.
- Compute 1000 paired source-cluster bootstrap resamples, seed 20261104, for
  all methods and all H-RCAM-minus-baseline differences. Keep every available
  background pair of a sampled source together; reuse the same source draw
  across methods. Retain existing point estimates and verify their intervals.
- Keep the main Markdown/LaTeX table in Table 3's compact AUROC/AP format;
  retain full intervals and paired differences in companion files. Bold each
  column's maximum at unrounded precision, regardless of method identity.
- Write to a new `full_baselines_v1` directory within the completed remote
  run. Preserve all initial records, summaries, map caches, and checkpoints.
  Check input hashes before and after analysis.
- Use approved logical CPUs 24-31, four numerical-library threads, and no GPU.
  Use existing Python 3.14/POT because the archived transport extension targets
  that interpreter. Record package versions and source hashes.
