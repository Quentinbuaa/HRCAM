# R1C1: DenseNet-121 Table 4 extension

Authorized by the user after reviewing the Pet/CUB results. This protocol is
fixed before fitting or evaluating DenseNet on the four additional datasets.

- Evaluate all four existing external tasks: Flowers-102, Caltech-101,
  single-object-class VOC 2012 and single-food-class UECFoodPixComplete.
  Reuse the exact existing prepared training/evaluation arrays, source order,
  class mappings, masks, and source-selection manifests. Verify original
  per-pair record hashes against the local verified copies.
- Use the same official frozen DenseNet-121 V1 checkpoint, ImageNet
  normalization, final activated spatial Grad-CAM representation, C=1
  logistic-regression head, solver=lbfgs, max_iter=1000, and batch size 16
  as in the completed Pet/CUB experiment. Seed=20261011, matching the original
  external protocol. Fit each head on its designated training images only.
- No metric recalibration, model/dataset selection, alternative training,
  outcome-based filtering, or post-result adjustments to H-RCAM. Report all
  four datasets, including weaker results.
- Generate black/gray/white follow-ups with unchanged foreground pixels.
  Save own-predicted and fixed-source-predicted maps. Apply the original map
  validity and inclusive 5-95% mask-area rules; report coverage explicitly.
- Primary Table 4 uses own-class targets and all nine original baseline
  definitions plus unchanged H-RCAM. Use raw defined distances without
  fitted clipping; exact transport uses grid=14. Retain 1000 source-cluster
  bootstrap intervals (seed=20261104) and paired differences, separately from
  the compact table.
- Use CPU affinity 24-31 and four numerical threads throughout. GPU extraction
  uses the original Python 3.10 environment. CPU transport analysis uses the
  existing Python 3.14/POT environment. Do not alter either environment.
- Write only a new external run. Preserve original ResNet and DenseNet runs.
  Save provenance, frozen-feature checks, gradient checks, complete records,
  full-resolution map caches, all rankings, and background-job status/logs.
- The background-sensitivity experiment is a separate proposed task; this
  authorization does not start natural-background or mask-error experiments.
- Do not edit the manuscript. Produce standalone Table 4-style artifacts.
