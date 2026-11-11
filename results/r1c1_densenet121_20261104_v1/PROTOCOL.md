# R1C1: architecture-transfer experiment, fixed before execution

Status: prepared on 4 November 2026; no DenseNet results inspected at preparation.

Question for proposed RQ1.4: Do H-RCAM's discrimination and relative gains
persist with a second CNN backbone on the original two development datasets?

## Architecture and scope

- Use Torchvision DenseNet-121 with `DenseNet121_Weights.IMAGENET1K_V1`.
  Its dense feature concatenation provides an architectural contrast with
  ResNet-18's residual addition. Spatial features and global average pooling
  permit the same class-specific Grad-CAM comparison and fitted linear head.
- References: https://arxiv.org/abs/1608.06993 and
  https://docs.pytorch.org/vision/main/models/generated/torchvision.models.densenet121.html
- Evaluate Oxford-IIIT Pet and CUB-200-2011, using the exact saved training
  indices and holdout source indices from `runs/pets_v1/splits.json` and
  `runs/cub_v1/splits.json`. These are the manuscript's two development
  datasets, not new independent datasets. Do not substitute the separate
  within-run development image split for the submitted evaluation images.
- This is a two-dataset extension with one additional backbone, not an
  architecture search. Report both datasets regardless of score ordering.

## Fixed experimental settings

1. Run on remote-5080 only on logical CPUs 24-31, using the existing `job.py`
   launcher. Set OMP, OpenBLAS, MKL, and PyTorch numerical threads to four.
   Use the original Python 3.10 `.venv`; do not modify other environments.
2. Preserve all original files. Write new checkpoints, caches, records, and
   summaries under a new run directory. Record original file hashes and check
   them again after the run. Refuse to replace a completed run.
3. Freeze all pretrained feature parameters and batch-normalization buffers.
   Fit only a multinomial logistic-regression head, C=1, solver=lbfgs,
   max_iter=1000, on the original head-training images. Record convergence.
   Do not tune the head, image subsets, architecture, targets, or metric by
   the evaluation results.
4. Reuse the original prepared 224x224 images and annotated masks, ImageNet
   normalization, and black/gray/white replacements before normalization.
   Assert that preserved foreground pixels are identical.
5. Extract Grad-CAM from the final activated DenseNet spatial representation
   (`relu(features.norm5)`), before global average pooling. Keep maps in
   float32; use bilinear interpolation with align_corners=False and each
   map's existing independent [0,1] normalization. Explain logits, not softmax.
   Compute actual autograd gradients with respect to this representation.
6. Evaluate own-predicted-class maps as the comparison matching the submitted
   main experiment. Also retain fixed-source-predicted-class maps as a
   diagnostic. This is not a change to the submitted primary convention.
7. Keep JSD in base 2, smoothing 1e-8, NCD=(1-Pearson)/2, and
   H-RCAM=2*JSD*NCD/(JSD+NCD+1e-12). Do not recalibrate or reselect components.
8. Apply the original 5-95% inclusive foreground-area and map-validity rules.
   Report invalid maps, analyzed sources/pairs, clean and transformed
   accuracy, and label-inconsistency prevalence. Do not exclude bad predictions.
9. Report AUROC and AP for H-RCAM, JSD, and NCD, with 1,000 source-cluster
   bootstrap resamples and paired H-RCAM-minus-component intervals. Keep all
   background pairs from each sampled source. Report each target convention
   on its valid set and their paired comparison on the common valid set.
   These intervals do not quantify training-seed or backbone uncertainty.
10. Include the existing ResNet-18 results as context, clearly identifying the
    different prediction-inconsistency outcomes produced by different models.
    Cross-backbone AUROC differences are not a direct classifier-quality test.
11. Following the pre-execution design audit, show paired component-difference
    intervals in the readable report, and check results on common valid pairs
    across both backbones. This is a component comparison (JSD and NCD), not a
    new claim of superiority over all nine candidates. Do not interpret a
    nonsignificant difference as equivalence between metrics or architectures.

## Verification and reporting

- Before full extraction, check the detached-feature Grad-CAM calculation
  against a full end-to-end autograd extraction on actual source images.
  Check the global-average-pooling/linear-head CAM identity as a second check.
- Independently check component scores against SciPy JSD and Pearson formulas.
- Verify that own/fixed maps and scores coincide when predictions coincide.
- Retain per-pair records, all evaluated source IDs, checkpoints, full-resolution
  map caches, source hashes, environment, CPU affinity, and readable reports.
- A mixed outcome is reportable: retained discrimination and superiority to
  individual components are different questions. Do not omit an unfavorable
  dataset or change the metric to restore its rank.
- Do not edit `Entropy-latest/Entropy-clean.tex` in this task. Prepare evidence
  and a suggested RQ1.4 presentation for author review first.
