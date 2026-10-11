# Caltech-101 independent validation of the frozen H-RCAM

Run date: 2026-10-11. Caltech-101 was not used to choose the components or
fusion of H-RCAM. The sample, classifier, perturbations, metric, exclusions,
and analysis were fixed in [`CALTECH101_EXTERNAL_PROTOCOL.md`](CALTECH101_EXTERNAL_PROTOCOL.md)
before examining Caltech-101 outcomes. No manuscript text was changed in
this run.

## Data and procedure

- [CaltechDATA](https://data.caltech.edu/records/mzrjq-6wc02) describes 101
  object categories and author-clicked object outlines. The official download
  returned HTTP 403 in our computing environment, so we obtained a mirror of its original
  `caltech-101.zip`, verified its published MD5
  (`3138e1922a9193bfa496528edbbc45d0`), and recorded its SHA256 in the
  [manifest](results/caltech101_external_v1/caltech101_manifest.json).
- Excluding `BACKGROUND_Google`, the archive has 8677 object images and an
  annotation for every object image. We prespecified 10 test images in each
  of 101 classes, yielding 1010 test sources and 7667 training images. The
  [exact test IDs](results/caltech101_external_v1/caltech101_selected_ids.json) are saved.
- The classifier was an ImageNet-pretrained frozen ResNet-18 with a
  101-class logistic-regression head (C=1), trained only on the 7667 training
  images. Its clean test accuracy was **90.99%**. The metric was not fitted
  or recalibrated on Caltech-101.
- Each source underwent black, gray, and white background replacement outside
  its author-clicked contour. A mask QA contact sheet was checked without
  outcome information but is not redistributed because it contains original
  dataset images. Foreground pixels designated by
  the masks were asserted unchanged. Eleven sources fell outside the frozen
  5–95% mask-area eligibility range. All 3030 attempted pairs had valid
  own-predicted-class Grad-CAM maps; **999 sources / 2997 pairs** were eligible.
  Of these, **555 pairs (18.52%)** changed predicted label.

## External result

| Fixed measure | AUROC | AP |
|---|---:|---:|
| **H-RCAM** | **0.8739** | 0.6367 |
| JSD | 0.8663 | 0.6165 |
| NCD | 0.8556 | 0.6188 |
| D1 centroid displacement | 0.8000 | 0.5544 |
| D2 KL divergence | 0.8160 | 0.4743 |
| D3 top-attribution mismatch | 0.8172 | 0.5633 |
| Foreground shift | 0.6770 | 0.3798 |
| Background increase | 0.5187 | 0.2520 |
| BAT | 0.7889 | 0.4824 |
| SpatialW1 | 0.8460 | 0.5991 |

H-RCAM's median score was 0.0162 for prediction-consistent pairs and 0.0744
for inconsistent pairs. Its AUROC was 0.8397 on black, 0.8993 on gray, and
0.8840 on white backgrounds. The source-cluster bootstrap 95% AUROC interval
was **[0.8538, 0.8920]** (1000 resamples). Paired AUROC-difference intervals
were **[0.0109, 0.0262] versus NCD** and **[0.0016, 0.0138] versus JSD**.
These intervals condition on this classifier, sample, and mask source.

The prespecified fusion comparison was:

| Fixed JSD/NCD fusion | AUROC | AP |
|---|---:|---:|
| Minimum | 0.8732 | 0.6295 |
| Maximum | 0.8590 | 0.6218 |
| Arithmetic | 0.8682 | 0.6348 |
| Geometric | 0.8719 | **0.6371** |
| **Harmonic (H-RCAM)** | **0.8739** | 0.6367 |

The harmonic rule had the highest AUROC among these rules, but geometric
fusion had a slightly higher AP. The AUROC difference versus geometric
fusion was only about 0.0020; the source-cluster bootstrap difference
interval was [0.0004, 0.0037]. This is evidence for this fixed evaluation,
not a universal fusion-optimality claim.

## Interpretation and limits

Together with the separate [Flowers-102 external test](FLOWERS102_EXTERNAL_VALIDATION.md),
this supplies two untouched datasets beyond the original Oxford-IIIT Pet
and CUB-200-2011 selection/evaluation datasets. Caltech-101 adds broad object
categories but is not a second fine-grained animal benchmark; many images
are centered with simple backgrounds. Its class-wise test partition is
prespecified and independent of training, but is **our partition**, not an
official fixed test split. The endpoint is *prediction-label inconsistency*,
not verified incorrectness or a causal robustness ground truth. AP should
be read alongside the 18.52% inconsistency prevalence; it is not directly
comparable with Flowers-102's AP at 46.67% prevalence.

The mask is an author-clicked contour, not a perfect pixel-level instance
segmentation. Some unmasked pixels within a contour could be background and
some object parts could lie outside it. These limitations, one classifier
per dataset, and a single perturbation family limit generalization. A prior
TrashCan exploratory stress test found an
unfavorable H-RCAM-versus-NCD result on a substantially different derived
underwater object-crop task; it remains part of the research record. Thus the
claim should be *independent discrimination in two additional datasets*, not
universal superiority or proof that H-RCAM measures causal robustness.

## Audit trail

- [Frozen protocol](CALTECH101_EXTERNAL_PROTOCOL.md)
- [Per-pair outcomes and scores](results/caltech101_external_v1/external_records.jsonl)
- [Transport-candidate scores](results/caltech101_external_v1/external_transport.jsonl)
- [Full aggregate summary](results/caltech101_external_v1/external_summary.json)
- [Compact generated report](results/caltech101_external_v1/external_report.md)
- [Evaluation image IDs](results/caltech101_external_v1/caltech101_selected_ids.json)
- Full source images and heatmap arrays are not redistributed in this repository.
