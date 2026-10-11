# Oxford Flowers-102 independent validation of the frozen H-RCAM

Run date: 2026-10-11. This analysis responds to the concern that Oxford-IIIT
Pet and CUB-200-2011 were used both to choose H-RCAM's components/fusion and
to report their performance. Flowers-102 was not used for that earlier choice.
The metric and evaluation protocol were recorded in
[`FLOWERS102_EXTERNAL_PROTOCOL.md`](FLOWERS102_EXTERNAL_PROTOCOL.md) before
reading any Flowers-102 model or metric outcomes. The manuscript was not
edited for this run.

## Dataset and fixed procedure

- [Oxford's original Flowers-102 distribution](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html):
  102 flower categories; official images, labels, splits and segmentations.
  The original images/labels/split file pass the published torchvision MD5
  values. File SHA256 hashes are in
  [`flowers_manifest.json`](results/flowers102_external_v1/flowers_manifest.json).
- Fit a 102-class logistic-regression head (C=1) to frozen ImageNet-pretrained
  ResNet-18 features on the official train+validation images (2040). No test
  images were used to fit the classifier or select the metric. Clean accuracy
  on the prespecified test sample was **87.55%**. Training accuracy was 100%,
  so the classification head may overfit; no tuning was performed after this
  observation.
- Before outcomes, select 10 official-test images per class, seed 20261011:
  **1020 source images**. All undergo black, gray and white constant-color
  background replacement. The exact IDs are in
  [`selected_ids.json`](results/flowers102_external_v1/selected_ids.json).
- The Oxford-provided segmentation JPEGs encode background as near-pure blue.
  A fixed RGB distance threshold from `(0,0,255)` of 110 was chosen after
  outcome-blind visual inspection. Only mask-designated background pixels are
  replaced; preserved pixels are checked for exact equality. Raw and decoded
  contact sheets were inspected but are not redistributed here because they
  contain original dataset images.
- Exclude masks occupying <5% or >95% of image area, as in the original
  Pet/CUB protocol. This removed **33 sources** and all their 99 pairs.
  Grad-CAM was valid for all 3060 attempted own-class pairs. Primary analysis
  thus includes **987 sources / 2961 pairs**, of which **1382 (46.67%)**
  changed predicted label. The endpoint is prediction inconsistency, not
  correctness or causal vulnerability.

## Independent result

| Measure | AUROC | AP |
|---|---:|---:|
| **H-RCAM** | **0.8361** | 0.8078 |
| NCD | 0.8173 | 0.7933 |
| JSD | 0.8230 | 0.7828 |
| D1 centroid displacement | 0.7450 | 0.7223 |
| D2 KL divergence | 0.7719 | 0.6857 |
| D3 top-attribution mismatch | 0.7837 | 0.7537 |
| Foreground shift | 0.6441 | 0.6267 |
| Background increase | 0.5082 | 0.5203 |
| BAT | 0.7395 | 0.6977 |
| SpatialW1 | 0.7969 | 0.7616 |

H-RCAM's source-cluster bootstrap 95% AUROC interval is **[0.8165, 0.8531]**.
The paired AUROC difference interval is **[0.0097, 0.0279] versus NCD** and
**[0.0068, 0.0193] versus JSD** (1000 source-level resamples, fixed seed).
Its median score was 0.0218 for prediction-consistent pairs and 0.0697 for
prediction-inconsistent pairs. H-RCAM AUROC by background was 0.8307 (black),
0.8581 (gray) and 0.8183 (white). All backgrounds were included in the pooled
analysis, not chosen by outcome.

The prespecified secondary fusion comparison is more nuanced:

| Fixed fusion rule | AUROC | AP |
|---|---:|---:|
| Minimum | 0.8304 | 0.7942 |
| Maximum | 0.8188 | 0.7934 |
| Arithmetic mean | 0.8281 | 0.8028 |
| Geometric mean | 0.8339 | **0.8082** |
| **Harmonic mean (H-RCAM)** | **0.8361** | 0.8078 |

Harmonic minus geometric AUROC is about 0.0022, with a source-cluster
bootstrap 95% difference interval of **[-0.0004, 0.0048]**. This does **not**
establish that the harmonic rule is better than geometric fusion on new data;
geometric fusion has marginally higher AP. The result supports the frozen
H-RCAM score as a useful discriminator and its advantage over the individual
NCD and JSD components, not a claim of universal fusion-rule optimality.

## Quality and interpretation limits

The Oxford-provided segmentations are not independently confirmed manual
ground-truth masks. Visual inspection found some omitted petals/stems and some
retained background, in addition to the 33 objectively ineligible masks.
Background replacement is therefore not perfectly foreground-preserving in
every source image. The intervals are conditional on this one classifier,
mask source and fixed test sample; they do not include model-training or
segmentation uncertainty. The high inconsistency prevalence (46.67%) should
be reported alongside AP. Prediction inconsistency is the external evaluation
label, not a measure of human-labelled failure.

The results are ready for manuscript adaptation after author review. The
discussion should replace the previous statement that H-RCAM was *not tested
on a new dataset*, while retaining the honest limitations above. Claims of
harmonic fusion being generally better than geometric fusion should be
narrowed rather than suppressed.

For completeness, an earlier TrashCan exploratory stress test
on a *derived* underwater-video, three-class object-crop task found H-RCAM
AUROC 0.629 versus NCD 0.682. That task differs materially from the paper's
fine-grained whole-image classification setting and is not substituted for
Flowers-102, but its unfavorable result remains in the research record and
precludes a universal superiority claim.

## Audit trail

- [Frozen protocol](FLOWERS102_EXTERNAL_PROTOCOL.md)
- [Per-pair metrics and outcomes](results/flowers102_external_v1/external_records.jsonl)
- [Per-pair transport candidates](results/flowers102_external_v1/external_transport.jsonl)
- [Full summary, including bootstrap intervals](results/flowers102_external_v1/external_summary.json)
- [Compact generated report](results/flowers102_external_v1/external_report.md)
- Full source images and heatmap arrays are not redistributed in this repository.
