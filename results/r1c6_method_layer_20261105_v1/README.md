# R1C6: attribution-method and layer sensitivity

**Complete and independently audited. No manuscript changes.**

Own-class targets; common valid pairs across all four configurations within each dataset.

| Configuration | Pet AUROC | Pet AP | CUB AUROC | CUB AP |
|---|---:|---:|---:|---:|
| Grad-CAM / layer4 (original) | 0.8952 | 0.6888 | 0.8054 | 0.8660 |
| Grad-CAM / layer3[0] | 0.7056 | 0.3057 | 0.6665 | 0.7550 |
| LayerCAM / layer4 | 0.7353 | 0.3902 | 0.6554 | 0.7460 |
| LayerCAM / layer3[0] | 0.8383 | 0.4787 | 0.7335 | 0.8037 |

## Setup and boundaries

The original fitted ResNet-18 classifiers, Pet/CUB source identities, masks, black/gray/white probes, target conventions, and H-RCAM score formula are unchanged. Grad-CAM and LayerCAM are evaluated at layer4 (7x7) and layer3[0] (14x14), a 2x2 method/layer comparison selected before results. No layer search, model training, recalibration, smoothing or cross-layer fusion was performed.

LayerCAM is a published nonnegative CAM method compatible with the unchanged H-RCAM inputs. It weights activations by local positive target gradients, while Grad-CAM uses spatially averaged channel gradients. The earlier layer equals layer3[-2] in ResNet-18; the authors recommend this type of layer choice to reduce grid effects from immediately following strided convolutions. See the [authors' code/guidance](https://github.com/PengtaoJiang/LayerCAM-jittor) and [LayerCAM paper](https://mftp.mmcheng.net/Papers/21TIP_LayerCAM.pdf).

## Common-set H-RCAM estimates and intervals

| Dataset | Target | Configuration | Pairs | AUROC (95% CI) | AP (95% CI) |
|---|---|---|---:|---:|---:|
| Oxford-IIIT Pet | own | Grad-CAM / layer4 (original) | 2968 | 0.8952 [0.8744, 0.9147] | 0.6888 [0.6313, 0.7426] |
| Oxford-IIIT Pet | own | Grad-CAM / layer3[0] | 2968 | 0.7056 [0.6701, 0.7394] | 0.3057 [0.2563, 0.3620] |
| Oxford-IIIT Pet | own | LayerCAM / layer4 | 2968 | 0.7353 [0.7009, 0.7701] | 0.3902 [0.3286, 0.4602] |
| Oxford-IIIT Pet | own | LayerCAM / layer3[0] | 2968 | 0.8383 [0.8123, 0.8634] | 0.4787 [0.4085, 0.5489] |
| Oxford-IIIT Pet | fixed | Grad-CAM / layer4 (original) | 2964 | 0.7960 [0.7625, 0.8255] | 0.5053 [0.4348, 0.5741] |
| Oxford-IIIT Pet | fixed | Grad-CAM / layer3[0] | 2964 | 0.6397 [0.5986, 0.6810] | 0.2505 [0.2124, 0.3000] |
| Oxford-IIIT Pet | fixed | LayerCAM / layer4 | 2964 | 0.6631 [0.6201, 0.6993] | 0.2991 [0.2393, 0.3624] |
| Oxford-IIIT Pet | fixed | LayerCAM / layer3[0] | 2964 | 0.6670 [0.6294, 0.7024] | 0.2752 [0.2238, 0.3415] |
| CUB-200-2011 | own | Grad-CAM / layer4 (original) | 2863 | 0.8054 [0.7843, 0.8264] | 0.8660 [0.8463, 0.8849] |
| CUB-200-2011 | own | Grad-CAM / layer3[0] | 2863 | 0.6665 [0.6403, 0.6918] | 0.7550 [0.7255, 0.7863] |
| CUB-200-2011 | own | LayerCAM / layer4 | 2863 | 0.6554 [0.6270, 0.6815] | 0.7460 [0.7150, 0.7750] |
| CUB-200-2011 | own | LayerCAM / layer3[0] | 2863 | 0.7335 [0.7079, 0.7571] | 0.8037 [0.7736, 0.8304] |
| CUB-200-2011 | fixed | Grad-CAM / layer4 (original) | 2834 | 0.7628 [0.7375, 0.7834] | 0.8427 [0.8188, 0.8635] |
| CUB-200-2011 | fixed | Grad-CAM / layer3[0] | 2834 | 0.6599 [0.6332, 0.6869] | 0.7559 [0.7265, 0.7864] |
| CUB-200-2011 | fixed | LayerCAM / layer4 | 2834 | 0.6025 [0.5730, 0.6297] | 0.7040 [0.6695, 0.7357] |
| CUB-200-2011 | fixed | LayerCAM / layer3[0] | 2834 | 0.6345 [0.6055, 0.6606] | 0.7334 [0.7015, 0.7651] |

Confidence intervals use 1000 paired source-cluster resamples (seed 20261105), retaining each source's available probes together. All configurations share the same source draws within a comparison. Paired configuration-minus-original AUROC/AP differences and their CIs are in [summary.json](analysis/summary.json). These are comparison-wise intervals, not simultaneous guarantees across all comparisons.

## Coverage and all-valid estimates

| Dataset | Target | Configuration | Eligible pairs | Valid pairs | Invalid pairs | All-valid AUROC | AP |
|---|---|---|---:|---:|---:|---:|---:|
| Oxford-IIIT Pet | own | Grad-CAM / layer4 (original) | 2979 | 2979 | 0 | 0.8953 | 0.6888 |
| Oxford-IIIT Pet | own | Grad-CAM / layer3[0] | 2979 | 2968 | 11 | 0.7056 | 0.3057 |
| Oxford-IIIT Pet | own | LayerCAM / layer4 | 2979 | 2979 | 0 | 0.7357 | 0.3911 |
| Oxford-IIIT Pet | own | LayerCAM / layer3[0] | 2979 | 2979 | 0 | 0.8377 | 0.4783 |
| Oxford-IIIT Pet | fixed | Grad-CAM / layer4 (original) | 2979 | 2979 | 0 | 0.7973 | 0.5095 |
| Oxford-IIIT Pet | fixed | Grad-CAM / layer3[0] | 2979 | 2964 | 15 | 0.6397 | 0.2505 |
| Oxford-IIIT Pet | fixed | LayerCAM / layer4 | 2979 | 2979 | 0 | 0.6645 | 0.3043 |
| Oxford-IIIT Pet | fixed | LayerCAM / layer3[0] | 2979 | 2979 | 0 | 0.6682 | 0.2808 |
| CUB-200-2011 | own | Grad-CAM / layer4 (original) | 2889 | 2889 | 0 | 0.8066 | 0.8684 |
| CUB-200-2011 | own | Grad-CAM / layer3[0] | 2889 | 2863 | 26 | 0.6665 | 0.7550 |
| CUB-200-2011 | own | LayerCAM / layer4 | 2889 | 2889 | 0 | 0.6567 | 0.7495 |
| CUB-200-2011 | own | LayerCAM / layer3[0] | 2889 | 2889 | 0 | 0.7353 | 0.8074 |
| CUB-200-2011 | fixed | Grad-CAM / layer4 (original) | 2889 | 2887 | 2 | 0.7658 | 0.8484 |
| CUB-200-2011 | fixed | Grad-CAM / layer3[0] | 2889 | 2836 | 53 | 0.6600 | 0.7562 |
| CUB-200-2011 | fixed | LayerCAM / layer4 | 2889 | 2889 | 0 | 0.6034 | 0.7093 |
| CUB-200-2011 | fixed | LayerCAM / layer3[0] | 2889 | 2889 | 0 | 0.6359 | 0.7393 |

The original Grad-CAM-layer4 all-valid estimates reproduce the submitted main/target results. Common-set estimates can differ if another configuration excludes pairs. Table 3 has not been changed.

## Score/rank agreement with the original configuration

| Dataset | Target | Alternative | Spearman | Median absolute H difference | 95th-percentile absolute difference |
|---|---|---|---:|---:|---:|
| Oxford-IIIT Pet | own | Grad-CAM / layer3[0] | 0.2614 | 0.143793 | 0.513214 |
| Oxford-IIIT Pet | own | LayerCAM / layer4 | 0.6678 | 0.004000 | 0.043279 |
| Oxford-IIIT Pet | own | LayerCAM / layer3[0] | 0.5324 | 0.012323 | 0.039406 |
| Oxford-IIIT Pet | fixed | Grad-CAM / layer3[0] | 0.2231 | 0.140230 | 0.498244 |
| Oxford-IIIT Pet | fixed | LayerCAM / layer4 | 0.6523 | 0.003754 | 0.030740 |
| Oxford-IIIT Pet | fixed | LayerCAM / layer3[0] | 0.4399 | 0.012016 | 0.036880 |
| CUB-200-2011 | own | Grad-CAM / layer3[0] | 0.1573 | 0.203180 | 0.569130 |
| CUB-200-2011 | own | LayerCAM / layer4 | 0.6162 | 0.010112 | 0.070699 |
| CUB-200-2011 | own | LayerCAM / layer3[0] | 0.4658 | 0.018025 | 0.057399 |
| CUB-200-2011 | fixed | Grad-CAM / layer3[0] | 0.2453 | 0.203643 | 0.568601 |
| CUB-200-2011 | fixed | LayerCAM / layer4 | 0.5571 | 0.008976 | 0.081728 |
| CUB-200-2011 | fixed | LayerCAM / layer3[0] | 0.4465 | 0.017072 | 0.065729 |

These descriptive correlations contain no pair-independent p-values. They measure agreement between method/layer settings, not the separate R2C4 correlation between JSD and NCD. Similar AUROC does not by itself establish unchanged score scales, case rankings, or attribution maps.

## Interpretation and released files

The comparison covers two CAM methods, two layers and two datasets with
the classifier and probes fixed. It shows method/layer dependence rather
than invariance across arbitrary explanation methods or layers. The primary
Grad-CAM/layer4 setting is retained. The experiment is summarized in
manuscript Section 5; no additional manuscript table is introduced.

- [All common-set estimates and CIs](table_with_ci.csv)
- [Full analysis](analysis/summary.json), including all-valid and matched pools
- [Independent audit](INDEPENDENT_AUDIT.json)
- `analysis/*_common_scores.jsonl`: cached pair-level scores for verification
- `pets_v1/records.jsonl`, `cub_v1/records.jsonl`: detailed derived records
- `code/`, `FROZEN_PROTOCOL.json`: executed extraction/analysis and protocol

Images, masks, checkpoints and native heatmap arrays are not included.
