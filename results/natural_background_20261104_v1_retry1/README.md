# RQ3.1 / Table 7: natural-background replacement

**Status: completed and independently audited.**

The additional photographic-background condition uses the original fitted ResNet-18 models and supplied masks on 993 Pet and 963 CUB sources. Each source receives one unique donor, with no donor reuse across datasets. Every foreground pixel is preserved. No classifier was retrained and no metric/calibration was refitted.

## Main comparison: own-class targets

| Method | Pet AUROC | Pet AP | CUB AUROC | CUB AP |
|---|---:|---:|---:|---:|
| JSD | 0.8889 | 0.6552 | 0.8038 | 0.9172 |
| NCD | 0.8870 | 0.7041 | 0.8080 | 0.9147 |
| CD | 0.8032 | 0.5548 | 0.7062 | 0.8707 |
| KLD | 0.8094 | 0.4151 | 0.6938 | 0.8442 |
| SNO | 0.8603 | 0.6507 | 0.7683 | 0.8913 |
| BAT | 0.8303 | 0.5510 | 0.7403 | 0.8915 |
| SpatialW1 | 0.8612 | 0.6149 | 0.7645 | 0.8960 |
| FAS | 0.6788 | 0.4079 | 0.6467 | 0.8468 |
| BAI | 0.6134 | 0.3511 | 0.6382 | 0.8385 |
| H-RCAM | **0.9020** | **0.7150** | **0.8186** | **0.9224** |

H-RCAM has the highest observed AUROC/AP in all four columns. These point-estimate ranks do not mean that every paired AP difference is distinguishable from zero.

## H-RCAM intervals and targeting sensitivity

| Dataset | Target | Valid pairs | Inconsistency | AUROC (95% CI) | AP (95% CI) |
|---|---|---:|---:|---:|---:|
| Oxford-IIIT Pet | own | 993 | 16.01% | 0.9020 [0.8753, 0.9253] | 0.7150 [0.6474, 0.7782] |
| Oxford-IIIT Pet | fixed | 993 | 16.01% | 0.7899 [0.7494, 0.8289] | 0.5054 [0.4276, 0.5798] |
| CUB-200-2011 | own | 963 | 73.21% | 0.8186 [0.7894, 0.8460] | 0.9224 [0.9027, 0.9381] |
| CUB-200-2011 | fixed | 962 | 73.18% | 0.7623 [0.7315, 0.7955] | 0.8990 [0.8785, 0.9193] |

There are no invalid own-class pairs. The CUB fixed-source analysis has one invalid map pair (962 valid pairs); it is retained in the records and disclosed, rather than silently removed.

**Target dependence remains:** under fixed-source targeting, JSD has higher AUROC/AP than H-RCAM on both datasets. The natural-background result supports transfer of the original own-class comparison; it does not resolve the reviewers' attribution-target concern.

| Fixed-source method | Pet AUROC | Pet AP | CUB AUROC | CUB AP |
|---|---:|---:|---:|---:|
| JSD | 0.8371 | 0.5886 | 0.8229 | 0.9314 |
| NCD | 0.7270 | 0.4265 | 0.7064 | 0.8646 |
| H-RCAM | 0.7899 | 0.5054 | 0.7623 | 0.8990 |

## Paired uncertainty

All intervals use 1000 paired source resamples, seed 20261104. There is one natural-background pair per source, so source resampling here also resamples individual natural pairs. The intervals condition on the frozen donor-selection/assignment protocol; there were no repeated donor assignments.

| Dataset | H-RCAM minus baseline | Metric | Difference | 95% paired CI |
|---|---|---|---:|---:|
| Oxford-IIIT Pet | JSD | auroc | +0.013039 | [+0.000895, +0.024998] |
| Oxford-IIIT Pet | JSD | ap | +0.059823 | [+0.034399, +0.088063] |
| Oxford-IIIT Pet | NCD | auroc | +0.014954 | [+0.005455, +0.024542] |
| Oxford-IIIT Pet | NCD | ap | +0.010968 | [-0.012533, +0.037159] |
| CUB-200-2011 | JSD | auroc | +0.014806 | [+0.000921, +0.028233] |
| CUB-200-2011 | JSD | ap | +0.005165 | [-0.001399, +0.012235] |
| CUB-200-2011 | NCD | auroc | +0.010572 | [+0.001326, +0.019660] |
| CUB-200-2011 | NCD | ap | +0.007738 | [+0.001733, +0.014059] |

The own-class AUROC differences against JSD and NCD have positive comparison-wise 95% intervals on both datasets. Pet AP versus NCD and CUB AP versus JSD have intervals containing zero. These are individual comparisons, without a simultaneous multiple-comparison guarantee. Full method intervals, paired differences, and group-median intervals are retained in [summary.json](analysis/summary.json) and [table_with_ci.csv](analysis/table_with_ci.csv).

## Comparison with the original controlled probes

| Dataset | Background | Pairs | Inconsistency | H-RCAM AUROC | AP |
|---|---|---:|---:|---:|---:|
| Oxford-IIIT Pet | black | 993 | 18.43% | 0.8779 | 0.7040 |
| Oxford-IIIT Pet | gray | 993 | 13.80% | 0.9126 | 0.6898 |
| Oxford-IIIT Pet | white | 993 | 14.50% | 0.8989 | 0.6768 |
| Oxford-IIIT Pet | pooled_constant | 2979 | 15.58% | 0.8953 | 0.6888 |
| Oxford-IIIT Pet | natural | 993 | 16.01% | 0.9020 | 0.7150 |
| CUB-200-2011 | black | 963 | 72.07% | 0.7850 | 0.8912 |
| CUB-200-2011 | gray | 963 | 57.32% | 0.8241 | 0.8535 |
| CUB-200-2011 | white | 963 | 60.44% | 0.8031 | 0.8585 |
| CUB-200-2011 | pooled_constant | 2889 | 63.27% | 0.8066 | 0.8684 |
| CUB-200-2011 | natural | 963 | 73.21% | 0.8186 | 0.9224 |

These cross-background comparisons are descriptive. The inconsistency outcomes differ across backgrounds. In particular, CUB natural-background AP is evaluated at 73.21% inconsistency prevalence, versus 63.27% for the pooled constant-color pairs; its higher AP alone is not evidence of stronger discrimination.

## Donor provenance and verification

The official BG-20k Drive endpoint was inaccessible. We used the public resized copy [unography/BG-20k-1200px](https://huggingface.co/datasets/unography/BG-20k-1200px), commit `f94ef4f1c95fd81c088bc0a0ee3b0bd451b8109c`, first parquet shard (3334 candidate images). The 497094652-byte download matched the advertised LFS SHA256. All 17 donor contact sheets were reviewed before inference; 183 obvious animal/animal-depiction or artwork/graphic/gradient rows were excluded. Sixty-four final composites were inspected before freezing the assignment. This thumbnail QA does not guarantee absence of every small or occluded object.

Selection used a fixed random permutation, seed 20261104, before model outputs. Donors were center-cropped with Pillow ImageOps.fit, bilinear, to 224x224. Foreground preservation was checked for all 1956 pairs. The first shard is a bounded donor pool; it is not a claim of representative sampling of the full BG-20k distribution.

Credit BG-20k to Li et al., *Bridging Composite and Real: Towards End-to-end Deep Image Matting*, as required by the [official dataset release](https://github.com/JizhiziLi/GFM#bg-20k).

The first extraction attempt stopped at a source-CAM replay check because cuDNN numerical behavior differed when a source was moved from the original final 7-image batch into a 16-image batch. The isolated retry kept the failed attempt, frozen donors, models, masks, and metric unchanged. It reproduced the exact original batch order/sizes with **zero map error on both datasets**, then reused those verified archived source maps. The independent audit matched the actual source/follow-up batch contexts.

- Original checkpoints and records are unchanged; own-class original predictions all match.
- Independent closed-form Grad-CAM replay: 16 sources per dataset; maximum map error below 4.2e-7.
- Independent SciPy JSD/Pearson checks on all valid map pairs: maximum error below 1.6e-15.
- All 52 reported AUROC/AP point entries independently reconstructed with pure-Python grouped rank counts; maximum error below 1.5e-15.
- All 44 core confidence intervals independently reconstructed with the same source draws and pure-Python rank/count metrics; maximum endpoint error below 1e-15. See [interval audit](INDEPENDENT_INTERVAL_AUDIT.json).
- CPU affinity 24-31 and four numerical threads.
- No model retraining, metric retuning, favorable-run filtering or imperfect-mask experiments.

CD/KLD/SNO use the original frozen calibration transform. Raw unclipped results and clipping counts are retained; they agree with the reported discrimination values at four decimal places.

## Interpretation and released files

This condition is integrated into RQ3.1 / Table 7 alongside the constant-color
probes. It supports discrimination under the tested photographic replacements;
it does not test imperfect masks or arbitrary natural-background distributions.

- `analysis/summary.json`, `analysis/table_with_ci.csv`: all estimates/intervals
- `analysis/*_scores.jsonl`: own-class pair scores for verification
- Dataset `records.jsonl` files: full derived extraction records
- `prepared/donor_assignment.json` and associated preparation checks:
  fixed donor identifiers, exclusions and preprocessing provenance
- `code/`, `FROZEN_INPUTS.json`: extraction/analysis and frozen inputs
- [Main audit](INDEPENDENT_AUDIT.json), [interval audit](INDEPENDENT_INTERVAL_AUDIT.json)

Donor images and composites are not redistributed. Acquire source/donor data
from the providers under their respective terms for map-level reproduction.
The final Table 7 color/probe intervals are released in
`../background_probe_ci_20261109_v1/`.
