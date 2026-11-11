# Tables 4 and 5: expanded baseline comparisons

The ten `*_scores.jsonl` files contain all thirteen scores for DenseNet-121
on six datasets and ResNet-18 on four independent datasets. Only L1, L2
and SSIM distances were newly computed from the saved maps; the other
scores, valid pairs, fitted classifiers and metric were unchanged.

`summary.json` contains full-precision AUROC/AP, marginal 95% intervals
and paired H-RCAM-minus-baseline intervals. `metrics_with_ci.csv` supplies
the 260 point estimates and intervals used by current Tables 4 and 5.
These comparisons supersede earlier baseline-only display files elsewhere
in the repository. Dataset identifiers and counts are retained explicitly.

`INDEPENDENT_AUDIT.json` reconstructs all 260 points, 272 newly computed
intervals and 240 direct map-pair checks. Source-cluster bootstrap uses
1,000 draws, shared across methods (DenseNet seed 20261104; ResNet external
seed 20261012). `code/`, `FROZEN_PROTOCOL.json` and `PROVENANCE.json`
preserve the scientific computation and unchanged input hashes.
