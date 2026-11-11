# Table7 uncertainty and JSD/NCD relationship analysis

Author approved creating the new Table7, a correlation figure, and the
related component-ratio analysis. Saved scores only; no fitting, inference,
new maps, data exclusions, tuning or canonical manuscript edits.

Table7: original ResNet18 own-class black/gray/white pairs on Pet/CUB,
2979/2889 pairs from993/963 sources. Use checked Table3 score JSONLs.
JSD and NCD remain unchanged; minimum/maximum/arithmetic/geometric formulas
match Section3.5/RQ3.2; H=2*J*C/(J+C+1e-12). Preserve all28 submitted
median-gap/AUROC point entries at four decimals. Replace the descriptive
rank column with paired H-minus-rule AUROC differences and CIs.

1000 source-image-cluster bootstrap resamples, seed20261011, matching
Tables2/3. Retain all three background pairs for every sampled source;
same draws across all rules. Compute median-group-gap CIs, marginal AUROC
CIs, and paired H-minus-rule AUROC CIs. Do not infer paired differences
from overlapping marginal intervals. Weighted exact rank/median calculation;
independent audit reconstructs every interval from materialized source
draws, scalar tied-rank counts and Python statistics.median. Verify14 median
gaps,14 AUROCs and12 paired differences, and40 intervals. Check replay
against prior Table3 AUROC intervals and prior Table2 H median interval.

Component analysis: original ResNet18 on all six datasets, same valid
own-class pairs, original JSD/NCD/H scores. Pet/CUB from Table3 run, four
verification datasets from completed table_layouts_baselines retry1.
Report Pearson(J,C), average-tied-rank Spearman(J,C), Spearman(H,J),
Spearman(H,C), per dataset plus pooled-development and pooled-all-six.
Pooled correlations are descriptive and may reflect dataset differences;
the per-dataset estimates are primary. No pair-independent p-values.

Report the J/C distribution (5/25/50/75/95th percentiles), J<C/J=C/C<J
fractions and the min(J,C)/max(J,C) distribution. Ratios with zero
denominator are missing and their counts are explicit. Use predeclared
0.1 imbalance threshold only as a descriptive tenfold-scale summary.
Report zero-component/both-zero counts and H=0 fraction. Do not select
thresholds by their relationship with outcomes or claim correlation alone
proves predictive complementarity. Audit correlations by independent
fsum-based covariance and manual average ranks; audit ratio quantiles
with a separate scalar/list implementation.

Figure1: six scatter panels in dataset order Pet,CUB,Flowers,Caltech,VOC,
Food. All pairs shown, colored by prediction consistency, with y=x guide,
Pearson/Spearman annotation and shared linear axes including all data.
Do not clip/downsample to improve appearance. Publication PDF plus PNG.
Figure2: ECDF of min/max ratio per dataset, same six cases, x in[0,1],
vertical0.1 reference. Zeros valid; both-zero missing and disclosed.
No correlation or ratio CIs are claimed by this descriptive figure.

Existing remote-5080 .venv, CPUs24–31, four numerical-library threads,
GPU disabled, durable job.py. Freeze code/protocol and preserve input
hashes. Do not rerun or hide mixed findings. CIs are conditional on the
selected score/rules and do not remove development-set selection bias.
Deliver standalone Markdown table/report and figures, not manuscript edits.
