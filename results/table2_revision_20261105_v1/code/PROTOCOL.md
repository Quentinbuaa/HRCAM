# Standalone revised Table 2

Author explicitly requested a Markdown Table 2 including L1/L2/SSIM and 95%
CIs replacing the pair-independent p-values. Use the checked Table 3 score
rows from `runs/table3_revision_20261105_v1`, preserving all 13 distance
definitions, classifiers, own-class targets, source/pair pools, probes and
frozen calibration. No new model inference, heatmaps, parameter selection,
environment changes or manuscript edits.

Original Table 2 has Delta median/mean/max/min, U, p and A_U. Preserve the
four descriptive differences and retain U only as a descriptive rank count.
Remove p entirely. Place a 95% source-cluster CI beside Delta median and
beside A_U; do not label the result as an independent-pair hypothesis test.
Split the Markdown into dataset panels for readability without dropping
original descriptive statistics. Preserve the original method order and add
L1/L2/SSIM before H-RCAM.

Delta statistics = prediction-inconsistent group minus prediction-consistent
group. U uses the consistent group first. A_U=1-U/(n0*n1), equivalently the
AUROC when prediction inconsistency is positive, with half-credit for ties.
Reuse Table 3's fully audited A_U/AUROC point estimates and CIs exactly.
Compute U from the exact rank effect, rounding only to the nearest half-integer.
Independently verify U by grouped integer rank counts, without running or
reporting a Mann-Whitney significance test.

Compute Delta median CIs from 1000 source-image-cluster bootstrap resamples,
seed 20261011, matching Table 3. Sample sources with replacement, retaining
all original background pairs and recomputing both outcome-group medians
within each draw. Same draws for every method. Use exact replicated-weight
order statistics for efficient medians and independently reconstruct every
median CI with ordinary Python statistics.median on materialized source draws.

Verify all 120 original descriptive/rank values (20 old rows x 6 statistics)
at their published precision against the verified submitted Table 2 fixture.
Independently verify all 156 new point entries (26 rows x 6), all 26 new
median-gap intervals, and exact reuse of the 26 Table 3 A_U intervals.
These CIs quantify sampling uncertainty for median/rank effects, not a
conversion of the former p-values or correction of development-set selection.
Max/min differences remain descriptive, without bootstrap-extreme claims.

Main artifact: Reviews/Table2_revised.md. Keep full-precision CSV/JSON,
source/protocol hashes and the independent audit. Original manuscript and
other table drafts remain unchanged. Execute only saved-score analysis on
remote-5080 via Tailscale 100.74.145.98, taskset CPUs 24-31, four numerical
threads, existing Python environment. No new training/inference is needed.
