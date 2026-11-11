# R2C3 Table8 and authorized R1C2 target-switch analysis

Author explicitly requested Table8/R2C3 first, then a more thorough R1C2
analysis. Table8 reuses the completed fixed-GT experiment, preserves the
eight submitted own/fixed point entries, adds four GT entries and existing
CIs. No rerun of that experiment and no manuscript edits.

R1C2 uses original ResNet18 Pet/CUB own/fixed cached maps and records only.
No training, inference, new attribution extraction, metric tuning or natural
backgrounds. Common valid own/fixed pairs must equal the audited Table8
three-way cohort: 2979 Pet/2887 CUB,993/963 sources. Preserve the two invalid
fixed-source CUB exclusions and all original predictions/masks.

For c0=f(source),c1=f(follow-up), define A=CAM(source,c0),
B=CAM(follow-up,c0), C=CAM(follow-up,c1). Compare A–B (background change
with fixed target), B–C (target change with fixed follow-up image), A–C
(own-class joint change). Recompute original A–B/A–C JSD,NCD,H as cache
identity checks; score B–C on all common pairs. Original [0,1] maps,
JSD smoothing1e-8/base2,NCD=(1-Pearson)/2,H=2JC/(J+C+1e-12).
Identical maps have zero distance. All contrasts use the same valid pool.

Do not call these distances additive causal components: H is nonlinear
and no additive/triangle decomposition is asserted. Target-only B–C is
zero by construction on prediction-consistent pairs; do not report its
AUROC as independent robustness validation. Quantify its distributions
and descriptive associations with score/rank changes on changed pairs.

Original-record audit established own/fixed negative scores are exactly
identical for all three components. Verify this again and B=C for Y=0.
For each positive pair, rank credit is fraction of the common negative
scores below it plus half-credit for ties. AUROC is the positive mean
credit. Therefore own-minus-fixed AUROC equals the mean positive credit
shift. Verify this exact accounting both on full data and every bootstrap
draw; it explains the discrimination change without implying a causal
percentage of attribution change.

1000 source-cluster resamples,seed20261011,retaining available background
pairs (some CUB sources have missing fixed pairs),same draws across targets
and components. Report own/fixed AUROC/AP and paired differences for
JSD,NCD,H. Check marginal replay against completed GT analysis. Decompose
the AUROC difference into positive-pair contributions partitioned by:
background(black/gray/white),source-correctness(correct/incorrect),and
correctness transition(correct→incorrect,incorrect→correct,incorrect→
different-incorrect). Each partition sums to the global difference.
These are contributions against the same full negative reference, not
within-stratum AUROCs or causal effect percentages.

Report changed-pair fractions where own score/rank credit rises/falls/
stays equal; score equality tolerance1e-12. Target-only mean/median/5/25/
50/75/95 quantiles are descriptive. Pearson/Spearman associations within
changed pairs have no pair-independent p-values. Fixed-GT/source-correct
results remain a separate diagnostic, not an accuracy-validity proof.

Independent audit: scalar rank counts, actual materialized source draws,
bisect-based credit accounting, fsum-based correlations/manual average
ranks, and independent SciPy JSD/Pearson map checks. Map audit sources:
eight evenly spaced common source IDs plus the first eight sources with
changed pairs and first eight with consistent pairs; all available probes
for their union. Retain every pair in performance estimates, not a sample.

Write only isolated run/results. Freeze protocol/code before new metrics.
Input hashes unchanged. remote-5080 via Tailscale,CPUs24–31,four numerical/
BLAS threads,two map-scoring workers,GPU disabled,durable launcher.
The result is a diagnostic of the tested own/fixed conventions, not a
demonstration of classifier reliability (R1C3) or causal robustness.
