# Agreed Table 2 layout: median and mean differences with CIs

Author authorized a revised standalone Markdown Table 2 and the response to
the Table 2 portion of R2C1. Columns: Measure, Δmedian [95% CI], Δmean
[95% CI], Δmax and Δmin. Include all 13 measures, retaining original order
and placing L1/L2/SSIM before H-RCAM. Remove U, p and A_U from the display.
Preserve the approved caption and all original descriptive point estimates.

Reuse the independently checked median CIs from
`runs/table2_revision_20261105_v1`. Input scores are the unchanged, audited
JSONL rows in `runs/table3_revision_20261105_v1`; verify their hashes against
the earlier Table 2 provenance. No new inference, maps, dependencies,
parameters, data selection, attribution targets or manuscript edits.

For new mean-gap intervals use the same 1000 source-image-cluster draws,
NumPy default_rng seed 20261011, source IDs sorted as in the original run.
Draw the original number of sources with replacement, retaining all three
background pairs for each sampled source. Recompute the prediction-consistent
and inconsistent means jointly within each draw and subtract inconsistent
minus consistent. Use percentile [2.5%,97.5%] bounds. Same draws for all
methods and both group summaries, preserving within-source dependence.

Use weighted sums for calculation, then independently materialize all 1000
source draws and use Python statistics.fmean to verify every mean interval.
Independently check all 104 descriptive point entries, exact reuse of all
26 previously audited median intervals, and 80 submitted Δ entries at their
published precision. Max/min remain descriptive without inferential CIs.
Do not infer between-method superiority from group-gap CIs or compare gaps
across incompatible score scales. Preserve selection-influenced framing.

Run on remote-5080 over Tailscale, CPUs 24–31, four numerical threads.
Freeze this protocol/code before execution. Preserve the earlier draft and
analysis record. Only after checks pass update Reviews/Table2_revised.md.
Draft a separate response and proposed local wording changes; make clear
that the submitted manuscript has not been edited and past-tense response
claims require subsequent manuscript integration.
