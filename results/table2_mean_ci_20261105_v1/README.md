# Table 2: group differences with bootstrap intervals

`summary.json` and `Table2_with_ci.csv` contain all 26 dataset/measure rows
at full precision. Each reports prediction-inconsistent minus consistent
differences in medians and means, their 95% confidence intervals, and
descriptive extrema. Extrema have no inferential intervals.

The analysis uses 1,000 source-image cluster bootstrap samples, retaining
all three background pairs from each sampled source together. Both group
summaries are recomputed within each draw; the same draws are used across
measures. Inputs are the cached score JSONLs in
`../table3_revision_20261105_v1/` and the archived median intervals in
`../table2_revision_20261105_v1/`. `PROVENANCE.json` records their hashes.

`INDEPENDENT_AUDIT.json` verifies 104 descriptive entries, 26 mean-gap
intervals, exact reuse of 26 median-gap intervals and agreement with the
previously published estimates at their printed precision. `code/` and
`FROZEN_PROTOCOL.json` preserve the executed computation. These are
comparison-wise, selection-influenced development-set intervals.
