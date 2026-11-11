# Attribution-target change diagnostics

The two `*_diagnostic_scores.jsonl` files retain source IDs, prediction
outcomes and target-specific derived scores. `summary.json`,
`component_target_comparisons.csv` and `rank_contributions.csv` support
the target-coupling analysis in RQ3.3. `code/` includes the frozen diagnostic
calculation, rank analysis and independent audit; no new models were fitted.

The three-convention results and confidence intervals used by current
Table 9 are in `../r2c3_fixed_gt_20261105_v1/`. The diagnostic analysis
examines the consequences of changing the explained class; it is not an
independent measure of predictive reliability.
