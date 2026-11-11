# Figure 5 and current Table 8: components and fusion

The run name and `Table7_with_ci.csv` retain their earlier numbering;
that CSV is the full-precision input to current manuscript Table 8.

- `fusion_summary.json`, `Table7_with_ci.csv`: seven rules per development
  dataset, median-gap/AUROC intervals and paired H-RCAM-minus-rule intervals.
- `component_summary.json`, `component_correlations.csv`,
  `component_ratios.csv`: Pearson/Spearman correlations, q/R quantiles and
  zero-denominator counts, by dataset and for the pooled data.
- Six `*_component_scores.jsonl` and two `*_fusion_scores.jsonl`: source
  identifiers, outcomes and full-precision scores.
- `presentation_v2/Figure_JSD_NCD_correlation.pdf`: Figure 5 scatter panels;
  `presentation_v2/plot_code.py`: the executed plotting code.
- `INDEPENDENT_AUDIT.json`: independent checks of points, intervals,
  correlations and ratio summaries. `PROVENANCE.json`, `code/` and
  `FROZEN_PROTOCOL.json` preserve inputs and computation.

Fusion comparisons use 1,000 source-cluster draws, shared across rules,
retaining all three background pairs together. Intervals are individual,
conditional development-set intervals, without correction for selection.
Component correlations use all 16,080 ResNet-18 own-class pairs on six
datasets and are descriptive, without pair-independent p-values.
