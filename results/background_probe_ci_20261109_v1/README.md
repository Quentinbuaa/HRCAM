# Table 7 / RQ3.1: background-probe AUROC intervals

`Table7_with_auroc_ci.csv` and `summary.json` contain the ten rows for black,
gray, white, Constant color and Natural probes on Pet/CUB. Per-color and
pooled estimates use cached Table 3 scores; Natural results use the frozen
photographic-background experiment in
`../natural_background_20261104_v1_retry1/`.

All intervals use source resampling with 1,000 draws. Constant-color draws
retain the three within-source pairs together, and the same draws are used
across color conditions. Natural intervals condition on the frozen donor
assignment (one pair per source). Individual intervals are not paired
contrasts across probe families. No additional model inference was needed.

`source_draw_counts.npz` and `bootstrap_auroc_samples.npz` preserve the
color draws/results. `code/` contains the executed calculation and audit;
`INDEPENDENT_AUDIT.json` records exact reconstruction of all 8,000
color/pooled bootstrap AUROCs and matching percentile endpoints.
