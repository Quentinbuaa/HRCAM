# Table 3: ResNet-18 on the two development datasets

`pets_v1_scores.jsonl` and `cub_v1_scores.jsonl` provide source identifiers,
background probes, prediction-inconsistency outcomes and all thirteen
scores (H-RCAM plus twelve individual comparison measures).
`summary.json` contains AUROC/AP, marginal 95% intervals and paired
H-RCAM-minus-baseline intervals. `Table3_with_ci.csv` is the display input.

The same source-image cluster draws are retained across measures. These
development datasets informed component/fusion selection, so the
intervals do not correct selection-induced optimism.
Executed code, protocol, provenance and the independent audit are included.
