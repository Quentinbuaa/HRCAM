# JSD-D4 fusion development results

Exploratory source-grouped development CV; not fresh confirmation

The learned target is observed label change. All three backgrounds from one source stay in one fold.

## own-class CAM

JSD-D4 Spearman correlation: 0.7270.

| Candidate | Pooled OOF AUROC | Mean fold AUROC | AP |
|---|---:|---:|---:|
| JSD | 0.8706 | 0.8712 | 0.5734 |
| D4 | 0.8714 | 0.8711 | 0.6153 |
| RCAM | 0.8551 | 0.8541 | 0.5920 |
| BAT | 0.7886 | 0.7901 | 0.4341 |
| equal_mean | 0.8869 | 0.8866 | 0.6349 |
| nonlinear_union | 0.8869 | 0.8866 | 0.6346 |
| sqrt_mean | 0.8883 | 0.8878 | 0.6382 |
| boundary_mean | 0.8735 | 0.8728 | 0.6063 |
| convex_linear | 0.8863 | 0.8860 | 0.6328 |
| LR_two | 0.8856 | 0.8860 | 0.6275 |
| LR_extended | 0.8925 | 0.8941 | 0.6457 |
| MLP_two | 0.8883 | 0.8891 | 0.6289 |
| MLP_extended | 0.8884 | 0.8888 | 0.6634 |

Fold-fitted JSD weights: [0.6000000000000001, 0.5, 0.5, 0.5, 0.5].

- LR_extended_minus_LR_two: 0.0069, conditional paired interval [-0.0028, 0.0173].
- MLP_extended_minus_MLP_two: 0.0001, conditional paired interval [-0.0098, 0.0097].

Recorded training warnings: 0.

## fixed-class CAM

JSD-D4 Spearman correlation: 0.6819.

| Candidate | Pooled OOF AUROC | Mean fold AUROC | AP |
|---|---:|---:|---:|
| JSD | 0.8040 | 0.8037 | 0.5013 |
| D4 | 0.7317 | 0.7328 | 0.3767 |
| RCAM | 0.7156 | 0.7176 | 0.3639 |
| BAT | 0.7409 | 0.7415 | 0.3926 |
| equal_mean | 0.7899 | 0.7900 | 0.4608 |
| nonlinear_union | 0.7899 | 0.7901 | 0.4606 |
| sqrt_mean | 0.7874 | 0.7876 | 0.4606 |
| boundary_mean | 0.7849 | 0.7859 | 0.4586 |
| convex_linear | 0.8011 | 0.8018 | 0.4956 |
| LR_two | 0.7994 | 0.8013 | 0.4951 |
| LR_extended | 0.8179 | 0.8214 | 0.5338 |
| MLP_two | 0.7996 | 0.8018 | 0.4911 |
| MLP_extended | 0.8099 | 0.8126 | 0.5418 |

Fold-fitted JSD weights: [0.9, 0.8, 0.9, 1.0, 1.0].

- LR_extended_minus_LR_two: 0.0185, conditional paired interval [0.0009, 0.0348].
- MLP_extended_minus_MLP_two: 0.0103, conditional paired interval [-0.0116, 0.0255].

Recorded training warnings: 0.

## Interpretation

- Fixed formulas are label-free; convex weights, LR and MLP are fitted using prediction-change outcomes.
- LR/MLP probabilities are learned scores, not automatically zero-preserving or monotone distances.
- Conditional bootstrap intervals reuse OOF predictions without refitting; multiple comparisons are exploratory.
- CV and source grouping control within-round leakage, not adaptivity across the research project.
- Fresh data and a frozen definition are required to confirm a selected improvement.
- Novelty and semantic usefulness are separate from a development AUROC gain.
