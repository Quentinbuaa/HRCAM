# RQ1 final artifacts

## Oxford-IIIT Pet
Pairs: 2979
Prediction-consistent: 2515
Prediction-inconsistent: 464
H-RCAM AUROC: 0.8953
H-RCAM AP: 0.6888

## CUB-200-2011
Pairs: 2889
Prediction-consistent: 1061
Prediction-inconsistent: 1828
H-RCAM AUROC: 0.8066
H-RCAM AP: 0.8684

Methods: D1, D2, D3 use frozen calibrated component scores; D4 is divided by 2.
BAT and SpatialW1 are computed on the holdout heatmaps with a 14x14 transport grid.
