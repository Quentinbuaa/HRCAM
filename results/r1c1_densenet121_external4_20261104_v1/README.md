# DenseNet-121: four independent datasets

`run_external.py`, `complete_external_baselines.py`, the frozen protocols
and independent audit preserve this extension to Flowers-102, Caltech-101,
VOC 2012 and UECFoodPixComplete. Each dataset directory contains source
selection, metadata, pair-level records and a summary. No component or
fusion-rule reselection was performed.

`full_baselines_v1/` contains the earlier baseline outputs. Current Table 5,
including L1/L2/SSIM and their paired comparisons, uses
`../table_layouts_baselines_20261105_v1_retry1/`. Use that expanded summary
for rankings. Images, masks, checkpoints and heatmaps are not redistributed.
