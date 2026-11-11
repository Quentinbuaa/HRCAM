# DenseNet-121: development-dataset extraction and analysis

`run_architecture.py` freezes ImageNet-pretrained DenseNet-121 features and
fits a multinomial logistic-regression head (C=1) on original training
images for each dataset. `PROTOCOL.md` records the prespecified setup.
Pet/CUB split identifiers, metadata, pair-level records and summaries are
included, together with `audit_results.py` and its audit output.

`full_baselines_v1/` preserves the earlier full-baseline computation.
For the complete twelve-baseline comparison and current manuscript Table 5,
use `../table_layouts_baselines_20261105_v1_retry1/`.
The original execution used the run name
`r1c1_densenet121_20261104_v1_retry1`; this archive omits that suffix.
