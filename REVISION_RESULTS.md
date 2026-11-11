# First-round revision: result and code index

This release archives completed analyses; publishing it did not run new
experiments. Original datasets, masks, donor photographs, model checkpoints
and native attribution maps are not redistributed. Obtain the source data
under the providers' terms. In particular, UECFoodPixComplete is restricted
to non-commercial research use. Dataset filenames and source identifiers
are retained to make selection and score-level verification possible.

## Current manuscript mapping

All paths below are relative to `results/`.

| Current item | Authoritative directory | Main result input |
|---|---|---|
| Table 2, group differences | `table2_mean_ci_20261105_v1/` | `summary.json`, `Table2_with_ci.csv` |
| Table 3, ResNet-18 development | `table3_revision_20261105_v1/` | `summary.json`, `Table3_with_ci.csv`, two score JSONLs |
| Table 4, ResNet-18 independent | `table_layouts_baselines_20261105_v1_retry1/` | `summary.json`, `metrics_with_ci.csv`, `resnet18__*_scores.jsonl` |
| Table 5 / RQ1.4, DenseNet-121 all six datasets | Same expanded-comparison directory | `densenet121__*_scores.jsonl`, summary and CSV |
| Table 7 / RQ3.1, background probes | `background_probe_ci_20261109_v1/` | `Table7_with_auroc_ci.csv`, `summary.json` |
| Photographic-background experiment | `natural_background_20261104_v1_retry1/` | `analysis/summary.json`, cached scores, donor-assignment identifiers |
| Table 8 / RQ3.2, fusion rules | `table7_components_20261105_v1/` | `Table7_with_ci.csv`, `fusion_summary.json` |
| Figure 5 and q/R diagnostics | Same components directory | `component_summary.json`, correlation/ratio CSVs, `presentation_v2/` plot and code |
| Table 9 / RQ3.3, target conventions | `r2c3_fixed_gt_20261105_v1/` | `table8_three_conventions_with_ci.csv`, `analysis/summary.json` and common scores |
| Target-switch analysis, RQ3.3 | `r1c2_target_switch_20261105_v1/` | `summary.json`, diagnostic JSONLs, rank/component CSVs |
| Method/layer experiment, Discussion | `r1c6_method_layer_20261105_v1/` | `README.md`, `table_with_ci.csv`, `analysis/summary.json`, common scores |

DenseNet extraction/training code and original intermediate summaries are
in `r1c1_densenet121_20261104_v1/` (Pet/CUB) and
`r1c1_densenet121_external4_20261104_v1/` (four independent datasets).
The expanded comparisons above supply the current rankings; earlier compact
baseline displays are retained only as intermediate computation artifacts.
`table2_revision_20261105_v1/` contains inputs to the final mean-CI extension,
not a competing final table. Legacy `Table7` and `table8` filenames in the
fusion and target runs correspond to current Tables 8 and 9, respectively.

## Score-level verification

From the repository root, run:

```bash
python verify_revision_release.py --report revision_verification.json
```

Only Python's standard library is needed. The verifier checks SHA-256 and
byte counts, recomputes tie-aware AUROC/AP, mean/median/extreme group gaps,
and Pearson/Spearman correlations from the released JSONLs, and validates
saved CI endpoints. It does not regenerate bootstrap samples or rerun
models. The frozen independent audits include the earlier, more extensive
map/CI checks. `REVISION_RELEASE_MANIFEST.json` records artifact hashes;
numeric data and executed scientific code are unchanged from the frozen
local analyses. Public README files have been updated for current numbering.

## Map-level reproduction and scope

Executed scientific scripts are under each run's `code/` or at its root.
They require the original datasets, fitted models and map caches, plus the
dependencies documented in the main README and frozen protocols. They are
execution snapshots, not a portable end-to-end installer. Original paths
use `runs/` for inputs and a Linux compute root; this release uses `results/`
for archives. Set the command-line root/run arguments and reproduce the
required source-data preparation before invoking extraction or bootstrap
scripts. DenseNet Pet/CUB's original execution name ends in `_retry1`, while
its archive directory does not. Absolute execution-root paths in frozen
provenance are historical references, not additional downloadable inputs.
Linux extraction snapshots bind logical CPUs 24-31 and four numerical
threads; adjust affinity to available cores when reproducing elsewhere.
Remote launch/transfer helpers are omitted because they do not implement
the scientific analysis. No claim of clean-environment end-to-end
reproduction is made by this release.

Development-set estimates and intervals are selection-influenced. The four
independent datasets did not inform component/fusion selection. Confidence
intervals use source-image cluster resampling, with common source draws
across methods for paired differences; individual intervals do not provide
simultaneous coverage or correct selection optimism. Natural-background
intervals condition on one frozen donor assignment. These results support
the evaluated prediction-consistency endpoint, architectures, attribution
settings and foreground-preserving probes, not unrestricted robustness.
