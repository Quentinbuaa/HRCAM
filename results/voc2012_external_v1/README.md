# PASCAL VOC 2012 external stress-test outputs

The [fixed protocol](../../VOC2012_EXTERNAL_PROTOCOL.md) predates outcome
inspection. `external_records.jsonl` stores all paired scores and prediction
outcomes; `external_transport.jsonl` supplies BAT and SpatialW1;
`external_summary.json` contains all aggregate analyses and source-cluster
intervals. `voc2012_selected_ids.json` records the official train/validation
split after outcome-blind single-class filtering, and
`voc2012_manifest.json` records archive identity and sample counts. Full
prepared arrays, source images, and heatmaps are not redistributed.
