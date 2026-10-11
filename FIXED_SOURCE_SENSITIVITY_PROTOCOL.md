# Fixed-source-class sensitivity protocol (11 October 2026)

Purpose: assess whether H-RCAM's association with prediction inconsistency
persists when source and follow-up attribution maps explain the same class.
This check was requested after reviewing the existing manuscript. It is a
retrospective sensitivity analysis, not an untouched independent validation.

1. Use the manuscript's exact Oxford-IIIT Pet and CUB-200-2011 source subsets
   (the original runs' `holdout` role), classifier checkpoints, supplied masks,
   normalization, layer4 Grad-CAM, and black/gray/white replacement conditions.
   The term "development datasets" describes their role in H-RCAM selection;
   it does not mean substituting the runs' separate `development` image split.
2. Keep the primary own-predicted-class convention unchanged. For the sensitivity
   convention, explain both images using the source image's predicted class.
3. Reuse the two conventions already saved together by the original extraction
   only after verifying hashes, source IDs, predictions, fresh autograd replay,
   and independently recomputed JSD/NCD components for all cached pairs.
4. Apply exactly the current bounded harmonic formula, with JSD in base 2,
   NCD = (1 - Pearson correlation)/2, and fusion epsilon 1e-12. Do not fit weights,
   choose components, normalize scores across datasets, or tune any settings.
5. Use the intersection of valid eligible pairs for the paired comparison;
   report coverage and exclusions by mode. Separately reproduce the original
   own-class manuscript scores on their original eligible subset. Confirm that
   consistent pairs have identical scores when their targets are identical.
6. Main endpoints: H-RCAM AUROC and AP for actual prediction inconsistency.
   Report both conventions and paired fixed-minus-own changes. Include JSD/NCD
   and per-background results as diagnostics, without claiming a new comparison
   against all nine individual candidates.
7. Use 1,000 paired source-image-cluster bootstrap resamples, seed 20261011,
   retaining all available background pairs of each sampled source. Report
   percentile 95% intervals. These quantify source-sampling uncertainty, not
   model-training variation. No post hoc significance-driven tuning.
8. Save the script, provenance, matched per-pair score arrays, JSON summaries,
   and a readable Markdown report in a new results directory. Original files,
   manuscript, and public GitHub release remain unchanged. Report negative
   or weaker results as well as positive results.
