# RQ3.4: Natural-image background replacement

Authorized by the author on 2026-11-04. This is a separate RQ, preserving
the existing RQ3.1. No manuscript edits and no imperfect-mask experiments.
Freeze this protocol and the donor-assignment manifest before inference.

## Fixed scope

- Existing fitted ResNet-18 classifiers from `runs/pets_v1` and `runs/cub_v1`.
  No retraining, component selection, or recalibration.
- Exactly the original holdout identities whose original mask occupies
  5%-95% of the image: expected 993 Pet and 963 CUB sources. Inclusion uses
  mask geometry only, before seeing natural-background predictions/maps.
- Original supplied masks, 224x224 images, normalization, last-stage Grad-CAM,
  JSD/NCD definitions and harmonic fusion remain unchanged.
- One photographic replacement per eligible source, with no donor reuse
  across either dataset. Do not choose backgrounds based on model results.
- Own-class targets for the original comparison; retain fixed-source targets
  as a separate diagnostic. Report all results, including invalid-map coverage.

## Background source and prespecified selection

Preferred source is BG-20k (Li et al., *Bridging Composite and Real: Towards
End-to-end Deep Image Matting*). Its official release describes backgrounds
excluding salient objects and specifies MIT licensing and citation.
The official Google Drive endpoint was inaccessible during preparation.
Use the public resized mirror `unography/BG-20k-1200px`, with commit pinned
from its API and the downloaded file verified against its advertised LFS
SHA256. Record that this is a third-party resized copy, not a newly verified
byte-identical copy of the original full-resolution release.

Use **only the lexicographically first parquet shard**
`data/train-00000-of-00006.parquet` (about 497 MB), chosen for bounded data
acquisition before any outputs are evaluated. This is a sample from one shard,
not a claim of a representative random sample of all BG-20k scenes.
Before inference, inspect contact sheets of the complete candidate shard and
record exclusions for obvious competing animals or nonphotographic graphics
(artwork, cartoons, digital patterns, logos, and plain gradients). This QA
extension was decided after the initial preview showed a small bird and some
abstract artwork, while no classifier outputs had been evaluated.
Scan the remaining rows in `numpy.random.default_rng(20261104).permutation` order.
Also reject undecodable images, images with either dimension below 224, and exact
duplicate resized RGB crops. Assign the first required number of eligible
unique donors to Pet then CUB, each in increasing original source-ID order.

Convert donors to RGB; use Pillow `ImageOps.fit` to center-crop preserving
aspect ratio and resample to 224x224 with bilinear interpolation. Composite before
normalization using `where(original_mask, original_image, donor)`.
Assert pixel-exact preservation of the entire foreground for every image.

Before inference, also inspect a prespecified contact sheet of 64 evenly spaced
assigned donor/composite pairs for decoding, crop, placement, and obvious
competing foreground animals. If a sampled donor contains an obvious cat,
dog, or bird, reject it by semantic QA and take the next unused eligible
donor in the fixed permutation. Record all exclusions and regenerate the
assignment and QA sheet before inference. This sampled QA does not certify
that every donor is free of every semantic object. The complete-shard review
likewise excludes obvious issues; it is not an exhaustively annotated object
inventory or a guarantee against small, occluded objects.

Save source IDs, donor row indices, original image dimensions, image/crop
hashes, rejection reasons, the crop array, and the final assignment manifest.
Hash/freeze these before model evaluation. Never alter assignment after
viewing classifier outcomes or H-RCAM scores.

## Analysis and reporting

- Report source/pair counts, invalid maps, inconsistency prevalence, source
  and transformed accuracy, H-RCAM/JSD/NCD AUROC and AP, group median gaps.
- Compare all nine original baseline distances and H-RCAM on exactly the
  same valid natural-replacement pairs. Use original definitions; no new
  min/max fitting. Retain raw distances and the original frozen calibration
  transform for CD/KLD/SNO comparisons when available; disclose clipping.
- Retain 1000 paired source bootstrap resamples, seed 20261104, for AUROC/AP
  and H-RCAM-minus-baseline differences. Keep all conditions/methods for
  each sampled source together. No pair-level Mann-Whitney/KS p-values.
- Compare against existing black/gray/white results descriptively. AP values
  can change with inconsistency prevalence. Any cross-background difference
  interval must resample common source IDs jointly and disclose coverage.
- Save representative composites chosen by source position before inference,
  not by score or whether H-RCAM wins. Keep complete records and hashes.
- Independently audit foreground preservation, original checkpoint hashes,
  replayed predictions/maps, score formulas, and table reconstruction.

## Execution

Remote host: remote-5080. Every Python process must run with `taskset -c 24-31`;
PyTorch/BLAS numerical threads must be four. New outputs go to a new run only.
Any optional parquet reader is installed into a task-specific dependency
directory, without modifying the existing experiment environments.

Sources:
- https://github.com/JizhiziLi/GFM#bg-20k
- https://jizhizili.github.io/files/gfm_datasets_agreements/BG-20k_Dataset_Release_Agreement.pdf
- https://huggingface.co/datasets/unography/BG-20k-1200px
