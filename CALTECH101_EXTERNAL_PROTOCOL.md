# Caltech-101 external stress test of the frozen H-RCAM

Fixed on 2026-10-11 before seeing any Caltech-101 classifier or H-RCAM
outcomes. The aim is to test portability beyond the development datasets
(Oxford-IIIT Pet, CUB-200-2011) and the first external Flowers-102 test. We
will report unfavorable results; this is not a search for a winning dataset.

## Data and sample

- Use the original [CaltechDATA](https://data.caltech.edu/records/mzrjq-6wc02)
  Caltech-101 archive (MD5 `3138e1922a9193bfa496528edbbc45d0`) with the
  hand-clicked object contour annotations. The official host returned 403 on
  our computing environment, so the original archive was obtained through a Hugging Face
  mirror and verified against the CaltechDATA MD5 before use.
- Exclude `BACKGROUND_Google`, which is not an object class, as torchvision
  does. Keep all 101 remaining classes. Map four annotation directory names
  using the documented torchvision mapping.
- The dataset has no official fixed classification test split. Sort image
  filenames in each class and, with NumPy seed 20261011, sample exactly 10
  images per class (1010 sources) for one sealed evaluation set. Use the
  remaining images to fit only the classifier head. Record every selected
  filename. No evaluation image influences model fitting or metric design.
- Rasterize hand-clicked outlines from `box_coord` and `obj_contour` using
  the creators' `show_annotation.m` coordinate convention. Before inference,
  inspect a deterministic class-spread mask QA contact sheet. Resize RGB to
  224x224 bilinear and masks to 224x224 nearest-neighbor. Exclude masks with
  area <5% or >95%, identically to Pet/CUB/Flowers. No outcome-dependent
  segmentation correction or exclusions.

## Frozen classifier, probe and metric

- Frozen ImageNet-pretrained ResNet-18, `layer4` Grad-CAM, and a 101-class
  multinomial logistic-regression head with C=1 and max_iter=1000 fitted on
  the designated training images. No hyperparameter tuning on evaluation.
- Replace only outside-contour pixels with RGB black (0), gray (128), and
  white (255); assert retained foreground pixels are exactly unchanged.
- Use the existing own-predicted-class Grad-CAM convention and validity
  criteria. The binary endpoint is observed prediction-label inconsistency,
  not independently verified incorrectness. H-RCAM remains
  `2*JSD*NCD/(JSD+NCD+1e-12)` without recalibration or new weights.

## Fixed analyses

- Pooled primary AUROC/AP for H-RCAM versus NCD and JSD; include all three
  backgrounds and report prevalence, clean classifier accuracy, map/mask
  coverage, and by-background AUROC.
- Report every previously compared candidate: D1, D2, D3, BAT, SpatialW1,
  foreground shift, and background increase, plus NCD/JSD/H-RCAM.
- Report fixed minimum, maximum, arithmetic and geometric fusion, but do not
  replace the harmonic H-RCAM if another rule wins.
- Source-cluster bootstrap (1000 resamples, seed 20261012) for H-RCAM AUROC
  and paired AUROC differences against NCD, JSD, arithmetic and geometric
  fusion. All backgrounds from a source stay in the same resampled cluster.
  Do not interpret a confidence interval spanning zero as an advantage.

Caltech-101 differs from the original fine-grained animal/flower tasks: its
categories are broad, its images are older and often centered, and the
background can be sparse. This makes it a deliberately harder domain-transfer
stress test, not a perfectly matched replicate. Its behavior will be reported
alongside, not substituted for, the Flowers-102 result.
