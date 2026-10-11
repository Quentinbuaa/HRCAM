# PASCAL VOC 2012 external stress test of frozen H-RCAM

Fixed on 2026-10-11 before reading any VOC 2012 classifier or metric
outcomes. This is the next external test after Flowers-102 and Caltech-101.
Its result will be reported whether favorable or unfavorable. Neither
H-RCAM's formula nor the nine baselines will be reselected after this test.

## Data and single-label adaptation

- Use the original [PASCAL VOC 2012 train/validation archive](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/)
  with official segmentation masks and `ImageSets/Segmentation/train.txt`
  and `val.txt`. The hidden official test set has no public foreground masks;
  therefore the official *validation* partition is the evaluation set.
- Read palette-mask label indices, not rendered RGB colors. Background is 0,
  object classes are 1--20, and the void/uncertain label is 255. Keep an
  image for this single-label experiment only if its segmentation contains
  exactly one distinct object class. This is an outcome-blind adaptation of
  VOC's potentially multi-label images, not an assertion that all VOC images
  are single-label. Use all such train images to fit the classifier head and
  all such val images for evaluation. No val image may enter training.
- Preserve both pixels of that object class and void/uncertain pixels;
  replace only mask-labelled background (0). Exclude a val source when its
  preserved-pixel fraction is outside the fixed inclusive interval
  [0.05, 0.95]. Do not correct or exclude images after examining outcomes.
  Report filtering and class counts. If a val class has no train example,
  exclude it by this prespecified rule and report the count.
- Resize RGB to 224x224 bilinear and masks to 224x224 nearest-neighbor.
  Conduct an outcome-blind mask/replacement visual QA contact sheet.

## Model, probes and frozen metrics

- Freeze the ImageNet-pretrained ResNet-18 backbone; fit a multinomial
  logistic-regression head with C=1 and max_iter=1000 on the designated
  single-class segmentation-train images only. Use `layer4` Grad-CAM and
  explain each image's own predicted class, as in the main analysis.
- For every eligible val source, make all three follow-ups by replacing only
  annotated background with black, gray or white constant RGB. Assert
  preserved pixels are unchanged. Record prediction-label inconsistency,
  which is the endpoint; do not equate it with ground-truth error.
- Keep JSD, NCD and H-RCAM's harmonic formula unchanged. Evaluate the same
  nine individual candidates (JSD, NCD, CD, KLD, SNO, BAT, SpatialW1, FAS,
  BAI), H-RCAM and fixed alternative JSD/NCD fusions. Report AUROC, AP,
  inconsistency prevalence, map/mask coverage, clean accuracy and results by
  background. AP must be interpreted relative to prevalence.
- Use 1000 source-cluster bootstrap replicates, seed 20261012, for H-RCAM
  AUROC and paired differences versus JSD, NCD, arithmetic and geometric
  fusion. All three follow-ups from a source remain in one cluster.

VOC is a harder domain transfer than Pet/CUB/Flowers/Caltech: it contains
diverse everyday objects and scenes, and its official task allows multiple
labels per image. The single-class subset and void-label handling limit
interpretation. The aim is to test the fixed metric transparently, not to
choose a dataset on which it wins.
