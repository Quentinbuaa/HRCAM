# UECFoodPixComplete external stress test of frozen H-RCAM

Fixed on 2026-10-11 before reading any UECFoodPixComplete classifier or
metric outcomes. This is a second new stress test, alongside the independently
fixed PASCAL VOC 2012 protocol. All outcomes will be retained, including
unfavorable ones. No H-RCAM component, formula, fusion rule, or baseline will
be changed after this test.

## Data and single-label adaptation

- Use the creators' [UECFoodPixComplete distribution](https://mm.cs.uec.ac.jp/uecfoodpix/),
  specifically the *manually* segmented Complete version and its official
  train/test partitions. The source page restricts use to non-commercial
  research; the raw archive will not be rehosted in our public repository.
- Its mask's red channel carries background 0 and food classes 1--102.
  Keep an image only if the mask contains exactly one distinct nonzero class.
  This is a prespecified single-label subset of a segmentation dataset, not a
  claim that the full food task is single-label. Use all such train images
  to fit the classifier head and all such test images for evaluation.
  If an evaluation class has no filtered training example, exclude it by
  this outcome-blind rule and report the count.
- Preserve every nonzero-mask pixel and replace only zero-mask background.
  Exclude an evaluation image when the foreground fraction is outside the
  fixed inclusive [0.05, 0.95] interval. Resize images to 224x224 bilinear
  and masks to 224x224 nearest-neighbor. Report all filtering/class counts
  and inspect a mask/replacement QA contact sheet before inference.

## Model, probes and frozen metrics

- Freeze the ImageNet-pretrained ResNet-18 backbone; fit a multinomial
  logistic-regression head with C=1 and max_iter=1000 on the designated
  filtered training split only. Use `layer4` Grad-CAM, explaining each
  source and follow-up image with its own predicted class.
- Make black, gray, and white constant-color background replacements for
  each selected test source, asserting that mask-designated foreground
  pixels remain unchanged. The endpoint is prediction-label inconsistency,
  not independently verified model error.
- Report H-RCAM, its JSD and NCD components, CD, KLD, SNO, BAT, SpatialW1,
  FAS, BAI, and the fixed minimum, maximum, arithmetic and geometric
  JSD/NCD fusions. Report AUROC, AP, inconsistency prevalence, coverage,
  clean accuracy and by-background discrimination. AP requires prevalence
  context.
- Use 1000 source-cluster bootstrap replicates with seed 20261012 for
  H-RCAM AUROC and paired AUROC differences against JSD, NCD, arithmetic
  and geometric fusion. Cluster the three follow-ups of each test source.

This is a transfer test to a different visual domain. It is conditional on
the derived single-class subset and one classifier, and is not evidence for
all food images or all perturbations. We will not select the reported
datasets based on which metric ranks first.
