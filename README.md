# H-RCAM: code and derived results

This repository contains the analysis code and derived results for the H-RCAM manuscript prepared for *Entropy*. Original images and annotations are not redistributed. Obtain them from the providers of [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/), [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/), [Oxford Flowers-102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/), [Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02), [PASCAL VOC 2012](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/), and [UECFoodPixComplete](https://mm.cs.uec.ac.jp/uecfoodpix/) as needed. UECFoodPixComplete permits non-commercial research use only; consult its provider's terms before obtaining it.

## Contents

- Root-level Python scripts implement data preparation, background replacement, attribution-map scoring, analysis, and table/figure generation. The principal entry points are `prepare_pets.py`, `prepare_cub.py`, `pets_experiment.py`, `cub_experiment.py`, `make_rq1_final_artifacts.py`, and `make_rq3_artifacts.py`.
- `results/pets_v1/` contains the available Oxford-IIIT Pet calibration, development, and holdout pair-level records, split indices, summaries, and analysis outputs.
- `results/cub_v1/` contains the CUB calibration, development, and holdout pair-level records, split indices, metadata, compact score arrays, and aggregate summaries. The holdout file contains 6000 records; 2889 own-predicted-class pairs meet the validity criteria used in the paper.
- `results/rq1_v3/`, `results/rq2_visual_cases_v1/`, and `results/rq3_v1/` contain available derived artifacts supporting the paper's RQ1–RQ3 displays. Development-stage selection files are also retained.
- `results/flowers102_external_v1/`, `results/caltech101_external_v1/`, `results/voc2012_external_v1/`, and `results/uecfoodpix_external_v1/` contain the four fixed-metric external evaluations reported in RQ1.3. Each has source selection, a dataset manifest, model metadata, per-pair candidate scores and prediction outcomes, transport-candidate scores, a full aggregate summary with source-cluster bootstrap intervals, and a compact report. No original images, masks, image-containing QA sheets, checkpoints, or heatmap arrays are included.
- The corresponding `prepare_*` and `*_experiment.py` scripts, `flowers_external_transport.py`, and `flowers_external_summary.py` reproduce the additional pipeline from legally obtained source datasets. Their frozen `*_EXTERNAL_PROTOCOL.md` files and detailed `*_EXTERNAL_VALIDATION.md` reports document the selection rules, results, and limitations.
- `fixed_source_sensitivity.py`, `FIXED_SOURCE_SENSITIVITY_PROTOCOL.md`, and `results/fixed_source_sensitivity_20261011_v1/` contain the unchanged-metric attribution-target sensitivity analysis in RQ3.3, including matched pair-level scores and bootstrap summaries. `verify_fixed_source_release.py` checks the released point estimates without source images, heatmaps, or a GPU.
- `vendor/` contains supporting model code and its original license notices.

## Independent validation (RQ1.3)

| Dataset | Valid sources / pairs | H-RCAM AUROC | H-RCAM AP | Full report |
|---|---:|---:|---:|---|
| Oxford Flowers-102 | 987 / 2961 | 0.8361 | 0.8078 | [Report](FLOWERS102_EXTERNAL_VALIDATION.md) |
| Caltech-101 | 999 / 2997 | 0.8739 | 0.6367 | [Report](CALTECH101_EXTERNAL_VALIDATION.md) |
| PASCAL VOC 2012, derived single-class task | 834 / 2502 | 0.8438 | 0.6661 | [Report](VOC2012_EXTERNAL_VALIDATION.md) |
| UECFoodPixComplete, derived single-class task | 584 / 1752 | 0.7808 | 0.8351 | [Report](UECFOODPIX_EXTERNAL_VALIDATION.md) |

The original Oxford-IIIT Pet and CUB-200-2011 datasets informed the choice of H-RCAM components and fusion. The four datasets above did not. The endpoint here is **change in the predicted label** under a foreground-preserving background replacement, not independently confirmed classification error. H-RCAM has the highest observed AUROC and AP among the nine individual comparison measures on all four, but the paired bootstrap AUROC advantage over JSD is not established on VOC 2012 or UECFoodPixComplete. AP values are not directly comparable across datasets with different positive-class prevalence. The VOC and UEC analyses use prespecified single-class subsets of segmentation datasets, not their full original tasks. See the reports and protocols before reusing these results.

## Attribution-target sensitivity (RQ3.3)

The matched-pair comparison holds the classifier, inputs, background replacements, Grad-CAM procedure, and H-RCAM formula fixed while changing only the follow-up explanation target. Own-class H-RCAM AUROC/AP are 0.8953/0.6888 on Pet and 0.8067/0.8684 on matched CUB pairs; fixed-source values are 0.7973/0.5095 and 0.7658/0.8484. Prediction-consistent scores are identical under both conventions. The primary results therefore compare explanations of actual predictions, not isolated attribution instability for one fixed class. The diagnostic results also show that H-RCAM's primary ranking above JSD does not extend to fixed-source targeting. See the [RQ3.3 report and reproduction instructions](results/fixed_source_sensitivity_20261011_v1/README.md).

## Environment and use

The experiments were run with Python 3.10, PyTorch 2.11.0, torchvision 0.26.0, NumPy 2.2.5, SciPy 1.15.3, scikit-learn 1.7.2, Pillow 12.3.0, and Matplotlib 3.10.9. Some scripts additionally require `requests`, OpenCV, or `segment-anything`. The experiment scripts use CUDA. Model weights are downloaded by the preparation scripts; they are not redistributed here.

For example, after installing compatible dependencies and obtaining the source datasets:

```bash
python prepare_pets.py
python pets_experiment.py --stage prepare --run pets_v1
python pets_experiment.py --stage development --run pets_v1
python analyze.py --stage development --run pets_v1 --bootstrap 400
python pets_experiment.py --stage holdout --run pets_v1

python prepare_cub.py
python cub_experiment.py --stage prepare --run cub_v1
python cub_experiment.py --stage development --run cub_v1
python analyze.py --stage development --run cub_v1 --bootstrap 400
python cub_experiment.py --stage holdout --run cub_v1
```

The experiment scripts write new records under `runs/<run>/`; the archived analysis outputs here are under `results/<run>/`. The holdout stages require development-stage configuration to have been frozen first; consult the scripts before starting a new run. For the external tests, follow the corresponding frozen protocols and use the dataset-specific preparation and experiment scripts, then `flowers_external_transport.py` and `flowers_external_summary.py` with the appropriate run arguments. Pair-level records include the source ID, background condition, predictions, consistency outcome, validity flags, and candidate dissimilarity components. The large heatmap arrays are not hosted here; they can be regenerated from the original images, segmentation annotations, and scripts. This repository has not yet passed a clean-environment, end-to-end reproduction check.

No software license has yet been assigned to the original code in this repository. The licenses and usage conditions of the source datasets and of the vendored code remain separate.
