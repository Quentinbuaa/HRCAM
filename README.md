# H-RCAM: code and derived results

This repository contains the analysis code and currently available derived results for the H-RCAM manuscript prepared for *Entropy*. The original images and segmentation annotations are not included; obtain them from the [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) and [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) providers.

## Contents

- Root-level Python scripts implement data preparation, background replacement, attribution-map scoring, analysis, and table/figure generation. The principal entry points are `prepare_pets.py`, `prepare_cub.py`, `pets_experiment.py`, `cub_experiment.py`, `make_rq1_final_artifacts.py`, and `make_rq3_artifacts.py`.
- `results/pets_v1/` contains the available Oxford-IIIT Pet calibration, development, and holdout pair-level records, split indices, summaries, and analysis outputs.
- `results/cub_v1/` contains the available CUB split indices, metadata, and aggregate summaries. **The complete CUB holdout pair-level records are not present in this local copy.** The file `results/rq1_v3/cub_v1_holdout_transport_own.jsonl` contains only transport-distance additions, not a replacement for those records.
- `results/rq1_v3/`, `results/rq2_visual_cases_v1/`, and `results/rq3_v1/` contain available derived artifacts supporting the paper's RQ1–RQ3 displays. Development-stage selection files are also retained.
- `vendor/` contains supporting model code and its original license notices.

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

The experiment scripts write new records under `runs/<run>/`; the archived analysis outputs here are under `results/<run>/`. The holdout stages require development-stage configuration to have been frozen first; consult the scripts before starting a new run. This repository has not yet passed a clean-environment, end-to-end reproduction check, and the missing CUB holdout records prevent a complete rebuild of every published table from archived pair-level data alone.

No software license has yet been assigned to the original code in this repository. The licenses and usage conditions of the source datasets and of the vendored code remain separate.
