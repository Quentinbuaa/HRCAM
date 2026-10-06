"""Feasibility baseline for TrashCan's original top-level object categories."""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

import experiment as exp
from pets_experiment import build_model, features
from trashcan_experiment import make_splits


ROOT = Path(__file__).resolve().parent


def main():
    exp.seed_all(20260923)
    exp.MEAN = [.485, .456, .406]
    exp.STD = [.229, .224, .225]
    with np.load(ROOT / "data/trashcan_coarse_3class.npz") as source:
        images = source["images"]
        labels = source["labels"]
        groups = source["groups"]
    splits = make_splits(labels, groups)
    training = splits["training"]
    assessment = splits["calibration"] + splits["development"]
    ids = training + assessment
    model = build_model()
    representation = features(model, images[ids])
    head = LogisticRegression(C=1, solver="lbfgs", max_iter=1000, n_jobs=1)
    head.fit(representation[:len(training)], labels[training])
    result = {"dataset": "TrashCan top-level object classification",
              "source_counts": {role: len(part) for role, part in splits.items()},
              "video_counts": {role: len(set(groups[part])) for role, part in splits.items()},
              "class_counts": {role: np.bincount(labels[part], minlength=3).tolist()
                               for role, part in splits.items()},
              "accuracy_training": float(head.score(representation[:len(training)], labels[training]))}
    positions = {source_id: pos for pos, source_id in enumerate(ids)}
    for role in ("calibration", "development"):
        part = splits[role]
        result[f"accuracy_{role}"] = float(head.score(representation[[positions[i] for i in part]],
                                                      labels[part]))
    output = ROOT / "results/trashcan_coarse_baseline.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
