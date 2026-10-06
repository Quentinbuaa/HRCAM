"""External H-RCAM validation on TrashCan derived material classification."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

import experiment as exp
from pets_experiment import build_model, features


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/trashcan_material_3class.npz"
N_CLASSES = 3
SEED = 20260923


def make_splits(labels, groups):
    folds = np.full(len(labels), -1, dtype=np.int64)
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
    for fold, (_, indices) in enumerate(splitter.split(np.zeros(len(labels)), labels, groups)):
        folds[indices] = fold
    assert np.all(folds >= 0)
    split_folds = {"training": set(range(6)), "calibration": {6},
                   "development": {7}, "holdout": {8, 9}}
    splits = {role: np.flatnonzero(np.isin(folds, list(chosen))).tolist()
              for role, chosen in split_folds.items()}
    assert len(set(sum(splits.values(), []))) == len(labels)
    group_sets = {role: set(groups[indices]) for role, indices in splits.items()}
    for role, current in group_sets.items():
        assert all(not current.intersection(others)
                   for key, others in group_sets.items() if key != role)
    for role, indices in splits.items():
        counts = np.bincount(labels[indices], minlength=N_CLASSES)
        if np.any(counts < 10):
            raise ValueError(f"Too few samples in {role}: {counts.tolist()}")
    return splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="trashcan_material_3class_v1")
    parser.add_argument("--stage", choices=["prepare", "development", "holdout"], required=True)
    args = parser.parse_args()
    exp.seed_all(SEED)
    exp.MEAN = [.485, .456, .406]
    exp.STD = [.229, .224, .225]
    with np.load(DATA) as source:
        data = {key: source[key] for key in ("images", "masks", "labels", "names", "groups")}
    splits = make_splits(data["labels"], data["groups"])
    output = ROOT / "runs" / args.run
    output.mkdir(parents=True, exist_ok=True)
    split_file = output / "splits.json"
    if split_file.exists() and json.loads(split_file.read_text()) != splits:
        raise ValueError("Existing split differs from new split")
    split_file.write_text(json.dumps(splits, indent=2))
    model = build_model()
    if args.stage == "prepare":
        classifier_file = output / "classifier.pt"
        if classifier_file.exists():
            raise FileExistsError(f"Refusing to replace {classifier_file}")
        print("Extracting TrashCan image features", flush=True)
        indices = splits["training"] + splits["calibration"] + splits["development"]
        representation = features(model, data["images"][indices])
        by_index = {source_id: pos for pos, source_id in enumerate(indices)}
        training_positions = [by_index[source_id] for source_id in splits["training"]]
        head = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, n_jobs=1)
        head.fit(representation[training_positions], data["labels"][splits["training"]])
        model.fc = torch.nn.Linear(512, N_CLASSES).cuda()
        with torch.no_grad():
            model.fc.weight.copy_(torch.as_tensor(head.coef_, device="cuda", dtype=torch.float32))
            model.fc.bias.copy_(torch.as_tensor(head.intercept_, device="cuda", dtype=torch.float32))
        accuracy = {}
        class_counts = {}
        for role, role_indices in splits.items():
            class_counts[role] = np.bincount(data["labels"][role_indices], minlength=N_CLASSES).tolist()
            if role != "holdout":
                positions = [by_index[source_id] for source_id in role_indices]
                accuracy[role] = float(head.score(representation[positions], data["labels"][role_indices]))
        metadata = {
            "dataset": "TrashCan 1.0, derived 3-class object-centered material classification",
            "classes": ["metal", "plastic", "wood"],
            "source": "https://conservancy.umn.edu/handle/11299/214865",
            "backbone": "ImageNet-pretrained ResNet-18, frozen",
            "head": "Multinomial logistic regression C=1, trained only on training videos",
            "masks": "COCO polygons for annotated target object, resized nearest-neighbor",
            "split": "StratifiedGroupKFold by source video; published frame-level split ignored",
            "seed": SEED, "source_counts": {role: len(values) for role, values in splits.items()},
            "video_counts": {role: len(set(data["groups"][values])) for role, values in splits.items()},
            "class_counts": class_counts, "accuracy": accuracy,
            "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [ROOT / "prepare_trashcan.py", ROOT / "trashcan_experiment.py"]},
        }
        torch.save(model.state_dict(), classifier_file)
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
        print(json.dumps(metadata, indent=2), flush=True)
    else:
        if args.stage == "holdout" and not (output / "frozen_analysis.json").exists():
            raise RuntimeError("Freeze metric definition before holdout extraction")
        model.fc = torch.nn.Linear(512, N_CLASSES).cuda()
        model.load_state_dict(torch.load(output / "classifier.pt", map_location="cuda", weights_only=True))
        model.eval()
        roles = ("calibration", "development") if args.stage == "development" else ("holdout",)
        for role in roles:
            exp.collect(role, splits[role], np.arange(len(data["labels"])), data["masks"],
                        data, model, output, batch=16)


if __name__ == "__main__":
    main()
