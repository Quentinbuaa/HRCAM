"""H-RCAM external validation on TrashCan top-level object categories."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

import experiment as exp
from pets_experiment import build_model, features
from trashcan_experiment import make_splits


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/trashcan_coarse_3class.npz"
CLASSES = ("rov", "bio", "trash")
SEED = 20260923


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="trashcan_coarse_3class_v1")
    parser.add_argument("--stage", required=True, choices=("prepare", "development", "holdout"))
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
        raise ValueError("Saved TrashCan split differs from regenerated split")
    split_file.write_text(json.dumps(splits, indent=2))
    model = build_model()
    if args.stage == "prepare":
        classifier_file = output / "classifier.pt"
        if classifier_file.exists():
            raise FileExistsError(f"Refusing to replace {classifier_file}")
        ids = splits["training"] + splits["calibration"] + splits["development"]
        print(f"Extracting features for {len(ids)} targets", flush=True)
        representation = features(model, data["images"][ids])
        train_count = len(splits["training"])
        head = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, n_jobs=1)
        head.fit(representation[:train_count], data["labels"][splits["training"]])
        model.fc = torch.nn.Linear(512, len(CLASSES)).cuda()
        with torch.no_grad():
            model.fc.weight.copy_(torch.as_tensor(head.coef_, device="cuda", dtype=torch.float32))
            model.fc.bias.copy_(torch.as_tensor(head.intercept_, device="cuda", dtype=torch.float32))
        by_index = {source_id: position for position, source_id in enumerate(ids)}
        accuracy = {}
        for role in ("training", "calibration", "development"):
            part = splits[role]
            positions = [by_index[i] for i in part]
            accuracy[role] = float(head.score(representation[positions], data["labels"][part]))
        metadata = {
            "dataset": "TrashCan 1.0; derived object-centered top-level classification",
            "classes": CLASSES, "source": "https://conservancy.umn.edu/handle/11299/214865",
            "backbone": "ImageNet-pretrained ResNet-18, frozen",
            "head": "Multinomial logistic regression C=1, trained only on training videos",
            "masks": "COCO instance polygons for each target object, resized nearest-neighbor",
            "split": "10-fold StratifiedGroupKFold by source video; published frame-level split ignored",
            "seed": SEED,
            "source_counts": {role: len(part) for role, part in splits.items()},
            "video_counts": {role: len(set(data["groups"][part])) for role, part in splits.items()},
            "class_counts": {role: np.bincount(data["labels"][part], minlength=3).tolist()
                             for role, part in splits.items()},
            "accuracy": accuracy,
            "source_hashes": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in (ROOT / "prepare_trashcan_coarse.py",
                                           ROOT / "trashcan_coarse_experiment.py")},
        }
        torch.save(model.state_dict(), classifier_file)
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
        print(json.dumps(metadata, indent=2), flush=True)
    else:
        if args.stage == "holdout" and not (output / "frozen_analysis.json").exists():
            raise RuntimeError("Freeze metric configuration before holdout extraction")
        model.fc = torch.nn.Linear(512, len(CLASSES)).cuda()
        model.load_state_dict(torch.load(output / "classifier.pt", map_location="cuda", weights_only=True))
        model.eval()
        roles = ("calibration", "development") if args.stage == "development" else ("holdout",)
        for role in roles:
            exp.collect(role, splits[role], np.arange(len(data["labels"])), data["masks"],
                        data, model, output, batch=16)


if __name__ == "__main__":
    main()
