"""Frozen H-RCAM external test on single-class PASCAL VOC 2012 images."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

import experiment as exp
from flowers_experiment import build_model, features

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="voc2012_external_v1")
    parser.add_argument("--stage", choices=["fit", "evaluate"], required=True)
    args = parser.parse_args()
    exp.seed_all(20261011)
    exp.MEAN = [.485, .456, .406]
    exp.STD = [.229, .224, .225]
    output = ROOT / "runs" / args.run
    output.mkdir(parents=True, exist_ok=True)
    train = np.load(ROOT / "data/voc2012_train.npz")
    test = np.load(ROOT / "data/voc2012_val_selected.npz")
    manifest = json.loads((ROOT / "data/voc2012_selected_ids.json").read_text())
    n_classes = len(manifest["class_mapping"])
    assert n_classes >= 2
    assert len(np.unique(train["names"])) == len(train["names"])
    assert len(np.unique(test["names"])) == len(test["names"])
    assert set(train["names"]).isdisjoint(set(test["names"]))
    assert set(np.unique(test["labels"])) <= set(np.unique(train["labels"]))
    split_path = output / "selected_ids.json"
    if not split_path.exists():
        split_path.write_text(json.dumps(manifest, indent=2))
    elif json.loads(split_path.read_text()) != manifest:
        raise ValueError("VOC 2012 filtered split changed")
    model = build_model()
    if args.stage == "fit":
        if (output / "classifier.pt").exists():
            raise FileExistsError("Refusing to refit existing classifier")
        f = features(model, train["images"])
        head = LogisticRegression(C=1., solver="lbfgs", max_iter=1000, n_jobs=1)
        head.fit(f, train["labels"])
        assert len(head.classes_) == n_classes
        assert np.array_equal(head.classes_, np.arange(n_classes))
        model.fc = torch.nn.Linear(512, n_classes).cuda()
        with torch.no_grad():
            model.fc.weight.copy_(torch.as_tensor(head.coef_, device="cuda", dtype=torch.float32))
            model.fc.bias.copy_(torch.as_tensor(head.intercept_, device="cuda", dtype=torch.float32))
        metadata = {
            "dataset": "PASCAL VOC 2012 single-object-class segmentation subset",
            "run": args.run,
            "backbone": "ImageNet-pretrained ResNet-18, frozen",
            "head": "Multinomial logistic regression C=1, max_iter=1000, official segmentation train only",
            "training_sources": len(train["labels"]),
            "evaluation_sources": len(test["labels"]),
            "classes": n_classes,
            "train_accuracy": float(head.score(f, train["labels"])),
            "iterations_by_class": head.n_iter_.tolist(),
            "masks": "Official VOC SegmentationClass masks; void and object pixels preserved",
            "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob("*.py")},
        }
        torch.save(model.state_dict(), output / "classifier.pt")
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
        print(json.dumps({key: value for key, value in metadata.items() if key != "source_hashes"}, indent=2), flush=True)
    else:
        if not (output / "classifier.pt").exists():
            raise FileNotFoundError("Fit classifier first")
        model.fc = torch.nn.Linear(512, n_classes).cuda()
        model.load_state_dict(torch.load(output / "classifier.pt", map_location="cuda", weights_only=True))
        model.eval()
        ids = np.arange(len(test["labels"]))
        exp.collect("external", ids, ids, test["masks"], test, model, output, batch=16)


if __name__ == "__main__":
    main()
