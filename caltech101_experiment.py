"""External H-RCAM stress test on Caltech-101 hand-drawn object masks."""
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
N_CLASSES = 101


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="caltech101_external_v1")
    parser.add_argument("--stage", choices=["fit", "evaluate"], required=True)
    args = parser.parse_args()
    exp.seed_all(20261011)
    exp.MEAN = [.485, .456, .406]
    exp.STD = [.229, .224, .225]
    output = ROOT / "runs" / args.run
    output.mkdir(parents=True, exist_ok=True)
    train = np.load(ROOT / "data/caltech101_train.npz")
    test = np.load(ROOT / "data/caltech101_test_selected.npz")
    manifest = json.loads((ROOT / "data/caltech101_selected_ids.json").read_text())
    assert len(test["images"]) == 1010
    assert len(np.unique(test["names"])) == 1010
    assert set(train["names"]).isdisjoint(set(test["names"]))
    split_path = output / "selected_ids.json"
    if not split_path.exists():
        split_path.write_text(json.dumps(manifest, indent=2))
    elif json.loads(split_path.read_text()) != manifest:
        raise ValueError("Caltech source split changed")
    model = build_model()
    if args.stage == "fit":
        if (output / "classifier.pt").exists():
            raise FileExistsError("Refusing to refit classifier")
        f = features(model, train["images"])
        head = LogisticRegression(C=1., solver="lbfgs", max_iter=1000, n_jobs=1)
        head.fit(f, train["labels"])
        model.fc = torch.nn.Linear(512, N_CLASSES).cuda()
        with torch.no_grad():
            model.fc.weight.copy_(torch.as_tensor(head.coef_, device="cuda", dtype=torch.float32))
            model.fc.bias.copy_(torch.as_tensor(head.intercept_, device="cuda", dtype=torch.float32))
        metadata = {"dataset": "Caltech-101", "run": args.run,
                    "backbone": "ImageNet-pretrained ResNet-18, frozen",
                    "head": "Multinomial logistic regression C=1, max_iter=1000, all non-evaluation images",
                    "training_sources": len(train["labels"]), "evaluation_sources": len(test["labels"]),
                    "classes": N_CLASSES, "seed": 20261011,
                    "train_accuracy": float(head.score(f, train["labels"])),
                    "iterations_by_class": head.n_iter_.tolist(),
                    "masks": "Author-drawn object contours rasterized by show_annotation.m convention",
                    "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob("*.py")}}
        torch.save(model.state_dict(), output / "classifier.pt")
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
        print(json.dumps({k: v for k, v in metadata.items() if k != "source_hashes"}, indent=2), flush=True)
    else:
        if not (output / "classifier.pt").exists():
            raise FileNotFoundError("Fit classifier first")
        model.fc = torch.nn.Linear(512, N_CLASSES).cuda()
        model.load_state_dict(torch.load(output / "classifier.pt", map_location="cuda", weights_only=True))
        model.eval()
        ids = np.arange(len(test["labels"]))
        exp.collect("external", ids, ids, test["masks"], test, model, output, batch=16)


if __name__ == "__main__":
    main()
