"""Frozen-method external test on Oxford Flowers-102; no metric selection."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torchvision.models import resnet18

import experiment as exp

ROOT = Path(__file__).resolve().parent
N_CLASSES = 102


def build_model():
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(ROOT / "assets/resnet18_imagenet.pth", map_location="cpu", weights_only=True))
    return model.cuda().eval()


def features(model, images, batch=64):
    head = model.fc
    model.fc = torch.nn.Identity()
    values = []
    with torch.inference_mode():
        for start in range(0, len(images), batch):
            values.append(model(exp.normalize(exp.tensors(images[start:start + batch]))).cpu().numpy())
            if start % 512 == 0:
                print(f"features {start}/{len(images)}", flush=True)
    model.fc = head
    return np.concatenate(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="flowers102_external_v1")
    parser.add_argument("--stage", choices=["fit", "evaluate"], required=True)
    args = parser.parse_args()
    exp.seed_all(20261011)
    exp.MEAN = [.485, .456, .406]
    exp.STD = [.229, .224, .225]
    output = ROOT / "runs" / args.run
    output.mkdir(parents=True, exist_ok=True)
    train = np.load(ROOT / "data/flowers_trainval.npz")
    test = np.load(ROOT / "data/flowers_test_selected.npz")
    manifest = json.loads((ROOT / "data/flowers_selected_ids.json").read_text())
    assert len(train["images"]) == 2040
    assert len(test["images"]) == 1020
    assert len(np.unique(test["names"])) == 1020
    if not (output / "selected_ids.json").exists():
        (output / "selected_ids.json").write_text(json.dumps(manifest, indent=2))
    elif json.loads((output / "selected_ids.json").read_text()) != manifest:
        raise ValueError("Flowers source sample changed")
    model = build_model()
    if args.stage == "fit":
        if (output / "classifier.pt").exists():
            raise FileExistsError("Refusing to refit existing classifier")
        f = features(model, train["images"])
        head = LogisticRegression(C=1., solver="lbfgs", max_iter=1000, n_jobs=1)
        head.fit(f, train["labels"])
        model.fc = torch.nn.Linear(512, N_CLASSES).cuda()
        with torch.no_grad():
            model.fc.weight.copy_(torch.as_tensor(head.coef_, device="cuda", dtype=torch.float32))
            model.fc.bias.copy_(torch.as_tensor(head.intercept_, device="cuda", dtype=torch.float32))
        metadata = {
            "dataset": "Oxford Flowers-102",
            "run": args.run,
            "backbone": "ImageNet-pretrained ResNet-18, frozen",
            "head": "Multinomial logistic regression C=1, max_iter=1000, official train+val only",
            "training_sources": len(train["labels"]),
            "evaluation_sources": len(test["labels"]),
            "classes": N_CLASSES,
            "seed": 20261011,
            "train_accuracy": float(head.score(f, train["labels"])),
            "iterations_by_class": head.n_iter_.tolist(),
            "masks": "Oxford-provided segmentations, not independently verified ground truth",
            "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob("*.py")},
        }
        torch.save(model.state_dict(), output / "classifier.pt")
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
        print(json.dumps(metadata, indent=2), flush=True)
    else:
        if not (output / "classifier.pt").exists():
            raise FileNotFoundError("Fit the classifier first")
        model.fc = torch.nn.Linear(512, N_CLASSES).cuda()
        model.load_state_dict(torch.load(output / "classifier.pt", map_location="cuda", weights_only=True))
        model.eval()
        ids = np.arange(len(test["labels"]))
        exp.collect("external", ids, ids, test["masks"], test, model, output, batch=16)


if __name__ == "__main__":
    main()
