"""Ground-truth-mask validation with an independently fitted classification head."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.linear_model import LogisticRegression
from torchvision.models import resnet18

import experiment as exp

ROOT = Path(__file__).resolve().parent


def build_model():
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(ROOT / "assets/resnet18_imagenet.pth", map_location="cpu", weights_only=True))
    return model.cuda().eval()


def features(model, images):
    features = []
    original_head = model.fc
    model.fc = torch.nn.Identity()
    with torch.inference_mode():
        for start in range(0, len(images), 64):
            features.append(model(exp.normalize(exp.tensors(images[start:start+64]))).cpu().numpy())
    model.fc = original_head
    return np.concatenate(features)


def ground_truth_qa(data, ids, output):
    chosen = ids[:24]
    sheet = Image.new("RGB", (4*336, 6*144), "white")
    draw = ImageDraw.Draw(sheet)
    for pos, index in enumerate(chosen):
        x, y = (pos % 4)*336, (pos // 4)*144
        image = data["images"][index]
        mask = data["masks"][index]
        altered = np.where(mask[:, :, None], image, 128).astype(np.uint8)
        for col, arr in enumerate([image, np.repeat(mask[:, :, None], 3, 2).astype(np.uint8)*255, altered]):
            sheet.paste(Image.fromarray(arr).resize((108, 108)), (x+col*112, y))
        draw.text((x, y+112), str(data["names"][index]), fill="black")
    sheet.save(output / "mask_qa.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="pets_v1")
    parser.add_argument("--stage", choices=["prepare", "development", "holdout"], required=True)
    args = parser.parse_args()
    exp.seed_all(20260911)
    exp.MEAN = [.485, .456, .406]
    exp.STD = [.229, .224, .225]
    output = ROOT / "runs" / args.run
    output.mkdir(parents=True, exist_ok=True)
    training = np.load(ROOT / "data/pets_trainval.npz")
    testing = np.load(ROOT / "data/pets_test.npz")
    data = {key: np.concatenate([training[key], testing[key]]) for key in ["images", "masks", "labels", "names"]}
    rng = np.random.default_rng(20260911)
    splits = {r: [] for r in ["training", "calibration", "development", "holdout"]}
    for label in range(37):
        ids = rng.permutation(np.flatnonzero(training["labels"] == label))
        n_train, n_cal = int(.7*len(ids)), int(.1*len(ids))
        splits["training"].extend(ids[:n_train].tolist())
        splits["calibration"].extend(ids[n_train:n_train+n_cal].tolist())
        splits["development"].extend(ids[n_train+n_cal:].tolist())
        ids = rng.permutation(np.flatnonzero(testing["labels"] == label))[:27] + len(training["labels"])
        splits["holdout"].extend(ids.tolist())
    for role in splits:
        splits[role] = rng.permutation(splits[role]).tolist()
    assert len(set(sum(splits.values(), []))) == sum(map(len, splits.values()))
    existing = output / "splits.json"
    if existing.exists() and json.loads(existing.read_text()) != splits:
        raise ValueError("Splits changed")
    existing.write_text(json.dumps(splits, indent=2))
    model = build_model()
    if args.stage == "prepare":
        print("Extracting train/calibration/development features", flush=True)
        f = features(model, training["images"])
        # C is fixed before seeing validation or holdout; backbone is frozen.
        head = LogisticRegression(C=1., solver="lbfgs", max_iter=1000)
        head.fit(f[splits["training"]], training["labels"][splits["training"]])
        model.fc = torch.nn.Linear(512, 37).cuda()
        with torch.no_grad():
            model.fc.weight.copy_(torch.as_tensor(head.coef_, device="cuda", dtype=torch.float32))
            model.fc.bias.copy_(torch.as_tensor(head.intercept_, device="cuda", dtype=torch.float32))
        metadata = {"dataset": "Oxford-IIIT Pet", "backbone": "ImageNet-pretrained ResNet-18, frozen",
                    "head": "Multinomial logistic regression C=1, independently trained on 70% of official trainval",
                    "masks": "Ground-truth trimaps; preserve foreground and unknown boundary",
                    "source_counts": {k: len(v) for k, v in splits.items()}, "seed": 20260911,
                    "accuracy_training": float(head.score(f[splits["training"]], training["labels"][splits["training"]])),
                    "accuracy_calibration": float(head.score(f[splits["calibration"]], training["labels"][splits["calibration"]])),
                    "accuracy_development": float(head.score(f[splits["development"]], training["labels"][splits["development"]])),
                    "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob("*.py")}}
        torch.save(model.state_dict(), output / "classifier.pt")
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
        ground_truth_qa(data, splits["development"], output)
        print(json.dumps(metadata, indent=2), flush=True)
    else:
        if args.stage == "holdout" and not (output / "frozen_analysis.json").exists():
            raise RuntimeError("Freeze configuration before accessing holdout")
        model.fc = torch.nn.Linear(512, 37).cuda()
        model.load_state_dict(torch.load(output / "classifier.pt", map_location="cuda", weights_only=True))
        model.eval()
        for role in (["calibration", "development"] if args.stage == "development" else ["holdout"]):
            exp.collect(role, splits[role], np.arange(len(data["labels"])), data["masks"], data, model, output, batch=16)


if __name__ == "__main__":
    main()
