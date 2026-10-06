"""Fetch public assets with pinned revisions and record checksums."""
import argparse
import hashlib
import json
import pickle
import tarfile
import time
from pathlib import Path

import numpy as np
import requests


ROOT = Path(__file__).resolve().parent


def fetch(url, path, expected_md5=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        for attempt in range(3):
            try:
                print(f"Downloading {url}", flush=True)
                with requests.get(url, stream=True, timeout=(15, 45)) as response:
                    response.raise_for_status()
                    with path.with_suffix(path.suffix + ".partial").open("wb") as stream:
                        for chunk in response.iter_content(1024 * 1024):
                            stream.write(chunk)
                path.with_suffix(path.suffix + ".partial").replace(path)
                break
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(2)
    content = path.read_bytes()
    if expected_md5 and hashlib.md5(content).hexdigest() != expected_md5:
        raise ValueError(f"Checksum mismatch: {path}")
    return {"url": url, "path": str(path.relative_to(ROOT)),
            "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", choices=["classifier", "segmentation", "sam", "data"], required=True)
    args = parser.parse_args()
    records = []
    if args.asset == "classifier":
        revision = "ccacbae365101b9bfa45e1f17f4acac30c2601b2"
        base = f"https://hf-mirror.com/edadaltocg/resnet18_cifar10/resolve/{revision}"
        for name in ["pytorch_model.bin", "config.json", "README.md", "hyperparameters.json"]:
            records.append(fetch(f"{base}/{name}", ROOT / "assets/classifier" / name))
        repo = "huyvnphan/PyTorch_CIFAR10"
        response = requests.get(f"https://api.github.com/repos/{repo}/commits/master", timeout=30)
        response.raise_for_status()
        sha = response.json()["sha"]
        for source, dest in [("cifar10_models/resnet.py", "resnet.py"), ("LICENSE", "LICENSE-resnet")]:
            records.append(fetch(f"https://raw.githubusercontent.com/{repo}/{sha}/{source}", ROOT / "vendor" / dest))
    elif args.asset == "segmentation":
        revision = "7f1d3b18bcf55a010ec4526087978d89bccd3097"
        records.append(fetch(f"https://hf-mirror.com/netradrishti/u2net-saliency/resolve/{revision}/models/u2net.pth", ROOT / "assets/u2net.pth"))
        repo = "xuebinqin/U-2-Net"
        response = requests.get(f"https://api.github.com/repos/{repo}/commits/master", timeout=30)
        response.raise_for_status()
        sha = response.json()["sha"]
        for source, dest in [("model/u2net.py", "u2net.py"), ("LICENSE", "LICENSE-u2net")]:
            records.append(fetch(f"https://raw.githubusercontent.com/{repo}/{sha}/{source}", ROOT / "vendor" / dest))
    elif args.asset == "sam":
        records.append(fetch("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth", ROOT / "assets/sam_vit_b_01ec64.pth"))
    else:
        archive = ROOT / "data/cifar-10-python.tar.gz"
        records.append(fetch("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", archive,
                             "c58f30108f718f92721af3b95e74349a"))
        # Read only the checksum-verified canonical batches; no archive extraction.
        with tarfile.open(archive) as tar:
            for split, names in [("train", [f"data_batch_{i}" for i in range(1, 6)]),
                                 ("test", ["test_batch"])]:
                batches = [pickle.load(tar.extractfile(f"cifar-10-batches-py/{name}"), encoding="bytes") for name in names]
                images = np.concatenate([b[b"data"] for b in batches]).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                labels = np.concatenate([np.asarray(b[b"labels"]) for b in batches])
                np.savez_compressed(ROOT / f"data/cifar10_{split}.npz", images=images, labels=labels)
    (ROOT / "assets").mkdir(exist_ok=True)
    (ROOT / f"assets/{args.asset}_manifest.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2), flush=True)


if __name__ == "__main__":
    main()
