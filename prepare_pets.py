"""Prepare an optional ground-truth-mask validation dataset from its authors."""
import io
import json
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

from prepare import fetch

ROOT = Path(__file__).resolve().parent


def main():
    base = "https://www.robots.ox.ac.uk/~vgg/data/pets/data"
    records = []
    for name in ["images.tar.gz", "annotations.tar.gz"]:
        records.append(fetch(f"{base}/{name}", ROOT / "data/pets" / name))
    records.append(fetch("https://download-r2.pytorch.org/models/resnet18-f37072fd.pth", ROOT / "assets/resnet18_imagenet.pth"))
    with tarfile.open(ROOT / "data/pets/images.tar.gz") as image_tar, tarfile.open(ROOT / "data/pets/annotations.tar.gz") as annotation_tar:
        for role in ["trainval", "test"]:
            rows = annotation_tar.extractfile(f"annotations/{role}.txt").read().decode().splitlines()
            images, masks, labels, names = [], [], [], []
            for line in rows:
                name, label, _, _ = line.split()
                image = Image.open(io.BytesIO(image_tar.extractfile(f"images/{name}.jpg").read())).convert("RGB")
                trimap = Image.open(io.BytesIO(annotation_tar.extractfile(f"annotations/trimaps/{name}.png").read()))
                # Preserve the pet and uncertain boundary; only definite background is replaced.
                mask = Image.fromarray((np.asarray(trimap) != 2).astype(np.uint8)*255)
                images.append(np.asarray(image.resize((224, 224), Image.Resampling.BILINEAR)))
                masks.append(np.asarray(mask.resize((224, 224), Image.Resampling.NEAREST)) > 0)
                labels.append(int(label)-1)
                names.append(name)
            np.savez_compressed(ROOT / f"data/pets_{role}.npz", images=np.asarray(images), masks=np.asarray(masks), labels=labels, names=names)
    (ROOT / "assets/pets_manifest.json").write_text(json.dumps(records, indent=2))
    print("Oxford-IIIT Pet prepared", flush=True)


if __name__ == "__main__":
    main()
