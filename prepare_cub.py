"""Prepare CUB-200-2011 with official segmentation masks."""
import io
import hashlib
import json
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent


def fetch(url, path, expected_md5=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_suffix(path.suffix + ".partial")
        for attempt in range(3):
            try:
                print(f"Downloading {url}", flush=True)
                request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as stream:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        stream.write(chunk)
                partial.replace(path)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
    content = path.read_bytes()
    if expected_md5 and hashlib.md5(content).hexdigest() != expected_md5:
        raise ValueError(f"Checksum mismatch: {path}")
    return {
        "url": url,
        "path": str(path.relative_to(ROOT)),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def read_text_member(tar, name):
    return tar.extractfile(f"CUB_200_2011/{name}").read().decode().splitlines()


def safe_extract(archive, output, marker):
    output.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        return
    with tarfile.open(archive) as tar:
        tar.extractall(output)
    marker.write_text("ok\n")


def main():
    records = []
    image_archive = ROOT / "data" / "cub" / "CUB_200_2011.tgz"
    segmentation_archive = ROOT / "data" / "cub" / "segmentations.tgz"
    records.append(
        fetch(
            "https://data.caltech.edu/api/records/65de6-vp158/files/CUB_200_2011.tgz/content",
            image_archive,
            "97eceeb196236b17998738112f37df78",
        )
    )
    records.append(
        fetch(
            "https://data.caltech.edu/api/records/w9d68-gec53/files/segmentations.tgz/content",
            segmentation_archive,
            "4d47ba1228eae64f2fa547c47bc65255",
        )
    )
    records.append(
        fetch(
            "https://download-r2.pytorch.org/models/resnet18-f37072fd.pth",
            ROOT / "assets" / "resnet18_imagenet.pth",
        )
    )
    extract_root = ROOT / "data" / "cub" / "extracted"
    safe_extract(image_archive, extract_root, extract_root / ".images_extracted")
    safe_extract(segmentation_archive, extract_root, extract_root / ".segmentations_extracted")
    cub_root = extract_root / "CUB_200_2011"
    seg_root = extract_root / "segmentations"
    image_rows = (cub_root / "images.txt").read_text().splitlines()
    label_rows = (cub_root / "image_class_labels.txt").read_text().splitlines()
    split_rows = (cub_root / "train_test_split.txt").read_text().splitlines()
    images_by_id = {int(row.split()[0]): row.split()[1] for row in image_rows}
    labels_by_id = {int(row.split()[0]): int(row.split()[1]) - 1 for row in label_rows}
    train_by_id = {int(row.split()[0]): int(row.split()[1]) == 1 for row in split_rows}
    collected = {"train": {"images": [], "masks": [], "labels": [], "names": []},
                 "test": {"images": [], "masks": [], "labels": [], "names": []}}
    for count, image_id in enumerate(sorted(images_by_id), 1):
        rel = images_by_id[image_id]
        role = "train" if train_by_id[image_id] else "test"
        image = Image.open(cub_root / "images" / rel).convert("RGB")
        mask_rel = rel.rsplit(".", 1)[0] + ".png"
        mask = Image.open(seg_root / mask_rel).convert("L")
        collected[role]["images"].append(np.asarray(image.resize((224, 224), Image.Resampling.BILINEAR)))
        collected[role]["masks"].append(np.asarray(mask.resize((224, 224), Image.Resampling.NEAREST)) > 0)
        collected[role]["labels"].append(labels_by_id[image_id])
        collected[role]["names"].append(rel)
        if count % 1000 == 0:
            print(f"Prepared {count}/{len(images_by_id)} CUB images", flush=True)
    for role, data in collected.items():
        np.savez_compressed(
            ROOT / f"data/cub_{role}.npz",
            images=np.asarray(data["images"], dtype=np.uint8),
            masks=np.asarray(data["masks"], dtype=bool),
            labels=np.asarray(data["labels"], dtype=np.int64),
            names=np.asarray(data["names"]),
        )
    (ROOT / "assets" / "cub_manifest.json").write_text(json.dumps(records, indent=2))
    print("CUB-200-2011 prepared", flush=True)


if __name__ == "__main__":
    main()
