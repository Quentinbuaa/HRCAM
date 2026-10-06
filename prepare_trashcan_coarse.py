"""Derive TrashCan's original trash/bio/ROV categories as target crops."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_trashcan import (CONTEXT_SCALE, MATERIAL, MAX_MASK_FRACTION,
                              MIN_AREA, MIN_BOX_SIDE, MIN_MASK_FRACTION, ROOT,
                              crop_box, polygon_mask)


OUTPUT = ROOT / "data/trashcan_coarse_3class.npz"
CLASSES = ("rov", "bio", "trash")


def coarse_label(category):
    if category == 1:
        return 0
    if 2 <= category <= 8:
        return 1
    if 9 <= category <= 16:
        return 2
    raise ValueError(category)


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace {OUTPUT}")
    images, masks, labels, names, groups = [], [], [], [], []
    excluded = Counter()
    for role in ("train", "val"):
        source = json.loads((MATERIAL / f"instances_{role}_trashcan.json").read_text())
        image_rows = {row["id"]: row for row in source["images"]}
        for row in source["annotations"]:
            x, y, width, height = row["bbox"]
            if min(width, height) < MIN_BOX_SIDE or row["area"] < MIN_AREA:
                excluded["too_small"] += 1
                continue
            polygons = row.get("segmentation")
            if row.get("iscrowd") or not isinstance(polygons, list) or not polygons:
                excluded["invalid_polygon"] += 1
                continue
            filename = image_rows[row["image_id"]]["file_name"]
            with Image.open(MATERIAL / role / filename) as raw:
                image = raw.convert("RGB")
            mask = polygon_mask(image.size, polygons)
            bounds = crop_box(row["bbox"], image.size)
            if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                excluded["invalid_crop"] += 1
                continue
            image = image.crop(bounds).resize((224, 224), Image.Resampling.BILINEAR)
            mask = mask.crop(bounds).resize((224, 224), Image.Resampling.NEAREST)
            target = np.asarray(mask) > 0
            fraction = float(target.mean())
            if not MIN_MASK_FRACTION <= fraction <= MAX_MASK_FRACTION:
                excluded["mask_fraction"] += 1
                continue
            images.append(np.asarray(image))
            masks.append(target)
            labels.append(coarse_label(row["category_id"]))
            names.append(f"{filename}#ann{row['id']}")
            groups.append(filename.split("_frame", 1)[0])
            if len(images) % 1000 == 0:
                print(f"Prepared {len(images)} targets", flush=True)
    np.savez_compressed(OUTPUT, images=np.asarray(images, dtype=np.uint8),
                        masks=np.asarray(masks, dtype=bool),
                        labels=np.asarray(labels, dtype=np.int64),
                        names=np.asarray(names), groups=np.asarray(groups))
    archive = ROOT / "assets/trashcan_kaggle.zip"
    metadata = {
        "dataset": "TrashCan 1.0, derived target-centered top-level classification",
        "classes": CLASSES, "n_instances": len(images), "n_videos": len(set(groups)),
        "class_counts": {name: labels.count(index) for index, name in enumerate(CLASSES)},
        "excluded": dict(excluded), "context_scale": CONTEXT_SCALE,
        "min_bbox_side": MIN_BOX_SIDE, "min_annotation_area": MIN_AREA,
        "mask_fraction": [MIN_MASK_FRACTION, MAX_MASK_FRACTION],
        "download_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "source": "https://conservancy.umn.edu/handle/11299/214865",
    }
    (ROOT / "data/trashcan_coarse_3class_manifest.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
