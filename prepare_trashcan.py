"""Build target-centered TrashCan material-classification images and masks.

TrashCan is an instance-segmentation benchmark, not an image-classification
benchmark. This derived task classifies annotated metal, plastic, and wood
debris instances. All instances from the same source video must share a split.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
MATERIAL = ROOT / "data/trashcan/dataset/material_version"
OUTPUT = ROOT / "data/trashcan_material_3class.npz"
LABELS = {12: "metal", 14: "plastic", 16: "wood"}
MIN_BOX_SIDE = 20
MIN_AREA = 300
MIN_MASK_FRACTION = 0.05
MAX_MASK_FRACTION = 0.75
CONTEXT_SCALE = 1.6


def crop_box(bbox, size):
    x, y, width, height = bbox
    side = CONTEXT_SCALE * max(width, height)
    cx, cy = x + width / 2, y + height / 2
    left = max(0, int(np.floor(cx - side / 2)))
    top = max(0, int(np.floor(cy - side / 2)))
    right = min(size[0], int(np.ceil(cx + side / 2)))
    bottom = min(size[1], int(np.ceil(cy + side / 2)))
    return (left, top, right, bottom)


def polygon_mask(size, polygons):
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        if len(polygon) >= 6 and len(polygon) % 2 == 0:
            draw.polygon(list(zip(polygon[::2], polygon[1::2])), fill=255)
    return mask


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace {OUTPUT}")
    images, masks, labels, names, groups = [], [], [], [], []
    exclusions = Counter()
    for role in ("train", "val"):
        annotations = json.loads((MATERIAL / f"instances_{role}_trashcan.json").read_text())
        images_by_id = {row["id"]: row for row in annotations["images"]}
        for count, row in enumerate(annotations["annotations"], 1):
            category = row["category_id"]
            if category not in LABELS:
                continue
            x, y, width, height = row["bbox"]
            if min(width, height) < MIN_BOX_SIDE or row["area"] < MIN_AREA:
                exclusions["too_small"] += 1
                continue
            polygons = row.get("segmentation")
            if row.get("iscrowd") or not isinstance(polygons, list) or not polygons:
                exclusions["invalid_polygon"] += 1
                continue
            image_row = images_by_id[row["image_id"]]
            file_name = image_row["file_name"]
            source = MATERIAL / role / file_name
            with Image.open(source) as raw:
                image = raw.convert("RGB")
            mask = polygon_mask(image.size, polygons)
            bounds = crop_box(row["bbox"], image.size)
            if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                exclusions["invalid_crop"] += 1
                continue
            image = image.crop(bounds).resize((224, 224), Image.Resampling.BILINEAR)
            mask = mask.crop(bounds).resize((224, 224), Image.Resampling.NEAREST)
            mask_array = np.asarray(mask) > 0
            fraction = float(mask_array.mean())
            if not MIN_MASK_FRACTION <= fraction <= MAX_MASK_FRACTION:
                exclusions["mask_fraction"] += 1
                continue
            images.append(np.asarray(image))
            masks.append(mask_array)
            labels.append(list(LABELS).index(category))
            names.append(f"{file_name}#ann{row['id']}")
            groups.append(file_name.split("_frame", 1)[0])
            if len(images) % 500 == 0:
                print(f"Prepared {len(images)} target instances", flush=True)
    if not images:
        raise RuntimeError("No qualifying TrashCan targets")
    np.savez_compressed(OUTPUT, images=np.asarray(images, dtype=np.uint8),
                        masks=np.asarray(masks, dtype=bool),
                        labels=np.asarray(labels, dtype=np.int64),
                        names=np.asarray(names), groups=np.asarray(groups))
    archive = ROOT / "assets/trashcan_kaggle.zip"
    metadata = {
        "dataset": "TrashCan 1.0, material version; derived object-centered classification",
        "source": "https://conservancy.umn.edu/handle/11299/214865",
        "download_mirror": "https://www.kaggle.com/datasets/mexwell/trashcan-1-0",
        "download_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "classes": list(LABELS.values()), "original_category_ids": list(LABELS),
        "image_size": [224, 224], "context_scale": CONTEXT_SCALE,
        "min_bbox_side": MIN_BOX_SIDE, "min_annotation_area": MIN_AREA,
        "allowed_mask_fraction": [MIN_MASK_FRACTION, MAX_MASK_FRACTION],
        "source_instances": len(images), "source_videos": len(set(groups)),
        "class_counts": {LABELS[cid]: labels.count(index) for index, cid in enumerate(LABELS)},
        "exclusions": dict(exclusions),
    }
    (ROOT / "data/trashcan_material_3class_manifest.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
