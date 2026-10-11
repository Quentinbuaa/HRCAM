"""Prepare VOC 2012 single-class segmentation images without outcome access."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "data" / "voc2012"
VOC = BASE / "VOCdevkit" / "VOC2012"
EXPECTED_MD5 = "6cd6e144f989b92b3379bac3b3de84fd"
CLASS_NAMES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def image_ids(split):
    return (VOC / "ImageSets" / "Segmentation" / f"{split}.txt").read_text().split()


def mask_for(image_id):
    with Image.open(VOC / "SegmentationClass" / f"{image_id}.png") as source:
        if source.mode not in ("P", "L"):
            raise ValueError(f"Expected indexed VOC mask for {image_id}: {source.mode}")
        mask = np.asarray(source)
    values = set(np.unique(mask).tolist())
    if not values <= set(range(21)) | {255}:
        raise ValueError(f"Unexpected VOC segmentation values for {image_id}: {values}")
    return mask


def scan_split(split):
    rows = []
    reasons = Counter()
    for image_id in image_ids(split):
        if not (VOC / "JPEGImages" / f"{image_id}.jpg").exists():
            reasons["missing_rgb"] += 1
            continue
        mask_file = VOC / "SegmentationClass" / f"{image_id}.png"
        if not mask_file.exists():
            reasons["missing_mask"] += 1
            continue
        mask = mask_for(image_id)
        object_classes = sorted(set(np.unique(mask).tolist()) - {0, 255})
        if len(object_classes) != 1:
            reasons["zero_or_multiple_object_classes"] += 1
            continue
        rows.append({"id": image_id, "voc_class": object_classes[0],
                     "class_name": CLASS_NAMES[object_classes[0] - 1]})
    return rows, dict(reasons)


def prepare_rows(role, rows, class_to_label, contact_sheet):
    images, masks, labels, names = [], [], [], []
    qa_rows = []
    if role == "val":
        for value in sorted(class_to_label):
            match = next((row for row in rows if row["voc_class"] == value), None)
            if match:
                qa_rows.append(match["id"])
        qa_rows.extend(row["id"] for row in rows if row["id"] not in qa_rows)
        qa_rows = qa_rows[:24]
    sheet = Image.new("RGB", (4 * 672, 6 * 175), "white") if role == "val" else None
    draw = ImageDraw.Draw(sheet) if sheet else None
    ineligible = 0
    for count, row in enumerate(rows):
        with Image.open(VOC / "JPEGImages" / f"{row['id']}.jpg") as source:
            image = np.asarray(source.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR))
        if role == "val":
            raw = mask_for(row["id"])
            foreground = np.asarray(Image.fromarray(raw).resize((224, 224), Image.Resampling.NEAREST)) != 0
            if not .05 <= foreground.mean() <= .95:
                ineligible += 1
            if row["id"] in qa_rows:
                pos = qa_rows.index(row["id"])
                x, y = (pos % 4) * 672, (pos // 4) * 175
                altered = np.where(foreground[:, :, None], image, 128).astype(np.uint8)
                mask_rgb = np.repeat(foreground[:, :, None], 3, axis=2).astype(np.uint8) * 255
                for col, visual in enumerate((image, mask_rgb, altered)):
                    sheet.paste(Image.fromarray(visual).resize((216, 146)), (x + col * 222, y))
                draw.text((x, y + 147), f"{row['id']} {row['class_name']} area={foreground.mean():.3f}", fill="black")
        else:
            foreground = np.zeros((224, 224), dtype=bool)
        images.append(image)
        masks.append(foreground)
        labels.append(class_to_label[row["voc_class"]])
        names.append(row["id"])
        if (count + 1) % 250 == 0:
            print(f"{role} {count + 1}/{len(rows)}", flush=True)
    path = ROOT / "data" / f"voc2012_{'train' if role == 'train' else 'val_selected'}.npz"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    np.savez_compressed(path, images=np.asarray(images, dtype=np.uint8),
                        masks=np.asarray(masks, dtype=bool), labels=np.asarray(labels, dtype=np.int64),
                        names=np.asarray(names))
    if sheet:
        sheet.save(contact_sheet)
    return {"sources": len(rows), "mask_area_ineligible": ineligible if role == "val" else None,
            "sha256": digest(path), "class_counts": dict(Counter(row["class_name"] for row in rows))}


def main():
    archive = BASE / "VOCtrainval_11-May-2012.tar"
    if digest(archive, "md5") != EXPECTED_MD5:
        raise ValueError("Original VOC 2012 archive MD5 mismatch")
    train_ids, val_ids = image_ids("train"), image_ids("val")
    if set(train_ids) & set(val_ids):
        raise ValueError("Official segmentation train/val IDs overlap")
    train_rows, train_reasons = scan_split("train")
    val_rows, val_reasons = scan_split("val")
    class_values = sorted({row["voc_class"] for row in train_rows})
    no_train = [row for row in val_rows if row["voc_class"] not in class_values]
    val_rows = [row for row in val_rows if row["voc_class"] in class_values]
    class_to_label = {value: label for label, value in enumerate(class_values)}
    assert len(class_to_label) >= 2
    manifest = {
        "official_train_images": len(train_ids), "official_val_images": len(val_ids),
        "train_filter_reasons": train_reasons, "val_filter_reasons": val_reasons,
        "val_without_train_class": [row["id"] for row in no_train],
        "class_mapping": {CLASS_NAMES[value - 1]: label for value, label in class_to_label.items()},
        "train": train_rows, "val": val_rows,
    }
    manifest_path = ROOT / "data" / "voc2012_selected_ids.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("VOC 2012 filtered split changed")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    qa_path = BASE / "voc2012_mask_qa.png"
    prepared = {
        "train": prepare_rows("train", train_rows, class_to_label, qa_path),
        "val": prepare_rows("val", val_rows, class_to_label, qa_path),
    }
    info = {
        "source": "PASCAL VOC 2012 original trainval archive",
        "url": "https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/",
        "archive_md5": EXPECTED_MD5, "archive_sha256": digest(archive),
        "manifest": str(manifest_path.relative_to(ROOT)), "prepared": prepared,
    }
    (ROOT / "assets" / "voc2012_manifest.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
