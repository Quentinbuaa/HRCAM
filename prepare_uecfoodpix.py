"""Outcome-blind single-class preparation of official UECFoodPixComplete."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "data" / "uecfoodpixcomplete"


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def split_root():
    candidates = [path for path in BASE.rglob("train")
                  if (path / "img").is_dir() and (path / "mask").is_dir()
                  and (path.parent / "test" / "img").is_dir()
                  and (path.parent / "test" / "mask").is_dir()]
    if len(candidates) != 1:
        raise ValueError(f"Expected one official train/test folder pair: {candidates}")
    return candidates[0].parent


def red_mask(path):
    with Image.open(path) as source:
        rgb = np.asarray(source.convert("RGB"))
    labels = rgb[:, :, 0]
    values = set(np.unique(labels).tolist())
    if not values <= set(range(103)):
        raise ValueError(f"Unexpected food class values in {path}: {values}")
    return labels


def scan_split(folder):
    rows = []
    reasons = Counter()
    images = sorted((folder / "img").glob("*.jpg"))
    for image in images:
        mask_path = folder / "mask" / f"{image.stem}.png"
        if not mask_path.exists():
            reasons["missing_mask"] += 1
            continue
        labels = red_mask(mask_path)
        object_classes = sorted(set(np.unique(labels).tolist()) - {0})
        if len(object_classes) != 1:
            reasons["zero_or_multiple_object_classes"] += 1
            continue
        rows.append({"id": image.stem, "food_class": object_classes[0]})
    return rows, dict(reasons), len(images)


def prepare_rows(role, folder, rows, class_to_label, contact_sheet):
    images, masks, labels, names = [], [], [], []
    ineligible = 0
    qa_rows = []
    if role == "test":
        for value in sorted(class_to_label):
            match = next((row for row in rows if row["food_class"] == value), None)
            if match:
                qa_rows.append(match["id"])
        qa_rows.extend(row["id"] for row in rows if row["id"] not in qa_rows)
        qa_rows = qa_rows[:24]
    sheet = Image.new("RGB", (4 * 672, 6 * 175), "white") if role == "test" else None
    draw = ImageDraw.Draw(sheet) if sheet else None
    for count, row in enumerate(rows):
        with Image.open(folder / "img" / f"{row['id']}.jpg") as source:
            image = np.asarray(source.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR))
        if role == "test":
            raw = red_mask(folder / "mask" / f"{row['id']}.png")
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
                draw.text((x, y + 147), f"{row['id']} class={row['food_class']} area={foreground.mean():.3f}", fill="black")
        else:
            foreground = np.zeros((224, 224), dtype=bool)
        images.append(image)
        masks.append(foreground)
        labels.append(class_to_label[row["food_class"]])
        names.append(row["id"])
        if (count + 1) % 500 == 0:
            print(f"{role} {count + 1}/{len(rows)}", flush=True)
    path = ROOT / "data" / f"uecfoodpix_{'train' if role == 'train' else 'test_selected'}.npz"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    np.savez_compressed(path, images=np.asarray(images, dtype=np.uint8),
                        masks=np.asarray(masks, dtype=bool), labels=np.asarray(labels, dtype=np.int64),
                        names=np.asarray(names))
    if sheet:
        sheet.save(contact_sheet)
    return {"sources": len(rows), "mask_area_ineligible": ineligible if role == "test" else None,
            "sha256": digest(path), "class_counts": dict(Counter(row["food_class"] for row in rows))}


def main():
    archive = BASE / "UECFOODPIXCOMPLETE.tar"
    root = split_root()
    train_rows, train_reasons, train_original = scan_split(root / "train")
    test_rows, test_reasons, test_original = scan_split(root / "test")
    class_values = sorted({row["food_class"] for row in train_rows})
    no_train = [row for row in test_rows if row["food_class"] not in class_values]
    test_rows = [row for row in test_rows if row["food_class"] in class_values]
    class_to_label = {value: label for label, value in enumerate(class_values)}
    assert len(class_to_label) >= 2
    manifest = {
        "official_train_images": train_original, "official_test_images": test_original,
        "train_filter_reasons": train_reasons, "test_filter_reasons": test_reasons,
        "test_without_train_class": [row["id"] for row in no_train],
        "class_mapping": {str(value): label for value, label in class_to_label.items()},
        "train": train_rows, "test": test_rows,
    }
    manifest_path = ROOT / "data" / "uecfoodpix_selected_ids.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("UECFoodPixComplete filtered split changed")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    qa_path = BASE / "uecfoodpix_mask_qa.png"
    prepared = {
        "train": prepare_rows("train", root / "train", train_rows, class_to_label, qa_path),
        "test": prepare_rows("test", root / "test", test_rows, class_to_label, qa_path),
    }
    info = {
        "source": "UECFoodPixComplete original author distribution, manual masks",
        "url": "https://mm.cs.uec.ac.jp/uecfoodpix/",
        "usage": "Non-commercial research only; do not redistribute raw images or masks",
        "archive_sha256": digest(archive),
        "manifest": str(manifest_path.relative_to(ROOT)), "prepared": prepared,
    }
    (ROOT / "assets" / "uecfoodpix_manifest.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
