"""Prepare Caltech-101 with author-drawn object contours, outcome-blind."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "data" / "caltech101"
IMAGES = BASE / "101_ObjectCategories"
ANNOTATIONS = BASE / "Annotations"
NAME_MAP = {"Faces": "Faces_2", "Faces_easy": "Faces_3",
            "Motorbikes": "Motorbikes_16", "airplanes": "Airplanes_Side_2"}
EXPECTED_MD5 = "3138e1922a9193bfa496528edbbc45d0"


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_contour_mask(annotation, size):
    data = loadmat(annotation)
    box = np.asarray(data["box_coord"], dtype=float).ravel()
    contour = np.asarray(data["obj_contour"], dtype=float)
    if box.size != 4 or contour.ndim != 2 or contour.shape[0] != 2 or contour.shape[1] < 3:
        raise ValueError(f"Invalid author contour: {annotation}")
    if not np.isfinite(box).all() or not np.isfinite(contour).all():
        raise ValueError(f"Nonfinite author contour: {annotation}")
    # show_annotation.m: plot(contour[0]+box_coord[2], contour[1]+box_coord[0]).
    # PIL has zero-based pixel coordinates; MATLAB's coordinates are one-based.
    points = [(float(contour[0, i] + box[2] - 1), float(contour[1, i] + box[0] - 1))
              for i in range(contour.shape[1])]
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).polygon(points, fill=255)
    return mask


def main():
    archive = BASE / "caltech-101.zip"
    if digest(archive, "md5") != EXPECTED_MD5:
        raise ValueError("CaltechDATA MD5 mismatch")
    classes = sorted(p.name for p in IMAGES.iterdir() if p.is_dir() and p.name != "BACKGROUND_Google")
    assert len(classes) == 101
    rng = np.random.default_rng(20261011)
    training_rows, test_rows = [], []
    missing_annotations = []
    class_counts = {}
    for label, category in enumerate(classes):
        source = sorted((IMAGES / category).glob("image_*.jpg"))
        rows = []
        for image in source:
            annotation = ANNOTATIONS / NAME_MAP.get(category, category) / image.name.replace("image_", "annotation_").replace(".jpg", ".mat")
            if not annotation.exists():
                missing_annotations.append(str(image.relative_to(IMAGES)))
                continue
            rows.append({"image": str(image.relative_to(IMAGES)), "annotation": str(annotation.relative_to(ANNOTATIONS)),
                         "label": label, "category": category})
        if len(rows) < 20:
            raise ValueError(f"Insufficient annotated images for {category}: {len(rows)}")
        selected = set(rng.choice(len(rows), 10, replace=False).tolist())
        for pos, row in enumerate(rows):
            (test_rows if pos in selected else training_rows).append(row)
        class_counts[category] = {"original_images": len(source), "matched_annotations": len(rows),
                                  "train": len(rows) - 10, "test": 10}
    manifest = {"seed": 20261011, "class_names": classes, "class_counts": class_counts,
                "missing_annotations": missing_annotations,
                "training": [r["image"] for r in training_rows], "test": [r["image"] for r in test_rows]}
    manifest_path = ROOT / "data" / "caltech101_selected_ids.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Caltech-101 source split changed")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    qa_selection = {next(i for i, row in enumerate(test_rows) if row["label"] == label): label for label in range(24)}
    sheet = Image.new("RGB", (4 * 672, 6 * 175), "white")
    draw = ImageDraw.Draw(sheet)
    prepared = {}
    for role, rows in (("train", training_rows), ("test_selected", test_rows)):
        arrays, masks, labels, names, areas = [], [], [], [], []
        for count, row in enumerate(rows):
            with Image.open(IMAGES / row["image"]) as source:
                image = source.convert("RGB")
            array = np.asarray(image.resize((224, 224), Image.Resampling.BILINEAR))
            if role == "test_selected":
                original_mask = read_contour_mask(ANNOTATIONS / row["annotation"], image.size)
                mask = np.asarray(original_mask.resize((224, 224), Image.Resampling.NEAREST)) > 0
                areas.append(float(mask.mean()))
                if count in qa_selection:
                    pos = qa_selection[count]
                    x, y = (pos % 4) * 672, (pos // 4) * 175
                    altered = np.where(mask[:, :, None], array, 128).astype(np.uint8)
                    mask_rgb = np.repeat(mask[:, :, None], 3, axis=2).astype(np.uint8) * 255
                    for col, visual in enumerate((array, mask_rgb, altered)):
                        sheet.paste(Image.fromarray(visual).resize((216, 146)), (x + col * 222, y))
                    draw.text((x, y + 147), f"{row['category']}/{Path(row['image']).name} area={mask.mean():.3f}", fill="black")
            else:
                mask = np.zeros((224, 224), dtype=bool)
            arrays.append(array)
            masks.append(mask)
            labels.append(row["label"])
            names.append(row["image"])
            if (count + 1) % 500 == 0:
                print(f"{role} {count + 1}/{len(rows)}", flush=True)
        path = ROOT / "data" / f"caltech101_{role}.npz"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        np.savez_compressed(path, images=np.asarray(arrays, dtype=np.uint8),
                            masks=np.asarray(masks, dtype=bool), labels=np.asarray(labels, dtype=np.int64),
                            names=np.asarray(names))
        prepared[role] = {"sources": len(rows), "sha256": digest(path),
                          "mask_area_min": min(areas) if areas else None,
                          "mask_area_median": float(np.median(areas)) if areas else None,
                          "mask_area_max": max(areas) if areas else None,
                          "mask_area_ineligible": sum(area < .05 or area > .95 for area in areas)}
    sheet.save(BASE / "caltech101_contour_mask_qa.png")
    info = {"source": "CaltechDATA archive via Hugging Face mirror; verified official MD5",
            "url": "https://data.caltech.edu/records/mzrjq-6wc02",
            "mirror": "https://huggingface.co/datasets/HuggingFaceM4/Caltech-101/resolve/main/caltech-101.zip",
            "md5": EXPECTED_MD5, "sha256": digest(archive),
            "missing_annotations": len(missing_annotations), "prepared": prepared}
    (ROOT / "assets" / "caltech101_manifest.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
