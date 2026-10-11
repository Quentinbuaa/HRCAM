"""Prepare Oxford Flowers-102 official test subset and provided masks."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "data" / "flowers"
EXPECTED_MD5 = {
    "102flowers.tgz": "52808999861908f626f3c1f4e79d11fa",
    "imagelabels.mat": "e0620be6f572b9609742df49c70aed4d",
    "setid.mat": "a5357ecc9cb78c4bef273ce3793fc85c",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5(path):
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def mask_from_provided_segmentation(segmentation):
    # The Oxford-provided JPG uses (0, 0, 255) to mark background. JPEG
    # compression changes this slightly near borders; this fixed RGB radius
    # was chosen from outcome-blind inspection, before classifier inference.
    a = np.asarray(segmentation, dtype=np.int32)
    blue_distance_squared = a[:, :, 0] ** 2 + a[:, :, 1] ** 2 + (a[:, :, 2] - 255) ** 2
    return blue_distance_squared >= 110 ** 2


def main():
    for filename, expected in EXPECTED_MD5.items():
        actual = md5(BASE / filename)
        if actual != expected:
            raise ValueError(f"Official checksum mismatch {filename}: {actual}")
    labels = np.asarray(loadmat(BASE / "imagelabels.mat", squeeze_me=True)["labels"], dtype=int).ravel() - 1
    sets = loadmat(BASE / "setid.mat", squeeze_me=True)
    train_ids = np.asarray(sets["trnid"], dtype=int).ravel()
    val_ids = np.asarray(sets["valid"], dtype=int).ravel()
    test_ids = np.asarray(sets["tstid"], dtype=int).ravel()
    assert len(labels) == 8189
    assert len(train_ids) == 1020 and len(val_ids) == 1020 and len(test_ids) == 6149
    assert len(set(train_ids) | set(val_ids) | set(test_ids)) == len(labels)
    trainval = sorted(np.concatenate([train_ids, val_ids]).tolist())
    rng = np.random.default_rng(20261011)
    selected = []
    for label in range(102):
        ids = np.sort(test_ids[labels[test_ids - 1] == label])
        assert len(ids) >= 10
        selected.extend(rng.choice(ids, 10, replace=False).tolist())
    selected = sorted(selected)
    manifest = {"seed": 20261011, "training_ids": trainval, "selected_test_ids": selected,
                "sampling": "10 official-test image IDs per class without replacement"}
    manifest_path = ROOT / "data" / "flowers_selected_ids.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("The preselected Flowers-102 sample changed")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    sample_check = [next(i for i in selected if labels[i - 1] == label) for label in range(24)]
    qa = Image.new("RGB", (4 * 672, 6 * 175), "white")
    draw = ImageDraw.Draw(qa)
    results = {}
    extracted = BASE / "extracted"
    if not (extracted / "jpg" / "image_00001.jpg").exists() or not (extracted / "segmim" / "segmim_00001.jpg").exists():
        raise FileNotFoundError("Extract both official archives before preparation")
    if True:
        for role, ids in (("trainval", trainval), ("test_selected", selected)):
            arrays, masks, targets, names, areas = [], [], [], [], []
            for count, image_id in enumerate(ids, start=1):
                image = load_rgb(extracted / "jpg" / f"image_{image_id:05d}.jpg")
                array = np.asarray(image.resize((224, 224), Image.Resampling.BILINEAR))
                if role == "test_selected":
                    segmentation = load_rgb(extracted / "segmim" / f"segmim_{image_id:05d}.jpg")
                    if segmentation.size != image.size:
                        raise ValueError(f"Image/segmentation size mismatch: {image_id}")
                    full_mask = mask_from_provided_segmentation(segmentation)
                    mask = np.asarray(Image.fromarray(full_mask.astype(np.uint8)).resize(
                        (224, 224), Image.Resampling.NEAREST), dtype=bool)
                    areas.append(float(mask.mean()))
                    if image_id in sample_check:
                        pos = sample_check.index(image_id)
                        x, y = (pos % 4) * 672, (pos // 4) * 175
                        foreground = np.where(mask[:, :, None], array, 128).astype(np.uint8)
                        binary = np.repeat(mask[:, :, None], 3, axis=2).astype(np.uint8) * 255
                        for col, a in enumerate((array, binary, foreground)):
                            qa.paste(Image.fromarray(a).resize((216, 146)), (x + col * 222, y))
                        draw.text((x, y + 147), f"id={image_id} label={labels[image_id - 1]} area={mask.mean():.3f}", fill="black")
                else:
                    mask = np.zeros((224, 224), dtype=bool)
                arrays.append(array)
                masks.append(mask)
                targets.append(int(labels[image_id - 1]))
                names.append(f"image_{image_id:05d}.jpg")
                if count % 500 == 0:
                    print(f"{role} prepared {count}/{len(ids)}", flush=True)
            path = ROOT / "data" / f"flowers_{role}.npz"
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite {path}")
            np.savez_compressed(path, images=np.asarray(arrays, dtype=np.uint8),
                                masks=np.asarray(masks, dtype=bool),
                                labels=np.asarray(targets, dtype=np.int64), names=np.asarray(names))
            results[role] = {"count": len(ids), "sha256": sha256(path),
                             "mask_area_min": min(areas) if areas else None,
                             "mask_area_median": float(np.median(areas)) if areas else None,
                             "mask_area_max": max(areas) if areas else None,
                             "mask_area_ineligible": sum(a < .05 or a > .95 for a in areas)}
    qa.save(BASE / "flowers_decoded_mask_qa.png")
    source = {name: {"url": f"https://www.robots.ox.ac.uk/~vgg/data/flowers/102/{name}",
                     "bytes": (BASE / name).stat().st_size, "sha256": sha256(BASE / name)}
              for name in ("102flowers.tgz", "102segmentations.tgz", "imagelabels.mat", "setid.mat")}
    (ROOT / "assets" / "flowers_manifest.json").write_text(json.dumps({"source": source, "prepared": results,
                                                                        "segmentation_decoding": "RGB Euclidean distance from (0,0,255) >= 110"}, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
