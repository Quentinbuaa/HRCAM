"""Outcome-blind inspection of Oxford's provided flower segmentations."""
import io
import json
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "data" / "flowers"


def member_image(archive, name):
    stream = archive.extractfile(name)
    if stream is None:
        raise FileNotFoundError(name)
    return Image.open(io.BytesIO(stream.read())).convert("RGB")


def main():
    labels = loadmat(BASE / "imagelabels.mat", squeeze_me=True)["labels"] - 1
    splits = loadmat(BASE / "setid.mat", squeeze_me=True)
    test = np.asarray(splits["tstid"], dtype=int).ravel()
    # One class-spread, deterministic sample; independent of all outcomes.
    rng = np.random.default_rng(20261011)
    chosen = []
    for label in range(24):
        ids = test[labels[test - 1] == label]
        chosen.append(int(rng.choice(ids)))
    sheet = Image.new("RGB", (4 * 600, 6 * 242), "white")
    draw = ImageDraw.Draw(sheet)
    records = []
    with tarfile.open(BASE / "102flowers.tgz") as images, tarfile.open(BASE / "102segmentations.tgz") as masks:
        for pos, image_id in enumerate(chosen):
            image = member_image(images, f"jpg/image_{image_id:05d}.jpg")
            seg = member_image(masks, f"segmim/segmim_{image_id:05d}.jpg")
            arr = np.asarray(seg)
            h, w = arr.shape[:2]
            corners = [arr[0, 0].tolist(), arr[0, -1].tolist(), arr[-1, 0].tolist(), arr[-1, -1].tolist()]
            central = arr[h // 2, w // 2].tolist()
            records.append({"id": image_id, "class": int(labels[image_id - 1]),
                            "image_size": image.size, "segmentation_size": seg.size,
                            "corners_rgb": corners, "center_rgb": central,
                            "channel_min": arr.min(axis=(0, 1)).tolist(),
                            "channel_max": arr.max(axis=(0, 1)).tolist()})
            x, y = (pos % 4) * 600, (pos // 4) * 242
            sheet.paste(image.resize((286, 210)), (x, y))
            sheet.paste(seg.resize((286, 210)), (x + 298, y))
            draw.text((x, y + 211), f"id={image_id} label={labels[image_id - 1]}", fill="black")
    sheet.save(BASE / "flowers_mask_raw_qa.png")
    (BASE / "flowers_mask_raw_qa.json").write_text(json.dumps(records, indent=2))
    print(json.dumps({"sample_size": len(records), "examples": records[:4]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
