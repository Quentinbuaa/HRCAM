"""Make a visual audit sheet for derived TrashCan targets and masks."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("material", "coarse"), default="material")
    args = parser.parse_args()
    data_path = ROOT / f"data/trashcan_{args.variant}_3class.npz"
    output = ROOT / f"data/trashcan_{args.variant}_3class_mask_qa.png"
    classes = ("metal", "plastic", "wood") if args.variant == "material" else ("rov", "bio", "trash")
    with np.load(data_path) as source:
        data = {key: source[key] for key in ("images", "masks", "labels", "names")}
    rng = np.random.default_rng(20260923)
    chosen = []
    for label in range(3):
        candidates = np.flatnonzero(data["labels"] == label)
        chosen.extend(rng.choice(candidates, size=min(8, len(candidates)), replace=False).tolist())
    sheet = Image.new("RGB", (4 * 336, 6 * 140), "white")
    draw = ImageDraw.Draw(sheet)
    for pos, index in enumerate(chosen):
        x, y = (pos % 4) * 336, (pos // 4) * 140
        image = data["images"][index]
        mask = data["masks"][index]
        altered = np.where(mask[:, :, None], image, 128).astype(np.uint8)
        panels = [image, np.repeat(mask[:, :, None], 3, axis=2).astype(np.uint8) * 255, altered]
        for column, panel in enumerate(panels):
            sheet.paste(Image.fromarray(panel).resize((108, 108)), (x + column * 112, y))
        draw.text((x, y + 112), f"{classes[int(data['labels'][index])]} {str(data['names'][index])[:29]}",
                  fill="black")
    sheet.save(output)
    print(output)


if __name__ == "__main__":
    main()
