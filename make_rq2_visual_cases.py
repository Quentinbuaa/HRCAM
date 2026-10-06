"""Create RQ2 visual case sheets from saved Pet holdout heatmaps."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "rq2_visual_cases_v1"
BACKGROUND_INDEX = {"black": 0, "gray": 1, "white": 2}
BACKGROUND_COLOR = {"black": 0, "gray": 128, "white": 255}
CASES = [
    ("Low", 6639, "gray"),
    ("Medium", 3797, "gray"),
    ("High", 4783, "black"),
    ("Very high", 5912, "black"),
]


def normalize01(values):
    arr = np.asarray(values, dtype=np.float32)
    mn = float(np.nanmin(arr))
    mx = float(np.nanmax(arr))
    if mx - mn < 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def heat_color(values):
    h = normalize01(values)
    return heat_color_fixed(h)


def heat_color_fixed(values):
    h = np.clip(np.asarray(values, dtype=np.float32), 0, 1)
    red = np.clip(1.5 * h, 0, 1)
    green = np.clip(1.5 * (1 - np.abs(h - 0.5) * 2), 0, 1)
    blue = np.clip(1.5 * (1 - h), 0, 1)
    return (np.stack([red, green, blue], axis=-1) * 255).astype(np.uint8)


def overlay(image, heatmap, alpha=0.45):
    base = np.asarray(image, dtype=np.float32)
    color = heat_color_fixed(heatmap).astype(np.float32)
    return np.clip((1 - alpha) * base + alpha * color, 0, 255).astype(np.uint8)


def resize(array, size):
    return Image.fromarray(array.astype(np.uint8)).resize(
        (size, size), Image.Resampling.BILINEAR
    )


def load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return None


def case_arrays(data, heatmaps, source_to_pos, source_id, background):
    pos = source_to_pos[source_id]
    image = data["images"][source_id]
    mask = heatmaps["masks"][pos]
    perturbed = np.where(
        mask[:, :, None], image, BACKGROUND_COLOR[background]
    ).astype(np.uint8)
    original_map = heatmaps["original_maps"][pos]
    perturbed_map = heatmaps["maps"][pos, BACKGROUND_INDEX[background], 0]
    # Use a fixed scale for differences so rows are visually comparable.
    diff = np.clip(np.abs(original_map - perturbed_map) * 3.0, 0, 1)
    return [
        image,
        perturbed,
        overlay(image, original_map),
        overlay(perturbed, perturbed_map),
        heat_color_fixed(diff),
    ]


def h_rcam(record):
    j = record["jsd"]
    c = record["d4"] / 2
    return 2 * j * c / (j + c + 1e-12)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = np.load(ROOT / "data" / "pets_trainval.npz")
    test = np.load(ROOT / "data" / "pets_test.npz")
    data = {
        key: np.concatenate([train[key], test[key]])
        for key in ["images", "masks", "labels", "names"]
    }
    heatmaps = np.load(ROOT / "runs" / "pets_v1" / "holdout_heatmaps.npz")
    source_to_pos = {int(s): i for i, s in enumerate(heatmaps["source_ids"])}
    records = [
        json.loads(line)
        for line in (ROOT / "runs" / "pets_v1" / "holdout_records.jsonl")
        .read_text()
        .splitlines()
        if line
    ]
    record_map = {
        (r["source_id"], r["background"], r["cam_mode"]): r for r in records
    }

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font = load_font(font_path, 14)
    small = load_font(font_path, 12)
    title_font = load_font(bold_path, 14)
    headers = ["Original", "Perturbed", "Original CAM", "Perturbed CAM", "CAM diff x3"]

    cell = 224
    label_h = 58
    cols = 5
    margin = 12
    header_h = 40
    sheet = Image.new(
        "RGB",
        (
            cols * cell + (cols + 1) * margin,
            header_h + len(CASES) * (cell + label_h + margin) + margin,
        ),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for col, title in enumerate(headers):
        x = margin + col * (cell + margin)
        draw.text((x, 12), title, fill=(0, 0, 0), font=title_font)

    for row, (level, source_id, background) in enumerate(CASES):
        y = header_h + row * (cell + label_h + margin)
        arrays = case_arrays(data, heatmaps, source_to_pos, source_id, background)
        for col, array in enumerate(arrays):
            x = margin + col * (cell + margin)
            sheet.paste(resize(array, cell), (x, y))
            draw.rectangle((x, y, x + cell - 1, y + cell - 1), outline=(210, 210, 210))
        r = record_map[(source_id, background, "own")]
        text = (
            f"{level}: source {source_id}, bg={background}, "
            f"pred {r['original_prediction']}->{r['transformed_prediction']}\n"
            f"H-RCAM={h_rcam(r):.4f}, JSD={r['jsd']:.4f}, "
            f"D4/2={r['d4'] / 2:.4f}, predJSD={r['prediction_jsd']:.4f}"
        )
        draw.text((margin, y + cell + 6), text, fill=(0, 0, 0), font=small)

    sheet.save(OUT / "rq2_visual_cases_grid.png")

    for level, source_id, background in CASES:
        arrays = case_arrays(data, heatmaps, source_to_pos, source_id, background)
        big = Image.new("RGB", (5 * 280 + 6 * 14, 360), "white")
        d = ImageDraw.Draw(big)
        for col, (title, array) in enumerate(zip(headers, arrays)):
            x = 14 + col * (280 + 14)
            d.text((x, 10), title, fill=(0, 0, 0), font=title_font)
            big.paste(resize(array, 280), (x, 36))
            d.rectangle((x, 36, x + 279, 36 + 279), outline=(210, 210, 210))
        r = record_map[(source_id, background, "own")]
        d.text(
            (14, 326),
            (
                f"{level}: source {source_id}, bg={background}, "
                f"pred {r['original_prediction']}->{r['transformed_prediction']}, "
                f"H-RCAM={h_rcam(r):.4f}, JSD={r['jsd']:.4f}, "
                f"D4/2={r['d4'] / 2:.4f}, predJSD={r['prediction_jsd']:.4f}"
            ),
            fill=(0, 0, 0),
            font=font,
        )
        slug = level.lower().replace(" ", "_")
        big.save(OUT / f"rq2_{slug}_{source_id}_{background}.png")

    for path in sorted(OUT.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
