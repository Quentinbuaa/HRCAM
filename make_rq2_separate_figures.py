"""Create separate RQ2 distribution and visual-example figures."""
import json
import math
import textwrap
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
    ("Extra-high", 5912, "black"),
]


def load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return None


def fonts():
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    return {
        "regular12": load_font(font_path, 12),
        "regular14": load_font(font_path, 14),
        "regular16": load_font(font_path, 16),
        "bold14": load_font(bold_path, 14),
        "bold16": load_font(bold_path, 16),
        "bold18": load_font(bold_path, 18),
    }


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


def resize_square(array, size):
    return Image.fromarray(array.astype(np.uint8)).resize(
        (size, size), Image.Resampling.BILINEAR
    )


def h_rcam(record):
    j = record["jsd"]
    c = record["d4"] / 2
    return 2 * j * c / (j + c + 1e-12)


def load_data():
    train = np.load(ROOT / "data" / "pets_trainval.npz")
    test = np.load(ROOT / "data" / "pets_test.npz")
    data = {key: np.concatenate([train[key], test[key]]) for key in ["images", "masks", "labels", "names"]}
    heatmaps = np.load(ROOT / "runs" / "pets_v1" / "holdout_heatmaps.npz")
    source_to_pos = {int(source): pos for pos, source in enumerate(heatmaps["source_ids"])}
    records = [
        json.loads(line)
        for line in (ROOT / "runs" / "pets_v1" / "holdout_records.jsonl").read_text().splitlines()
        if line
    ]
    record_map = {(r["source_id"], r["background"], r["cam_mode"]): r for r in records}
    consistent = [
        r for r in records
        if r["cam_mode"] == "own" and r["valid_cam"] and r["mask_eligible"] and not r["inconsistent"]
    ]
    values = np.array([h_rcam(r) for r in consistent], dtype=float)
    return data, heatmaps, source_to_pos, record_map, values


def draw_centered(draw, xy, text, font, fill):
    x0, y0, x1, y1 = xy
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((x0 + (x1 - x0 - tw) / 2, y0 + (y1 - y0 - th) / 2), text, font=font, fill=fill)


def make_distribution(values, q1, q2, q9, fontset):
    width, height = 1500, 520
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 115, 42, width - 45, height - 96
    cap = 0.06
    clipped = np.clip(values, 0, cap)
    bins = np.linspace(0, cap, 49)
    counts, edges = np.histogram(clipped, bins=bins)
    total = len(values)
    percentages = counts / total * 100.0
    ymax = max(1e-12, float(percentages.max()))
    colors = {
        "Low": (232, 246, 237),
        "Medium": (247, 242, 220),
        "High": (246, 231, 232),
        "Extra-high": (246, 224, 243),
        "bar": (59, 95, 135),
        "axis": (45, 45, 45),
    }
    regions = [
        (0, q1, "Low"),
        (q1, q2, "Medium"),
        (q2, q9, "High"),
        (q9, cap, "Extra-high"),
    ]
    for start, end, label in regions:
        x0 = left + (right - left) * start / cap
        x1 = left + (right - left) * min(end, cap) / cap
        draw.rectangle((x0, top, x1, bottom), fill=colors[label])

    for pct, start, end in zip(percentages, edges[:-1], edges[1:]):
        x0 = left + (right - left) * start / cap
        x1 = left + (right - left) * end / cap
        y0 = bottom - (bottom - top) * pct / ymax
        draw.rectangle((x0 + 1, y0, x1 - 1, bottom), fill=colors["bar"])

    draw.line((left, bottom, right, bottom), fill=colors["axis"], width=2)
    draw.line((left, top, left, bottom), fill=colors["axis"], width=2)
    for tick in [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]:
        x = left + (right - left) * tick / cap
        draw.line((x, bottom, x, bottom + 7), fill=colors["axis"], width=1)
        draw_centered(draw, (x - 25, bottom + 12, x + 25, bottom + 32), f"{tick:.2f}", fontset["regular12"], colors["axis"])
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        y = bottom - (bottom - top) * frac
        pct = frac * ymax
        draw.line((left - 6, y, left, y), fill=colors["axis"], width=1)
        draw.text((left - 54, y - 8), f"{pct:.1f}", font=fontset["regular12"], fill=colors["axis"])

    threshold_colors = {
        "q33": (18, 120, 80),
        "q67": (130, 90, 40),
        "q90": (130, 30, 100),
    }
    for q, name in [(q1, "q33"), (q2, "q67"), (q9, "q90")]:
        x = left + (right - left) * q / cap
        draw.line((x, top, x, bottom), fill=threshold_colors[name], width=5)

    draw_centered(draw, (left + 350, height - 48, right - 350, height - 22), "H-RCAM score", fontset["regular16"], colors["axis"])
    ylabel = Image.new("RGBA", (190, 36), (255, 255, 255, 0))
    ydraw = ImageDraw.Draw(ylabel)
    ydraw.text((0, 7), "Percentage (%)", font=fontset["regular16"], fill=colors["axis"])
    image.paste(ylabel.rotate(90, expand=True), (26, top + 92), ylabel.rotate(90, expand=True))

    legend_items = [
        ("Low", f"<= {q1:.4f}"),
        ("Medium", f"{q1:.4f}-{q2:.4f}"),
        ("High", f"{q2:.4f}-{q9:.4f}"),
        ("Extra-high", f"> {q9:.4f}"),
    ]
    lx, ly = left + 26, 16
    for label, value in legend_items:
        draw.rectangle((lx, ly + 3, lx + 18, ly + 21), fill=colors[label], outline=(160, 160, 160))
        draw.text((lx + 25, ly), f"{label}: {value}", font=fontset["regular14"], fill=(35, 35, 35))
        lx += 255 if label != "Extra-high" else 0

    draw.text((right - 280, height - 48), "x-axis capped at 0.06", font=fontset["regular12"], fill=(80, 80, 80))
    return image


def case_arrays(data, heatmaps, source_to_pos, source_id, background):
    pos = source_to_pos[source_id]
    original = data["images"][source_id]
    mask = heatmaps["masks"][pos]
    perturbed = np.where(mask[:, :, None], original, BACKGROUND_COLOR[background]).astype(np.uint8)
    original_map = heatmaps["original_maps"][pos]
    perturbed_map = heatmaps["maps"][pos, BACKGROUND_INDEX[background], 0]
    return [
        original,
        perturbed,
        overlay(original, original_map),
        overlay(perturbed, perturbed_map),
    ]


def make_visual_grid(data, heatmaps, source_to_pos, record_map, fontset):
    headers = ["Prediction-Unchanged Case", "Source image", "Transferred image", "Source CAM", "Transferred CAM"]
    cell = 180
    caption_w = 315
    margin = 18
    header_h = 52
    row_gap = 16
    width = caption_w + 4 * cell + 6 * margin
    height = header_h + len(CASES) * cell + (len(CASES) + 1) * row_gap
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    x_positions = [margin, margin + caption_w + margin]
    for _ in range(3):
        x_positions.append(x_positions[-1] + cell + margin)
    col_widths = [caption_w, cell, cell, cell, cell]

    for x, w, header in zip(x_positions, col_widths, headers):
        draw_centered(draw, (x, 12, x + w, 38), header, fontset["bold16"], (20, 20, 20))

    fills = {
        "Low": (241, 249, 244),
        "Medium": (250, 247, 232),
        "High": (250, 239, 240),
        "Extra-high": (248, 235, 246),
    }
    comments = {
        "Low": "Comment: the attribution of the transferred CAM remains stable.",
        "Medium": "Comment: the transferred CAM shows moderate attribution shift.",
        "High": "Comment: the transferred CAM shows clear attribution shift.",
        "Extra-high": "Comment: the transferred CAM shows noticeable attribution shift.",
    }
    case_titles = {
        "Low": "Low H-RCAM",
        "Medium": "Medium H-RCAM",
        "High": "High H-RCAM",
        "Extra-high": "Extra-high H-RCAM",
    }
    for row, (level, source_id, background) in enumerate(CASES):
        y = header_h + row * (cell + row_gap)
        r = record_map[(source_id, background, "own")]
        score = h_rcam(r)
        draw.rectangle((margin // 2, y - row_gap // 2, width - margin // 2, y + cell + row_gap // 2), fill=fills[level])
        draw.text((x_positions[0] + 8, y + 16), case_titles[level], font=fontset["bold18"], fill=(20, 20, 20))
        caption_lines = [
            f"source {source_id}",
            f"background={background}",
            f"prediction {r['original_prediction']} -> {r['transformed_prediction']}",
            f"H-RCAM={score:.4f}",
        ]
        caption_lines.extend(textwrap.wrap(comments[level], width=36))
        for i, line in enumerate(caption_lines):
            draw.text((x_positions[0] + 8, y + 45 + i * 18), line, font=fontset["regular14"], fill=(35, 35, 35))
        for x, array in zip(x_positions[1:], case_arrays(data, heatmaps, source_to_pos, source_id, background)):
            image.paste(resize_square(array, cell), (x, y))
            draw.rectangle((x, y, x + cell - 1, y + cell - 1), outline=(190, 190, 190), width=1)
    return image


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fontset = fonts()
    data, heatmaps, source_to_pos, record_map, values = load_data()
    q1, q2, q9 = np.quantile(values, [1 / 3, 2 / 3, 0.9])
    distribution = make_distribution(values, q1, q2, q9, fontset)
    grid = make_visual_grid(data, heatmaps, source_to_pos, record_map, fontset)
    distribution.save(OUT / "rq2_distribution_only.png")
    grid.save(OUT / "rq2_visual_grid_caption_col.png")
    print(OUT / "rq2_distribution_only.png")
    print(OUT / "rq2_visual_grid_caption_col.png")


if __name__ == "__main__":
    main()
