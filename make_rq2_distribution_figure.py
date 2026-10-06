"""Create an integrated RQ2 distribution-and-example figure."""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "rq2_visual_cases_v1"
BACKGROUND_INDEX = {"black": 0, "gray": 1, "white": 2}
BACKGROUND_COLOR = {"black": 0, "gray": 128, "white": 255}

# Representative prediction-consistent cases selected from low, medium, high,
# and very-high H-RCAM regions.
CASES = [
    ("Low", 6639, "gray"),
    ("Medium", 3797, "gray"),
    ("High", 4783, "black"),
    ("Very high", 5912, "black"),
]


def load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return None


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


def draw_centered(draw, xy, text, font, fill):
    x0, y0, x1, y1 = xy
    box = draw.textbbox((0, 0), text, font=font)
    tw = box[2] - box[0]
    th = box[3] - box[1]
    draw.text((x0 + (x1 - x0 - tw) / 2, y0 + (y1 - y0 - th) / 2), text, font=font, fill=fill)


def case_images(data, heatmaps, source_to_pos, source_id, background):
    pos = source_to_pos[source_id]
    image = data["images"][source_id]
    mask = heatmaps["masks"][pos]
    perturbed = np.where(mask[:, :, None], image, BACKGROUND_COLOR[background]).astype(np.uint8)
    original_map = heatmaps["original_maps"][pos]
    perturbed_map = heatmaps["maps"][pos, BACKGROUND_INDEX[background], 0]
    diff = np.clip(np.abs(original_map - perturbed_map) * 3.0, 0, 1)
    return (
        image,
        perturbed,
        overlay(image, original_map),
        overlay(perturbed, perturbed_map),
        heat_color_fixed(diff),
    )


def draw_histogram(draw, values, thresholds, case_scores, box, fonts):
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    pad_l, pad_r, pad_t, pad_b = 76, 28, 38, 62
    ax_l, ax_t = left + pad_l, top + pad_t
    ax_r, ax_b = right - pad_r, bottom - pad_b
    ax_w, ax_h = ax_r - ax_l, ax_b - ax_t

    # The distribution is long-tailed. A capped x-axis keeps the low/medium/high
    # region readable and reports the cap explicitly in the axis title.
    cap = 0.06
    clipped = np.clip(values, 0, cap)
    bins = np.linspace(0, cap, 49)
    counts, edges = np.histogram(clipped, bins=bins)
    ymax = max(1, int(counts.max()))

    colors = {
        "low": (232, 246, 237),
        "medium": (244, 241, 220),
        "high": (246, 231, 232),
        "bar": (59, 95, 135),
        "line": (45, 45, 45),
    }
    q1, q2, q9 = thresholds
    regions = [
        (0, q1, colors["low"], "Low"),
        (q1, q2, colors["medium"], "Medium"),
        (q2, q9, colors["high"], "High"),
        (q9, cap, (241, 220, 225), "Very high"),
    ]
    for start, end, color, label in regions:
        x0 = ax_l + ax_w * start / cap
        x1 = ax_l + ax_w * min(end, cap) / cap
        draw.rectangle((x0, ax_t, x1, ax_b), fill=color)
        draw_centered(draw, (x0, ax_t + 2, x1, ax_t + 28), label, fonts["bold14"], (40, 40, 40))

    bar_gap = 2
    for count, start, end in zip(counts, edges[:-1], edges[1:]):
        x0 = ax_l + ax_w * start / cap
        x1 = ax_l + ax_w * end / cap
        y0 = ax_b - ax_h * math.sqrt(count / ymax)
        draw.rectangle((x0 + bar_gap / 2, y0, x1 - bar_gap / 2, ax_b), fill=colors["bar"])

    draw.line((ax_l, ax_b, ax_r, ax_b), fill=colors["line"], width=2)
    draw.line((ax_l, ax_t, ax_l, ax_b), fill=colors["line"], width=2)

    for tick in [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]:
        x = ax_l + ax_w * tick / cap
        draw.line((x, ax_b, x, ax_b + 6), fill=colors["line"], width=1)
        draw_centered(draw, (x - 22, ax_b + 10, x + 22, ax_b + 28), f"{tick:.2f}", fonts["regular12"], colors["line"])
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        count = int(round((frac ** 2) * ymax))
        y = ax_b - ax_h * frac
        draw.line((ax_l - 5, y, ax_l, y), fill=colors["line"], width=1)
        draw.text((left + 18, y - 8), str(count), font=fonts["regular12"], fill=colors["line"])

    for q, name in [(q1, "q33"), (q2, "q67"), (q9, "q90")]:
        x = ax_l + ax_w * q / cap
        draw.line((x, ax_t, x, ax_b), fill=(115, 80, 60), width=2)
        draw.text((x + 4, ax_t + 30), f"{name}={q:.4f}", font=fonts["regular12"], fill=(80, 55, 45))

    marker_colors = {
        "Low": (18, 120, 80),
        "Medium": (155, 110, 20),
        "High": (165, 42, 42),
        "Very high": (120, 30, 100),
    }
    for label, score in case_scores:
        x = ax_l + ax_w * min(score, cap) / cap
        draw.line((x, ax_t - 8, x, ax_b), fill=marker_colors[label], width=3)
        draw.ellipse((x - 5, ax_t - 16, x + 5, ax_t - 6), fill=marker_colors[label])
        draw.text((x + 6, ax_t - 21), label, font=fonts["regular12"], fill=marker_colors[label])

    draw.text((left + 10, top + 8), "H-RCAM distribution among prediction-consistent holdout pairs", font=fonts["bold18"], fill=(20, 20, 20))
    draw.text((ax_l + 170, bottom - 34), "H-RCAM score (x-axis capped at 0.06; higher tail retained in summary statistics)", font=fonts["regular13"], fill=(40, 40, 40))
    draw.text((left + 12, ax_t + 40), "count", font=fonts["regular12"], fill=(40, 40, 40))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
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
    own_consistent = [
        r for r in records
        if r["cam_mode"] == "own" and r["valid_cam"] and r["mask_eligible"] and not r["inconsistent"]
    ]
    values = np.array([h_rcam(r) for r in own_consistent], dtype=float)
    q1, q2, q9 = np.quantile(values, [1 / 3, 2 / 3, 0.9])

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    fonts = {
        "regular12": load_font(font_path, 12),
        "regular13": load_font(font_path, 13),
        "regular14": load_font(font_path, 14),
        "bold14": load_font(bold_path, 14),
        "bold16": load_font(bold_path, 16),
        "bold18": load_font(bold_path, 18),
    }

    width, height = 1800, 2280
    figure = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(figure)

    case_scores = []
    for label, source_id, background in CASES:
        case_scores.append((label, h_rcam(record_map[(source_id, background, "own")])))
    draw_histogram(draw, values, (q1, q2, q9), case_scores, (70, 45, width - 70, 455), fonts)

    panel_top = 505
    panel_gap_x = 44
    panel_gap_y = 42
    panel_w = (width - 2 * 70 - panel_gap_x) // 2
    panel_h = 790
    thumb = 175
    headers = ["Original", "Perturbed", "Original CAM", "Perturbed CAM", "CAM diff x3"]
    region_fills = {
        "Low": (241, 249, 244),
        "Medium": (250, 247, 232),
        "High": (250, 239, 240),
        "Very high": (248, 235, 246),
    }
    text_color = (25, 25, 25)

    for idx, (label, source_id, background) in enumerate(CASES):
        grid_row, grid_col = divmod(idx, 2)
        x0 = 70 + grid_col * (panel_w + panel_gap_x)
        y0 = panel_top + grid_row * (panel_h + panel_gap_y)
        x1, y1 = x0 + panel_w, y0 + panel_h
        draw.rectangle((x0, y0, x1, y1), fill=region_fills[label], outline=(210, 210, 210), width=2)
        r = record_map[(source_id, background, "own")]
        score = h_rcam(r)
        draw.text((x0 + 18, y0 + 14), f"{label} attribution shift", font=fonts["bold18"], fill=text_color)
        draw.text((x0 + 18, y0 + 45), f"source {source_id}, background={background}, pred {r['original_prediction']}->{r['transformed_prediction']}", font=fonts["regular14"], fill=text_color)
        draw.text((x0 + 18, y0 + 70), f"H={score:.4f} | JSD={r['jsd']:.4f} | D4/2={r['d4'] / 2:.4f} | predJSD={r['prediction_jsd']:.4f}", font=fonts["regular14"], fill=text_color)

        arrays = case_images(data, heatmaps, source_to_pos, source_id, background)
        for j, (title, array) in enumerate(zip(headers, arrays)):
            if j < 4:
                row, col = divmod(j, 2)
                tx = x0 + 82 + col * (thumb + 90)
                ty = y0 + 126 + row * (thumb + 38)
            else:
                tx = x0 + (panel_w - thumb) // 2
                ty = y0 + 126 + 2 * (thumb + 38)
            draw.text((tx, ty - 23), title, font=fonts["bold14"], fill=text_color)
            figure.paste(resize_square(array, thumb), (tx, ty))
            draw.rectangle((tx, ty, tx + thumb - 1, ty + thumb - 1), outline=(190, 190, 190), width=1)

        if label == "Low":
            comment = "Prediction unchanged; attention remains visually stable."
        elif label == "Medium":
            comment = "Prediction unchanged; moderate attribution displacement is visible."
        elif label == "High":
            comment = "Prediction unchanged; attention shifts despite stable output."
        else:
            comment = "Prediction unchanged; strong attribution shift remains visible."
        draw.text((x0 + 18, y1 - 32), comment, font=fonts["regular14"], fill=text_color)

    summary = (
        f"Prediction-consistent pairs: n={len(values)}; "
        f"median={np.median(values):.4f}; q90={np.quantile(values, 0.9):.4f}; "
        f"q95={np.quantile(values, 0.95):.4f}; max={values.max():.4f}."
    )
    draw.text((70, height - 50), summary, font=fonts["regular14"], fill=(35, 35, 35))

    path = OUT / "rq2_distribution_with_cases.png"
    figure.save(path)
    print(path)


if __name__ == "__main__":
    main()
