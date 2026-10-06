"""Create an auditable contact sheet and panel assets for Figure 1 selection."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BACKGROUND_INDEX = {"black": 0, "gray": 1, "white": 2}
BACKGROUND_COLOR = {"black": 0, "gray": 128, "white": 255}


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(path, size)


def hrcam(record):
    jsd = record["jsd"]
    ncd = record["d4"] / 2.0
    return 2.0 * jsd * ncd / (jsd + ncd + 1e-12)


def heat_color(values):
    heat = np.clip(np.asarray(values, dtype=np.float32), 0, 1)
    red = np.clip(1.5 * heat, 0, 1)
    green = np.clip(1.5 * (1 - np.abs(heat - 0.5) * 2), 0, 1)
    blue = np.clip(1.5 * (1 - heat), 0, 1)
    return (np.stack([red, green, blue], axis=-1) * 255).astype(np.uint8)


def overlay(image, heatmap, alpha=0.45):
    base = np.asarray(image, dtype=np.float32)
    color = heat_color(heatmap).astype(np.float32)
    return np.clip((1 - alpha) * base + alpha * color, 0, 255).astype(np.uint8)


def load(root):
    train = np.load(root / "data" / "pets_trainval.npz")
    test = np.load(root / "data" / "pets_test.npz")
    data = {
        key: np.concatenate([train[key], test[key]])
        for key in ["images", "masks", "labels", "names"]
    }
    heatmaps = np.load(root / "runs" / "pets_v1" / "holdout_heatmaps.npz")
    source_to_pos = {
        int(source): pos for pos, source in enumerate(heatmaps["source_ids"])
    }
    records = [
        json.loads(line)
        for line in (root / "runs" / "pets_v1" / "holdout_records.jsonl")
        .read_text()
        .splitlines()
        if line
    ]
    return data, heatmaps, source_to_pos, records


def arrays(data, heatmaps, source_to_pos, source_id, background):
    pos = source_to_pos[source_id]
    original = data["images"][source_id]
    mask = heatmaps["masks"][pos]
    followup = np.where(
        mask[:, :, None], original, BACKGROUND_COLOR[background]
    ).astype(np.uint8)
    source_map = heatmaps["original_maps"][pos]
    followup_map = heatmaps["maps"][pos, BACKGROUND_INDEX[background], 0]
    return [
        original,
        followup,
        overlay(original, source_map),
        overlay(followup, followup_map),
    ]


def candidates(records, limit):
    eligible = [
        record
        for record in records
        if record["cam_mode"] == "own"
        and record["valid_cam"]
        and record["mask_eligible"]
        and not record["inconsistent"]
        and record["source_id"] not in {6639, 3797, 4783, 5912}
    ]
    eligible.sort(key=hrcam, reverse=True)
    chosen, seen = [], set()
    for record in eligible:
        if record["source_id"] in seen:
            continue
        chosen.append(record)
        seen.add(record["source_id"])
        if len(chosen) == limit:
            break
    return chosen


def make_sheet(data, heatmaps, source_to_pos, records, output, limit=12):
    selected = candidates(records, limit)
    thumb, caption_w, gap = 112, 188, 8
    block_w = caption_w + 4 * thumb + 5 * gap
    block_h = thumb + 48
    columns = 2
    rows = (len(selected) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * block_w, rows * block_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, record in enumerate(selected):
        column, row = index % columns, index // columns
        left, top = column * block_w, row * block_h
        draw.rectangle(
            (left + 3, top + 3, left + block_w - 4, top + block_h - 4),
            outline=(170, 170, 170),
            width=1,
        )
        score = hrcam(record)
        lines = [
            f"source {record['source_id']}",
            f"background={record['background']}",
            f"prediction {record['original_prediction']} -> {record['transformed_prediction']}",
            f"H-RCAM={score:.4f}",
            f"JSD={record['jsd']:.4f}",
            f"NCD={record['d4']/2:.4f}",
        ]
        draw.text((left + 12, top + 12), lines[0], font=font(17, True), fill=(20, 20, 20))
        for line_index, line in enumerate(lines[1:]):
            draw.text(
                (left + 12, top + 38 + line_index * 18),
                line,
                font=font(13),
                fill=(40, 40, 40),
            )
        panel_left = left + caption_w
        labels = ["Source", "Follow-up", "Source CAM", "Follow-up CAM"]
        for panel_index, (label, array) in enumerate(
            zip(labels, arrays(data, heatmaps, source_to_pos, record["source_id"], record["background"]))
        ):
            x = panel_left + gap + panel_index * (thumb + gap)
            sheet.paste(
                Image.fromarray(array).resize((thumb, thumb), Image.Resampling.BILINEAR),
                (x, top + 8),
            )
            draw.text((x, top + thumb + 12), label, font=font(12, True), fill=(25, 25, 25))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def save_panels(data, heatmaps, source_to_pos, source_id, background, output_dir):
    names = ["source", "followup", "source_cam", "followup_cam"]
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, array in zip(
        names, arrays(data, heatmaps, source_to_pos, source_id, background)
    ):
        Image.fromarray(array).save(output_dir / f"{name}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--sheet", type=Path)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--source", type=int)
    parser.add_argument("--background", choices=sorted(BACKGROUND_INDEX))
    parser.add_argument("--panel-dir", type=Path)
    args = parser.parse_args()
    data, heatmaps, source_to_pos, records = load(args.root)
    if args.sheet:
        make_sheet(data, heatmaps, source_to_pos, records, args.sheet, args.limit)
    if args.source is not None:
        if not args.background or not args.panel_dir:
            parser.error("--source requires --background and --panel-dir")
        save_panels(
            data,
            heatmaps,
            source_to_pos,
            args.source,
            args.background,
            args.panel_dir,
        )


if __name__ == "__main__":
    main()
