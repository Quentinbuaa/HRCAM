"""Generate RQ1 figures and baseline table from saved holdout records."""
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import gaussian_kde, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from metrics import apply_minmax


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "rq1_v1"


def load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return None


def fontset():
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    return {
        "regular11": load_font(font_path, 11),
        "regular12": load_font(font_path, 12),
        "regular13": load_font(font_path, 13),
        "regular14": load_font(font_path, 14),
        "regular16": load_font(font_path, 16),
        "bold13": load_font(bold_path, 13),
        "bold14": load_font(bold_path, 14),
        "bold16": load_font(bold_path, 16),
        "bold18": load_font(bold_path, 18),
    }


def draw_centered(draw, xy, text, font, fill):
    x0, y0, x1, y1 = xy
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((x0 + (x1 - x0 - tw) / 2, y0 + (y1 - y0 - th) / 2), text, font=font, fill=fill)


def h_rcam(jsd, d4):
    c = d4 / 2
    return 2 * jsd * c / (jsd + c + 1e-12)


def load_holdout_own():
    records = [
        json.loads(line)
        for line in (ROOT / "runs" / "pets_v1" / "holdout_records.jsonl").read_text().splitlines()
        if line
    ]
    rows = sorted(
        [
            r for r in records
            if r["cam_mode"] == "own" and r["valid_cam"] and r["mask_eligible"]
        ],
        key=lambda r: (r["source_id"], r["background"]),
    )
    y = np.array([r["inconsistent"] for r in rows], dtype=int)
    jsd = np.array([r["jsd"] for r in rows], dtype=float)
    d4 = np.array([r["d4"] for r in rows], dtype=float)
    original = np.array([[r[n] for n in ["d1", "d2", "d3", "d4"]] for r in rows], dtype=float)
    config = json.loads((ROOT / "runs" / "pets_v1" / "frozen_analysis.json").read_text())
    norm, _ = apply_minmax(original, config["normalization"]["own"])
    scores = {
        "D1 centroid shift": norm[:, 0],
        "D2 KL distance": norm[:, 1],
        "D3 top-region distance": norm[:, 2],
        "D4/2 correlation distance": d4 / 2,
        "Original RCAM": norm.mean(axis=1),
        "JSD": jsd,
        "Arithmetic JSD-D4 fusion": (jsd + d4 / 2) / 2,
        "Geometric JSD-D4 fusion": np.sqrt(jsd * d4 / 2),
        "H-RCAM (harmonic fusion)": h_rcam(jsd, d4),
        "Confidence drop": np.array([r["confidence_drop"] for r in rows], dtype=float),
        "Absolute confidence drop": np.abs(np.array([r["confidence_drop"] for r in rows], dtype=float)),
        "Prediction JSD": np.array([r["prediction_jsd"] for r in rows], dtype=float),
    }
    return rows, y, scores


def save_table(y, scores):
    categories = {
        "D1 centroid shift": "Historic component",
        "D2 KL distance": "Historic component",
        "D3 top-region distance": "Historic component",
        "D4/2 correlation distance": "Historic component",
        "Original RCAM": "Historic aggregate",
        "JSD": "Single attention baseline",
        "Arithmetic JSD-D4 fusion": "Fixed fusion",
        "Geometric JSD-D4 fusion": "Fixed fusion",
        "H-RCAM (harmonic fusion)": "Proposed",
        "Confidence drop": "Output reference",
        "Absolute confidence drop": "Output reference",
        "Prediction JSD": "Output reference",
    }
    rows = []
    for name, score in scores.items():
        rows.append(
            {
                "Category": categories[name],
                "Method": name,
                "AUROC": roc_auc_score(y, score),
                "AP": average_precision_score(y, score),
            }
        )
    rows.sort(key=lambda r: (r["Category"] != "Proposed", -r["AUROC"]))
    # Keep manuscript order rather than sorted-by-score within all categories.
    order = list(categories)
    rows = sorted(rows, key=lambda r: order.index(r["Method"]))

    with (OUT / "table_rq1_metric_discrimination.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Category", "Method", "AUROC", "AP"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Category": row["Category"],
                    "Method": row["Method"],
                    "AUROC": f"{row['AUROC']:.4f}",
                    "AP": f"{row['AP']:.4f}",
                }
            )

    md = ["| Category | Method | AUROC | AP |", "|---|---|---:|---:|"]
    tex = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Category & Method & AUROC & AP \\",
        r"\midrule",
    ]
    for row in rows:
        md.append(f"| {row['Category']} | {row['Method']} | {row['AUROC']:.4f} | {row['AP']:.4f} |")
        tex.append(
            f"{row['Category']} & {row['Method']} & {row['AUROC']:.4f} & {row['AP']:.4f} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_rq1_metric_discrimination.md").write_text("\n".join(md) + "\n")
    (OUT / "table_rq1_metric_discrimination.tex").write_text("\n".join(tex) + "\n")


def draw_violin_box(draw, values, cx, width, y_to_px, color, outline, fontset):
    vals = np.asarray(values, dtype=float)
    grid = np.linspace(0, 0.18, 240)
    clipped = np.clip(vals, 0, 0.18)
    kde = gaussian_kde(clipped)
    density = kde(grid)
    density = density / max(density.max(), 1e-12) * width / 2
    left_points = [(cx - density[i], y_to_px(grid[i])) for i in range(len(grid))]
    right_points = [(cx + density[i], y_to_px(grid[i])) for i in range(len(grid) - 1, -1, -1)]
    draw.polygon(left_points + right_points, fill=color, outline=outline)
    q05, q25, q50, q75, q95 = np.quantile(vals, [0.05, 0.25, 0.5, 0.75, 0.95])
    q05, q25, q50, q75, q95 = [min(v, 0.18) for v in [q05, q25, q50, q75, q95]]
    box_w = 72
    draw.line((cx, y_to_px(q05), cx, y_to_px(q95)), fill=(35, 35, 35), width=2)
    draw.rectangle((cx - box_w / 2, y_to_px(q75), cx + box_w / 2, y_to_px(q25)), fill=(255, 255, 255), outline=(35, 35, 35), width=2)
    draw.line((cx - box_w / 2, y_to_px(q50), cx + box_w / 2, y_to_px(q50)), fill=(35, 35, 35), width=3)
    max_v = vals.max()
    draw.text((cx - 58, y_to_px(min(max_v, 0.18)) - 18), f"max={max_v:.3f}", font=fontset["regular11"], fill=(65, 65, 65))


def make_distribution_figure(y, h_scores, fontset):
    consistent = h_scores[y == 0]
    inconsistent = h_scores[y == 1]
    width, height = 900, 640
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 120, 55, width - 50, height - 105
    axis = (45, 45, 45)

    def y_to_px(v):
        return bottom - (bottom - top) * min(v, 0.18) / 0.18

    draw.line((left, bottom, right, bottom), fill=axis, width=2)
    draw.line((left, top, left, bottom), fill=axis, width=2)
    for tick in np.arange(0, 0.181, 0.03):
        ypx = y_to_px(float(tick))
        draw.line((left - 6, ypx, left, ypx), fill=axis, width=1)
        draw.text((left - 54, ypx - 8), f"{tick:.2f}", font=fontset["regular12"], fill=axis)
        draw.line((left, ypx, right, ypx), fill=(232, 232, 232), width=1)
    draw_centered(draw, (left + 120, height - 54, right - 120, height - 20), "Prediction consistency group", fontset["regular16"], axis)
    ylabel = Image.new("RGBA", (170, 30), (255, 255, 255, 0))
    yd = ImageDraw.Draw(ylabel)
    yd.text((0, 3), "H-RCAM score", font=fontset["regular16"], fill=axis)
    rotated = ylabel.rotate(90, expand=True)
    image.paste(rotated, (33, top + 150), rotated)

    draw_violin_box(draw, consistent, left + 210, 190, y_to_px, (220, 238, 226), (35, 120, 80), fontset)
    draw_violin_box(draw, inconsistent, left + 520, 190, y_to_px, (240, 219, 224), (160, 55, 70), fontset)
    draw_centered(draw, (left + 80, bottom + 16, left + 340, bottom + 46), f"Prediction-consistent\nn={len(consistent)}", fontset["bold14"], axis)
    draw_centered(draw, (left + 390, bottom + 16, left + 650, bottom + 46), f"Prediction-inconsistent\nn={len(inconsistent)}", fontset["bold14"], axis)
    draw.text((left + 18, top + 10), "Violin + boxplot; y-axis capped at 0.18", font=fontset["regular12"], fill=(70, 70, 70))
    med_c = np.median(consistent)
    med_i = np.median(inconsistent)
    draw.text((left + 335, top + 45), f"median: {med_c:.4f} -> {med_i:.4f}", font=fontset["bold14"], fill=(35, 35, 35))
    image.save(OUT / "fig_rq1_hrcam_consistent_vs_inconsistent.png")


def quantile_curve(x, y, n_bins=18):
    order = np.argsort(x)
    splits = np.array_split(order, n_bins)
    rows = []
    for idx in splits:
        if len(idx) < 4:
            continue
        rows.append(
            (
                float(np.median(x[idx])),
                float(np.quantile(y[idx], 0.25)),
                float(np.median(y[idx])),
                float(np.quantile(y[idx], 0.75)),
            )
        )
    return rows


def draw_relation_panel(draw, box, x, y, label, color, fontset):
    left, top, right, bottom = box
    axis = (45, 45, 45)
    x_cap = 0.14
    y_cap = max(np.quantile(y, 0.98), 1e-6)
    y_cap = min(max(y_cap, 0.02), 0.55)

    def x_px(v):
        return left + (right - left) * min(v, x_cap) / x_cap

    def y_px(v):
        return bottom - (bottom - top) * min(v, y_cap) / y_cap

    draw.rectangle((left, top, right, bottom), outline=(55, 55, 55), width=2)
    for tick in np.linspace(0, x_cap, 8):
        xp = x_px(tick)
        draw.line((xp, bottom, xp, bottom + 5), fill=axis, width=1)
        draw_centered(draw, (xp - 20, bottom + 9, xp + 20, bottom + 26), f"{tick:.2f}", fontset["regular11"], axis)
    for tick in np.linspace(0, y_cap, 5):
        yp = y_px(tick)
        draw.line((left - 5, yp, left, yp), fill=axis, width=1)
        draw.text((left - 55, yp - 7), f"{tick:.2f}", font=fontset["regular11"], fill=axis)
        draw.line((left, yp, right, yp), fill=(234, 234, 234), width=1)

    pts = quantile_curve(x, y)
    polygon = [(x_px(a), y_px(q1)) for a, q1, med, q3 in pts]
    polygon += [(x_px(a), y_px(q3)) for a, q1, med, q3 in reversed(pts)]
    draw.polygon(polygon, fill=tuple(list(color) + [70]) if len(color) == 3 else color)
    # PIL RGB images ignore alpha in tuple, so blend manually with light fill.
    draw.polygon(polygon, fill=tuple(int(0.75 * 255 + 0.25 * c) for c in color))
    curve = [(x_px(a), y_px(med)) for a, q1, med, q3 in pts]
    if len(curve) > 1:
        draw.line(curve, fill=color, width=4)
    for xp, yp in curve:
        draw.ellipse((xp - 4, yp - 4, xp + 4, yp + 4), fill=color)
    rho = spearmanr(x, y).statistic
    draw.text((left + 12, top + 10), f"{label}  (Spearman rho={rho:.3f})", font=fontset["bold14"], fill=axis)
    return rho


def make_relation_figure(rows, y_label, h_scores, fontset):
    pred_jsd = np.array([r["prediction_jsd"] for r in rows], dtype=float)
    abs_conf = np.abs(np.array([r["confidence_drop"] for r in rows], dtype=float))
    width, height = 1420, 580
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw_relation_panel(draw, (105, 62, 680, 460), h_scores, pred_jsd, "Prediction JSD", (55, 100, 150), fontset)
    draw_relation_panel(draw, (810, 62, 1385, 460), h_scores, abs_conf, "Absolute confidence drop", (150, 80, 70), fontset)
    draw_centered(draw, (390, height - 54, 1100, height - 18), "H-RCAM score (binned by quantiles; line = median, band = interquartile range)", fontset["regular16"], (45, 45, 45))
    ylabel = Image.new("RGBA", (200, 30), (255, 255, 255, 0))
    yd = ImageDraw.Draw(ylabel)
    yd.text((0, 3), "Output-level change", font=fontset["regular16"], fill=(45, 45, 45))
    rotated = ylabel.rotate(90, expand=True)
    image.paste(rotated, (28, 180), rotated)
    image.save(OUT / "fig_rq1_hrcam_vs_output_change.png")


def save_readme(rows, y, scores):
    h = scores["H-RCAM (harmonic fusion)"]
    consistent = h[y == 0]
    inconsistent = h[y == 1]
    text = f"""# RQ1 artifacts

Source: Oxford-IIIT Pet holdout, own-class Grad-CAM, valid and mask-eligible pairs.

Pairs: {len(y)}
Prediction-consistent pairs: {(y == 0).sum()}
Prediction-inconsistent pairs: {(y == 1).sum()}

H-RCAM consistent median: {np.median(consistent):.4f}
H-RCAM inconsistent median: {np.median(inconsistent):.4f}
H-RCAM AUROC: {roc_auc_score(y, h):.4f}
H-RCAM AP: {average_precision_score(y, h):.4f}

Generated files:

- fig_rq1_hrcam_consistent_vs_inconsistent.png
- table_rq1_metric_discrimination.csv
- table_rq1_metric_discrimination.md
- table_rq1_metric_discrimination.tex
- fig_rq1_hrcam_vs_output_change.png
"""
    (OUT / "RQ1_ARTIFACTS.md").write_text(text)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fs = fontset()
    rows, y, scores = load_holdout_own()
    save_table(y, scores)
    make_distribution_figure(y, scores["H-RCAM (harmonic fusion)"], fs)
    make_relation_figure(rows, y, scores["H-RCAM (harmonic fusion)"], fs)
    save_readme(rows, y, scores)
    for path in sorted(OUT.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
