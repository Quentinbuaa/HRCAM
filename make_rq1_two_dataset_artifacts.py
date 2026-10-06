"""Generate RQ1 artifacts for Pets and CUB holdout datasets."""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import gaussian_kde, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from metrics import apply_minmax


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "rq1_v2"
DATASETS = [
    ("pets_v1", "Oxford-IIIT Pet"),
    ("cub_v1", "CUB-200-2011"),
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
        "r11": load_font(font_path, 11),
        "r12": load_font(font_path, 12),
        "r13": load_font(font_path, 13),
        "r14": load_font(font_path, 14),
        "r16": load_font(font_path, 16),
        "b13": load_font(bold_path, 13),
        "b14": load_font(bold_path, 14),
        "b16": load_font(bold_path, 16),
        "b18": load_font(bold_path, 18),
    }


def center(draw, box, text, font, fill):
    x0, y0, x1, y1 = box
    bb = draw.textbbox((0, 0), text, font=font)
    draw.text((x0 + (x1 - x0 - (bb[2] - bb[0])) / 2, y0 + (y1 - y0 - (bb[3] - bb[1])) / 2), text, font=font, fill=fill)


def hrcam(jsd, d4):
    c = d4 / 2
    return 2 * jsd * c / (jsd + c + 1e-12)


def load_dataset(run, label):
    out = ROOT / "runs" / run
    rows = [
        json.loads(line)
        for line in (out / "holdout_records.jsonl").read_text().splitlines()
        if line
    ]
    rows = sorted(
        [r for r in rows if r["cam_mode"] == "own" and r["valid_cam"] and r["mask_eligible"]],
        key=lambda r: (r["source_id"], r["background"]),
    )
    y = np.array([r["inconsistent"] for r in rows], dtype=int)
    jsd = np.array([r["jsd"] for r in rows], dtype=float)
    d4 = np.array([r["d4"] for r in rows], dtype=float)
    raw = np.array([[r[k] for k in ["d1", "d2", "d3", "d4"]] for r in rows], dtype=float)
    config = json.loads((out / "frozen_analysis.json").read_text())
    norm, _ = apply_minmax(raw, config["normalization"]["own"])
    scores = {
        "D1 centroid shift": norm[:, 0],
        "D2 KL distance": norm[:, 1],
        "D3 top-region distance": norm[:, 2],
        "D4/2 correlation distance": d4 / 2,
        "Original RCAM": norm.mean(axis=1),
        "JSD": jsd,
        "Arithmetic JSD-D4 fusion": (jsd + d4 / 2) / 2,
        "Geometric JSD-D4 fusion": np.sqrt(jsd * d4 / 2),
        "H-RCAM (harmonic fusion)": hrcam(jsd, d4),
        "Confidence drop": np.array([r["confidence_drop"] for r in rows], dtype=float),
        "Absolute confidence drop": np.abs(np.array([r["confidence_drop"] for r in rows], dtype=float)),
        "Prediction JSD": np.array([r["prediction_jsd"] for r in rows], dtype=float),
    }
    return {"run": run, "label": label, "rows": rows, "y": y, "scores": scores}


def draw_violin(draw, vals, cx, width, ypx, color, outline, fs):
    vals = np.asarray(vals, dtype=float)
    ymax = 0.18
    grid = np.linspace(0, ymax, 220)
    kde = gaussian_kde(np.clip(vals, 0, ymax))
    dens = kde(grid)
    dens = dens / max(dens.max(), 1e-12) * width / 2
    left = [(cx - dens[i], ypx(grid[i])) for i in range(len(grid))]
    right = [(cx + dens[i], ypx(grid[i])) for i in range(len(grid) - 1, -1, -1)]
    draw.polygon(left + right, fill=color, outline=outline)
    q05, q25, q50, q75, q95 = np.quantile(vals, [0.05, 0.25, 0.5, 0.75, 0.95])
    q05, q25, q50, q75, q95 = [min(v, ymax) for v in [q05, q25, q50, q75, q95]]
    draw.line((cx, ypx(q05), cx, ypx(q95)), fill=(35, 35, 35), width=2)
    draw.rectangle((cx - 34, ypx(q75), cx + 34, ypx(q25)), fill=(255, 255, 255), outline=(35, 35, 35), width=2)
    draw.line((cx - 34, ypx(q50), cx + 34, ypx(q50)), fill=(35, 35, 35), width=3)
    draw.text((cx - 44, ypx(min(vals.max(), ymax)) - 16), f"max={vals.max():.3f}", font=fs["r11"], fill=(65, 65, 65))
    draw.text((cx + 40, ypx(q50) - 8), f"median={q50:.4f}", font=fs["b13"], fill=(35, 35, 35))


def make_violin_figure(datasets, fs):
    width, height = 1500, 650
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    panel_w = 650
    top, bottom = 70, 520
    axis = (45, 45, 45)
    for i, data in enumerate(datasets):
        left = 95 + i * 705
        right = left + panel_w
        def ypx(v):
            return bottom - (bottom - top) * min(v, 0.18) / 0.18
        draw.text((left + 210, 26), f"({chr(97+i)}) {data['label']}", font=fs["b16"], fill=axis)
        draw.line((left, bottom, right, bottom), fill=axis, width=2)
        draw.line((left, top, left, bottom), fill=axis, width=2)
        for tick in np.arange(0, 0.181, 0.03):
            y = ypx(float(tick))
            draw.line((left - 6, y, left, y), fill=axis, width=1)
            if i == 0:
                draw.text((left - 54, y - 8), f"{tick:.2f}", font=fs["r12"], fill=axis)
            draw.line((left, y, right, y), fill=(232, 232, 232), width=1)
        h = data["scores"]["H-RCAM (harmonic fusion)"]
        y = data["y"]
        draw_violin(draw, h[y == 0], left + 205, 170, ypx, (220, 238, 226), (35, 120, 80), fs)
        draw_violin(draw, h[y == 1], left + 465, 170, ypx, (240, 219, 224), (160, 55, 70), fs)
        center(draw, (left + 90, bottom + 15, left + 320, bottom + 58), f"Prediction-consistent\nn={(y==0).sum()}", fs["b13"], axis)
        center(draw, (left + 350, bottom + 15, left + 590, bottom + 58), f"Prediction-inconsistent\nn={(y==1).sum()}", fs["b13"], axis)
        x0, y0 = left + 315, ypx(np.median(h[y == 0]))
        x1, y1 = left + 355, ypx(np.median(h[y == 1]))
        draw.line((x0, y0, x1, y1), fill=(90, 90, 90), width=2)
        draw.polygon([(x1, y1), (x1 - 8, y1 - 4), (x1 - 7, y1 + 5)], fill=(90, 90, 90))
    ylabel = Image.new("RGBA", (170, 30), (255, 255, 255, 0))
    yd = ImageDraw.Draw(ylabel)
    yd.text((0, 3), "H-RCAM score", font=fs["r16"], fill=axis)
    rotated = ylabel.rotate(90, expand=True)
    image.paste(rotated, (28, 235), rotated)
    center(draw, (480, height - 42, 1030, height - 10), "Prediction consistency group", fs["r16"], axis)
    image.save(OUT / "fig_rq1_hrcam_consistent_vs_inconsistent_2datasets.png")


def draw_box(draw, values, cx, width, ypx, fill):
    vals = np.asarray(values, dtype=float)
    q05, q25, q50, q75, q95 = np.quantile(vals, [0.05, 0.25, 0.5, 0.75, 0.95])
    ymax = 1.0
    q05, q25, q50, q75, q95 = [min(v, ymax) for v in [q05, q25, q50, q75, q95]]
    draw.line((cx, ypx(q05), cx, ypx(q95)), fill=(45, 45, 45), width=2)
    draw.rectangle((cx - width / 2, ypx(q75), cx + width / 2, ypx(q25)), fill=fill, outline=(35, 35, 35), width=2)
    draw.line((cx - width / 2, ypx(q50), cx + width / 2, ypx(q50)), fill=(35, 35, 35), width=3)


def make_metric_distribution_figure(datasets, fs):
    methods = [
        ("Original RCAM", "Original\nRCAM"),
        ("JSD", "JSD"),
        ("D4/2 correlation distance", "D4/2"),
        ("Arithmetic JSD-D4 fusion", "Arithmetic\nfusion"),
        ("Geometric JSD-D4 fusion", "Geometric\nfusion"),
        ("H-RCAM (harmonic fusion)", "H-RCAM"),
    ]
    width, height = 1500, 780
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    axis = (45, 45, 45)
    row_h = 320
    for row, data in enumerate(datasets):
        left, right = 115, width - 45
        top = 55 + row * 350
        bottom = top + 250

        def ypx(v):
            return bottom - (bottom - top) * min(v, 1.0)

        draw.text((left + 520, top - 35), f"({chr(97 + row)}) {data['label']}", font=fs["b16"], fill=axis)
        draw.line((left, bottom, right, bottom), fill=axis, width=2)
        draw.line((left, top, left, bottom), fill=axis, width=2)
        for tick in np.linspace(0, 1, 6):
            y = ypx(float(tick))
            draw.line((left - 6, y, left, y), fill=axis, width=1)
            draw.text((left - 44, y - 8), f"{tick:.1f}", font=fs["r12"], fill=axis)
            draw.line((left, y, right, y), fill=(234, 234, 234), width=1)
        y = data["y"]
        group_gap = (right - left) / len(methods)
        for i, (key, label) in enumerate(methods):
            center_x = left + group_gap * (i + 0.5)
            values = data["scores"][key]
            draw_box(draw, values[y == 0], center_x - 28, 42, ypx, (220, 238, 226))
            draw_box(draw, values[y == 1], center_x + 28, 42, ypx, (240, 219, 224))
            center(draw, (center_x - 65, bottom + 10, center_x + 65, bottom + 48), label, fs["r12"], axis)
        if row == 0:
            draw.rectangle((right - 225, top + 8, right - 205, top + 28), fill=(220, 238, 226), outline=axis)
            draw.text((right - 195, top + 8), "Prediction-consistent", font=fs["r13"], fill=axis)
            draw.rectangle((right - 225, top + 34, right - 205, top + 54), fill=(240, 219, 224), outline=axis)
            draw.text((right - 195, top + 34), "Prediction-inconsistent", font=fs["r13"], fill=axis)
    ylabel = Image.new("RGBA", (180, 30), (255, 255, 255, 0))
    yd = ImageDraw.Draw(ylabel)
    yd.text((0, 3), "Metric score", font=fs["r16"], fill=axis)
    rotated = ylabel.rotate(90, expand=True)
    image.paste(rotated, (28, 320), rotated)
    center(draw, (520, height - 46, 980, height - 14), "Attention-map metric", fs["r16"], axis)
    image.save(OUT / "fig_rq1_metric_distributions_by_method_2datasets.png")


def make_relation_figure(datasets, fs):
    width, height = 1450, 760
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    axis = (45, 45, 45)
    colors = [(55, 100, 150), (150, 80, 70)]
    for row, data in enumerate(datasets):
        for col, metric in enumerate(["Prediction JSD", "Absolute confidence drop"]):
            x = data["scores"]["H-RCAM (harmonic fusion)"]
            y = data["scores"][metric]
            left = 105 + col * 675
            top = 58 + row * 330
            right = left + 560
            bottom = top + 250
            x_cap = 0.14
            y_cap = min(max(np.quantile(y, .98), .05), .55)
            def xp(v):
                return left + (right - left) * min(v, x_cap) / x_cap
            def yp(v):
                return bottom - (bottom - top) * min(v, y_cap) / y_cap
            draw.rectangle((left, top, right, bottom), outline=axis, width=2)
            for t in np.linspace(0, x_cap, 5):
                draw.line((xp(t), bottom, xp(t), bottom + 5), fill=axis)
                center(draw, (xp(t)-20, bottom+8, xp(t)+20, bottom+26), f"{t:.2f}", fs["r11"], axis)
            for t in np.linspace(0, y_cap, 4):
                draw.line((left - 5, yp(t), left, yp(t)), fill=axis)
                draw.text((left - 48, yp(t)-7), f"{t:.2f}", font=fs["r11"], fill=axis)
                draw.line((left, yp(t), right, yp(t)), fill=(234, 234, 234))
            order = np.argsort(x)
            bins = np.array_split(order, 18)
            pts = []
            for idx in bins:
                pts.append((np.median(x[idx]), np.quantile(y[idx], .25), np.median(y[idx]), np.quantile(y[idx], .75)))
            fill = tuple(int(.75*255 + .25*c) for c in colors[col])
            poly = [(xp(a), yp(q1)) for a, q1, m, q3 in pts] + [(xp(a), yp(q3)) for a, q1, m, q3 in reversed(pts)]
            draw.polygon(poly, fill=fill)
            curve = [(xp(a), yp(m)) for a, q1, m, q3 in pts]
            draw.line(curve, fill=colors[col], width=4)
            rho = spearmanr(x, y).statistic
            draw.text((left + 12, top + 10), f"{data['label']}: {metric} (rho={rho:.3f})", font=fs["b13"], fill=axis)
    center(draw, (430, height - 44, 1030, height - 12), "H-RCAM score (quantile-binned median; band = interquartile range)", fs["r16"], axis)
    image.save(OUT / "fig_rq1_hrcam_vs_output_change_2datasets.png")


def make_table(datasets):
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
    for data in datasets:
        y = data["y"]
        for method, score in data["scores"].items():
            rows.append({
                "Dataset": data["label"],
                "Category": categories[method],
                "Method": method,
                "AUROC": roc_auc_score(y, score),
                "AP": average_precision_score(y, score),
            })
    with (OUT / "table_rq1_metric_discrimination_2datasets.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Dataset", "Category", "Method", "AUROC", "AP"])
        writer.writeheader()
        for r in rows:
            writer.writerow({**{k: r[k] for k in ["Dataset", "Category", "Method"]}, "AUROC": f"{r['AUROC']:.4f}", "AP": f"{r['AP']:.4f}"})
    lines = ["| Dataset | Category | Method | AUROC | AP |", "|---|---|---|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['Dataset']} | {r['Category']} | {r['Method']} | {r['AUROC']:.4f} | {r['AP']:.4f} |")
    (OUT / "table_rq1_metric_discrimination_2datasets.md").write_text("\n".join(lines) + "\n")
    tex = [r"\begin{tabular}{lllrr}", r"\toprule", r"Dataset & Category & Method & AUROC & AP \\", r"\midrule"]
    for r in rows:
        tex.append(f"{r['Dataset']} & {r['Category']} & {r['Method']} & {r['AUROC']:.4f} & {r['AP']:.4f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_rq1_metric_discrimination_2datasets.tex").write_text("\n".join(tex) + "\n")


def make_readme(datasets):
    lines = ["# RQ1 two-dataset artifacts", ""]
    for d in datasets:
        y = d["y"]
        h = d["scores"]["H-RCAM (harmonic fusion)"]
        lines += [
            f"## {d['label']}",
            "",
            f"Pairs: {len(y)}",
            f"Prediction-consistent: {(y == 0).sum()}",
            f"Prediction-inconsistent: {(y == 1).sum()}",
            f"H-RCAM AUROC: {roc_auc_score(y, h):.4f}",
            f"H-RCAM AP: {average_precision_score(y, h):.4f}",
            "",
        ]
    lines += [
        "Generated files:",
        "",
        "- fig_rq1_hrcam_consistent_vs_inconsistent_2datasets.png",
        "- fig_rq1_metric_distributions_by_method_2datasets.png",
        "- fig_rq1_hrcam_vs_output_change_2datasets.png",
        "- table_rq1_metric_discrimination_2datasets.csv/md/tex",
    ]
    (OUT / "RQ1_TWO_DATASET_ARTIFACTS.md").write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fs = fonts()
    datasets = [load_dataset(run, label) for run, label in DATASETS]
    make_violin_figure(datasets, fs)
    make_metric_distribution_figure(datasets, fs)
    make_relation_figure(datasets, fs)
    make_table(datasets)
    make_readme(datasets)
    for p in sorted(OUT.iterdir()):
        print(p)


if __name__ == "__main__":
    main()
