"""Final RQ1 artifacts: group divergence and discrimination across two datasets."""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import gaussian_kde, ks_2samp, mannwhitneyu
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from metrics import apply_minmax
from transport_metrics import distances


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "rq1_v3"
BACKGROUNDS = ["black", "gray", "white"]
DATASETS = [("pets_v1", "Oxford-IIIT Pet"), ("cub_v1", "CUB-200-2011")]
METHODS = [
    ("D1", "D1"),
    ("D2", "D2"),
    ("D3", "D3"),
    ("D4_half", "D4/2"),
    ("JSD", "JSD"),
    ("BAT", "BAT"),
    ("SpatialW1", "SpatialW1"),
    ("ForegroundShift", "Foreground shift"),
    ("BackgroundIncrease", "Background increase"),
    ("HRCAM", "H-RCAM"),
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
        "r10": load_font(font_path, 10),
        "r11": load_font(font_path, 11),
        "r12": load_font(font_path, 12),
        "r13": load_font(font_path, 13),
        "r14": load_font(font_path, 14),
        "r16": load_font(font_path, 16),
        "b12": load_font(bold_path, 12),
        "b13": load_font(bold_path, 13),
        "b14": load_font(bold_path, 14),
        "b16": load_font(bold_path, 16),
    }


def center(draw, box, text, font, fill):
    x0, y0, x1, y1 = box
    bb = draw.textbbox((0, 0), text, font=font)
    draw.text((x0 + (x1 - x0 - (bb[2] - bb[0])) / 2, y0 + (y1 - y0 - (bb[3] - bb[1])) / 2), text, font=font, fill=fill)


def hrcam(j, d4):
    c = d4 / 2
    return 2 * j * c / (j + c + 1e-12)


def transport_cache(run):
    output = OUT / f"{run}_holdout_transport_own.jsonl"
    if output.exists():
        rows = [json.loads(line) for line in output.read_text().splitlines() if line]
        return {(r["source_id"], r["background"]): r for r in rows}

    run_dir = ROOT / "runs" / run
    records = [
        json.loads(line)
        for line in (run_dir / "holdout_records.jsonl").read_text().splitlines()
        if line
    ]
    selected = [
        r for r in records
        if r["cam_mode"] == "own" and r["valid_cam"] and r["mask_eligible"]
    ]
    archive = np.load(run_dir / "holdout_heatmaps.npz")
    source_ids = archive["source_ids"]
    positions = {int(source): index for index, source in enumerate(source_ids)}
    originals = archive["original_maps"]
    maps = archive["maps"]
    masks = archive["masks"]

    rows = []
    for count, r in enumerate(selected):
        pos = positions[r["source_id"]]
        bg = BACKGROUNDS.index(r["background"])
        values = distances(originals[pos], maps[pos, bg, 0], masks[pos], grid=14)
        rows.append({
            "source_id": r["source_id"],
            "background": r["background"],
            "spatial_w1": values["spatial_w1"],
            "BAT": values["BAT"],
            "foreground_mass_change": values["foreground_mass_change"],
        })
        if count % 500 == 0:
            print(run, count, len(selected), flush=True)
    output.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {(r["source_id"], r["background"]): r for r in rows}


def load_dataset(run, label):
    run_dir = ROOT / "runs" / run
    records = [
        json.loads(line)
        for line in (run_dir / "holdout_records.jsonl").read_text().splitlines()
        if line
    ]
    rows = sorted(
        [r for r in records if r["cam_mode"] == "own" and r["valid_cam"] and r["mask_eligible"]],
        key=lambda r: (r["source_id"], r["background"]),
    )
    y = np.asarray([r["inconsistent"] for r in rows], dtype=int)
    raw = np.asarray([[r[k] for k in ["d1", "d2", "d3", "d4"]] for r in rows], dtype=float)
    config = json.loads((run_dir / "frozen_analysis.json").read_text())
    norm, _ = apply_minmax(raw, config["normalization"]["own"])
    transport = transport_cache(run)
    jsd = np.asarray([r["jsd"] for r in rows], dtype=float)
    d4 = np.asarray([r["d4"] for r in rows], dtype=float)
    scores = {
        "D1": norm[:, 0],
        "D2": norm[:, 1],
        "D3": norm[:, 2],
        "D4_half": d4 / 2,
        "JSD": jsd,
        "BAT": np.asarray([transport[(r["source_id"], r["background"])]["BAT"] for r in rows], dtype=float),
        "SpatialW1": np.asarray([transport[(r["source_id"], r["background"])]["spatial_w1"] for r in rows], dtype=float),
        "ForegroundShift": np.asarray([r["foreground_shift"] for r in rows], dtype=float),
        "BackgroundIncrease": np.asarray([r["background_increase"] for r in rows], dtype=float),
        "HRCAM": hrcam(jsd, d4),
    }
    return {"run": run, "label": label, "rows": rows, "y": y, "scores": scores}


def format_p(value):
    if value < 1e-300:
        return "<1e-300"
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def make_rq11_table(datasets):
    table_rows = []
    for data_index, data in enumerate(datasets, start=1):
        y = data["y"]
        n_consistent = int((y == 0).sum())
        n_changed = int((y == 1).sum())
        for key, label in METHODS:
            score = data["scores"][key]
            a = score[y == 0]
            b = score[y == 1]
            u = mannwhitneyu(a, b, alternative="two-sided")
            ks = ks_2samp(a, b, alternative="two-sided", mode="auto")
            table_rows.append({
                "Dataset": f"Dataset {data_index}",
                "Method": label,
                "Delta_median": np.median(b) - np.median(a),
                "Delta_mean": np.mean(b) - np.mean(a),
                "Delta_max": np.max(b) - np.max(a),
                "Delta_min": np.min(b) - np.min(a),
                "U_statistic": u.statistic,
                "U_p_value": u.pvalue,
                "U_effect": 1 - u.statistic / (n_consistent * n_changed),
                "KS_statistic": ks.statistic,
                "KS_p_value": ks.pvalue,
            })
    with (OUT / "table_rq1_1_group_divergence.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table_rows[0]))
        writer.writeheader()
        for r in table_rows:
            writer.writerow({
                k: (f"{v:.6g}" if isinstance(v, float) else v)
                for k, v in r.items()
            })
    best_effect = {
        f"Dataset {data_index}": max(
            r["U_effect"] for r in table_rows if r["Dataset"] == f"Dataset {data_index}"
        )
        for data_index, _ in enumerate(datasets, start=1)
    }
    best_p = {
        f"Dataset {data_index}": min(
            r["U_p_value"] for r in table_rows if r["Dataset"] == f"Dataset {data_index}"
        )
        for data_index, _ in enumerate(datasets, start=1)
    }
    best_u = {
        f"Dataset {data_index}": min(
            r["U_statistic"] for r in table_rows if r["Dataset"] == f"Dataset {data_index}"
        )
        for data_index, _ in enumerate(datasets, start=1)
    }

    def bold_md(text, condition):
        return f"**{text}**" if condition else text

    md = [
        "| Dataset | Method | $\\Delta$ median | $\\Delta$ mean | $\\Delta$ max | $\\Delta$ min | U | p-value | A_U |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in table_rows:
        is_best_effect = np.isclose(r["U_effect"], best_effect[r["Dataset"]])
        is_best_p = r["U_p_value"] == best_p[r["Dataset"]]
        is_best_u = r["U_statistic"] == best_u[r["Dataset"]]
        method = bold_md(r["Method"], is_best_effect)
        md.append(
            f"| {r['Dataset']} | {method} | {r['Delta_median']:.4f} | {r['Delta_mean']:.4f} | "
            f"{r['Delta_max']:.4f} | {r['Delta_min']:.4f} | {bold_md(f'{r['U_statistic']:.1f}', is_best_u)} | "
            f"{bold_md(format_p(r['U_p_value']), is_best_p)} | {bold_md(f'{r['U_effect']:.4f}', is_best_effect)} |"
        )
    (OUT / "table_rq1_1_group_divergence.md").write_text("\n".join(md) + "\n")

    tex = [
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Dataset & Method & \multicolumn{4}{c}{Basic Statistic Variations} & \multicolumn{3}{c}{Consistency Hypothesis Test} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-9}",
        r"& & $\Delta$ Median & $\Delta$ Mean & $\Delta$ Max & $\Delta$ Min & U & p & $A_U$ \\",
        r"\midrule",
    ]

    def bold_tex(text, condition):
        return rf"\textbf{{{text}}}" if condition else text

    previous_dataset = None
    for r in table_rows:
        is_best_effect = np.isclose(r["U_effect"], best_effect[r["Dataset"]])
        is_best_p = r["U_p_value"] == best_p[r["Dataset"]]
        is_best_u = r["U_statistic"] == best_u[r["Dataset"]]
        dataset_rows = [row for row in table_rows if row["Dataset"] == r["Dataset"]]
        dataset_cell = rf"\multirow{{{len(dataset_rows)}}}{{*}}{{{r['Dataset']}}}" if r["Dataset"] != previous_dataset else ""
        if previous_dataset is not None and r["Dataset"] != previous_dataset:
            tex.append(r"\midrule")
        previous_dataset = r["Dataset"]
        tex.append(
            f"{dataset_cell} & {bold_tex(r['Method'], is_best_effect)} & {r['Delta_median']:.4f} & "
            f"{r['Delta_mean']:.4f} & {r['Delta_max']:.4f} & {r['Delta_min']:.4f} & "
            f"{bold_tex(f'{r['U_statistic']:.1f}', is_best_u)} & {bold_tex(format_p(r['U_p_value']), is_best_p)} & "
            f"{bold_tex(f'{r['U_effect']:.4f}', is_best_effect)} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_rq1_1_group_divergence.tex").write_text("\n".join(tex) + "\n")


def draw_violin(draw, vals, cx, width, ypx, color, outline, fs):
    vals = np.asarray(vals)
    grid = np.linspace(0, 0.18, 220)
    kde = gaussian_kde(np.clip(vals, 0, 0.18))
    dens = kde(grid)
    dens = dens / max(dens.max(), 1e-12) * width / 2
    left = [(cx - dens[i], ypx(grid[i])) for i in range(len(grid))]
    right = [(cx + dens[i], ypx(grid[i])) for i in range(len(grid) - 1, -1, -1)]
    draw.polygon(left + right, fill=color, outline=outline)
    q05, q25, q50, q75, q95 = np.quantile(vals, [.05, .25, .5, .75, .95])
    q05, q25, q50, q75, q95 = [min(v, .18) for v in [q05, q25, q50, q75, q95]]
    draw.line((cx, ypx(q05), cx, ypx(q95)), fill=(35, 35, 35), width=2)
    draw.rectangle((cx - 34, ypx(q75), cx + 34, ypx(q25)), fill=(255, 255, 255), outline=(35, 35, 35), width=2)
    draw.line((cx - 34, ypx(q50), cx + 34, ypx(q50)), fill=(35, 35, 35), width=3)
    draw.text((cx + 42, ypx(q50) - 8), f"median={q50:.4f}", font=fs["b13"], fill=(35, 35, 35))
    draw.text((cx - 42, ypx(min(vals.max(), .18)) - 16), f"max={vals.max():.3f}", font=fs["r11"], fill=(65, 65, 65))


def make_hrcam_violin(datasets, fs):
    width, height = 1500, 650
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    axis = (45, 45, 45)
    top, bottom = 70, 520
    for i, data in enumerate(datasets):
        left = 95 + i * 705
        right = left + 650

        def ypx(v):
            return bottom - (bottom - top) * min(v, .18) / .18

        center(draw, (left, 18, right, 50), f"Dataset {i + 1}", fs["b16"], axis)
        draw.line((left, bottom, right, bottom), fill=axis, width=2)
        draw.line((left, top, left, bottom), fill=axis, width=2)
        for tick in np.arange(0, .181, .03):
            y = ypx(float(tick))
            draw.line((left - 6, y, left, y), fill=axis)
            if i == 0:
                draw.text((left - 54, y - 8), f"{tick:.2f}", font=fs["r12"], fill=axis)
            draw.line((left, y, right, y), fill=(232, 232, 232))
        score = data["scores"]["HRCAM"]
        y = data["y"]
        consistent = score[y == 0]
        changed = score[y == 1]
        draw_violin(draw, score[y == 0], left + 205, 170, ypx, (220, 238, 226), (35, 120, 80), fs)
        draw_violin(draw, score[y == 1], left + 465, 170, ypx, (240, 219, 224), (160, 55, 70), fs)
        center(draw, (left + 80, bottom + 15, left + 320, bottom + 58), f"Prediction-consistent\nn={(y == 0).sum()}", fs["b13"], axis)
        center(draw, (left + 350, bottom + 15, left + 590, bottom + 58), f"Prediction-inconsistent\nn={(y == 1).sum()}", fs["b13"], axis)
        u = mannwhitneyu(consistent, changed, alternative="two-sided")
        box_lines = [
            f"delta_median={np.median(changed) - np.median(consistent):.4f}",
            f"delta_mean={np.mean(changed) - np.mean(consistent):.4f}",
            f"delta_min={np.min(changed) - np.min(consistent):.4f}",
            f"delta_max={np.max(changed) - np.max(consistent):.4f}",
            f"U={u.statistic:.1f}",
            f"p={format_p(u.pvalue)}",
        ]
        bx0, by0, bx1, by1 = left + 298, top + 88, left + 438, top + 202
        draw.rounded_rectangle((bx0, by0, bx1, by1), radius=4, fill=(255, 255, 255), outline=(95, 95, 95), width=1)
        draw.multiline_text((bx0 + 9, by0 + 8), "\n".join(box_lines), font=fs["r11"], fill=axis, spacing=3)
    ylabel = Image.new("RGBA", (170, 30), (255, 255, 255, 0))
    yd = ImageDraw.Draw(ylabel)
    yd.text((0, 3), "H-RCAM score", font=fs["r16"], fill=axis)
    rot = ylabel.rotate(90, expand=True)
    img.paste(rot, (18, 235), rot)
    center(draw, (480, height - 42, 1030, height - 10), "Prediction consistency group", fs["r16"], axis)
    img.save(OUT / "fig_rq1_1_hrcam_group_divergence_2datasets.png")


def make_auc_ap_curve(datasets, fs):
    width, height = 1600, 720
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    axis = (45, 45, 45)
    palette = {
        "D1": (130, 130, 130),
        "D2": (160, 160, 160),
        "D3": (100, 120, 160),
        "D4/2": (53, 111, 175),
        "JSD": (74, 150, 104),
        "BAT": (166, 112, 62),
        "SpatialW1": (137, 99, 168),
        "Foreground shift": (184, 140, 42),
        "Background increase": (145, 145, 85),
        "H-RCAM": (175, 45, 64),
    }
    for i, data in enumerate(datasets):
        left = 95 + i * 755
        right = left + 610
        top, bottom = 80, 555

        def xp(v):
            return left + (right - left) * v

        def yp(v):
            return bottom - (bottom - top) * v

        center(draw, (left, 25, right, 55), f"Dataset {i + 1}", fs["b16"], axis)
        draw.line((left, bottom, right, bottom), fill=axis, width=2)
        draw.line((left, top, left, bottom), fill=axis, width=2)
        for tick in np.linspace(0, 1, 6):
            pos_y = yp(float(tick))
            pos_x = xp(float(tick))
            draw.line((left - 6, pos_y, left, pos_y), fill=axis)
            draw.line((pos_x, bottom, pos_x, bottom + 6), fill=axis)
            if i == 0:
                draw.text((left - 42, pos_y - 8), f"{tick:.1f}", font=fs["r12"], fill=axis)
            draw.text((pos_x - 10, bottom + 13), f"{tick:.1f}", font=fs["r11"], fill=axis)
            draw.line((left, pos_y, right, pos_y), fill=(232, 232, 232))
            draw.line((pos_x, top, pos_x, bottom), fill=(242, 242, 242))
        draw.line((left, bottom, right, top), fill=(185, 185, 185), width=2)
        y = data["y"]
        aucs = []
        for key, label in METHODS:
            score = data["scores"][key]
            fpr, tpr, _ = roc_curve(y, score)
            auc_value = roc_auc_score(y, score)
            aucs.append((label, auc_value))
            pts = [(xp(float(fpr[j])), yp(float(tpr[j]))) for j in range(len(fpr))]
            color = palette[label]
            line_width = 5 if label == "H-RCAM" else 2
            draw.line(pts, fill=color, width=line_width)

        legend_x = right - 190
        legend_y = top + 22
        draw.rounded_rectangle((legend_x - 14, legend_y - 12, right - 12, legend_y + 252), radius=4, fill=(255, 255, 255), outline=(205, 205, 205))
        for j, (label, auc_value) in enumerate(sorted(aucs, key=lambda item: item[1], reverse=True)):
            y0 = legend_y + j * 24
            color = palette[label]
            line_width = 5 if label == "H-RCAM" else 2
            text_font = fs["b12"] if label == "H-RCAM" else fs["r11"]
            draw.line((legend_x, y0 + 8, legend_x + 28, y0 + 8), fill=color, width=line_width)
            draw.text((legend_x + 35, y0), f"{label}: {auc_value:.4f}", font=text_font, fill=axis)
    ylabel = Image.new("RGBA", (180, 30), (255, 255, 255, 0))
    yd = ImageDraw.Draw(ylabel)
    yd.text((0, 3), "True positive rate", font=fs["r16"], fill=axis)
    rot = ylabel.rotate(90, expand=True)
    img.paste(rot, (25, 270), rot)
    center(draw, (570, height - 50, 1030, height - 18), "False positive rate", fs["r16"], axis)
    img.save(OUT / "fig_rq1_2_roc_curves_2datasets.png")
    img.save(OUT / "fig_rq1_2_auroc_ap_by_method_2datasets.png")


def make_auc_ap_table(datasets):
    rows = []
    for data in datasets:
        y = data["y"]
        for key, label in METHODS:
            score = data["scores"][key]
            rows.append({
                "Dataset": data["label"],
                "Method": label,
                "AUROC": roc_auc_score(y, score),
                "AP": average_precision_score(y, score),
            })
    with (OUT / "table_rq1_2_auroc_ap.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    md = [
        "| Dataset | Method | AUROC | AP |",
        "|---|---|---:|---:|",
    ]
    for r in rows:
        md.append(f"| {r['Dataset']} | {r['Method']} | {r['AUROC']:.4f} | {r['AP']:.4f} |")
    (OUT / "table_rq1_2_auroc_ap.md").write_text("\n".join(md) + "\n")


def make_readme(datasets):
    lines = ["# RQ1 final artifacts", ""]
    for data in datasets:
        y = data["y"]
        h = data["scores"]["HRCAM"]
        lines += [
            f"## {data['label']}",
            f"Pairs: {len(y)}",
            f"Prediction-consistent: {(y == 0).sum()}",
            f"Prediction-inconsistent: {(y == 1).sum()}",
            f"H-RCAM AUROC: {roc_auc_score(y, h):.4f}",
            f"H-RCAM AP: {average_precision_score(y, h):.4f}",
            "",
        ]
    lines += [
        "Methods: D1, D2, D3 use frozen calibrated component scores; D4 is divided by 2.",
        "BAT and SpatialW1 are computed on the holdout heatmaps with a 14x14 transport grid.",
    ]
    (OUT / "RQ1_FINAL_ARTIFACTS.md").write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fs = fonts()
    datasets = [load_dataset(run, label) for run, label in DATASETS]
    make_hrcam_violin(datasets, fs)
    make_rq11_table(datasets)
    make_auc_ap_curve(datasets, fs)
    make_auc_ap_table(datasets)
    make_readme(datasets)
    for path in sorted(OUT.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
