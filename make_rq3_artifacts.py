"""RQ3 sensitivity and ablation tables."""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "rq3_v1"
DATASETS = [("pets_v1", "Dataset 1"), ("cub_v1", "Dataset 2")]
BACKGROUNDS = ["black", "gray", "white"]
FUSIONS = [
    ("JSD", "JSD"),
    ("D4_half", "D4/2"),
    ("Min", "min(JSD,D4/2)"),
    ("Max", "max(JSD,D4/2)"),
    ("Arithmetic", "Arithmetic"),
    ("Geometric", "Geometric"),
    ("HRCAM", "H-RCAM"),
]


def records_path(run):
    for base in [ROOT / "runs" / run, ROOT / "results" / run]:
        path = base / "holdout_records.jsonl"
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing holdout_records.jsonl for {run}")


def load_rows(run):
    rows = [
        json.loads(line)
        for line in records_path(run).read_text().splitlines()
        if line
    ]
    return [
        r for r in rows
        if r["cam_mode"] == "own" and r["valid_cam"] and r["mask_eligible"]
    ]


def hrcam(jsd, d4):
    c = d4 / 2
    return 2 * jsd * c / (jsd + c + 1e-12)


def score_rows(rows, method):
    jsd = np.asarray([r["jsd"] for r in rows], dtype=float)
    c = np.asarray([r["d4"] / 2 for r in rows], dtype=float)
    if method == "JSD":
        return jsd
    if method == "D4_half":
        return c
    if method == "Min":
        return np.minimum(jsd, c)
    if method == "Max":
        return np.maximum(jsd, c)
    if method == "Arithmetic":
        return (jsd + c) / 2
    if method == "Geometric":
        return np.sqrt(jsd * c)
    if method == "HRCAM":
        return 2 * jsd * c / (jsd + c + 1e-12)
    raise KeyError(method)


def delta_stats(y, score):
    consistent = score[y == 0]
    changed = score[y == 1]
    return {
        "Delta_median": np.median(changed) - np.median(consistent),
        "Delta_mean": np.mean(changed) - np.mean(consistent),
        "Delta_max": np.max(changed) - np.max(consistent),
        "Delta_min": np.min(changed) - np.min(consistent),
    }


def format_paper_float(value):
    return f"{value:.4f}"


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: (f"{value:.6g}" if isinstance(value, float) else value)
                for key, value in row.items()
            })


def bold_tex(text, condition):
    return rf"\textbf{{{text}}}" if condition else text


def bold_md(text, condition):
    return f"**{text}**" if condition else text


def make_background_table(datasets):
    rows = []
    for dataset, label, data_rows in datasets:
        for background in BACKGROUNDS + ["Overall"]:
            subset = data_rows if background == "Overall" else [r for r in data_rows if r["background"] == background]
            y = np.asarray([r["inconsistent"] for r in subset], dtype=int)
            score = hrcam(
                np.asarray([r["jsd"] for r in subset], dtype=float),
                np.asarray([r["d4"] for r in subset], dtype=float),
            )
            row = {
                "Dataset": label,
                "Background": background,
                "n": len(subset),
                "Inconsistency_rate": float(y.mean()),
                **delta_stats(y, score),
                "AUROC": roc_auc_score(y, score),
            }
            rows.append(row)

    write_csv(OUT / "table_rq3_1_background_sensitivity.csv", rows)
    best_auc = {
        label: max(r["AUROC"] for r in rows if r["Dataset"] == label)
        for _, label, _ in datasets
    }
    md = [
        "| Dataset | Background | n | Inconsistency rate | $\\Delta$ median | $\\Delta$ mean | $\\Delta$ max | $\\Delta$ min | AUROC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        is_best = row["AUROC"] == best_auc[row["Dataset"]]
        background = bold_md(row["Background"], row["Background"] == "Overall")
        md.append(
            f"| {row['Dataset']} | {background} | {row['n']} | {row['Inconsistency_rate']:.4f} | "
            f"{row['Delta_median']:.4f} | {row['Delta_mean']:.4f} | {row['Delta_max']:.4f} | "
            f"{row['Delta_min']:.4f} | {bold_md(f'{row['AUROC']:.4f}', is_best)} |"
        )
    (OUT / "table_rq3_1_background_sensitivity.md").write_text("\n".join(md) + "\n")

    tex = [
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Dataset & Background & n & Inconsistency & \multicolumn{4}{c}{Basic Statistic Variations} & Discrimination \\",
        r"\cmidrule(lr){5-8}\cmidrule(lr){9-9}",
        r"& & & Rate & $\Delta$ Median & $\Delta$ Mean & $\Delta$ Max & $\Delta$ Min & AUROC \\",
        r"\midrule",
    ]
    previous = None
    for row in rows:
        if previous is not None and row["Dataset"] != previous:
            tex.append(r"\midrule")
        dataset_cell = rf"\multirow{{4}}{{*}}{{{row['Dataset']}}}" if row["Dataset"] != previous else ""
        previous = row["Dataset"]
        is_best = row["AUROC"] == best_auc[row["Dataset"]]
        background = bold_tex(row["Background"], row["Background"] == "Overall")
        auc_text = f"{row['AUROC']:.4f}"
        tex.append(
            f"{dataset_cell} & {background} & {row['n']} & {row['Inconsistency_rate']:.4f} & "
            f"{row['Delta_median']:.4f} & {row['Delta_mean']:.4f} & {row['Delta_max']:.4f} & "
            f"{row['Delta_min']:.4f} & {bold_tex(auc_text, is_best)} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_rq3_1_background_sensitivity.tex").write_text("\n".join(tex) + "\n")


def make_fusion_table(datasets):
    rows = []
    for dataset, label, data_rows in datasets:
        y = np.asarray([r["inconsistent"] for r in data_rows], dtype=int)
        dataset_rows = []
        for key, method_label in FUSIONS:
            score = score_rows(data_rows, key)
            row = {
                "Dataset": label,
                "Fusion_method": method_label,
                **delta_stats(y, score),
                "AUROC": roc_auc_score(y, score),
            }
            dataset_rows.append(row)
        ranked = sorted(dataset_rows, key=lambda r: r["AUROC"], reverse=True)
        ranks = {id(row): rank for rank, row in enumerate(ranked, start=1)}
        for row in dataset_rows:
            row["Rank"] = ranks[id(row)]
        rows.extend(dataset_rows)

    write_csv(OUT / "table_rq3_2_fusion_ablation.csv", rows)
    md = [
        "| Dataset | Fusion method | $\\Delta$ median | $\\Delta$ mean | $\\Delta$ max | $\\Delta$ min | AUROC | Rank |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        best = row["Rank"] == 1
        method = bold_md(row["Fusion_method"], best)
        md.append(
            f"| {row['Dataset']} | {method} | {row['Delta_median']:.4f} | {row['Delta_mean']:.4f} | "
            f"{row['Delta_max']:.4f} | {row['Delta_min']:.4f} | {bold_md(f'{row['AUROC']:.4f}', best)} | "
            f"{bold_md(str(row['Rank']), best)} |"
        )
    (OUT / "table_rq3_2_fusion_ablation.md").write_text("\n".join(md) + "\n")

    tex = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Dataset & Fusion Method & \multicolumn{4}{c}{Basic Statistic Variations} & \multicolumn{2}{c}{Discrimination Performance} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}",
        r"& & $\Delta$ Median & $\Delta$ Mean & $\Delta$ Max & $\Delta$ Min & AUROC & Rank \\",
        r"\midrule",
    ]
    previous = None
    for row in rows:
        if previous is not None and row["Dataset"] != previous:
            tex.append(r"\midrule")
        dataset_rows = [r for r in rows if r["Dataset"] == row["Dataset"]]
        dataset_cell = rf"\multirow{{{len(dataset_rows)}}}{{*}}{{{row['Dataset']}}}" if row["Dataset"] != previous else ""
        previous = row["Dataset"]
        best = row["Rank"] == 1
        auc_text = f"{row['AUROC']:.4f}"
        tex.append(
            f"{dataset_cell} & {bold_tex(row['Fusion_method'], best)} & {row['Delta_median']:.4f} & "
            f"{row['Delta_mean']:.4f} & {row['Delta_max']:.4f} & {row['Delta_min']:.4f} & "
            f"{bold_tex(auc_text, best)} & {bold_tex(str(row['Rank']), best)} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_rq3_2_fusion_ablation.tex").write_text("\n".join(tex) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    datasets = [(run, label, load_rows(run)) for run, label in DATASETS]
    make_background_table(datasets)
    make_fusion_table(datasets)
    for path in sorted(OUT.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
