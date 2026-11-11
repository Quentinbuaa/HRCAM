"""Compute Table-3-style DenseNet comparison from saved maps, CPU only."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8388608), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


METHODS = ["JSD", "NCD", "CD", "KLD", "SNO", "BAT", "SpatialW1", "FAS", "BAI", "H-RCAM"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    selected = set(os.sched_getaffinity(0)) & set(range(24, 32))
    if not selected:
        raise RuntimeError("Approved CPUs unavailable")
    os.sched_setaffinity(0, selected)
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "4"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(args.root))
    import numpy as np
    import scipy
    import sklearn
    from sklearn.metrics import roc_auc_score, average_precision_score
    from metrics import heatmap_metrics
    from transport_metrics import distances
    import ot
    output = args.run / "full_baselines_v1"
    output.mkdir(exist_ok=False)
    started = time.time()
    write_json(output / "status.json", {"state": "running", "started": started})
    comparisons = []
    for name, label in (("pets_v1", "Oxford-IIIT Pet"), ("cub_v1", "CUB-200-2011")):
        folder = args.run / name
        inputs = {file: folder / file for file in ("holdout_records.jsonl", "holdout_sources.npz", "holdout_heatmaps.npy")}
        before = {key: sha256(path) for key, path in inputs.items()}
        records = [json.loads(line) for line in inputs["holdout_records.jsonl"].read_text().splitlines()]
        rows = sorted([r for r in records if r["cam_mode"] == "own" and r["valid_cam"] and r["mask_eligible"]],
                      key=lambda r: (r["source_id"], r["background"]))
        maps = np.load(inputs["holdout_heatmaps.npy"], mmap_mode="r")
        with np.load(inputs["holdout_sources.npz"]) as archive:
            source_ids, masks = archive["source_ids"], archive["masks"]
        positions = {int(source): pos for pos, source in enumerate(source_ids)}
        score_rows = []
        for index, row in enumerate(rows):
            pos = positions[row["source_id"]]
            bg = ("black", "gray", "white").index(row["background"])
            first, second, mask = maps[pos, 0], maps[pos, 1 + bg * 2], masks[pos]
            basic = heatmap_metrics(first, second, mask)
            assert basic["valid_cam"]
            np.testing.assert_allclose(basic["jsd"], row["jsd"], atol=3e-14, rtol=0)
            np.testing.assert_allclose(basic["d4"] / 2, row["ncd"], atol=3e-14, rtol=0)
            trans = distances(first, second, mask, grid=14)
            j, c = row["jsd"], row["ncd"]
            score_rows.append({"source_id": row["source_id"], "background": row["background"],
                "inconsistent": row["inconsistent"], "JSD": j, "NCD": c,
                "CD": basic["d1"], "KLD": basic["d2"], "SNO": basic["d3"],
                "BAT": trans["BAT"], "SpatialW1": trans["spatial_w1"],
                "FAS": basic["foreground_shift"], "BAI": basic["background_increase"],
                "H-RCAM": 2 * j * c / (j + c + 1e-12)})
            if index % 300 == 0:
                print(name + " saved-map distances " + str(index) + "/" + str(len(rows)), flush=True)
        (output / (name + "_scores.jsonl")).write_text("".join(json.dumps(r) + "\n" for r in score_rows))
        y = np.array([r["inconsistent"] for r in score_rows], dtype=int)
        scores = {method: np.array([r[method] for r in score_rows]) for method in METHODS}
        ranking = {method: {"auroc": float(roc_auc_score(y, values)), "ap": float(average_precision_score(y, values))}
                   for method, values in scores.items()}
        unique = np.unique([r["source_id"] for r in score_rows])
        source = np.array([r["source_id"] for r in score_rows])
        groups = [np.flatnonzero(source == value) for value in unique]
        rng = np.random.default_rng(20261104)
        samples = {method: {"auroc": [], "ap": []} for method in METHODS}
        for repeat in range(1000):
            ids = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            assert len(np.unique(y[ids])) == 2
            for method, values in scores.items():
                samples[method]["auroc"].append(float(roc_auc_score(y[ids], values[ids])))
                samples[method]["ap"].append(float(average_precision_score(y[ids], values[ids])))
        differences = {}
        for method in METHODS:
            for metric in ("auroc", "ap"):
                ranking[method][metric + "_ci95"] = np.quantile(samples[method][metric], [.025, .975]).tolist()
            if method != "H-RCAM":
                differences[method] = {metric: {"difference": ranking["H-RCAM"][metric] - ranking[method][metric],
                    "ci95": np.quantile(np.array(samples["H-RCAM"][metric]) - samples[method][metric], [.025, .975]).tolist()}
                    for metric in ("auroc", "ap")}
        previous = json.loads((folder / "summary.json").read_text())["own"]["ranking"]
        for method in ("JSD", "NCD", "H-RCAM"):
            old = previous["HRCAM" if method == "H-RCAM" else method]
            for metric in ("auroc", "ap"):
                np.testing.assert_allclose(ranking[method][metric], old[metric], atol=3e-14, rtol=0)
                np.testing.assert_allclose(ranking[method][metric + "_ci95"], old[metric + "_ci95"], atol=3e-13, rtol=0)
        # Reproduce the original Table 3 raw-distance ranks to distinguish
        # affine rescaling from calibration clipping in its legacy generator.
        original = [json.loads(line) for line in (args.root / "runs" / name / "holdout_records.jsonl").read_text().splitlines()]
        original = [r for r in original if r["cam_mode"] == "own" and r["mask_eligible"] and r["valid_cam"]]
        config = json.loads((args.root / "runs" / name / "frozen_analysis.json").read_text())["normalization"]["own"]
        legacy_check = {}
        for column, (method, key) in enumerate((("CD", "d1"), ("KLD", "d2"), ("SNO", "d3"))):
            values = np.array([r[key] for r in original])
            target = np.array([r["inconsistent"] for r in original])
            low, high = config["minimum"][column], config["maximum"][column]
            clipped = np.clip((values - low) / (high - low), 0, 1)
            legacy_check[method] = {"raw_auroc": float(roc_auc_score(target, values)),
                "legacy_auroc": float(roc_auc_score(target, clipped)),
                "raw_ap": float(average_precision_score(target, values)),
                "legacy_ap": float(average_precision_score(target, clipped))}
            for metric in ("auroc", "ap"):
                assert round(legacy_check[method]["raw_" + metric], 4) == round(legacy_check[method]["legacy_" + metric], 4)
        after = {key: sha256(path) for key, path in inputs.items()}
        assert before == after
        result = {"dataset": label, "run": name, "valid_pairs": len(y), "valid_sources": len(unique),
                  "inconsistent_pairs": int(y.sum()), "ranking": ranking, "hrcam_minus_baseline": differences,
                  "original_resnet_raw_vs_legacy": legacy_check, "input_sha256": before}
        write_json(output / (name + "_summary.json"), result)
        comparisons.append(result)
        print(name + " all-baseline analysis complete", flush=True)
    write_json(output / "summary.json", comparisons)
    metadata = {"cpu_affinity": sorted(os.sched_getaffinity(0)), "gpu_used": False,
                "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                "sklearn": sklearn.__version__, "POT": ot.__version__, "bootstrap": 1000,
                "seed": 20261104, "normalization": "raw defined candidate distances; no fitted clipping",
                "source_sha256": {file: sha256(args.root / file) for file in ("metrics.py", "transport_metrics.py")},
                "script_sha256": sha256(Path(__file__)),
                "protocol_sha256": sha256(Path(__file__).with_name("FULL_BASELINES_PROTOCOL.md"))}
    write_json(output / "metadata.json", metadata)
    lines = ["# R1C1: DenseNet-121 comparison in the style of Table 3", "",
             "Own-predicted-class targets; exact original Pet/CUB source subsets. Higher AUROC/AP are better.", "",
             "| Measure | Pet AUROC | Pet AP | CUB AUROC | CUB AP |", "|---|---:|---:|---:|---:|"]
    tex = [r"\begin{table}[H]", r"\caption{Discrimination of prediction inconsistency by the baseline heatmap-dissimilarity metrics and H-RCAM with a frozen DenseNet-121 backbone. Higher AUROC and AP are better.}",
           r"\label{tab:rq1-architecture}", r"\centering", r"\fontsize{9}{11}\selectfont", r"\setlength{\tabcolsep}{5pt}",
           r"\renewcommand{\arraystretch}{1.08}", r"\begin{tabular*}{0.8\textwidth}{@{\extracolsep{\fill}}lcccc@{}}",
           r"\toprule", r"& \multicolumn{2}{c}{\textbf{Oxford-IIIT Pet}} & \multicolumn{2}{c}{\textbf{CUB-200-2011}} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r"\textbf{Measure} & \textbf{AUROC} & \textbf{AP} & \textbf{AUROC} & \textbf{AP} \\", r"\midrule"]
    csv_rows = []
    for method in METHODS:
        cells, tex_cells = [], []
        for result in comparisons:
            for metric in ("auroc", "ap"):
                value = result["ranking"][method][metric]
                best = value == max(row[metric] for row in result["ranking"].values())
                text = "{:.4f}".format(value)
                cells.append("**" + text + "**" if best else text)
                tex_cells.append(r"\textbf{" + text + "}" if best else text)
                score = result["ranking"][method]
                csv_rows.append({"dataset": result["dataset"], "measure": method, "metric": metric,
                                 "estimate": value, "ci95_low": score[metric + "_ci95"][0],
                                 "ci95_high": score[metric + "_ci95"][1]})
        lines.append("| " + method + " | " + " | ".join(cells) + " |")
        label = r"\textbf{H-RCAM}" if method == "H-RCAM" else method
        tex.append(label + " & " + " & ".join(tex_cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    lines += ["", "Pet: 2979 pairs from 993 valid sources. CUB: 2889 pairs from 963 valid sources.",
              "Bold marks the largest unrounded point estimate in each column, not a significance test.",
              "The main table mirrors Table 3; all 95% source-cluster intervals are retained in table_rq1_4_with_ci.csv and summary.json.",
              "Raw defined distances were used without fitted clipping. Raw CD/KLD/SNO reproduce the original ResNet Table 3 values to four decimals.",
              "No backbone fitting or new attribution extraction was performed in this saved-map extension.", ""]
    (output / "table_rq1_4_densenet121.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "table_rq1_4_densenet121.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    with (output / "table_rq1_4_with_ci.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    write_json(output / "status.json", {"state": "complete", "started": started, "ended": time.time(), "gpu_used": False})
    print("Completed Table 3 extension: " + str(output), flush=True)


if __name__ == "__main__":
    main()
