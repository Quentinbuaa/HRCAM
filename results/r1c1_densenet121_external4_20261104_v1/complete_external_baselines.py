"""Compute Table-4-style DenseNet comparison on all four external datasets, CPU only."""
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
    for name, label in (("flowers102_external_v1", "Oxford Flowers-102"), ("caltech101_external_v1", "Caltech-101"), ("voc2012_external_v1", "PASCAL VOC 2012"), ("uecfoodpix_external_v1", "UECFoodPixComplete")):
        folder = args.run / name
        inputs = {file: folder / file for file in ("external_records.jsonl", "external_sources.npz", "external_heatmaps.npy")}
        before = {key: sha256(path) for key, path in inputs.items()}
        records = [json.loads(line) for line in inputs["external_records.jsonl"].read_text().splitlines()]
        rows = sorted([r for r in records if r["cam_mode"] == "own" and r["valid_cam"] and r["mask_eligible"]],
                      key=lambda r: (r["source_id"], r["background"]))
        maps = np.load(inputs["external_heatmaps.npy"], mmap_mode="r")
        with np.load(inputs["external_sources.npz"]) as archive:
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
        legacy_check = {"note": "External Table 4 uses raw distances; no legacy calibration transfer."}
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
    lines=["# DenseNet-121: Table 4-style independent-dataset comparison", "", "Own-class targets; all four prespecified independent datasets. Higher AUROC/AP are better.", "", "| Measure | Flowers AUROC | Flowers AP | Caltech AUROC | Caltech AP | VOC AUROC | VOC AP | Food AUROC | Food AP |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    tex=[r"\begin{table}[!htbp]", r"\caption{Discrimination of prediction inconsistency on four independent datasets with a frozen DenseNet-121 backbone. Higher AUROC and AP are better.}", r"\label{tab:rq1-independent-architecture}", r"\centering", r"\setlength{\tabcolsep}{3.5pt}", r"\renewcommand{\arraystretch}{1.08}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lcccccccc}", r"\toprule", r"& \multicolumn{2}{c}{\textbf{Oxford Flowers-102}} & \multicolumn{2}{c}{\textbf{Caltech-101}} & \multicolumn{2}{c}{\textbf{VOC 2012}} & \multicolumn{2}{c}{\textbf{UECFoodPixComplete}} \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}", r"\textbf{Measure} & \textbf{AUROC} & \textbf{AP} & \textbf{AUROC} & \textbf{AP} & \textbf{AUROC} & \textbf{AP} & \textbf{AUROC} & \textbf{AP} \\",r"\midrule"]
    csv_rows=[]
    for method in METHODS:
        cells=[];tex_cells=[]
        for result in comparisons:
            for metric in ("auroc","ap"):
                value=result["ranking"][method][metric]
                best=value==max(r[metric] for r in result["ranking"].values())
                text="{:.4f}".format(value)
                cells.append("**"+text+"**" if best else text)
                tex_cells.append(r"\textbf{"+text+"}" if best else text)
                score=result["ranking"][method]
                csv_rows.append(dict(dataset=result["dataset"],measure=method,metric=metric,estimate=value,ci95_low=score[metric+"_ci95"][0],ci95_high=score[metric+"_ci95"][1]))
        lines.append("| "+method+" | "+" | ".join(cells)+" |")
        tex.append((r"\textbf{H-RCAM}" if method=="H-RCAM" else method)+" & "+" & ".join(tex_cells)+r" \\")
    tex += [r"\bottomrule",r"\end{tabular}",r"}",r"\end{table}"]
    lines += ["", "Bold marks observed column maxima, not statistical significance. All datasets are retained regardless of ranking.", "Raw distances, no metric recalibration; 1000 paired source-cluster bootstrap samples with seed 20261104. Full intervals and paired differences are in summary.json and table4_with_ci.csv.", ""]
    for result in comparisons:lines.append(result['dataset']+": "+str(result['valid_pairs'])+" valid pairs from "+str(result['valid_sources'])+" sources.")
    (output/'table4_densenet121.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (output/'table4_densenet121.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')
    with (output/'table4_with_ci.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    write_json(output / "status.json", {"state": "complete", "started": started, "ended": time.time(), "gpu_used": False})
    print("Completed Table 3 extension: " + str(output), flush=True)


if __name__ == "__main__":
    main()
