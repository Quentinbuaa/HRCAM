"""Summarize the prespecified H-RCAM score with video-cluster uncertainty."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parent
SCORE_NAMES = ("H-RCAM", "JSD", "NCD")


def summarize(rows, groups, bootstrap, seed):
    valid = [row for row in rows if row["valid_cam"] and row["mask_eligible"]
             and row["jsd"] is not None and row["d4"] is not None]
    label = np.asarray([row["inconsistent"] for row in valid], dtype=int)
    jsd = np.asarray([row["jsd"] for row in valid], dtype=float)
    ncd = np.asarray([row["d4"] / 2 for row in valid], dtype=float)
    scores = {"H-RCAM": 2 * jsd * ncd / (jsd + ncd + 1e-12),
              "JSD": jsd, "NCD": ncd}
    video = np.asarray([groups[row["source_id"]] for row in valid])
    video_rows = defaultdict(list)
    for index, key in enumerate(video):
        video_rows[key].append(index)
    videos = list(video_rows)
    rng = np.random.default_rng(seed)
    bootstrap_ids = []
    for _ in range(bootstrap):
        sampled = rng.choice(videos, size=len(videos), replace=True)
        indices = np.fromiter((i for key in sampled for i in video_rows[key]), dtype=int)
        if len(np.unique(label[indices])) == 2:
            bootstrap_ids.append(indices)
    result = {
        "n_pairs_all": len(rows), "n_pairs_valid": len(valid),
        "n_videos": len(videos), "inconsistency_rate_all": float(np.mean([r["inconsistent"] for r in rows])),
        "inconsistency_rate_valid": float(label.mean()),
        "clean_accuracy": float(np.mean([r["original_correct"] for r in rows])),
        "transformed_accuracy": float(np.mean([r["transformed_correct"] for r in rows])),
        "metrics": {}, "paired_auroc_differences": {}, "by_background": {},
    }
    for name in SCORE_NAMES:
        score = scores[name]
        if len(np.unique(label)) < 2:
            result["metrics"][name] = None
            continue
        auc = float(roc_auc_score(label, score))
        ap = float(average_precision_score(label, score))
        auc_samples = [roc_auc_score(label[ids], score[ids]) for ids in bootstrap_ids]
        result["metrics"][name] = {
            "auroc": auc, "average_precision": ap,
            "auroc_video_cluster_ci95": np.quantile(auc_samples, [.025, .975]).tolist(),
        }
    if len(np.unique(label)) == 2:
        for comparator in ("JSD", "NCD"):
            differences = [roc_auc_score(label[ids], scores["H-RCAM"][ids])
                           - roc_auc_score(label[ids], scores[comparator][ids])
                           for ids in bootstrap_ids]
            result["paired_auroc_differences"][f"H-RCAM minus {comparator}"] = {
                "point": result["metrics"]["H-RCAM"]["auroc"]
                         - result["metrics"][comparator]["auroc"],
                "video_cluster_ci95": np.quantile(differences, [.025, .975]).tolist(),
            }
    for background in ("black", "gray", "white"):
        ids = np.flatnonzero([row["background"] == background for row in valid])
        y = label[ids]
        result["by_background"][background] = {
            "n_pairs": len(ids), "inconsistency_rate": float(y.mean()),
            "hrcam_auroc": float(roc_auc_score(y, scores["H-RCAM"][ids]))
            if len(np.unique(y)) == 2 else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="trashcan_material_3class_v1")
    parser.add_argument("--data", default="trashcan_material_3class.npz")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    with np.load(ROOT / "data" / args.data) as source:
        groups = source["groups"]
    output = ROOT / "runs" / args.run
    report = {}
    for role in ("development", "holdout"):
        path = output / f"{role}_records.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        report[role] = {}
        for mode in ("own", "fixed"):
            subset = [row for row in rows if row["cam_mode"] == mode]
            report[role][mode] = summarize(subset, groups, args.bootstrap,
                                           20260923 + (role == "holdout") * 10 + (mode == "fixed"))
    destination = output / "trashcan_external_summary.json"
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
