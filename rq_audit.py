"""Focused post hoc RQ1/RQ2 audit of saved records; no fitting or inference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from metrics import apply_minmax


NAMES = ["d1", "d2", "d3", "d4", "RCAM"]
BACKGROUNDS = ["black", "gray", "white"]


def describe(values):
    return {"n": len(values), "mean": float(np.mean(values)),
            "q10_q25_q50_q75_q90": np.quantile(values, [.1, .25, .5, .75, .9]).tolist()}


def audit(folder, repeats):
    raw_path = folder / "holdout_records.jsonl"
    rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line]
    config = json.loads((folder / "frozen_analysis.json").read_text())
    result = {"scope": "Post hoc audit of existing Pet holdout; no new model or metric selection",
              "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
              "bootstrap_repeats": repeats, "modes": {}}
    mode_rows = {}
    for mode in ["own", "fixed"]:
        selected = [r for r in rows if r["cam_mode"] == mode and r["mask_eligible"] and r["valid_cam"]]
        mode_rows[mode] = selected
        data = np.load(folder / f"holdout_{mode}_scores.npz")
        ids = np.array([r["source_id"] for r in selected])
        y = np.array([r["inconsistent"] for r in selected], int)
        assert np.array_equal(ids, data["source_ids"])
        assert np.array_equal(y, data["inconsistent"])
        components = np.array([[r[n] for n in NAMES[:4]] for r in selected])
        scaled, _ = apply_minmax(components, config["normalization"][mode])
        np.testing.assert_allclose(scaled.mean(1), data["RCAM"], atol=1e-12)
        scores = {n: data[n] for n in NAMES}
        assert all(np.isfinite(s).all() for s in scores.values())
        unique = np.unique(ids)
        groups = [np.flatnonzero(ids == source) for source in unique]
        assert all(len(g) == 3 for g in groups)
        matrix = np.array([[scores["RCAM"][next(i for i in g if selected[i]["background"] == b)]
                            for b in BACKGROUNDS] for g in groups])
        rng = np.random.default_rng(20260913)
        auc_boot = {n: [] for n in ["RCAM", "d4"]}
        background_boot = []
        for _ in range(repeats):
            sample = rng.integers(0, len(groups), len(groups))
            background_boot.append(matrix[sample].mean(0))
            draw = np.concatenate([groups[i] for i in sample])
            if len(np.unique(y[draw])) == 2:
                for n in auc_boot:
                    auc_boot[n].append(roc_auc_score(y[draw], scores[n][draw]))
        delta = np.array(auc_boot["d4"]) - np.array(auc_boot["RCAM"])
        summary = {"sources": len(unique), "pairs": len(selected),
                   "auroc": {n: float(roc_auc_score(y, s)) for n, s in scores.items()},
                   "D4_minus_RCAM_auroc": {
                       "estimate": float(roc_auc_score(y, scores["d4"]) - roc_auc_score(y, scores["RCAM"])),
                       "paired_cluster_ci95": np.quantile(delta, [.025, .975]).tolist()},
                   "label_groups": {}, "backgrounds": {}, "background_mean_differences": {}}
        for label, mask in [
            ("consistent", y == 0), ("inconsistent", y == 1),
            ("consistent_and_original_correct", (y == 0) & np.array([r["original_correct"] for r in selected])),
        ]:
            summary["label_groups"][label] = {n: describe(s[mask]) for n, s in scores.items()}
        consistent = y == 0
        summary["consistent_spearman_names"] = NAMES
        summary["consistent_spearman"] = spearmanr(np.stack([scores[n][consistent] for n in NAMES], axis=1)).statistic.tolist()
        for j, bg in enumerate(BACKGROUNDS):
            mask = np.array([r["background"] == bg for r in selected])
            summary["backgrounds"][bg] = {
                "RCAM": describe(matrix[:, j]), "label_consistency": float(1-y[mask].mean()),
                "RCAM_consistent": describe(scores["RCAM"][mask & consistent])}
        background_boot = np.asarray(background_boot)
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            summary["background_mean_differences"][f"{BACKGROUNDS[a]}_minus_{BACKGROUNDS[b]}"] = {
                "estimate": float(np.mean(matrix[:, a]-matrix[:, b])),
                "paired_source_ci95": np.quantile(background_boot[:, a]-background_boot[:, b], [.025, .975]).tolist()}
        result["modes"][mode] = summary
    fixed = {(r["source_id"], r["background"]): r for r in mode_rows["fixed"]}
    differences = [abs(r[n]-fixed[(r["source_id"], r["background"])][n])
                   for r in mode_rows["own"] if not r["inconsistent"] for n in NAMES[:4]]
    result["consistent_own_fixed_raw_components_max_abs_difference"] = max(differences)
    result["interpretation"] = [
        "AUC evaluates association with label changes, not all properties of a robustness diagnostic.",
        "These paired intervals address the observed comparison; they do not make a post hoc analysis prespecified.",
        "Continuous score variation within consistent cases does not by itself validate hidden vulnerability.",
        "Own/fixed raw CAM components should agree when labels agree; separately calibrated RCAM scales can differ.",
        "Background colors are separate interventions, not an ordered perturbation-strength scale.",
        "Background intervals are pointwise descriptive intervals, not multiplicity-adjusted significance tests.",
        "Single model and supplemental dataset; cannot replace original CIFAR/TinyImageNet results without revising scope.",
    ]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "rq_audit.json"
    report = args.output_dir / "rq_audit.md"
    if path.exists() or report.exists():
        raise FileExistsError("Choose a new output directory; preserve completed audits")
    result = audit(args.run_dir, args.bootstrap)
    path.write_text(json.dumps(result, indent=2))
    lines = ["# Focused RQ1/RQ2 audit", "", result["scope"], "",
             "Source images are bootstrap clusters; all three backgrounds stay together.", ""]
    for mode, s in result["modes"].items():
        d = s["D4_minus_RCAM_auroc"]
        lo, hi = d["paired_cluster_ci95"]
        lines += [f"## {mode}-class CAM", "",
                  f"{s['sources']} sources, {s['pairs']} pairs.", "",
                  "| Metric | AUROC for label inconsistency |", "|---|---:|"]
        lines += [f"| {n} | {v:.4f} |" for n, v in s["auroc"].items()]
        lines += ["", f"D4 minus RCAM AUROC: {d['estimate']:.4f}; paired 95% CI [{lo:.4f}, {hi:.4f}].", "",
                  "| Label group | Pairs | RCAM Q10 | Median | Q90 |", "|---|---:|---:|---:|---:|"]
        for label, values in s["label_groups"].items():
            v = values["RCAM"]
            q = v["q10_q25_q50_q75_q90"]
            lines.append(f"| {label} | {v['n']} | {q[0]:.4f} | {q[2]:.4f} | {q[4]:.4f} |")
        lines += ["", "| Background | Mean RCAM | Label consistency |", "|---|---:|---:|"]
        for bg, v in s["backgrounds"].items():
            lines.append(f"| {bg} | {v['RCAM']['mean']:.4f} | {v['label_consistency']:.4f} |")
        lines.append("")
        for pair, v in s["background_mean_differences"].items():
            lo, hi = v["paired_source_ci95"]
            lines.append(f"- {pair}: {v['estimate']:.4f}, pointwise paired 95% CI [{lo:.4f}, {hi:.4f}].")
        lines.append("")
    lines += ["## Interpretation", ""] + [f"- {v}" for v in result["interpretation"]]
    lines += ["", "Consistent-case maximum own/fixed raw-component difference: "
              f"{result['consistent_own_fixed_raw_components_max_abs_difference']:.8g}."]
    report.write_text("\n".join(lines)+"\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
