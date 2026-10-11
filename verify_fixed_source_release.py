"""Verify published RQ3.3 derived scores without images, heatmaps, or a GPU.

Requires NumPy and scikit-learn. Run from any directory with:
    python verify_fixed_source_release.py
This verifies score-level results, not end-to-end Grad-CAM extraction.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results/fixed_source_sensitivity_20261011_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((ROOT / "fixed_source_sensitivity.py").read_bytes()).hexdigest()
    require(digest == summary["script_sha256"], "Original analysis script hash mismatch")
    print("Dataset | matched pairs | own AUROC / AP | fixed AUROC / AP")
    for dataset in summary["datasets"]:
        run = dataset["source_run"]
        with np.load(RESULTS / (run + "_matched_scores.npz"), allow_pickle=False) as data:
            y = data["inconsistent"]
            ids = data["source_ids"]
            backgrounds = data["backgrounds"]
            coverage = dataset["coverage"]
            require(set(np.unique(y).tolist()) == {0, 1}, run + ": invalid outcome labels")
            require(len(y) == coverage["matched_pairs"], run + ": pair count mismatch")
            require(len(np.unique(ids)) == coverage["matched_sources"], run + ": source count mismatch")
            require(int(y.sum()) == coverage["matched_inconsistent"], run + ": inconsistent count mismatch")
            keys = list(zip(ids.tolist(), backgrounds.tolist()))
            require(len(set(keys)) == len(keys), run + ": duplicate source/background pairs")
            require(set(backgrounds.tolist()) == {"black", "gray", "white"}, run + ": background mismatch")
            for mode in ("own", "fixed"):
                jsd, ncd = data[mode + "_JSD"], data[mode + "_NCD"]
                expected_score = 2 * jsd * ncd / (jsd + ncd + 1e-12)
                require(np.allclose(data[mode + "_HRCAM"], expected_score, rtol=0, atol=1e-12),
                        run + ": harmonic score mismatch")
                for component in ("JSD", "NCD", "HRCAM"):
                    name = mode + "_" + component
                    values = data[name]
                    require(values.shape == y.shape and np.isfinite(values).all(), run + ": invalid scores")
                    observed = {"auroc": float(roc_auc_score(y, values)),
                                "ap": float(average_precision_score(y, values))}
                    for metric, value in observed.items():
                        require(abs(value - dataset["ranking"][name][metric]) < 1e-12,
                                run + ": " + name + " " + metric + " mismatch")
            require(np.array_equal(data["own_HRCAM"][y == 0], data["fixed_HRCAM"][y == 0]),
                    run + ": consistent-pair scores differ")
            own = dataset["ranking"]["own_HRCAM"]
            fixed = dataset["ranking"]["fixed_HRCAM"]
            print(f"{dataset['dataset']} | {len(y)} | {own['auroc']:.4f} / {own['ap']:.4f} | "
                  f"{fixed['auroc']:.4f} / {fixed['ap']:.4f}")
    print("PASS: script hash, coverage, harmonic scores, all component AUROC/AP values, "
          "and unchanged prediction-consistent scores.")


if __name__ == "__main__":
    main()
