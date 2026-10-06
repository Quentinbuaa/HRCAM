"""Freeze calibration/development choices; audit and evaluate held-out records."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from metrics import apply_minmax, fit_minmax

ROOT = Path(__file__).resolve().parent
COMPONENTS = ["d1", "d2", "d3", "d4"]


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def eligible(rows, mode):
    return [r for r in rows if r["cam_mode"] == mode and r["mask_eligible"] and r["valid_cam"]]


def features(rows):
    return np.array([[r["d1"], r["jsd"], r["d3"], r["d4"]/2, r["foreground_shift"]] for r in rows])


def score_rows(rows, params, weights):
    raw = np.asarray([[r[c] for c in COMPONENTS] for r in rows])
    scaled, clipped = apply_minmax(raw, params)
    f = features(rows)
    scores = {c: raw[:, i] for i, c in enumerate(COMPONENTS)}
    scores.update({"RCAM": scaled.mean(1), "BRCAM": f[:, :4].mean(1),
                   "jsd": f[:, 1],
                   "foreground_shift": f[:, 4], "background_increase": np.array([r["background_increase"] for r in rows]),
                   "learned_bounded": f @ weights,
                   "confidence_drop": np.array([r["confidence_drop"] for r in rows]),
                   "prediction_jsd": np.array([r["prediction_jsd"] for r in rows])})
    assert np.isfinite(np.stack(list(scores.values()))).all()
    assert ((scores["RCAM"] >= 0) & (scores["RCAM"] <= 1)).all()
    return scores, clipped


def auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None


def freeze(output):
    path = output / "frozen_analysis.json"
    if path.exists():
        print("Using existing frozen configuration", flush=True)
        return json.loads(path.read_text())
    if (output / "holdout_records.jsonl").exists():
        raise RuntimeError("Holdout already exists; refuse to fit a new configuration")
    calibration = load(output / "calibration_records.jsonl")
    development = load(output / "development_records.jsonl")
    parameters = {}
    for mode in ["own", "fixed"]:
        rows = eligible(calibration, mode)
        parameters[mode] = fit_minmax([[r[c] for c in COMPONENTS] for r in rows])
    dev = eligible(development, "fixed")
    y = np.array([r["inconsistent"] for r in dev], dtype=int)
    f = features(dev)
    rng = np.random.default_rng(20260911)
    candidates = np.concatenate([np.eye(5), np.ones((1, 5))/5, rng.dirichlet(np.ones(5), size=512)])
    values = [auc(y, f @ w) for w in candidates]
    best = int(np.argmax(values))
    weights = candidates[best]
    scores, _ = score_rows(dev, parameters["fixed"], weights)
    candidate_aucs = {name: auc(y, scores[name]) for name in ["BRCAM", "foreground_shift", "learned_bounded"]}
    selected = max(candidate_aucs, key=candidate_aucs.get)
    result = {"normalization": parameters, "learned_weights": weights.tolist(),
              "learned_features": ["d1", "jsd", "d3", "d4_half", "foreground_shift"],
              "selection_target": "label_inconsistency", "selection_mode": "fixed",
              "development_candidate_auroc": candidate_aucs, "selected_candidate": selected,
              "selection_warning": "Development-only optimization; learned score is supervised; holdout is required.",
              "calibration_sha256": hashlib.sha256((output / "calibration_records.jsonl").read_bytes()).hexdigest(),
              "development_sha256": hashlib.sha256((output / "development_records.jsonl").read_bytes()).hexdigest()}
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


def cluster_bootstrap(rows, scores, repeats=400):
    y = np.array([r["inconsistent"] for r in rows], int)
    ids = np.array([r["source_id"] for r in rows])
    unique = np.unique(ids)
    groups = [np.flatnonzero(ids == source) for source in unique]
    rng = np.random.default_rng(20260912)
    estimates = {name: {"auc": [], "ap": []} for name in scores}
    for _ in range(repeats):
        draw = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(y[draw])) < 2:
            continue
        for name, values in scores.items():
            estimates[name]["auc"].append(roc_auc_score(y[draw], values[draw]))
            estimates[name]["ap"].append(average_precision_score(y[draw], values[draw]))
    result = {}
    for name, values in scores.items():
        result[name] = {"auroc": auc(y, values), "average_precision": float(average_precision_score(y, values)),
                        "auroc_ci95": np.quantile(estimates[name]["auc"], [.025, .975]).tolist(),
                        "ap_ci95": np.quantile(estimates[name]["ap"], [.025, .975]).tolist()}
    differences = {}
    for name in ["BRCAM", "learned_bounded", "foreground_shift"]:
        differences[name] = {}
        for reference in ["RCAM", "d4", "jsd"]:
            diffs = np.asarray(estimates[name]["auc"]) - estimates[reference]["auc"]
            differences[name][reference] = {"auroc_difference": result[name]["auroc"] - result[reference]["auroc"],
                                            "ci95": np.quantile(diffs, [.025, .975]).tolist()}
    return result, differences


def plots(rows, scores, output, role, mode):
    y = np.array([r["inconsistent"] for r in rows], int)
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in ["d4", "RCAM", "BRCAM", "foreground_shift", "learned_bounded", "prediction_jsd"]:
        x, t, _ = roc_curve(y, scores[name])
        ax.plot(x, t, label=f"{name} ({roc_auc_score(y, scores[name]):.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"{role}: {mode}-class CAM")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / f"{role}_{mode}_roc.png", dpi=180)
    plt.close(fig)
    names = ["RCAM", "BRCAM", "d4", "foreground_shift"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
    for ax, name in zip(axes, names):
        ax.boxplot([scores[name][y == 0], scores[name][y == 1]], tick_labels=["Consistent", "Changed"], showfliers=False)
        ax.set_title(name)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(f"{role}: {mode}-class CAM; outliers hidden only in visualization")
    fig.tight_layout()
    fig.savefig(output / f"{role}_{mode}_boxplots.png", dpi=180)
    plt.close(fig)


def analyze(output, role, config, boot):
    all_rows = load(output / f"{role}_records.jsonl")
    result = {"role": role, "bootstrap_repeats": boot, "selected_on_development": config["selected_candidate"], "modes": {}}
    for mode in ["own", "fixed"]:
        all_mode = [r for r in all_rows if r["cam_mode"] == mode]
        rows = eligible(all_rows, mode)
        scores, clipped = score_rows(rows, config["normalization"][mode], np.asarray(config["learned_weights"]))
        summary = {"records_total": len(all_mode), "records_mask_eligible": sum(r["mask_eligible"] for r in all_mode),
                   "records_analyzed": len(rows), "unique_sources_analyzed": len(set(r["source_id"] for r in rows)),
                   "invalid_cams_total": sum(not r["valid_cam"] for r in all_mode),
                   "clipping_fraction": clipped.mean(axis=0).tolist(),
                   "positive_prevalence_analyzed": float(np.mean([r["inconsistent"] for r in rows])),
                   "clean_accuracy_all_sources": float(np.mean([r["original_correct"] for r in all_mode])),
                   "transformed_accuracy_all": float(np.mean([r["transformed_correct"] for r in all_mode])),
                   "label_consistency_all": 1-float(np.mean([r["inconsistent"] for r in all_mode]))}
        correct_rows = np.array([r["original_correct"] for r in rows], bool)
        y = np.array([r["inconsistent"] for r in rows], int)
        summary["initially_correct_n"] = int(correct_rows.sum())
        summary["initially_correct_auroc"] = {name: auc(y[correct_rows], s[correct_rows]) for name, s in scores.items()}
        summary["ranking"], summary["paired_auroc_differences"] = cluster_bootstrap(rows, scores, repeats=boot)
        summary["backgrounds"] = {}
        for bg in ["black", "gray", "white"]:
            mask = np.array([r["background"] == bg for r in rows])
            summary["backgrounds"][bg] = {"n": int(mask.sum()), "RCAM_mean": float(scores["RCAM"][mask].mean()),
                                            "consistency": float(1-y[mask].mean())}
        mean_by_groups = sum(scores["RCAM"][y == c].sum() for c in [0, 1]) / len(rows)
        mean_by_bg = sum(g["n"]*g["RCAM_mean"] for g in summary["backgrounds"].values()) / len(rows)
        assert np.isclose(mean_by_groups, mean_by_bg, atol=1e-12)
        summary["weighted_mean_audit_pass"] = True
        result["modes"][mode] = summary
        np.savez_compressed(output / f"{role}_{mode}_scores.npz", source_ids=[r["source_id"] for r in rows],
                            inconsistent=y, original_correct=correct_rows, **scores)
        plots(rows, scores, output, role, mode)
        print(f"{role}/{mode}: n={len(rows)}, AUC=" + str({n: round(v["auroc"], 4) for n, v in summary["ranking"].items()}), flush=True)
    (output / f"{role}_summary.json").write_text(json.dumps(result, indent=2))
    metadata = json.loads((output / "metadata.json").read_text())
    dataset = metadata.get("dataset", "CIFAR-10")
    lines = [f"# {dataset} {role} results", "", "Exploratory independent reproduction; not a publication-ready confirmation.", ""]
    for mode, summary in result["modes"].items():
        lines += [f"## {mode}-class CAM", "", f"Analyzed: {summary['records_analyzed']}/{summary['records_total']} pairs. Positive prevalence: {summary['positive_prevalence_analyzed']:.3f}.", "",
                  "| Score | AUROC (95% cluster CI) | Average precision |", "|---|---:|---:|"]
        for name, r in summary["ranking"].items():
            lines.append(f"| {name} | {r['auroc']:.4f} ({r['auroc_ci95'][0]:.4f}, {r['auroc_ci95'][1]:.4f}) | {r['average_precision']:.4f} |")
        lines += ["", f"Clean accuracy: {summary['clean_accuracy_all_sources']:.4f}; transformed accuracy: {summary['transformed_accuracy_all']:.4f}; label consistency: {summary['label_consistency_all']:.4f}.", ""]
    lines += ["## Limitations", "", "- Review the run metadata and protocol for mask provenance and model-selection exposure.",
              "- One dataset and one classifier checkpoint; results do not establish general superiority.",
              "- Label inconsistency is not synonymous with actual classification failure.",
              "- Learned bounded score is supervised and must not be described as a label-free metric.",
              "- Confidence intervals reflect sampled source images, not training-seed variation."]
    (output / f"{role}_report.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="cifar10_v1")
    parser.add_argument("--stage", choices=["development", "holdout"], required=True)
    parser.add_argument("--bootstrap", type=int, default=400)
    args = parser.parse_args()
    output = ROOT / "runs" / args.run
    if args.stage == "development":
        config = freeze(output)
    else:
        config = json.loads((output / "frozen_analysis.json").read_text())
        for role in ["calibration", "development"]:
            checksum = hashlib.sha256((output / f"{role}_records.jsonl").read_bytes()).hexdigest()
            if checksum != config[f"{role}_sha256"]:
                raise ValueError("Selection records changed after freezing")
    analyze(output, args.stage, config, args.bootstrap)


if __name__ == "__main__":
    main()
