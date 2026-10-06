"""Source-grouped development CV for predefined JSD-D4 fusion candidates."""
import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from metrics import apply_minmax


ROOT = Path(__file__).resolve().parent
EXTRA_NAMES = ["foreground_shift", "background_increase", "spatial_w1", "BAT"]


def source_folds(rows):
    ids = np.array([r["source_id"] for r in rows])
    y = np.array([r["inconsistent"] for r in rows], int)
    unique = np.unique(ids)
    group_rows = [np.flatnonzero(ids == sid) for sid in unique]
    if not all(len(g) == 3 for g in group_rows):
        raise ValueError("Expected three eligible background pairs per source")
    strata = np.array([y[g].sum() for g in group_rows])
    folds = np.full(len(rows), -1, int)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260917)
    for fold, (_, valid_sources) in enumerate(splitter.split(unique, strata)):
        folds[np.concatenate([group_rows[i] for i in valid_sources])] = fold
    assert (folds >= 0).all()
    return folds


def fitted_scores(x, y, folds, kind):
    predictions = np.full(len(y), np.nan)
    metadata = []
    for fold in range(5):
        train, valid = folds != fold, folds == fold
        scaler = StandardScaler().fit(x[train])
        if kind == "LR":
            model = LogisticRegression(C=1., max_iter=1000)
        else:
            model = MLPClassifier(hidden_layer_sizes=(8,), activation="tanh", solver="lbfgs",
                                  alpha=1., max_iter=1000, max_fun=50000, random_state=20260917+fold)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(scaler.transform(x[train]), y[train])
        predictions[valid] = model.predict_proba(scaler.transform(x[valid]))[:, 1]
        info = {"fold": fold, "n_training_pairs": int(train.sum()), "n_validation_pairs": int(valid.sum()),
                "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
                "warnings": [str(w.message) for w in caught],
                "zero_distance_prediction": float(model.predict_proba(scaler.transform(np.zeros((1, x.shape[1]))))[0, 1])}
        if kind == "LR":
            info.update(coefficients=model.coef_[0].tolist(), intercept=float(model.intercept_[0]))
        else:
            info.update(iterations=int(model.n_iter_), weights=[v.tolist() for v in model.coefs_],
                        intercepts=[v.tolist() for v in model.intercepts_])
        metadata.append(info)
    assert np.isfinite(predictions).all()
    return predictions, metadata


def evaluate(rows, scores, folds):
    ids = np.array([r["source_id"] for r in rows])
    y = np.array([r["inconsistent"] for r in rows], int)
    groups = [np.flatnonzero(ids == sid) for sid in np.unique(ids)]
    ranking = {}
    for name, s in scores.items():
        fold_aucs = [float(roc_auc_score(y[folds == f], s[folds == f])) for f in range(5)]
        ranking[name] = {"pooled_auroc": float(roc_auc_score(y, s)),
                         "pooled_AP": float(average_precision_score(y, s)),
                         "fold_auroc": fold_aucs, "mean_fold_auroc": float(np.mean(fold_aucs))}
    rng = np.random.default_rng(20260918)
    boot = {n: [] for n in scores}
    for _ in range(400):
        draw = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(y[draw])) < 2:
            continue
        for n, s in scores.items():
            boot[n].append(roc_auc_score(y[draw], s[draw]))
    differences = {}
    comparisons = [(n, ref) for n in scores if n not in ["JSD", "D4", "RCAM", "BAT"] for ref in ["JSD", "D4"]]
    comparisons += [("LR_extended", "LR_two"), ("MLP_extended", "MLP_two")]
    for name, ref in comparisons:
        delta = np.array(boot[name])-np.array(boot[ref])
        differences[f"{name}_minus_{ref}"] = {
            "estimate": ranking[name]["pooled_auroc"]-ranking[ref]["pooled_auroc"],
            "conditional_ci95": np.quantile(delta, [.025, .975]).tolist(),
            "fold_differences": (np.array(ranking[name]["fold_auroc"])-ranking[ref]["fold_auroc"]).tolist()}
    return {"ranking": ranking, "paired_differences": differences}


def run(folder, transport_folder, output):
    raw_path = folder / "development_records.jsonl"
    raw = [json.loads(line) for line in raw_path.read_text().splitlines() if line]
    boundary_path = transport_folder / "candidate_records.jsonl"
    boundary_rows = [json.loads(line) for line in boundary_path.read_text().splitlines() if line]
    key = lambda r: (r["source_id"], r["background"], r["cam_mode"])
    boundary = {key(r): r for r in boundary_rows}
    assert len(boundary) == len(boundary_rows)
    config = json.loads((folder / "frozen_analysis.json").read_text())
    result = {"scope": "Exploratory source-grouped development CV; not fresh confirmation",
              "target": "prediction-label inconsistency", "bootstrap_repeats": 400,
              "extra_features": EXTRA_NAMES, "modes": {},
              "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [raw_path, boundary_path]},
              "source_sha256": {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
                                for n in ["fusion_development.py", "FUSION_PROTOCOL.md"]}}
    shared_source_folds = None
    for mode in ["own", "fixed"]:
        rows = sorted([r for r in raw if r["cam_mode"] == mode and r["valid_cam"] and r["mask_eligible"]],
                      key=lambda r: (r["source_id"], r["background"]))
        ids = np.array([r["source_id"] for r in rows])
        y = np.array([r["inconsistent"] for r in rows], int)
        folds = source_folds(rows)
        source_assignment = {int(sid): int(fold) for sid, fold in zip(ids, folds)}
        if shared_source_folds is None:
            shared_source_folds = source_assignment
        assert source_assignment == shared_source_folds
        for f in range(5):
            assert not set(ids[folds == f]) & set(ids[folds != f])
        j = np.array([r["jsd"] for r in rows])
        c = np.array([r["d4"]/2 for r in rows])
        extra = np.array([[r["foreground_shift"], r["background_increase"],
                           boundary[key(r)]["spatial_w1"], boundary[key(r)]["BAT"]] for r in rows])
        x = np.column_stack([j, c])
        extended = np.column_stack([x, extra])
        assert np.isfinite(extended).all() and extended.min() >= 0 and extended.max() <= 1+1e-9
        original = np.array([[r[n] for n in ["d1", "d2", "d3", "d4"]] for r in rows])
        norm, _ = apply_minmax(original, config["normalization"][mode])
        scores = {"JSD": j, "D4": c, "RCAM": norm.mean(1), "BAT": extra[:, -1],
                  "equal_mean": (j+c)/2, "nonlinear_union": 1-(1-j)*(1-c),
                  "sqrt_mean": (np.sqrt(j)+np.sqrt(c))/2,
                  "boundary_mean": (j+c+extra[:, -1])/3}
        scores["convex_linear"] = np.full(len(y), np.nan)
        learned = {"convex_linear": []}
        for fold in range(5):
            train, valid = folds != fold, folds == fold
            weights = np.linspace(0, 1, 11)
            aucs = [roc_auc_score(y[train], w*j[train]+(1-w)*c[train]) for w in weights]
            w = float(weights[int(np.argmax(aucs))])
            scores["convex_linear"][valid] = w*j[valid]+(1-w)*c[valid]
            learned["convex_linear"].append({"fold": fold, "JSD_weight": w})
        for kind in ["LR", "MLP"]:
            for feature_name, features in [("two", x), ("extended", extended)]:
                name = f"{kind}_{feature_name}"
                print(mode, "fitting", name, flush=True)
                scores[name], learned[name] = fitted_scores(features, y, folds, kind)
        summary = evaluate(rows, scores, folds)
        summary.update(n_pairs=len(rows), n_sources=len(source_assignment),
                       label_inconsistent_pairs=int(y.sum()),
                       JSD_D4_spearman=float(spearmanr(j, c).statistic), learned=learned)
        result["modes"][mode] = summary
        np.savez_compressed(output / f"{mode}_oof_scores.npz", source_ids=ids,
                            backgrounds=[r["background"] for r in rows], inconsistent=y, fold=folds, **scores)
        print(mode, {n: round(s["pooled_auroc"], 4) for n, s in summary["ranking"].items()}, flush=True)
    (output / "source_folds.json").write_text(json.dumps(shared_source_folds, indent=2))
    (output / "summary.json").write_text(json.dumps(result, indent=2))
    lines = ["# JSD-D4 fusion development results", "", result["scope"], "",
             "The learned target is observed label change. All three backgrounds from one source stay in one fold.", ""]
    for mode, s in result["modes"].items():
        lines += [f"## {mode}-class CAM", "", f"JSD-D4 Spearman correlation: {s['JSD_D4_spearman']:.4f}.", "",
                  "| Candidate | Pooled OOF AUROC | Mean fold AUROC | AP |", "|---|---:|---:|---:|"]
        for n, v in s["ranking"].items():
            lines.append(f"| {n} | {v['pooled_auroc']:.4f} | {v['mean_fold_auroc']:.4f} | {v['pooled_AP']:.4f} |")
        lines += ["", "Fold-fitted JSD weights: "+str([v["JSD_weight"] for v in s["learned"]["convex_linear"]])+".", ""]
        for name in ["LR_extended_minus_LR_two", "MLP_extended_minus_MLP_two"]:
            v = s["paired_differences"][name]
            lo, hi = v["conditional_ci95"]
            lines.append(f"- {name}: {v['estimate']:.4f}, conditional paired interval [{lo:.4f}, {hi:.4f}].")
        warnings_count = sum(len(f.get("warnings", [])) for value in s["learned"].values() for f in value)
        lines += ["", f"Recorded training warnings: {warnings_count}.", ""]
    lines += ["## Interpretation", "",
              "- Fixed formulas are label-free; convex weights, LR and MLP are fitted using prediction-change outcomes.",
              "- LR/MLP probabilities are learned scores, not automatically zero-preserving or monotone distances.",
              "- Conditional bootstrap intervals reuse OOF predictions without refitting; multiple comparisons are exploratory.",
              "- CV and source grouping control within-round leakage, not adaptivity across the research project.",
              "- Fresh data and a frozen definition are required to confirm a selected improvement.",
              "- Novelty and semantic usefulness are separate from a development AUROC gain."]
    (output / "report.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--transport-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    run(args.run_dir, args.transport_dir, args.output_dir)
