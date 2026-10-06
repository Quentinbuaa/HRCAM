"""Development-only evaluation and controlled examples; no holdout access."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from metrics import heatmap_metrics, apply_minmax
from transport_metrics import distances, ot


ROOT = Path(__file__).resolve().parent
BACKGROUNDS = ["black", "gray", "white"]


def synthetic(config):
    mask = np.zeros((56, 56), bool)
    mask[:, :28] = True

    def point(y, x):
        a = np.zeros((56, 56))
        a[y, x] = 1
        return a

    a = point(28, 12)
    cases = [
        ("identity", a, a),
        ("near_shift", a, point(28, 16)),
        ("far_shift", a, point(28, 24)),
        ("within_foreground", point(28, 20), point(28, 24)),
        ("cross_into_background", point(28, 24), point(28, 28)),
        ("fixed_centroid_redistribution", point(28, 12)+point(28, 20), point(24, 16)+point(32, 16)),
    ]
    output = []
    for name, first, second in cases:
        old = heatmap_metrics(first, second, mask)
        scaled, clipped = apply_minmax([[old[n] for n in ["d1", "d2", "d3", "d4"]]], config["normalization"]["fixed"])
        output.append({"case": name, **{n: old[n] for n in ["d1", "d2", "d3", "d4", "jsd"]},
                       "RCAM": float(scaled.mean()), "RCAM_calibration_clipped": bool(clipped.any()),
                       **distances(first, second, mask, 14)})
    by_name = {r["case"]: r for r in output}
    assert np.isclose(by_name["near_shift"]["d4"], by_name["far_shift"]["d4"])
    assert by_name["near_shift"]["spatial_w1"] < by_name["far_shift"]["spatial_w1"]
    assert np.isclose(by_name["within_foreground"]["spatial_w1"], by_name["cross_into_background"]["spatial_w1"])
    assert by_name["within_foreground"]["BAT"] < by_name["cross_into_background"]["BAT"]
    assert by_name["fixed_centroid_redistribution"]["d1"] < 1e-10
    assert by_name["fixed_centroid_redistribution"]["spatial_w1"] > 0
    return output


def summarize(rows, scores, repeats=400):
    y = np.array([r["inconsistent"] for r in rows], int)
    ids = np.array([r["source_id"] for r in rows])
    groups = [np.flatnonzero(ids == sid) for sid in np.unique(ids)]
    result = {"n_pairs": len(rows), "n_sources": len(groups), "positive_prevalence": float(y.mean()),
              "ranking": {}, "consistent": {}, "paired_differences": {}}
    for name, s in scores.items():
        result["ranking"][name] = {"auroc": float(roc_auc_score(y, s)),
                                    "AP": float(average_precision_score(y, s))}
        result["consistent"][name] = {"q10_median_q90": np.quantile(s[y == 0], [.1, .5, .9]).tolist()}
    bootstrap = {n: [] for n in ["RCAM", "d4", "spatial_w1", "BAT"]}
    rng = np.random.default_rng(20260914)
    for _ in range(repeats):
        draw = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(y[draw])) < 2:
            continue
        for n in bootstrap:
            bootstrap[n].append(roc_auc_score(y[draw], scores[n][draw]))
    for name in ["spatial_w1", "BAT"]:
        for ref in ["RCAM", "d4"]:
            delta = np.array(bootstrap[name])-np.array(bootstrap[ref])
            result["paired_differences"][f"{name}_minus_{ref}"] = {
                "estimate": result["ranking"][name]["auroc"]-result["ranking"][ref]["auroc"],
                "ci95": np.quantile(delta, [.025, .975]).tolist()}
    names = list(scores)
    result["consistent_correlation_names"] = names
    result["consistent_spearman"] = spearmanr(np.stack([scores[n][y == 0] for n in names], axis=1)).statistic.tolist()
    return result


def run(folder, output):
    config = json.loads((folder / "frozen_analysis.json").read_text())
    result = {"scope": "Exploratory development only; no fresh confirmatory result",
              "primary_mode": "fixed", "grid": 14, "POT_version": ot.__version__,
              "synthetic": synthetic(config), "modes": {},
              "code_sha256": {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest()
                              for n in ["transport_metrics.py", "transport_development.py", "TRANSPORT_CANDIDATE_PROTOCOL.md"]}}
    print("Loading development heatmaps", flush=True)
    archive = np.load(folder / "development_heatmaps.npz")
    source_ids, originals, maps, masks = (archive[k] for k in ["source_ids", "original_maps", "maps", "masks"])
    assert maps.shape[:3] == (len(source_ids), 3, 2)
    positions = {int(sid): i for i, sid in enumerate(source_ids)}
    rows = [json.loads(line) for line in (folder / "development_records.jsonl").read_text().splitlines() if line]
    result["records_sha256"] = hashlib.sha256((folder / "development_records.jsonl").read_bytes()).hexdigest()
    new_rows = []
    for mode_index, mode in enumerate(["own", "fixed"]):
        selected = [r for r in rows if r["cam_mode"] == mode and r["valid_cam"] and r["mask_eligible"]]
        saved = np.load(folder / f"development_{mode}_scores.npz")
        assert np.array_equal(saved["source_ids"], [r["source_id"] for r in selected])
        scores = {n: saved[n] for n in ["d1", "d2", "d3", "d4", "RCAM", "jsd"]}
        candidates = {n: [] for n in ["spatial_w1", "BAT"]}
        elapsed = []
        for count, r in enumerate(selected):
            i, bg = positions[r["source_id"]], BACKGROUNDS.index(r["background"])
            start = time.perf_counter()
            values = distances(originals[i], maps[i, bg, mode_index], masks[i], 14)
            elapsed.append(time.perf_counter()-start)
            for n in candidates:
                candidates[n].append(values[n])
            new_rows.append({"source_id": r["source_id"], "background": r["background"], "cam_mode": mode, **values})
            if count % 300 == 0:
                print(f"{mode} {count}/{len(selected)}; transport seconds {sum(elapsed):.2f}", flush=True)
        scores.update({n: np.asarray(s) for n, s in candidates.items()})
        result["modes"][mode] = summarize(selected, scores)
        result["modes"][mode]["median_pair_seconds_both_candidates"] = float(np.median(elapsed))
        np.savez_compressed(output / f"development_{mode}_scores.npz", source_ids=saved["source_ids"],
                            inconsistent=saved["inconsistent"], **scores)
        print(mode, result["modes"][mode]["ranking"], flush=True)
    result["limitations"] = [
        "Candidate designed after earlier RCAM findings; development performance is not confirmation.",
        "Transport-based saliency comparison is established prior art; BAT novelty is unverified.",
        "Synthetic cases isolate specified properties and do not prove classifier vulnerability.",
        "The boundary penalty is an explicit fixed semantic assumption, not empirically optimal.",
        "Attribution moving onto the foreground is scored as change too; consult the signed mass change.",
        "A stable background-dependent or wrong classifier can still have zero distance.",
        "Pooling loses within-cell detail; the formal metric applies to represented distributions.",
        "Both candidates add mask/transport costs; a larger AUROC alone is not the sole acceptance criterion.",
    ]
    (output / "candidate_records.jsonl").write_text("".join(json.dumps(r)+"\n" for r in new_rows))
    (output / "summary.json").write_text(json.dumps(result, indent=2))
    lines = ["# Transport candidate development results", "", result["scope"], "",
             "## Controlled examples", "", "| Case | D1 | D4 | JSD | RCAM | Spatial W1 | BAT |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in result["synthetic"]:
        lines.append("| "+r["case"]+" | "+" | ".join(f"{r[n]:.4f}" for n in ["d1", "d4", "jsd", "RCAM", "spatial_w1", "BAT"])+" |")
    lines += ["", "RCAM uses frozen calibration scales; synthetic extrapolation is clipped and flagged in JSON.", ""]
    for mode, s in result["modes"].items():
        lines += [f"## {mode}-class CAM", "", f"{s['n_pairs']} pairs, {s['n_sources']} sources.", "",
                  "| Score | Development AUROC | AP |", "|---|---:|---:|"]
        for n, v in s["ranking"].items():
            lines.append(f"| {n} | {v['auroc']:.4f} | {v['AP']:.4f} |")
        lines.append("")
        for pair, v in s["paired_differences"].items():
            lo, hi = v["ci95"]
            lines.append(f"- {pair}: {v['estimate']:.4f}, paired 95% interval [{lo:.4f}, {hi:.4f}].")
        lines += ["", f"Median time for both candidate scores: {1000*s['median_pair_seconds_both_candidates']:.2f} ms/pair.", ""]
    lines += ["## Limitations", ""]+["- "+s for s in result["limitations"]]
    (output / "report.md").write_text("\n".join(lines)+"\n")
    print("Saved", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    run(args.run_dir, args.output_dir)
