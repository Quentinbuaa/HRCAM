"""Audit-ready, fixed-score summary of an untouched external test sample."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, mannwhitneyu
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
BACKGROUNDS = ("black", "gray", "white")
BASELINES = ("H-RCAM", "NCD", "JSD", "D1", "D2", "D3", "Foreground shift", "Background increase")
FUSIONS = ("Minimum", "Maximum", "Arithmetic", "Geometric")


def scores(rows):
    j = np.asarray([r["jsd"] for r in rows], dtype=float)
    c = np.asarray([r["d4"] for r in rows], dtype=float) / 2
    return {
        "H-RCAM": 2 * j * c / (j + c + 1e-12),
        "NCD": c,
        "JSD": j,
        "Minimum": np.minimum(j, c),
        "Maximum": np.maximum(j, c),
        "Arithmetic": (j + c) / 2,
        "Geometric": np.sqrt(j * c),
        "D1": np.asarray([r["d1"] for r in rows], dtype=float),
        "D2": np.asarray([r["d2"] for r in rows], dtype=float),
        "D3": np.asarray([r["d3"] for r in rows], dtype=float),
        "Foreground shift": np.asarray([r["foreground_shift"] for r in rows], dtype=float),
        "Background increase": np.asarray([r["background_increase"] for r in rows], dtype=float),
    }


def score_stats(y, values):
    if len(np.unique(y)) != 2:
        return {"auroc": None, "ap": None, "median_consistent": None, "median_inconsistent": None}
    return {
        "auroc": float(roc_auc_score(y, values)),
        "ap": float(average_precision_score(y, values)),
        "median_consistent": float(np.median(values[y == 0])),
        "median_inconsistent": float(np.median(values[y == 1])),
    }


def interval(values):
    return [float(x) for x in np.quantile(values, [.025, .5, .975])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="flowers102_external_v1")
    parser.add_argument("--output-stem", default="external_summary")
    parser.add_argument("--expected-sources", type=int, default=1020)
    parser.add_argument("--dataset-label", default="Oxford Flowers-102")
    parser.add_argument("--protocol", default="FLOWERS102_EXTERNAL_PROTOCOL.md")
    args = parser.parse_args()
    folder = ROOT / "runs" / args.run
    if not args.output_stem.replace("_", "").isalnum():
        raise ValueError("Use an alphanumeric output stem")
    result = folder / f"{args.output_stem}.json"
    if result.exists():
        raise FileExistsError("Refusing to replace existing summary")
    rows = [json.loads(line) for line in (folder / "external_records.jsonl").read_text().splitlines() if line]
    own = [r for r in rows if r["cam_mode"] == "own"]
    assert len(own) == args.expected_sources * 3
    for source in set(r["source_id"] for r in own):
        assert {r["background"] for r in own if r["source_id"] == source} == set(BACKGROUNDS)
    eligible = [r for r in own if r["mask_eligible"] and r["valid_cam"]]
    y = np.asarray([int(r["inconsistent"]) for r in eligible], dtype=int)
    source_ids = np.asarray([r["source_id"] for r in eligible], dtype=int)
    score = scores(eligible)
    transport_path = folder / "external_transport.jsonl"
    transport_status = "not computed"
    methods = list(BASELINES)
    if transport_path.exists():
        transport_rows = [json.loads(line) for line in transport_path.read_text().splitlines() if line]
        transport = {(r["source_id"], r["background"]): r for r in transport_rows}
        keys = [(r["source_id"], r["background"]) for r in eligible]
        if len(transport_rows) == len(eligible) and set(transport) == set(keys):
            score["BAT"] = np.asarray([transport[key]["BAT"] for key in keys], dtype=float)
            score["SpatialW1"] = np.asarray([transport[key]["SpatialW1"] for key in keys], dtype=float)
            methods.extend(["BAT", "SpatialW1"])
            transport_status = "complete"
        else:
            transport_status = f"partial {len(transport_rows)}/{len(eligible)}; omitted from comparison"
    if not all(np.isfinite(values).all() for values in score.values()):
        raise ValueError("Nonfinite candidate score among eligible records")
    if len(np.unique(y)) != 2:
        raise RuntimeError("Only one outcome class; cannot calculate AUROC")
    per_method = {name: score_stats(y, score[name]) for name in methods}
    per_fusion = {name: score_stats(y, score[name]) for name in FUSIONS}
    by_background = {}
    for bg in BACKGROUNDS:
        select = np.asarray([r["background"] == bg for r in eligible])
        by_background[bg] = {
            "pairs": int(select.sum()),
            "inconsistent": int(y[select].sum()),
            "prevalence": float(y[select].mean()),
            "H-RCAM": score_stats(y[select], score["H-RCAM"][select]),
            "NCD": score_stats(y[select], score["NCD"][select]),
            "JSD": score_stats(y[select], score["JSD"][select]),
        }
    consistent = score["H-RCAM"][y == 0]
    changed = score["H-RCAM"][y == 1]
    u = mannwhitneyu(consistent, changed, alternative="two-sided")
    ks = ks_2samp(consistent, changed, alternative="two-sided")
    rng = np.random.default_rng(20261012)
    ids = np.unique(source_ids)
    groups = {i: np.flatnonzero(source_ids == i) for i in ids}
    auc_samples = {name: [] for name in ("H-RCAM", "NCD", "JSD", "Arithmetic", "Geometric")}
    delta_samples = {f"H-RCAM minus {name}": [] for name in ("NCD", "JSD", "Arithmetic", "Geometric")}
    for _ in range(1000):
        draw = rng.choice(ids, len(ids), replace=True)
        positions = np.concatenate([groups[i] for i in draw])
        yy = y[positions]
        if len(np.unique(yy)) != 2:
            continue
        for name in auc_samples:
            auc_samples[name].append(roc_auc_score(yy, score[name][positions]))
        for name in ("NCD", "JSD", "Arithmetic", "Geometric"):
            delta_samples[f"H-RCAM minus {name}"].append(auc_samples["H-RCAM"][-1] - auc_samples[name][-1])
    summary = {
        "protocol": args.protocol,
        "run": args.run,
        "all_sources": args.expected_sources,
        "all_own_mode_pairs": len(own),
        "mask_eligible_pairs": sum(r["mask_eligible"] for r in own),
        "valid_cam_pairs": sum(r["valid_cam"] for r in own),
        "analyzed_pairs": len(eligible),
        "analyzed_sources": len(ids),
        "inconsistent_pairs": int(y.sum()),
        "prevalence": float(y.mean()),
        "clean_source_accuracy": float(np.mean([r["original_correct"] for r in own if r["background"] == "gray"])),
        "methods": per_method,
        "transport_status": transport_status,
        "fusion_ablation": per_fusion,
        "by_background": by_background,
        "H-RCAM_group_tests": {
            "mann_whitney_u": float(u.statistic), "mann_whitney_p": float(u.pvalue),
            "rank_effect_AU": float(1 - u.statistic / (len(consistent) * len(changed))),
            "ks_statistic": float(ks.statistic), "ks_p": float(ks.pvalue),
        },
        "source_cluster_bootstrap": {
            "seed": 20261012, "requested_replicates": 1000,
            "valid_replicates": len(auc_samples["H-RCAM"]),
            "auroc_intervals": {name: interval(values) for name, values in auc_samples.items()},
            "paired_auroc_difference_intervals": {name: interval(values) for name, values in delta_samples.items()},
        },
    }
    result.write_text(json.dumps(summary, indent=2))
    lines = [f"# {args.dataset_label} untouched external test", "", f"Run: `{args.run}`; protocol: `{args.protocol}`.", "",
             f"Analyzed {len(eligible)}/{len(own)} pairs from {len(ids)}/{args.expected_sources} sources. Inconsistent: {int(y.sum())} ({y.mean():.2%}).",
             f"Clean source accuracy: {summary['clean_source_accuracy']:.2%}. Transport candidates: {transport_status}.", "",
             "| Measure | AUROC | AP | Median unchanged | Median changed |", "|---|---:|---:|---:|---:|"]
    for name, item in per_method.items():
        lines.append(f"| {name} | {item['auroc']:.4f} | {item['ap']:.4f} | {item['median_consistent']:.4f} | {item['median_inconsistent']:.4f} |")
    lines.extend(["", "Fixed JSD/NCD fusion rules:", "", "| Rule | AUROC | AP |", "|---|---:|---:|"])
    for name, item in per_fusion.items():
        lines.append(f"| {name} | {item['auroc']:.4f} | {item['ap']:.4f} |")
    lines.extend(["", "Source-cluster bootstrap AUROC difference intervals (2.5%, median, 97.5%):"])
    for name, values in summary["source_cluster_bootstrap"]["paired_auroc_difference_intervals"].items():
        lines.append(f"- {name}: {values}")
    lines.extend(["", "This is an external validation of the frozen metric, not proof of causal robustness."])
    (folder / f"{args.output_stem.replace('summary', 'report')}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
