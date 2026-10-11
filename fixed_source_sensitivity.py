"""Audit cached own/fixed-source Grad-CAMs and evaluate unchanged H-RCAM.

Writes only to a new results directory. Does not train, select, or recalibrate.
Run on remote-5080 with the original .venv and job.py CPU-affinity restriction.
"""
import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

BACKGROUNDS = ("black", "gray", "white")
SPECS = {
    "pets_v1": {
        "dataset": "Oxford-IIIT Pet", "classes": 37,
        "train": "pets_trainval.npz", "test": "pets_test.npz",
        "model_sha256": "f3edfdb3b780e21a407a1aefec16634aef331fddaaacad0064d828f998d6e50c",
        "records_sha256": "88efda909786478419329fc4b1ffafcf995238d7007f1b90fe141c2fd30af0fd",
        "published_n": 2979, "published_auroc": .8953, "published_ap": .6888,
    },
    "cub_v1": {
        "dataset": "CUB-200-2011", "classes": 200,
        "train": "cub_train.npz", "test": "cub_test.npz",
        "model_sha256": "9784845363e53b3a77f3892b27ff0969152afd0e7200dc304f4b1cecede5dc18",
        "records_sha256": "f8d8e1d4fdefbb56a23ff6b15dd809d4b1dd63967e0ed72008935d77b6d282d9",
        "published_n": 2889, "published_auroc": .8066, "published_ap": .8684,
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def components(first, second):
    """Independent vectorized JSD and NCD; inputs are N x H x W maps."""
    a = np.asarray(first, dtype=np.float64).reshape(len(first), -1)
    b = np.asarray(second, dtype=np.float64).reshape(len(second), -1)
    finite = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    valid = (finite & (a.min(1) >= 0) & (b.min(1) >= 0)
             & (a.sum(1) > 1e-8) & (b.sum(1) > 1e-8)
             & (a.std(1) > 1e-8) & (b.std(1) > 1e-8))
    jsd, ncd = np.full(len(a), np.nan), np.full(len(a), np.nan)
    av, bv = a[valid], b[valid]
    if len(av):
        p, q = av + 1e-8, bv + 1e-8
        p /= p.sum(1, keepdims=True)
        q /= q.sum(1, keepdims=True)
        mixture = (p + q) / 2
        jsd[valid] = np.clip((np.sum(p * np.log2(p / mixture), axis=1)
                              + np.sum(q * np.log2(q / mixture), axis=1)) / 2, 0, 1)
        ac = av - av.mean(1, keepdims=True)
        bc = bv - bv.mean(1, keepdims=True)
        correlation = (np.sum(ac * bc, axis=1)
                       / np.sqrt(np.sum(ac * ac, axis=1) * np.sum(bc * bc, axis=1)))
        ncd[valid] = np.clip((1 - correlation) / 2, 0, 1)
    return valid, jsd, ncd


def harmonic(jsd, ncd):
    return 2 * jsd * ncd / (jsd + ncd + 1e-12)


def estimates(y, scores):
    return {name: {"auroc": float(roc_auc_score(y, values)),
                   "ap": float(average_precision_score(y, values))}
            for name, values in scores.items()}


def bootstrap(y, source_ids, scores, repeats, seed):
    """Resample sources, retaining all available background pairs; same draw for all scores."""
    unique = np.unique(source_ids)
    groups = [np.flatnonzero(source_ids == source) for source in unique]
    rng = np.random.default_rng(seed)
    samples = {name: {"auroc": [], "ap": []} for name in scores}
    for _ in range(repeats):
        indices = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(y[indices])) < 2:
            continue
        for name, value in estimates(y[indices], {k: v[indices] for k, v in scores.items()}).items():
            for metric in ("auroc", "ap"):
                samples[name][metric].append(value[metric])
    result = estimates(y, scores)
    for name in scores:
        for metric in ("auroc", "ap"):
            result[name][metric + "_ci95"] = np.quantile(samples[name][metric], [.025, .975]).tolist()
    differences = {}
    for first, second in (("fixed_HRCAM", "own_HRCAM"),
                          ("fixed_HRCAM", "fixed_JSD"),
                          ("fixed_HRCAM", "fixed_NCD")):
        key = first + "_minus_" + second
        differences[key] = {}
        for metric in ("auroc", "ap"):
            delta = np.asarray(samples[first][metric]) - samples[second][metric]
            differences[key][metric] = {
                "difference": result[first][metric] - result[second][metric],
                "ci95": np.quantile(delta, [.025, .975]).tolist(),
            }
    return {"ranking": result, "paired_differences": differences,
            "successful_resamples": len(samples["own_HRCAM"]["auroc"])}


def replay(root, spec, folder, ids, maps, original_maps, masks, rows):
    """Fresh autograd replay, explicitly choosing original predicted targets."""
    import torch
    import torch.nn.functional as functional
    from torchvision.models import resnet18
    import experiment as exp

    exp.seed_all(20260911)
    with np.load(root / "data" / spec["train"]) as training:
        offset = len(training["labels"])
    with np.load(root / "data" / spec["test"]) as testing:
        # Match original batch size/order to eliminate batch-dependent roundoff.
        selected = ids[:16] - offset
        assert (selected >= 0).all()
        images = testing["images"][selected]
        data_masks = testing["masks"][selected]
    assert np.array_equal(data_masks, masks[:16])
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(512, spec["classes"])
    model.load_state_dict(torch.load(folder / "classifier.pt", map_location="cpu", weights_only=True))
    model.cuda().eval()

    def get_cam(array, fixed=None):
        x = torch.from_numpy(array.transpose(0, 3, 1, 2).copy()).cuda().float() / 255
        x = ((x - x.new_tensor([.485, .456, .406])[None, :, None, None])
             / x.new_tensor([.229, .224, .225])[None, :, None, None])
        captured = []
        handle = model.layer4.register_forward_hook(lambda module, inputs, output: captured.append(output))
        try:
            logits = model(x)
            pred = logits.argmax(1)
            targets = pred if fixed is None else torch.as_tensor(fixed, device="cuda")
            activation = captured[0]
            gradient, = torch.autograd.grad(logits.gather(1, targets[:, None]).sum(), activation)
            cam = (activation * gradient.mean((2, 3), keepdim=True)).sum(1, keepdim=True).relu()
            cam = functional.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[:, 0]
            cam = cam - cam.amin((1, 2), keepdim=True)
            cam = cam / cam.amax((1, 2), keepdim=True).clamp_min(1e-12)
            return pred.detach().cpu().numpy(), cam.detach().cpu().numpy()
        finally:
            handle.remove()

    pred0, cam0 = get_cam(images)
    errors = {"source": float(np.max(np.abs(cam0 - original_maps[:16])))}
    for bg_index, (bg, color) in enumerate(zip(BACKGROUNDS, (0, 128, 255))):
        followup = np.where(masks[:16, :, :, None], images, color).astype(np.uint8)
        assert np.array_equal(followup[masks[:16]], images[masks[:16]])
        pred1, own = get_cam(followup)
        pred2, fixed = get_cam(followup, pred0)
        assert np.array_equal(pred1, pred2)
        for i, source in enumerate(ids[:16]):
            row = rows[(int(source), bg, "own")]
            assert row["original_prediction"] == int(pred0[i])
            assert row["transformed_prediction"] == int(pred1[i])
        errors[bg + "_own"] = float(np.max(np.abs(own - maps[:16, bg_index, 0])))
        errors[bg + "_fixed_source"] = float(np.max(np.abs(fixed - maps[:16, bg_index, 1])))
    assert max(errors.values()) < 3e-5, errors
    del model
    torch.cuda.empty_cache()
    return {"source_images": len(images), "background_pairs": 3 * len(images),
            "maximum_absolute_errors": errors, "target": "source predicted class for both images",
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}


def analyze_run(root, name, output, repeats):
    spec = SPECS[name]
    folder = root / "runs" / name
    originals = {file: sha256(folder / file)
                 for file in ("classifier.pt", "holdout_records.jsonl", "splits.json", "holdout_heatmaps.npz")}
    assert originals["classifier.pt"] == spec["model_sha256"]
    assert originals["holdout_records.jsonl"] == spec["records_sha256"]
    original_rows = [json.loads(line) for line in (folder / "holdout_records.jsonl").read_text().splitlines()]
    rows = {(r["source_id"], r["background"], r["cam_mode"]): r for r in original_rows}
    assert len(rows) == len(original_rows), "Duplicate source/background/mode"
    print(name + ": loading both cached map conventions", flush=True)
    with np.load(folder / "holdout_heatmaps.npz") as cache:
        ids, maps = cache["source_ids"], cache["maps"]
        original_maps, masks = cache["original_maps"], cache["masks"]
    splits = json.loads((folder / "splits.json").read_text())
    assert np.array_equal(ids, splits["holdout"])
    assert maps.shape == (len(ids), 3, 2, 224, 224)
    print(name + ": checking actual GPU Grad-CAM targets on original first batch", flush=True)
    gpu_check = replay(root, spec, folder, ids, maps, original_maps, masks, rows)
    keys = [(int(source), bg) for source in ids for bg in BACKGROUNDS]
    metrics = {mode: {} for mode in ("own", "fixed")}
    max_metric_error = 0.
    for mode_index, mode in enumerate(("own", "fixed")):
        print(name + ": independently recomputing all " + mode + " JSD/NCD components", flush=True)
        for start in range(0, len(ids), 16):
            stop = min(start + 16, len(ids))
            for bg_index, bg in enumerate(BACKGROUNDS):
                valid, jsd, ncd = components(original_maps[start:stop], maps[start:stop, bg_index, mode_index])
                for i, source in enumerate(ids[start:stop]):
                    key = (int(source), bg)
                    row = rows[(*key, mode)]
                    assert bool(valid[i]) == row["valid_cam"]
                    fraction = float(masks[start + i].mean())
                    assert abs(fraction - row["mask_fraction"]) < 1e-12
                    assert row["mask_eligible"] == bool(.05 <= fraction <= .95)
                    if valid[i]:
                        error = max(abs(jsd[i] - row["jsd"]), abs(ncd[i] - row["d4"] / 2))
                        max_metric_error = max(max_metric_error, error)
                        assert error < 1e-10, (key, mode, error)
                    metrics[mode][key] = (bool(valid[i] and row["mask_eligible"]), jsd[i], ncd[i])
    del maps, original_maps, masks
    gc.collect()
    matched = [key for key in keys if metrics["own"][key][0] and metrics["fixed"][key][0]]
    y = np.array([rows[(*key, "own")]["inconsistent"] for key in matched], dtype=int)
    source_ids = np.array([key[0] for key in matched])
    bg_names = np.array([key[1] for key in matched])
    scores = {}
    for mode in ("own", "fixed"):
        scores[mode + "_JSD"] = np.array([metrics[mode][key][1] for key in matched])
        scores[mode + "_NCD"] = np.array([metrics[mode][key][2] for key in matched])
        scores[mode + "_HRCAM"] = harmonic(scores[mode + "_JSD"], scores[mode + "_NCD"])
    for key in keys:
        own_row, fixed_row = rows[(*key, "own")], rows[(*key, "fixed")]
        for field in ("original_prediction", "transformed_prediction", "inconsistent", "mask_eligible"):
            assert own_row[field] == fixed_row[field]
        assert own_row["inconsistent"] == (own_row["original_prediction"] != own_row["transformed_prediction"])
    consistent_error = float(np.max(np.abs(scores["own_HRCAM"][y == 0] - scores["fixed_HRCAM"][y == 0])))
    assert consistent_error < 1e-10, consistent_error
    # Reproduce the manuscript on its original own-class valid subset, not a silently changed subset.
    own_keys = [key for key in keys if metrics["own"][key][0]]
    own_y = np.array([rows[(*key, "own")]["inconsistent"] for key in own_keys], int)
    own_s = harmonic(np.array([metrics["own"][key][1] for key in own_keys]),
                     np.array([metrics["own"][key][2] for key in own_keys]))
    primary = estimates(own_y, {"HRCAM": own_s})["HRCAM"]
    assert len(own_keys) == spec["published_n"]
    assert abs(primary["auroc"] - spec["published_auroc"]) < 5e-5
    assert abs(primary["ap"] - spec["published_ap"]) < 5e-5
    coverage = {"all_candidate_pairs": len(keys), "all_source_images": len(ids),
                "mask_eligible_pairs": sum(rows[(*key, "own")]["mask_eligible"] for key in keys),
                "own_valid_eligible_pairs": len(own_keys),
                "fixed_valid_eligible_pairs": sum(metrics["fixed"][key][0] for key in keys),
                "matched_pairs": len(matched), "matched_sources": len(np.unique(source_ids)),
                "matched_consistent": int((y == 0).sum()), "matched_inconsistent": int(y.sum()),
                "inconsistency_prevalence": float(y.mean()),
                "own_only_valid_eligible_pairs": sum(metrics["own"][key][0] and not metrics["fixed"][key][0] for key in keys),
                "fixed_only_valid_eligible_pairs": sum(metrics["fixed"][key][0] and not metrics["own"][key][0] for key in keys)}
    print(name + ": coverage " + json.dumps(coverage), flush=True)
    print(name + ": point estimates " + json.dumps(estimates(y, scores)), flush=True)
    print(name + ": paired source-cluster bootstrap", flush=True)
    inference = bootstrap(y, source_ids, scores, repeats, 20261011)
    strata = {}
    for bg in BACKGROUNDS:
        selected = bg_names == bg
        strata[bg] = {"n": int(selected.sum()), "inconsistent": int(y[selected].sum()),
                      "ranking": estimates(y[selected], {k: v[selected] for k, v in scores.items()})}
    medians = {mode: {"consistent": float(np.median(scores[mode + "_HRCAM"][y == 0])),
                      "inconsistent": float(np.median(scores[mode + "_HRCAM"][y == 1]))}
               for mode in ("own", "fixed")}
    result = {"dataset": spec["dataset"], "source_run": name, "source_role": "holdout",
              "coverage": coverage, "primary_own_class_reproduction": primary,
              "fresh_autograd_replay": gpu_check, "maximum_component_error": float(max_metric_error),
              "consistent_pair_maximum_hrcam_difference": consistent_error,
              "group_medians": medians, "backgrounds": strata, **inference,
              "original_file_sha256": originals}
    for file, value in originals.items():
        assert sha256(folder / file) == value, "Original file was modified"
    (output / (name + "_summary.json")).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(output / (name + "_matched_scores.npz"), source_ids=source_ids,
                        backgrounds=bg_names, inconsistent=y, **scores)
    print(name + ": complete", flush=True)
    return result


def report(results, repeats):
    lines = ["# Fixed-source-class sensitivity check: two development datasets", "",
             "The original classifier checkpoints, image subsets, masks, three background replacements,",
             "Grad-CAM layer, normalization, and H-RCAM formula are unchanged. No weights or thresholds",
             "were fitted. Original own-class and fixed-source-class maps were already saved together",
             "during the original extraction; this analysis verifies and reuses those caches.", "",
             "The source map explains the source image's predicted class in both conventions. The",
             "follow-up map explains its own prediction (primary convention) or the source prediction",
             "(sensitivity convention). The evaluation outcome remains actual prediction inconsistency.", "",
             "Results below use the intersection of valid pairs across both conventions. Percentile",
             f"95% intervals use {repeats} paired source-image-cluster bootstrap resamples; all available",
             "backgrounds of each sampled source are retained together. These are retrospective",
             "sensitivity results on the development datasets, not new independent verification.", "",
             "## Main comparison", "",
             "| Dataset | Convention | Matched pairs (inconsistent) | H-RCAM AUROC (95% CI) | H-RCAM AP (95% CI) |",
             "|---|---|---:|---:|---:|"]
    for result in results:
        c = result["coverage"]
        for mode, label in (("own", "Each image's own predicted class"), ("fixed", "Fixed source-predicted class")):
            s = result["ranking"][mode + "_HRCAM"]
            lines.append(f"| {result['dataset']} | {label} | {c['matched_pairs']} ({c['matched_inconsistent']}) | "
                         f"{s['auroc']:.4f} [{s['auroc_ci95'][0]:.4f}, {s['auroc_ci95'][1]:.4f}] | "
                         f"{s['ap']:.4f} [{s['ap_ci95'][0]:.4f}, {s['ap_ci95'][1]:.4f}] |")
    lines += ["", "## Paired change: fixed-source minus own-class H-RCAM", "",
              "| Dataset | AUROC difference (95% CI) | AP difference (95% CI) |", "|---|---:|---:|"]
    for result in results:
        delta = result["paired_differences"]["fixed_HRCAM_minus_own_HRCAM"]
        values = [f"{delta[m]['difference']:+.4f} [{delta[m]['ci95'][0]:+.4f}, {delta[m]['ci95'][1]:+.4f}]"
                  for m in ("auroc", "ap")]
        lines.append(f"| {result['dataset']} | " + " | ".join(values) + " |")
    lines += ["", "## Component comparison under fixed-source targeting", "",
              "| Dataset | Measure | AUROC | AP |", "|---|---|---:|---:|"]
    for result in results:
        for name in ("JSD", "NCD", "HRCAM"):
            s = result["ranking"]["fixed_" + name]
            lines.append(f"| {result['dataset']} | {name} | {s['auroc']:.4f} | {s['ap']:.4f} |")
    lines += ["", "## Checks and coverage", ""]
    for result in results:
        c, p = result["coverage"], result["primary_own_class_reproduction"]
        lines += [f"### {result['dataset']}", "",
                  f"- Reproduced manuscript own-class H-RCAM: AUROC {p['auroc']:.4f}, AP {p['ap']:.4f}.",
                  f"- Eligible valid pairs: own {c['own_valid_eligible_pairs']}; fixed {c['fixed_valid_eligible_pairs']}; matched {c['matched_pairs']}.",
                  f"- Own-valid pairs excluded by fixed-map invalidity: {c['own_only_valid_eligible_pairs']}.",
                  f"- Matched sources: {c['matched_sources']}; consistent pairs: {c['matched_consistent']}; inconsistent pairs: {c['matched_inconsistent']}.",
                  f"- Maximum H-RCAM difference for consistent pairs: {result['consistent_pair_maximum_hrcam_difference']:.3g} (same target expected).",
                  f"- Maximum independently recomputed JSD/NCD error: {result['maximum_component_error']:.3g}.",
                  f"- Fresh gradient replay: {result['fresh_autograd_replay']['background_pairs']} pairs for both conventions; maximum map error {max(result['fresh_autograd_replay']['maximum_absolute_errors'].values()):.3g}.",
                  "- Model, records, splits, and cached maps retain their original SHA256 hashes.", ""]
    lines += ["## Interpretation boundaries", "",
              "A drop under fixed-source targeting indicates sensitivity of the observed discrimination",
              "to the explanation-target convention. It is not a causal decomposition of the score.",
              "Remaining AUROC above 0.5 supports an association with label inconsistency even when",
              "the target class is held fixed, but does not establish prediction correctness, universal",
              "robustness, or superiority over all candidates. This check compares H-RCAM and its two",
              "components only; transport and other individual candidates were not reranked.", "",
              "The original manuscript, original extraction files, and GitHub release are not changed.", ""]
    return "\n".join(lines)


def self_test():
    a = np.array([[[0., 1.], [2., 3.]]])
    valid, j, c = components(a, a.copy())
    assert valid.all() and np.allclose(j, 0) and np.allclose(c, 0)
    valid, j, c = components(a, 3 - a)
    assert valid.all() and np.allclose(c, 1) and (j > 0).all()
    valid, _, _ = components(a, np.zeros_like(a))
    assert not valid.any()
    assert np.allclose(harmonic(np.array([0., .5, 1.]), np.array([1., .5, 1.])), [0., .5, 1.])
    y = np.tile([0, 1], 12)
    ids = np.repeat(np.arange(8), 3)
    scores = {name: y * .5 + np.linspace(0, .1, len(y))
              for name in ("own_HRCAM", "fixed_HRCAM", "fixed_JSD", "fixed_NCD")}
    res = bootstrap(y, ids, scores, 20, 1)
    assert res["paired_differences"]["fixed_HRCAM_minus_own_HRCAM"]["auroc"]["ci95"] == [0., 0.]
    print("Self-tests passed", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        return
    if args.output is None:
        parser.error("--output is required and must be a new directory")
    assert set(os.sched_getaffinity(0)) <= set(range(24, 32)), "Use approved CPU affinity 24-31"
    args.output.mkdir(parents=True, exist_ok=False)
    results = [analyze_run(args.root, name, args.output, args.bootstrap) for name in SPECS]
    summary = {"analysis": "fixed-source-class sensitivity, unchanged H-RCAM",
               "bootstrap_repeats": args.bootstrap, "bootstrap_seed": 20261011,
               "cpu_affinity": sorted(os.sched_getaffinity(0)),
               "script_sha256": sha256(Path(__file__)), "datasets": results}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (args.output / "REPORT.md").write_text(report(results, args.bootstrap))
    print("Saved " + str(args.output / "REPORT.md"), flush=True)


if __name__ == "__main__":
    main()
