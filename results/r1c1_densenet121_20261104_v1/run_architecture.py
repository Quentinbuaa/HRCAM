"""Fixed DenseNet-121 transfer check for R1C1; writes only a new run.

Run through job.py or taskset -c 24-31 on remote-5080. --self-test is CPU-only.
The submitted manuscript and the original runs are never modified.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import warnings

SAFE_CPUS = frozenset(range(24, 32))
BACKGROUNDS = (("black", 0), ("gray", 128), ("white", 255))
SPECS = {
    "pets_v1": {
        "dataset": "Oxford-IIIT Pet", "classes": 37, "seed": 20260911,
        "train": "pets_trainval.npz", "test": "pets_test.npz",
        "records_sha256": "88efda909786478419329fc4b1ffafcf995238d7007f1b90fe141c2fd30af0fd",
        "model_sha256": "f3edfdb3b780e21a407a1aefec16634aef331fddaaacad0064d828f998d6e50c",
    },
    "cub_v1": {
        "dataset": "CUB-200-2011", "classes": 200, "seed": 20260914,
        "train": "cub_train.npz", "test": "cub_test.npz",
        "records_sha256": "f8d8e1d4fdefbb56a23ff6b15dd809d4b1dd63967e0ed72008935d77b6d282d9",
        "model_sha256": "9784845363e53b3a77f3892b27ff0969152afd0e7200dc304f4b1cecede5dc18",
    },
}


def bind_resources(require_linux=True):
    if hasattr(os, "sched_getaffinity"):
        selected = set(os.sched_getaffinity(0)) & SAFE_CPUS
        if not selected:
            raise RuntimeError("Approved CPUs 24-31 unavailable")
        os.sched_setaffinity(0, selected)
        assert set(os.sched_getaffinity(0)) == selected
    elif require_linux:
        raise RuntimeError("The experiment requires the approved Linux host")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "4"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def components(first, second):
    import numpy as np
    a = np.asarray(first, dtype=np.float64).reshape(len(first), -1)
    b = np.asarray(second, dtype=np.float64).reshape(len(second), -1)
    valid = (np.isfinite(a).all(1) & np.isfinite(b).all(1)
             & (a.min(1) >= 0) & (b.min(1) >= 0)
             & (a.sum(1) > 1e-8) & (b.sum(1) > 1e-8)
             & (a.std(1) > 1e-8) & (b.std(1) > 1e-8))
    jsd, ncd = np.full(len(a), np.nan), np.full(len(a), np.nan)
    av, bv = a[valid], b[valid]
    if len(av):
        p, q = av + 1e-8, bv + 1e-8
        p /= p.sum(1, keepdims=True)
        q /= q.sum(1, keepdims=True)
        m = (p + q) / 2
        jsd[valid] = np.clip((np.sum(p * np.log2(p / m), 1)
                              + np.sum(q * np.log2(q / m), 1)) / 2, 0, 1)
        ac, bc = av - av.mean(1, keepdims=True), bv - bv.mean(1, keepdims=True)
        rho = np.sum(ac * bc, 1) / np.sqrt(np.sum(ac * ac, 1) * np.sum(bc * bc, 1))
        ncd[valid] = np.clip((1 - rho) / 2, 0, 1)
    return valid, jsd, ncd


def scores_from_rows(rows):
    import numpy as np
    j = np.array([row["jsd"] for row in rows])
    c = np.array([row["ncd"] for row in rows])
    return {"HRCAM": 2 * j * c / (j + c + 1e-12), "JSD": j, "NCD": c}


def point_estimates(y, scores):
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score
    if len(np.unique(y)) != 2:
        return {name: {"auroc": None, "ap": None} for name in scores}
    return {name: {"auroc": float(roc_auc_score(y, values)),
                   "ap": float(average_precision_score(y, values))}
            for name, values in scores.items()}


def inference(rows, repeats, seed, score_override=None):
    import numpy as np
    y = np.array([r["inconsistent"] for r in rows], dtype=int)
    source_ids = np.array([r["source_id"] for r in rows])
    scores = scores_from_rows(rows) if score_override is None else score_override
    ranking = point_estimates(y, scores)
    if len(np.unique(y)) != 2:
        return {"state": "single_outcome_class", "ranking": ranking, "successful_resamples": 0}
    unique = np.unique(source_ids)
    groups = [np.flatnonzero(source_ids == source) for source in unique]
    rng = np.random.default_rng(seed)
    draws = {name: {metric: [] for metric in ("auroc", "ap")} for name in scores}
    for _ in range(repeats):
        ids = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(y[ids])) < 2:
            continue
        estimates = point_estimates(y[ids], {name: values[ids] for name, values in scores.items()})
        for name in scores:
            for metric in ("auroc", "ap"):
                draws[name][metric].append(estimates[name][metric])
    successful = len(next(iter(draws.values()))["auroc"])
    if successful < max(2, int(repeats * .95)):
        raise RuntimeError("Too few valid source-cluster bootstrap resamples")
    for name in scores:
        for metric in ("auroc", "ap"):
            ranking[name][metric + "_ci95"] = np.quantile(draws[name][metric], [.025, .975]).tolist()
    contrasts = [("HRCAM", "JSD"), ("HRCAM", "NCD")]
    if "own_HRCAM" in scores:
        contrasts = [("fixed_HRCAM", "own_HRCAM")]
    differences = {}
    for first, second in contrasts:
        differences[first + "_minus_" + second] = {
            metric: {"difference": ranking[first][metric] - ranking[second][metric],
                     "ci95": np.quantile(np.array(draws[first][metric])
                                         - np.array(draws[second][metric]), [.025, .975]).tolist()}
            for metric in ("auroc", "ap")}
    return {"state": "complete", "ranking": ranking, "paired_differences": differences,
            "successful_resamples": successful}


def summarize(rows, repeats, seed):
    import numpy as np
    result = {}
    for mode in ("own", "fixed"):
        all_mode = [r for r in rows if r["cam_mode"] == mode]
        valid = [r for r in all_mode if r["mask_eligible"] and r["valid_cam"]]
        if not valid:
            raise RuntimeError("No valid pairs")
        source_rows = {r["source_id"]: r for r in all_mode}
        result[mode] = {
            "coverage": {"all_sources": len(source_rows), "all_pairs": len(all_mode),
                         "valid_sources": len({r["source_id"] for r in valid}), "valid_pairs": len(valid),
                         "mask_ineligible_pairs": sum(not r["mask_eligible"] for r in all_mode),
                         "invalid_map_pairs": sum(not r["valid_cam"] for r in all_mode),
                         "inconsistent_pairs": sum(r["inconsistent"] for r in valid),
                         "inconsistency_prevalence": float(np.mean([r["inconsistent"] for r in valid]))},
            "accuracy_all_sources": float(np.mean([r["original_correct"] for r in source_rows.values()])),
            "transformed_accuracy_all_pairs": float(np.mean([r["transformed_correct"] for r in all_mode])),
            "accuracy_valid_sources": float(np.mean([r["original_correct"] for r in
                {r["source_id"]: r for r in valid}.values()])),
            "transformed_accuracy_valid_pairs": float(np.mean([r["transformed_correct"] for r in valid])),
            **inference(valid, repeats, seed),
            "backgrounds": {},
        }
        for name, _ in BACKGROUNDS:
            subset = [r for r in valid if r["background"] == name]
            result[mode]["backgrounds"][name] = {
                "n": len(subset), "inconsistent": sum(r["inconsistent"] for r in subset),
                "ranking": point_estimates(np.array([r["inconsistent"] for r in subset]), scores_from_rows(subset))}
    own = {(r["source_id"], r["background"]): r for r in rows
           if r["cam_mode"] == "own" and r["mask_eligible"] and r["valid_cam"]}
    fixed = {(r["source_id"], r["background"]): r for r in rows
             if r["cam_mode"] == "fixed" and r["mask_eligible"] and r["valid_cam"]}
    keys = sorted(own.keys() & fixed.keys())
    own_rows, fixed_rows = [own[k] for k in keys], [fixed[k] for k in keys]
    assert all(a["inconsistent"] == b["inconsistent"] for a, b in zip(own_rows, fixed_rows))
    matched_scores = {"own_HRCAM": scores_from_rows(own_rows)["HRCAM"],
                      "fixed_HRCAM": scores_from_rows(fixed_rows)["HRCAM"]}
    result["matched_targets"] = {"n_pairs": len(keys),
        **inference(own_rows, repeats, seed, score_override=matched_scores)}
    return result


def common_backbone_comparison(first_rows, second_rows, first_summary, second_summary, repeats, seed):
    """Keep exactly the same eligible pairs for both backbones; outcomes may differ."""
    result = {}
    for mode in ("own", "fixed"):
        selected = []
        for rows in (first_rows, second_rows):
            selected.append({(r["source_id"], r["background"]): r for r in rows
                             if r["cam_mode"] == mode and r["mask_eligible"] and r["valid_cam"]})
        keys = sorted(selected[0].keys() & selected[1].keys())
        for key in keys:
            assert selected[0][key]["label"] == selected[1][key]["label"]
            assert selected[0][key]["mask_fraction"] == selected[1][key]["mask_fraction"]
        entry = {"n_pairs": len(keys), "n_sources": len({key[0] for key in keys}),
                 "DenseNet121_excluded_valid_pairs": len(selected[0]) - len(keys),
                 "ResNet18_excluded_valid_pairs": len(selected[1]) - len(keys)}
        for label, lookup, summary in zip(("DenseNet121", "ResNet18"), selected, (first_summary, second_summary)):
            subset = [lookup[key] for key in keys]
            # Reuse exactly the same analysis when there is no coverage change.
            unchanged = len(keys) == summary[mode]["coverage"]["valid_pairs"]
            estimate = ({key: summary[mode][key] for key in
                         ("state", "ranking", "successful_resamples", "paired_differences")}
                        if unchanged and summary[mode]["state"] == "complete"
                        else inference(subset, repeats, seed))
            entry[label] = {"inconsistent_pairs": sum(r["inconsistent"] for r in subset), **estimate}
        result[mode] = entry
    return result


def normalize_images(array):
    import numpy as np
    import torch
    x = torch.from_numpy(np.ascontiguousarray(array.transpose(0, 3, 1, 2))).to("cuda", torch.float32) / 255
    return (x - x.new_tensor([.485, .456, .406])[None, :, None, None]) / x.new_tensor([.229, .224, .225])[None, :, None, None]


def normalize_cam(native, size=(224, 224)):
    import torch.nn.functional as functional
    cam = functional.interpolate(native, size=size, mode="bilinear", align_corners=False)[:, 0]
    cam = cam - cam.amin((1, 2), keepdim=True)
    return cam / cam.amax((1, 2), keepdim=True).clamp_min(1e-12)


def gradcam_from_features(features, head, fixed_targets=None):
    import torch
    import torch.nn.functional as functional
    with torch.enable_grad():
        activation = features.detach().requires_grad_(True)
        logits = head(functional.adaptive_avg_pool2d(activation, (1, 1)).flatten(1))
        own = logits.argmax(1)
        targets = [own] if fixed_targets is None else [own, fixed_targets]
        maps = []
        for i, target in enumerate(targets):
            objective = logits.gather(1, target[:, None]).sum()
            gradient = torch.autograd.grad(objective, activation, retain_graph=i + 1 < len(targets))[0]
            native = (gradient.mean((2, 3), keepdim=True) * activation).sum(1, keepdim=True).relu()
            maps.append(normalize_cam(native).detach())
    return logits.detach(), maps


def self_test():
    import numpy as np
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import pearsonr
    rng = np.random.default_rng(20261104)
    a, b = rng.random((7, 8, 8)), rng.random((7, 8, 8))
    a[0], b[0] = 0, 0
    valid, j, c = components(a, b)
    assert not valid[0] and valid[1:].all()
    for i in range(1, 7):
        p, q = a[i].ravel() + 1e-8, b[i].ravel() + 1e-8
        np.testing.assert_allclose(j[i], jensenshannon(p / p.sum(), q / q.sum(), base=2) ** 2, atol=2e-15)
        np.testing.assert_allclose(c[i], (1 - pearsonr(a[i].ravel(), b[i].ravel()).statistic) / 2, atol=2e-15)
    same, sj, sc = components(a[1:], a[1:])
    assert same.all()
    np.testing.assert_allclose(sj, 0, atol=1e-15)
    np.testing.assert_allclose(sc, 0, atol=1e-15)
    # Each source includes both outcomes. Resampling whole sources must leave
    # this constructed source-level balance unchanged, with paired differences zero.
    rows = [{"source_id": source, "inconsistent": bool(k), "jsd": .1 + .1 * k,
             "ncd": .1 + .1 * k} for source in range(10) for k in (0, 1)]
    result = inference(rows, 30, 20261104)
    assert result["ranking"]["HRCAM"]["auroc_ci95"] == [1., 1.]
    assert result["paired_differences"]["HRCAM_minus_JSD"]["auroc"]["ci95"] == [0., 0.]
    records = []
    for row in rows:
        for mode in ("own", "fixed"):
            records.append(dict(row, cam_mode=mode, background="black" if row["inconsistent"] else "gray",
                                label=row["source_id"] % 2, mask_fraction=.5, mask_eligible=True,
                                valid_cam=True, original_correct=True,
                                transformed_correct=not row["inconsistent"]))
    complete_records = [dict(row) for row in records]
    # A legitimate invalid fixed-target map is excluded only from the fixed
    # and matched sets; it must not remove the corresponding own-target pair.
    records[-1]["valid_cam"] = False
    records[-1]["jsd"] = records[-1]["ncd"] = None
    coverage = summarize(list(reversed(records)), 30, 20261104)
    assert coverage["own"]["coverage"]["valid_pairs"] == 20
    assert coverage["fixed"]["coverage"]["valid_pairs"] == 19
    assert coverage["matched_targets"]["n_pairs"] == 19
    assert coverage["matched_targets"]["paired_differences"]["fixed_HRCAM_minus_own_HRCAM"]["auroc"]["ci95"] == [0., 0.]
    reference = summarize(complete_records, 30, 20261104)
    common = common_backbone_comparison(records, complete_records, coverage, reference, 30, 20261104)
    assert common["own"]["n_pairs"] == 20 and common["fixed"]["n_pairs"] == 19
    assert common["fixed"]["ResNet18_excluded_valid_pairs"] == 1
    print(json.dumps({"self_test": "passed", "checks": ["SciPy JSD", "SciPy Pearson", "zero maps",
        "identical maps", "paired source bootstrap", "mode-specific invalidity and matched coverage",
        "common-backbone comparison"]}))


def model_verification(model, images):
    import numpy as np
    import torch
    import torch.nn.functional as functional
    x = normalize_images(images[:4]).requires_grad_(True)
    with torch.enable_grad():
        activation = model.features(x)
        logits = model.classifier(functional.adaptive_avg_pool2d(activation, 1).flatten(1))
        targets = logits.argmax(1)
        gradient = torch.autograd.grad(logits.gather(1, targets[:, None]).sum(), activation)[0]
        full_cam = normalize_cam((gradient.mean((2, 3), keepdim=True) * activation).sum(1, keepdim=True).relu())
    detached_logits, detached_maps = gradcam_from_features(activation, model.classifier)
    torch.testing.assert_close(logits, detached_logits, rtol=0, atol=0)
    torch.testing.assert_close(full_cam, detached_maps[0], rtol=0, atol=0)
    weights = model.classifier.weight[targets][:, :, None, None]
    closed_cam = normalize_cam((activation.detach() * weights).sum(1, keepdim=True).relu())
    torch.testing.assert_close(closed_cam, full_cam, rtol=2e-4, atol=3e-5)
    # The frozen backbone must remain in evaluation mode, with no trainable parameters.
    assert not model.features.training and not any(p.requires_grad for p in model.features.parameters())
    original = detached_maps[0].cpu().numpy()
    valid, j, c = components(original, original)
    np.testing.assert_allclose(j[valid], 0, atol=1e-15)
    np.testing.assert_allclose(c[valid], 0, atol=1e-15)
    return {"full_autograd_map_max_error": float((full_cam - detached_maps[0]).abs().max()),
            "closed_form_cam_max_error": float((full_cam - closed_cam).abs().max()),
            "n_sources": len(x), "valid_identity_maps": int(valid.sum())}


def run_dataset(args, name, spec):
    import numpy as np
    import sklearn
    import torch
    import torch.nn.functional as functional
    import torchvision
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from torchvision.models import DenseNet121_Weights, densenet121
    original = args.root / "runs" / name
    output = args.output / name
    output.mkdir(parents=True, exist_ok=False)
    tracked = {"split": original / "splits.json", "baseline_checkpoint": original / "classifier.pt",
               "baseline_records": original / "holdout_records.jsonl",
               "prepared_train": args.root / "data" / spec["train"],
               "prepared_test": args.root / "data" / spec["test"]}
    before = {key: sha256(path) for key, path in tracked.items()}
    assert before["baseline_checkpoint"] == spec["model_sha256"]
    assert before["baseline_records"] == spec["records_sha256"]
    splits = json.loads(tracked["split"].read_text())
    all_roles = [index for ids in splits.values() for index in ids]
    assert len(all_roles) == len(set(all_roles))
    write_json(output / "splits.json", splits)
    torch.manual_seed(spec["seed"])
    np.random.seed(spec["seed"])
    torch.cuda.manual_seed_all(spec["seed"])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    weights = DenseNet121_Weights.IMAGENET1K_V1
    model = densenet121(weights=weights, progress=False)
    # Expose the final activated tensor as a module without changing DenseNet's forward.
    model.features = torch.nn.Sequential(model.features, torch.nn.ReLU(inplace=False))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.classifier = torch.nn.Linear(1024, spec["classes"])
    model = model.cuda().eval()
    backbone_before = {key: value.detach().cpu().clone() for key, value in model.features.state_dict().items()}
    print(name + ": loading original prepared images and splits", flush=True)
    with np.load(tracked["prepared_train"]) as archive:
        train_images, train_labels = archive["images"], archive["labels"]
    train_ids = np.asarray(splits["training"], dtype=int)
    assert (train_ids >= 0).all() and (train_ids < len(train_labels)).all()
    train_features = []
    with torch.inference_mode():
        for start in range(0, len(train_ids), args.batch):
            chosen = train_ids[start:start + args.batch]
            activation = model.features(normalize_images(train_images[chosen]))
            train_features.append(functional.adaptive_avg_pool2d(activation, 1).flatten(1).cpu().numpy())
            if start % (args.batch * 20) == 0:
                print(name + ": training features " + str(start) + "/" + str(len(train_ids)), flush=True)
    train_features = np.concatenate(train_features)
    np.savez_compressed(output / "training_features.npz", source_ids=train_ids, features=train_features,
                        labels=train_labels[train_ids])
    print(name + ": fitting fixed-C logistic regression head", flush=True)
    head = LogisticRegression(C=1., solver="lbfgs", max_iter=1000)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        head.fit(train_features, train_labels[train_ids])
    fit_warnings = [str(w.message) for w in caught]
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        write_json(output / "fit_failure.json", {"warnings": fit_warnings, "n_iter": head.n_iter_.tolist()})
        raise RuntimeError("Head failed to converge; inspect numerical fit without changing C or evaluation subset")
    assert np.array_equal(head.classes_, np.arange(spec["classes"]))
    with torch.no_grad():
        model.classifier.weight.copy_(torch.as_tensor(head.coef_, device="cuda", dtype=torch.float32))
        model.classifier.bias.copy_(torch.as_tensor(head.intercept_, device="cuda", dtype=torch.float32))
    torch.save(model.state_dict(), output / "classifier.pt")
    n_training = len(train_labels)
    holdout_ids = np.asarray(splits["holdout"], dtype=int)
    with np.load(tracked["prepared_test"]) as archive:
        testing = {key: archive[key] for key in ("images", "labels", "masks")}
    assert (holdout_ids >= n_training).all() and (holdout_ids < n_training + len(testing["labels"])).all()
    selected = holdout_ids - n_training
    images, labels, masks = (testing[key][selected] for key in ("images", "labels", "masks"))
    del train_images, train_labels, train_features, testing
    gc.collect()
    verification = model_verification(model, images)
    cache = np.lib.format.open_memmap(output / "holdout_heatmaps.npy", mode="w+", dtype="float32",
                                     shape=(len(images), 7, 224, 224))
    np.savez_compressed(output / "holdout_sources.npz", source_ids=holdout_ids, labels=labels, masks=masks)
    records = []
    started = time.time()
    for start in range(0, len(images), args.batch):
        stop = min(start + args.batch, len(images))
        array, foreground = images[start:stop], masks[start:stop].astype(bool)
        with torch.no_grad():
            original_features = model.features(normalize_images(array))
        original_logits, source_maps = gradcam_from_features(original_features, model.classifier)
        source_map = source_maps[0].cpu().numpy()
        pred0 = original_logits.argmax(1)
        p0 = original_logits.softmax(1).cpu().numpy().astype(np.float64)
        cache[start:stop, 0] = source_map
        for bg_index, (background, color) in enumerate(BACKGROUNDS):
            altered = np.where(foreground[:, :, :, None], array, color).astype(np.uint8)
            assert np.array_equal(altered[foreground], array[foreground])
            with torch.no_grad():
                follow_features = model.features(normalize_images(altered))
            follow_logits, follow_maps = gradcam_from_features(follow_features, model.classifier, pred0)
            pred1 = follow_logits.argmax(1)
            p1 = follow_logits.softmax(1).cpu().numpy().astype(np.float64)
            consistent = (pred0 == pred1).cpu().numpy()
            if consistent.any():
                np.testing.assert_array_equal(follow_maps[0].cpu().numpy()[consistent],
                                              follow_maps[1].cpu().numpy()[consistent])
            for mode_index, mode in enumerate(("own", "fixed")):
                follow_map = follow_maps[mode_index].cpu().numpy()
                cache[start:stop, 1 + bg_index * 2 + mode_index] = follow_map
                valid, jsd, ncd = components(source_map, follow_map)
                for i in range(len(array)):
                    mask_fraction = float(foreground[i].mean())
                    first, second = int(pred0[i]), int(pred1[i])
                    label = int(labels[start + i])
                    records.append({"source_id": int(holdout_ids[start + i]), "role": "holdout",
                        "background": background, "cam_mode": mode, "label": label,
                        "original_prediction": first, "transformed_prediction": second,
                        "original_correct": first == label, "transformed_correct": second == label,
                        "inconsistent": first != second, "mask_fraction": mask_fraction,
                        "mask_eligible": .05 <= mask_fraction <= .95, "valid_cam": bool(valid[i]),
                        "jsd": float(jsd[i]) if valid[i] else None,
                        "ncd": float(ncd[i]) if valid[i] else None,
                        "original_confidence": float(p0[i, first]),
                        "confidence_drop": float(p0[i, first] - p1[i, first])})
        if start % (args.batch * 5) == 0:
            print(name + ": extraction " + str(stop) + "/" + str(len(images))
                  + " elapsed=" + str(round(time.time() - started, 1)) + "s", flush=True)
    cache.flush()
    del cache
    for key, value in model.features.state_dict().items():
        assert torch.equal(value.detach().cpu(), backbone_before[key]), key
    (output / "holdout_records.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in records), encoding="utf-8")
    print(name + ": computing source-cluster intervals", flush=True)
    result = summarize(records, args.bootstrap, 20261104)
    baseline_records = [json.loads(line) for line in tracked["baseline_records"].read_text().splitlines()]
    assert {r["source_id"] for r in baseline_records} == set(holdout_ids.tolist())
    for row in baseline_records:
        row["ncd"] = row["d4"] / 2 if row["valid_cam"] else None
    result["resnet18_context"] = summarize(baseline_records, args.bootstrap, 20261104)
    result["common_backbone_pairs"] = common_backbone_comparison(
        records, baseline_records, result, result["resnet18_context"], args.bootstrap, 20261104)
    after = {key: sha256(path) for key, path in tracked.items()}
    assert before == after, "Original artifacts changed"
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    metadata = {"dataset": spec["dataset"], "architecture": "DenseNet-121",
        "weights": str(weights), "weights_url": weights.url, "weights_sha256": sha256(checkpoint),
        "gradcam_layer": "relu(features.norm5), final activated spatial tensor before global average pooling",
        "head": {"C": 1., "solver": "lbfgs", "max_iter": 1000, "n_iter": head.n_iter_.tolist(),
                 "warnings": fit_warnings},
        "source_counts": {key: len(ids) for key, ids in splits.items()},
        "seed": spec["seed"], "bootstrap_seed": 20261104, "bootstrap_resamples": args.bootstrap,
        "batch_size": args.batch, "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "torch_threads": torch.get_num_threads(), "python": sys.version, "torch": torch.__version__,
        "torchvision": torchvision.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__,
        "device": torch.cuda.get_device_name(), "script_sha256": sha256(__file__),
        "protocol_sha256": sha256(Path(__file__).with_name("PROTOCOL.md")),
        "original_files": {key: {"path": str(tracked[key]), "sha256_before": before[key],
                                  "sha256_after": after[key]} for key in tracked},
        "verification": verification, "backbone_unchanged": True,
        "heatmap_cache_layout": ["source", "black_own", "black_fixed", "gray_own", "gray_fixed", "white_own", "white_fixed"]}
    write_json(output / "metadata.json", metadata)
    write_json(output / "summary.json", result)
    del model, backbone_before
    torch.cuda.empty_cache()
    gc.collect()
    return {"dataset": spec["dataset"], "metadata": metadata, "results": result}


def make_report(results, output):
    lines = ["# R1C1: fixed DenseNet-121 architecture-transfer results", "",
        "Both prespecified datasets are reported. No architecture or metric was selected by these results.", "",
        "| Dataset | Architecture | Target | Valid pairs | Inconsistency | H-RCAM AUROC (95% CI) | AP (95% CI) |", 
        "|---|---|---|---:|---:|---:|---:|"]
    for item in results:
        result = item["results"]
        for architecture, data in (("ResNet-18 (existing)", result["resnet18_context"]), ("DenseNet-121", result)):
            for mode in ("own", "fixed"):
                subset = data[mode]
                score = subset["ranking"]["HRCAM"]
                auc = ("{:.4f} [{:.4f}, {:.4f}]".format(score["auroc"], *score["auroc_ci95"])
                       if score["auroc"] is not None else "not estimable (one outcome class)")
                ap = ("{:.4f} [{:.4f}, {:.4f}]".format(score["ap"], *score["ap_ci95"])
                      if score["ap"] is not None else "not estimable (one outcome class)")
                lines.append("| {} | {} | {} | {} | {:.2%} | {} | {} |".format(
                    item["dataset"], architecture, mode, subset["coverage"]["valid_pairs"],
                    subset["coverage"]["inconsistency_prevalence"], auc, ap))
    lines += ["", "## Component comparisons", "",
        "| Dataset | Architecture | Target | Measure | AUROC (95% CI) | AP (95% CI) |",
        "|---|---|---|---|---:|---:|"]
    for item in results:
        for architecture, data in (("ResNet-18 (existing)", item["results"]["resnet18_context"]),
                                   ("DenseNet-121", item["results"])):
            for mode in ("own", "fixed"):
                for name, score in data[mode]["ranking"].items():
                    auc = ("{:.4f} [{:.4f}, {:.4f}]".format(score["auroc"], *score["auroc_ci95"])
                           if score["auroc"] is not None else "not estimable")
                    ap = ("{:.4f} [{:.4f}, {:.4f}]".format(score["ap"], *score["ap_ci95"])
                          if score["ap"] is not None else "not estimable")
                    lines.append("| {} | {} | {} | {} | {} | {} |".format(item["dataset"], architecture, mode, name, auc, ap))
    lines += ["", "## Paired H-RCAM-minus-component differences", "",
        "Positive differences favor H-RCAM. Intervals use the same source-cluster resamples.", "",
        "| Dataset | Architecture | Target | Contrast | AUROC difference (95% CI) | AP difference (95% CI) |",
        "|---|---|---|---|---:|---:|"]
    for item in results:
        for architecture, data in (("ResNet-18 (existing)", item["results"]["resnet18_context"]),
                                   ("DenseNet-121", item["results"])):
            for mode in ("own", "fixed"):
                for contrast, values in data[mode].get("paired_differences", {}).items():
                    auc = "{:+.4f} [{:+.4f}, {:+.4f}]".format(values["auroc"]["difference"], *values["auroc"]["ci95"])
                    ap = "{:+.4f} [{:+.4f}, {:+.4f}]".format(values["ap"]["difference"], *values["ap"]["ci95"])
                    lines.append("| {} | {} | {} | {} | {} | {} |".format(item["dataset"], architecture, mode, contrast, auc, ap))
    lines += ["", "## Coverage and classifier accuracy", "",
        "| Dataset | Architecture | Target | Valid pairs | Invalid-map pairs | Mask-ineligible pairs | Clean accuracy (all sources) | Transformed accuracy (all pairs) |",
        "|---|---|---|---:|---:|---:|---:|---:|"]
    for item in results:
        for architecture, data in (("ResNet-18 (existing)", item["results"]["resnet18_context"]),
                                   ("DenseNet-121", item["results"])):
            for mode in ("own", "fixed"):
                subset = data[mode]
                count = subset["coverage"]
                lines.append("| {} | {} | {} | {} | {} | {} | {:.2%} | {:.2%} |".format(
                    item["dataset"], architecture, mode, count["valid_pairs"], count["invalid_map_pairs"],
                    count["mask_ineligible_pairs"], subset["accuracy_all_sources"], subset["transformed_accuracy_all_pairs"]))
    lines += ["", "## Common valid pairs across backbones", "",
        "The pair identities are identical within each row; each backbone retains its own prediction outcomes.", "",
        "| Dataset | Target | Common pairs | DenseNet H-RCAM AUROC | ResNet H-RCAM AUROC | DenseNet valid pairs excluded | ResNet valid pairs excluded |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for item in results:
        for mode, subset in item["results"]["common_backbone_pairs"].items():
            dense = subset["DenseNet121"]["ranking"]["HRCAM"]["auroc"]
            resnet = subset["ResNet18"]["ranking"]["HRCAM"]["auroc"]
            lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(item["dataset"], mode, subset["n_pairs"],
                "{:.4f}".format(dense) if dense is not None else "not estimable",
                "{:.4f}".format(resnet) if resnet is not None else "not estimable",
                subset["DenseNet121_excluded_valid_pairs"], subset["ResNet18_excluded_valid_pairs"]))
    lines += ["", "Full uncertainty, paired differences, validity coverage, per-background results,",
        "accuracy, and provenance are in the JSON summaries. Different backbones produce different",
        "prediction-inconsistency outcomes; their AUROCs are not classifier-accuracy comparisons.", "",
        "These are architecture-transfer results on the original development datasets, not new",
        "independent-dataset validation. One fitted head per dataset was evaluated; source-cluster",
        "intervals do not quantify training variability. The manuscript has not been changed.", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    bind_resources(require_linux=not args.self_test)
    if args.self_test:
        self_test()
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; do not silently run the experiment on CPU")
    if "5080" not in torch.cuda.get_device_name():
        raise RuntimeError("This protocol is authorized for the remote-5080 GPU")
    args.root = args.root.resolve()
    args.output = (args.output or args.root / "runs/r1c1_densenet121_20261104_v1").resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = Path(__file__).with_name("PROTOCOL.md")
    write_json(args.output / "run_status.json", {"state": "running", "started": time.time(),
        "pid": os.getpid(), "host": platform.node(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "script_sha256": sha256(__file__), "protocol_sha256": sha256(protocol),
        "prespecified_architecture": "DenseNet-121", "prespecified_datasets": list(SPECS)})
    (args.output / "PROTOCOL.md").write_bytes(protocol.read_bytes())
    (args.output / "run_architecture.py").write_bytes(Path(__file__).read_bytes())
    try:
        self_test()
        results = []
        for name, spec in SPECS.items():
            results.append(run_dataset(args, name, spec))
            write_json(args.output / "partial_results.json", results)
        make_report(results, args.output)
        write_json(args.output / "summary.json", results)
        write_json(args.output / "run_status.json", {"state": "complete", "ended": time.time(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)), "datasets": list(SPECS)})
        print("Experiment complete: " + str(args.output), flush=True)
    except Exception as exc:
        write_json(args.output / "run_status.json", {"state": "failed", "ended": time.time(), "error": repr(exc),
            "cpu_affinity": sorted(os.sched_getaffinity(0))})
        raise


if __name__ == "__main__":
    main()
