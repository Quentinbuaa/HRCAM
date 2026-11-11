"""Independently audit completed R1C1 records, checkpoints, and sampled maps.

Uses official Torchvision forwarding, closed-form CAM, SciPy component
definitions, and independent record-to-table reconstruction. No fitting.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    approved = set(os.sched_getaffinity(0)) & set(range(24, 32))
    if not approved:
        raise RuntimeError("Approved CPUs unavailable")
    os.sched_setaffinity(0, approved)
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "4"
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import pearsonr
    from sklearn.metrics import average_precision_score, roc_auc_score
    from torchvision.models import DenseNet121_Weights, densenet121
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    assert "5080" in torch.cuda.get_device_name()
    combined = [{"dataset":label,"metadata":json.loads((args.run/name/"metadata.json").read_text()),"results":json.loads((args.run/name/"summary.json").read_text())} for name,label in [("flowers102_external_v1","Oxford Flowers-102"),("caltech101_external_v1","Caltech-101"),("voc2012_external_v1","PASCAL VOC 2012"),("uecfoodpix_external_v1","UECFoodPixComplete")]]
    audit = {"state": "running", "cpu_affinity": sorted(os.sched_getaffinity(0)), "datasets": {}}
    specs = {"Oxford Flowers-102": ("flowers102_external_v1","flowers_trainval.npz","flowers_test_selected.npz",102),
             "Caltech-101": ("caltech101_external_v1","caltech101_train.npz","caltech101_test_selected.npz",101),
             "PASCAL VOC 2012": ("voc2012_external_v1","voc2012_train.npz","voc2012_val_selected.npz",20),
             "UECFoodPixComplete": ("uecfoodpix_external_v1","uecfoodpix_train.npz","uecfoodpix_test_selected.npz",102)}
    for item in combined:
        name, training_file, testing_file, classes = specs[item["dataset"]]
        folder = args.run / name
        metadata = json.loads((folder / "metadata.json").read_text())
        summary = json.loads((folder / "summary.json").read_text())
        assert metadata == item["metadata"] and summary == item["results"]
        assert metadata["script_sha256"] == digest(args.run / "run_external.py")
        assert metadata["helper_sha256"] == digest(args.root / "runs/r1c1_densenet121_20261104_v1_retry1/run_architecture.py")
        assert set(metadata["cpu_affinity"]) <= set(range(24, 32))
        assert metadata["threads"] == 4 and metadata["backbone_unchanged"]
        for key, value in metadata["source_hashes"].items():
            assert digest(Path(metadata["source_paths"][key])) == value
        rows = [json.loads(line) for line in (folder / "external_records.jsonl").read_text().splitlines()]
        rows_by_key = {(r["source_id"], r["background"], r["cam_mode"]): r for r in rows}
        assert len(rows_by_key) == len(rows) == metadata["evaluation_sources"] * 6
        for r in rows:
            assert r["inconsistent"] == (r["original_prediction"] != r["transformed_prediction"])
            assert r["original_correct"] == (r["original_prediction"] == r["label"])
            assert r["transformed_correct"] == (r["transformed_prediction"] == r["label"])
            assert r["mask_eligible"] == (.05 <= r["mask_fraction"] <= .95)
            partner = rows_by_key[(r["source_id"], r["background"], "fixed" if r["cam_mode"] == "own" else "own")]
            assert r["original_prediction"] == partner["original_prediction"]
            assert r["transformed_prediction"] == partner["transformed_prediction"]
            if not r["inconsistent"]:
                assert r["valid_cam"] == partner["valid_cam"]
                assert r["jsd"] == partner["jsd"] and r["ncd"] == partner["ncd"]
        for mode in ("own", "fixed"):
            valid_rows = [r for r in rows if r["cam_mode"] == mode and r["mask_eligible"] and r["valid_cam"]]
            y = np.array([r["inconsistent"] for r in valid_rows])
            j = np.array([r["jsd"] for r in valid_rows])
            c = np.array([r["ncd"] for r in valid_rows])
            arrays = {"HRCAM": 2 * j * c / (j + c + 1e-12), "JSD": j, "NCD": c}
            assert summary[mode]["coverage"]["valid_pairs"] == len(y)
            assert summary[mode]["coverage"]["inconsistent_pairs"] == int(y.sum())
            assert summary[mode]["successful_resamples"] == 1000
            for metric, values in arrays.items():
                np.testing.assert_allclose(summary[mode]["ranking"][metric]["auroc"], roc_auc_score(y, values), atol=1e-15)
                np.testing.assert_allclose(summary[mode]["ranking"][metric]["ap"], average_precision_score(y, values), atol=1e-15)
            for contrast, values in summary[mode]["paired_differences"].items():
                first, second = contrast.split("_minus_")
                for metric in ("auroc", "ap"):
                    expected = summary[mode]["ranking"][first][metric] - summary[mode]["ranking"][second][metric]
                    np.testing.assert_allclose(values[metric]["difference"], expected, atol=1e-15)
        state = torch.load(folder / "classifier.pt", map_location="cpu", weights_only=True)
        converted = {key.replace("features.0.", "features.", 1): value for key, value in state.items()}
        model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1, progress=False)
        feature_tensors = 0
        for key, value in model.state_dict().items():
            if key.startswith("features."):
                assert torch.equal(value, converted[key]), key
                feature_tensors += 1
        model.classifier = torch.nn.Linear(1024, classes)
        model.load_state_dict(converted, strict=True)
        model = model.cuda().eval()
        with np.load(folder / "external_sources.npz") as source_cache:
            source_ids = source_cache["source_ids"]
            cached_labels, cached_masks = source_cache["labels"], source_cache["masks"]
        np.testing.assert_array_equal(source_ids, np.arange(metadata["evaluation_sources"]))
        with np.load(args.root / "data" / training_file) as archive:
            training_count = len(archive["labels"])
        with np.load(args.root / "data" / testing_file) as archive:
            selected = source_ids
            images = archive["images"][selected[:16]]
            np.testing.assert_array_equal(cached_labels, archive["labels"][selected])
            np.testing.assert_array_equal(cached_masks, archive["masks"][selected])
        maps = np.load(folder / "external_heatmaps.npy", mmap_mode="r")
        assert maps.shape == (len(source_ids), 7, 224, 224)

        def extract(array, target=None):
            x = torch.tensor(array.transpose(0, 3, 1, 2).copy(), device="cuda", dtype=torch.float32) / 255
            x = (x - x.new_tensor([.485, .456, .406])[None, :, None, None]) / x.new_tensor([.229, .224, .225])[None, :, None, None]
            with torch.no_grad():
                features = functional.relu(model.features(x), inplace=False)
                logits = model(x)
                predictions = logits.argmax(1)
                target = predictions if target is None else target
                cam = (features * model.classifier.weight[target][:, :, None, None]).sum(1, keepdim=True).relu()
                cam = functional.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[:, 0]
                cam -= cam.amin((1, 2), keepdim=True)
                cam /= cam.amax((1, 2), keepdim=True).clamp_min(1e-12)
            return predictions, cam.cpu().numpy()

        pred0, source_maps = extract(images)
        map_errors = [float(np.max(np.abs(source_maps - maps[:16, 0])))]
        jsd_errors, ncd_errors = [], []
        for bg_index, (background, color) in enumerate((("black", 0), ("gray", 128), ("white", 255))):
            masks = cached_masks[:16].astype(bool)
            altered = np.where(masks[:, :, :, None], images, color).astype(np.uint8)
            assert np.array_equal(altered[masks], images[masks])
            for mode_index, mode in enumerate(("own", "fixed")):
                pred1, replay = extract(altered, None if mode == "own" else pred0)
                saved = maps[:16, 1 + bg_index * 2 + mode_index]
                map_errors.append(float(np.max(np.abs(replay - saved))))
                np.testing.assert_allclose(replay, saved, rtol=2e-4, atol=3e-5)
                for i, source_id in enumerate(source_ids[:16]):
                    row = rows_by_key[(int(source_id), background, mode)]
                    assert row["original_prediction"] == int(pred0[i])
                    assert row["transformed_prediction"] == int(pred1[i])
                    if not row["valid_cam"]:
                        continue
                    a = np.asarray(maps[i, 0], dtype=np.float64).ravel()
                    b = np.asarray(saved[i], dtype=np.float64).ravel()
                    p, q = a + 1e-8, b + 1e-8
                    expected_j = float(jensenshannon(p / p.sum(), q / q.sum(), base=2) ** 2)
                    expected_c = float((1 - pearsonr(a, b).statistic) / 2)
                    jsd_errors.append(abs(expected_j - row["jsd"]))
                    ncd_errors.append(abs(expected_c - row["ncd"]))
                    assert jsd_errors[-1] < 3e-14 and ncd_errors[-1] < 3e-14
        assert max(map_errors) < 3e-5
        audit["datasets"][name] = {"state": "passed", "source_count": len(source_ids),
            "record_count": len(rows), "official_pretrained_feature_tensors_equal": feature_tensors,
            "replayed_sources": len(images), "replayed_pairs_per_target": len(images) * 3,
            "maximum_closed_form_map_error": max(map_errors),
            "maximum_independent_jsd_error": max(jsd_errors),
            "maximum_independent_ncd_error": max(ncd_errors),
            "all_point_estimates_reproduced": True, "consistent_target_scores_identical": True,
            "original_provenance_hashes_unchanged": True}
        del model, maps
        torch.cuda.empty_cache()
        print(json.dumps(audit["datasets"][name]), flush=True)
    audit["state"] = "passed"
    audit["audit_script_sha256"] = digest(Path(__file__))
    (args.run / "INDEPENDENT_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("Independent audit passed", flush=True)


if __name__ == "__main__":
    main()
