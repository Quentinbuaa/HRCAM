"""CIFAR-10 background perturbation and class-controlled Grad-CAM experiment."""
import argparse
import hashlib
import importlib.util
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from metrics import heatmap_metrics

ROOT = Path(__file__).resolve().parent
MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.201]
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def vendor(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"vendor/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False


def classifier():
    model = vendor("resnet").resnet18()
    model.maxpool = torch.nn.Identity()
    state = torch.load(ROOT / "assets/classifier/pytorch_model.bin", map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    return model.cuda().eval()


def tensors(images):
    return torch.from_numpy(np.ascontiguousarray(images.transpose(0, 3, 1, 2))).float().cuda() / 255


def normalize(x):
    return (x - x.new_tensor(MEAN)[None, :, None, None]) / x.new_tensor(STD)[None, :, None, None]


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.activation = None
        self.handle = model.layer4.register_forward_hook(self.capture)

    def capture(self, module, inputs, output):
        self.activation = output

    def __call__(self, images, fixed_targets=None):
        with torch.enable_grad():
            logits = self.model(normalize(images))
            own = logits.argmax(1)
            targets = own if fixed_targets is None else fixed_targets
            modes = [own, targets]
            maps = []
            for index, target in enumerate(modes):
                objective = logits.gather(1, target[:, None]).sum()
                gradient = torch.autograd.grad(objective, self.activation, retain_graph=index == 0)[0]
                cam = (gradient.mean(dim=(2, 3), keepdim=True) * self.activation).sum(1, keepdim=True).relu()
                cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
                cam = cam - cam.amin(dim=(1, 2), keepdim=True)
                cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)
                maps.append(cam.detach().cpu().numpy())
            return logits.detach(), maps[0], maps[1]

    def close(self):
        self.handle.remove()


def make_splits(labels, n, seed):
    if n % 10:
        raise ValueError("Each split size must be divisible by ten")
    rng = np.random.default_rng(seed)
    result = {role: [] for role in ["calibration", "development", "holdout"]}
    for label in range(10):
        shuffled = rng.permutation(np.flatnonzero(labels == label))
        if 3 * n // 10 > len(shuffled):
            raise ValueError("Insufficient samples")
        for i, role in enumerate(result):
            result[role].extend(shuffled[i*n//10:(i+1)*n//10].tolist())
    for role in result:
        result[role] = rng.permutation(result[role]).tolist()
    assert len(set(sum(result.values(), []))) == 3 * n
    return result


def full_accuracy(model, data, batch=256):
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(data["labels"]), batch):
            predictions.extend(model(normalize(tensors(data["images"][start:start+batch]))).argmax(1).cpu().tolist())
    return float(np.mean(np.asarray(predictions) == data["labels"]))


def masks_for(indices, images, output, batch=4):
    cache = output / "masks.npz"
    if cache.exists():
        saved = np.load(cache)
        if not np.array_equal(saved["indices"], indices):
            raise ValueError("Existing mask indices do not match")
        return saved["masks"]
    net = vendor("u2net").U2NET(3, 1)
    net.load_state_dict(torch.load(ROOT / "assets/u2net.pth", map_location="cpu", weights_only=True))
    net.cuda().eval()
    masks = []
    started = time.time()
    with torch.inference_mode():
        for start in range(0, len(indices), batch):
            # Match the upstream ToTensorLab flag=0 image/max normalization.
            x = tensors(images[np.asarray(indices[start:start+batch])])
            x = F.interpolate(x, (320, 320), mode="bilinear", align_corners=False)
            x = x / x.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
            x = (x - x.new_tensor([.485, .456, .406])[None, :, None, None]) / x.new_tensor([.229, .224, .225])[None, :, None, None]
            y = net(x)[0]
            y = y - y.amin(dim=(2, 3), keepdim=True)
            y = y / y.amax(dim=(2, 3), keepdim=True).clamp_min(1e-8)
            y = F.interpolate(y, (32, 32), mode="bilinear", align_corners=False)[:, 0]
            masks.extend((y > .5).cpu().numpy())
            if start % 100 == 0:
                print(f"masks {start}/{len(indices)} elapsed={time.time()-started:.1f}s", flush=True)
    masks = np.asarray(masks)
    np.savez_compressed(cache, indices=indices, masks=masks)
    del net
    torch.cuda.empty_cache()
    return masks


def contact_sheet(indices, images, labels, masks, output):
    selected = []
    for label in range(10):
        selected.extend([i for i, source in enumerate(indices) if labels[source] == label][:4])
    sheet = Image.new("RGB", (4 * 330, 10 * 110), "white")
    draw = ImageDraw.Draw(sheet)
    for count, position in enumerate(selected):
        source = indices[position]
        original = images[source]
        altered = np.where(masks[position, :, :, None], original, 128).astype(np.uint8)
        mask_rgb = np.repeat(masks[position, :, :, None], 3, axis=2).astype(np.uint8) * 255
        x, y = (count % 4) * 330, (count // 4) * 110
        for column, arr in enumerate([original, mask_rgb, altered]):
            sheet.paste(Image.fromarray(arr).resize((96, 80), Image.Resampling.NEAREST), (x+column*108, y))
        draw.text((x, y+81), f"{source} {CLASSES[labels[source]]} area={masks[position].mean():.2f}", fill="black")
    sheet.save(output / "mask_qa.png")


def collect(role, indices, all_indices, masks, data, model, output, batch):
    target_file = output / f"{role}_records.jsonl"
    if target_file.exists():
        raise FileExistsError(f"Refusing to replace completed records: {target_file}")
    mask_map = dict(zip(all_indices, masks))
    cam = GradCAM(model)
    records = []
    heatmaps = []
    original_heatmaps = []
    source_ids = []
    foreground_masks = []
    started = time.time()
    for start in range(0, len(indices), batch):
        ids = indices[start:start+batch]
        array = data["images"][ids]
        foreground = np.asarray([mask_map[i] for i in ids])
        original_logits, original_maps, _ = cam(tensors(array))
        original_predictions = original_logits.argmax(1)
        prob0 = original_logits.softmax(1).cpu().numpy()
        by_bg = []
        for background, color in [("black", 0), ("gray", 128), ("white", 255)]:
            altered = np.where(foreground[:, :, :, None], array, color).astype(np.uint8)
            assert np.array_equal(altered[foreground], array[foreground])
            logits, own_maps, fixed_maps = cam(tensors(altered), original_predictions)
            probabilities = logits.softmax(1).cpu().numpy()
            predictions = logits.argmax(1).cpu().numpy()
            by_bg.append(np.stack([own_maps, fixed_maps], axis=1))
            for i, source in enumerate(ids):
                p, q = prob0[i].astype(float), probabilities[i].astype(float)
                p, q = np.maximum(p, 1e-12), np.maximum(q, 1e-12)
                p, q = p/p.sum(), q/q.sum()
                m = (p+q)/2
                for mode, h in [("own", own_maps[i]), ("fixed", fixed_maps[i])]:
                    row = {"source_id": int(source), "role": role, "background": background, "cam_mode": mode,
                           "label": int(data["labels"][source]), "original_prediction": int(original_predictions[i]),
                           "transformed_prediction": int(predictions[i]), "mask_fraction": float(foreground[i].mean()),
                           "mask_eligible": bool(.05 <= foreground[i].mean() <= .95),
                           "original_correct": bool(int(original_predictions[i]) == data["labels"][source]),
                           "transformed_correct": bool(predictions[i] == data["labels"][source]),
                           "inconsistent": bool(int(original_predictions[i]) != predictions[i]),
                           "original_confidence": float(p[int(original_predictions[i])]),
                           "confidence_drop": float(p[int(original_predictions[i])] - q[int(original_predictions[i])]),
                           "prediction_jsd": float((np.sum(p*np.log(p/m))+np.sum(q*np.log(q/m)))/(2*np.log(2)))}
                    row.update(heatmap_metrics(original_maps[i], h, foreground[i]))
                    row = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in row.items()}
                    records.append(row)
        heatmaps.append(np.stack(by_bg, axis=1))
        original_heatmaps.append(original_maps)
        source_ids.extend(ids)
        foreground_masks.extend(foreground)
        if start % (batch * 5) == 0:
            print(f"{role} {start}/{len(indices)} elapsed={time.time()-started:.1f}s", flush=True)
    cam.close()
    # Store records only after the entire role finishes successfully.
    target_file.write_text("".join(json.dumps(r, allow_nan=False)+"\n" for r in records))
    np.savez_compressed(output / f"{role}_heatmaps.npz", source_ids=source_ids,
                        maps=np.concatenate(heatmaps), original_maps=np.concatenate(original_heatmaps),
                        masks=np.asarray(foreground_masks))
    print(f"Saved {len(records)} {role} records", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="cifar10_v1")
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--stage", choices=["prepare", "development", "holdout"], required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--mask-method", choices=["u2net", "sam"], default="u2net")
    args = parser.parse_args()
    seed_all(args.seed)
    output = ROOT / "runs" / args.run
    output.mkdir(parents=True, exist_ok=True)
    data = np.load(ROOT / "data/cifar10_test.npz")
    splits = make_splits(data["labels"], args.size, args.seed)
    split_path = output / "splits.json"
    if split_path.exists() and json.loads(split_path.read_text()) != splits:
        raise ValueError("Run already has a different split")
    split_path.write_text(json.dumps(splits, indent=2))
    if args.stage == "prepare":
        model = classifier()
        accuracy = full_accuracy(model, data)
        print(f"Full test accuracy: {accuracy:.4%}", flush=True)
        metadata = {"arguments": vars(args), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                    "clean_test_accuracy": accuracy, "n_clean_test": len(data["labels"]),
                    "checkpoint_sha256": hashlib.sha256((ROOT / "assets/classifier/pytorch_model.bin").read_bytes()).hexdigest(),
                    "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob("*.py")}}
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
        if accuracy < .90:
            raise RuntimeError("Clean accuracy gate failed; stop and inspect classifier")
        del model
        torch.cuda.empty_cache()
        all_ids = sum(splits.values(), [])
        if args.mask_method == "sam":
            from sam_masks import generate
            masks = generate(all_ids, data["images"], output)
        else:
            masks = masks_for(all_ids, data["images"], output)
        contact_sheet(all_ids, data["images"], data["labels"], masks, output)
    else:
        if args.stage == "holdout" and not (output / "frozen_analysis.json").exists():
            raise RuntimeError("Freeze development selection before opening holdout")
        saved = np.load(output / "masks.npz")
        model = classifier()
        for role in (["calibration", "development"] if args.stage == "development" else ["holdout"]):
            collect(role, splits[role], saved["indices"], saved["masks"], data, model, output, args.batch)


if __name__ == "__main__":
    main()
