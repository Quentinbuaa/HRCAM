"""Compare saved model predictions/CAMs across existing Python environments."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from experiment import GradCAM, classifier, seed_all, tensors

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    seed_all(20260911)
    model = classifier()
    data = np.load(ROOT / "data/cifar10_test.npz")
    x = tensors(data["images"][:32])
    cam = GradCAM(model)
    cam(x)
    torch.cuda.synchronize()
    start = time.perf_counter()
    logits, own, fixed = cam(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    out = ROOT / "environment_checks"
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / f"{args.tag}.npz", logits=logits.cpu().numpy(), cam=own)
    report = {"torch": torch.__version__, "cuda": torch.version.cuda,
              "elapsed_seconds_32_examples": elapsed, "device": torch.cuda.get_device_name(0)}
    (out / f"{args.tag}.json").write_text(json.dumps(report, indent=2))
    if (out / "py310.npz").exists() and (out / "py314.npz").exists():
        a, b = np.load(out / "py310.npz"), np.load(out / "py314.npz")
        comparison = {"max_abs_logit_difference": float(np.max(np.abs(a["logits"]-b["logits"]))),
                      "max_abs_cam_difference": float(np.max(np.abs(a["cam"]-b["cam"]))),
                      "same_predictions": bool(np.array_equal(a["logits"].argmax(1), b["logits"].argmax(1)))}
        (out / "comparison.json").write_text(json.dumps(comparison, indent=2))
        print(json.dumps(comparison), flush=True)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
