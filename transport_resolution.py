"""Outcome-independent resolution sensitivity on the first 16 eligible dev sources."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import spearmanr

from transport_metrics import distances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    archive = np.load(args.run_dir / "development_heatmaps.npz")
    ids, originals, maps, masks = (archive[k] for k in ["source_ids", "original_maps", "maps", "masks"])
    raw = [json.loads(s) for s in (args.run_dir / "development_records.jsonl").read_text().splitlines() if s]
    eligible = {r["source_id"] for r in raw if r["cam_mode"] == "fixed" and r["mask_eligible"] and r["valid_cam"]}
    chosen = [i for i, sid in enumerate(ids) if sid in eligible][:16]
    results = []
    for count, i in enumerate(chosen):
        for b in range(3):
            scores = {}
            for grid in [7, 14, 28]:
                start = time.perf_counter()
                scores[grid] = distances(originals[i], maps[i, b, 1], masks[i], grid)
                scores[grid]["seconds"] = time.perf_counter()-start
            results.append({"source_id": int(ids[i]), "background": ["black", "gray", "white"][b], "grids": scores})
        print(f"resolution {count+1}/{len(chosen)} sources", flush=True)
    summary = {"scope": "Development descriptive sensitivity only; first 16 eligible sources, fixed CAM target",
               "sources": [int(ids[i]) for i in chosen], "n_pairs": len(results), "metrics": {},
               "timing": {g: float(np.median([r["grids"][g]["seconds"] for r in results])) for g in [7, 14, 28]}}
    for name in ["spatial_w1", "BAT"]:
        summary["metrics"][name] = {}
        for a, b in [(7, 14), (14, 28)]:
            x = np.array([r["grids"][a][name] for r in results])
            y = np.array([r["grids"][b][name] for r in results])
            summary["metrics"][name][f"{a}_versus_{b}"] = {
                "spearman": float(spearmanr(x, y).statistic),
                "mean_absolute_difference": float(np.mean(abs(x-y))),
                "max_absolute_difference": float(np.max(abs(x-y)))}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output_dir / "records.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
