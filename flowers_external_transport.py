"""Optional fixed BAT/SpatialW1 baselines for external pairs."""
import argparse
import json
from pathlib import Path

import numpy as np

from transport_metrics import distances

ROOT = Path(__file__).resolve().parent
BACKGROUNDS = ("black", "gray", "white")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="flowers102_external_v1")
    parser.add_argument("--expected-sources", type=int, default=1020)
    args = parser.parse_args()
    folder = ROOT / "runs" / args.run
    target = folder / "external_transport.jsonl"
    rows = [json.loads(line) for line in (folder / "external_records.jsonl").read_text().splitlines() if line]
    rows = [r for r in rows if r["cam_mode"] == "own" and r["valid_cam"] and r["mask_eligible"]]
    rows.sort(key=lambda r: (r["source_id"], r["background"]))
    assert rows and len(rows) <= args.expected_sources * 3
    done = {}
    if target.exists():
        for line in target.read_text().splitlines():
            if line:
                item = json.loads(line)
                key = (item["source_id"], item["background"])
                if key in done:
                    raise ValueError(f"Duplicate transport record {key}")
                done[key] = item
    archive = np.load(folder / "external_heatmaps.npz")
    ids = archive["source_ids"]
    positions = {int(value): pos for pos, value in enumerate(ids)}
    if len(positions) != args.expected_sources:
        raise ValueError("Missing or duplicate source heatmaps")
    original = archive["original_maps"]
    changed = archive["maps"]
    masks = archive["masks"]
    with target.open("a") as stream:
        for count, row in enumerate(rows, start=1):
            key = (row["source_id"], row["background"])
            if key in done:
                continue
            pos = positions[row["source_id"]]
            bg = BACKGROUNDS.index(row["background"])
            values = distances(original[pos], changed[pos, bg, 0], masks[pos], grid=14)
            item = {"source_id": row["source_id"], "background": row["background"],
                    "BAT": values["BAT"], "SpatialW1": values["spatial_w1"]}
            stream.write(json.dumps(item) + "\n")
            stream.flush()
            done[key] = item
            if count % 100 == 0:
                print(f"transport {count}/{len(rows)}", flush=True)
    if len(done) != len(rows):
        raise ValueError(f"Incomplete transport records: {len(done)}/{len(rows)}")
    print(f"Complete {len(done)} fixed-transport baselines", flush=True)


if __name__ == "__main__":
    main()
