"""Audit the downloaded TrashCan-Material COCO annotations."""

import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MATERIAL = ROOT / "data/trashcan/dataset/material_version"


def video_id(filename: str) -> str:
    return filename.split("_frame", 1)[0]


def main() -> None:
    video_roles = defaultdict(set)
    for role in ("train", "val"):
        path = MATERIAL / f"instances_{role}_trashcan.json"
        data = json.loads(path.read_text())
        categories = {row["id"]: row["name"] for row in data["categories"]}
        images = {row["id"]: row for row in data["images"]}
        counts = Counter(row["category_id"] for row in data["annotations"])
        image_counts = defaultdict(set)
        class_videos = defaultdict(set)
        videos = set()
        for row in data["annotations"]:
            image_counts[row["category_id"]].add(row["image_id"])
            class_videos[row["category_id"]].add(video_id(images[row["image_id"]]["file_name"]))
        for row in images.values():
            vid = video_id(row["file_name"])
            videos.add(vid)
            video_roles[vid].add(role)
        print(f"{role}: {len(images)} images; {len(data['annotations'])} objects; "
              f"{len(videos)} videos")
        for cid in sorted(categories):
            print(f"  {cid:2d} {categories[cid]:20s} "
                  f"objects={counts[cid]:5d} images={len(image_counts[cid]):5d} "
                  f"videos={len(class_videos[cid]):4d}")
        for row in list(images.values())[:3]:
            print("  example:", row)
        for row in data["annotations"][:2]:
            print("  annotation:", {k: row.get(k) for k in
                                   ("id", "image_id", "category_id", "bbox", "area", "iscrowd")})
            print("  segmentation type:", type(row.get("segmentation")).__name__)
    overlap = [vid for vid, roles in video_roles.items() if len(roles) > 1]
    print(f"Video overlap across published train/val: {len(overlap)}")
    print("Overlapping video examples:", overlap[:10])


if __name__ == "__main__":
    main()
