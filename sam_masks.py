"""Class-agnostic SAM masks with an explicit central-object selection rule."""
import json
import time
from pathlib import Path

import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import segment_anything.automatic_mask_generator as amg_module
from torchvision.ops.boxes import batched_nms


def cpu_nms(boxes, scores, idxs, iou_threshold):
    # The CPU torchvision wheel provides NMS; SAM inference itself runs on CUDA.
    return batched_nms(boxes.cpu(), scores.cpu(), idxs.cpu(), iou_threshold).to(boxes.device)


def generate(indices, images, output):
    cache = output / "masks.npz"
    if cache.exists():
        saved = np.load(cache)
        if not np.array_equal(saved["indices"], indices):
            raise ValueError("SAM mask cache has a different source split")
        return saved["masks"]
    root = Path(__file__).resolve().parent
    amg_module.batched_nms = cpu_nms
    model = sam_model_registry["vit_b"]()
    model.load_state_dict(torch.load(root / "assets/sam_vit_b_01ec64.pth", map_location="cpu", weights_only=True))
    model.cuda().eval()
    generator = SamAutomaticMaskGenerator(model, points_per_side=8, points_per_batch=64,
                                          pred_iou_thresh=.80, stability_score_thresh=.90,
                                          crop_n_layers=0, min_mask_region_area=0)
    masks, quality = [], []
    started = time.time()
    with torch.inference_mode():
        for position, source in enumerate(indices):
            proposals = generator.generate(images[source])
            ranked = []
            for proposal in proposals:
                mask = proposal["segmentation"]
                area = float(mask.mean())
                if not .05 <= area <= .95:
                    continue
                h, w = mask.shape
                central = float(mask[h//4:3*h//4, w//4:3*w//4].mean())
                border = float(np.concatenate([mask[0], mask[-1], mask[1:-1, 0], mask[1:-1, -1]]).mean())
                score = proposal["predicted_iou"] * proposal["stability_score"] * np.sqrt(area) * (.25+.75*central) * (1-border)
                ranked.append((float(score), proposal))
            if ranked:
                score, chosen = max(ranked, key=lambda item: item[0])
                masks.append(chosen["segmentation"])
                quality.append({"source_id": int(source), "proposal_count": len(proposals), "eligible_proposals": len(ranked),
                                "selection_score": score, "predicted_iou": float(chosen["predicted_iou"]),
                                "stability_score": float(chosen["stability_score"])})
            else:
                masks.append(np.zeros(images[source].shape[:2], bool))
                quality.append({"source_id": int(source), "proposal_count": len(proposals), "eligible_proposals": 0})
            if position % 50 == 0:
                print(f"SAM {position}/{len(indices)} elapsed={time.time()-started:.1f}s", flush=True)
    masks = np.asarray(masks)
    np.savez_compressed(cache, indices=indices, masks=masks)
    (output / "mask_quality.json").write_text(json.dumps(quality, indent=2))
    return masks
