"""Experiment: MedSAM3 refinement strategies for cardiomegaly.

Tests approaches to fix "heart prompt → lung mask" failure mode.
Saves overlay images to scripts/results/ for visual comparison.

Approaches tested:
1. Baseline: text prompt "heart" / "cardiomegaly"
2. Alternative text prompts: "cardiac silhouette", "mediastinum", etc.
3. Box prompts: positive heart box + negative lung box (Sam3 native)
4. Mask-out: blank lung pixels then re-segment
5. Multi-mask ranking: get all candidates, filter by anatomy
"""

import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.medsam3_tool import MedSAM3Tool
from tools.overlay import SEGMENT_COLORS, overlay_mask_on_image, overlay_multiple_masks

# --- Config ---
SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "fixtures", "sample_cxr_vqa")
IMAGE_PATH = os.path.join(SAMPLE_DIR, "cardiomegaly.png")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "medsam3-merged")

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_image():
    img = Image.open(IMAGE_PATH).convert("RGB")
    return np.array(img)


def save_overlay(img_np, masks, title, filename):
    """Save a multi-mask overlay with title."""
    if not masks:
        # Save original with "no masks" label
        pil = Image.fromarray(img_np)
        draw = ImageDraw.Draw(pil)
        draw.text((10, 10), f"{title}: NO MASKS", fill=(255, 0, 0))
        pil.save(os.path.join(RESULTS_DIR, filename))
        print(f"  Saved {filename} (no masks)")
        return

    pairs = []
    for i, m in enumerate(masks):
        concept = m.get("concept", f"mask_{i}")
        score = m.get("score", 0)
        label = f"{concept} ({score:.2f})"
        pairs.append((m["mask"], label))

    overlay = overlay_multiple_masks(img_np, pairs)
    # Add title text
    draw = ImageDraw.Draw(overlay)
    draw.text((10, 10), title, fill=(255, 255, 255))
    overlay.save(os.path.join(RESULTS_DIR, filename))
    print(f"  Saved {filename} ({len(masks)} masks)")


def save_single_mask_overlay(img_np, mask_dict, title, filename):
    """Save single mask overlay with bbox and info."""
    mask = mask_dict["mask"]
    concept = mask_dict.get("concept", "unknown")
    score = mask_dict.get("score", 0)
    bbox = mask_dict.get("bbox", (0, 0, 0, 0))

    overlay = overlay_mask_on_image(img_np, mask, title=f"{concept} ({score:.2f})")
    draw = ImageDraw.Draw(overlay)
    draw.text((10, 10), title, fill=(255, 255, 255))

    # Draw bbox
    x0, y0, x1, y1 = bbox
    draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 0), width=3)

    # Add mask stats
    area = int(np.sum(mask > 0))
    h, w = mask.shape[:2]
    area_pct = area / (h * w) * 100
    ys, xs = np.where(mask > 0)
    if len(xs) > 0:
        cx, cy = int(xs.mean()), int(ys.mean())
        cx_pct = cx / w * 100
        cy_pct = cy / h * 100
        info = f"area={area_pct:.1f}% centroid=({cx_pct:.0f}%,{cy_pct:.0f}%)"
    else:
        info = "empty mask"
    draw.text((10, 30), info, fill=(255, 255, 255))

    overlay.save(os.path.join(RESULTS_DIR, filename))
    print(f"  Saved {filename}: {concept} score={score:.3f} {info}")


def mask_iou(m1, m2):
    """Compute IoU between two binary masks."""
    m1b = m1 > 0
    m2b = m2 > 0
    intersection = np.sum(m1b & m2b)
    union = np.sum(m1b | m2b)
    return intersection / union if union > 0 else 0


def mask_centroid_pct(mask):
    """Return centroid as (x%, y%) of image size."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (50, 50)
    h, w = mask.shape[:2]
    return (xs.mean() / w * 100, ys.mean() / h * 100)


def mask_area_pct(mask):
    """Return mask area as percentage of image area."""
    return np.sum(mask > 0) / (mask.shape[0] * mask.shape[1]) * 100


# ======================================
# Experiment 1: Baseline text prompts
# ======================================
def experiment_1_baseline(medsam3, img_np):
    """Test basic text prompts for heart/cardiomegaly."""
    print("\n=== Experiment 1: Baseline text prompts ===")

    prompts = ["heart", "cardiomegaly", "cardiac silhouette"]
    for prompt in prompts:
        t0 = time.perf_counter()
        results = medsam3.segment_concepts(
            img_np, [prompt],
            threshold=0.3,
            mask_threshold=0.3,
            max_masks_per_concept=5,
        )
        dt = time.perf_counter() - t0
        print(f"\n  Prompt: '{prompt}' → {len(results)} masks ({dt:.1f}s)")

        save_overlay(img_np, results, f"Exp1: '{prompt}'", f"exp1_{prompt.replace(' ', '_')}.png")
        for i, r in enumerate(results):
            save_single_mask_overlay(
                img_np, r,
                f"'{prompt}' mask {i+1}",
                f"exp1_{prompt.replace(' ', '_')}_mask{i+1}.png",
            )

    return results  # return last for further use


# ======================================
# Experiment 2: More alternative prompts
# ======================================
def experiment_2_alt_prompts(medsam3, img_np):
    """Try broader set of text prompts."""
    print("\n=== Experiment 2: Alternative text prompts ===")

    prompts = [
        "heart shadow",
        "enlarged heart",
        "mediastinum",
        "cardiac",
        "left ventricle",
        "cardiac enlargement",
    ]

    all_results = {}
    for prompt in prompts:
        t0 = time.perf_counter()
        results = medsam3.segment_concepts(
            img_np, [prompt],
            threshold=0.3,
            mask_threshold=0.3,
            max_masks_per_concept=5,
        )
        dt = time.perf_counter() - t0
        all_results[prompt] = results
        print(f"\n  Prompt: '{prompt}' → {len(results)} masks ({dt:.1f}s)")

        save_overlay(img_np, results, f"Exp2: '{prompt}'", f"exp2_{prompt.replace(' ', '_')}.png")
        for i, r in enumerate(results[:3]):  # max 3 per prompt
            save_single_mask_overlay(
                img_np, r,
                f"'{prompt}' mask {i+1}",
                f"exp2_{prompt.replace(' ', '_')}_mask{i+1}.png",
            )

    return all_results


# ======================================
# Experiment 3: Segment lungs first, then use as negative context
# ======================================
def experiment_3_lung_negative(medsam3, img_np):
    """Segment lungs, then try to segment heart while masking out lungs."""
    print("\n=== Experiment 3: Lung mask → mask-out → re-segment ===")

    # Step 1: Get lung masks
    lung_results = medsam3.segment_concepts(
        img_np, ["lung", "left lung", "right lung"],
        threshold=0.3,
        mask_threshold=0.3,
        max_masks_per_concept=3,
    )
    print(f"  Found {len(lung_results)} lung masks")
    save_overlay(img_np, lung_results, "Exp3: Lung masks", "exp3_lung_masks.png")

    if not lung_results:
        print("  No lung masks found, skipping")
        return

    # Combine all lung masks into one
    combined_lung = np.zeros(img_np.shape[:2], dtype=np.uint8)
    for lr in lung_results:
        combined_lung = np.maximum(combined_lung, lr["mask"])

    # Step 2: Mask out lung regions (approach 4A: set to median intensity)
    median_val = int(np.median(img_np[combined_lung == 0]))
    img_masked = img_np.copy()
    img_masked[combined_lung > 0] = median_val

    # Save masked image
    Image.fromarray(img_masked).save(os.path.join(RESULTS_DIR, "exp3_masked_image.png"))
    print(f"  Saved masked image (lung pixels → median={median_val})")

    # Step 3: Segment heart on masked image
    heart_prompts = ["heart", "cardiomegaly", "cardiac silhouette"]
    for prompt in heart_prompts:
        results = medsam3.segment_concepts(
            img_masked, [prompt],
            threshold=0.3,
            mask_threshold=0.3,
            max_masks_per_concept=5,
        )
        print(f"  Masked image + '{prompt}' → {len(results)} masks")

        # Show on ORIGINAL image for comparison
        save_overlay(img_np, results, f"Exp3 masked+'{prompt}'", f"exp3_masked_{prompt.replace(' ', '_')}.png")
        for i, r in enumerate(results[:3]):
            save_single_mask_overlay(
                img_np, r,
                f"Masked+'{prompt}' mask {i+1}",
                f"exp3_masked_{prompt.replace(' ', '_')}_mask{i+1}.png",
            )

    # Step 4: Also try mediastinum crop approach (4B)
    # Find mediastinum band = region between lungs
    ys_lung, xs_lung = np.where(combined_lung > 0)
    if len(xs_lung) > 0:
        h, w = img_np.shape[:2]
        # Mediastinum: center band between lung edges, lower half
        left_edge = int(np.percentile(xs_lung, 30))
        right_edge = int(np.percentile(xs_lung, 70))
        top = h // 3  # lower 2/3
        bottom = h

        # Crop and segment
        crop = img_np[top:bottom, left_edge:right_edge].copy()
        if crop.shape[0] > 50 and crop.shape[1] > 50:
            results_crop = medsam3.segment_concepts(
                crop, ["heart"],
                threshold=0.2,
                mask_threshold=0.2,
                max_masks_per_concept=5,
            )
            print(f"  Mediastinum crop + 'heart' → {len(results_crop)} masks")

            # Map back to full image
            for r in results_crop:
                full_mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
                m = r["mask"]
                if m.ndim == 3:
                    m = m[:, :, 0]
                full_mask[top:top+m.shape[0], left_edge:left_edge+m.shape[1]] = m
                r["mask"] = full_mask
                x0, y0, x1, y1 = r["bbox"]
                r["bbox"] = (x0 + left_edge, y0 + top, x1 + left_edge, y1 + top)

            save_overlay(img_np, results_crop, "Exp3: Mediastinum crop", "exp3_mediastinum_crop.png")

    return lung_results


# ======================================
# Experiment 4: Box prompts (Sam3 native)
# ======================================
def experiment_4_box_prompts(medsam3, img_np, lung_masks=None):
    """Use Sam3's native box prompt with positive/negative labels."""
    print("\n=== Experiment 4: Box prompts (positive + negative) ===")

    h, w = img_np.shape[:2]

    # Ground truth heart bbox from samples.json (approx)
    # cardiomegaly bbox: [691, 1375, 1653, 1831] in original coords
    heart_box = [691, 1375, 1653, 1831]  # x0, y0, x1, y1

    # If we have lung masks, compute lung bbox for negative
    lung_boxes = []
    if lung_masks:
        for lm in lung_masks:
            ys, xs = np.where(lm["mask"] > 0)
            if len(xs) > 0:
                lung_boxes.append([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])

    # Use Sam3 processor directly with box prompts
    processor = medsam3._concept_processor
    model = medsam3._concept_model
    device = medsam3.device

    # Preprocess image
    from PIL import Image as PILImage
    pil_img = PILImage.fromarray(img_np)

    # 4A: Positive heart box only
    print(f"\n  4A: Positive heart box {heart_box}")
    img_inputs = processor(images=pil_img, return_tensors="pt")
    img_inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in img_inputs.items()}

    with torch.no_grad():
        vision_embeds = model.get_vision_features(pixel_values=img_inputs["pixel_values"])

    original_sizes = img_inputs.get("original_sizes")
    target_sizes = original_sizes.tolist() if original_sizes is not None else [[h, w]]

    # Box prompt: [image_level[box_coords]] with label [image_level[label]]
    # input_boxes: 3 levels = [batch, num_boxes, 4]
    # input_boxes_labels: 2 levels = [batch, num_boxes]
    box_inputs = processor(
        input_boxes=[[[heart_box[0], heart_box[1], heart_box[2], heart_box[3]]]],
        input_boxes_labels=[[1]],
        original_sizes=[[h, w]],
        return_tensors="pt",
    )
    box_inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in box_inputs.items()}

    with torch.no_grad():
        outputs = model(vision_embeds=vision_embeds, **box_inputs)

    post = processor.post_process_instance_segmentation(
        outputs, threshold=0.3, mask_threshold=0.3, target_sizes=target_sizes,
    )
    if post and post[0].get("masks") is not None:
        masks = post[0]["masks"]
        scores = post[0].get("scores", [])
        results_4a = []
        for j in range(len(masks)):
            m = masks[j].cpu().numpy().astype(np.uint8) if isinstance(masks[j], torch.Tensor) else np.array(masks[j], dtype=np.uint8)
            if m.ndim == 3:
                m = m[0]
            s = float(scores[j]) if j < len(scores) else 0
            bbox_tensor = post[0].get("boxes", [])
            b = tuple(int(x) for x in bbox_tensor[j].tolist()) if j < len(bbox_tensor) else (0, 0, 0, 0)
            results_4a.append({"mask": m, "score": s, "bbox": b, "concept": "box_heart"})
        print(f"  → {len(results_4a)} masks")
        save_overlay(img_np, results_4a, "Exp4A: Heart box only", "exp4a_heart_box.png")
        for i, r in enumerate(results_4a[:3]):
            save_single_mask_overlay(img_np, r, f"Heart box mask {i+1}", f"exp4a_heart_box_mask{i+1}.png")
    else:
        print("  → No masks from box prompt")
        results_4a = []

    # 4B: Positive heart box + negative lung boxes
    if lung_boxes:
        print(f"\n  4B: Heart box + {len(lung_boxes)} negative lung boxes")
        all_boxes = [heart_box] + lung_boxes
        all_labels = [1] + [0] * len(lung_boxes)

        box_inputs_neg = processor(
            input_boxes=[all_boxes],
            input_boxes_labels=[all_labels],
            original_sizes=[[h, w]],
            return_tensors="pt",
        )
        box_inputs_neg = {k: v.to(device) if hasattr(v, "to") else v for k, v in box_inputs_neg.items()}

        with torch.no_grad():
            outputs_neg = model(vision_embeds=vision_embeds, **box_inputs_neg)

        post_neg = processor.post_process_instance_segmentation(
            outputs_neg, threshold=0.3, mask_threshold=0.3, target_sizes=target_sizes,
        )
        if post_neg and post_neg[0].get("masks") is not None:
            masks = post_neg[0]["masks"]
            scores = post_neg[0].get("scores", [])
            results_4b = []
            for j in range(len(masks)):
                m = masks[j].cpu().numpy().astype(np.uint8) if isinstance(masks[j], torch.Tensor) else np.array(masks[j], dtype=np.uint8)
                if m.ndim == 3:
                    m = m[0]
                s = float(scores[j]) if j < len(scores) else 0
                bbox_tensor = post_neg[0].get("boxes", [])
                b = tuple(int(x) for x in bbox_tensor[j].tolist()) if j < len(bbox_tensor) else (0, 0, 0, 0)
                results_4b.append({"mask": m, "score": s, "bbox": b, "concept": "box_heart-lung_neg"})
            print(f"  → {len(results_4b)} masks")
            save_overlay(img_np, results_4b, "Exp4B: Heart box + neg lung", "exp4b_heart_neg_lung.png")
            for i, r in enumerate(results_4b[:3]):
                save_single_mask_overlay(img_np, r, f"Heart+neg mask {i+1}", f"exp4b_neg_mask{i+1}.png")
        else:
            print("  → No masks from box+neg prompt")

    # 4C: Text + box combined
    print(f"\n  4C: Text 'heart' + heart box combined")
    text_inputs = processor(text="heart", return_tensors="pt")
    text_inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in text_inputs.items()}

    # Merge text + box inputs
    combined = {**text_inputs}
    combined["input_boxes"] = box_inputs["input_boxes"]
    combined["input_boxes_labels"] = box_inputs["input_boxes_labels"]

    with torch.no_grad():
        outputs_combined = model(vision_embeds=vision_embeds, **combined)

    post_comb = processor.post_process_instance_segmentation(
        outputs_combined, threshold=0.3, mask_threshold=0.3, target_sizes=target_sizes,
    )
    if post_comb and post_comb[0].get("masks") is not None:
        masks = post_comb[0]["masks"]
        scores = post_comb[0].get("scores", [])
        results_4c = []
        for j in range(len(masks)):
            m = masks[j].cpu().numpy().astype(np.uint8) if isinstance(masks[j], torch.Tensor) else np.array(masks[j], dtype=np.uint8)
            if m.ndim == 3:
                m = m[0]
            s = float(scores[j]) if j < len(scores) else 0
            bbox_tensor = post_comb[0].get("boxes", [])
            b = tuple(int(x) for x in bbox_tensor[j].tolist()) if j < len(bbox_tensor) else (0, 0, 0, 0)
            results_4c.append({"mask": m, "score": s, "bbox": b, "concept": "text+box_heart"})
        print(f"  → {len(results_4c)} masks")
        save_overlay(img_np, results_4c, "Exp4C: Text+box 'heart'", "exp4c_text_box.png")
        for i, r in enumerate(results_4c[:3]):
            save_single_mask_overlay(img_np, r, f"Text+box mask {i+1}", f"exp4c_text_box_mask{i+1}.png")
    else:
        print("  → No masks from text+box combined")


# ======================================
# Experiment 5: Multi-mask ranking with anatomy filter
# ======================================
def experiment_5_ranking(medsam3, img_np, lung_masks=None):
    """Get many candidate masks, rank by anatomy constraints."""
    print("\n=== Experiment 5: Multi-mask ranking with anatomy filter ===")

    h, w = img_np.shape[:2]

    # Get ALL heart-related masks from multiple prompts
    prompts = ["heart", "cardiomegaly", "cardiac silhouette", "cardiac", "mediastinum"]
    all_candidates = []
    for prompt in prompts:
        results = medsam3.segment_concepts(
            img_np, [prompt],
            threshold=0.2,  # lower threshold = more candidates
            mask_threshold=0.2,
            max_masks_per_concept=5,
        )
        for r in results:
            r["prompt"] = prompt
            all_candidates.append(r)

    print(f"  Total candidates from {len(prompts)} prompts: {len(all_candidates)}")

    # Combine lung masks for overlap check
    combined_lung = np.zeros((h, w), dtype=np.uint8)
    if lung_masks:
        for lm in lung_masks:
            combined_lung = np.maximum(combined_lung, lm["mask"])

    # Score each candidate
    scored = []
    for i, c in enumerate(all_candidates):
        mask = c["mask"]
        cx_pct, cy_pct = mask_centroid_pct(mask)
        area = mask_area_pct(mask)
        lung_overlap = mask_iou(mask, combined_lung) if lung_masks else 0

        # Scoring:
        # - Reward: central location (heart is ~45-55% horizontal, ~55-75% vertical)
        # - Reward: reasonable area (5-30% for heart)
        # - Penalty: high overlap with lungs
        # - Reward: higher model score

        loc_score = max(0, 1 - abs(cx_pct - 50) / 30) * max(0, 1 - abs(cy_pct - 65) / 25)
        area_score = max(0, 1 - abs(area - 15) / 20) if 2 < area < 40 else 0
        lung_penalty = lung_overlap * 2
        model_score = c["score"]

        total = (
            0.3 * loc_score
            + 0.2 * area_score
            + 0.3 * model_score
            - 0.4 * lung_penalty
        )

        scored.append({
            **c,
            "total_score": total,
            "loc_score": loc_score,
            "area_score": area_score,
            "lung_overlap": lung_overlap,
            "cx_pct": cx_pct,
            "cy_pct": cy_pct,
            "area_pct": area,
        })

    # Sort by total score
    scored.sort(key=lambda x: x["total_score"], reverse=True)

    # Print rankings
    print("\n  Rankings:")
    for i, s in enumerate(scored[:10]):
        print(
            f"  #{i+1}: prompt='{s['prompt']}' total={s['total_score']:.3f} "
            f"model={s['score']:.3f} loc={s['loc_score']:.2f} area={s['area_pct']:.1f}% "
            f"lung_iou={s['lung_overlap']:.2f} centroid=({s['cx_pct']:.0f}%,{s['cy_pct']:.0f}%)"
        )

    # Save top 3 individually
    for i, s in enumerate(scored[:3]):
        save_single_mask_overlay(
            img_np, s,
            f"Ranked #{i+1}: '{s['prompt']}' (score={s['total_score']:.2f})",
            f"exp5_ranked_{i+1}.png",
        )

    # Save top-1 vs worst comparison
    if len(scored) >= 2:
        comparison = [scored[0], scored[-1]]
        save_overlay(
            img_np, comparison,
            f"Exp5: Best vs Worst",
            "exp5_best_vs_worst.png",
        )

    return scored


# ======================================
# Experiment 6: Combined pipeline
# ======================================
def experiment_6_combined(medsam3, img_np, lung_masks):
    """Combined pipeline: mask-out + multi-prompt + box + ranking.

    Strategy:
    1. Build lung mask from existing lung_masks
    2. Create masked image (lungs → median fill)
    3. Run multiple text prompts on BOTH original and masked image
    4. Run box prompts (text+box combined) on original
    5. Collect ALL candidates, deduplicate by IoU
    6. Rank by anatomy (location + area + lung penalty)
    7. Also try: take top candidate, use its bbox as a box prompt for refinement
    """
    print("\n=== Experiment 6: Combined pipeline ===")

    h, w = img_np.shape[:2]
    gt_bbox = [691, 1375, 1653, 1831]  # ground truth heart bbox

    # --- Step 1: Build combined lung mask ---
    combined_lung = np.zeros((h, w), dtype=np.uint8)
    if lung_masks:
        for lm in lung_masks:
            combined_lung = np.maximum(combined_lung, lm["mask"])
    lung_area = mask_area_pct(combined_lung)
    print(f"  Combined lung mask: {lung_area:.1f}% of image")

    # --- Step 2: Create masked image ---
    median_val = int(np.median(img_np[combined_lung == 0]))
    img_masked = img_np.copy()
    img_masked[combined_lung > 0] = median_val

    # --- Step 3: Multi-prompt on original + masked ---
    prompts = ["heart", "cardiomegaly", "cardiac silhouette", "cardiac",
               "heart shadow", "enlarged heart", "left ventricle"]
    all_candidates = []

    # 3a: Original image
    for prompt in prompts:
        results = medsam3.segment_concepts(
            img_np, [prompt],
            threshold=0.2, mask_threshold=0.2, max_masks_per_concept=5,
        )
        for r in results:
            r["prompt"] = prompt
            r["source"] = "original"
            all_candidates.append(r)

    n_orig = len(all_candidates)
    print(f"  Original image: {n_orig} candidates from {len(prompts)} prompts")

    # 3b: Masked image (lungs removed)
    for prompt in prompts:
        results = medsam3.segment_concepts(
            img_masked, [prompt],
            threshold=0.2, mask_threshold=0.2, max_masks_per_concept=5,
        )
        for r in results:
            r["prompt"] = prompt
            r["source"] = "masked"
            all_candidates.append(r)

    n_masked = len(all_candidates) - n_orig
    print(f"  Masked image: {n_masked} candidates from {len(prompts)} prompts")

    # --- Step 4: Box prompts on original ---
    processor = medsam3._concept_processor
    model = medsam3._concept_model
    device = medsam3.device

    pil_img = Image.fromarray(img_np)
    img_inputs = processor(images=pil_img, return_tensors="pt")
    img_inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in img_inputs.items()}

    with torch.no_grad():
        vision_embeds = model.get_vision_features(pixel_values=img_inputs["pixel_values"])

    original_sizes = img_inputs.get("original_sizes")
    target_sizes = original_sizes.tolist() if original_sizes is not None else [[h, w]]

    # 4a: Text "heart" + GT heart box
    text_inputs = processor(text="heart", return_tensors="pt")
    text_inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in text_inputs.items()}
    box_inputs = processor(
        input_boxes=[[[gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3]]]],
        input_boxes_labels=[[1]],
        original_sizes=[[h, w]],
        return_tensors="pt",
    )
    box_inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in box_inputs.items()}

    combined_inputs = {**text_inputs}
    combined_inputs["input_boxes"] = box_inputs["input_boxes"]
    combined_inputs["input_boxes_labels"] = box_inputs["input_boxes_labels"]

    with torch.no_grad():
        outputs = model(vision_embeds=vision_embeds, **combined_inputs)

    post = processor.post_process_instance_segmentation(
        outputs, threshold=0.2, mask_threshold=0.2, target_sizes=target_sizes,
    )
    if post and post[0].get("masks") is not None:
        masks_t = post[0]["masks"]
        scores_t = post[0].get("scores", [])
        boxes_t = post[0].get("boxes", [])
        for j in range(len(masks_t)):
            m = masks_t[j].cpu().numpy().astype(np.uint8) if isinstance(masks_t[j], torch.Tensor) else np.array(masks_t[j], dtype=np.uint8)
            if m.ndim == 3:
                m = m[0]
            s = float(scores_t[j]) if j < len(scores_t) else 0
            b = tuple(int(x) for x in boxes_t[j].tolist()) if j < len(boxes_t) else (0, 0, 0, 0)
            all_candidates.append({"mask": m, "score": s, "bbox": b, "concept": "text+box", "prompt": "heart+box", "source": "box_prompt"})

    # 4b: Text "cardiac silhouette" + GT heart box
    text_inputs2 = processor(text="cardiac silhouette", return_tensors="pt")
    text_inputs2 = {k: v.to(device) if hasattr(v, "to") else v for k, v in text_inputs2.items()}
    combined_inputs2 = {**text_inputs2}
    combined_inputs2["input_boxes"] = box_inputs["input_boxes"]
    combined_inputs2["input_boxes_labels"] = box_inputs["input_boxes_labels"]

    with torch.no_grad():
        outputs2 = model(vision_embeds=vision_embeds, **combined_inputs2)

    post2 = processor.post_process_instance_segmentation(
        outputs2, threshold=0.2, mask_threshold=0.2, target_sizes=target_sizes,
    )
    if post2 and post2[0].get("masks") is not None:
        masks_t = post2[0]["masks"]
        scores_t = post2[0].get("scores", [])
        boxes_t = post2[0].get("boxes", [])
        for j in range(len(masks_t)):
            m = masks_t[j].cpu().numpy().astype(np.uint8) if isinstance(masks_t[j], torch.Tensor) else np.array(masks_t[j], dtype=np.uint8)
            if m.ndim == 3:
                m = m[0]
            s = float(scores_t[j]) if j < len(scores_t) else 0
            b = tuple(int(x) for x in boxes_t[j].tolist()) if j < len(boxes_t) else (0, 0, 0, 0)
            all_candidates.append({"mask": m, "score": s, "bbox": b, "concept": "text+box", "prompt": "cardiac_silhouette+box", "source": "box_prompt"})

    # 4c: Negative lung boxes + positive heart box (no text)
    lung_boxes = []
    if lung_masks:
        for lm in lung_masks:
            ys, xs = np.where(lm["mask"] > 0)
            if len(xs) > 0:
                lung_boxes.append([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])

    if lung_boxes:
        all_boxes = [gt_bbox] + lung_boxes
        all_labels = [1] + [0] * len(lung_boxes)
        neg_box_inputs = processor(
            input_boxes=[all_boxes],
            input_boxes_labels=[all_labels],
            original_sizes=[[h, w]],
            return_tensors="pt",
        )
        neg_box_inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in neg_box_inputs.items()}

        with torch.no_grad():
            outputs_neg = model(vision_embeds=vision_embeds, **neg_box_inputs)

        post_neg = processor.post_process_instance_segmentation(
            outputs_neg, threshold=0.2, mask_threshold=0.2, target_sizes=target_sizes,
        )
        if post_neg and post_neg[0].get("masks") is not None:
            masks_t = post_neg[0]["masks"]
            scores_t = post_neg[0].get("scores", [])
            boxes_t = post_neg[0].get("boxes", [])
            for j in range(len(masks_t)):
                m = masks_t[j].cpu().numpy().astype(np.uint8) if isinstance(masks_t[j], torch.Tensor) else np.array(masks_t[j], dtype=np.uint8)
                if m.ndim == 3:
                    m = m[0]
                s = float(scores_t[j]) if j < len(scores_t) else 0
                b = tuple(int(x) for x in boxes_t[j].tolist()) if j < len(boxes_t) else (0, 0, 0, 0)
                all_candidates.append({"mask": m, "score": s, "bbox": b, "concept": "neg_lung_box", "prompt": "heart_box+neg_lung", "source": "neg_box"})

    n_box = len(all_candidates) - n_orig - n_masked
    print(f"  Box prompts: {n_box} candidates")
    print(f"  Total candidates before dedup: {len(all_candidates)}")

    # --- Step 5: Deduplicate by IoU ---
    # Remove near-identical masks (IoU > 0.85)
    deduped = []
    for c in all_candidates:
        is_dup = False
        for idx, existing in enumerate(deduped):
            if mask_iou(c["mask"], existing["mask"]) > 0.85:
                # Keep the one with higher model score
                if c["score"] > existing["score"]:
                    deduped[idx] = c
                is_dup = True
                break
        if not is_dup:
            deduped.append(c)

    print(f"  After dedup (IoU>0.85): {len(deduped)} unique candidates")

    # --- Step 6: Rank by anatomy ---
    scored = []
    for c in deduped:
        mask = c["mask"]
        cx_pct, cy_pct = mask_centroid_pct(mask)
        area = mask_area_pct(mask)
        lung_overlap = mask_iou(mask, combined_lung)

        # Heart location prior: centroid ~(45-55%, 55-75%)
        loc_score = max(0, 1 - abs(cx_pct - 50) / 30) * max(0, 1 - abs(cy_pct - 65) / 25)

        # Heart area: 5-25% of CXR, penalize outside range
        if area < 1:
            area_score = 0  # too tiny, noise
        elif area < 5:
            area_score = area / 5 * 0.5  # partial credit for small
        elif area <= 25:
            area_score = 1.0 - abs(area - 12) / 20  # sweet spot ~12%
        elif area <= 40:
            area_score = max(0, 0.5 - (area - 25) / 30)  # penalize large
        else:
            area_score = 0  # way too big

        # Lung overlap: heavy penalty
        lung_penalty = lung_overlap * 2.5

        # Bonus for masked-image source (less lung contamination)
        source_bonus = 0.1 if c.get("source") == "masked" else 0
        # Bonus for box-prompted (guided)
        box_bonus = 0.05 if c.get("source") in ("box_prompt", "neg_box") else 0

        model_score = c["score"]

        total = (
            0.30 * loc_score
            + 0.20 * area_score
            + 0.20 * model_score
            + source_bonus
            + box_bonus
            - 0.40 * lung_penalty
        )

        scored.append({
            **c,
            "total_score": total,
            "loc_score": loc_score,
            "area_score": area_score,
            "lung_overlap": lung_overlap,
            "cx_pct": cx_pct,
            "cy_pct": cy_pct,
            "area_pct": area,
        })

    scored.sort(key=lambda x: x["total_score"], reverse=True)

    # Print rankings
    print("\n  Combined rankings (top 15):")
    for i, s in enumerate(scored[:15]):
        src = s.get("source", "?")
        print(
            f"  #{i+1}: prompt='{s.get('prompt', '?')}' src={src} "
            f"total={s['total_score']:.3f} model={s['score']:.3f} "
            f"loc={s['loc_score']:.2f} area={s['area_pct']:.1f}% "
            f"lung_iou={s['lung_overlap']:.2f} centroid=({s['cx_pct']:.0f}%,{s['cy_pct']:.0f}%)"
        )

    # Save top 5 individually
    for i, s in enumerate(scored[:5]):
        src = s.get("source", "?")
        save_single_mask_overlay(
            img_np, s,
            f"Combined #{i+1}: '{s.get('prompt','?')}' src={src} (score={s['total_score']:.2f})",
            f"exp6_combined_rank{i+1}.png",
        )

    # Save top-3 overlay
    top3 = scored[:3]
    if top3:
        save_overlay(
            img_np, top3,
            "Exp6 combined: top 3",
            "exp6_combined_top3.png",
        )

    # --- Step 7: Refinement — use top candidate bbox as box prompt ---
    if scored:
        best = scored[0]
        best_bbox = list(best["bbox"])
        best_area = best["area_pct"]
        best_cx, best_cy = best["cx_pct"], best["cy_pct"]
        print(f"\n  Step 7: Refine best candidate (area={best_area:.1f}%, centroid=({best_cx:.0f}%,{best_cy:.0f}%))")
        print(f"    Using bbox {best_bbox} as positive box prompt")

        # Expand bbox by 20% for refinement
        bw = best_bbox[2] - best_bbox[0]
        bh = best_bbox[3] - best_bbox[1]
        expand = 0.2
        refine_bbox = [
            max(0, int(best_bbox[0] - bw * expand)),
            max(0, int(best_bbox[1] - bh * expand)),
            min(w, int(best_bbox[2] + bw * expand)),
            min(h, int(best_bbox[3] + bh * expand)),
        ]
        print(f"    Expanded bbox (20%): {refine_bbox}")

        # Text + expanded box
        for text_prompt in ["heart", "cardiac silhouette"]:
            text_in = processor(text=text_prompt, return_tensors="pt")
            text_in = {k: v.to(device) if hasattr(v, "to") else v for k, v in text_in.items()}
            refine_box_in = processor(
                input_boxes=[[[refine_bbox[0], refine_bbox[1], refine_bbox[2], refine_bbox[3]]]],
                input_boxes_labels=[[1]],
                original_sizes=[[h, w]],
                return_tensors="pt",
            )
            refine_box_in = {k: v.to(device) if hasattr(v, "to") else v for k, v in refine_box_in.items()}

            refine_combined = {**text_in}
            refine_combined["input_boxes"] = refine_box_in["input_boxes"]
            refine_combined["input_boxes_labels"] = refine_box_in["input_boxes_labels"]

            with torch.no_grad():
                refine_out = model(vision_embeds=vision_embeds, **refine_combined)

            refine_post = processor.post_process_instance_segmentation(
                refine_out, threshold=0.2, mask_threshold=0.2, target_sizes=target_sizes,
            )
            if refine_post and refine_post[0].get("masks") is not None:
                masks_r = refine_post[0]["masks"]
                scores_r = refine_post[0].get("scores", [])
                boxes_r = refine_post[0].get("boxes", [])
                refine_results = []
                for j in range(len(masks_r)):
                    m = masks_r[j].cpu().numpy().astype(np.uint8) if isinstance(masks_r[j], torch.Tensor) else np.array(masks_r[j], dtype=np.uint8)
                    if m.ndim == 3:
                        m = m[0]
                    s = float(scores_r[j]) if j < len(scores_r) else 0
                    b = tuple(int(x) for x in boxes_r[j].tolist()) if j < len(boxes_r) else (0, 0, 0, 0)
                    refine_results.append({"mask": m, "score": s, "bbox": b, "concept": f"refine_{text_prompt}"})

                # Score & pick best non-lung mask
                for r in refine_results:
                    cx_p, cy_p = mask_centroid_pct(r["mask"])
                    a_p = mask_area_pct(r["mask"])
                    lo = mask_iou(r["mask"], combined_lung)
                    r["cx_pct"] = cx_p
                    r["cy_pct"] = cy_p
                    r["area_pct"] = a_p
                    r["lung_overlap"] = lo
                    print(f"    Refine '{text_prompt}': area={a_p:.1f}% centroid=({cx_p:.0f}%,{cy_p:.0f}%) lung_iou={lo:.2f} score={r['score']:.3f}")

                slug = text_prompt.replace(" ", "_")
                save_overlay(img_np, refine_results, f"Exp6 refine: '{text_prompt}'+box", f"exp6_refine_{slug}.png")
                for j, r in enumerate(refine_results[:3]):
                    save_single_mask_overlay(img_np, r, f"Refine '{text_prompt}' mask {j+1}", f"exp6_refine_{slug}_mask{j+1}.png")
            else:
                print(f"    Refine '{text_prompt}': no masks")

    # --- Step 8: Nuclear option — mask-out lungs + text + box together ---
    print(f"\n  Step 8: Masked image + text + box prompt")
    pil_masked = Image.fromarray(img_masked)
    masked_img_inputs = processor(images=pil_masked, return_tensors="pt")
    masked_img_inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in masked_img_inputs.items()}

    with torch.no_grad():
        masked_vision_embeds = model.get_vision_features(pixel_values=masked_img_inputs["pixel_values"])

    for text_prompt in ["heart", "cardiac silhouette"]:
        text_in = processor(text=text_prompt, return_tensors="pt")
        text_in = {k: v.to(device) if hasattr(v, "to") else v for k, v in text_in.items()}
        box_in = processor(
            input_boxes=[[[gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3]]]],
            input_boxes_labels=[[1]],
            original_sizes=[[h, w]],
            return_tensors="pt",
        )
        box_in = {k: v.to(device) if hasattr(v, "to") else v for k, v in box_in.items()}

        nuke_inputs = {**text_in}
        nuke_inputs["input_boxes"] = box_in["input_boxes"]
        nuke_inputs["input_boxes_labels"] = box_in["input_boxes_labels"]

        with torch.no_grad():
            nuke_out = model(vision_embeds=masked_vision_embeds, **nuke_inputs)

        nuke_post = processor.post_process_instance_segmentation(
            nuke_out, threshold=0.2, mask_threshold=0.2, target_sizes=target_sizes,
        )
        if nuke_post and nuke_post[0].get("masks") is not None:
            masks_n = nuke_post[0]["masks"]
            scores_n = nuke_post[0].get("scores", [])
            boxes_n = nuke_post[0].get("boxes", [])
            nuke_results = []
            for j in range(len(masks_n)):
                m = masks_n[j].cpu().numpy().astype(np.uint8) if isinstance(masks_n[j], torch.Tensor) else np.array(masks_n[j], dtype=np.uint8)
                if m.ndim == 3:
                    m = m[0]
                s = float(scores_n[j]) if j < len(scores_n) else 0
                b = tuple(int(x) for x in boxes_n[j].tolist()) if j < len(boxes_n) else (0, 0, 0, 0)
                nuke_results.append({"mask": m, "score": s, "bbox": b, "concept": f"masked+{text_prompt}+box"})

            for r in nuke_results:
                cx_p, cy_p = mask_centroid_pct(r["mask"])
                a_p = mask_area_pct(r["mask"])
                lo = mask_iou(r["mask"], combined_lung)
                r["cx_pct"] = cx_p
                r["cy_pct"] = cy_p
                r["area_pct"] = a_p
                r["lung_overlap"] = lo
                print(f"    Masked+'{text_prompt}'+box: area={a_p:.1f}% centroid=({cx_p:.0f}%,{cy_p:.0f}%) lung_iou={lo:.2f} score={r['score']:.3f}")

            slug = text_prompt.replace(" ", "_")
            # Show masks on ORIGINAL image for visual comparison
            save_overlay(img_np, nuke_results, f"Exp6 masked+'{text_prompt}'+box", f"exp6_masked_box_{slug}.png")
            for j, r in enumerate(nuke_results[:3]):
                save_single_mask_overlay(img_np, r, f"Masked+box '{text_prompt}' mask {j+1}", f"exp6_masked_box_{slug}_mask{j+1}.png")
        else:
            print(f"    Masked+'{text_prompt}'+box: no masks")

    return scored


# ======================================
# Main
# ======================================
def main():
    print("Loading image...")
    img_np = load_image()
    print(f"  Image shape: {img_np.shape}")

    # Draw ground truth bbox on image for reference
    gt_bbox = [691, 1375, 1653, 1831]
    ref_img = Image.fromarray(img_np)
    draw = ImageDraw.Draw(ref_img)
    draw.rectangle(gt_bbox, outline=(0, 255, 0), width=4)
    draw.text((gt_bbox[0], gt_bbox[1] - 20), "GT: cardiomegaly", fill=(0, 255, 0))
    ref_img.save(os.path.join(RESULTS_DIR, "reference_gt_bbox.png"))
    print("  Saved reference image with GT bbox")

    print("\nLoading MedSAM3...")
    t0 = time.perf_counter()
    medsam3 = MedSAM3Tool(checkpoint=MODEL_PATH)
    print(f"  Model loaded in {time.perf_counter() - t0:.1f}s")

    # Run experiments (pass --skip-done to skip 1-3 if already completed)
    skip_done = "--skip-done" in sys.argv
    if not skip_done:
        experiment_1_baseline(medsam3, img_np)
        experiment_2_alt_prompts(medsam3, img_np)
    if not skip_done:
        lung_masks = experiment_3_lung_negative(medsam3, img_np)
        experiment_4_box_prompts(medsam3, img_np, lung_masks)
        experiment_5_ranking(medsam3, img_np, lung_masks)
    else:
        # Still need lung masks for exp6
        print("\n  Getting lung masks (needed for combined pipeline)...")
        lung_masks = medsam3.segment_concepts(
            img_np, ["lung", "left lung", "right lung"],
            threshold=0.3, mask_threshold=0.3, max_masks_per_concept=3,
        )
        print(f"  Found {len(lung_masks)} lung masks")

    experiment_6_combined(medsam3, img_np, lung_masks)

    print(f"\n{'='*60}")
    print(f"All results saved to: {RESULTS_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
