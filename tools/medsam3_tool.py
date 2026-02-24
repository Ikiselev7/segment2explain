from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np


try:
    import torch
    from transformers import Sam3Model, Sam3Processor
except Exception:
    torch = None
    Sam3Model = None
    Sam3Processor = None

logger = logging.getLogger(__name__)


@dataclass
class MedSAM3Tool:
    """
    MedSAM3 wrapper for text-prompted concept segmentation via Sam3Model.

    Default checkpoint: models/medsam3-merged
    """

    checkpoint: str = "models/medsam3-merged"
    device: str | None = None

    # Lazy-loaded model stack (only initialized when needed)
    _concept_model: object = field(default=None, init=False, repr=False)
    _concept_processor: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if torch is None:
            raise RuntimeError("Missing dependencies for MedSAM3Tool. Install torch and transformers.")
        if self.device is None:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(
            "MedSAM3Tool initialized: checkpoint=%s device=%s (models loaded lazily)",
            self.checkpoint,
            self.device,
        )

    def _ensure_concept(self) -> None:
        """Lazy-load Sam3Model + Sam3Processor for concept segmentation."""
        if self._concept_model is not None:
            return
        logger.info("Loading MedSAM3 concept model: checkpoint=%s device=%s", self.checkpoint, self.device)
        t0 = time.perf_counter()
        self._concept_processor = Sam3Processor.from_pretrained(self.checkpoint)
        self._concept_model = Sam3Model.from_pretrained(
            self.checkpoint,
            dtype=torch.float32,
        )
        self._concept_model.to(self.device)
        self._concept_model.eval()
        dt = time.perf_counter() - t0
        param_count = sum(p.numel() for p in self._concept_model.parameters()) / 1e6
        logger.info("MedSAM3 concept model loaded in %.1fs (%.0fM params)", dt, param_count)

    def segment_concepts(
        self,
        image_rgb_uint8: np.ndarray,
        concepts: list[str],
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        max_masks_per_concept: int = 5,
        nms_iou_thresh: float = 0.5,
        max_total_masks: int = 20,
        preprocess: bool = True,
    ) -> list[dict]:
        """
        Text-prompted concept segmentation using Sam3Model.

        For each concept string, find all matching instances in the image.

        Returns list of dicts:
            {"mask": ndarray(H,W,uint8), "bbox": (x0,y0,x1,y1), "score": float, "concept": str}
        """
        from PIL import Image

        image_for_seg = _preprocess_medical_image(image_rgb_uint8) if preprocess else _coerce_rgb_uint8(image_rgb_uint8)
        h, w = image_for_seg.shape[:2]
        logger.info(
            "MedSAM3 segment_concepts: image=%dx%d concepts=%s preprocess=%s threshold=%.2f",
            w,
            h,
            concepts,
            preprocess,
            threshold,
        )
        t0 = time.perf_counter()

        self._ensure_concept()
        pil_image = Image.fromarray(image_for_seg)

        # Pre-compute vision embeddings once
        img_inputs = self._concept_processor(images=pil_image, return_tensors="pt")
        img_inputs = {
            k: v.to(torch.float32).to(self.device)
            if hasattr(v, "dtype") and v.dtype == torch.float64
            else v.to(self.device)
            if hasattr(v, "to")
            else v
            for k, v in img_inputs.items()
        }

        with torch.no_grad():
            vision_embeds = self._concept_model.get_vision_features(
                pixel_values=img_inputs["pixel_values"],
            )

        original_sizes = img_inputs.get("original_sizes")
        if original_sizes is not None:
            target_sizes = original_sizes.tolist()
        else:
            target_sizes = [[h, w]]

        all_results = []
        for concept in concepts:
            try:
                text_inputs = self._concept_processor(text=concept, return_tensors="pt")
                text_inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in text_inputs.items()}

                with torch.no_grad():
                    outputs = self._concept_model(
                        vision_embeds=vision_embeds,
                        **text_inputs,
                    )

                post = self._concept_processor.post_process_instance_segmentation(
                    outputs,
                    threshold=threshold,
                    mask_threshold=mask_threshold,
                    target_sizes=target_sizes,
                )

                if not post:
                    continue

                result = post[0]
                masks = result.get("masks", [])
                scores = result.get("scores", [])
                boxes = result.get("boxes", [])

                for j in range(min(len(masks), max_masks_per_concept)):
                    mask_data = masks[j]
                    if isinstance(mask_data, torch.Tensor):
                        mask = mask_data.cpu().numpy().astype(np.uint8)
                    else:
                        mask = np.array(mask_data, dtype=np.uint8)
                    if mask.ndim == 3:
                        mask = mask[0]
                    if mask.sum() == 0:
                        continue

                    score = float(scores[j]) if j < len(scores) else 0.5

                    # Get bbox from mask or from boxes output
                    if j < len(boxes):
                        box = boxes[j]
                        if isinstance(box, torch.Tensor):
                            box = box.cpu().tolist()
                        bbox = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                    else:
                        ys, xs = np.where(mask > 0)
                        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

                    all_results.append(
                        {
                            "mask": mask,
                            "bbox": bbox,
                            "score": score,
                            "concept": concept,
                        }
                    )

                logger.info(
                    "MedSAM3 concept '%s': found %d instance(s)", concept, min(len(masks), max_masks_per_concept)
                )

            except Exception as e:
                logger.warning("MedSAM3 concept '%s' failed: %s", concept, e)
                continue

        if nms_iou_thresh > 0:
            all_results = _nms_result_dicts(all_results, iou_thresh=nms_iou_thresh)

        if max_total_masks > 0:
            all_results = all_results[:max_total_masks]

        dt = time.perf_counter() - t0
        logger.info(
            "MedSAM3 segment_concepts: %d concepts -> %d total instances in %.1fs",
            len(concepts),
            len(all_results),
            dt,
        )
        return all_results

    def segment_concept_with_spatial_prior(
        self,
        image_rgb_uint8: np.ndarray,
        concept: str,
        positive_box: tuple[int, int, int, int] | None = None,
        positive_points: list[tuple[int, int]] | None = None,
        negative_boxes: list[tuple[int, int, int, int]] | None = None,
        threshold: float = 0.2,
        mask_threshold: float = 0.2,
        max_masks: int = 1,
        preprocess: bool = True,
    ) -> list[dict]:
        """Text-prompted segmentation with attention-derived spatial priors.

        Combines text prompt with positive box/point prompts from MedGemma
        attention heatmaps to guide segmentation to the correct region.

        Args:
            image_rgb_uint8: RGB uint8 image (H, W, 3)
            concept: Medical concept to search for
            positive_box: (x_min, y_min, x_max, y_max) attention-derived box, label=1
            positive_points: List of (x, y) attention peaks, label=1
            negative_boxes: List of (x_min, y_min, x_max, y_max) regions to avoid, label=0
            threshold: Score threshold for post-processing
            mask_threshold: Mask threshold for post-processing
            max_masks: Maximum masks to return
            preprocess: Apply medical image preprocessing

        Returns:
            List of dicts: {mask, bbox, score, concept}
        """
        from PIL import Image

        # If no spatial priors, fall back to text-only
        if positive_box is None and not positive_points and not negative_boxes:
            return self.segment_concepts(
                image_rgb_uint8, [concept],
                threshold=threshold, mask_threshold=mask_threshold,
                max_masks_per_concept=max_masks, preprocess=preprocess,
            )

        image_for_seg = _preprocess_medical_image(image_rgb_uint8) if preprocess else _coerce_rgb_uint8(image_rgb_uint8)
        h, w = image_for_seg.shape[:2]
        logger.info(
            "MedSAM3 segment_concept_with_spatial_prior: concept='%s' pos_box=%s pos_pts=%d neg_boxes=%d image=%dx%d",
            concept,
            positive_box is not None,
            len(positive_points or []),
            len(negative_boxes or []),
            w, h,
        )
        t0 = time.perf_counter()

        self._ensure_concept()
        pil_image = Image.fromarray(image_for_seg)

        # Pre-compute vision embeddings
        img_inputs = self._concept_processor(images=pil_image, return_tensors="pt")
        img_inputs = {
            k: v.to(torch.float32).to(self.device)
            if hasattr(v, "dtype") and v.dtype == torch.float64
            else v.to(self.device)
            if hasattr(v, "to")
            else v
            for k, v in img_inputs.items()
        }

        with torch.no_grad():
            vision_embeds = self._concept_model.get_vision_features(
                pixel_values=img_inputs["pixel_values"],
            )

        original_sizes = img_inputs.get("original_sizes")
        target_sizes = original_sizes.tolist() if original_sizes is not None else [[h, w]]

        # Prepare text inputs
        text_inputs = self._concept_processor(text=concept, return_tensors="pt")
        text_inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in text_inputs.items()}

        # Build combined box prompt (positive + negative)
        all_boxes: list[list[int]] = []
        all_labels: list[int] = []

        if positive_box is not None:
            all_boxes.append([int(x) for x in positive_box])
            all_labels.append(1)

        if negative_boxes:
            for neg_box in negative_boxes:
                all_boxes.append([int(x) for x in neg_box])
                all_labels.append(0)

        combined = {**text_inputs}

        if all_boxes:
            box_inputs = self._concept_processor(
                input_boxes=[all_boxes],
                input_boxes_labels=[all_labels],
                original_sizes=[[h, w]],
                return_tensors="pt",
            )
            box_inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in box_inputs.items()}
            combined["input_boxes"] = box_inputs["input_boxes"]
            combined["input_boxes_labels"] = box_inputs["input_boxes_labels"]

        # Add point prompts if available
        if positive_points:
            point_coords = [[list(pt) for pt in positive_points]]
            point_labels = [[1] * len(positive_points)]
            point_inputs = self._concept_processor(
                input_points=point_coords,
                input_points_labels=point_labels,
                original_sizes=[[h, w]],
                return_tensors="pt",
            )
            point_inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in point_inputs.items()}
            combined["input_points"] = point_inputs.get("input_points")
            combined["input_points_labels"] = point_inputs.get("input_points_labels")

        results = []
        try:
            with torch.no_grad():
                outputs = self._concept_model(vision_embeds=vision_embeds, **combined)

            post = self._concept_processor.post_process_instance_segmentation(
                outputs, threshold=threshold, mask_threshold=mask_threshold,
                target_sizes=target_sizes,
            )

            if post:
                masks = post[0].get("masks", [])
                scores = post[0].get("scores", [])
                boxes = post[0].get("boxes", [])

                for j in range(min(len(masks), max_masks)):
                    mask_data = masks[j]
                    if isinstance(mask_data, torch.Tensor):
                        mask = mask_data.cpu().numpy().astype(np.uint8)
                    else:
                        mask = np.array(mask_data, dtype=np.uint8)
                    if mask.ndim == 3:
                        mask = mask[0]
                    if mask.sum() == 0:
                        continue

                    score = float(scores[j]) if j < len(scores) else 0.5

                    if j < len(boxes):
                        box = boxes[j]
                        if isinstance(box, torch.Tensor):
                            box = box.cpu().tolist()
                        bbox = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                    else:
                        ys, xs = np.where(mask > 0)
                        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

                    results.append({
                        "mask": mask,
                        "bbox": bbox,
                        "score": score,
                        "concept": concept,
                    })

        except Exception as e:
            logger.warning("MedSAM3 concept '%s' with spatial prior failed: %s", concept, e)

        dt = time.perf_counter() - t0
        logger.info(
            "MedSAM3 segment_concept_with_spatial_prior: concept='%s' -> %d masks in %.1fs",
            concept, len(results), dt,
        )
        return results

    def segment_concept_with_negatives(
        self,
        image_rgb_uint8: np.ndarray,
        concept: str,
        negative_boxes: list[tuple[int, int, int, int]],
        threshold: float = 0.2,
        mask_threshold: float = 0.2,
        max_masks: int = 1,
        preprocess: bool = True,
    ) -> list[dict]:
        """Text-prompted segmentation with negative box constraints.

        Tells MedSAM3: "find this concept, but NOT in these regions."

        Args:
            image_rgb_uint8: RGB uint8 image (H, W, 3)
            concept: Medical concept to search for
            negative_boxes: list of (x_min, y_min, x_max, y_max) pixel-coord
                            regions to avoid
            threshold: Score threshold for post-processing
            mask_threshold: Mask threshold for post-processing
            max_masks: Maximum masks to return
            preprocess: Apply medical image preprocessing

        Returns:
            List of dicts: {mask, bbox, score, concept}
        """
        from PIL import Image

        if not negative_boxes:
            return self.segment_concepts(
                image_rgb_uint8, [concept],
                threshold=threshold, mask_threshold=mask_threshold,
                max_masks_per_concept=max_masks, preprocess=preprocess,
            )

        image_for_seg = _preprocess_medical_image(image_rgb_uint8) if preprocess else _coerce_rgb_uint8(image_rgb_uint8)
        h, w = image_for_seg.shape[:2]
        logger.info(
            "MedSAM3 segment_concept_with_negatives: concept='%s' neg_boxes=%d image=%dx%d",
            concept, len(negative_boxes), w, h,
        )
        t0 = time.perf_counter()

        self._ensure_concept()
        pil_image = Image.fromarray(image_for_seg)

        # Pre-compute vision embeddings
        img_inputs = self._concept_processor(images=pil_image, return_tensors="pt")
        img_inputs = {
            k: v.to(torch.float32).to(self.device)
            if hasattr(v, "dtype") and v.dtype == torch.float64
            else v.to(self.device)
            if hasattr(v, "to")
            else v
            for k, v in img_inputs.items()
        }

        with torch.no_grad():
            vision_embeds = self._concept_model.get_vision_features(
                pixel_values=img_inputs["pixel_values"],
            )

        original_sizes = img_inputs.get("original_sizes")
        if original_sizes is not None:
            target_sizes = original_sizes.tolist()
        else:
            target_sizes = [[h, w]]

        # Prepare text inputs for the concept
        text_inputs = self._concept_processor(text=concept, return_tensors="pt")
        text_inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in text_inputs.items()}

        # Prepare negative box inputs
        # All boxes are negative (label=0): "don't segment here"
        neg_boxes_list = [[int(x) for x in box] for box in negative_boxes]
        neg_labels = [0] * len(neg_boxes_list)

        box_inputs = self._concept_processor(
            input_boxes=[neg_boxes_list],
            input_boxes_labels=[neg_labels],
            original_sizes=[[h, w]],
            return_tensors="pt",
        )
        box_inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in box_inputs.items()}

        # Merge text + negative box inputs
        combined = {**text_inputs}
        combined["input_boxes"] = box_inputs["input_boxes"]
        combined["input_boxes_labels"] = box_inputs["input_boxes_labels"]

        results = []
        try:
            with torch.no_grad():
                outputs = self._concept_model(vision_embeds=vision_embeds, **combined)

            post = self._concept_processor.post_process_instance_segmentation(
                outputs, threshold=threshold, mask_threshold=mask_threshold,
                target_sizes=target_sizes,
            )

            if post:
                masks = post[0].get("masks", [])
                scores = post[0].get("scores", [])
                boxes = post[0].get("boxes", [])

                for j in range(min(len(masks), max_masks)):
                    mask_data = masks[j]
                    if isinstance(mask_data, torch.Tensor):
                        mask = mask_data.cpu().numpy().astype(np.uint8)
                    else:
                        mask = np.array(mask_data, dtype=np.uint8)
                    if mask.ndim == 3:
                        mask = mask[0]
                    if mask.sum() == 0:
                        continue

                    score = float(scores[j]) if j < len(scores) else 0.5

                    if j < len(boxes):
                        box = boxes[j]
                        if isinstance(box, torch.Tensor):
                            box = box.cpu().tolist()
                        bbox = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                    else:
                        ys, xs = np.where(mask > 0)
                        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

                    results.append({
                        "mask": mask,
                        "bbox": bbox,
                        "score": score,
                        "concept": concept,
                    })

        except Exception as e:
            logger.warning("MedSAM3 concept '%s' with negatives failed: %s", concept, e)

        dt = time.perf_counter() - t0
        logger.info(
            "MedSAM3 segment_concept_with_negatives: concept='%s' -> %d masks in %.1fs",
            concept, len(results), dt,
        )
        return results


def _coerce_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Ensure image is RGB uint8 for MedSAM3."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.ndim == 3 and arr.shape[-1] >= 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _looks_grayscale_rgb(image_rgb_uint8: np.ndarray, tolerance: float = 2.0) -> bool:
    """Heuristic check for grayscale-like RGB images (common for X-rays)."""
    if image_rgb_uint8.ndim != 3 or image_rgb_uint8.shape[-1] != 3:
        return False
    arr = image_rgb_uint8.astype(np.int16)
    rg = np.mean(np.abs(arr[..., 0] - arr[..., 1]))
    rb = np.mean(np.abs(arr[..., 0] - arr[..., 2]))
    gb = np.mean(np.abs(arr[..., 1] - arr[..., 2]))
    return max(rg, rb, gb) <= tolerance


def _preprocess_medical_image(
    image_rgb_uint8: np.ndarray,
    lower_pct: float = 0.5,
    upper_pct: float = 99.5,
) -> np.ndarray:
    """Apply lightweight contrast normalization for grayscale-like medical inputs."""
    rgb = _coerce_rgb_uint8(image_rgb_uint8)
    if not _looks_grayscale_rgb(rgb):
        return rgb

    gray = rgb[..., 0].astype(np.float32)
    lo = float(np.percentile(gray, lower_pct))
    hi = float(np.percentile(gray, upper_pct))
    if hi <= lo + 1e-6:
        return rgb

    scaled = np.clip((gray - lo) / (hi - lo), 0.0, 1.0)
    out = (scaled * 255.0).astype(np.uint8)
    return np.stack([out, out, out], axis=-1)


def _nms_result_dicts(results: list[dict], iou_thresh: float = 0.5) -> list[dict]:
    """Run NMS on list-of-dict results while preserving extra keys like concept."""
    if not results:
        return []

    import torch
    from torchvision.ops import nms

    boxes = []
    scores = []
    kept_indices = []
    for i, result in enumerate(results):
        bbox = result.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = bbox
        if x1 <= x0 or y1 <= y0:
            continue
        boxes.append([float(x0), float(y0), float(x1), float(y1)])
        scores.append(float(result.get("score", 0.0)))
        kept_indices.append(i)

    if not boxes:
        return []

    keep = nms(
        torch.tensor(boxes, dtype=torch.float32),
        torch.tensor(scores, dtype=torch.float32),
        iou_thresh,
    )
    return [results[kept_indices[k.item()]] for k in keep]
