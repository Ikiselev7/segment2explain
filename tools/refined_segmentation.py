"""Refined segmentation pipeline.

Passes MedGemma-provided concepts directly to MedSAM3 segment_concepts(),
returning 1 best mask per concept.

Usage:
    results = refined_segment(img_np, ["heart", "right lung", "nodule"], medsam3_tool)
"""

from __future__ import annotations

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    m1b = m1 > 0
    m2b = m2 > 0
    intersection = int(np.sum(m1b & m2b))
    union = int(np.sum(m1b | m2b))
    return intersection / union if union > 0 else 0.0


def mask_area_pct(mask: np.ndarray) -> float:
    """Return mask area as percentage of image area (0-100)."""
    total = mask.shape[0] * mask.shape[1]
    return float(np.sum(mask > 0)) / total * 100 if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD = 0.2
DEFAULT_MASK_THRESHOLD = 0.2
DEFAULT_CROSS_DEDUP_IOU = 0.7


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _dedup_candidates(
    candidates: list[dict],
    iou_thresh: float,
) -> list[dict]:
    """Remove near-duplicate masks, keeping higher-score version."""
    if not candidates:
        return []

    deduped: list[dict] = []
    for c in candidates:
        is_dup = False
        for idx, existing in enumerate(deduped):
            if mask_iou(c["mask"], existing["mask"]) > iou_thresh:
                if c["score"] > existing["score"]:
                    deduped[idx] = c
                is_dup = True
                break
        if not is_dup:
            deduped.append(c)

    return deduped


def refined_segment(
    img_np: np.ndarray,
    concepts: list[str],
    medsam3_tool: object,
    config: object | None = None,
    proposed_bbox: list[int] | None = None,
) -> list[dict]:
    """Run concept segmentation: 1 best mask per concept from MedSAM3.

    Args:
        img_np: RGB uint8 image (H, W, 3)
        concepts: Medical concepts to search for (e.g. ["heart", "right lung"])
        medsam3_tool: MedSAM3Tool instance
        config: Unused, kept for API compatibility
        proposed_bbox: Unused, kept for API compatibility

    Returns:
        List of dicts: {mask, bbox, score, concept, area_pct}
    """
    if not concepts:
        logger.warning("Refined seg: no concepts provided")
        return []

    t0 = time.perf_counter()
    logger.info("Refined seg: searching for concepts=%s", concepts)

    # 1 mask per concept from MedSAM3
    all_candidates = medsam3_tool.segment_concepts(
        img_np,
        concepts,
        threshold=DEFAULT_THRESHOLD,
        mask_threshold=DEFAULT_MASK_THRESHOLD,
        max_masks_per_concept=1,
    )

    if not all_candidates:
        dt = time.perf_counter() - t0
        logger.warning("Refined seg: no candidates found (%.1fs)", dt)
        return []

    # Add area_pct to each result
    for c in all_candidates:
        c["area_pct"] = mask_area_pct(c["mask"])

    # Cross-concept dedup to avoid overlapping masks
    results = _dedup_candidates(all_candidates, DEFAULT_CROSS_DEDUP_IOU)
    results.sort(key=lambda x: x.get("score", 0), reverse=True)

    dt = time.perf_counter() - t0
    logger.info(
        "Refined seg: done in %.1fs: %d concepts → %d results",
        dt, len(concepts), len(results),
    )

    for i, r in enumerate(results):
        logger.info(
            "Refined seg: #%d score=%.3f area=%.1f%% concept=%s",
            i + 1, r.get("score", 0), r.get("area_pct", 0), r.get("concept", "?"),
        )

    return results


def refined_segment_with_priors(
    img_np: np.ndarray,
    concepts: list[str],
    medsam3_tool: object,
    spatial_priors: dict[str, tuple[int, int, int, int]],
    config: object | None = None,
) -> list[dict]:
    """Run segmentation using attention-derived spatial priors.

    For each concept, if a prior box exists, uses text+positive_box prompt.
    Falls back to text-only for concepts without priors.

    Args:
        img_np: RGB uint8 image (H, W, 3)
        concepts: Medical concepts to search for
        medsam3_tool: MedSAM3Tool instance
        spatial_priors: dict mapping concept → (x0, y0, x1, y1) attention box
        config: Unused, kept for API compatibility

    Returns:
        List of dicts: {mask, bbox, score, concept, area_pct}
    """
    if not concepts:
        return []

    t0 = time.perf_counter()
    concepts_with_prior = [c for c in concepts if c in spatial_priors]
    concepts_without_prior = [c for c in concepts if c not in spatial_priors]
    logger.info(
        "Refined seg (priors): %d/%d concepts with spatial priors",
        len(concepts_with_prior), len(concepts),
    )

    all_candidates = []

    # Concepts with spatial priors: use text + positive box
    for concept in concepts_with_prior:
        box = spatial_priors[concept]
        logger.info("Refined seg (prior): concept='%s' box=%s", concept, box)
        results = medsam3_tool.segment_concept_with_spatial_prior(
            img_np, concept,
            positive_box=box,
            threshold=DEFAULT_THRESHOLD,
            mask_threshold=DEFAULT_MASK_THRESHOLD,
            max_masks=1,
        )
        all_candidates.extend(results)

    # Concepts without priors: text-only fallback
    if concepts_without_prior:
        fallback = medsam3_tool.segment_concepts(
            img_np, concepts_without_prior,
            threshold=DEFAULT_THRESHOLD,
            mask_threshold=DEFAULT_MASK_THRESHOLD,
            max_masks_per_concept=1,
        )
        all_candidates.extend(fallback)

    if not all_candidates:
        dt = time.perf_counter() - t0
        logger.warning("Refined seg (priors): no candidates found (%.1fs)", dt)
        return []

    # Add area_pct and dedup
    for c in all_candidates:
        c["area_pct"] = mask_area_pct(c["mask"])

    results = _dedup_candidates(all_candidates, DEFAULT_CROSS_DEDUP_IOU)
    results.sort(key=lambda x: x.get("score", 0), reverse=True)

    dt = time.perf_counter() - t0
    logger.info(
        "Refined seg (priors): done in %.1fs: %d concepts → %d results",
        dt, len(concepts), len(results),
    )
    for i, r in enumerate(results):
        logger.info(
            "Refined seg (prior): #%d score=%.3f area=%.1f%% concept=%s",
            i + 1, r.get("score", 0), r.get("area_pct", 0), r.get("concept", "?"),
        )

    return results


def refined_segment_with_attention_overlay(
    img_np: np.ndarray,
    concepts: list[str],
    medsam3_tool: object,
    heatmaps: dict[str, "np.ndarray"],
    spatial_priors: dict[str, tuple[int, int, int, int]] | None = None,
    config: object | None = None,
) -> list[dict]:
    """Run segmentation using attention heatmaps as image overlays.

    For each concept, modulates the image brightness using the attention
    heatmap (dim low-attention regions) and optionally combines with
    spatial-prior box prompts.

    Args:
        img_np: RGB uint8 image (H, W, 3)
        concepts: Medical concepts to search for
        medsam3_tool: MedSAM3Tool instance
        heatmaps: dict mapping concept → 2D heatmap (from MedGemma attention)
        spatial_priors: Optional dict mapping concept → (x0, y0, x1, y1)
        config: Unused, kept for API compatibility

    Returns:
        List of dicts: {mask, bbox, score, concept, area_pct}
    """
    from models.attention_prior import apply_heatmap_overlay

    if not concepts:
        return []

    t0 = time.perf_counter()
    concepts_with_heatmap = [c for c in concepts if c in heatmaps]
    concepts_without_heatmap = [c for c in concepts if c not in heatmaps]
    logger.info(
        "Refined seg (overlay): %d/%d concepts with heatmaps",
        len(concepts_with_heatmap), len(concepts),
    )

    all_candidates = []

    # Concepts with heatmaps: overlay + optional box
    for concept in concepts_with_heatmap:
        overlaid_img = apply_heatmap_overlay(img_np, heatmaps[concept])
        box = (spatial_priors or {}).get(concept)
        if box is not None:
            results = medsam3_tool.segment_concept_with_spatial_prior(
                overlaid_img, concept,
                positive_box=box,
                threshold=DEFAULT_THRESHOLD,
                mask_threshold=DEFAULT_MASK_THRESHOLD,
                max_masks=1,
            )
        else:
            results = medsam3_tool.segment_concepts(
                overlaid_img, [concept],
                threshold=DEFAULT_THRESHOLD,
                mask_threshold=DEFAULT_MASK_THRESHOLD,
                max_masks_per_concept=1,
            )
        all_candidates.extend(results)

    # Concepts without heatmaps: text-only fallback
    if concepts_without_heatmap:
        fallback = medsam3_tool.segment_concepts(
            img_np, concepts_without_heatmap,
            threshold=DEFAULT_THRESHOLD,
            mask_threshold=DEFAULT_MASK_THRESHOLD,
            max_masks_per_concept=1,
        )
        all_candidates.extend(fallback)

    if not all_candidates:
        dt = time.perf_counter() - t0
        logger.warning("Refined seg (overlay): no candidates found (%.1fs)", dt)
        return []

    for c in all_candidates:
        c["area_pct"] = mask_area_pct(c["mask"])

    results = _dedup_candidates(all_candidates, DEFAULT_CROSS_DEDUP_IOU)
    results.sort(key=lambda x: x.get("score", 0), reverse=True)

    dt = time.perf_counter() - t0
    logger.info(
        "Refined seg (overlay): done in %.1fs: %d concepts → %d results",
        dt, len(concepts), len(results),
    )
    for i, r in enumerate(results):
        logger.info(
            "Refined seg (overlay): #%d score=%.3f area=%.1f%% concept=%s",
            i + 1, r.get("score", 0), r.get("area_pct", 0), r.get("concept", "?"),
        )

    return results


def refined_segment_with_negatives(
    img_np: np.ndarray,
    concepts: list[str],
    medsam3_tool: object,
    negative_boxes: list[tuple[int, int, int, int]],
    config: object | None = None,
) -> list[dict]:
    """Re-segment unmatched concepts, excluding regions of matched segments.

    For each concept, calls segment_concept_with_negatives() with the
    provided negative boxes (bboxes of already-matched segments).

    Args:
        img_np: RGB uint8 image (H, W, 3)
        concepts: Unmatched concepts to search for
        medsam3_tool: MedSAM3Tool instance
        negative_boxes: Bboxes of already-found segments to avoid
        config: Unused, kept for API compatibility

    Returns:
        List of dicts: {mask, bbox, score, concept, area_pct}
    """
    if not concepts:
        return []

    t0 = time.perf_counter()
    logger.info(
        "Refined seg (negatives): concepts=%s neg_boxes=%d",
        concepts, len(negative_boxes),
    )

    all_candidates = []
    for concept in concepts:
        results = medsam3_tool.segment_concept_with_negatives(
            img_np, concept, negative_boxes,
            threshold=DEFAULT_THRESHOLD,
            mask_threshold=DEFAULT_MASK_THRESHOLD,
            max_masks=1,
        )
        all_candidates.extend(results)

    if not all_candidates:
        dt = time.perf_counter() - t0
        logger.warning("Refined seg (negatives): no candidates found (%.1fs)", dt)
        return []

    for c in all_candidates:
        c["area_pct"] = mask_area_pct(c["mask"])

    deduped = _dedup_candidates(all_candidates, DEFAULT_CROSS_DEDUP_IOU)
    deduped.sort(key=lambda x: x.get("score", 0), reverse=True)

    dt = time.perf_counter() - t0
    logger.info(
        "Refined seg (negatives): done in %.1fs: %d concepts → %d results",
        dt, len(concepts), len(deduped),
    )

    return deduped
