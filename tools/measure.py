from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np


def _bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def measure_mask(
    mask: np.ndarray,
    pixel_spacing: Tuple[float, float] | None = None,
) -> Dict[str, Any]:
    """Minimal measurements for PoC.

    Args:
        mask: Binary mask (H, W).
        pixel_spacing: Optional (row_mm, col_mm) from DICOM for mm conversion.

    Returns:
      - area_px, bbox_px, max_diameter_px, centroid_px (always)
      - area_mm2, max_diameter_mm, pixel_spacing_mm (when pixel_spacing provided)
    """
    mask = (mask > 0).astype(np.uint8)
    area = int(mask.sum())
    x0, y0, x1, y1 = _bbox_from_mask(mask)
    dx = max(0, x1 - x0)
    dy = max(0, y1 - y0)
    max_diam = float((dx * dx + dy * dy) ** 0.5)

    if area > 0:
        ys, xs = np.where(mask > 0)
        cx = float(xs.mean())
        cy = float(ys.mean())
    else:
        cx = cy = 0.0

    result: Dict[str, Any] = {
        "area_px": area,
        "bbox_px": [x0, y0, x1, y1],
        "max_diameter_px": round(max_diam, 2),
        "centroid_px": [round(cx, 2), round(cy, 2)],
    }

    if pixel_spacing is not None:
        row_mm, col_mm = pixel_spacing
        result["area_mm2"] = round(area * row_mm * col_mm, 2)
        diam_mm = float(((dx * col_mm) ** 2 + (dy * row_mm) ** 2) ** 0.5)
        result["max_diameter_mm"] = round(diam_mm, 2)
        result["pixel_spacing_mm"] = [row_mm, col_mm]

    return result
