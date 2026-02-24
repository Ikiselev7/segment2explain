from __future__ import annotations

from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SEGMENT_COLORS: List[Tuple[int, int, int]] = [
    (255, 0, 0),      # Red
    (0, 128, 255),     # Blue
    (0, 200, 0),       # Green
    (255, 165, 0),     # Orange
    (128, 0, 255),     # Purple
    (255, 255, 0),     # Yellow
    (0, 200, 200),     # Cyan
    (255, 0, 128),     # Pink
    (128, 128, 0),     # Olive
    (0, 128, 128),     # Teal
]

COLOR_NAMES: List[str] = [
    "Red", "Blue", "Green", "Orange", "Purple",
    "Yellow", "Cyan", "Pink", "Olive", "Teal",
]


def _mask_contour(mask: np.ndarray, thickness: int = 3) -> np.ndarray:
    """Extract contour pixels from a binary mask.

    A pixel is on the contour if it is inside the mask but within
    `thickness` pixels of the mask boundary.

    Returns a binary uint8 array of contour pixels.
    """
    m = (mask > 0).astype(np.uint8)
    eroded = m.copy()
    for _ in range(thickness):
        padded = np.pad(eroded, 1, mode="constant", constant_values=0)
        eroded = (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1]   # up
            & padded[2:, 1:-1]    # down
            & padded[1:-1, :-2]   # left
            & padded[1:-1, 2:]    # right
        )
    return (m - eroded).astype(np.uint8)


def overlay_mask_on_image(
    image_rgb_uint8: np.ndarray,
    mask_01: np.ndarray,
    title: str = "",
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.2,
    contour_thickness: int = 6,
) -> Image.Image:
    """
    Create a PIL image with mask contour overlay (underlying image visible).
    """
    img = Image.fromarray(image_rgb_uint8).convert("RGB")
    out_np = np.array(img, dtype=np.float32)

    # Semi-transparent fill
    fill_mask = mask_01 > 0
    color_arr = np.array(color, dtype=np.float32)
    out_np[fill_mask] = (1 - alpha) * out_np[fill_mask] + alpha * color_arr

    # Opaque contour outline
    contour = _mask_contour(mask_01, thickness=contour_thickness)
    out_np[contour > 0] = color_arr

    out = Image.fromarray(out_np.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(out)

    if title:
        draw.rectangle([0, 0, out.size[0], 28], fill=(0, 0, 0))
        draw.text((8, 6), title, fill=(255, 255, 255))

    return out


def overlay_multiple_masks(
    image_rgb_uint8: np.ndarray,
    masks: List[Tuple[np.ndarray, str]],
    alpha: float = 0.35,
    contour_thickness: int = 3,
    color_indices: List[int] | None = None,
) -> Image.Image:
    """
    Create overlay with multiple mask contours, each in a distinct color.

    Args:
        image_rgb_uint8: Original image as RGB uint8 numpy array.
        masks: List of (mask_array, label) tuples.
        alpha: Opacity for semi-transparent mask fill (0.0–1.0).
        contour_thickness: Width of contour lines in pixels.
        color_indices: Optional list of stable color indices per mask.
            If None, sequential indexing is used.

    Returns:
        PIL Image with filled masks and contour outlines in different colors.
    """
    img = Image.fromarray(image_rgb_uint8).convert("RGB")
    out_np = np.array(img, dtype=np.float32)

    for i, (mask_01, label) in enumerate(masks):
        cidx = color_indices[i] if color_indices else i
        color = SEGMENT_COLORS[cidx % len(SEGMENT_COLORS)]
        color_arr = np.array(color, dtype=np.float32)

        # Semi-transparent fill
        fill_mask = mask_01 > 0
        out_np[fill_mask] = (1 - alpha) * out_np[fill_mask] + alpha * color_arr

        # Opaque contour outline
        contour = _mask_contour(mask_01, thickness=contour_thickness)
        out_np[contour > 0] = color_arr

    out = Image.fromarray(out_np.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(out)

    for i, (mask_01, label) in enumerate(masks):
        cidx = color_indices[i] if color_indices else i
        color = SEGMENT_COLORS[cidx % len(SEGMENT_COLORS)]
        mask = (mask_01 > 0).astype(np.uint8)
        ys, xs = np.where(mask == 1)
        if len(xs) == 0 or len(ys) == 0:
            continue
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        # Label near top-left of bbox
        draw.rectangle([x0, y0, x0 + len(label) * 8 + 8, y0 + 18], fill=color)
        draw.text((x0 + 4, y0 + 2), label, fill=(255, 255, 255))

    return out
