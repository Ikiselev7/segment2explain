"""Image upload, storage, overlay rendering, and contour extraction."""

from __future__ import annotations

import logging
import uuid
from io import BytesIO
from typing import Any

import cv2
import numpy as np
from PIL import Image

from backend.dicom import is_dicom, parse_dicom
from tools.overlay import SEGMENT_COLORS, overlay_multiple_masks

logger = logging.getLogger(__name__)

# In-memory image store: {image_id: {"array": np.ndarray, "pil": PIL.Image}}
_image_store: dict[str, dict[str, Any]] = {}

# Segment mask store: {image_id: {seg_id: (mask, label, color_idx)}}
_segment_masks: dict[str, dict[str, tuple[np.ndarray, str, int]]] = {}

# Attention heatmap store: {image_id: {concept: heatmap_2d_array}}
_heatmap_store: dict[str, dict[str, np.ndarray]] = {}


def store_image(img_bytes: bytes) -> tuple[str, int, int, tuple[float, float] | None]:
    """Store an uploaded image. Returns (image_id, width, height, pixel_spacing).

    pixel_spacing is (row_mm, col_mm) extracted from DICOM, or None for non-DICOM.
    """
    pixel_spacing: tuple[float, float] | None = None

    if is_dicom(img_bytes):
        img_np, pixel_spacing = parse_dicom(img_bytes)
        pil_img = Image.fromarray(img_np)
    else:
        pil_img = Image.open(BytesIO(img_bytes))
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        img_np = np.array(pil_img)

    image_id = str(uuid.uuid4())
    _image_store[image_id] = {
        "array": img_np,
        "pil": pil_img,
        "pixel_spacing": pixel_spacing,
    }
    h, w = img_np.shape[:2]
    logger.info("Stored image %s: %dx%d pixel_spacing=%s", image_id, w, h, pixel_spacing)
    return image_id, w, h, pixel_spacing


def get_image_array(image_id: str) -> np.ndarray | None:
    """Get stored image as numpy array, or None if not found."""
    entry = _image_store.get(image_id)
    return entry["array"] if entry else None


def get_image_pil(image_id: str) -> Image.Image | None:
    """Get stored image as PIL Image, or None if not found."""
    entry = _image_store.get(image_id)
    return entry["pil"] if entry else None


def get_image_bytes(image_id: str) -> bytes | None:
    """Get stored image as PNG bytes, or None if not found."""
    pil_img = get_image_pil(image_id)
    if pil_img is None:
        return None
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def get_pixel_spacing(image_id: str) -> tuple[float, float] | None:
    """Get pixel spacing (row_mm, col_mm) for image, or None if unavailable."""
    entry = _image_store.get(image_id)
    if entry is None:
        return None
    return entry.get("pixel_spacing")


def store_segment_masks(
    image_id: str,
    masks_and_labels: dict[str, tuple[np.ndarray, str, int]],
) -> None:
    """Store segment masks for overlay rendering."""
    _segment_masks[image_id] = masks_and_labels


def render_overlay(
    image_id: str,
    segments: dict[str, dict] | None = None,
    highlight: str | None = None,
) -> bytes | None:
    """Render overlay PNG with segment fills, contours and labels.

    Uses stored segment masks from the pipeline if available.
    Returns PNG bytes, or None if image not found.
    """
    img_np = get_image_array(image_id)
    if img_np is None:
        return None

    # Use stored masks from pipeline (populated by WS adapter)
    stored = _segment_masks.get(image_id, {})
    if not stored and (not segments or not segments):
        buf = BytesIO()
        Image.fromarray(img_np).save(buf, format="PNG")
        return buf.getvalue()

    if stored:
        mask_label_pairs = [(mask, label) for mask, label, _ in stored.values()]
        color_indices = [cidx for _, _, cidx in stored.values()]
    else:
        mask_label_pairs = [(seg["mask"], f"Seg {sid}") for sid, seg in segments.items()]
        color_indices = None

    overlay_img = overlay_multiple_masks(img_np, mask_label_pairs, color_indices=color_indices)
    buf = BytesIO()
    overlay_img.save(buf, format="PNG")
    return buf.getvalue()


def mask_to_contour_points(mask: np.ndarray) -> list[list[list[int]]]:
    """Extract simplified contour polygons from binary mask.

    Returns a list of polygons, where each polygon is a list of [x, y]
    points. Multiple polygons occur when the mask has disconnected regions.
    Each polygon is simplified with Douglas-Peucker.
    """
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[list[int]]] = []
    for contour in contours:
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape(-1, 2).tolist()
        if len(points) >= 3:
            polygons.append(points)
    return polygons


def segment_color_hex(index: int) -> str:
    """Get hex color for segment at given index."""
    rgb = SEGMENT_COLORS[index % len(SEGMENT_COLORS)]
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def store_heatmaps(image_id: str, heatmaps: dict[str, np.ndarray]) -> None:
    """Store attention heatmaps for visualization."""
    _heatmap_store[image_id] = heatmaps


def get_heatmap_concepts(image_id: str) -> list[str]:
    """Get list of concept names with stored heatmaps."""
    return list(_heatmap_store.get(image_id, {}).keys())


def render_heatmap_png(image_id: str, concept: str) -> bytes | None:
    """Render a colorized heatmap PNG for a concept.

    Resizes the heatmap to match the original image dimensions and
    applies JET colormap. Returns PNG bytes, or None if not found.
    """
    heatmaps = _heatmap_store.get(image_id)
    if heatmaps is None or concept not in heatmaps:
        return None
    img_np = get_image_array(image_id)
    if img_np is None:
        return None

    heatmap = heatmaps[concept]
    h, w = img_np.shape[:2]

    # Resize and apply colormap
    resized = cv2.resize(heatmap.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    # Normalize to [0, 255]
    vmin, vmax = resized.min(), resized.max()
    if vmax - vmin > 1e-8:
        resized = (resized - vmin) / (vmax - vmin)
    colored = cv2.applyColorMap((resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    # BGR → RGB
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    buf = BytesIO()
    Image.fromarray(colored).save(buf, format="PNG")
    return buf.getvalue()


def cleanup_image(image_id: str) -> None:
    """Remove stored image to free memory."""
    _image_store.pop(image_id, None)
    _segment_masks.pop(image_id, None)
    _heatmap_store.pop(image_id, None)
