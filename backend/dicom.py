"""DICOM file parsing: pixel spacing extraction and windowed 8-bit conversion."""

from __future__ import annotations

import logging
from io import BytesIO

import numpy as np

logger = logging.getLogger(__name__)

# DICOM magic: "DICM" at byte offset 128
_DICM_OFFSET = 128
_DICM_MAGIC = b"DICM"


def is_dicom(data: bytes) -> bool:
    """Check if raw bytes look like a DICOM file."""
    if len(data) > _DICM_OFFSET + 4:
        if data[_DICM_OFFSET : _DICM_OFFSET + 4] == _DICM_MAGIC:
            return True
    return False


def parse_dicom(data: bytes) -> tuple[np.ndarray, tuple[float, float] | None]:
    """Parse DICOM bytes into an RGB uint8 image and optional pixel spacing.

    Returns:
        (rgb_uint8_array, pixel_spacing_mm)
        pixel_spacing_mm is (row_mm, col_mm) or None if not available.
    """
    import pydicom

    ds = pydicom.dcmread(BytesIO(data))
    pixel_array = ds.pixel_array.astype(np.float64)

    # --- Extract pixel spacing ---
    pixel_spacing: tuple[float, float] | None = None
    for attr in ("PixelSpacing", "ImagerPixelSpacing"):
        val = getattr(ds, attr, None)
        if val is not None:
            try:
                row_mm, col_mm = float(val[0]), float(val[1])
                pixel_spacing = (row_mm, col_mm)
                logger.info("DICOM %s: row=%.4f mm, col=%.4f mm", attr, row_mm, col_mm)
                break
            except (IndexError, TypeError, ValueError):
                continue

    # --- Apply Rescale Slope / Intercept (→ Hounsfield units for CT) ---
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    if slope != 1.0 or intercept != 0.0:
        pixel_array = pixel_array * slope + intercept

    # --- Windowing ---
    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)

    if wc is not None and ww is not None:
        # Handle multi-value window (take first)
        if hasattr(wc, "__iter__") and not isinstance(wc, str):
            wc = float(wc[0])
        else:
            wc = float(wc)
        if hasattr(ww, "__iter__") and not isinstance(ww, str):
            ww = float(ww[0])
        else:
            ww = float(ww)

        lo = wc - ww / 2.0
        hi = wc + ww / 2.0
    else:
        # Percentile fallback
        lo = float(np.percentile(pixel_array, 0.5))
        hi = float(np.percentile(pixel_array, 99.5))

    if hi <= lo:
        hi = lo + 1.0

    # Map to 0-255
    img_f = np.clip((pixel_array - lo) / (hi - lo), 0.0, 1.0)
    img_u8 = (img_f * 255.0).astype(np.uint8)

    # Convert to RGB
    if img_u8.ndim == 2:
        rgb = np.stack([img_u8, img_u8, img_u8], axis=-1)
    elif img_u8.ndim == 3 and img_u8.shape[-1] == 1:
        rgb = np.repeat(img_u8, 3, axis=-1)
    elif img_u8.ndim == 3 and img_u8.shape[-1] >= 3:
        rgb = img_u8[..., :3]
    else:
        rgb = np.stack([img_u8, img_u8, img_u8], axis=-1)

    logger.info(
        "DICOM parsed: %dx%d, pixel_spacing=%s",
        rgb.shape[1], rgb.shape[0],
        pixel_spacing,
    )
    return rgb, pixel_spacing
