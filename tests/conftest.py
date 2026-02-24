"""Pytest configuration and shared fixtures for segment2explain-poc tests."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_cxr_vqa_dir(fixtures_dir: Path) -> Path:
    """Return path to VinDr-CXR-VQA sample data directory."""
    return fixtures_dir / "sample_cxr_vqa"


@pytest.fixture
def sample_image_np() -> np.ndarray:
    """Create a simple RGB test image as numpy array."""
    # Create a 512x512 RGB image with some structure
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    # Add some "abnormality" - a bright square
    img[200:300, 200:300] = [200, 200, 200]
    return img


@pytest.fixture
def sample_bbox() -> tuple[int, int, int, int]:
    """Return a sample bounding box (xmin, ymin, xmax, ymax)."""
    return (200, 200, 299, 299)


@pytest.fixture
def sample_mask() -> np.ndarray:
    """Create a simple binary mask."""
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[200:300, 200:300] = 1
    return mask


@pytest.fixture
def sample_measurements() -> dict[str, Any]:
    """Return sample measurement data matching tools/measure.py output."""
    return {
        "area_px": 10000,
        "bbox_px": [200, 200, 299, 299],
        "max_diameter_px": 140.71,
        "centroid_px": [249.5, 249.5],
    }
