"""Tests for overlay_multiple_masks in tools/overlay.py."""

import numpy as np
from PIL import Image

from tools.overlay import SEGMENT_COLORS, overlay_multiple_masks


class TestOverlayMultipleMasks:
    """Test multi-color overlay rendering."""

    def _make_image(self, w=200, h=200):
        return np.zeros((h, w, 3), dtype=np.uint8) + 128  # gray

    def _make_mask(self, h=200, w=200, x0=10, y0=10, x1=50, y1=50):
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y0:y1, x0:x1] = 1
        return mask

    def test_returns_pil_image(self):
        img = self._make_image()
        mask = self._make_mask()
        result = overlay_multiple_masks(img, [(mask, "A")])
        assert isinstance(result, Image.Image)

    def test_correct_size(self):
        img = self._make_image(300, 400)
        mask = self._make_mask(400, 300)
        result = overlay_multiple_masks(img, [(mask, "A")])
        assert result.size == (300, 400)

    def test_empty_masks_list(self):
        img = self._make_image()
        result = overlay_multiple_masks(img, [])
        assert isinstance(result, Image.Image)

    def test_multiple_masks_different_colors(self):
        img = self._make_image()
        mask_a = self._make_mask(x0=10, y0=10, x1=50, y1=50)
        mask_b = self._make_mask(x0=100, y0=100, x1=150, y1=150)
        result = overlay_multiple_masks(img, [(mask_a, "A"), (mask_b, "B")])
        result_np = np.array(result)
        # Contour pixels at mask edge should have different colors
        pixel_a = result_np[10, 10]  # edge of mask A
        pixel_b = result_np[100, 100]  # edge of mask B
        assert not np.array_equal(pixel_a, pixel_b)

    def test_many_masks_wraps_colors(self):
        img = self._make_image()
        masks = []
        for i in range(len(SEGMENT_COLORS) + 2):
            m = self._make_mask(x0=i * 5, y0=i * 5, x1=i * 5 + 10, y1=i * 5 + 10)
            masks.append((m, f"S{i}"))
        result = overlay_multiple_masks(img, masks)
        assert isinstance(result, Image.Image)


class TestSegmentColors:
    """Test color palette."""

    def test_has_at_least_10_colors(self):
        assert len(SEGMENT_COLORS) >= 10

    def test_colors_are_rgb_tuples(self):
        for color in SEGMENT_COLORS:
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)
