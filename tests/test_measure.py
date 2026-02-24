"""Tests for tools/measure.py — pixel and mm-based measurements."""

from __future__ import annotations

import numpy as np
import pytest

from tools.measure import measure_mask


class TestMeasureMaskPixelOnly:
    def test_basic_rectangle(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 30:60] = 1  # 20x30 rectangle
        m = measure_mask(mask)

        assert m["area_px"] == 20 * 30
        assert m["bbox_px"] == [30, 20, 59, 39]
        assert m["max_diameter_px"] > 0
        assert len(m["centroid_px"]) == 2
        # No mm fields when pixel_spacing not provided
        assert "area_mm2" not in m
        assert "max_diameter_mm" not in m
        assert "pixel_spacing_mm" not in m

    def test_empty_mask(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        m = measure_mask(mask)
        assert m["area_px"] == 0


class TestMeasureMaskWithSpacing:
    def test_mm_fields_computed(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:30] = 1  # 20x20 pixel square
        pixel_spacing = (0.5, 0.5)  # 0.5mm per pixel

        m = measure_mask(mask, pixel_spacing=pixel_spacing)

        # Pixel fields still present
        assert m["area_px"] == 400
        # mm fields
        assert m["area_mm2"] == pytest.approx(400 * 0.5 * 0.5)  # 100 mm²
        assert m["max_diameter_mm"] > 0
        assert m["pixel_spacing_mm"] == [0.5, 0.5]

    def test_asymmetric_spacing(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0:10, 0:20] = 1  # 10 rows x 20 cols
        pixel_spacing = (0.3, 0.5)  # row=0.3mm, col=0.5mm

        m = measure_mask(mask, pixel_spacing=pixel_spacing)

        assert m["area_mm2"] == pytest.approx(200 * 0.3 * 0.5)  # 30 mm²
        # Diameter: dx=19*0.5=9.5, dy=9*0.3=2.7 → sqrt(90.25+7.29)=sqrt(97.54)
        expected_diam = ((19 * 0.5) ** 2 + (9 * 0.3) ** 2) ** 0.5
        assert m["max_diameter_mm"] == pytest.approx(expected_diam, abs=0.1)
        assert m["pixel_spacing_mm"] == [0.3, 0.5]

    def test_none_spacing_no_mm_fields(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:20, 10:20] = 1
        m = measure_mask(mask, pixel_spacing=None)

        assert "area_mm2" not in m
        assert "max_diameter_mm" not in m
