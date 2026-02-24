"""Tests for the refined segmentation pipeline."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from tools.refined_segmentation import (
    _dedup_candidates,
    mask_area_pct,
    mask_iou,
    refined_segment,
)


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------

class TestMaskIou:
    def test_identical_masks(self):
        m = np.zeros((64, 64), dtype=np.uint8)
        m[10:30, 10:30] = 1
        assert mask_iou(m, m) == pytest.approx(1.0)

    def test_disjoint_masks(self):
        m1 = np.zeros((64, 64), dtype=np.uint8)
        m1[0:10, 0:10] = 1
        m2 = np.zeros((64, 64), dtype=np.uint8)
        m2[50:60, 50:60] = 1
        assert mask_iou(m1, m2) == pytest.approx(0.0)

    def test_partial_overlap(self):
        m1 = np.zeros((64, 64), dtype=np.uint8)
        m1[0:20, 0:20] = 1  # 400 px
        m2 = np.zeros((64, 64), dtype=np.uint8)
        m2[10:30, 10:30] = 1  # 400 px, overlap = 10×10 = 100
        # union = 400 + 400 - 100 = 700
        assert mask_iou(m1, m2) == pytest.approx(100 / 700)

    def test_empty_masks(self):
        m = np.zeros((64, 64), dtype=np.uint8)
        assert mask_iou(m, m) == pytest.approx(0.0)


class TestMaskAreaPct:
    def test_full_mask(self):
        m = np.ones((100, 100), dtype=np.uint8)
        assert mask_area_pct(m) == pytest.approx(100.0)

    def test_quarter_mask(self):
        m = np.zeros((100, 100), dtype=np.uint8)
        m[0:50, 0:50] = 1
        assert mask_area_pct(m) == pytest.approx(25.0)

    def test_empty_mask(self):
        m = np.zeros((64, 64), dtype=np.uint8)
        assert mask_area_pct(m) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Dedup tests
# ---------------------------------------------------------------------------

class TestDedupCandidates:
    def _make_candidate(self, mask, score=0.5):
        return {"mask": mask, "score": score, "bbox": (0, 0, 10, 10)}

    def test_identical_masks_kept_higher_score(self):
        m = np.zeros((64, 64), dtype=np.uint8)
        m[10:30, 10:30] = 1
        c1 = self._make_candidate(m.copy(), score=0.3)
        c2 = self._make_candidate(m.copy(), score=0.8)
        result = _dedup_candidates([c1, c2], iou_thresh=0.85)
        assert len(result) == 1
        assert result[0]["score"] == 0.8

    def test_disjoint_masks_both_kept(self):
        m1 = np.zeros((64, 64), dtype=np.uint8)
        m1[0:10, 0:10] = 1
        m2 = np.zeros((64, 64), dtype=np.uint8)
        m2[50:60, 50:60] = 1
        c1 = self._make_candidate(m1, score=0.5)
        c2 = self._make_candidate(m2, score=0.5)
        result = _dedup_candidates([c1, c2], iou_thresh=0.85)
        assert len(result) == 2

    def test_empty_input(self):
        assert _dedup_candidates([], iou_thresh=0.85) == []


# ---------------------------------------------------------------------------
# Pipeline integration tests (mocked)
# ---------------------------------------------------------------------------

class TestRefinedSegment:
    def _make_fake_medsam3(self, results):
        """Create a MagicMock that returns given results from segment_concepts."""
        mock = MagicMock()
        mock.segment_concepts.return_value = results
        return mock

    def test_empty_concepts_returns_empty(self):
        mock = self._make_fake_medsam3([])
        result = refined_segment(np.zeros((64, 64, 3), dtype=np.uint8), [], mock)
        assert result == []

    def test_returns_results_with_score_and_area(self):
        """Pipeline should return results with score and area_pct."""
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:44, 20:44] = 1
        candidates = [{"mask": mask, "bbox": (20, 20, 43, 43), "score": 0.7, "concept": "heart"}]
        mock = self._make_fake_medsam3(candidates)

        output = refined_segment(
            np.zeros((64, 64, 3), dtype=np.uint8) + 128,
            ["heart"],
            mock,
        )

        assert len(output) == 1
        assert output[0]["score"] == 0.7
        assert "area_pct" in output[0]
        assert output[0]["concept"] == "heart"

    def test_no_candidates_returns_empty(self):
        mock = self._make_fake_medsam3([])
        output = refined_segment(
            np.zeros((64, 64, 3), dtype=np.uint8) + 128,
            ["test"],
            mock,
        )
        assert output == []

    def test_multiple_concepts_each_gets_result(self):
        """Different concepts should each get their mask."""
        mask_heart = np.zeros((64, 64), dtype=np.uint8)
        mask_heart[20:44, 20:44] = 1
        mask_lung = np.zeros((64, 64), dtype=np.uint8)
        mask_lung[5:15, 5:15] = 1

        candidates = [
            {"mask": mask_heart, "bbox": (20, 20, 43, 43), "score": 0.8, "concept": "heart"},
            {"mask": mask_lung, "bbox": (5, 5, 14, 14), "score": 0.7, "concept": "lung"},
        ]
        mock = self._make_fake_medsam3(candidates)

        output = refined_segment(
            np.zeros((64, 64, 3), dtype=np.uint8) + 128,
            ["heart", "lung"],
            mock,
        )

        assert len(output) == 2
        concepts = {r["concept"] for r in output}
        assert "heart" in concepts
        assert "lung" in concepts

    def test_overlapping_concepts_deduped(self):
        """Highly overlapping masks from different concepts get deduped."""
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:50, 10:50] = 1
        candidates = [
            {"mask": mask.copy(), "bbox": (10, 10, 49, 49), "score": 0.9, "concept": "heart"},
            {"mask": mask.copy(), "bbox": (10, 10, 49, 49), "score": 0.6, "concept": "cardiac"},
        ]
        mock = self._make_fake_medsam3(candidates)

        output = refined_segment(
            np.zeros((64, 64, 3), dtype=np.uint8) + 128,
            ["heart", "cardiac"],
            mock,
        )

        assert len(output) == 1
        assert output[0]["score"] == 0.9

    def test_calls_segment_concepts_with_max_1(self):
        """Pipeline should pass max_masks_per_concept=1 to segment_concepts."""
        mock = self._make_fake_medsam3([])

        refined_segment(
            np.zeros((64, 64, 3), dtype=np.uint8),
            ["heart", "nodule", "right lung"],
            mock,
        )

        mock.segment_concepts.assert_called_once()
        call_kwargs = mock.segment_concepts.call_args[1]
        assert call_kwargs.get("max_masks_per_concept") == 1
        call_concepts = call_kwargs.get("concepts") or mock.segment_concepts.call_args[0][1]
        assert call_concepts == ["heart", "nodule", "right lung"]
