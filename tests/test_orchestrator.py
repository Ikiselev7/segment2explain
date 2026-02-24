"""Tests for orchestrator.py - state management and rendering."""

import numpy as np
import pytest

from orchestrator import JobState, Step, create_job_state, render_steps_markdown, ensure_rgb_uint8


class TestJobState:
    """Test JobState dataclass functionality."""

    def test_create_job_state(self):
        """Test creating a new JobState."""
        state = create_job_state()
        assert state.job_id is None
        assert state.image is None
        assert state.chat == []
        assert state.steps == []
        assert state.segments == {}
        assert state.highlight == "ALL"
        assert state._seg_counter == 0

    def test_next_segment_id(self):
        """Test segment ID generation (A, B, C, ...)."""
        state = create_job_state()
        assert state.next_segment_id() == "A"
        assert state.next_segment_id() == "B"
        assert state.next_segment_id() == "C"
        # After Z, should use S0, S1, ...
        state._seg_counter = 26
        assert state.next_segment_id() == "S26"

    def test_add_segment(self, sample_mask, sample_bbox, sample_measurements):
        """Test adding a segment to state."""
        state = create_job_state()
        state.add_segment(
            segment_id="A",
            label="Test region",
            mask=sample_mask,
            bbox=sample_bbox,
            created_by_step="S2",
            measurements=sample_measurements,
        )

        assert "A" in state.segments
        seg = state.segments["A"]
        assert seg["segment_id"] == "A"
        assert seg["label"] == "Test region"
        assert seg["bbox"] == sample_bbox
        assert seg["created_by_step"] == "S2"
        assert seg["measurements"] == sample_measurements
        assert seg["mask"].dtype == np.float32  # Should be converted


class TestStep:
    """Test Step dataclass."""

    def test_step_creation(self):
        """Test creating a Step with default values."""
        step = Step(id="S1", name="Parse request")
        assert step.id == "S1"
        assert step.name == "Parse request"
        assert step.status == "queued"
        assert step.detail == ""
        assert step.segment_ids == []

    def test_step_with_segments(self):
        """Test Step with segment IDs."""
        step = Step(
            id="S2",
            name="Segment ROI",
            status="done",
            detail="Produced Segment A",
            segment_ids=["A"],
        )
        assert step.segment_ids == ["A"]
        assert step.status == "done"


class TestRenderStepsMarkdown:
    """Test steps rendering to markdown."""

    def test_render_empty_steps(self):
        """Test rendering with no steps."""
        result = render_steps_markdown([])
        assert result == "_No steps yet._"

    def test_render_single_step(self):
        """Test rendering a single step."""
        steps = [Step(id="S1", name="Parse request", status="done", detail="Request received.")]
        result = render_steps_markdown(steps)
        assert "S1" in result
        assert "Parse request" in result
        assert "✅" in result  # done icon
        assert "Request received." in result

    def test_render_step_with_segments(self):
        """Test rendering step with segment IDs."""
        steps = [
            Step(
                id="S2",
                name="Segment ROI",
                status="done",
                detail="Using user box",
                segment_ids=["A"],
            )
        ]
        result = render_steps_markdown(steps)
        assert "Segment A" in result

    def test_render_multiple_steps_with_statuses(self):
        """Test rendering steps with different statuses."""
        steps = [
            Step(id="S1", name="Parse", status="done"),
            Step(id="S2", name="Segment", status="running"),
            Step(id="S3", name="Explain", status="queued"),
            Step(id="S4", name="Failed", status="failed", detail="Error occurred"),
        ]
        result = render_steps_markdown(steps)
        assert "✅" in result  # done
        assert "🔄" in result  # running
        assert "⏳" in result  # queued
        assert "❌" in result  # failed


class TestEnsureRgbUint8:
    """Test ensure_rgb_uint8 image conversion."""

    def test_already_rgb_uint8(self, sample_image_np):
        """Test with already correct format."""
        result = ensure_rgb_uint8(sample_image_np)
        assert result.dtype == np.uint8
        assert result.shape == (512, 512, 3)
        np.testing.assert_array_equal(result, sample_image_np)

    def test_grayscale_conversion(self):
        """Test converting grayscale to RGB."""
        gray = np.ones((100, 100), dtype=np.uint8) * 128
        result = ensure_rgb_uint8(gray)
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8
        # All channels should be same value
        assert np.all(result[:, :, 0] == 128)
        assert np.all(result[:, :, 1] == 128)
        assert np.all(result[:, :, 2] == 128)

    def test_rgba_conversion(self):
        """Test converting RGBA to RGB (drop alpha)."""
        rgba = np.ones((100, 100, 4), dtype=np.uint8) * 200
        result = ensure_rgb_uint8(rgba)
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_float_to_uint8(self):
        """Test converting float to uint8."""
        float_img = np.ones((100, 100, 3), dtype=np.float32) * 0.5  # 0-1 range
        result = ensure_rgb_uint8(float_img)
        assert result.dtype == np.uint8
        # Note: values will be clipped to 0-255, so 0.5 -> 0 in naive conversion
        # The function clips to 0-255 range
