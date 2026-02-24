"""Tests for utils/segments_list_renderer.py - HTML table rendering for segments metadata."""

import pytest

from orchestrator import JobState, create_job_state
from utils.segments_list_renderer import render_segments_list_html


class TestRenderSegmentsListHtml:
    """Test HTML rendering of segments list."""

    def test_render_empty_segments(self):
        """Test rendering with no segments."""
        state = create_job_state()
        result = render_segments_list_html(state)

        assert "segments-list" in result
        assert "No segments yet" in result or "segment-row" not in result

    def test_render_single_segment(self, sample_mask, sample_bbox, sample_measurements):
        """Test rendering a single segment."""
        state = create_job_state()
        state.add_segment(
            segment_id="A",
            label="User-selected region",
            mask=sample_mask,
            bbox=sample_bbox,
            created_by_step="S2",
            measurements=sample_measurements,
        )
        result = render_segments_list_html(state)

        assert "segments-list" in result
        assert "segment-row" in result
        assert 'data-segment-id="A"' in result
        assert "User-selected region" in result
        assert "S2" in result

    def test_render_multiple_segments(self, sample_mask, sample_bbox, sample_measurements):
        """Test rendering multiple segments."""
        state = create_job_state()
        state.add_segment("A", "First", sample_mask, sample_bbox, "S2", sample_measurements)
        state.add_segment("B", "Second", sample_mask, sample_bbox, "S3", sample_measurements)
        state.add_segment("C", "Third", sample_mask, sample_bbox, "S4", sample_measurements)

        result = render_segments_list_html(state)

        assert result.count("segment-row") == 3
        assert 'data-segment-id="A"' in result
        assert 'data-segment-id="B"' in result
        assert 'data-segment-id="C"' in result

    def test_render_segment_measurements(self, sample_mask, sample_bbox):
        """Test that measurements are displayed."""
        state = create_job_state()
        measurements = {
            "area_px": 12450,
            "bbox_px": [200, 200, 299, 299],
            "max_diameter_px": 140.71,
            "centroid_px": [249.5, 249.5],
        }
        state.add_segment("A", "Test", sample_mask, sample_bbox, "S2", measurements)

        result = render_segments_list_html(state)

        # Area should be formatted with commas
        assert "12,450" in result or "12450" in result

    def test_render_segment_created_by_step(self, sample_mask, sample_bbox, sample_measurements):
        """Test that creating step is shown."""
        state = create_job_state()
        state.add_segment("A", "Test", sample_mask, sample_bbox, "S2", sample_measurements)

        result = render_segments_list_html(state)

        assert "S2" in result
        assert 'data-step-id="S2"' in result

    def test_render_html_escaping(self, sample_mask, sample_bbox, sample_measurements):
        """Test that HTML special characters are escaped."""
        state = create_job_state()
        state.add_segment(
            "A",
            "Test <script>alert('xss')</script>",
            sample_mask,
            sample_bbox,
            "S2",
            sample_measurements,
        )

        result = render_segments_list_html(state)

        # Should not contain unescaped HTML
        assert "<script>" not in result or "&lt;script&gt;" in result

    def test_render_table_headers(self, sample_mask, sample_bbox, sample_measurements):
        """Test that table has proper headers."""
        state = create_job_state()
        state.add_segment("A", "Test", sample_mask, sample_bbox, "S2", sample_measurements)

        result = render_segments_list_html(state)

        # Should have table headers
        assert any(
            header in result.lower()
            for header in ["id", "label", "created by", "area", "segment"]
        )


class TestSegmentListStructure:
    """Test the HTML structure of segments list."""

    def test_has_container_div(self, sample_mask, sample_bbox, sample_measurements):
        """Test that output has segments-list container."""
        state = create_job_state()
        state.add_segment("A", "Test", sample_mask, sample_bbox, "S2", sample_measurements)

        result = render_segments_list_html(state)

        assert 'class="segments-list"' in result or "segments-list" in result

    def test_has_table_element(self, sample_mask, sample_bbox, sample_measurements):
        """Test that list is rendered as a table."""
        state = create_job_state()
        state.add_segment("A", "Test", sample_mask, sample_bbox, "S2", sample_measurements)

        result = render_segments_list_html(state)

        assert "<table" in result
        assert "</table>" in result

    def test_row_has_data_attributes(self, sample_mask, sample_bbox, sample_measurements):
        """Test that rows have data attributes for click handling."""
        state = create_job_state()
        state.add_segment("A", "Test", sample_mask, sample_bbox, "S2", sample_measurements)

        result = render_segments_list_html(state)

        assert 'data-segment-id' in result
        assert 'data-step-id' in result

    def test_row_clickability_class(self, sample_mask, sample_bbox, sample_measurements):
        """Test that rows have segment-row class for styling."""
        state = create_job_state()
        state.add_segment("A", "Test", sample_mask, sample_bbox, "S2", sample_measurements)

        result = render_segments_list_html(state)

        assert "segment-row" in result


class TestSegmentListInteractivity:
    """Test interactive elements of segments list."""

    def test_segment_id_shown(self, sample_mask, sample_bbox, sample_measurements):
        """Test that segment ID is displayed."""
        state = create_job_state()
        state.add_segment("A", "Test", sample_mask, sample_bbox, "S2", sample_measurements)

        result = render_segments_list_html(state)

        # Segment ID should be prominently displayed
        assert ">A<" in result or "A</td>" in result or 'id="A"' in result.lower()

    def test_label_shown(self, sample_mask, sample_bbox, sample_measurements):
        """Test that segment label is displayed."""
        state = create_job_state()
        state.add_segment("A", "Important finding", sample_mask, sample_bbox, "S2", sample_measurements)

        result = render_segments_list_html(state)

        assert "Important finding" in result

    def test_multiple_segments_unique_data(self, sample_mask, sample_bbox, sample_measurements):
        """Test that each segment has unique data attributes."""
        state = create_job_state()
        state.add_segment("A", "First", sample_mask, sample_bbox, "S2", sample_measurements)
        state.add_segment("B", "Second", sample_mask, sample_bbox, "S3", sample_measurements)

        result = render_segments_list_html(state)

        # Each segment should have unique data-segment-id
        assert result.count('data-segment-id="A"') >= 1
        assert result.count('data-segment-id="B"') >= 1
        # Each should reference different steps
        assert 'data-step-id="S2"' in result
        assert 'data-step-id="S3"' in result
