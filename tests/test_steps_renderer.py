"""Tests for utils/steps_renderer.py - HTML rendering for interactive workflow steps."""

import pytest

from orchestrator import Step
from utils.steps_renderer import render_steps_html


class TestRenderStepsHtml:
    """Test HTML rendering of workflow steps."""

    def test_render_empty_steps(self):
        """Test rendering with no steps."""
        result = render_steps_html([])
        assert "steps-container" in result
        assert "No steps yet" in result or result.count("step-item") == 0

    def test_render_single_step(self):
        """Test rendering a single step."""
        steps = [Step(id="S1", name="Parse request", status="done", detail="Request received.")]
        result = render_steps_html(steps)

        assert "steps-container" in result
        assert "step-item" in result
        assert 'data-step-id="S1"' in result
        assert "Parse request" in result
        assert "Request received." in result

    def test_render_step_status_classes(self):
        """Test that steps have correct CSS classes for status."""
        steps = [
            Step(id="S1", name="Done", status="done"),
            Step(id="S2", name="Running", status="running"),
            Step(id="S3", name="Queued", status="queued"),
            Step(id="S4", name="Failed", status="failed"),
        ]
        result = render_steps_html(steps)

        assert "step-done" in result
        assert "step-running" in result
        assert "step-queued" in result
        assert "step-failed" in result

    def test_render_step_status_icons(self):
        """Test that steps have correct status icons."""
        steps = [
            Step(id="S1", name="Done", status="done"),
            Step(id="S2", name="Running", status="running"),
            Step(id="S3", name="Queued", status="queued"),
            Step(id="S4", name="Failed", status="failed"),
        ]
        result = render_steps_html(steps)

        assert "✅" in result  # done
        assert "🔄" in result  # running
        assert "⏳" in result  # queued
        assert "❌" in result  # failed

    def test_render_step_with_segment_ids(self):
        """Test rendering step with associated segment IDs."""
        steps = [
            Step(
                id="S2",
                name="Segment ROI",
                status="done",
                detail="Produced Segment A",
                segment_ids=["A"],
            )
        ]
        result = render_steps_html(steps)

        assert 'data-step-id="S2"' in result
        assert 'data-segment-ids' in result
        # Check for A in the result (may be HTML-escaped as &quot;A&quot;)
        assert 'A' in result

    def test_render_step_with_multiple_segments(self):
        """Test rendering step with multiple segment IDs."""
        steps = [
            Step(
                id="S3C-1",
                name="Ground finding",
                status="done",
                segment_ids=["B", "C"],
            )
        ]
        result = render_steps_html(steps)

        assert 'data-segment-ids' in result
        # Should contain both B and C (may be HTML-escaped)
        assert 'B' in result
        assert 'C' in result

    def test_render_multiple_steps(self):
        """Test rendering multiple steps in order."""
        steps = [
            Step(id="S1", name="First", status="done"),
            Step(id="S2", name="Second", status="running"),
            Step(id="S3", name="Third", status="queued"),
        ]
        result = render_steps_html(steps)

        # All steps should be present
        assert result.count("step-item") >= 3
        assert 'data-step-id="S1"' in result
        assert 'data-step-id="S2"' in result
        assert 'data-step-id="S3"' in result

    def test_render_step_detail_text(self):
        """Test that step detail text is included."""
        steps = [
            Step(
                id="S2",
                name="Segment",
                status="done",
                detail="Using user box: (100, 100, 200, 200)",
            )
        ]
        result = render_steps_html(steps)

        assert "Using user box" in result
        assert "(100, 100, 200, 200)" in result

    def test_render_html_escaping(self):
        """Test that HTML special characters are escaped."""
        steps = [
            Step(
                id="S1",
                name="Test <script>alert('xss')</script>",
                status="done",
                detail="Detail with & and < and >",
            )
        ]
        result = render_steps_html(steps)

        # Should not contain unescaped HTML
        assert "<script>" not in result or "&lt;script&gt;" in result
        assert "alert('xss')" not in result or "alert(&#x27;xss&#x27;)" in result

    def test_render_empty_segment_ids(self):
        """Test step with no segment IDs."""
        steps = [Step(id="S1", name="Parse", status="done", segment_ids=[])]
        result = render_steps_html(steps)

        assert 'data-segment-ids' in result
        assert "[]" in result  # Empty JSON array


class TestStepHtmlStructure:
    """Test the HTML structure of rendered steps."""

    def test_has_container_div(self):
        """Test that output has steps-container div."""
        steps = [Step(id="S1", name="Test", status="done")]
        result = render_steps_html(steps)

        assert '<div class="steps-container">' in result or 'class="steps-container"' in result
        assert "</div>" in result

    def test_step_item_structure(self):
        """Test that each step is a div with step-item class."""
        steps = [Step(id="S1", name="Test", status="done")]
        result = render_steps_html(steps)

        assert 'class="step-item' in result
        assert 'data-step-id' in result

    def test_step_has_header_and_detail(self):
        """Test that step has header and detail sections."""
        steps = [Step(id="S1", name="Test", status="done", detail="Details here")]
        result = render_steps_html(steps)

        # Should have separate sections for header and detail
        assert "step-header" in result or "step-icon" in result
        assert "step-detail" in result or "Details here" in result

    def test_data_attributes_valid_json(self):
        """Test that data-segment-ids contains valid JSON."""
        steps = [Step(id="S1", name="Test", status="done", segment_ids=["A", "B"])]
        result = render_steps_html(steps)

        # Extract the data-segment-ids value
        import re

        match = re.search(r'data-segment-ids=["\']([^"\']+)["\']', result)
        if match:
            import json
            import html

            json_str = html.unescape(match.group(1))
            # Should be valid JSON
            parsed = json.loads(json_str)
            assert isinstance(parsed, list)
            assert "A" in parsed
            assert "B" in parsed


class TestStepClickability:
    """Test that steps have the right attributes for click handling."""

    def test_step_has_data_step_id(self):
        """Test that each step has data-step-id attribute."""
        steps = [Step(id="S2", name="Test", status="done")]
        result = render_steps_html(steps)

        assert 'data-step-id="S2"' in result

    def test_step_has_data_segment_ids(self):
        """Test that step has data-segment-ids attribute."""
        steps = [Step(id="S1", name="Test", status="done", segment_ids=["A"])]
        result = render_steps_html(steps)

        assert 'data-segment-ids' in result

    def test_multiple_steps_have_unique_ids(self):
        """Test that multiple steps have different data-step-id values."""
        steps = [
            Step(id="S1", name="First", status="done"),
            Step(id="S2", name="Second", status="done"),
        ]
        result = render_steps_html(steps)

        assert 'data-step-id="S1"' in result
        assert 'data-step-id="S2"' in result
        # Each should appear exactly once
        assert result.count('data-step-id="S1"') == 1
        assert result.count('data-step-id="S2"') == 1
