"""Tests for prompts/templates.py - tool result message builder."""

from prompts.templates import build_tool_result_message


class TestBuildToolResultMessage:
    """Test tool result message builder for R2 evidence."""

    def test_returns_string(self):
        result = build_tool_result_message("A", "opacity")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_segment_id(self):
        result = build_tool_result_message("B", "nodule")
        assert "B" in result
        assert "Segment B" in result

    def test_includes_label(self):
        result = build_tool_result_message("A", "enlarged cardiac silhouette")
        assert "enlarged cardiac silhouette" in result

    def test_includes_color_when_provided(self):
        result = build_tool_result_message("A", "heart", "red")
        assert "red" in result

    def test_no_color_when_omitted(self):
        result = build_tool_result_message("A", "heart")
        assert result == "- Segment A: heart"

    def test_format_with_color(self):
        result = build_tool_result_message("C", "effusion", "blue")
        assert result == "- Segment C (blue): effusion"

    def test_format_with_description(self):
        result = build_tool_result_message("A", "nodule", "red", "pulmonary nodule in right upper lobe")
        assert result == "- Segment A (red): nodule — pulmonary nodule in right upper lobe"

    def test_description_without_color(self):
        result = build_tool_result_message("B", "heart", description="enlarged cardiac silhouette")
        assert result == "- Segment B: heart — enlarged cardiac silhouette"

    def test_no_description(self):
        result = build_tool_result_message("A", "heart", "red", "")
        assert result == "- Segment A (red): heart"
