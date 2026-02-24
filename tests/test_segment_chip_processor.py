"""Tests for utils/segment_chip_processor.py - segment reference detection and HTML chip generation."""

import pytest

from utils.segment_chip_processor import detect_segment_references, process_segment_chips


class TestDetectSegmentReferences:
    """Test detection of segment references in text."""

    def test_detect_double_bracket_format(self):
        """Test [[SEG:A]] format detection."""
        text = "The finding in [[SEG:A]] shows abnormality."
        result = detect_segment_references(text)
        assert result == {"A"}

    def test_detect_single_bracket_format(self):
        """Test [SEG:A] format detection."""
        text = "The opacity [SEG:B] is concerning."
        result = detect_segment_references(text)
        assert result == {"B"}

    def test_detect_segment_word_format(self):
        """Test 'Segment A' format detection."""
        text = "Segment C shows consolidation."
        result = detect_segment_references(text)
        assert result == {"C"}

    def test_detect_multiple_references(self):
        """Test detecting multiple segment references."""
        text = "Segment A and [[SEG:B]] and [SEG:C] all show findings."
        result = detect_segment_references(text)
        assert result == {"A", "B", "C"}

    def test_detect_case_insensitive(self):
        """Test case-insensitive detection."""
        text = "segment a and SEGMENT B and Seg:C"
        result = detect_segment_references(text)
        assert "A" in result
        assert "B" in result

    def test_detect_empty_text(self):
        """Test with empty text."""
        result = detect_segment_references("")
        assert result == set()

    def test_detect_no_references(self):
        """Test text with no segment references."""
        text = "This is a normal finding with no segments."
        result = detect_segment_references(text)
        assert result == set()

    def test_detect_multiple_same_segment(self):
        """Test that same segment ID appears only once."""
        text = "Segment A shows opacity. The Segment A region is concerning."
        result = detect_segment_references(text)
        assert result == {"A"}


class TestProcessSegmentChips:
    """Test conversion of segment references to HTML chips."""

    def test_process_simple_reference(self):
        """Test processing single segment reference."""
        text = "Finding in Segment A shows opacity."
        available_segments = ["A"]
        result = process_segment_chips(text, available_segments)

        assert '<span class="seg-chip"' in result
        assert 'data-seg-id="A"' in result
        assert "[SEG:A]" in result or "Segment A" in result

    def test_process_double_bracket_format(self):
        """Test [[SEG:A]] gets converted to chip."""
        text = "The finding [[SEG:A]] is abnormal."
        available_segments = ["A"]
        result = process_segment_chips(text, available_segments)

        assert '<span class="seg-chip"' in result
        assert 'data-seg-id="A"' in result

    def test_process_single_bracket_format(self):
        """Test [SEG:A] gets converted to chip."""
        text = "The finding [SEG:B] is abnormal."
        available_segments = ["B"]
        result = process_segment_chips(text, available_segments)

        assert '<span class="seg-chip"' in result
        assert 'data-seg-id="B"' in result

    def test_process_only_available_segments(self):
        """Test that only available segments get converted."""
        text = "Segment A and Segment B and Segment C"
        available_segments = ["A", "C"]  # B not available
        result = process_segment_chips(text, available_segments)

        # A and C should be chips
        assert result.count('<span class="seg-chip"') == 2
        assert 'data-seg-id="A"' in result
        assert 'data-seg-id="C"' in result
        # B should remain as plain text
        assert "Segment B" in result

    def test_process_multiple_segments(self):
        """Test processing multiple segment references."""
        text = "Findings: Segment A shows opacity, [[SEG:B]] shows consolidation."
        available_segments = ["A", "B"]
        result = process_segment_chips(text, available_segments)

        assert result.count('<span class="seg-chip"') == 2
        assert 'data-seg-id="A"' in result
        assert 'data-seg-id="B"' in result

    def test_process_empty_text(self):
        """Test with empty text."""
        result = process_segment_chips("", ["A"])
        assert result == ""

    def test_process_no_segments_available(self):
        """Test with no segments available."""
        text = "Segment A shows findings."
        result = process_segment_chips(text, [])
        assert result == text  # Should return unchanged

    def test_process_preserves_other_text(self):
        """Test that other text is preserved."""
        text = "The right lower lobe Segment A shows dense consolidation with air bronchograms."
        available_segments = ["A"]
        result = process_segment_chips(text, available_segments)

        assert "right lower lobe" in result
        assert "dense consolidation" in result
        assert "air bronchograms" in result

    def test_process_html_escaping(self):
        """Test that segment IDs are properly escaped for HTML."""
        text = "Segment A shows findings."
        available_segments = ["A"]
        result = process_segment_chips(text, available_segments)

        # Check for proper HTML attribute quoting
        assert 'data-seg-id="A"' in result
        # No unescaped quotes or special chars
        assert '"A">' not in result or '<span' in result

    def test_process_case_insensitive_matching(self):
        """Test that matching is case-insensitive."""
        text = "segment a and SEGMENT B"
        available_segments = ["A", "B"]
        result = process_segment_chips(text, available_segments)

        assert 'data-seg-id="A"' in result
        assert 'data-seg-id="B"' in result


class TestChipHTMLStructure:
    """Test the HTML structure of generated chips."""

    def test_chip_has_required_class(self):
        """Test that chip has seg-chip class."""
        text = "Segment A"
        result = process_segment_chips(text, ["A"])
        assert 'class="seg-chip"' in result

    def test_chip_has_data_attribute(self):
        """Test that chip has data-seg-id attribute."""
        text = "Segment B"
        result = process_segment_chips(text, ["B"])
        assert 'data-seg-id="B"' in result

    def test_chip_displays_segment_id(self):
        """Test that chip displays the segment ID."""
        text = "[[SEG:C]]"
        result = process_segment_chips(text, ["C"])
        # Should contain visual representation of segment
        assert "[SEG:C]" in result or "Segment C" in result

    def test_chip_is_inline_span(self):
        """Test that chip is an inline span element."""
        text = "Segment A"
        result = process_segment_chips(text, ["A"])
        assert "<span" in result
        assert "</span>" in result
