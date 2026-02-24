"""Tests for utils/finding_injector.py - post-process text to inject segment chips after grounding."""

import pytest

from utils.finding_injector import inject_segment_chips_for_findings


class TestInjectSegmentChipsForFindings:
    """Test chip injection into completed text after grounding."""

    def test_inject_single_finding(self):
        """Test injecting chip for single finding."""
        original_text = "There is an opacity in the right lower lobe."
        findings = [
            {"label": "opacity", "description": "opacity in the right lower lobe"}
        ]
        new_segments = ["B"]  # Segment B was created for this finding

        result = inject_segment_chips_for_findings(original_text, findings, new_segments)

        # Should insert [SEG:B] chip near the finding mention
        assert "[SEG:B]" in result or "SEG:B" in result

    def test_inject_multiple_findings(self):
        """Test injecting chips for multiple findings."""
        original_text = "There is an opacity and a nodule visible."
        findings = [
            {"label": "opacity", "description": "opacity"},
            {"label": "nodule", "description": "nodule"},
        ]
        new_segments = ["B", "C"]

        result = inject_segment_chips_for_findings(original_text, findings, new_segments)

        # Should have chips for both findings
        assert "B" in result
        assert "C" in result

    def test_inject_preserves_original_text(self):
        """Test that original text content is preserved."""
        original_text = "The right lower lobe shows dense consolidation with air bronchograms."
        findings = [{"label": "consolidation", "description": "consolidation"}]
        new_segments = ["B"]

        result = inject_segment_chips_for_findings(original_text, findings, new_segments)

        # Original words should still be present
        assert "right lower lobe" in result
        assert "dense consolidation" in result
        assert "air bronchograms" in result

    def test_inject_no_findings(self):
        """Test with no findings (returns original text)."""
        original_text = "No abnormalities detected."
        result = inject_segment_chips_for_findings(original_text, [], [])

        assert result == original_text

    def test_inject_mismatched_findings_segments(self):
        """Test graceful handling when findings and segments don't match."""
        original_text = "There is an opacity."
        findings = [{"label": "opacity", "description": "opacity"}]
        new_segments = []  # No segments created

        result = inject_segment_chips_for_findings(original_text, findings, new_segments)

        # Should return original text if no segments available
        assert result == original_text or "SEG" not in result

    def test_inject_empty_original_text(self):
        """Test with empty original text."""
        result = inject_segment_chips_for_findings("", [{"label": "test"}], ["B"])
        assert result == ""

    def test_inject_matching_by_description(self):
        """Test that matching uses finding description."""
        original_text = "There is a large mass in the upper zone."
        findings = [
            {"label": "mass", "description": "large mass in the upper zone"}
        ]
        new_segments = ["B"]

        result = inject_segment_chips_for_findings(original_text, findings, new_segments)

        # Should find the description in the text and inject chip nearby
        assert "B" in result

    def test_inject_case_insensitive_matching(self):
        """Test that matching is case-insensitive."""
        original_text = "There is an Opacity in the right lung."
        findings = [{"label": "opacity", "description": "opacity"}]
        new_segments = ["B"]

        result = inject_segment_chips_for_findings(original_text, findings, new_segments)

        assert "B" in result


class TestChipInjectionStrategies:
    """Test different strategies for chip injection."""

    def test_inject_at_end_of_finding_mention(self):
        """Test chip is injected near the finding mention."""
        original_text = "Segment A shows opacity. The opacity appears dense."
        findings = [{"label": "opacity", "description": "dense opacity"}]
        new_segments = ["B"]

        result = inject_segment_chips_for_findings(original_text, findings, new_segments)

        # Chip should be near "opacity" mentions (but not replace existing Segment A)
        assert "opacity" in result
        assert "B" in result
        assert "Segment A" in result  # Original reference preserved

    def test_inject_appends_to_finding(self):
        """Test that chip is appended to finding description."""
        original_text = "There is a nodule."
        findings = [{"label": "nodule", "description": "nodule"}]
        new_segments = ["B"]

        result = inject_segment_chips_for_findings(original_text, findings, new_segments)

        # Common patterns: "nodule [SEG:B]" or "nodule (Segment B)"
        assert "nodule" in result.lower()
        assert "B" in result

    def test_inject_does_not_duplicate_existing_chips(self):
        """Test that existing chips are not duplicated."""
        original_text = "Segment A shows opacity [SEG:A]."
        findings = [{"label": "opacity", "description": "opacity"}]
        new_segments = ["A"]  # Same segment already referenced

        result = inject_segment_chips_for_findings(original_text, findings, new_segments)

        # Should not add another chip for A
        # Count occurrences of SEG:A or Segment A
        count_a = result.count("SEG:A") + result.count("Segment A")
        assert count_a <= 2  # At most original + maybe one injection


class TestRobustness:
    """Test robustness and edge cases."""

    def test_handles_special_characters_in_findings(self):
        """Test handling findings with special characters."""
        original_text = "The finding (opacity) is visible."
        findings = [{"label": "opacity", "description": "finding (opacity)"}]
        new_segments = ["B"]

        # Should not crash on special regex characters
        result = inject_segment_chips_for_findings(original_text, findings, new_segments)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_handles_unicode_in_text(self):
        """Test handling unicode characters."""
        original_text = "The finding → opacity is visible."
        findings = [{"label": "opacity", "description": "opacity"}]
        new_segments = ["B"]

        result = inject_segment_chips_for_findings(original_text, findings, new_segments)
        assert isinstance(result, str)
        assert "→" in result  # Unicode preserved

    def test_handles_long_text(self):
        """Test with longer text."""
        original_text = """
        The chest radiograph shows a large opacity in the right lower lobe.
        The opacity measures approximately 5cm in diameter.
        There is associated volume loss.
        The finding is concerning for consolidation or mass.
        """.strip()
        findings = [{"label": "opacity", "description": "large opacity"}]
        new_segments = ["B"]

        result = inject_segment_chips_for_findings(original_text, findings, new_segments)

        # Should still contain original content
        assert "right lower lobe" in result
        assert "5cm" in result
        assert "B" in result
