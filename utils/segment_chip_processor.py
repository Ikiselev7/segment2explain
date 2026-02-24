"""Segment chip processor - detect and convert segment references to interactive HTML chips.

This module provides functionality to:
1. Detect segment references in text (e.g., [[SEG:A]], [SEG:A], "Segment A")
2. Convert references to clickable HTML chips with data attributes
3. Only process segments that actually exist in the state

Example:
    >>> text = "The finding in Segment A shows opacity."
    >>> available_segments = ["A", "B"]
    >>> result = process_segment_chips(text, available_segments)
    >>> print(result)
    'The finding in <span class="seg-chip" data-seg-id="A">[SEG:A]</span> shows opacity.'
"""

import re


def detect_segment_references(text: str) -> set[str]:
    """Detect all segment references in text.

    Supports multiple formats:
    - [[SEG:A]] - double bracket format
    - [SEG:A] - single bracket format
    - "Segment A" - word format

    Args:
        text: Input text to search for segment references

    Returns:
        Set of segment IDs (uppercase letters) found in text

    Example:
        >>> detect_segment_references("Segment A and [[SEG:B]]")
        {'A', 'B'}
    """
    if not text:
        return set()

    matches = set()

    # Pattern 1: [[SEG:X]] format
    pattern1 = re.compile(r"\[\[SEG:([A-Z])\]\]", re.IGNORECASE)
    matches.update(m.upper() for m in pattern1.findall(text))

    # Pattern 2: [SEG:X] format
    pattern2 = re.compile(r"\[SEG:([A-Z])\]", re.IGNORECASE)
    matches.update(m.upper() for m in pattern2.findall(text))

    # Pattern 3: "Segment X" or "segment X" format
    pattern3 = re.compile(r"\bSegment\s+([A-Z])\b", re.IGNORECASE)
    matches.update(m.upper() for m in pattern3.findall(text))

    return matches


def process_segment_chips(text: str, available_segments: list[str]) -> str:
    """Convert segment references to clickable HTML chips.

    Only processes segments that exist in available_segments.
    Segments not in the list are left as plain text.

    Args:
        text: Input text with segment references
        available_segments: List of segment IDs that actually exist

    Returns:
        Text with segment references converted to HTML chips

    Example:
        >>> text = "Findings: Segment A shows opacity, [[SEG:B]] shows consolidation."
        >>> available = ["A", "B"]
        >>> result = process_segment_chips(text, available)
        >>> # Result contains clickable chips for A and B
    """
    if not text:
        return ""

    if not available_segments:
        return text

    # Convert to set for fast lookup (uppercase)
    seg_set = {seg.upper() for seg in available_segments}

    # Process each pattern in order (most specific first)

    # Pattern 1: [[SEG:X]] → chip
    def replace_double_bracket(match):
        seg_id = match.group(1).upper()
        if seg_id in seg_set:
            return f'<span class="seg-chip" data-seg-id="{seg_id}">[SEG:{seg_id}]</span>'
        return match.group(0)  # Leave unchanged

    text = re.sub(r"\[\[SEG:([A-Z])\]\]", replace_double_bracket, text, flags=re.IGNORECASE)

    # Pattern 2: [SEG:X] → chip (only if not already inside a span)
    def replace_single_bracket(match):
        seg_id = match.group(1).upper()
        if seg_id in seg_set:
            return f'<span class="seg-chip" data-seg-id="{seg_id}">[SEG:{seg_id}]</span>'
        return match.group(0)

    # Use negative lookahead to avoid matching inside span tags
    text = re.sub(
        r"\[SEG:([A-Z])\](?![^<]*</span>)",
        replace_single_bracket,
        text,
        flags=re.IGNORECASE
    )

    # Pattern 3: "Segment X" → chip (only if not already inside a span)
    def replace_segment_word(match):
        seg_id = match.group(1).upper()
        if seg_id in seg_set:
            return f'<span class="seg-chip" data-seg-id="{seg_id}">[SEG:{seg_id}]</span>'
        return match.group(0)

    # Don't replace if already inside a span tag (avoid double-processing)
    # Use negative lookbehind to avoid matching inside HTML tags
    text = re.sub(
        r'\bSegment\s+([A-Z])\b(?![^<]*</span>)',
        replace_segment_word,
        text,
        flags=re.IGNORECASE
    )

    return text


def _escape_html(text: str) -> str:
    """Escape HTML special characters.

    Args:
        text: Text to escape

    Returns:
        Escaped text safe for HTML
    """
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )
