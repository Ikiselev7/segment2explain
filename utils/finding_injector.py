"""Finding injector - post-process text to inject segment chips after grounding.

This module provides functionality to:
1. Take original preliminary text
2. Match findings to text mentions
3. Inject segment chips at appropriate locations
4. Preserve original text structure

This is used in the two-stage answer strategy:
- Stage A: Quick answer with preliminary text
- Stage B: Ground findings → create segments
- Final: Inject chips into Stage A text

Example:
    >>> original = "There is an opacity in the right lung."
    >>> findings = [{"label": "opacity", "description": "opacity in right lung"}]
    >>> segments = ["B"]
    >>> result = inject_segment_chips_for_findings(original, findings, segments)
    >>> # Result: "There is an opacity [SEG:B] in the right lung."
"""

import re


def inject_segment_chips_for_findings(
    original_text: str,
    findings: list[dict],
    new_segments: list[str],
) -> str:
    """Inject segment chips into text for grounded findings.

    Takes the original text and injects [SEG:X] chips near mentions of
    findings that were successfully grounded.

    Strategy:
    1. Match each finding to its segment (by index)
    2. Find finding description/label in original text
    3. Inject chip after the mention (append mode)
    4. Avoid duplicating existing chips

    Args:
        original_text: The preliminary answer text
        findings: List of finding dicts with 'label' and optionally 'description'
        new_segments: List of segment IDs created during grounding (B, C, D, ...)

    Returns:
        Text with chips injected at appropriate locations

    Example:
        >>> text = "There is an opacity and a nodule."
        >>> findings = [
        ...     {"label": "opacity", "description": "opacity"},
        ...     {"label": "nodule", "description": "nodule"}
        ... ]
        >>> segments = ["B", "C"]
        >>> inject_segment_chips_for_findings(text, findings, segments)
        'There is an opacity [SEG:B] and a nodule [SEG:C].'
    """
    if not original_text:
        return ""

    if not findings or not new_segments:
        return original_text

    # Work with a mutable version
    result = original_text

    # Process findings in reverse order to preserve positions
    # (injecting at later positions first doesn't affect earlier positions)
    for idx in range(min(len(findings), len(new_segments)) - 1, -1, -1):
        finding = findings[idx]
        seg_id = new_segments[idx]

        # Get searchable text (prefer description, fall back to label)
        search_term = finding.get("description", finding.get("label", ""))

        if not search_term:
            continue

        # Check if this segment is already referenced
        if f"[SEG:{seg_id}]" in result or f"Segment {seg_id}" in result:
            continue  # Skip if already present

        # Escape special regex characters in search term
        search_term_escaped = re.escape(search_term)

        # Find the search term (case-insensitive, whole words preferred)
        # Try exact phrase first
        pattern = re.compile(rf"\b{search_term_escaped}\b", re.IGNORECASE)
        match = pattern.search(result)

        if not match:
            # Try just the label if description didn't match
            label = finding.get("label", "")
            if label and label != search_term:
                label_escaped = re.escape(label)
                pattern = re.compile(rf"\b{label_escaped}\b", re.IGNORECASE)
                match = pattern.search(result)

        if match:
            # Inject chip after the match
            insert_pos = match.end()
            chip = f" [SEG:{seg_id}]"

            # Insert at position
            result = result[:insert_pos] + chip + result[insert_pos:]

    return result
