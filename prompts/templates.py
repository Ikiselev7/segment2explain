from __future__ import annotations


def build_tool_result_message(
    segment_id: str,
    label: str,
    color_name: str = "",
    description: str = "",
) -> str:
    """Format a segment reference line for MedGemma R2 evidence.

    Uses concept label + validated description so R2 gets specific
    context about what each segment represents.
    """
    color_part = f" ({color_name})" if color_name else ""
    desc_part = f" — {description}" if description else ""
    return f"- Segment {segment_id}{color_part}: {label}{desc_part}"
