"""Segments list renderer - convert segments to interactive HTML table.

This module provides HTML rendering for the segments list with:
- Table view of all segments with metadata
- Data attributes for click handling (segment-to-step navigation)
- Measurements display (area, etc.)
- Proper HTML escaping for safety

Example:
    >>> from orchestrator import create_job_state
    >>> state = create_job_state()
    >>> # ... add segments to state ...
    >>> html = render_segments_list_html(state)
    >>> # Returns interactive HTML table
"""

import html

from orchestrator import JobState


def render_segments_list_html(state: JobState) -> str:
    """Render segments as interactive HTML table.

    Creates a table with columns:
    - ID: Segment identifier (A, B, C, ...)
    - Label: Segment description
    - Created By: Step that created this segment
    - Area: Measurement in pixels

    Each row includes data attributes:
    - data-segment-id: For identifying clicked segments
    - data-step-id: For navigating to creating step

    Args:
        state: JobState containing segments

    Returns:
        HTML string with interactive segments table

    Example:
        >>> state = create_job_state()
        >>> # Add segment...
        >>> html = render_segments_list_html(state)
        >>> # Contains: <tr class="segment-row" data-segment-id="A" ...>
    """
    if not state.segments:
        return '<div class="segments-list"><p class="no-segments">No segments yet.</p></div>'

    lines = ['<div class="segments-list">']
    lines.append('<table>')

    # Table header
    lines.append('''
    <thead>
        <tr>
            <th>ID</th>
            <th>Label</th>
            <th>Created By</th>
            <th>Area</th>
        </tr>
    </thead>
    ''')

    # Table body
    lines.append('<tbody>')

    for seg_id, seg in state.segments.items():
        # Escape text content
        seg_id_escaped = html.escape(seg_id)
        label_escaped = html.escape(seg["label"])
        step_id_escaped = html.escape(seg["created_by_step"])

        # Format measurements
        area_px = seg["measurements"]["area_px"]
        area_formatted = f"{area_px:,}px"  # Format with commas

        lines.append(f'''
        <tr class="segment-row"
            data-segment-id="{seg_id_escaped}"
            data-step-id="{step_id_escaped}">
            <td><strong>{seg_id_escaped}</strong></td>
            <td>{label_escaped}</td>
            <td>{step_id_escaped}</td>
            <td>{area_formatted}</td>
        </tr>
        ''')

    lines.append('</tbody>')
    lines.append('</table>')
    lines.append('</div>')

    return '\n'.join(lines)
