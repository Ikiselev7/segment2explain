"""Steps renderer - convert workflow steps to interactive HTML.

This module provides HTML rendering for workflow steps with:
- Interactive step cards with data attributes for click/hover handling
- Status indicators (icons and CSS classes)
- Segment ID associations for linking
- Proper HTML escaping for safety

Example:
    >>> from orchestrator import Step
    >>> steps = [Step(id="S1", name="Parse", status="done", segment_ids=["A"])]
    >>> html = render_steps_html(steps)
    >>> # Returns interactive HTML with data attributes
"""

import html
import json

from orchestrator import Step


def render_steps_html(steps: list[Step]) -> str:
    """Render workflow steps as interactive HTML.

    Creates a container with step cards that include:
    - data-step-id: For identifying clicked steps
    - data-segment-ids: JSON array of associated segment IDs
    - CSS classes for status: step-queued, step-running, step-done, step-failed
    - Status icons: ⏳ queued, 🔄 running, ✅ done, ❌ failed

    Args:
        steps: List of Step objects to render

    Returns:
        HTML string with interactive step cards

    Example:
        >>> steps = [Step(id="S2", name="Segment ROI", status="done", segment_ids=["A"])]
        >>> html = render_steps_html(steps)
        >>> # Contains: <div class="step-item step-done" data-step-id="S2" ...>
    """
    if not steps:
        return '<div class="steps-container"><p class="no-steps">No steps yet.</p></div>'

    lines = ['<div class="steps-container">']

    for step in steps:
        # Status icon
        status_icon = {
            "queued": "⏳",
            "running": "🔄",
            "done": "✅",
            "failed": "❌",
        }.get(step.status, "•")

        # CSS class for status
        status_class = f"step-{step.status}"

        # Segment IDs as JSON array (HTML-escaped)
        segment_ids_json = html.escape(json.dumps(step.segment_ids))

        # Escape text content
        step_id_escaped = html.escape(step.id)
        name_escaped = html.escape(step.name)
        detail_escaped = html.escape(step.detail) if step.detail else ""

        # Build step HTML
        lines.append(f'''
        <div class="step-item {status_class}"
             data-step-id="{step_id_escaped}"
             data-segment-ids='{segment_ids_json}'>
            <div class="step-header">
                <span class="step-icon">{status_icon}</span>
                <strong>{step_id_escaped}</strong> — {name_escaped}
            </div>
            <div class="step-detail">{detail_escaped}</div>
        </div>
        ''')

    lines.append('</div>')
    return '\n'.join(lines)


def _escape_html(text: str) -> str:
    """Escape HTML special characters.

    Args:
        text: Text to escape

    Returns:
        Escaped text safe for HTML

    Note:
        This is a fallback; we use html.escape() from stdlib primarily.
    """
    return html.escape(text)
