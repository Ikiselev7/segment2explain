"""Utility modules for segment2explain-poc.

This package contains helper modules for:
- Segment chip processing (converting text references to HTML)
- Steps rendering (HTML rendering for interactive steps)
- Segments list rendering (HTML table for segments metadata)
- Finding injection (post-processing text to inject chips after grounding)
"""

from utils.finding_injector import inject_segment_chips_for_findings
from utils.segment_chip_processor import detect_segment_references, process_segment_chips
from utils.segments_list_renderer import render_segments_list_html
from utils.steps_renderer import render_steps_html


__all__ = [
    "detect_segment_references",
    "process_segment_chips",
    "render_steps_html",
    "render_segments_list_html",
    "inject_segment_chips_for_findings",
]

__version__ = "0.1.0"
