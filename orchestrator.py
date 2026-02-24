from __future__ import annotations

import string
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from PIL import Image

Status = Literal["queued", "running", "done", "failed"]


@dataclass
class Step:
    id: str
    name: str
    status: Status = "queued"
    detail: str = ""
    segment_ids: List[str] = field(default_factory=list)


@dataclass
class JobState:
    job_id: Optional[str] = None
    image: Optional[np.ndarray] = None  # HxWx3 uint8
    chat: List[dict] = field(default_factory=list)  # openai-style {role, content}
    steps: List[Step] = field(default_factory=list)
    segments: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    highlight: str = "ALL"
    proposed_bboxes: List[Tuple[Tuple[int, int, int, int], str]] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)
    pixel_spacing: Optional[Tuple[float, float]] = None  # (row_mm, col_mm) from DICOM

    _seg_counter: int = 0

    def next_segment_id(self) -> str:
        letters = string.ascii_uppercase
        seg_id = letters[self._seg_counter] if self._seg_counter < len(letters) else f"S{self._seg_counter}"
        self._seg_counter += 1
        return seg_id

    def add_segment(
        self,
        segment_id: str,
        label: str,
        mask: np.ndarray,
        bbox: Tuple[int, int, int, int],
        created_by_step: str,
        measurements: Dict[str, Any],
    ) -> None:
        # color_idx is assigned at creation and never changes — survives filtering
        color_idx = self._seg_counter - 1  # next_segment_id() already incremented
        self.segments[segment_id] = {
            "segment_id": segment_id,
            "label": label,
            "mask": mask.astype(np.float32),
            "bbox": bbox,
            "created_by_step": created_by_step,
            "measurements": measurements,
            "color_idx": color_idx,
        }


def create_job_state() -> JobState:
    return JobState()


def ensure_rgb_uint8(img: Any) -> np.ndarray:
    """Coerce input image to RGB uint8 numpy array."""
    if isinstance(img, Image.Image):
        img = np.array(img.convert("RGB"))
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def render_steps_markdown(steps: List[Step]) -> str:
    lines = []
    for s in steps:
        icon = {
            "queued": "⏳",
            "running": "🔄",
            "done": "✅",
            "failed": "❌",
        }.get(s.status, "•")
        segs = ", ".join([f"Segment {sid}" for sid in s.segment_ids]) if s.segment_ids else ""
        detail = s.detail or ""
        if segs:
            detail = (detail + " | " + segs).strip(" |")
        lines.append(f"- **{s.id}** {icon} — {s.name}  \n  {detail}")
    return "\n".join(lines) if lines else "_No steps yet._"


def build_annotated_image(state: JobState, highlight: Optional[str] = None):
    """Return (base_image, [(mask_or_bbox, label), ...]) for overlay rendering."""
    if state.image is None:
        return None

    img = state.image
    hl = highlight or state.highlight or "ALL"

    annotations = []

    # Determine which segments to show
    seg_ids: List[str] = list(state.segments.keys())

    if hl.startswith("SEG:"):
        target = hl.split(":", 1)[1]
        seg_ids = [target] if target in state.segments else []
    elif hl != "ALL":
        # interpret as step id
        step = next((s for s in state.steps if s.id == hl), None)
        if step and step.segment_ids:
            seg_ids = step.segment_ids
        else:
            seg_ids = list(state.segments.keys())

    for sid in seg_ids:
        seg = state.segments.get(sid)
        if not seg:
            continue
        mask = seg["mask"]
        label = f"{sid}: {seg['label']}"
        annotations.append((mask, label))

    # Draw proposed bboxes (from MedGemma R1) if no segments yet or showing all
    if state.proposed_bboxes and (hl == "ALL" or not state.segments):
        for bbox, bbox_label in state.proposed_bboxes:
            annotations.append((bbox, f"bbox: {bbox_label}"))

    return (img, annotations)


def make_side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
    left = left.convert("RGB")
    right = right.convert("RGB")
    h = max(left.height, right.height)
    w = left.width + right.width
    out = Image.new("RGB", (w, h), (0, 0, 0))
    out.paste(left, (0, 0))
    out.paste(right, (left.width, 0))
    return out
