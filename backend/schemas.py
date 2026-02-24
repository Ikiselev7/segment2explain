"""Pydantic models for WebSocket messages and REST API responses."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


# --- REST API ---

class ImageUploadResponse(BaseModel):
    image_id: str
    width: int
    height: int
    url: str
    pixel_spacing_mm: list[float] | None = None


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict[str, bool]


# --- WebSocket: Client → Server ---

class StartJobMessage(BaseModel):
    type: str = "start_job"
    image_id: str
    prompt: str


class CancelJobMessage(BaseModel):
    type: str = "cancel_job"


# --- WebSocket: Server → Client ---

class WsMessage(BaseModel):
    """Base class for all server→client WS messages."""
    type: str


class JobStartedMessage(WsMessage):
    type: str = "job_started"
    job_id: str


class StepData(BaseModel):
    id: str
    name: str
    status: str
    detail: str
    segment_ids: list[str]


class StepAddedMessage(WsMessage):
    type: str = "step_added"
    step: StepData


class StepUpdatedMessage(WsMessage):
    type: str = "step_updated"
    step_id: str
    status: str
    detail: str


class SegmentMeasurements(BaseModel):
    area_px: int = 0
    bbox_px: list[int] = []
    max_diameter_px: float = 0.0
    centroid_px: list[float] = []
    area_mm2: float | None = None
    max_diameter_mm: float | None = None
    pixel_spacing_mm: list[float] | None = None


class SegmentData(BaseModel):
    id: str
    label: str
    description: str = ""
    created_by_step: str = ""
    bbox: list[int] = []
    contour_points: list[list[list[int]]] = []
    color: str = "#3b82f6"
    measurements: SegmentMeasurements = SegmentMeasurements()
    concept: str = ""


class SegmentAddedMessage(WsMessage):
    type: str = "segment_added"
    segment: SegmentData


class SegmentUpdatedMessage(WsMessage):
    type: str = "segment_updated"
    segment_id: str
    label: str
    description: str = ""


class SegmentRemovedMessage(WsMessage):
    type: str = "segment_removed"
    segment_id: str


class ChatMessageData(WsMessage):
    type: str = "chat_message"
    role: str
    content: str
    reasoning: bool = False


class ChatDeltaMessage(WsMessage):
    type: str = "chat_delta"
    text: str
    replace: bool = False


class OverlayReadyMessage(WsMessage):
    type: str = "overlay_ready"
    url: str


class MeasurementsMessage(WsMessage):
    type: str = "measurements"
    data: dict[str, Any]


class DebugMessage(WsMessage):
    type: str = "debug"
    data: dict[str, Any]


class JobCompletedMessage(WsMessage):
    type: str = "job_completed"
    job_id: str


class JobFailedMessage(WsMessage):
    type: str = "job_failed"
    error: str


class ConceptLinkedMessage(WsMessage):
    """Sent after a concept is segmented to link concept terms to a segment."""
    type: str = "concept_linked"
    concept: str
    segment_id: str
    aliases: list[str] = []
    color: str = "#3b82f6"


class HeatmapsReadyMessage(WsMessage):
    """Sent after PRIOR step with available attention heatmap concept names."""
    type: str = "heatmaps_ready"
    concepts: list[str]
    image_id: str
