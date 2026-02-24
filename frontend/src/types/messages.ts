/** WebSocket message types — mirrors backend/schemas.py */

// Client → Server
export interface StartJobMessage {
  type: "start_job";
  image_id: string;
  prompt: string;
  mode?: "sequential" | "parallel";
}

export interface CancelJobMessage {
  type: "cancel_job";
}

export type ClientMessage = StartJobMessage | CancelJobMessage;

// Server → Client
export interface JobStartedMessage {
  type: "job_started";
  job_id: string;
}

export interface StepData {
  id: string;
  name: string;
  status: string;
  detail: string;
  segment_ids: string[];
}

export interface StepAddedMessage {
  type: "step_added";
  step: StepData;
}

export interface StepUpdatedMessage {
  type: "step_updated";
  step_id: string;
  status: string;
  detail: string;
}

export interface SegmentMeasurements {
  area_px: number;
  bbox_px: number[];
  max_diameter_px: number;
  centroid_px: number[];
  area_mm2?: number | null;
  max_diameter_mm?: number | null;
  pixel_spacing_mm?: number[] | null;
}

export interface SegmentData {
  id: string;
  label: string;
  description: string;
  created_by_step: string;
  bbox: number[];
  contour_points: number[][][];
  color: string;
  concept?: string;
  measurements: SegmentMeasurements;
}

export interface SegmentAddedMessage {
  type: "segment_added";
  segment: SegmentData;
}

export interface SegmentUpdatedMessage {
  type: "segment_updated";
  segment_id: string;
  label: string;
  description: string;
}

export interface SegmentRemovedMessage {
  type: "segment_removed";
  segment_id: string;
}

export interface ChatMessageMessage {
  type: "chat_message";
  role: "user" | "assistant";
  content: string;
  reasoning?: boolean;
}

export interface ChatDeltaMessage {
  type: "chat_delta";
  text: string;
  replace?: boolean;
}

export interface OverlayReadyMessage {
  type: "overlay_ready";
  url: string;
}

export interface MeasurementsMessage {
  type: "measurements";
  data: Record<string, SegmentMeasurements>;
}

export interface DebugMessage {
  type: "debug";
  data: Record<string, unknown>;
}

export interface JobCompletedMessage {
  type: "job_completed";
  job_id: string;
}

export interface JobFailedMessage {
  type: "job_failed";
  error: string;
}

export interface ConceptLinkedMessage {
  type: "concept_linked";
  concept: string;
  segment_id: string;
  aliases: string[];
  color: string;
}

export interface HeatmapsReadyMessage {
  type: "heatmaps_ready";
  concepts: string[];
  image_id: string;
}

export interface ErrorMessage {
  type: "error";
  error: string;
}

export type ServerMessage =
  | JobStartedMessage
  | StepAddedMessage
  | StepUpdatedMessage
  | SegmentAddedMessage
  | SegmentUpdatedMessage
  | SegmentRemovedMessage
  | ChatMessageMessage
  | ChatDeltaMessage
  | OverlayReadyMessage
  | MeasurementsMessage
  | DebugMessage
  | ConceptLinkedMessage
  | HeatmapsReadyMessage
  | JobCompletedMessage
  | JobFailedMessage
  | ErrorMessage;
