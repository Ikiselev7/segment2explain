/** Frontend pipeline state types */

export type StepStatus = "queued" | "running" | "done" | "failed";

export interface Step {
  id: string;
  name: string;
  status: StepStatus;
  detail: string;
  segmentIds: string[];
}

export interface Segment {
  id: string;
  label: string;
  description: string;
  createdByStep: string;
  bbox: number[];
  contourPoints: number[][][];
  color: string;
  concept: string;
  measurements: {
    area_px: number;
    bbox_px: number[];
    max_diameter_px: number;
    centroid_px: number[];
    area_mm2?: number | null;
    max_diameter_mm?: number | null;
    pixel_spacing_mm?: number[] | null;
  };
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  reasoning?: boolean;
}

export interface ConceptLink {
  concept: string;
  segmentId: string;
  aliases: string[];
  color: string;
}
