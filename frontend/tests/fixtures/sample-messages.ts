/** Pre-recorded WebSocket message sequences for testing */

import type { ServerMessage } from "../../src/types/messages";

/** Single-segment pipeline run: "Where is the nodule?" */
export const pipelineMessages: ServerMessage[] = [
  { type: "job_started", job_id: "test-job-1" },

  // S1: Upload step
  {
    type: "step_added",
    step: {
      id: "S1",
      name: "Upload",
      status: "done",
      detail: "Image uploaded",
      segment_ids: [],
    },
  },

  // R1: Extract concepts
  {
    type: "step_added",
    step: {
      id: "R1",
      name: "Extract concepts",
      status: "running",
      detail: "Extracting anatomical concepts...",
      segment_ids: [],
    },
  },
  {
    type: "step_updated",
    step_id: "R1",
    status: "done",
    detail: "Concepts: nodule, mass",
  },

  // R1 reasoning (dimmed in chat)
  {
    type: "chat_message",
    role: "assistant",
    content:
      "**R1 reasoning:** User asks about a nodule or mass. I should segment potential nodular opacities and surrounding lung tissue for context.",
    reasoning: true,
  },

  // SEG: Refined segmentation
  {
    type: "step_added",
    step: {
      id: "SEG",
      name: "Refined segmentation",
      status: "running",
      detail: "Running MedSAM3...",
      segment_ids: [],
    },
  },
  {
    type: "segment_added",
    segment: {
      id: "A",
      label: "Nodule",
      description: "",
      created_by_step: "SEG",
      bbox: [120, 80, 200, 160],
      contour_points: [
        [
          [120, 80],
          [200, 80],
          [200, 160],
          [120, 160],
        ],
      ],
      color: "#e6194b",
      measurements: {
        area_px: 6400,
        bbox_px: [120, 80, 200, 160],
        max_diameter_px: 80,
        centroid_px: [160, 120],
      },
    },
  },
  {
    type: "step_updated",
    step_id: "SEG",
    status: "done",
    detail: "Found 1 segment",
  },

  // FA: Identify segment A
  {
    type: "step_added",
    step: {
      id: "FA",
      name: "Identify segment A",
      status: "running",
      detail: "Identifying region...",
      segment_ids: ["A"],
    },
  },
  {
    type: "step_updated",
    step_id: "FA",
    status: "done",
    detail: "Identified: Nodule",
  },

  // M1: Match relevance
  {
    type: "step_added",
    step: {
      id: "M1",
      name: "Match relevance",
      status: "running",
      detail: "Judging concept matches...",
      segment_ids: ["A"],
    },
  },
  {
    type: "step_updated",
    step_id: "M1",
    status: "done",
    detail: "All segments relevant.",
  },

  // R2: Analysis (streamed)
  {
    type: "step_added",
    step: {
      id: "R2",
      name: "Analysis",
      status: "running",
      detail: "Generating analysis...",
      segment_ids: ["A"],
    },
  },
  {
    type: "chat_delta",
    text: "The image shows a ",
  },
  {
    type: "chat_delta",
    text: "round nodular opacity ",
  },
  {
    type: "chat_delta",
    text: "in the right upper lobe. ",
  },
  {
    type: "chat_delta",
    text: "[[SEG:A]] measures approximately 2cm. ",
  },
  {
    type: "chat_delta",
    text: "This finding warrants further evaluation with CT.",
  },
  {
    type: "step_updated",
    step_id: "R2",
    status: "done",
    detail: "Analysis complete",
  },

  { type: "job_completed", job_id: "test-job-1" },
];

/** Multi-segment pipeline run: "Describe this chest X-ray" */
export const multiSegmentPipelineMessages: ServerMessage[] = [
  { type: "job_started", job_id: "test-job-2" },

  {
    type: "step_added",
    step: {
      id: "S1",
      name: "Upload",
      status: "done",
      detail: "Image uploaded",
      segment_ids: [],
    },
  },

  // R1: Extract concepts
  {
    type: "step_added",
    step: {
      id: "R1",
      name: "Extract concepts",
      status: "running",
      detail: "Extracting anatomical concepts...",
      segment_ids: [],
    },
  },
  {
    type: "step_updated",
    step_id: "R1",
    status: "done",
    detail: "Concepts: heart, left lung",
  },

  // SEG: Segmentation
  {
    type: "step_added",
    step: {
      id: "SEG",
      name: "Refined segmentation",
      status: "running",
      detail: "Running MedSAM3...",
      segment_ids: [],
    },
  },
  {
    type: "segment_added",
    segment: {
      id: "A",
      label: "Heart",
      description: "",
      created_by_step: "SEG",
      bbox: [100, 150, 250, 350],
      contour_points: [
        [
          [100, 150],
          [250, 150],
          [250, 350],
          [100, 350],
        ],
      ],
      color: "#e6194b",
      measurements: {
        area_px: 30000,
        bbox_px: [100, 150, 250, 350],
        max_diameter_px: 200,
        centroid_px: [175, 250],
      },
    },
  },
  {
    type: "segment_added",
    segment: {
      id: "B",
      label: "Left lung",
      description: "",
      created_by_step: "SEG",
      bbox: [260, 50, 450, 400],
      contour_points: [
        [
          [260, 50],
          [450, 50],
          [450, 400],
          [260, 400],
        ],
      ],
      color: "#3cb44b",
      measurements: {
        area_px: 66500,
        bbox_px: [260, 50, 450, 400],
        max_diameter_px: 350,
        centroid_px: [355, 225],
      },
    },
  },
  {
    type: "step_updated",
    step_id: "SEG",
    status: "done",
    detail: "Found 2 segments",
  },

  // Identify steps
  {
    type: "step_added",
    step: {
      id: "FA",
      name: "Identify segment A",
      status: "done",
      detail: "Identified: Heart",
      segment_ids: ["A"],
    },
  },
  {
    type: "step_added",
    step: {
      id: "FB",
      name: "Identify segment B",
      status: "done",
      detail: "Identified: Left lung",
      segment_ids: ["B"],
    },
  },

  // M1: Match relevance
  {
    type: "step_added",
    step: {
      id: "M1",
      name: "Match relevance",
      status: "done",
      detail: "All segments relevant.",
      segment_ids: ["A", "B"],
    },
  },

  // R2: Analysis
  {
    type: "step_added",
    step: {
      id: "R2",
      name: "Analysis",
      status: "running",
      detail: "Generating analysis...",
      segment_ids: ["A", "B"],
    },
  },
  {
    type: "chat_delta",
    text: "This chest X-ray shows [[SEG:A]] (heart) and [[SEG:B]] (left lung). ",
  },
  {
    type: "chat_delta",
    text: "The cardiac silhouette appears mildly enlarged.",
  },
  {
    type: "step_updated",
    step_id: "R2",
    status: "done",
    detail: "Analysis complete",
  },

  { type: "job_completed", job_id: "test-job-2" },
];

/** Parallel mode pipeline run: answer first, then concepts and segments */
export const parallelPipelineMessages: ServerMessage[] = [
  { type: "job_started", job_id: "test-job-parallel" },

  // S1: Parse request
  {
    type: "step_added",
    step: {
      id: "S1",
      name: "Parse request",
      status: "done",
      detail: "Image received.",
      segment_ids: [],
    },
  },

  // ANSWER: Stream the answer (replace-delta for first chunk)
  {
    type: "step_added",
    step: {
      id: "ANSWER",
      name: "MedGemma: answer",
      status: "running",
      detail: "Streaming\u2026",
      segment_ids: [],
    },
  },
  {
    type: "chat_delta",
    text: "The heart appears mildly enlarged, ",
    replace: true,
  },
  {
    type: "chat_delta",
    text: "suggesting cardiomegaly. ",
  },
  {
    type: "chat_delta",
    text: "The lungs show no focal consolidation or pleural effusion.",
  },
  {
    type: "step_updated",
    step_id: "ANSWER",
    status: "done",
    detail: "Answer complete.",
  },

  // SELECT: Concept selection (xgrammar enum, uses KV cache)
  {
    type: "step_added",
    step: {
      id: "SELECT",
      name: "Select concepts",
      status: "done",
      detail: "Selected: heart, lungs",
      segment_ids: [],
    },
  },

  // SEG: Progressive segmentation (runs after SELECT)
  {
    type: "step_added",
    step: {
      id: "SEG",
      name: "MedSAM3: segmentation",
      status: "running",
      detail: "Segmenting concepts\u2026",
      segment_ids: [],
    },
  },
  {
    type: "segment_added",
    segment: {
      id: "A",
      label: "heart (A)",
      description: "",
      created_by_step: "SEG",
      bbox: [100, 150, 250, 350],
      contour_points: [
        [
          [100, 150],
          [250, 150],
          [250, 350],
          [100, 350],
        ],
      ],
      color: "#e6194b",
      measurements: {
        area_px: 30000,
        bbox_px: [100, 150, 250, 350],
        max_diameter_px: 200,
        centroid_px: [175, 250],
      },
    },
  },
  {
    type: "segment_added",
    segment: {
      id: "B",
      label: "lungs (B)",
      description: "",
      created_by_step: "SEG",
      bbox: [260, 50, 450, 400],
      contour_points: [
        [
          [260, 50],
          [450, 50],
          [450, 400],
          [260, 400],
        ],
      ],
      color: "#3cb44b",
      measurements: {
        area_px: 66500,
        bbox_px: [260, 50, 450, 400],
        max_diameter_px: 350,
        centroid_px: [355, 225],
      },
    },
  },
  {
    type: "step_updated",
    step_id: "SEG",
    status: "done",
    detail: "Found 2 region(s).",
  },

  // LINK: Deterministic alias extraction (after SEG)
  {
    type: "step_added",
    step: {
      id: "LINK",
      name: "Extract highlights",
      status: "running",
      detail: "Scanning answer phrases\u2026",
      segment_ids: [],
    },
  },
  {
    type: "step_updated",
    step_id: "LINK",
    status: "done",
    detail: "Found 3 highlight phrase(s)",
  },

  // Concept linking
  {
    type: "concept_linked",
    concept: "heart",
    segment_id: "A",
    aliases: ["cardiac silhouette", "cardiomegaly"],
    color: "#e6194b",
  },
  {
    type: "concept_linked",
    concept: "lungs",
    segment_id: "B",
    aliases: ["lung", "consolidation"],
    color: "#3cb44b",
  },

  { type: "job_completed", job_id: "test-job-parallel" },
];

/** Error scenario */
export const errorPipelineMessages: ServerMessage[] = [
  { type: "job_started", job_id: "test-job-3" },
  { type: "job_failed", error: "MedGemma inference failed: CUDA out of memory" },
];

/** Segment removal scenario */
export const segmentRemovalMessages: ServerMessage[] = [
  { type: "job_started", job_id: "test-job-4" },
  {
    type: "step_added",
    step: {
      id: "S1",
      name: "Upload",
      status: "done",
      detail: "Image uploaded",
      segment_ids: [],
    },
  },
  {
    type: "segment_added",
    segment: {
      id: "A",
      label: "Candidate",
      description: "",
      created_by_step: "SEG",
      bbox: [10, 10, 50, 50],
      contour_points: [],
      color: "#e6194b",
      measurements: {
        area_px: 1600,
        bbox_px: [10, 10, 50, 50],
        max_diameter_px: 40,
        centroid_px: [30, 30],
      },
    },
  },
  {
    type: "segment_removed",
    segment_id: "A",
  },
  { type: "job_completed", job_id: "test-job-4" },
];
