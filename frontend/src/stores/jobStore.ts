/** Zustand store for pipeline job state */

import { create } from "zustand";
import type { ChatMessage, ConceptLink, Segment, Step, StepStatus } from "../types/pipeline";

interface JobState {
  // Image
  imageId: string | null;
  imageUrl: string | null;
  imageWidth: number;
  imageHeight: number;

  // Job lifecycle
  jobId: string | null;
  isRunning: boolean;
  error: string | null;

  // Pipeline state
  steps: Step[];
  segments: Map<string, Segment>;
  chatMessages: ChatMessage[];
  streamingText: string;
  overlayUrl: string | null;

  // Parallel mode
  conceptLinks: ConceptLink[];
  pipelineMode: "sequential" | "parallel";

  // Hover cross-referencing
  hoveredSegmentId: string | null;
  hoverSource: "image" | "chat" | null;

  // Visualization toggles
  showSegments: boolean;
  heatmapConcepts: string[];
  heatmapImageId: string | null;
  selectedHeatmapConcept: string | null;
  showHeatmap: boolean;

  // Actions
  setImage: (id: string, url: string, width: number, height: number) => void;
  beginJob: () => void;
  startJob: (jobId: string) => void;
  addStep: (step: Step) => void;
  updateStep: (stepId: string, status: StepStatus, detail: string) => void;
  addSegment: (segment: Segment) => void;
  updateSegment: (segmentId: string, label: string, description: string) => void;
  removeSegment: (segmentId: string) => void;
  addChatMessage: (msg: ChatMessage) => void;
  appendStreamingText: (text: string) => void;
  replaceStreamingText: (text: string) => void;
  finalizeStreaming: () => void;
  setOverlayUrl: (url: string) => void;
  addConceptLink: (link: ConceptLink) => void;
  setPipelineMode: (mode: "sequential" | "parallel") => void;
  setHoveredSegment: (id: string | null, source: "image" | "chat" | null) => void;
  toggleSegments: () => void;
  setHeatmapConcepts: (concepts: string[], imageId: string) => void;
  setSelectedHeatmapConcept: (concept: string | null) => void;
  toggleHeatmap: () => void;
  completeJob: () => void;
  failJob: (error: string) => void;
  clearJob: () => void;
}

export const useJobStore = create<JobState>((set) => ({
  // Initial state
  imageId: null,
  imageUrl: null,
  imageWidth: 0,
  imageHeight: 0,
  jobId: null,
  isRunning: false,
  error: null,
  steps: [],
  segments: new Map(),
  chatMessages: [],
  streamingText: "",
  overlayUrl: null,
  conceptLinks: [],
  pipelineMode: "sequential",
  hoveredSegmentId: null,
  hoverSource: null,
  showSegments: true,
  heatmapConcepts: [],
  heatmapImageId: null,
  selectedHeatmapConcept: null,
  showHeatmap: false,

  // Actions
  setImage: (id, url, width, height) =>
    set({ imageId: id, imageUrl: url, imageWidth: width, imageHeight: height }),

  beginJob: () =>
    set({
      isRunning: true,
      error: null,
      steps: [],
      segments: new Map(),
      chatMessages: [],
      streamingText: "",
      overlayUrl: null,
      conceptLinks: [],
      heatmapConcepts: [],
      heatmapImageId: null,
      selectedHeatmapConcept: null,
      showHeatmap: false,
    }),

  startJob: (jobId) =>
    set({ jobId, isRunning: true }),

  addStep: (step) =>
    set((state) => ({ steps: [...state.steps, step] })),

  updateStep: (stepId, status, detail) =>
    set((state) => ({
      steps: state.steps.map((s) =>
        s.id === stepId ? { ...s, status, detail } : s
      ),
    })),

  addSegment: (segment) =>
    set((state) => {
      const newMap = new Map(state.segments);
      newMap.set(segment.id, segment);
      return { segments: newMap };
    }),

  updateSegment: (segmentId, label, description) =>
    set((state) => {
      const newMap = new Map(state.segments);
      const existing = newMap.get(segmentId);
      if (existing) {
        newMap.set(segmentId, { ...existing, label, description });
      }
      return { segments: newMap };
    }),

  removeSegment: (segmentId) =>
    set((state) => {
      const newMap = new Map(state.segments);
      newMap.delete(segmentId);
      return { segments: newMap };
    }),

  addChatMessage: (msg) =>
    set((state) => {
      // Streaming reasoning: replace the last reasoning message unless
      // they belong to different steps (e.g., **FA:** vs **FB:**).
      if (msg.reasoning && state.chatMessages.length > 0) {
        const last = state.chatMessages[state.chatMessages.length - 1];
        if (last.reasoning) {
          const prefixOf = (s: string) => s.match(/^\*\*\w+[.:]\*\*/)?.[0] ?? "";
          const newPfx = prefixOf(msg.content);
          const oldPfx = prefixOf(last.content);
          // Append only when both have a bold prefix AND they differ
          const differentSteps = newPfx !== "" && oldPfx !== "" && newPfx !== oldPfx;
          if (!differentSteps) {
            return {
              chatMessages: [...state.chatMessages.slice(0, -1), msg],
            };
          }
        }
      }
      return { chatMessages: [...state.chatMessages, msg] };
    }),

  appendStreamingText: (text) =>
    set((state) => ({ streamingText: state.streamingText + text })),

  replaceStreamingText: (text) => set({ streamingText: text }),

  finalizeStreaming: () =>
    set((state) => {
      if (!state.streamingText) return state;
      return {
        chatMessages: [
          ...state.chatMessages,
          { role: "assistant" as const, content: state.streamingText },
        ],
        streamingText: "",
      };
    }),

  setOverlayUrl: (url) => set({ overlayUrl: url }),

  addConceptLink: (link) =>
    set((state) => ({ conceptLinks: [...state.conceptLinks, link] })),

  setPipelineMode: (mode) => set({ pipelineMode: mode }),

  setHoveredSegment: (id, source) =>
    set({ hoveredSegmentId: id, hoverSource: source }),

  setHeatmapConcepts: (concepts, imageId) =>
    set({
      heatmapConcepts: concepts,
      heatmapImageId: imageId,
      selectedHeatmapConcept: concepts.length > 0 ? concepts[0] : null,
    }),

  setSelectedHeatmapConcept: (concept) =>
    set({ selectedHeatmapConcept: concept }),

  toggleSegments: () =>
    set((state) => ({ showSegments: !state.showSegments })),

  toggleHeatmap: () =>
    set((state) => ({ showHeatmap: !state.showHeatmap })),

  completeJob: () =>
    set((state) => {
      // Finalize any remaining streaming text
      const msgs = state.streamingText
        ? [
            ...state.chatMessages,
            { role: "assistant" as const, content: state.streamingText },
          ]
        : state.chatMessages;
      return { isRunning: false, chatMessages: msgs, streamingText: "" };
    }),

  failJob: (error) => set({ isRunning: false, error }),

  clearJob: () =>
    set({
      jobId: null,
      isRunning: false,
      error: null,
      steps: [],
      segments: new Map(),
      chatMessages: [],
      streamingText: "",
      overlayUrl: null,
      conceptLinks: [],
      hoveredSegmentId: null,
      hoverSource: null,
      heatmapConcepts: [],
      heatmapImageId: null,
      selectedHeatmapConcept: null,
      showHeatmap: false,
    }),
}));
