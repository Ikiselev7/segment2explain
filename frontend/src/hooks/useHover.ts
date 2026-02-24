/** Hook for bidirectional segment hover cross-referencing */

import { useCallback } from "react";
import { useJobStore } from "../stores/jobStore";

export function useHover() {
  const hoveredSegmentId = useJobStore((s) => s.hoveredSegmentId);
  const hoverSource = useJobStore((s) => s.hoverSource);
  const setHoveredSegment = useJobStore((s) => s.setHoveredSegment);

  const hoverFromImage = useCallback(
    (segmentId: string | null) => {
      setHoveredSegment(segmentId, segmentId ? "image" : null);
    },
    [setHoveredSegment]
  );

  const hoverFromChat = useCallback(
    (segmentId: string | null) => {
      setHoveredSegment(segmentId, segmentId ? "chat" : null);
    },
    [setHoveredSegment]
  );

  const clearHover = useCallback(() => {
    setHoveredSegment(null, null);
  }, [setHoveredSegment]);

  return {
    hoveredSegmentId,
    hoverSource,
    hoverFromImage,
    hoverFromChat,
    clearHover,
  };
}
