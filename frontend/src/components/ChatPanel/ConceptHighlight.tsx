/**
 * Inline highlight for concept terms in parallel mode.
 * Renders as underlined text with segment color, with bidirectional
 * hover interaction (hovering highlights segment on canvas and vice versa).
 */

import { useHover } from "../../hooks/useHover";

interface Props {
  text: string;
  segmentId: string;
  color: string;
}

export function ConceptHighlight({ text, segmentId, color }: Props) {
  const { hoveredSegmentId, hoverFromChat, clearHover } = useHover();
  const isActive = hoveredSegmentId === segmentId;

  return (
    <span
      className={`concept-highlight ${isActive ? "concept-highlight-active" : ""}`}
      style={{
        borderBottom: `2px solid ${color}`,
        backgroundColor: isActive ? `${color}30` : "transparent",
      }}
      onMouseEnter={() => hoverFromChat(segmentId)}
      onMouseLeave={clearHover}
    >
      {text}
    </span>
  );
}
