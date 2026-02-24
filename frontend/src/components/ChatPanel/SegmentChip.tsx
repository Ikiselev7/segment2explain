import { useHover } from "../../hooks/useHover";
import { useJobStore } from "../../stores/jobStore";

interface Props {
  segmentId: string;
}

export function SegmentChip({ segmentId }: Props) {
  const segment = useJobStore((s) => s.segments.get(segmentId));
  const { hoveredSegmentId, hoverFromChat, clearHover } = useHover();
  const isHighlighted = hoveredSegmentId === segmentId;
  const label = segment?.label || `Segment ${segmentId}`;
  const color = segment?.color || "#3b82f6";

  return (
    <span
      className={`seg-chip ${isHighlighted ? "seg-chip-highlighted" : ""}`}
      style={{
        borderLeft: `3px solid ${color}`,
        backgroundColor: isHighlighted ? `${color}44` : `${color}20`,
        ...(isHighlighted ? { boxShadow: `0 0 8px ${color}80` } : {}),
      }}
      onMouseEnter={() => hoverFromChat(segmentId)}
      onMouseLeave={clearHover}
    >
      <span className="seg-chip-dot" style={{ backgroundColor: color }} />
      {label}
    </span>
  );
}
