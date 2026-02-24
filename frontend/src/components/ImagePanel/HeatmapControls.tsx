import { useJobStore } from "../../stores/jobStore";

/**
 * Small overlay controls for image visualization toggles.
 * Positioned top-right over the image.
 * - Segments toggle: show/hide segment overlays
 * - Heatmap toggle: show/hide attention heatmap (hover-driven)
 */
export function HeatmapControls() {
  const segments = useJobStore((s) => s.segments);
  const showSegments = useJobStore((s) => s.showSegments);
  const toggleSegments = useJobStore((s) => s.toggleSegments);
  const heatmapConcepts = useJobStore((s) => s.heatmapConcepts);
  const showHeatmap = useJobStore((s) => s.showHeatmap);
  const toggleHeatmap = useJobStore((s) => s.toggleHeatmap);
  const selectedHeatmapConcept = useJobStore((s) => s.selectedHeatmapConcept);

  const hasSegments = segments.size > 0;
  const hasHeatmaps = heatmapConcepts.length > 0;

  if (!hasSegments && !hasHeatmaps) return null;

  const heatmapTitle = showHeatmap
    ? selectedHeatmapConcept
      ? `Attention: ${selectedHeatmapConcept}`
      : "Hover a segment to see its attention map"
    : "Show attention map";

  return (
    <div className="heatmap-controls">
      {hasSegments && (
        <button
          className={`heatmap-toggle ${showSegments ? "heatmap-toggle-active" : ""}`}
          onClick={toggleSegments}
          title={showSegments ? "Hide segments" : "Show segments"}
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path
              d="M2 4h12M2 8h12M2 12h12"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
            />
          </svg>
        </button>
      )}
      {hasHeatmaps && (
        <button
          className={`heatmap-toggle ${showHeatmap ? "heatmap-toggle-active" : ""}`}
          onClick={toggleHeatmap}
          title={heatmapTitle}
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <rect
              x="1"
              y="1"
              width="14"
              height="14"
              rx="2"
              fill={showHeatmap ? "currentColor" : "none"}
              stroke="currentColor"
              strokeWidth="1.5"
              opacity={showHeatmap ? 0.3 : 1}
            />
            <circle cx="5" cy="8" r="2.5" fill="currentColor" opacity="0.8" />
            <circle cx="10" cy="6" r="2" fill="currentColor" opacity="0.5" />
            <circle cx="8" cy="11" r="1.5" fill="currentColor" opacity="0.6" />
          </svg>
        </button>
      )}
      {showHeatmap && selectedHeatmapConcept && (
        <div className="heatmap-concept-label">{selectedHeatmapConcept}</div>
      )}
    </div>
  );
}
