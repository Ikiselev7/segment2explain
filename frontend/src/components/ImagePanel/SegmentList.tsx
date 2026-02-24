import { useHover } from "../../hooks/useHover";
import { useJobStore } from "../../stores/jobStore";

function formatArea(m: {
  area_px: number;
  area_mm2?: number | null;
}): string {
  if (m.area_mm2 != null) {
    if (m.area_mm2 >= 100) {
      return `${(m.area_mm2 / 100).toFixed(1)} cm\u00B2`;
    }
    return `${m.area_mm2.toFixed(1)} mm\u00B2`;
  }
  return `${m.area_px.toLocaleString()} px`;
}

function formatDiameter(m: {
  max_diameter_px: number;
  max_diameter_mm?: number | null;
}): string {
  if (m.max_diameter_mm != null) {
    if (m.max_diameter_mm >= 10) {
      return `${(m.max_diameter_mm / 10).toFixed(1)} cm`;
    }
    return `${m.max_diameter_mm.toFixed(1)} mm`;
  }
  return `${m.max_diameter_px.toFixed(1)} px`;
}

export function SegmentList() {
  const segments = useJobStore((s) => s.segments);
  const { hoveredSegmentId, hoverFromImage, clearHover } = useHover();

  if (segments.size === 0) return null;

  return (
    <div className="segments-list">
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Label</th>
            <th>Area</th>
            <th>Diameter</th>
          </tr>
        </thead>
        <tbody>
          {Array.from(segments.values()).map((seg) => (
            <tr
              key={seg.id}
              className={`segment-row ${
                hoveredSegmentId === seg.id ? "segment-row-highlighted" : ""
              }`}
              onMouseEnter={() => hoverFromImage(seg.id)}
              onMouseLeave={clearHover}
            >
              <td>
                <span
                  className="segment-color-dot"
                  style={{ backgroundColor: seg.color }}
                />
                {seg.id}
              </td>
              <td>{seg.label}</td>
              <td>{formatArea(seg.measurements)}</td>
              <td>{formatDiameter(seg.measurements)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
