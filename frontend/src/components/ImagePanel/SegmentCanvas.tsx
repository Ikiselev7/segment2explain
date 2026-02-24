import { useCallback, useEffect, useRef } from "react";
import { useHover } from "../../hooks/useHover";
import { useJobStore } from "../../stores/jobStore";
import type { Segment } from "../../types/pipeline";
import { HeatmapControls } from "./HeatmapControls";

/** Parse hex color "#rrggbb" to [r, g, b] */
function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [
    parseInt(h.substring(0, 2), 16),
    parseInt(h.substring(2, 4), 16),
    parseInt(h.substring(4, 6), 16),
  ];
}

/** Build a Path2D from segment contour polygons (scaled to canvas).
 *  Each segment can have multiple polygons (disconnected mask regions).
 *  Each polygon is drawn as a separate sub-path.
 */
function buildContourPath(
  seg: Segment,
  scaleX: number,
  scaleY: number,
): Path2D | null {
  if (seg.contourPoints.length === 0) return null;
  const path = new Path2D();
  for (const polygon of seg.contourPoints) {
    if (polygon.length < 3) continue;
    path.moveTo(polygon[0][0] * scaleX, polygon[0][1] * scaleY);
    for (let i = 1; i < polygon.length; i++) {
      path.lineTo(polygon[i][0] * scaleX, polygon[i][1] * scaleY);
    }
    path.closePath();
  }
  return path;
}

/** Build a smooth Path2D using quadratic Bezier curves through midpoints.
 *  Original vertices become control points; midpoints between consecutive
 *  vertices lie on the curve. Produces visually smooth contours while
 *  staying close to the original polygon.
 */
function buildSmoothContourPath(
  seg: Segment,
  scaleX: number,
  scaleY: number,
): Path2D | null {
  if (seg.contourPoints.length === 0) return null;
  const path = new Path2D();
  for (const polygon of seg.contourPoints) {
    if (polygon.length < 3) continue;
    const n = polygon.length;
    // Start at midpoint between last and first vertex
    const mx0 =
      ((polygon[n - 1][0] + polygon[0][0]) / 2) * scaleX;
    const my0 =
      ((polygon[n - 1][1] + polygon[0][1]) / 2) * scaleY;
    path.moveTo(mx0, my0);
    for (let i = 0; i < n; i++) {
      const cpx = polygon[i][0] * scaleX;
      const cpy = polygon[i][1] * scaleY;
      const next = polygon[(i + 1) % n];
      const mx = ((polygon[i][0] + next[0]) / 2) * scaleX;
      const my = ((polygon[i][1] + next[1]) / 2) * scaleY;
      path.quadraticCurveTo(cpx, cpy, mx, my);
    }
    path.closePath();
  }
  return path;
}

/** Draw a segment fill (contour polygon or bbox fallback) */
function drawSegmentFill(
  ctx: CanvasRenderingContext2D,
  seg: Segment,
  scaleX: number,
  scaleY: number,
  alpha: number,
) {
  const [r, g, b] = hexToRgb(seg.color);

  if (seg.contourPoints.length > 0) {
    const path = buildSmoothContourPath(seg, scaleX, scaleY);
    if (path) {
      ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
      ctx.fill(path);
    }
  } else if (seg.bbox.length === 4) {
    const [x0, y0, x1, y1] = seg.bbox;
    ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
    ctx.fillRect(
      x0 * scaleX,
      y0 * scaleY,
      (x1 - x0) * scaleX,
      (y1 - y0) * scaleY,
    );
  }
}

/** Draw a segment outline (contour or bbox) */
function drawSegmentOutline(
  ctx: CanvasRenderingContext2D,
  seg: Segment,
  scaleX: number,
  scaleY: number,
  lineWidth: number,
) {
  const [r, g, b] = hexToRgb(seg.color);
  ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, 0.9)`;
  ctx.lineWidth = lineWidth;

  ctx.lineJoin = "round";
  if (seg.contourPoints.length > 0) {
    const path = buildSmoothContourPath(seg, scaleX, scaleY);
    if (path) ctx.stroke(path);
  } else if (seg.bbox.length === 4) {
    const [x0, y0, x1, y1] = seg.bbox;
    ctx.strokeRect(
      x0 * scaleX,
      y0 * scaleY,
      (x1 - x0) * scaleX,
      (y1 - y0) * scaleY,
    );
  }
}

/**
 * Canvas overlay for segment visualization and hover hit-testing.
 * Draws all segments with colored fills; highlights hovered segment.
 */
export function SegmentCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const segments = useJobStore((s) => s.segments);
  const imageUrl = useJobStore((s) => s.imageUrl);
  const imageWidth = useJobStore((s) => s.imageWidth);
  const imageHeight = useJobStore((s) => s.imageHeight);
  const { hoveredSegmentId, hoverFromImage, clearHover } = useHover();
  const showSegments = useJobStore((s) => s.showSegments);

  // Heatmap overlay state
  const showHeatmap = useJobStore((s) => s.showHeatmap);
  const selectedHeatmapConcept = useJobStore((s) => s.selectedHeatmapConcept);
  const heatmapImageId = useJobStore((s) => s.heatmapImageId);
  const heatmapConcepts = useJobStore((s) => s.heatmapConcepts);
  const conceptLinks = useJobStore((s) => s.conceptLinks);
  const setSelectedHeatmapConcept = useJobStore((s) => s.setSelectedHeatmapConcept);
  const heatmapImgRef = useRef<HTMLImageElement | undefined>(undefined);
  const heatmapLoadedRef = useRef<string | null>(null); // tracks loaded concept+imageId

  // Map hovered segment → concept for heatmap display
  useEffect(() => {
    if (!showHeatmap || !hoveredSegmentId || heatmapConcepts.length === 0) return;

    // 1. Use segment's concept field directly (most reliable)
    const seg = segments.get(hoveredSegmentId);
    if (seg?.concept && heatmapConcepts.includes(seg.concept)) {
      setSelectedHeatmapConcept(seg.concept);
      return;
    }

    // 2. Try conceptLinks (parallel mode alias mapping)
    const link = conceptLinks.find((l) => l.segmentId === hoveredSegmentId);
    if (link && heatmapConcepts.includes(link.concept)) {
      setSelectedHeatmapConcept(link.concept);
      return;
    }

    // 3. Fallback: fuzzy match segment label (longest concept first)
    if (seg) {
      const segLabel = seg.label.toLowerCase();
      const sorted = [...heatmapConcepts].sort((a, b) => b.length - a.length);
      const match = sorted.find(
        (c) => segLabel.includes(c.toLowerCase()) || c.toLowerCase().includes(segLabel),
      );
      if (match) {
        setSelectedHeatmapConcept(match);
      }
    }
  }, [showHeatmap, hoveredSegmentId, conceptLinks, segments, heatmapConcepts, setSelectedHeatmapConcept]);

  // Load heatmap image when concept or visibility changes
  useEffect(() => {
    if (!showHeatmap || !selectedHeatmapConcept || !heatmapImageId) {
      heatmapImgRef.current = undefined;
      heatmapLoadedRef.current = null;
      drawRef.current();
      return;
    }
    const key = `${heatmapImageId}:${selectedHeatmapConcept}`;
    if (heatmapLoadedRef.current === key) return;

    const img = new Image();
    const encodedConcept = encodeURIComponent(selectedHeatmapConcept);
    img.src = `/api/images/${heatmapImageId}/heatmap/${encodedConcept}`;
    img.onload = () => {
      heatmapImgRef.current = img;
      heatmapLoadedRef.current = key;
      drawRef.current();
    };
    img.onerror = () => {
      heatmapImgRef.current = undefined;
      heatmapLoadedRef.current = null;
    };
  }, [showHeatmap, selectedHeatmapConcept, heatmapImageId]);

  // Ref-based draw function so ResizeObserver can call it synchronously
  const drawRef = useRef<() => void>(() => {});

  // Keep drawRef.current up-to-date with latest state
  drawRef.current = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx || !imageWidth || !imageHeight || canvas.width === 0) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw heatmap overlay if active
    if (showHeatmap && heatmapImgRef.current) {
      ctx.globalAlpha = 0.45;
      ctx.drawImage(heatmapImgRef.current, 0, 0, canvas.width, canvas.height);
      ctx.globalAlpha = 1.0;
    }

    if (segments.size === 0 || !showSegments) return;

    const scaleX = canvas.width / imageWidth;
    const scaleY = canvas.height / imageHeight;

    for (const [, seg] of segments) {
      const isHovered = hoveredSegmentId === seg.id;
      const fillAlpha = isHovered ? 0.15 : 0.30;
      drawSegmentFill(ctx, seg, scaleX, scaleY, fillAlpha);
    }

    if (hoveredSegmentId) {
      const seg = segments.get(hoveredSegmentId);
      if (seg) {
        drawSegmentOutline(ctx, seg, scaleX, scaleY, 3);
      }
    }
  };

  // Sync canvas dimensions with the displayed image element.
  // Only assigns canvas.width/height when they actually change
  // (assigning always clears the canvas). Draws immediately after
  // resize to avoid blank-frame flicker.
  const syncCanvasSize = useCallback(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;
    const img = container.querySelector("img");
    if (!img || img.clientWidth === 0 || img.clientHeight === 0) return;

    const needsResize =
      canvas.width !== img.clientWidth || canvas.height !== img.clientHeight;

    if (needsResize) {
      canvas.width = img.clientWidth;
      canvas.height = img.clientHeight;
    }

    canvas.style.left = `${img.offsetLeft}px`;
    canvas.style.top = `${img.offsetTop}px`;

    // Redraw immediately (synchronously) after resize so the browser
    // never paints a blank canvas frame.
    if (needsResize) {
      drawRef.current();
    }
  }, []);

  // ResizeObserver for window/container resizes
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver(syncCanvasSize);
    observer.observe(container);
    return () => observer.disconnect();
  }, [imageUrl, syncCanvasSize]);

  // Draw all segments + hover highlight + heatmap (reacts to state changes)
  useEffect(() => {
    drawRef.current();
  }, [segments, hoveredSegmentId, imageWidth, imageHeight, showHeatmap, selectedHeatmapConcept, showSegments]);

  // Mouse move for hit-testing (polygon-based when contours available)
  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas || segments.size === 0 || !imageWidth || !imageHeight) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      const scaleX = canvas.width / imageWidth;
      const scaleY = canvas.height / imageHeight;

      // Check contour polygons first, then bbox fallback
      for (const [, seg] of segments) {
        if (seg.contourPoints.length > 0) {
          const path = buildContourPath(seg, scaleX, scaleY);
          if (path && ctx.isPointInPath(path, mouseX, mouseY)) {
            hoverFromImage(seg.id);
            return;
          }
        } else if (seg.bbox.length === 4) {
          const [x0, y0, x1, y1] = seg.bbox;
          const px = (mouseX / canvas.width) * imageWidth;
          const py = (mouseY / canvas.height) * imageHeight;
          if (px >= x0 && px <= x1 && py >= y0 && py <= y1) {
            hoverFromImage(seg.id);
            return;
          }
        }
      }
      hoverFromImage(null);
    },
    [segments, imageWidth, imageHeight, hoverFromImage],
  );

  return (
    <div ref={containerRef} className="image-container">
      {imageUrl && (
        <>
          <img src={imageUrl} alt="Medical image" onLoad={syncCanvasSize} />
          <canvas
            ref={canvasRef}
            onMouseMove={handleMouseMove}
            onMouseLeave={clearHover}
            style={{ cursor: "crosshair" }}
          />
          <HeatmapControls />
        </>
      )}
    </div>
  );
}
