/** WebSocket connection hook with reconnect and message dispatch */

import { useCallback, useEffect, useRef } from "react";
import { useJobStore } from "../stores/jobStore";
import type { ClientMessage, ServerMessage } from "../types/messages";
import type { StepStatus } from "../types/pipeline";

const WS_URL = `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/ws/pipeline`;
const RECONNECT_DELAY = 2000;

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const {
    startJob,
    addStep,
    updateStep,
    addSegment,
    updateSegment,
    removeSegment,
    addChatMessage,
    appendStreamingText,
    replaceStreamingText,
    setOverlayUrl,
    addConceptLink,
    setHeatmapConcepts,
    completeJob,
    failJob,
  } = useJobStore();

  const handleMessage = useCallback(
    (event: MessageEvent) => {
      let msg: ServerMessage;
      try {
        msg = JSON.parse(event.data);
      } catch {
        return;
      }

      switch (msg.type) {
        case "job_started":
          startJob(msg.job_id);
          break;

        case "step_added":
          addStep({
            id: msg.step.id,
            name: msg.step.name,
            status: msg.step.status as StepStatus,
            detail: msg.step.detail,
            segmentIds: msg.step.segment_ids,
          });
          break;

        case "step_updated":
          updateStep(msg.step_id, msg.status as StepStatus, msg.detail);
          break;

        case "segment_added": {
          // Normalize contour_points: old format [[x,y],...] → new format [[[x,y],...]]
          let contours = msg.segment.contour_points;
          if (
            contours.length > 0 &&
            typeof contours[0][0] === "number"
          ) {
            // Old flat format — wrap in array to create single polygon
            contours = [contours as unknown as number[][]];
          }
          addSegment({
            id: msg.segment.id,
            label: msg.segment.label,
            description: msg.segment.description,
            createdByStep: msg.segment.created_by_step,
            bbox: msg.segment.bbox,
            contourPoints: contours,
            color: msg.segment.color,
            concept: msg.segment.concept ?? "",
            measurements: msg.segment.measurements,
          });
          break;
        }

        case "segment_updated":
          updateSegment(msg.segment_id, msg.label, msg.description);
          break;

        case "segment_removed":
          removeSegment(msg.segment_id);
          break;

        case "chat_message":
          addChatMessage({ role: msg.role, content: msg.content, reasoning: msg.reasoning });
          break;

        case "chat_delta":
          if (msg.replace) {
            replaceStreamingText(msg.text);
          } else {
            appendStreamingText(msg.text);
          }
          break;

        case "overlay_ready":
          setOverlayUrl(msg.url);
          break;

        case "concept_linked":
          addConceptLink({
            concept: msg.concept,
            segmentId: msg.segment_id,
            aliases: msg.aliases,
            color: msg.color,
          });
          break;

        case "heatmaps_ready":
          setHeatmapConcepts(msg.concepts, msg.image_id);
          break;

        case "job_completed":
          completeJob();
          break;

        case "job_failed":
          failJob(msg.error);
          break;

        case "measurements":
        case "debug":
          // Stored in debug panel if needed
          break;

        case "error":
          console.warn("WS error:", msg.error);
          break;
      }
    },
    [
      startJob, addStep, updateStep,
      addSegment, updateSegment, removeSegment,
      addChatMessage, appendStreamingText, replaceStreamingText,
      setOverlayUrl, addConceptLink, setHeatmapConcepts, completeJob, failJob,
    ]
  );

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("WebSocket connected");
    };

    ws.onmessage = handleMessage;

    ws.onclose = () => {
      console.log("WebSocket closed, reconnecting…");
      reconnectTimer.current = setTimeout(connect, RECONNECT_DELAY);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, [handleMessage]);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  const send = useCallback((msg: ClientMessage) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  }, []);

  return { send };
}
