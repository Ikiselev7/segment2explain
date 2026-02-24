"""WebSocket endpoint wrapping the synchronous run_job() pipeline.

Runs run_job() in a thread, diffs each yield against previous state,
and sends granular WS messages for each change.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import threading
import traceback
import uuid
from queue import Empty, Queue
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from backend.image_service import (
    get_image_array,
    get_pixel_spacing,
    mask_to_contour_points,
    segment_color_hex,
    store_heatmaps,
    store_segment_masks,
)
from backend.pipeline import run_job, run_parallel_job
from orchestrator import create_job_state
from backend.schemas import (
    ChatDeltaMessage,
    ChatMessageData,
    DebugMessage,
    JobCompletedMessage,
    JobFailedMessage,
    JobStartedMessage,
    MeasurementsMessage,
    OverlayReadyMessage,
    ConceptLinkedMessage,
    HeatmapsReadyMessage,
    SegmentAddedMessage,
    SegmentData,
    SegmentMeasurements,
    SegmentRemovedMessage,
    SegmentUpdatedMessage,
    StepAddedMessage,
    StepData,
    StepUpdatedMessage,
)

logger = logging.getLogger(__name__)


class _CancelledError(Exception):
    """Raised when a job is cancelled."""


async def ws_pipeline(websocket: WebSocket) -> None:
    """Handle a WebSocket connection for pipeline execution."""
    await websocket.accept()
    logger.info("WebSocket connected")

    cancel_event = threading.Event()

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"type": "error", "error": "Invalid JSON"}))
                continue

            msg_type = msg.get("type")

            if msg_type == "start_job":
                image_id = msg.get("image_id")
                prompt = msg.get("prompt", "")
                if not image_id or not prompt:
                    await websocket.send_text(json.dumps({"type": "error", "error": "Missing image_id or prompt"}))
                    continue

                mode = msg.get("mode", "sequential")
                cancel_event.clear()
                await _run_pipeline(websocket, image_id, prompt, cancel_event, mode=mode)

            elif msg_type == "cancel_job":
                cancel_event.set()

            else:
                await websocket.send_text(json.dumps({"type": "error", "error": f"Unknown message type: {msg_type}"}))

    except WebSocketDisconnect:
        cancel_event.set()
        logger.info("WebSocket disconnected")


async def _run_pipeline(
    websocket: WebSocket,
    image_id: str,
    prompt: str,
    cancel_event: threading.Event,
    *,
    mode: str = "sequential",
) -> None:
    """Run run_job() or run_parallel_job() in a thread and stream diffs as WS messages."""
    img_np = get_image_array(image_id)
    if img_np is None:
        await _send(websocket, JobFailedMessage(error="Image not found"))
        return

    # Generate job_id and send job_started IMMEDIATELY (before model loading)
    job_id = str(uuid.uuid4())
    await _send(websocket, JobStartedMessage(job_id=job_id))

    # Pre-create state with the job_id so pipeline uses the same one
    state = create_job_state()
    state.job_id = job_id
    state.pixel_spacing = get_pixel_spacing(image_id)

    queue: Queue[dict | None] = Queue()

    def _worker() -> None:
        """Thread target: run the sync generator and push yields to queue."""
        try:
            if mode == "parallel":
                gen = run_parallel_job(image=img_np, user_prompt=prompt, state=state)
            else:
                gen = run_job(image=img_np, user_prompt=prompt, compare_baseline=False, state=state)
            for yield_tuple in gen:
                if cancel_event.is_set():
                    queue.put({"_type": "cancelled"})
                    return
                chat, steps_html, annotated_img, meas_json, debug_json = yield_tuple

                # Extract masks and segment metadata before deepcopy
                # (_sync_segment_meta stores these in debug_json by reference)
                masks = debug_json.pop("_masks", {})
                seg_data = debug_json.get("_segment_data", {})
                # Extract attention heatmaps (numpy arrays, by reference)
                attn_heatmaps = debug_json.pop("_attention_heatmaps", None)

                queue.put({
                    "chat": copy.deepcopy(chat),
                    "meas": copy.deepcopy(meas_json),
                    "debug": copy.deepcopy(debug_json),
                    "masks": masks,  # numpy arrays by reference (not deepcopy'd)
                    "seg_data": copy.deepcopy(seg_data),
                    "attn_heatmaps": attn_heatmaps,
                })
            queue.put(None)  # Sentinel: done
        except Exception as e:
            queue.put({"_type": "error", "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()})

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    # State tracking for diffing
    prev_chat: list[dict] = []
    prev_meas: dict = {}
    prev_debug: dict = {}
    prev_assistant_text = ""
    known_steps: dict[str, dict] = {}  # step_id -> {status, detail}
    known_segments: dict[str, dict] = {}  # seg_id -> {label, color, ...}
    sent_concept_links: set[str] = set()  # track sent concept_linked segment_ids
    sent_heatmaps = False  # track whether heatmaps_ready was sent

    try:
        while True:
            try:
                item = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: queue.get(timeout=0.1)
                )
            except Empty:
                # Check if thread is still alive
                if not thread.is_alive():
                    break
                continue

            if item is None:
                # Pipeline completed
                break

            if isinstance(item, dict) and item.get("_type") == "error":
                await _send(websocket, JobFailedMessage(error=item["error"]))
                logger.error("Pipeline error: %s", item.get("tb", item["error"]))
                return

            if isinstance(item, dict) and item.get("_type") == "cancelled":
                await _send(websocket, JobFailedMessage(error="Job cancelled"))
                return

            chat = item.get("chat", [])
            meas = item.get("meas", {})
            debug = item.get("debug", {})
            masks = item.get("masks", {})
            seg_data = item.get("seg_data", {})

            # Diff steps from debug
            await _diff_steps(websocket, debug, known_steps)

            # Diff segments using real metadata from pipeline
            await _diff_segments(websocket, meas, seg_data, masks, known_segments, image_id)

            # Diff chat — send deltas for streaming assistant text
            await _diff_chat(websocket, chat, prev_chat, prev_assistant_text)
            if chat:
                last_msg = chat[-1]
                if isinstance(last_msg, dict) and last_msg.get("role") == "assistant":
                    prev_assistant_text = last_msg.get("content", "")
            prev_chat = copy.deepcopy(chat)

            # Send measurements update
            if meas != prev_meas:
                await _send(websocket, MeasurementsMessage(data=meas))
                prev_meas = copy.deepcopy(meas)

            # Send debug update (filter out internal keys and binary data)
            if debug != prev_debug:
                safe_debug = {k: v for k, v in debug.items()
                              if not isinstance(v, (bytes, memoryview))
                              and not k.startswith("_")}
                await _send(websocket, DebugMessage(data=safe_debug))
                prev_debug = copy.deepcopy(debug)

            # Send concept links (parallel mode)
            concept_links = debug.get("_concept_links")
            if concept_links:
                for cl in concept_links:
                    sid = cl.get("segment_id", "")
                    if sid and sid not in sent_concept_links:
                        await _send(websocket, ConceptLinkedMessage(
                            concept=cl["concept"],
                            segment_id=sid,
                            aliases=cl.get("aliases", []),
                            color=cl.get("color", "#3b82f6"),
                        ))
                        sent_concept_links.add(sid)

            # Store attention heatmaps and notify frontend
            attn_heatmaps = item.get("attn_heatmaps")
            if attn_heatmaps and not sent_heatmaps:
                store_heatmaps(image_id, attn_heatmaps)
                await _send(websocket, HeatmapsReadyMessage(
                    concepts=list(attn_heatmaps.keys()),
                    image_id=image_id,
                ))
                sent_heatmaps = True

            # Store masks for overlay endpoint
            if masks:
                mask_label_pairs = {}
                for sid, mask in masks.items():
                    meta = seg_data.get(sid, {})
                    label = meta.get("label", f"Seg {sid}")
                    cidx = meta.get("color_idx", 0)
                    mask_label_pairs[sid] = (mask, label, cidx)
                store_segment_masks(image_id, mask_label_pairs)

        await _send(websocket, JobCompletedMessage(job_id=job_id))

    except WebSocketDisconnect:
        cancel_event.set()
        logger.info("WebSocket disconnected during pipeline")

    thread.join(timeout=5)


async def _diff_steps(
    websocket: WebSocket,
    debug: dict,
    known_steps: dict[str, dict],
) -> None:
    """Extract step info from debug and send step_added/step_updated messages.

    Since run_job() doesn't directly expose steps through the yield tuple in a
    structured way, we infer step transitions from debug keys.
    """
    # The debug dict accumulates step-related info. We can detect steps from
    # known patterns in the debug keys.
    # For a more robust approach, we'd need to modify run_job() to expose steps.
    # For now, we infer from debug keys:
    step_hints = []

    if "R1_raw" in debug or "segmentation_mode" in debug:
        step_hints.append(("S1", "Parse request", "done", "Image received."))

    if "R1_raw" in debug:
        concepts = debug.get("R1_concepts", [])
        detail = f"Concepts: {', '.join(concepts)}" if concepts else "Extracting concepts…"
        step_hints.append(("R1", "MedGemma: extract concepts", "done", detail))

    if "SEG_segment_count" in debug:
        seg_count = debug["SEG_segment_count"]
        step_hints.append(("SEG", "MedSAM3: segmentation", "done",
                          f"Found {seg_count} region(s)."))

    # Filter steps
    for fk in [k for k in debug if k.startswith("filter_F")]:
        seg_id = fk.replace("filter_F", "")
        step_hints.append((f"F{seg_id}", f"MedGemma: filter Seg {seg_id}", "done", "Filtered"))

    if "R2_raw" in debug:
        step_hints.append(("R2", "MedGemma: analysis", "done", "Complete"))

    # Parallel mode steps
    if "ANSWER_raw" in debug:
        step_hints.append(("ANSWER", "MedGemma: answer", "done", "Answer complete"))
    if "CONCEPTS_list" in debug:
        concepts = debug["CONCEPTS_list"]
        detail = f"Selected {len(concepts)} concept(s)" if concepts else "No concepts"
        step_hints.append(("SELECT", "Select concepts", "done", detail))
    if "PRIOR_mode" in debug:
        n_heatmaps = len(debug.get("PRIOR_heatmaps", []))
        n_boxes = len(debug.get("PRIOR_boxes", {}))
        step_hints.append(("PRIOR", "Attention priors", "done",
                          f"mode={debug['PRIOR_mode']}: {n_heatmaps} heatmaps, {n_boxes} boxes"))

    for step_id, name, status, detail in step_hints:
        if step_id not in known_steps:
            known_steps[step_id] = {"status": status, "detail": detail}
            await _send(websocket, StepAddedMessage(
                step=StepData(id=step_id, name=name, status=status, detail=detail, segment_ids=[])
            ))
        elif known_steps[step_id]["status"] != status or known_steps[step_id]["detail"] != detail:
            known_steps[step_id] = {"status": status, "detail": detail}
            await _send(websocket, StepUpdatedMessage(step_id=step_id, status=status, detail=detail))


async def _diff_segments(
    websocket: WebSocket,
    meas: dict,
    seg_data: dict,
    masks: dict,
    known_segments: dict[str, dict],
    image_id: str,
) -> None:
    """Detect new/removed/updated segments using pipeline metadata."""
    current_ids = set(meas.keys()) | set(seg_data.keys())
    known_ids = set(known_segments.keys())

    # New segments
    for seg_id in current_ids - known_ids:
        seg_meas = meas.get(seg_id, {})
        meta = seg_data.get(seg_id, {})
        idx = meta.get("color_idx", len(known_segments))
        color = segment_color_hex(idx)

        # Get label from pipeline metadata
        label = meta.get("label", f"Segment {seg_id}")
        description = meta.get("description", "")
        created_by = meta.get("created_by_step", "")
        concept = meta.get("concept", "")

        # Extract contour points from mask
        contour_points: list[list[list[int]]] = []
        if seg_id in masks:
            try:
                contour_points = mask_to_contour_points(masks[seg_id])
            except Exception:
                logger.warning("Failed to extract contours for segment %s", seg_id)

        bbox = seg_meas.get("bbox_px", meta.get("bbox", []))
        measurements = SegmentMeasurements(
            area_px=seg_meas.get("area_px", 0),
            bbox_px=bbox if isinstance(bbox, list) else list(bbox),
            max_diameter_px=seg_meas.get("max_diameter_px", 0.0),
            centroid_px=seg_meas.get("centroid_px", []),
            area_mm2=seg_meas.get("area_mm2"),
            max_diameter_mm=seg_meas.get("max_diameter_mm"),
            pixel_spacing_mm=seg_meas.get("pixel_spacing_mm"),
        )

        await _send(websocket, SegmentAddedMessage(
            segment=SegmentData(
                id=seg_id,
                label=label,
                description=description,
                created_by_step=created_by,
                bbox=bbox if isinstance(bbox, list) else list(bbox),
                contour_points=contour_points,
                color=color,
                measurements=measurements,
                concept=concept,
            )
        ))
        known_segments[seg_id] = {"label": label, "description": description, "color": color, "concept": concept}

    # Updated segments (label/description changed by filter step)
    for seg_id in current_ids & known_ids:
        meta = seg_data.get(seg_id, {})
        new_label = meta.get("label", "")
        new_desc = meta.get("description", "")
        prev = known_segments[seg_id]
        if new_label and (new_label != prev.get("label") or new_desc != prev.get("description")):
            await _send(websocket, SegmentUpdatedMessage(
                segment_id=seg_id, label=new_label, description=new_desc,
            ))
            known_segments[seg_id]["label"] = new_label
            known_segments[seg_id]["description"] = new_desc

    # Removed segments
    for seg_id in known_ids - current_ids:
        await _send(websocket, SegmentRemovedMessage(segment_id=seg_id))
    for seg_id in known_ids - current_ids:
        known_segments.pop(seg_id, None)


async def _diff_chat(
    websocket: WebSocket,
    chat: list[dict],
    prev_chat: list[dict],
    prev_assistant_text: str,
) -> None:
    """Send chat message diffs: new user messages + assistant streaming deltas."""
    if not chat:
        return

    # New messages (added since last yield): user + reasoning
    new_start = len(prev_chat)
    for i in range(new_start, len(chat)):
        msg = chat[i]
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "user":
            await _send(websocket, ChatMessageData(role="user", content=msg.get("content", "")))
        elif msg.get("reasoning"):
            await _send(websocket, ChatMessageData(
                role="assistant", content=msg.get("content", ""), reasoning=True,
            ))

    # Streaming / in-place updates of the last assistant message.
    # New appended messages are handled by the loop above; this section
    # only fires for in-place mutations (same list length or grew without
    # a matching new-message entry).
    if chat:
        last = chat[-1]
        if isinstance(last, dict) and last.get("role") == "assistant":
            current_text = last.get("content", "")
            is_reasoning = bool(last.get("reasoning"))
            is_in_place = len(chat) <= len(prev_chat)

            if is_reasoning and is_in_place:
                # R1 reasoning streamed via in-place update of chat[-1]
                if current_text and current_text != prev_assistant_text:
                    await _send(websocket, ChatMessageData(
                        role="assistant", content=current_text, reasoning=True,
                    ))
            elif not is_reasoning and current_text and current_text != prev_assistant_text:
                # Regular assistant streaming (R2 deltas, status updates)
                if current_text.startswith(prev_assistant_text) and prev_assistant_text:
                    delta = current_text[len(prev_assistant_text):]
                    if delta:
                        await _send(websocket, ChatDeltaMessage(text=delta))
                else:
                    # Text changed completely (e.g., placeholder replaced by
                    # answer).  Send as a replace-delta so the frontend puts
                    # it in the streaming buffer instead of creating a new
                    # chat message (which would cause duplication).
                    await _send(websocket, ChatDeltaMessage(text=current_text, replace=True))


async def _send(websocket: WebSocket, msg: Any) -> None:
    """Send a Pydantic model as JSON over WebSocket."""
    try:
        await websocket.send_text(msg.model_dump_json())
    except Exception:
        pass  # Connection may be closed
