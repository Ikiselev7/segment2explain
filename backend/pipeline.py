"""Unified pipeline: R1 → SEG → F1..Fn → R2.

Contains run_job(), JSON parsing, concept extraction, filtering,
and all supporting helpers.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from collections.abc import Generator
from typing import Any

import numpy as np
from PIL import Image

from backend.config import (
    ATTENTION_PRIOR_ENABLED,
    ATTENTION_PRIOR_MODE,
    ITERATIVE_REFINEMENT_ENABLED,
    MAX_REFINEMENT_ITERATIONS,
    MEDSAM3_AUTO_PREPROCESS,
    MEDSAM3_CONCEPT_MASK_THRESHOLD,
    MEDSAM3_CONCEPT_MAX_MASKS_PER_CONCEPT,
    MEDSAM3_CONCEPT_THRESHOLD,
    MEDSAM3_MAX_MASKS,
    MEDSAM3_NMS_IOU_THRESH,
    REFINED_SEG_ENABLED,
    XGRAMMAR_ENABLED,
)
from backend.dependencies import get_medgemma, get_medsam3
from orchestrator import (
    JobState,
    Step,
    build_annotated_image,
    create_job_state,
    ensure_rgb_uint8,
)
from prompts.templates import build_tool_result_message
from tools.measure import measure_mask
from tools.overlay import (
    COLOR_NAMES,
    SEGMENT_COLORS,
    overlay_mask_on_image,
    overlay_multiple_masks,
)
from tools.refined_segmentation import (
    refined_segment,
    refined_segment_with_attention_overlay,
    refined_segment_with_negatives,
    refined_segment_with_priors,
)
from utils.steps_renderer import render_steps_html

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------
# JSON schemas for xgrammar constrained decoding
# -----------------------------
_MATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "matched": {"type": "array", "items": {"type": "string"}},
        "not_matched": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["matched", "not_matched"],
    "additionalProperties": False,
}

_compiled_grammars: dict[str, object] = {}


def _build_concept_select_schema() -> dict:
    """Build xgrammar-compatible JSON schema with CXR vocabulary as enum.

    Each concept term must be one of the known vocabulary entries.
    """
    vocab = _load_cxr_vocabulary()
    return {
        "type": "object",
        "properties": {
            "concepts": {
                "type": "array",
                "items": {"type": "string", "enum": vocab},
            },
        },
        "required": ["concepts"],
        "additionalProperties": False,
    }


def _get_grammar(schema_name: str) -> object | None:
    """Get a compiled xgrammar grammar by name, lazily compiling on first use."""
    if not XGRAMMAR_ENABLED:
        return None
    if schema_name in _compiled_grammars:
        return _compiled_grammars[schema_name]
    schemas: dict[str, dict] = {
        "match": _MATCH_SCHEMA,
        "concept_select": _build_concept_select_schema(),
    }
    schema = schemas.get(schema_name)
    if schema is None:
        return None
    medgemma = get_medgemma()
    compiled = medgemma.compile_json_schema(schema, cache_key=schema_name)
    if compiled is not None:
        _compiled_grammars[schema_name] = compiled
    return compiled


# -----------------------------
# JSON parsing helpers
# -----------------------------


def parse_tool_calls(text: str) -> list[dict]:
    """Extract <TOOL_CALL>...</TOOL_CALL> blocks from model output."""
    pattern = r"<TOOL_CALL>(.*?)</TOOL_CALL>"
    calls = []
    for match in re.finditer(pattern, text, re.DOTALL):
        raw = match.group(1).strip()
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "bbox_px" in obj:
                calls.append(obj)
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse tool call JSON: %s", raw[:200])
            continue
    return calls


def strip_tool_calls(text: str) -> str:
    """Remove <TOOL_CALL>...</TOOL_CALL> blocks from display text."""
    return re.sub(r"<TOOL_CALL>.*?</TOOL_CALL>", "", text, flags=re.DOTALL).strip()


def _has_bbox(r: dict) -> bool:
    """Check if a region dict has any bbox key."""
    return isinstance(r, dict) and ("bbox_pct" in r or "bbox_px" in r)


def _extract_regions_json(text: str) -> list[dict]:
    """Parse region proposals from MedGemma's JSON output.

    Accepts both {"regions": [...]} and bare [...] formats.
    Accepts both bbox_pct (percentage) and bbox_px (pixel) keys.
    """
    if not text:
        return []
    # Try to find JSON in the text
    for candidate in re.findall(r"\{.*\}", text, flags=re.DOTALL):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and "regions" in obj:
                regions = obj["regions"]
                if isinstance(regions, list):
                    return [r for r in regions if _has_bbox(r)]
        except (json.JSONDecodeError, ValueError):
            continue
    # Try bare array
    for candidate in re.findall(r"\[.*\]", text, flags=re.DOTALL):
        try:
            arr = json.loads(candidate)
            if isinstance(arr, list):
                return [r for r in arr if _has_bbox(r)]
        except (json.JSONDecodeError, ValueError):
            continue
    return []


def _region_to_bbox_px(
    region: dict,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int] | None:
    """Convert a region's bbox (pct or px) to clamped pixel coordinates.

    Returns (x_min, y_min, x_max, y_max) or None if invalid.
    """
    try:
        if "bbox_pct" in region:
            pct = region["bbox_pct"]
            x0 = float(pct[0]) / 100.0 * img_w
            y0 = float(pct[1]) / 100.0 * img_h
            x1 = float(pct[2]) / 100.0 * img_w
            y1 = float(pct[3]) / 100.0 * img_h
        elif "bbox_px" in region:
            raw = region["bbox_px"]
            x0, y0, x1, y1 = float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])
        else:
            return None

        # Clamp to image bounds
        x0 = max(0, min(int(x0), img_w - 1))
        y0 = max(0, min(int(y0), img_h - 1))
        x1 = max(0, min(int(x1), img_w - 1))
        y1 = max(0, min(int(y1), img_h - 1))

        # Ensure min < max with at least 1px
        if x1 <= x0:
            x1 = min(x0 + 1, img_w - 1)
        if y1 <= y0:
            y1 = min(y0 + 1, img_h - 1)

        return (x0, y0, x1, y1)
    except (TypeError, IndexError, ValueError) as e:
        logger.warning("Invalid bbox in region: %s (%s)", region, e)
        return None


def _parse_validation_json(text: str) -> dict | None:
    """Parse MedGemma validation response.

    Expects {"status": "ok"} or {"status": "adjust", "bbox_pct": [x0, y0, x1, y1]}.
    Returns the parsed dict or None if invalid.
    """
    if not text:
        return None
    for match in re.finditer(r"\{[^{}]*\}", text):
        try:
            obj = json.loads(match.group())
            if "status" not in obj:
                continue
            if obj["status"] == "ok":
                return obj
            if obj["status"] == "adjust" and "bbox_pct" in obj:
                pct = obj["bbox_pct"]
                if isinstance(pct, list) and len(pct) == 4:
                    return obj
        except json.JSONDecodeError:
            continue
    return None


def _strip_medgemma_thinking(text: str) -> str:
    """Strip MedGemma ``<unused94>thought ...`` chain-of-thought prefix."""
    cleaned = re.sub(r"<unused\d+>(?:thought|end_of_thought)?\s*", "", text)
    return cleaned.strip()


# Pattern that marks the start of MedGemma degeneration in R2 streaming.
# Once the model emits <unusedNN> tokens or ```json/``` code fences after
# the main content, all subsequent text is garbage and should be discarded.
_DEGENERATION_RE = re.compile(r"<unused\d+>|```json\s*[\[{]|```\s*\n\s*[\[{]", re.DOTALL)


def _detect_content_degeneration(text: str) -> bool:
    """Detect repetitive content patterns (not just token artifacts).

    Returns True if the text shows signs of content-level repetition
    such as repeated sentence openings or looping phrases.
    """
    sentences = re.split(r"[.!?]\s+", text)
    if len(sentences) < 5:
        return False
    # Check for repeated sentence openings (3+ with same 30-char prefix)
    starts = [s[:30].lower().strip() for s in sentences if len(s) > 30]
    if starts:
        from collections import Counter
        start_counts = Counter(starts)
        if any(c >= 3 for c in start_counts.values()):
            return True
    return False


def _clean_r2_stream(accumulated: str) -> tuple[str, bool]:
    """Clean accumulated R2 text and detect degeneration.

    Returns (cleaned_text, should_stop).
    """
    m = _DEGENERATION_RE.search(accumulated)
    if m:
        # Truncate at the degeneration marker
        cleaned = accumulated[: m.start()].rstrip()
        return cleaned, True
    # Check for content-level repetition
    if _detect_content_degeneration(accumulated):
        return accumulated, True
    return accumulated, False


_CONCEPTS_HEADER_RE = re.compile(
    r"(?:^|\n)\s*\d*\.?\s*\**(?:CONCEPTS?|TARGETS?|SEGMENT)\**[:\s]",
    re.IGNORECASE,
)
_BULLET_RE = re.compile(
    r"^\s*[-*\u2022]\s+(.+)",  # "- Heart (cardiac silhouette)"
    re.MULTILINE,
)


def _extract_concepts_from_bullets(text: str) -> list[str] | None:
    """Fallback: extract concept phrases from bullet-pointed text.

    When MedGemma produces structured text with a CONCEPTS section and
    bullet points instead of JSON, we extract the bullet items as concepts.
    Parenthetical annotations like "(cardiac silhouette)" are stripped.
    """
    # Find the CONCEPTS section; if absent, scan the whole text
    header_match = _CONCEPTS_HEADER_RE.search(text)
    search_text = text[header_match.start():] if header_match else text
    bullets = _BULLET_RE.findall(search_text)
    if not bullets:
        return None
    concepts: list[str] = []
    for raw in bullets:
        # Strip parenthetical annotations: "Heart (cardiac silhouette)" → "Heart"
        cleaned = re.sub(r"\s*\(.*?\)\s*", " ", raw).strip()
        # Strip trailing punctuation
        cleaned = cleaned.rstrip(".,;:")
        if cleaned and len(cleaned.split()) <= 5:
            concepts.append(cleaned)
    return concepts if concepts else None


def _parse_concepts_json(text: str) -> list[str] | None:
    """Parse MedGemma concepts suggestion response.

    Handles ``{"concepts": [...]}`` object format, bare JSON arrays,
    code fences, and falls back to extracting bullet-pointed concepts
    from structured text when the model doesn't produce JSON.
    Returns list of concept strings, or None if unparseable.
    """
    if not text:
        return None
    cleaned = _strip_medgemma_thinking(text)
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned).strip()
    # Try object format first: {"concepts": [...], ...}
    for match in re.finditer(r"\{[^}]*\}", cleaned, re.DOTALL):
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict) and "concepts" in obj:
                arr = obj["concepts"]
                if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
                    return arr
        except json.JSONDecodeError:
            continue
    # Fallback: bare array
    for match in re.finditer(r"\[.*?\]", cleaned, re.DOTALL):
        try:
            arr = json.loads(match.group())
            if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
                return arr
        except json.JSONDecodeError:
            continue
    # Fallback: extract concepts from bullet-point lists.
    # MedGemma sometimes lists concepts as "- Heart (cardiac silhouette)"
    # instead of producing JSON.
    return _extract_concepts_from_bullets(cleaned)


def _extract_thinking(text: str) -> tuple[str, str]:
    """Extract MedGemma's chain-of-thought reasoning from response.

    MedGemma may produce reasoning via:
    1. ``<unusedNN>thought … end_of_thought`` tokens (native thinking mode)
    2. Free text before the JSON output (prompted reasoning)

    Returns ``(thinking_text, cleaned_text)`` where *cleaned_text* has
    thinking tokens stripped and is ready for JSON parsing.
    """
    if not text:
        return "", ""

    # 1. Capture native thinking tokens (<unusedNN>thought ... end_of_thought)
    thinking_parts: list[str] = []
    for m in re.finditer(
        r"<unused\d+>\s*thought\s+(.*?)\s*end_of_thought",
        text,
        re.DOTALL,
    ):
        part = m.group(1).strip()
        if part:
            thinking_parts.append(part)

    # 2. Strip all <unusedNN> markers
    cleaned = re.sub(r"<unused\d+>(?:thought|end_of_thought)?\s*", "", text).strip()
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned).strip()

    # 3. Capture free text before first JSON object
    json_pos = cleaned.find("{")
    if json_pos > 0:
        pre_json = cleaned[:json_pos].strip()
        if pre_json:
            thinking_parts.append(pre_json)

    thinking = "\n".join(thinking_parts).strip()
    return thinking, cleaned


_GENERIC_CONCEPT_TERMS = {
    "abnormality",
    "abnormalities",
    "anything",
    "area",
    "cxr",
    "finding",
    "findings",
    "image",
    "medical image",
    "region",
    "scan",
    "structure",
    "structures",
    "x ray",
    "xray",
}

def _normalize_concept_text(text: str) -> str:
    """Normalize free-form model text to a segmentation prompt candidate.

    Strips HTML tags, punctuation, and excess whitespace. Does NOT apply
    any hardcoded anatomy mappings — the model's output is used as-is.
    """
    cleaned = text.lower()
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = cleaned.replace("_", " ")
    cleaned = re.sub(r"[^a-z0-9/ -]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -")
    return cleaned


def _split_compound_concepts(text: str) -> list[str]:
    """Split compound concepts like 'nodule/mass' or 'heart and lungs'."""
    work = text.replace("&", " and ")
    work = work.replace("/", " and ")
    work = re.sub(r"[|,;]+", " and ", work)
    parts = [p.strip() for p in re.split(r"\band\b", work) if p.strip()]
    return parts or [text.strip()]


def _prepare_medsam3_concepts(
    user_prompt: str,
    parsed_concepts: list[str] | None,
    max_concepts: int = 8,
) -> list[str]:
    """Convert raw MedGemma concepts into clean MedSAM3 text prompts.

    No hardcoded concept hints or fallback heuristics — we trust MedGemma
    to produce the right concepts. Only normalizes text and deduplicates.
    """
    candidates: list[str] = []
    raw_items = parsed_concepts if parsed_concepts is not None else []
    for raw in raw_items:
        for piece in _split_compound_concepts(raw):
            normalized = _normalize_concept_text(piece)
            if normalized:
                candidates.append(normalized)

    deduped: list[str] = []
    seen: set[str] = set()
    for concept in candidates:
        concept = _normalize_concept_text(concept)
        if not concept or concept in _GENERIC_CONCEPT_TERMS:
            continue
        if len(concept.split()) > 4:
            continue
        if concept in seen:
            continue
        seen.add(concept)
        deduped.append(concept)
        if len(deduped) >= max_concepts:
            break

    return deduped


def _parse_classify_json(text: str) -> dict | None:
    """Parse MedGemma segment classification response.

    Returns {"name": ..., "description": ..., "relevant": bool} or None.
    The ``relevant`` field is normalized from string if needed.
    """
    if not text:
        return None
    cleaned = _strip_medgemma_thinking(text)
    for match in re.finditer(r"\{[^{}]*\}", cleaned):
        try:
            obj = json.loads(match.group())
            if obj.get("name"):
                # Normalize relevant field (string → bool)
                rel = obj.get("relevant")
                if isinstance(rel, str):
                    obj["relevant"] = rel.lower().strip() in ("true", "yes", "1")
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _parse_concept_match_json(text: str) -> dict | None:
    """Parse concept-level match result from MedGemma.

    Expects ``{"matched": ["concept1", ...], "not_matched": ["concept3", ...]}``
    Returns the parsed dict with ``matched`` and ``not_matched`` lists, or None.
    """
    if not text:
        return None
    cleaned = _strip_medgemma_thinking(text)
    for m in re.finditer(r'\{[^{}]*"matched"\s*:\s*\[.*?\].*?\}', cleaned, re.DOTALL):
        try:
            obj = json.loads(m.group())
            if isinstance(obj.get("matched"), list) and isinstance(obj.get("not_matched"), list):
                return obj
        except json.JSONDecodeError:
            continue
    return None


# -----------------------------
#   Segment metadata sync for WS adapter
# -----------------------------


def _sync_segment_meta(state: JobState) -> None:
    """Sync segment metadata + masks into state.debug for the WS adapter.

    Called before each yield so the WS adapter can extract labels,
    contour points, and masks without needing direct access to state.segments.
    """
    state.debug["_segment_data"] = {
        sid: {
            "label": s.get("validated_name") or s["label"],
            "bbox": list(s["bbox"]),
            "description": s.get("validated_desc", ""),
            "created_by_step": s["created_by_step"],
            "color_idx": s.get("color_idx", 0),
            "concept": s.get("concept", ""),
        }
        for sid, s in state.segments.items()
    }
    # Masks stored by reference (not deepcopy'd by WS adapter).
    state.debug["_masks"] = {sid: s["mask"] for sid, s in state.segments.items()}


# -----------------------------
# Main pipeline: S1 → R1 → SEG → F1..Fn → R2
# MedGemma decides what concepts to segment for any query type.
# -----------------------------


def run_job(
    image: Any,
    user_prompt: str,
    compare_baseline: bool,
    state: JobState | None,
) -> Generator[tuple[list[dict], str, Any, dict[str, Any], dict[str, Any]], None, None]:
    """Unified pipeline: R1 → SEG → F1..Fn → R2.

    MedGemma extracts concepts from any query type, then MedSAM3 segments,
    MedGemma filters, and MedGemma analyzes with evidence.

    Yields: (chat, steps_html, annotated_image, measurements_json, debug_json)
    """
    if state is None:
        state = create_job_state()

    job_t0 = time.perf_counter()

    # Validate
    if image is None:
        logger.warning("run_job: no image provided")
        state.chat.append({"role": "user", "content": user_prompt or ""})
        state.chat.append({"role": "assistant", "content": "Please upload an image first."})
        yield state.chat, render_steps_html(state.steps), None, {}, {"error": "no image"}
        return

    # Convert image to numpy RGB uint8
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        img_np = image
    else:
        img_np = np.array(image)
    img_np = ensure_rgb_uint8(img_np)

    state.image = img_np
    state.job_id = state.job_id or str(uuid.uuid4())
    state.debug = {}
    img_h, img_w = img_np.shape[:2]

    logger.info(
        "run_job START job=%s image=%dx%d prompt='%s' baseline=%s",
        state.job_id[:8],
        img_w,
        img_h,
        user_prompt[:80],
        compare_baseline,
    )

    # Add user message + placeholder assistant message
    state.chat.append({"role": "user", "content": user_prompt})
    state.chat.append({"role": "assistant", "content": "Analyzing image…"})

    # Initialize steps
    state.steps = [
        Step(id="S1", name="Parse request", status="done", detail="Image received.", segment_ids=[]),
    ]

    pil_image = Image.fromarray(img_np)

    state.steps.append(
        Step(id="R1", name="MedGemma: extract concepts", status="queued", detail="", segment_ids=[]),
    )
    _sync_segment_meta(state); yield state.chat, render_steps_html(state.steps), build_annotated_image(state), {}, state.debug

    # Load prompts
    with open(os.path.join(BASE_DIR, "prompts", "extract_target_prompt.txt"), encoding="utf-8") as f:
        extract_target_prompt = f.read()
    with open(os.path.join(BASE_DIR, "prompts", "cxr_concepts.txt"), encoding="utf-8") as f:
        cxr_lines = [
            line.strip() for line in f
            if line.strip() and not line.strip().startswith("#")
        ]
    extract_target_prompt = extract_target_prompt.replace(
        "{cxr_concepts}", ", ".join(cxr_lines)
    )
    with open(os.path.join(BASE_DIR, "prompts", "analysis_prompt.txt"), encoding="utf-8") as f:
        analysis_prompt = f.read()
    with open(os.path.join(BASE_DIR, "prompts", "filter_segment_prompt.txt"), encoding="utf-8") as f:
        filter_prompt = f.read()
    with open(os.path.join(BASE_DIR, "prompts", "match_segments_prompt.txt"), encoding="utf-8") as f:
        match_prompt = f.read()

    medgemma = get_medgemma()
    meas_json: dict[str, Any] = {}

    # ---- STEP R1: MedGemma reasons about query and extracts target concepts ----
    state.steps[1].status = "running"
    state.steps[1].detail = "Extracting target concepts from question…"
    state.chat[-1] = {"role": "assistant", "content": "Identifying target concepts…"}
    _sync_segment_meta(state); yield state.chat, render_steps_html(state.steps), build_annotated_image(state), meas_json, state.debug

    # Ask MedGemma to reason about the question and propose concepts.
    # No xgrammar here — let MedGemma think freely (chain-of-thought),
    # then parse concepts JSON from the output.
    # Reasoning text is streamed to chat in real-time.
    concepts_text = ""
    json_started = False
    try:
        for delta in medgemma.chat_stream(
            user_content=f"{extract_target_prompt}\n\nUser question: {user_prompt}",
            images=[pil_image],
            system_prompt=None,
            max_new_tokens=500,
            do_sample=False,
            capture_attention=False,  # GradCAM does its own forward pass
        ):
            concepts_text += delta
            # Stream reasoning to chat until JSON output starts
            if not json_started:
                display = re.sub(r"<unused\d+>(?:thought|end_of_thought)?\s*", "", concepts_text).strip()
                display = re.sub(r"```(?:json)?\s*", "", display).strip()
                json_pos = display.find("{")
                if json_pos >= 0:
                    json_started = True
                    reasoning_so_far = display[:json_pos].strip()
                else:
                    reasoning_so_far = display
                if reasoning_so_far:
                    state.chat[-1] = {"role": "assistant", "content": f"**R1 reasoning:** {reasoning_so_far}", "reasoning": True}
                    _sync_segment_meta(state); yield state.chat, render_steps_html(state.steps), build_annotated_image(state), meas_json, state.debug
    except Exception as e:
        state.steps[1].status = "failed"
        state.steps[1].detail = f"Concept extraction failed: {e}"
        state.chat[-1] = {"role": "assistant", "content": f"Concept extraction failed: {e}"}
        _sync_segment_meta(state); yield state.chat, render_steps_html(state.steps), build_annotated_image(state), meas_json, state.debug
        return

    state.debug["R1_raw"] = concepts_text
    thinking, cleaned_text = _extract_thinking(concepts_text)
    raw_concepts = _parse_concepts_json(cleaned_text)
    concepts = _prepare_medsam3_concepts(user_prompt, raw_concepts)
    state.debug["R1_concepts_raw"] = raw_concepts
    state.debug["R1_concepts"] = concepts
    state.debug["R1_thinking"] = thinking
    state.debug["segmentation_mode"] = "medsam3_refined_segment"
    logger.info(
        "R1: parsed_concepts=%s prepared_concepts=%s thinking='%s' raw='%s'",
        raw_concepts,
        concepts,
        thinking[:100],
        concepts_text[:200],
    )

    if not concepts:
        state.steps[1].status = "failed"
        state.steps[1].detail = "No usable concepts for MedSAM3."
        state.chat[-1] = {
            "role": "assistant",
            "content": "Could not prepare segmentation concepts from the question. Please ask with a specific target.",
        }
        _sync_segment_meta(state); yield state.chat, render_steps_html(state.steps), build_annotated_image(state), meas_json, state.debug
        return

    state.steps[1].status = "done"
    state.steps[1].detail = f"Concepts: {', '.join(concepts)}"
    if thinking:
        # Finalize reasoning message (was being streamed into state.chat[-1])
        state.chat[-1] = {"role": "assistant", "content": f"**R1 reasoning:** {thinking}", "reasoning": True}
    state.chat.append({"role": "assistant", "content": f"Searching for: **{', '.join(concepts)}**"})
    _sync_segment_meta(state); yield state.chat, render_steps_html(state.steps), build_annotated_image(state), meas_json, state.debug

    # ---- STEP PRIOR: Extract attention-based spatial priors (optional) ----
    img_h, img_w = img_np.shape[:2]
    spatial_priors: dict[str, tuple[int, int, int, int]] = {}
    attention_heatmaps: dict[str, np.ndarray] = {}
    if ATTENTION_PRIOR_ENABLED:
        prior_step = Step(id="PRIOR", name="Attention priors", status="running", detail="Extracting spatial priors…", segment_ids=[])
        state.steps.append(prior_step)
        _sync_segment_meta(state); yield state.chat, render_steps_html(state.steps), build_annotated_image(state), meas_json, state.debug

        try:
            from models.attention_prior import heatmap_to_box

            # GradCAM first (gradient-based attribution), attention fallback
            attention_heatmaps = medgemma.extract_concept_heatmaps_gradcam(pil_image, concepts)
            if attention_heatmaps:
                logger.info("PRIOR: GradCAM produced %d heatmaps", len(attention_heatmaps))
            else:
                logger.info("PRIOR: GradCAM unavailable, falling back to attention")
                attention_heatmaps = medgemma.extract_concept_heatmaps(pil_image, concepts)

            use_boxes = ATTENTION_PRIOR_MODE in ("box", "both")
            if use_boxes:
                for concept, hmap in attention_heatmaps.items():
                    box = heatmap_to_box(hmap, img_h, img_w)
                    if box is not None:
                        spatial_priors[concept] = box
                        logger.info("PRIOR: concept='%s' → box=%s", concept, box)
                    else:
                        logger.info("PRIOR: concept='%s' → no significant region", concept)

            mode_detail = f"mode={ATTENTION_PRIOR_MODE}"
            n_heatmaps = len(attention_heatmaps)
            n_boxes = len(spatial_priors)
            prior_step.status = "done"
            prior_step.detail = f"{mode_detail}: {n_heatmaps} heatmaps, {n_boxes} boxes"
            state.debug["PRIOR_mode"] = ATTENTION_PRIOR_MODE
            state.debug["PRIOR_heatmaps"] = list(attention_heatmaps.keys())
            if spatial_priors:
                state.debug["PRIOR_boxes"] = {c: list(b) for c, b in spatial_priors.items()}
            if attention_heatmaps:
                state.debug["_attention_heatmaps"] = attention_heatmaps
        except Exception as e:
            logger.warning("PRIOR step failed (continuing without priors): %s", e)
            prior_step.status = "done"
            prior_step.detail = f"Failed: {e}"
            spatial_priors = {}
            attention_heatmaps = {}

        _sync_segment_meta(state); yield state.chat, render_steps_html(state.steps), build_annotated_image(state), meas_json, state.debug

    # Clean up capture state
    medgemma.invalidate_cache()

    # ---- STEP SEG: MedSAM3 refined segmentation ----
    seg_step = Step(id="SEG", name="MedSAM3: refined segmentation", status="running", detail="Running combined pipeline…", segment_ids=[])
    state.steps.append(seg_step)
    _sync_segment_meta(state); yield state.chat, render_steps_html(state.steps), build_annotated_image(state), meas_json, state.debug

    medsam3 = get_medsam3()
    try:
        use_overlay = ATTENTION_PRIOR_MODE in ("overlay", "both") and attention_heatmaps
        use_box_only = ATTENTION_PRIOR_MODE == "box" and spatial_priors

        if use_overlay:
            auto_results = refined_segment_with_attention_overlay(
                img_np, concepts, medsam3, attention_heatmaps,
                spatial_priors=spatial_priors if ATTENTION_PRIOR_MODE == "both" else None,
            )
        elif use_box_only:
            auto_results = refined_segment_with_priors(
                img_np, concepts, medsam3, spatial_priors,
            )
        elif REFINED_SEG_ENABLED:
            auto_results = refined_segment(
                img_np,
                concepts,
                medsam3,
            )
        else:
            auto_results = medsam3.segment_concepts(
                img_np,
                concepts,
                threshold=MEDSAM3_CONCEPT_THRESHOLD,
                mask_threshold=MEDSAM3_CONCEPT_MASK_THRESHOLD,
                max_masks_per_concept=MEDSAM3_CONCEPT_MAX_MASKS_PER_CONCEPT,
                nms_iou_thresh=MEDSAM3_NMS_IOU_THRESH,
                max_total_masks=MEDSAM3_MAX_MASKS,
                preprocess=MEDSAM3_AUTO_PREPROCESS,
            )
    except Exception as e:
        seg_step.status = "failed"
        seg_step.detail = f"MedSAM3 segmentation failed: {e}"
        state.chat[-1] = {"role": "assistant", "content": f"Segmentation failed: {e}"}
        _sync_segment_meta(state); yield state.chat, render_steps_html(state.steps), build_annotated_image(state), meas_json, state.debug
        return

    if not auto_results:
        seg_step.status = "done"
        seg_step.detail = "No matching regions found."
        logger.info("SEG: no results, will use image-only fallback")

    # Store segments
    for result in auto_results:
        seg_id = state.next_segment_id()
        mask = result["mask"]
        bbox = result["bbox"]
        meas = measure_mask(mask, pixel_spacing=state.pixel_spacing)
        concept = result.get("concept")
        label = f"{concept} ({seg_id})" if concept else f"Region {seg_id}"
        state.add_segment(
            segment_id=seg_id,
            label=label,
            mask=mask,
            bbox=bbox,
            created_by_step="SEG",
            measurements=meas,
        )
        # Store original concept for validation step
        state.segments[seg_id]["concept"] = concept or ""
        seg_step.segment_ids.append(seg_id)
        meas_json[seg_id] = meas

    if auto_results:
        seg_step.status = "done"
        seg_step.detail = f"Found {len(auto_results)} region(s)."
        state.chat[-1] = {"role": "assistant", "content": f"Found {len(auto_results)} region(s). Filtering with MedGemma…"}
    else:
        seg_step.status = "done"
        seg_step.detail = "No matching regions found."
        state.chat[-1] = {"role": "assistant", "content": "No regions found. Will analyze image directly…"}
    state.debug["SEG_segment_count"] = len(auto_results)
    _sync_segment_meta(state); yield state.chat, render_steps_html(state.steps), build_annotated_image(state), meas_json, state.debug

    # ---- IDENTIFY + MATCH + ITERATIVE REFINEMENT LOOP ----
    r1_thinking = state.debug.get("R1_thinking", "")
    identified: dict[str, dict] = {}  # seg_id → {"name", "description", "concept"}
    matched_concepts: set[str] = set()
    matched_segment_ids: set[str] = set()

    # Helper: yield shorthand
    def _yield_state():
        return state.chat, render_steps_html(state.steps), build_annotated_image(state), meas_json, state.debug

    # Run identify (F steps) for a set of segments, returning identified dict.
    # This is a generator that yields state updates.
    def _run_identify_steps(seg_ids_to_identify: list[str]):
        newly_identified: dict[str, dict] = {}
        for seg_id in seg_ids_to_identify:
            seg = state.segments.get(seg_id)
            if seg is None:
                continue
            cidx = seg.get("color_idx", 0)
            seg_color = SEGMENT_COLORS[cidx % len(SEGMENT_COLORS)]
            seg_color_name = COLOR_NAMES[cidx % len(COLOR_NAMES)]
            concept = seg.get("concept", "")
            f_step = Step(
                id=f"F{seg_id}",
                name=f"MedGemma: identify Seg {seg_id}",
                status="running",
                detail="Identifying region…",
                segment_ids=[seg_id],
            )
            state.steps.append(f_step)
            state.chat.append({"role": "assistant", "content": f"Identifying segment {seg_id}…"})
            _sync_segment_meta(state); yield _yield_state()

            single_overlay = overlay_mask_on_image(
                img_np, seg["mask"], title=f"Segment {seg_id} ({seg_color_name})", color=seg_color,
            )

            filter_text = ""
            json_started = False
            try:
                for delta in medgemma.chat_stream(
                    user_content=filter_prompt,
                    images=[pil_image, single_overlay],
                    system_prompt=None,
                    max_new_tokens=350,
                    do_sample=False,
                ):
                    filter_text += delta
                    if not json_started:
                        display = re.sub(r"<unused\d+>(?:thought|end_of_thought)?\s*", "", filter_text).strip()
                        display = re.sub(r"```(?:json)?\s*", "", display).strip()
                        json_pos = display.find("{")
                        if json_pos >= 0:
                            json_started = True
                            reasoning_so_far = display[:json_pos].strip()
                        else:
                            reasoning_so_far = display
                        if reasoning_so_far:
                            state.chat[-1] = {"role": "assistant", "content": f"**F{seg_id}:** {reasoning_so_far}", "reasoning": True}
                            _sync_segment_meta(state); yield _yield_state()
            except Exception as e:
                logger.warning("Identify failed for segment %s: %s", seg_id, e)
                f_step.status = "done"
                f_step.detail = f"Identify error: {e}"
                _sync_segment_meta(state); yield _yield_state()
                continue

            state.debug[f"filter_F{seg_id}"] = filter_text
            thinking_f, cleaned_f = _extract_thinking(filter_text)
            classification = _parse_classify_json(cleaned_f)
            logger.info("F%s: identify result for seg %s (concept=%s): %s", seg_id, seg_id, concept, classification)

            if classification:
                validated_name = classification.get("name", "")
                validated_desc = classification.get("description", "")
                seg["validated_name"] = validated_name
                seg["validated_desc"] = validated_desc
                newly_identified[seg_id] = {"name": validated_name, "description": validated_desc, "concept": concept}
                f_step.status = "done"
                f_step.detail = f"Identified: {validated_name}"
            else:
                f_step.status = "done"
                f_step.detail = "Identification unparseable"

            if thinking_f:
                state.chat[-1] = {"role": "assistant", "content": f"**F{seg_id}:** {thinking_f}", "reasoning": True}
            else:
                state.chat[-1] = {"role": "assistant", "content": f"**F{seg_id}** `{filter_text.strip()}`", "reasoning": True}
            _sync_segment_meta(state); yield _yield_state()

        return newly_identified

    # --- Initial identify pass ---
    initial_seg_ids = list(state.segments.keys())
    identify_gen = _run_identify_steps(initial_seg_ids)
    # Consume generator, collecting newly_identified via return value
    try:
        while True:
            yield next(identify_gen)
    except StopIteration as e:
        new_ids = e.value or {}
    identified.update(new_ids)

    # --- Iterative match + re-segment loop ---
    max_iter = MAX_REFINEMENT_ITERATIONS if ITERATIVE_REFINEMENT_ENABLED else 0

    for iteration in range(1 + max_iter):  # iteration 0 = initial match, 1..max_iter = refinement rounds
        match_n = iteration + 1  # M1, M2, M3
        unmatched_concepts = [c for c in concepts if c.lower() not in {mc.lower() for mc in matched_concepts}]

        # Build pairs for all identified segments whose concept is still unmatched
        current_pairs = {
            sid: info for sid, info in identified.items()
            if info.get("concept", "").lower() in {c.lower() for c in unmatched_concepts}
        }
        # Also include already-matched pairs for context
        matched_pairs = {
            sid: info for sid, info in identified.items()
            if sid in matched_segment_ids
        }
        all_pairs = {**matched_pairs, **current_pairs}

        if not current_pairs:
            # All concepts already matched or nothing to evaluate
            break

        # ---- MATCH STEP M{n} ----
        match_step = Step(
            id=f"M{match_n}", name=f"MedGemma: concept match (round {match_n})", status="running",
            detail="Matching concepts to segments…", segment_ids=list(current_pairs.keys()),
        )
        state.steps.append(match_step)
        state.chat.append({"role": "assistant", "content": f"Judging concept matches (round {match_n})…"})
        _sync_segment_meta(state); yield _yield_state()

        pairs_text = "\n".join(
            f"- Segment {sid}: expected \"{info['concept']}\" → found \"{info['name']}\" ({info['description']})"
            for sid, info in all_pairs.items()
        )
        concepts_list_text = ", ".join(unmatched_concepts)
        match_user_content = match_prompt.format(
            r1_thinking=r1_thinking or "(No explicit reasoning available)",
            concepts_list=concepts_list_text,
            pairs=pairs_text,
        )

        match_grammar = _get_grammar("match")
        match_lp = medgemma.json_logits_processor(match_grammar) if match_grammar else None

        match_text = ""
        match_error = False
        try:
            for delta in medgemma.chat_stream(
                user_content=match_user_content,
                images=None,
                system_prompt=None,
                max_new_tokens=350,
                do_sample=False,
                logits_processor=match_lp,
            ):
                match_text += delta
        except Exception as e:
            logger.warning("Match step M%d failed: %s", match_n, e)
            match_step.status = "done"
            match_step.detail = f"Match error: {e}"
            match_error = True
            _sync_segment_meta(state); yield _yield_state()

        state.debug[f"match_M{match_n}"] = match_text
        match_result = _parse_concept_match_json(match_text) if not match_error else None
        logger.info("M%d: match result: %s", match_n, match_result)

        if match_result:
            newly_matched = set(match_result.get("matched", []))
            not_matched_concepts = set(match_result.get("not_matched", []))
            matched_concepts |= newly_matched

            # Track which segments are confirmed matched
            for sid, info in identified.items():
                concept_lower = info.get("concept", "").lower()
                if any(concept_lower == mc.lower() for mc in newly_matched):
                    matched_segment_ids.add(sid)

            if not match_error:
                if not_matched_concepts:
                    match_step.status = "done"
                    match_step.detail = f"Matched: {', '.join(newly_matched)}. Unmatched: {', '.join(not_matched_concepts)}"
                    state.chat[-1] = {"role": "assistant", "content": f"Matched {len(newly_matched)} concept(s). {len(not_matched_concepts)} unmatched.", "reasoning": True}
                else:
                    match_step.status = "done"
                    match_step.detail = f"All concepts matched: {', '.join(newly_matched)}"
                    state.chat[-1] = {"role": "assistant", "content": "All concepts matched.", "reasoning": True}

                # Immediately remove segments for not_matched concepts
                # so UI reflects the change before re-segmentation starts
                not_matched_lower = {c.lower() for c in not_matched_concepts}
                for sid in list(state.segments.keys()):
                    if sid not in matched_segment_ids:
                        info = identified.get(sid, {})
                        seg_concept = info.get("concept", "")
                        if seg_concept and seg_concept.lower() in not_matched_lower:
                            logger.info("M%d: removing unmatched segment %s (concept=%s)", match_n, sid, seg_concept)
                            del state.segments[sid]
                            meas_json.pop(sid, None)
                            for st in state.steps:
                                if st.id == f"F{sid}":
                                    found_name = info.get("name", "")
                                    st.detail = f"Rejected: {found_name} (unmatched concept)"
                                    break

                _sync_segment_meta(state); yield _yield_state()
        else:
            # Parse failure or error: assume all matched (conservative)
            if not match_error:
                match_step.status = "done"
                match_step.detail = "Match unparseable, assuming all matched."
                state.chat[-1] = {"role": "assistant", "content": "Match unparseable, assuming all matched.", "reasoning": True}
                _sync_segment_meta(state); yield _yield_state()
            matched_concepts = set(concepts)
            break

        # Check if re-segmentation needed
        still_unmatched = [c for c in concepts if c.lower() not in {mc.lower() for mc in matched_concepts}]

        if not still_unmatched:
            break  # All concepts matched

        if iteration >= max_iter:
            break  # Max iterations reached

        # ---- RE-SEGMENT STEP (SEG{n+1}) ----
        seg_n = iteration + 2  # SEG2, SEG3

        # Collect negative boxes from matched segments
        negative_boxes = []
        for sid in matched_segment_ids:
            seg = state.segments.get(sid)
            if seg:
                negative_boxes.append(seg["bbox"])
        # Also add bboxes from unmatched segments (they're in the wrong place)
        for sid in list(state.segments.keys()):
            if sid not in matched_segment_ids:
                seg = state.segments.get(sid)
                if seg and seg["bbox"] not in negative_boxes:
                    negative_boxes.append(seg["bbox"])

        if not negative_boxes:
            logger.info("No negative boxes available, skipping re-segmentation")
            break

        reseg_step = Step(
            id=f"SEG{seg_n}",
            name=f"MedSAM3: re-segment (round {seg_n})",
            status="running",
            detail=f"Re-segmenting: {', '.join(still_unmatched)} (avoiding {len(negative_boxes)} region(s))",
            segment_ids=[],
        )
        state.steps.append(reseg_step)
        state.chat.append({"role": "assistant", "content": f"Re-segmenting {len(still_unmatched)} unmatched concept(s) with negative constraints…"})
        _sync_segment_meta(state); yield _yield_state()

        try:
            new_results = refined_segment_with_negatives(
                img_np, still_unmatched, medsam3, negative_boxes=negative_boxes,
            )
        except Exception as e:
            logger.warning("Re-segmentation SEG%d failed: %s", seg_n, e)
            reseg_step.status = "done"
            reseg_step.detail = f"Re-segmentation failed: {e}"
            _sync_segment_meta(state); yield _yield_state()
            break

        if not new_results:
            reseg_step.status = "done"
            reseg_step.detail = "No new regions found."
            _sync_segment_meta(state); yield _yield_state()
            break

        # Store new segments
        new_seg_ids = []
        for result in new_results:
            seg_id = state.next_segment_id()
            mask = result["mask"]
            bbox = result["bbox"]
            meas = measure_mask(mask, pixel_spacing=state.pixel_spacing)
            concept = result.get("concept")
            label = f"{concept} ({seg_id})" if concept else f"Region {seg_id}"
            state.add_segment(
                segment_id=seg_id, label=label, mask=mask, bbox=bbox,
                created_by_step=f"SEG{seg_n}", measurements=meas,
            )
            state.segments[seg_id]["concept"] = concept or ""
            reseg_step.segment_ids.append(seg_id)
            meas_json[seg_id] = meas
            new_seg_ids.append(seg_id)

        reseg_step.status = "done"
        reseg_step.detail = f"Found {len(new_results)} new region(s)."
        state.debug[f"SEG{seg_n}_segment_count"] = len(new_results)
        _sync_segment_meta(state); yield _yield_state()

        # ---- IDENTIFY NEW SEGMENTS ----
        identify_gen = _run_identify_steps(new_seg_ids)
        try:
            while True:
                yield next(identify_gen)
        except StopIteration as e:
            new_ids = e.value or {}
        identified.update(new_ids)
        # Loop continues to next match round

    state.debug["refinement_iterations"] = len([k for k in state.debug if k.startswith("match_M")])
    state.debug["final_matched_concepts"] = list(matched_concepts)

    # ---- CLEANUP: Remove any remaining segments not confirmed as matched ----
    # Most unmatched segments are already removed after each M step, but this
    # catches edge cases (e.g., old segments for concepts matched in a later round
    # by a different segment).
    for sid in list(state.segments.keys()):
        if sid not in matched_segment_ids:
            info = identified.get(sid, {})
            concept = info.get("concept", "")
            logger.info("Cleanup: removing non-matched segment %s (concept=%s)", sid, concept)
            del state.segments[sid]
            meas_json.pop(sid, None)
            for st in state.steps:
                if st.id == f"F{sid}":
                    found_name = info.get("name", "")
                    st.detail = f"Rejected: {found_name} (unmatched concept)"
                    break

    _sync_segment_meta(state); yield _yield_state()

    # ---- Emit concept links for heatmap hover (same as parallel flow) ----
    from backend.image_service import segment_color_hex

    evidence_concept_links = []
    for sid, seg in state.segments.items():
        info = identified.get(sid, {})
        seg_concept = info.get("concept", seg.get("concept", ""))
        color = segment_color_hex(seg.get("color_idx", 0))
        evidence_concept_links.append({
            "concept": seg_concept,
            "segment_id": sid,
            "aliases": [],
            "color": color,
        })
    if evidence_concept_links:
        state.debug["_concept_links"] = evidence_concept_links
        logger.info("Evidence concept links: %s", evidence_concept_links)
        _sync_segment_meta(state); yield _yield_state()

    state.debug["classified_count"] = len(state.segments)

    if not state.segments:
        # ---- FALLBACK: Image-only analysis (no useful masks) ----
        state.debug["fallback_image_only"] = True
        logger.info("No segments after classification, falling back to image-only analysis")

        r2_step = Step(id="R2", name="MedGemma: image-only analysis", status="running", detail="Streaming…", segment_ids=[])
        state.steps.append(r2_step)
        state.chat[-1] = {"role": "assistant", "content": "No relevant segments found. Analyzing image directly…"}
        _sync_segment_meta(state); yield state.chat, render_steps_html(state.steps), build_annotated_image(state), meas_json, state.debug

        fallback_content = (
            f"{user_prompt}\n\n"
            "NOTE: Automated segmentation did not find relevant regions for this question. "
            "Please analyze the image directly and answer the question based on what you observe."
        )

        assistant_text = ""
        try:
            for delta in medgemma.chat_stream(
                user_content=fallback_content,
                images=[pil_image],
                system_prompt=analysis_prompt,
                max_new_tokens=1200,
                do_sample=False,
                repetition_penalty=1.3,
            ):
                assistant_text += delta
                cleaned, should_stop = _clean_r2_stream(assistant_text)
                state.chat[-1] = {"role": "assistant", "content": cleaned}
                _sync_segment_meta(state); yield state.chat, render_steps_html(state.steps), build_annotated_image(state), meas_json, state.debug
                if should_stop:
                    logger.info("R2 (fallback): degeneration detected, stopping stream")
                    break

            r2_step.status = "done"
            r2_step.detail = "Image-only analysis (no segments)."
        except Exception as e:
            r2_step.status = "failed"
            r2_step.detail = f"Analysis failed: {e}"
            state.chat[-1] = {"role": "assistant", "content": f"Analysis failed: {e}"}

        assistant_text, _ = _clean_r2_stream(assistant_text)
        state.debug["R2_raw"] = assistant_text
        _sync_segment_meta(state); yield state.chat, render_steps_html(state.steps), build_annotated_image(state), meas_json, state.debug

        job_dt = time.perf_counter() - job_t0
        logger.info("run_job DONE (fallback) job=%s total=%.1fs", state.job_id[:8], job_dt)
        return

    seg_count_before = state.debug.get("SEG_segment_count", 0)
    rejected = seg_count_before - len(state.segments)
    if rejected > 0:
        state.chat.append({"role": "assistant", "content": f"Kept {len(state.segments)} region(s), rejected {rejected}. Analyzing…"})
    else:
        state.chat.append({"role": "assistant", "content": f"Kept {len(state.segments)} region(s). Analyzing…"})
    _sync_segment_meta(state); yield state.chat, render_steps_html(state.steps), build_annotated_image(state), meas_json, state.debug

    # ---- STEP R2: MedGemma analyzes with evidence ----
    r2_step = Step(id="R2", name="MedGemma: grounded analysis", status="running", detail="Streaming…", segment_ids=[])
    state.steps.append(r2_step)
    state.chat[-1] = {"role": "assistant", "content": "Analyzing with segmentation evidence…"}
    _sync_segment_meta(state); yield state.chat, render_steps_html(state.steps), build_annotated_image(state), meas_json, state.debug

    # Build multi-color overlay + evidence using stable color_idx
    mask_label_pairs = [
        (seg["mask"], f"Seg {seg_id} ({COLOR_NAMES[seg.get('color_idx', 0) % len(COLOR_NAMES)]})")
        for seg_id, seg in state.segments.items()
    ]
    multi_overlay = overlay_multiple_masks(img_np, mask_label_pairs, color_indices=[
        seg.get("color_idx", 0) for seg in state.segments.values()
    ])

    evidence_lines = []
    for seg_id, seg in state.segments.items():
        color_name = COLOR_NAMES[seg.get("color_idx", 0) % len(COLOR_NAMES)]
        concept = seg.get("concept", "")
        validated_desc = seg.get("validated_desc", "")
        evidence_lines.append(
            build_tool_result_message(
                segment_id=seg_id,
                label=concept or seg["label"],
                color_name=color_name,
                description=validated_desc,
            )
        )

    evidence_text = "\n".join(evidence_lines)
    analysis_user_content = f"{user_prompt}\n\nSegmented regions:\n{evidence_text}"

    # Optional baseline comparison
    baseline_text = ""
    if compare_baseline:
        try:
            for delta in medgemma.chat_stream(
                user_content="Answer the user's question about the image. Do not mention segmentation.\n\nUser: "
                + user_prompt,
                images=[pil_image],
                system_prompt=None,
                max_new_tokens=800,
                do_sample=False,
                repetition_penalty=1.3,
            ):
                baseline_text += delta
        except Exception as e:
            baseline_text = f"(Baseline failed: {e})"

    # Stream the grounded analysis
    assistant_text = ""
    try:
        for delta in medgemma.chat_stream(
            user_content=analysis_user_content,
            images=[pil_image, multi_overlay],
            system_prompt=analysis_prompt,
            max_new_tokens=1200,
            do_sample=False,
            repetition_penalty=1.3,
        ):
            assistant_text += delta
            cleaned, should_stop = _clean_r2_stream(assistant_text)
            state.chat[-1] = {
                "role": "assistant",
                "content": _format_assistant(
                    cleaned, baseline_text, compare_baseline, list(state.segments.keys())
                ),
            }
            _sync_segment_meta(state); yield state.chat, render_steps_html(state.steps), build_annotated_image(state), meas_json, state.debug
            if should_stop:
                logger.info("R2: degeneration detected, stopping stream")
                break

        r2_step.status = "done"
        r2_step.detail = f"Grounded in {len(state.segments)} segment(s)."
    except Exception as e:
        r2_step.status = "failed"
        r2_step.detail = f"Analysis failed: {e}"
        state.chat[-1] = {"role": "assistant", "content": f"Analysis failed: {e}"}

    # Clean final text (strip any remaining thinking tokens)
    assistant_text, _ = _clean_r2_stream(assistant_text)
    state.debug["R2_raw"] = assistant_text
    _sync_segment_meta(state); yield state.chat, render_steps_html(state.steps), build_annotated_image(state), meas_json, state.debug

    job_dt = time.perf_counter() - job_t0
    logger.info(
        "run_job DONE job=%s total=%.1fs segments=%d steps=%d",
        state.job_id[:8],
        job_dt,
        len(state.segments),
        len(state.steps),
    )


def _format_assistant(grounded: str, baseline: str, show_baseline: bool, segments: list[str]) -> str:
    """Format assistant response text.

    Note: The React frontend has its own chip parser (segmentChipParser.ts)
    that converts [SEG:A] patterns to interactive SegmentChip components.
    We no longer wrap chips in HTML spans here.
    """
    if not show_baseline:
        return grounded

    return f"## Grounded (Segment→Measure→Explain)\n{grounded}\n\n---\n## Baseline (MedGemma only)\n{baseline}\n"


_cxr_vocab_cache: list[str] | None = None


def _load_cxr_vocabulary() -> list[str]:
    """Load and cache the CXR concept vocabulary from file."""
    global _cxr_vocab_cache
    if _cxr_vocab_cache is not None:
        return _cxr_vocab_cache
    vocab_path = os.path.join(BASE_DIR, "prompts", "cxr_concepts.txt")
    with open(vocab_path, encoding="utf-8") as f:
        _cxr_vocab_cache = [
            line.strip() for line in f
            if line.strip() and not line.strip().startswith("#")
        ]
    return _cxr_vocab_cache


def _normalize_term_to_vocabulary(term: str, vocab: list[str]) -> str:
    """Map a free-text concept term to the closest CXR vocabulary entry.

    Matching priority:
    1. Exact match (case-insensitive)
    2. Simple plural/singular normalization
    3. Longest vocab entry contained in the term
    4. Shortest vocab entry containing the term
    5. Word-level Jaccard overlap > 0.5
    Falls back to original term if no match found.
    """
    term_lower = term.lower().strip()
    if not term_lower:
        return term
    vocab_map = {v.lower(): v for v in vocab}

    # 1. Exact match
    if term_lower in vocab_map:
        return vocab_map[term_lower]

    # 2. Singular/plural normalization
    if term_lower.endswith("s") and term_lower[:-1] in vocab_map:
        return vocab_map[term_lower[:-1]]
    if (term_lower + "s") in vocab_map:
        return vocab_map[term_lower + "s"]

    # 3. Longest vocab entry contained in term
    best_contained = ""
    for vl, vo in vocab_map.items():
        if vl in term_lower and len(vl) > len(best_contained):
            best_contained = vo
    if best_contained:
        return best_contained

    # 4. Shortest vocab entry containing the term
    shortest_containing: str | None = None
    for vl, vo in vocab_map.items():
        if term_lower in vl:
            if shortest_containing is None or len(vl) < len(shortest_containing):
                shortest_containing = vo
    if shortest_containing:
        return shortest_containing

    # 5. Word-level Jaccard overlap
    term_words = set(term_lower.split())
    best_overlap = 0.0
    best_match: str | None = None
    for vl, vo in vocab_map.items():
        vocab_words = set(vl.split())
        intersection = len(term_words & vocab_words)
        union = len(term_words | vocab_words)
        if union > 0:
            jaccard = intersection / union
            if jaccard > 0.5 and jaccard > best_overlap:
                best_overlap = jaccard
                best_match = vo
    if best_match:
        return best_match

    return term


def _normalize_concept_entries(entries: list[dict]) -> list[dict]:
    """Normalize concept entry terms to CXR vocabulary.

    For each entry, maps the "term" to the closest vocabulary match.
    If the term changes, the original term is added to aliases so the
    frontend can still highlight it in the answer text.
    """
    vocab = _load_cxr_vocabulary()
    normalized = []
    for entry in entries:
        original_term = entry.get("term", "")
        vocab_term = _normalize_term_to_vocabulary(original_term, vocab)
        aliases = list(entry.get("aliases", []))
        if vocab_term.lower() != original_term.lower() and original_term not in aliases:
            aliases.append(original_term)
        normalized.append({"term": vocab_term, "aliases": aliases})
    return normalized


def _alias_in_text(alias: str, text: str) -> bool:
    """Check if alias appears in text with word boundaries."""
    try:
        return bool(re.search(
            r"\b" + re.escape(alias.lower()) + r"\b",
            text.lower(),
        ))
    except re.error:
        return alias.lower() in text.lower()


def _validate_concept_aliases(
    concepts: list[dict], answer_text: str
) -> list[dict]:
    """Filter concepts: keep only those whose term or an alias appears in the answer.

    Uses word-boundary matching and minimum alias length (3 chars).
    """
    valid = []
    for c in concepts:
        term = c.get("term", "")
        aliases = c.get("aliases", [])
        # Check if term or any alias appears in answer text (word-boundary)
        found = _alias_in_text(term, answer_text)
        if not found:
            for alias in aliases:
                if len(alias.strip()) >= 3 and _alias_in_text(alias, answer_text):
                    found = True
                    break
        if found:
            # Filter aliases: must be in text (word-boundary) and >= 3 chars
            valid_aliases = [
                a for a in aliases
                if len(a.strip()) >= 3 and _alias_in_text(a, answer_text)
            ]
            valid.append({"term": term, "aliases": valid_aliases})
    return valid


# -----------------------------
# Deterministic alias extraction for parallel pipeline
# -----------------------------

_ALIAS_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
    "of", "to", "and", "or", "no", "not", "this", "that", "with",
    "for", "from", "by", "as", "it", "its", "be", "has", "have",
    "been", "there", "which", "also", "may", "can", "does", "do",
})


def _strip_markdown(text: str) -> str:
    """Remove markdown bold/italic markers from text."""
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # **bold**
    text = re.sub(r"__([^_]+)__", r"\1", text)  # __bold__
    text = re.sub(r"\*([^*]+)\*", r"\1", text)  # *italic*
    text = re.sub(r"_([^_]+)_", r"\1", text)  # _italic_
    return text


def _extract_contextual_aliases(
    concept_term: str,
    answer_text: str,
    other_segment_concepts: set[str],
) -> list[str]:
    """Extract multi-word phrases from answer_text containing root words of concept_term.

    For concept "heart", finds phrases like "heart borders", "heart shadow" in the text.
    Skips phrases that exactly match the concept or another segment's concept.
    """
    roots = [w for w in concept_term.lower().split()
             if len(w) >= 4 and w not in _ALIAS_STOPWORDS]
    if not roots:
        return []

    # Strip markdown formatting before extraction
    clean_text = _strip_markdown(answer_text)
    sentences = re.split(r"[.!?;\n]+", clean_text)
    aliases: set[str] = set()
    concept_lower = concept_term.lower()
    other_lower = {c.lower() for c in other_segment_concepts}
    clean_lower = clean_text.lower()

    for sentence in sentences:
        words = sentence.split()
        for root in roots:
            for i, word in enumerate(words):
                if root not in word.lower():
                    continue
                # Phrase window: up to 2 words before, up to 2 after
                start = max(0, i - 2)
                end = min(len(words), i + 3)
                for s in range(start, i + 1):
                    for e in range(i + 1, end + 1):
                        phrase = " ".join(words[s:e]).strip(" ,;:*()[]")
                        if len(phrase) < 3:
                            continue
                        phrase_lower = phrase.lower()
                        if phrase_lower == concept_lower:
                            continue
                        if phrase_lower in other_lower:
                            continue
                        # Skip if phrase contains another segment concept
                        if any(oc in phrase_lower for oc in other_lower if len(oc) >= 4):
                            continue
                        phrase_words = phrase_lower.split()
                        if all(w in _ALIAS_STOPWORDS or len(w) < 3 for w in phrase_words):
                            continue
                        # Skip phrases longer than 5 words or containing commas (list fragments)
                        if len(phrase_words) > 5 or "," in phrase:
                            continue
                        if phrase_lower in clean_lower:
                            aliases.add(phrase)

    # Deduplicate: remove substrings (keep longer phrases)
    sorted_aliases = sorted(aliases, key=len, reverse=True)
    final: list[str] = []
    for a in sorted_aliases:
        if not any(a.lower() in kept.lower() for kept in final):
            final.append(a)
    return final


def _cross_reference_prescan_aliases(
    prescan_terms: list[str],
    segment_concepts: dict[str, str],
    answer_text: str,
) -> dict[str, list[str]]:
    """Find prescan terms not used as segment concepts and assign as aliases.

    If prescan found "opacity" but no segment has concept "opacity",
    check if "opacity" is related to an existing segment concept
    (e.g., "lung opacity" contains "opacity") and add it as alias.
    """
    answer_lower = answer_text.lower()
    used_concepts = {c.lower() for c in segment_concepts.values() if c}
    unused = [t for t in prescan_terms if t.lower() not in used_concepts]

    extra_aliases: dict[str, list[str]] = {}

    for term in unused:
        if not _alias_in_text(term, answer_text):
            continue
        term_lower = term.lower()
        best_concept: str | None = None
        best_score = 0.0
        for seg_concept in set(segment_concepts.values()):
            sc_lower = seg_concept.lower()
            # Containment: "opacity" in "lung opacity" or vice versa
            if term_lower in sc_lower or sc_lower in term_lower:
                score = len(set(term_lower.split()) & set(sc_lower.split())) + 1.0
                if score > best_score:
                    best_score = score
                    best_concept = seg_concept
                continue
            # Word overlap
            t_words = set(term_lower.split())
            s_words = set(sc_lower.split())
            overlap = t_words & s_words
            if overlap:
                score = len(overlap) / len(t_words | s_words)
                if score > 0.3 and score > best_score:
                    best_score = score
                    best_concept = seg_concept

        if best_concept:
            extra_aliases.setdefault(best_concept, []).append(term)

    return extra_aliases


# -----------------------------
# Vocabulary pre-scan for parallel pipeline
# -----------------------------

_PRESCAN_VOCAB_CACHE: list[tuple[re.Pattern, str]] | None = None


def _prescan_vocab_terms(text: str) -> list[str]:
    """CPU-scan text for CXR vocabulary term matches.

    Called during ANSWER streaming to identify concept terms without MedGemma.
    Sorted by length descending so "pleural effusion" matches before "effusion".
    """
    global _PRESCAN_VOCAB_CACHE
    if _PRESCAN_VOCAB_CACHE is None:
        vocab = _load_cxr_vocabulary()
        sorted_vocab = sorted(vocab, key=len, reverse=True)
        _PRESCAN_VOCAB_CACHE = [
            (re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE), term)
            for term in sorted_vocab
        ]

    matched: list[str] = []
    seen: set[str] = set()
    for pattern, term in _PRESCAN_VOCAB_CACHE:
        if term.lower() not in seen and pattern.search(text):
            matched.append(term)
            seen.add(term.lower())
    return matched


# -----------------------------
# Concept selection helpers for parallel pipeline
# -----------------------------


def _build_select_prompt() -> str:
    """Build the SELECT step prompt with the CXR vocabulary list."""
    prompt_path = os.path.join(BASE_DIR, "prompts", "concept_select_prompt.txt")
    with open(prompt_path, encoding="utf-8") as f:
        template = f.read()
    vocab = _load_cxr_vocabulary()
    return template.format(cxr_concepts=", ".join(vocab))


def _parse_select_json(text: str) -> list[str]:
    """Parse concept selection response: {"concepts": ["term1", ...]}."""
    if not text:
        return []
    cleaned = _strip_medgemma_thinking(text)
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned).strip()
    # Try complete JSON first
    for match in re.finditer(r"\{.*\}", cleaned, re.DOTALL):
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict) and "concepts" in obj:
                arr = obj["concepts"]
                if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
                    return arr
        except json.JSONDecodeError:
            continue
    # Fallback: extract quoted strings from truncated JSON
    strings = re.findall(r'"([^"]{2,})"', cleaned)
    if strings:
        # Filter out JSON keys like "concepts", "type", etc.
        concepts = [s for s in strings if s.lower() not in ("concepts", "type", "object", "array", "string")]
        if concepts:
            logger.info("_parse_select_json: salvaged %d concepts from truncated JSON", len(concepts))
            return concepts
    return []


_NEGATION_PREFIX_RE = re.compile(
    r"\b(?:no|not|without|denied|denies|negative|rule[ds]?\s+out|"
    r"no\s+evidence\s+of|no\s+signs?\s+of|no\s+obvious)\b",
    re.IGNORECASE,
)
_NEGATION_SUFFIX_RE = re.compile(
    r"\b(?:absent|unlikely|unremarkable|ruled\s+out|not\s+seen|"
    r"not\s+identified|not\s+detected|not\s+observed)\b",
    re.IGNORECASE,
)


def _is_negated(concept: str, text: str) -> bool:
    """Check if a concept is mentioned only in a negated context.

    Looks for negation words in a window before AND after each occurrence.
    Only returns True if ALL occurrences are negated.
    """
    concept_lower = concept.lower()
    text_lower = text.lower()
    matches = list(re.finditer(re.escape(concept_lower), text_lower))
    if not matches:
        return False
    for m in matches:
        # Get the sentence containing this match (split on periods)
        sent_start = text_lower.rfind(".", 0, m.start())
        sent_start = sent_start + 1 if sent_start >= 0 else 0
        sent_end = text_lower.find(".", m.end())
        sent_end = sent_end if sent_end >= 0 else len(text_lower)
        prefix = text_lower[sent_start : m.start()]
        suffix = text_lower[m.end() : sent_end]
        has_negation = (
            _NEGATION_PREFIX_RE.search(prefix) or _NEGATION_SUFFIX_RE.search(suffix)
        )
        if not has_negation:
            return False
    return True


def _fuzzy_validate_concepts(concepts: list[str], answer_text: str) -> list[str]:
    """Validate selected concepts against answer text with fuzzy matching.

    Ensures each concept was actually discussed in the answer (word-boundary check).
    For multi-word concepts, requires ALL significant words (len>=4) to appear.
    Rejects concepts that only appear in negated context.
    """
    answer_lower = answer_text.lower()
    validated = []
    for concept in concepts:
        concept_lower = concept.lower()
        # Exact match (full concept phrase)
        pattern = re.compile(r"\b" + re.escape(concept_lower) + r"\b", re.IGNORECASE)
        if pattern.search(answer_lower):
            # Check negation
            if _is_negated(concept_lower, answer_lower):
                logger.debug("SELECT: rejected negated concept: %s", concept)
                continue
            validated.append(concept)
            continue
        # Fuzzy: for multi-word concepts, require ALL significant words present
        words = concept_lower.split()
        if len(words) >= 2:
            significant = [w for w in words if len(w) >= 4]
            if significant and all(
                re.search(r"\b" + re.escape(w) + r"\b", answer_lower)
                for w in significant
            ):
                # Check negation using the full concept phrase (approximate)
                if _is_negated(concept_lower, answer_lower):
                    logger.debug("SELECT: rejected negated concept (fuzzy): %s", concept)
                    continue
                validated.append(concept)
    return validated


def _dedup_concepts(concepts: list[str]) -> list[str]:
    """Deduplicate concepts, preferring more specific terms.

    "left lung" subsumes "lung" (if both present, keep specific).
    Also removes exact duplicates.
    """
    result: list[str] = []
    seen_lower: set[str] = set()
    for c in concepts:
        c_lower = c.lower()
        # Skip exact duplicates
        if c_lower in seen_lower:
            continue
        # Skip if a more specific concept already includes this one
        if any(c_lower in r.lower() and c_lower != r.lower() for r in result):
            continue
        # Remove less specific concepts already in result
        result = [r for r in result if not (r.lower() in c_lower and r.lower() != c_lower)]
        seen_lower = {r.lower() for r in result}
        result.append(c)
        seen_lower.add(c_lower)
    return result


# -----------------------------
# Parallel pipeline: ANSWER → SELECT → SEG → LINK
# -----------------------------


def run_parallel_job(
    image: Any,
    user_prompt: str,
    state: JobState | None = None,
) -> Generator[tuple[list[dict], str, Any, dict[str, Any], dict[str, Any]], None, None]:
    """Parallel pipeline: stream answer, select concepts (xgrammar enum), segment, link.

    Steps: S1 → ANSWER (streaming + KV cache) → SELECT (KV cache + enum grammar)
           → SEG (MedSAM3 on selected concepts) → LINK (deterministic aliases)

    Yields: (chat, steps_html, annotated_image, measurements_json, debug_json)
    """
    if state is None:
        state = create_job_state()

    job_t0 = time.perf_counter()

    # Validate
    if image is None:
        logger.warning("run_parallel_job: no image provided")
        state.chat.append({"role": "user", "content": user_prompt or ""})
        state.chat.append({"role": "assistant", "content": "Please upload an image first."})
        yield state.chat, render_steps_html(state.steps), None, {}, {"error": "no image"}
        return

    # Convert image to numpy RGB uint8
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        img_np = image
    else:
        img_np = np.array(image)
    img_np = ensure_rgb_uint8(img_np)

    state.image = img_np
    state.job_id = state.job_id or str(uuid.uuid4())
    state.debug = {}
    img_h, img_w = img_np.shape[:2]

    logger.info(
        "run_parallel_job START job=%s image=%dx%d prompt='%s'",
        state.job_id[:8], img_w, img_h, user_prompt[:80],
    )

    # Add user message + empty assistant placeholder (avoids stale text in chat)
    state.chat.append({"role": "user", "content": user_prompt})
    state.chat.append({"role": "assistant", "content": ""})

    pil_image = Image.fromarray(img_np)
    meas_json: dict[str, Any] = {}

    # ---- STEP S1: Parse request ----
    state.steps = [
        Step(id="S1", name="Parse request", status="done", detail="Image received.", segment_ids=[]),
    ]
    state.debug["segmentation_mode"] = "parallel"
    _sync_segment_meta(state)
    yield state.chat, render_steps_html(state.steps), build_annotated_image(state), meas_json, state.debug

    # ---- STEP ANSWER: MedGemma streams answer with KV cache ----
    answer_step = Step(id="ANSWER", name="MedGemma: answer", status="running", detail="Streaming…", segment_ids=[])
    state.steps.append(answer_step)
    _sync_segment_meta(state)
    yield state.chat, render_steps_html(state.steps), build_annotated_image(state), meas_json, state.debug

    # Load answer prompt
    with open(os.path.join(BASE_DIR, "prompts", "answer_prompt.txt"), encoding="utf-8") as f:
        answer_prompt = f.read()

    medgemma = get_medgemma()

    answer_text = ""
    prescan_terms: list[str] = []
    try:
        for delta in medgemma.chat_stream_with_cache(
            user_content=user_prompt,
            images=[pil_image],
            system_prompt=answer_prompt,
            max_new_tokens=1200,
            do_sample=False,
            repetition_penalty=1.3,
            capture_attention=False,  # GradCAM does its own forward pass
        ):
            answer_text += delta
            cleaned, should_stop = _clean_r2_stream(answer_text)
            state.chat[-1] = {"role": "assistant", "content": cleaned}

            # CPU vocab pre-scan during streaming (cosmetic — for step detail display only)
            prescan_terms = _prescan_vocab_terms(cleaned)
            if prescan_terms:
                answer_step.detail = f"Streaming… Found: {', '.join(prescan_terms[:5])}"

            _sync_segment_meta(state)
            yield state.chat, render_steps_html(state.steps), build_annotated_image(state), meas_json, state.debug
            if should_stop:
                logger.info("ANSWER: degeneration detected, stopping stream")
                break

        answer_step.status = "done"
        answer_step.detail = "Answer complete."
    except Exception as e:
        answer_step.status = "failed"
        answer_step.detail = f"Answer failed: {e}"
        state.chat[-1] = {"role": "assistant", "content": f"Answer failed: {e}"}
        _sync_segment_meta(state)
        yield state.chat, render_steps_html(state.steps), build_annotated_image(state), meas_json, state.debug
        return

    answer_text, _ = _clean_r2_stream(answer_text)
    state.debug["ANSWER_raw"] = answer_text

    # Final prescan for debug info only
    prescan_terms = _prescan_vocab_terms(answer_text)
    state.debug["_prescan_terms"] = prescan_terms
    logger.info("ANSWER complete, pre-scanned vocab terms: %s", prescan_terms)

    _sync_segment_meta(state)
    yield state.chat, render_steps_html(state.steps), build_annotated_image(state), meas_json, state.debug

    # ---- STEP SELECT: MedGemma concept selection (KV cache + xgrammar enum) ----
    select_step = Step(id="SELECT", name="Select concepts", status="running", detail="Selecting concepts…", segment_ids=[])
    state.steps.append(select_step)
    _sync_segment_meta(state)
    yield state.chat, render_steps_html(state.steps), build_annotated_image(state), meas_json, state.debug

    select_prompt = _build_select_prompt()
    grammar = _get_grammar("concept_select")
    lp = medgemma.json_logits_processor(grammar) if grammar else None

    select_json_text = ""
    select_degenerated = False
    try:
        for delta in medgemma.chat_continue_cached(
            user_content=select_prompt,
            assistant_response=answer_text,
            max_new_tokens=600,
            do_sample=False,
            logits_processor=lp,
            repetition_penalty=1.3,
        ):
            select_json_text += delta
            # Degeneration guard: if SELECT output gets unreasonably long,
            # it's likely looping. Truncate and try to salvage.
            if len(select_json_text) > 1500:
                logger.warning("SELECT: output too long (%d chars), likely degeneration — truncating", len(select_json_text))
                select_degenerated = True
                break
    except Exception as e:
        logger.warning("SELECT (MedGemma concept selection) failed: %s", e)

    state.debug["SELECT_raw"] = select_json_text

    # Parse, validate, dedup, and cap selected concepts
    selected_concepts = _parse_select_json(select_json_text)
    logger.info("SELECT raw concepts (%d): %s", len(selected_concepts), selected_concepts)
    selected_concepts = _fuzzy_validate_concepts(selected_concepts, answer_text)
    selected_concepts = _dedup_concepts(selected_concepts)
    if len(selected_concepts) > 8:
        logger.info("SELECT: trimming from %d to 8 concepts", len(selected_concepts))
        selected_concepts = selected_concepts[:8]

    # Normalize for MedSAM3
    concept_terms = _prepare_medsam3_concepts(user_prompt, selected_concepts)
    state.debug["CONCEPTS_list"] = concept_terms

    if not concept_terms:
        select_step.status = "done"
        select_step.detail = "No concepts selected."
        state.debug["SEG_segment_count"] = 0
        _sync_segment_meta(state)
        yield state.chat, render_steps_html(state.steps), build_annotated_image(state), meas_json, state.debug
        job_dt = time.perf_counter() - job_t0
        logger.info("run_parallel_job DONE (no concepts) job=%s total=%.1fs", state.job_id[:8], job_dt)
        return

    select_step.status = "done"
    select_step.detail = f"Selected: {', '.join(concept_terms[:5])}"
    logger.info("SELECT validated concepts: %s", concept_terms)
    _sync_segment_meta(state)
    yield state.chat, render_steps_html(state.steps), build_annotated_image(state), meas_json, state.debug

    # ---- STEP PRIOR: Extract attention-based spatial priors (optional) ----
    spatial_priors: dict[str, tuple[int, int, int, int]] = {}
    attention_heatmaps: dict[str, np.ndarray] = {}
    if ATTENTION_PRIOR_ENABLED:
        prior_step = Step(id="PRIOR", name="Attention priors", status="running", detail="Extracting spatial priors…", segment_ids=[])
        state.steps.append(prior_step)
        _sync_segment_meta(state)
        yield state.chat, render_steps_html(state.steps), build_annotated_image(state), meas_json, state.debug

        try:
            from models.attention_prior import heatmap_to_box

            # GradCAM first (gradient-based attribution), attention fallback
            attention_heatmaps = medgemma.extract_concept_heatmaps_gradcam(pil_image, concept_terms)
            if attention_heatmaps:
                logger.info("PRIOR: GradCAM produced %d heatmaps", len(attention_heatmaps))
            else:
                logger.info("PRIOR: GradCAM unavailable, falling back to attention")
                attention_heatmaps = medgemma.extract_concept_heatmaps(pil_image, concept_terms)

            # Convert heatmaps to boxes if mode uses boxes
            use_boxes = ATTENTION_PRIOR_MODE in ("box", "both")
            if use_boxes:
                for concept, hmap in attention_heatmaps.items():
                    box = heatmap_to_box(hmap, img_h, img_w)
                    if box is not None:
                        spatial_priors[concept] = box
                        logger.info("PRIOR: concept='%s' → box=%s", concept, box)
                    else:
                        logger.info("PRIOR: concept='%s' → no significant region", concept)

            mode_detail = f"mode={ATTENTION_PRIOR_MODE}"
            n_heatmaps = len(attention_heatmaps)
            n_boxes = len(spatial_priors)
            prior_step.status = "done"
            prior_step.detail = f"{mode_detail}: {n_heatmaps} heatmaps, {n_boxes} boxes"
            state.debug["PRIOR_mode"] = ATTENTION_PRIOR_MODE
            state.debug["PRIOR_heatmaps"] = list(attention_heatmaps.keys())
            if spatial_priors:
                state.debug["PRIOR_boxes"] = {c: list(b) for c, b in spatial_priors.items()}
            if attention_heatmaps:
                state.debug["_attention_heatmaps"] = attention_heatmaps
        except Exception as e:
            logger.warning("PRIOR step failed (continuing without priors): %s", e)
            prior_step.status = "done"
            prior_step.detail = f"Failed: {e}"
            spatial_priors = {}
            attention_heatmaps = {}

        _sync_segment_meta(state)
        yield state.chat, render_steps_html(state.steps), build_annotated_image(state), meas_json, state.debug

    # Invalidate KV cache — no longer needed after PRIOR
    medgemma.invalidate_cache()

    # ---- STEP SEG: MedSAM3 segments using selected concepts ----
    seg_step = Step(id="SEG", name="MedSAM3: segmentation", status="running", detail="Segmenting concepts…", segment_ids=[])
    state.steps.append(seg_step)
    _sync_segment_meta(state)
    yield state.chat, render_steps_html(state.steps), build_annotated_image(state), meas_json, state.debug

    medsam3 = get_medsam3()
    try:
        use_overlay = ATTENTION_PRIOR_MODE in ("overlay", "both") and attention_heatmaps
        use_box_only = ATTENTION_PRIOR_MODE == "box" and spatial_priors

        if use_overlay:
            # Overlay mode: attention heatmap as alpha channel + optional box prompts
            auto_results = refined_segment_with_attention_overlay(
                img_np, concept_terms, medsam3, attention_heatmaps,
                spatial_priors=spatial_priors if ATTENTION_PRIOR_MODE == "both" else None,
            )
        elif use_box_only:
            # Box-only mode: spatial prior boxes
            auto_results = refined_segment_with_priors(
                img_np, concept_terms, medsam3, spatial_priors,
            )
        elif REFINED_SEG_ENABLED:
            auto_results = refined_segment(img_np, concept_terms, medsam3)
        else:
            auto_results = medsam3.segment_concepts(
                img_np, concept_terms,
                threshold=MEDSAM3_CONCEPT_THRESHOLD,
                mask_threshold=MEDSAM3_CONCEPT_MASK_THRESHOLD,
                max_masks_per_concept=MEDSAM3_CONCEPT_MAX_MASKS_PER_CONCEPT,
                nms_iou_thresh=MEDSAM3_NMS_IOU_THRESH,
                max_total_masks=MEDSAM3_MAX_MASKS,
                preprocess=MEDSAM3_AUTO_PREPROCESS,
            )
    except Exception as e:
        seg_step.status = "failed"
        seg_step.detail = f"Segmentation failed: {e}"
        logger.warning("SEG failed in parallel mode: %s", e)
        _sync_segment_meta(state)
        yield state.chat, render_steps_html(state.steps), build_annotated_image(state), meas_json, state.debug
        job_dt = time.perf_counter() - job_t0
        logger.info("run_parallel_job DONE (seg failed) job=%s total=%.1fs", state.job_id[:8], job_dt)
        return

    # Store segments progressively (yield after each)
    for result in auto_results:
        seg_id = state.next_segment_id()
        mask = result["mask"]
        bbox = result["bbox"]
        meas = measure_mask(mask, pixel_spacing=state.pixel_spacing)
        concept = result.get("concept", "")
        label = f"{concept} ({seg_id})" if concept else f"Region {seg_id}"
        state.add_segment(
            segment_id=seg_id, label=label, mask=mask, bbox=bbox,
            created_by_step="SEG", measurements=meas,
        )
        state.segments[seg_id]["concept"] = concept
        seg_step.segment_ids.append(seg_id)
        meas_json[seg_id] = meas
        _sync_segment_meta(state)
        yield state.chat, render_steps_html(state.steps), build_annotated_image(state), meas_json, state.debug

    seg_step.status = "done"
    seg_step.detail = f"Found {len(auto_results)} region(s)."
    state.debug["SEG_segment_count"] = len(auto_results)
    _sync_segment_meta(state)
    yield state.chat, render_steps_html(state.steps), build_annotated_image(state), meas_json, state.debug

    # ---- STEP LINK: Deterministic alias extraction → concept linking ----
    link_step = Step(id="LINK", name="Extract highlights", status="running", detail="Scanning answer phrases…", segment_ids=[])
    state.steps.append(link_step)
    _sync_segment_meta(state)
    yield state.chat, render_steps_html(state.steps), build_annotated_image(state), meas_json, state.debug

    # Deterministic root-phrase extraction
    seg_concepts = {sid: seg.get("concept", "") for sid, seg in state.segments.items()}
    all_concepts_set = set(seg_concepts.values())
    deterministic_aliases: dict[str, list[str]] = {}
    for seg_concept in all_concepts_set:
        if not seg_concept:
            continue
        other = all_concepts_set - {seg_concept}
        aliases = _extract_contextual_aliases(seg_concept, answer_text, other)
        deterministic_aliases[seg_concept] = aliases

    # Cross-reference prescan terms not used as segment concepts
    cross_ref = _cross_reference_prescan_aliases(prescan_terms, seg_concepts, answer_text)
    for concept, extra in cross_ref.items():
        existing = deterministic_aliases.get(concept, [])
        existing_lower = {e.lower() for e in existing}
        for a in extra:
            if a.lower() not in existing_lower:
                existing.append(a)
                existing_lower.add(a.lower())
        deterministic_aliases[concept] = existing

    det_alias_count = sum(len(v) for v in deterministic_aliases.values())
    logger.info("LINK: deterministic aliases=%d for %d concepts", det_alias_count, len(deterministic_aliases))

    link_step.status = "done"
    link_step.detail = f"Found {det_alias_count} highlight phrase(s)"
    _sync_segment_meta(state)
    yield state.chat, render_steps_html(state.steps), build_annotated_image(state), meas_json, state.debug

    # ---- Concept linking: deterministic aliases → segment IDs ----
    from backend.image_service import segment_color_hex

    concept_links = []
    for seg_id, seg in state.segments.items():
        seg_concept = seg.get("concept", "")
        color = segment_color_hex(seg.get("color_idx", 0))
        aliases = list(deterministic_aliases.get(seg_concept, []))

        concept_links.append({
            "concept": seg_concept,
            "segment_id": seg_id,
            "aliases": aliases,
            "color": color,
        })

    state.debug["_concept_links"] = concept_links
    logger.info("Concept links: %s", concept_links)

    _sync_segment_meta(state)
    yield state.chat, render_steps_html(state.steps), build_annotated_image(state), meas_json, state.debug

    job_dt = time.perf_counter() - job_t0
    logger.info(
        "run_parallel_job DONE job=%s total=%.1fs segments=%d concepts=%d links=%d",
        state.job_id[:8], job_dt, len(state.segments), len(concept_terms), len(concept_links),
    )
