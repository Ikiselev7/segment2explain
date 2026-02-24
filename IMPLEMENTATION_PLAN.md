# Segment2Explain PoC - Implementation Plan

## Overview

This document outlines the complete implementation plan for transforming segment2explain_poc into a **chat-first, evidence-first UI** with interactive segment grounding.

**Goal:** Enable bidirectional linking between workflow steps, segmented regions, and text explanations while maintaining clinical positioning for realistic adoption.

## Status: Phase 0 Complete ✅

Last Updated: 2026-02-07

---

## Phase 0: Repository Setup & Testing Infrastructure ✅

**Status:** COMPLETED

**Duration:** 2-3 hours

### Completed Tasks

- [x] Initialize uv for dependency management
- [x] Create pyproject.toml with all production + dev dependencies
- [x] Configure ruff.toml for code quality (line-length=120, modern Python)
- [x] Set up pytest directory structure
- [x] Create initial test suite (13 tests, 100% pass rate)
- [x] Create fixtures directory for VinDr-CXR-VQA samples
- [x] Update README.md with development setup instructions

### Verification

```bash
uv run pytest -v        # 13 tests passing
uv run ruff check .     # Linter working
uv sync --dev           # All dependencies installed
```

### Manual Task (Deferred)

- [ ] Download 5-10 VinDr-CXR-VQA sample cases
  - See `tests/fixtures/README.md` for format specification
  - Can be completed later for integration testing

---

## Phase 1: Foundation (Non-Breaking)

**Status:** PENDING

**Duration:** ~4 hours

**Goal:** Create utility modules without modifying existing UI

### Tasks

1. **Create `utils/` directory structure**
   ```bash
   mkdir -p utils
   touch utils/__init__.py
   ```

2. **Implement `utils/segment_chip_processor.py`**
   - Detect segment references using regex: `[[SEG:A]]`, `[SEG:A]`, "Segment A"
   - Replace with HTML: `<span class="seg-chip" data-seg-id="A">[SEG:A]</span>`
   - Only process segments that exist in state
   - Write tests in `tests/test_segment_chip_processor.py`

3. **Implement `utils/steps_renderer.py`**
   - Convert `Step` objects to interactive HTML
   - Include data attributes: `data-step-id`, `data-segment-ids`
   - Add status classes: `step-queued`, `step-running`, `step-done`, `step-failed`
   - Write tests in `tests/test_steps_renderer.py`

4. **Implement `utils/segments_list_renderer.py`**
   - Render segments as HTML table
   - Include: ID, label, created-by step, status, area
   - Add data attributes for click handling
   - Write tests in `tests/test_segments_list_renderer.py`

5. **Implement `utils/finding_injector.py`** (Phase 3 dependency)
   - Post-process text to inject chips after grounding
   - Match finding descriptions to insertion points
   - Write tests in `tests/test_finding_injector.py`

### Verification Criteria

```bash
# All tests pass
uv run pytest

# No linting errors
uv run ruff check .

# Code formatted
uv run ruff format .

# New modules importable
python -c "from utils.segment_chip_processor import process_segment_chips; print('OK')"
```

### Files Created

- `utils/__init__.py`
- `utils/segment_chip_processor.py`
- `utils/steps_renderer.py`
- `utils/segments_list_renderer.py`
- `utils/finding_injector.py`
- `tests/test_segment_chip_processor.py`
- `tests/test_steps_renderer.py`
- `tests/test_segments_list_renderer.py`
- `tests/test_finding_injector.py`

---

## Phase 2: Clickable Segment Chips

**Status:** PENDING

**Duration:** ~4 hours

**Goal:** Make segment references in text clickable with highlight functionality

### Tasks

1. **Add CSS for segment chips**
   - Create `gr.HTML` component with inline styles
   - Blue background, white text, rounded corners
   - Hover effects

2. **Add JavaScript bridge components**
   - Hidden `gr.Textbox(elem_id="chip_click_bridge")`
   - JavaScript to attach click listeners
   - Update bridge value → trigger Gradio `.change()` event

3. **Modify `_format_assistant()` in app.py**
   - Import `process_segment_chips` from utils
   - Process text before returning
   - Pass available segment IDs

4. **Wire bridge to highlight update**
   ```python
   chip_click_bridge.change(
       lambda seg_id, st: (build_annotated_image(st, highlight=seg_id), seg_id),
       inputs=[chip_click_bridge, state],
       outputs=[annotated, highlight]
   )
   ```

5. **Test interactivity**
   - Upload image, draw box, ask question
   - Verify "Segment A" becomes clickable
   - Click chip → image highlights correct segment

### Verification Criteria

```bash
# Tests pass
uv run pytest

# Manual UI test
uv run python app.py
# 1. Upload image, draw box
# 2. Submit prompt
# 3. Click [SEG:A] chip in response
# 4. Verify image highlights Segment A
```

### Files Modified

- `app.py`: Add CSS HTML, bridge component, modify `_format_assistant()`
- Tests updated for new behavior

---

## Phase 3: Two-Stage Answer Strategy

**Status:** PENDING

**Duration:** ~6 hours (most complex)

**Goal:** Fast preliminary answer + async grounding pass with chip injection

### Current Flow

```
S1 Parse → S2 Segment ROI → S3 Explain (full) → [S4/S5 Extra findings]
```

### New Flow

```
S1 Parse → S2 Segment primary ROI (Segment A)
├─→ S3A: Quick preliminary explanation (300 tokens, references Segment A)
└─→ S3B: Draft structured findings JSON (hidden from user)
    └─→ S3C-1, S3C-2, ...: Ground each finding
        ├─→ Call MedSAM3 for each finding → creates Segments B, C, D
        ├─→ Show progress: "Grounding 1/3 findings... ✅"
        └─→ After all grounding: update S3A message with injected chips
```

### Tasks

1. **Add new prompt templates to `prompts/templates.py`**
   - `build_quick_assessment_prompt()` - for S3A (fast answer)
   - `build_findings_json_prompt()` - for S3B (structured findings)

2. **Restructure `run_job()` in app.py**
   - Split S3 into S3A (quick), S3B (draft), S3C-N (grounding loop)
   - Store original S3A text in `state.debug["s3a_original"]`
   - Add progress indicators in step details

3. **Implement grounding loop (S3C)**
   - Parse findings JSON from S3B
   - For each finding:
     - Create step `S3C-{idx}`
     - Call MedSAM3 with bbox
     - Create new segment (B, C, D, ...)
     - Update progress: "Grounding 1/3 findings... ✅"

4. **Chip injection after grounding**
   - Use `inject_segment_chips()` from utils
   - Update S3A message content with chips
   - Yield updated chat state

5. **Handle edge cases**
   - JSON parse failure → fallback to no grounding
   - MedSAM3 failure → mark step as failed, don't break pipeline
   - Empty findings → graceful completion

### Verification Criteria

```bash
# Tests pass with new pipeline
uv run pytest

# Manual UI test
uv run python app.py
# 1. Upload image
# 2. Enable "Ground additional findings"
# 3. Submit prompt
# 4. Verify:
#    - S3A appears <5 seconds
#    - S3B drafts findings
#    - S3C-1, S3C-2 create Segments B, C
#    - S3A message updates with [SEG:B], [SEG:C] chips
```

### Files Modified

- `app.py`: Restructure `run_job()` generator
- `prompts/templates.py`: Add new prompt builders
- `utils/finding_injector.py`: Chip injection logic
- Tests for new pipeline behavior

---

## Phase 4: Bidirectional Step ↔ Segment Linking

**Status:** PENDING

**Duration:** ~4 hours

**Goal:** Interactive steps with hover/click to highlight segments and vice versa

### Tasks

1. **Replace steps markdown with HTML**
   - Change `steps_md = gr.Markdown()` to `steps_html = gr.HTML()`
   - Update all `render_steps_markdown()` calls to `render_steps_html()`

2. **Add JavaScript for step interactions**
   - Click step → update highlight (via hidden bridge)
   - Hover step → CSS visual feedback only
   - Step cards include data attributes: `data-step-id`, `data-segment-ids`

3. **Wire step click to highlight update**
   ```python
   step_click_bridge.change(
       lambda step_id, st: build_annotated_image(st, highlight=step_id),
       inputs=[step_click_bridge, state],
       outputs=[annotated]
   )
   ```

4. **Add segment → step navigation**
   - When segment clicked in list → scroll to creating step
   - Use JavaScript `scrollIntoView({behavior: 'smooth'})`

5. **Test interactions**
   - Click step S2 → highlights Segment A
   - Click step S3C-1 → highlights Segment B
   - Hover step → visual highlight

### Verification Criteria

```bash
# Tests pass
uv run pytest

# Manual UI test
uv run python app.py
# 1. Create multiple segments
# 2. Click step S2 → Segment A highlights
# 3. Click step S3C-1 → Segment B highlights
# 4. Hover effects work
```

### Files Modified

- `app.py`: Replace markdown with HTML, add bridges, wire events
- `orchestrator.py`: Add `render_steps_html()`
- `utils/steps_renderer.py`: HTML rendering with data attributes

---

## Phase 5: Dedicated Segments List Component

**Status:** PENDING

**Duration:** ~3 hours

**Goal:** Interactive HTML table showing all segments with metadata

### Tasks

1. **Add segments list HTML component**
   ```python
   segments_list_html = gr.HTML()
   ```

2. **Render segments as table**
   - Columns: ID, Label, Created By, Status, Area
   - Rows have data attributes: `data-segment-id`, `data-step-id`
   - Click row → highlight on image + scroll to step

3. **Wire segment click handlers**
   ```python
   segment_click_bridge.change(
       lambda seg_id, st: (build_annotated_image(st, highlight=seg_id), seg_id),
       inputs=[segment_click_bridge, state],
       outputs=[annotated, highlight]
   )
   ```

4. **Add to run_job() yields**
   - Update segments list HTML on every segment creation
   - Real-time updates as grounding progresses

5. **Style with CSS**
   - Hover effects on rows
   - Status icons (✅ done, ⏳ pending, ❌ failed)

### Verification Criteria

```bash
# Tests pass
uv run pytest

# Manual UI test
uv run python app.py
# 1. Create 3+ segments
# 2. Verify list shows all segments
# 3. Click row → highlights segment on image
# 4. Click row → scrolls to creating step
```

### Files Modified

- `app.py`: Add segments_list_html component, wire to yields
- `utils/segments_list_renderer.py`: HTML table rendering
- CSS for table styling

---

## Testing Strategy

### Unit Tests

Each utility module has dedicated tests:
- `test_segment_chip_processor.py`: Regex patterns, HTML generation
- `test_finding_injector.py`: Chip injection logic
- `test_steps_renderer.py`: HTML output structure
- `test_segments_list_renderer.py`: Table rendering

### Integration Tests

- `test_full_pipeline.py`: End-to-end workflow (requires models)
- `test_vindr_integration.py`: VinDr-CXR-VQA cases (when data available)

### Quality Gates (Every Phase)

```bash
# All tests must pass
uv run pytest -v

# No linting errors
uv run ruff check .

# Code formatted
uv run ruff format .

# Coverage >80% for new code
uv run pytest --cov=utils
```

### Manual UI Testing Checklist

After each phase:
- [ ] App launches without errors
- [ ] Can upload image and draw box
- [ ] Can submit prompt and get response
- [ ] New features work as expected
- [ ] No regressions in existing features

---

## Clinical Positioning

### Core Value Proposition

**"Evidence-grounded interpretation + measurement assistant for clinician-in-loop workflows"**

NOT autonomous diagnosis.

### Best-Fit Use Cases

1. **Follow-up measurements** (oncology tumor tracking, nodule growth)
2. **Education & training** (residents learning to interpret images)
3. **QA/peer review** (auditable evidence trails)
4. **Annotation acceleration** (research dataset creation)

### Explicit Limitations

- ❌ Not for autonomous diagnosis or triage
- ❌ Not for diffuse/fuzzy findings without clear boundaries
- ❌ Must minimize interaction burden (<15 seconds)
- ❌ 3D volumetric workflows not yet supported
- ❌ Grounding budget enforced (top 1-3 findings)

---

## Dataset Integration: VinDr-CXR-VQA

### Why This Dataset

- 4,394 CXRs with 17,597 QA pairs
- Radiologist-verified bounding boxes
- Clinical reasoning explanations
- Structured questions: "Where?", "What?", "Is there?"

### Demo Examples (5 Cases)

1. **Single finding** - "Where is the infiltration?"
2. **Multiple findings** - "What abnormalities are present?"
3. **Negative case** - "Is there any abnormality?"
4. **Ambiguous finding** - "What is the opacity?"
5. **Follow-up scenario** - Two time points for measurement tracking

### Evaluation Metrics

**Grounding Quality:**
- IoU > 0.5 (acceptable), IoU > 0.7 (good)
- Dice coefficient for pixel-level comparison

**Reasoning Quality:**
- Segment reference compliance: % of findings with `[SEG:X]` or "Segment X"
- Evidence grounding rate: % of MedGemma findings successfully grounded (target >80%)

**UX Metrics:**
- Time to first answer (S3A): <5 seconds
- Total grounding time (S3B+S3C): <30 seconds for 2-3 findings
- Chip click → highlight latency: <100ms

---

## Implementation Timeline

| Phase | Duration | Tasks | Cumulative |
|-------|----------|-------|------------|
| 0 ✅  | 2-3h    | Setup + testing infrastructure | 3h |
| 1     | 4h      | Foundation utils modules | 7h |
| 2     | 4h      | Clickable chips | 11h |
| 3     | 6h      | Two-stage pipeline | 17h |
| 4     | 4h      | Bidirectional linking | 21h |
| 5     | 3h      | Segments list | 24h |

**Total estimated effort:** ~24 hours (3 working days)

---

## Rollback Strategy

Each phase is independently reversible:

- **Phase 2 fails:** Remove chip processor, revert `_format_assistant()`
- **Phase 3 fails:** Disable grounding flag, use original single-stage S3
- **Phase 4 fails:** Revert to markdown steps, remove JavaScript bridges
- **Phase 5 fails:** Hide segments list component

All changes maintain backwards compatibility with `JobState` structure.

---

## Key Differentiators for Judges

1. ✅ **Every segment reference is clickable and traceable**
2. ✅ **Steps show real-time progress with visual linking to image**
3. ✅ **Two-stage answer demonstrates methodical evidence generation**
4. ✅ **Segments list provides professional, comprehensive view of all evidence**
5. ✅ **Honest clinical positioning** (not overpromising capabilities)

The enhanced UI makes it immediately obvious that **"text is tied to pixels"** through evidence-grounded workflow steps.

---

## References

- Plan file: `.claude/plans/polymorphic-conjuring-rocket.md`
- Fixtures documentation: `tests/fixtures/README.md`
- Development setup: `README.md` Section 4
- Agent navigation: `AGENTS.md`
