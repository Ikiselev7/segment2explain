# Agent Navigation Guide - Segment2Explain PoC

## Purpose

This document provides AI coding agents with comprehensive instructions on:
- How to navigate this codebase
- Where to find project information
- How to run tests and quality checks
- Development workflow best practices
- Quality gates that must pass before completing work

**Target audience:** Claude, Cursor, GitHub Copilot, and other AI coding assistants

---

## Quick Start

### 1. Understanding the Project

**What is this?**
- Medical imaging decision-support PoC combining MedSAM3 (segmentation) + MedGemma (reasoning)
- Core innovation: **"no evidence → no claim"** - every explanation must reference segmented regions with measurements
- Goal: Transform into interactive, chat-first, evidence-first UI
- Segmentation backend policy: use `MedSAM3` for all visual operations (segment-everything, concept segmentation, mask generation); do not introduce legacy MedSAM or generic SAM3 tracker/pipeline backends.

**Read these files first (in order):**

1. `README.md` - User-facing documentation, setup instructions
2. `IMPLEMENTATION_PLAN.md` - Complete implementation plan (6 phases)
3. `.claude/plans/polymorphic-conjuring-rocket.md` - Detailed design decisions
4. `tests/fixtures/README.md` - Dataset integration strategy

### 2. Environment Setup

```bash
# Verify uv is installed
which uv  # Should output: /opt/homebrew/bin/uv

# Install all dependencies
uv sync --dev

# Verify installation
uv run pytest --version
uv run ruff --version
```

### 3. Project Structure

```
segment2explain_poc/
├── IMPLEMENTATION_PLAN.md       # ← START HERE for tasks
├── AGENTS.md                    # ← This file
├── README.md                    # User documentation
├── pyproject.toml               # Dependencies + config
├── ruff.toml                    # Linting rules
│
├── main.py                      # FastAPI entry point (uvicorn on port 8000)
├── orchestrator.py              # State management (JobState, Step, segments)
│
├── backend/                     # FastAPI backend
│   ├── main.py                  # FastAPI app, routes, static serving
│   ├── pipeline.py              # Pipeline logic (run_job generator)
│   ├── ws.py                    # WebSocket adapter (state diffing)
│   ├── schemas.py               # Pydantic WS message schemas
│   ├── image_service.py         # Image upload/storage/overlay
│   ├── config.py                # Environment config
│   └── dependencies.py          # Model singletons
│
├── frontend/                    # React + TypeScript + Vite SPA
│   ├── src/                     # Components, stores, hooks, types
│   └── tests/                   # Playwright e2e tests
│
├── models/
│   └── medgemma_torch.py        # MedGemma wrapper with streaming
│
├── tools/
│   ├── medsam3_tool.py          # MedSAM3 segmentation wrapper
│   ├── measure.py               # Mask measurements (area, bbox, diameter)
│   └── overlay.py               # Visualization (mask overlays)
│
├── prompts/
│   ├── system_prompt.txt        # Evidence-based reasoning rules
│   └── templates.py             # Grounded prompts, evidence packets
│
├── utils/                       # ← Phase 1+ utility modules
│   ├── segment_chip_processor.py  # Detect + convert segment refs to HTML
│   ├── steps_renderer.py          # Render steps as interactive HTML
│   ├── segments_list_renderer.py  # Render segments list as HTML table
│   └── finding_injector.py        # Post-process text to inject chips
│
└── tests/
    ├── conftest.py              # Shared fixtures (sample images, mocks)
    ├── test_orchestrator.py     # State management tests
    ├── test_segment_chip_processor.py
    ├── test_steps_renderer.py
    ├── test_segments_list_renderer.py
    ├── test_finding_injector.py
    └── fixtures/
        └── sample_cxr_vqa/      # VinDr-CXR-VQA test data (manual download)
```

---

## Development Workflow

### Before Starting Any Work

1. **Read the current phase in `IMPLEMENTATION_PLAN.md`**
   - Understand phase goals
   - Review task list
   - Check verification criteria

2. **Check current test status**
   ```bash
   uv run pytest -v
   ```

3. **Check code quality**
   ```bash
   uv run ruff check .
   ```

### While Working

1. **Write tests FIRST** (TDD approach)
   - Create test file: `tests/test_<module_name>.py`
   - Write failing tests for new functionality
   - Implement feature until tests pass

2. **Follow existing patterns**
   - Use dataclasses for state (`@dataclass`)
   - Type hints required (modern Python: `dict` not `Dict`, `list` not `List`)
   - Generators for streaming updates

3. **Import structure**
   ```python
   # Standard library
   import os
   import json
   from dataclasses import dataclass
   from typing import Any

   # Third-party
   import numpy as np
   from PIL import Image

   # Local
   from orchestrator import JobState, Step
   from utils.segment_chip_processor import process_segment_chips
   ```

### After Implementing Changes

**MANDATORY QUALITY GATES** (all must pass):

```bash
# 1. All tests pass
uv run pytest -v
# Expected: No failures, all tests green

# 2. No linting errors
uv run ruff check .
# Expected: No errors (warnings acceptable if documented)

# 3. Code formatted
uv run ruff format .
# Expected: Files reformatted if needed

# 4. Frontend e2e tests (if UI changes)
cd frontend && npx playwright test
# Expected: All Playwright tests pass
```

**If any gate fails, fix before proceeding to next task.**

---

## Key Architecture Patterns

### 1. State Management (orchestrator.py)

**JobState is the single source of truth:**

```python
@dataclass
class JobState:
    job_id: str | None
    image: np.ndarray | None  # HxWx3 uint8
    chat: list[dict]           # OpenAI-style messages
    steps: list[Step]          # Workflow steps
    segments: dict[str, dict]  # Segment ID → segment data
    highlight: str             # Current highlight selection
    debug: dict[str, Any]      # Debug info
```

**When modifying state:**
- Never mutate directly - use helper methods
- `state.next_segment_id()` → generates A, B, C, ...
- `state.add_segment()` → adds new segment with all metadata

### 2. Streaming Updates (backend/pipeline.py)

**run_job() is a generator yielding state snapshots:**

```python
def run_job(...) -> Generator[tuple[...], None, None]:
    state = create_job_state()
    _sync_segment_meta(state)
    yield state.chat, render_steps(...), build_annotated_image(state), meas_json, state.debug
```

**Pattern:** Always yield complete state tuple. The WebSocket adapter (`backend/ws.py`) diffs consecutive yields and sends incremental messages to the frontend.

### 3. Evidence Grounding Contract

**System prompt enforces:** "no evidence → no claim"

```python
# Building evidence packet
evidence_packet = {
    "segment_id": "A",
    "label": "User-selected region",
    "measurements": {
        "area_px": 12450,
        "bbox_px": [200, 200, 299, 299],
        "max_diameter_px": 140.71,
        "centroid_px": [249.5, 249.5]
    },
    "bbox_px": [200, 200, 299, 299],
    "image_size_px": [512, 512]
}

# Pass to MedGemma with overlay image
grounded_prompt = build_grounded_user_prompt(user_prompt, evidence_packet)
medgemma.chat_stream(grounded_prompt, images=[original_img, overlay_img])
```

**Rule:** MedGemma must reference Segment IDs in explanations.

### 4. Frontend Architecture (React + TypeScript)

**State management:** Zustand store (`frontend/src/stores/jobStore.ts`)
**WebSocket:** `useWebSocket` hook connects to `ws://localhost:8000/ws/{imageId}`
**Cross-referencing:** `useHover` hook provides bidirectional hover (image canvas ↔ chat chips ↔ segment table)
**Tests:** Playwright e2e tests with mock WebSocket backend (`frontend/tests/`)

---

## Testing Guidelines

### Test Structure

```python
# tests/test_<module>.py
"""Tests for <module description>."""

import pytest
from module import function_to_test


class TestFeatureName:
    """Test <feature> functionality."""

    def test_basic_case(self):
        """Test basic functionality with valid input."""
        result = function_to_test(valid_input)
        assert result == expected_output

    def test_edge_case(self):
        """Test edge case: empty input."""
        result = function_to_test("")
        assert result is None

    def test_error_handling(self):
        """Test graceful failure on invalid input."""
        with pytest.raises(ValueError):
            function_to_test(invalid_input)
```

### Fixtures (conftest.py)

**Available fixtures:**

- `sample_image_np` - 512x512 RGB test image
- `sample_bbox` - (200, 200, 299, 299)
- `sample_mask` - Binary mask matching bbox
- `sample_measurements` - Measurement dict

**Usage:**

```python
def test_with_fixtures(sample_image_np, sample_bbox):
    """Test using shared fixtures."""
    result = segment_box(sample_image_np, sample_bbox)
    assert result.shape == (512, 512)
```

### Running Tests

```bash
# All tests
uv run pytest

# Specific file
uv run pytest tests/test_orchestrator.py

# Specific test
uv run pytest tests/test_orchestrator.py::TestJobState::test_create_job_state

# With coverage
uv run pytest --cov=. --cov-report=term-missing

# Verbose output
uv run pytest -v

# Stop on first failure
uv run pytest -x

# Run only failed tests from last run
uv run pytest --lf
```

---

## Code Quality Standards

### Ruff Configuration (ruff.toml)

- Line length: 120 characters
- Target: Python 3.10+
- Enabled rules: E, F, W, I, N, UP, B, C4, SIM
- Auto-fixable: All enabled rules

### Common Linting Issues

**Issue:** `UP035` - Use modern type hints
```python
# Bad
from typing import Dict, List, Optional
def foo(x: Optional[Dict[str, List[int]]]) -> None: pass

# Good
def foo(x: dict[str, list[int]] | None) -> None: pass
```

**Issue:** `F401` - Unused import
```python
# Bad
import time  # imported but never used

# Good
# Remove the import or use it
```

**Issue:** `SIM115` - Use context manager
```python
# Bad
f = open("file.txt", "r")
data = f.read()
f.close()

# Good
with open("file.txt", "r") as f:
    data = f.read()
```

### Auto-Fixing

```bash
# Fix all auto-fixable issues
uv run ruff check . --fix

# Format all files
uv run ruff format .
```

---

## Common Tasks

### UI Verification

**Run Playwright e2e tests after UI/pipeline changes:**

```bash
cd frontend && npx playwright test
```

**Headless pipeline tests (no browser needed):**

```bash
uv run python run_samples.py nodule_mass
```

### Adding a New Utility Module

1. **Create module file**
   ```bash
   touch utils/new_module.py
   ```

2. **Add to utils/__init__.py**
   ```python
   from utils.new_module import main_function

   __all__ = ["main_function", ...]
   ```

3. **Create test file**
   ```bash
   touch tests/test_new_module.py
   ```

4. **Write tests first**
   ```python
   def test_main_function():
       result = main_function(input_data)
       assert result == expected
   ```

5. **Implement module**

6. **Verify**
   ```bash
   uv run pytest tests/test_new_module.py
   uv run ruff check utils/new_module.py
   ```

### Modifying the Pipeline (backend/pipeline.py)

**Before modifying:**
1. Understand current flow in `run_job()`
2. Identify yield points (where WS adapter sends updates)
3. Check how state is updated

**While modifying:**
1. Maintain generator pattern
2. Call `_sync_segment_meta(state)` before every `yield`
3. Update step status before yielding
4. Handle errors gracefully (try/except, mark step as failed)

**After modifying:**
```bash
uv run pytest tests/ -v  # Backend tests
cd frontend && npx playwright test  # Frontend e2e tests
```

### Adding New Dependencies

```bash
# Production dependency
uv add package-name

# Dev dependency
uv add --dev package-name

# Verify it's in pyproject.toml
cat pyproject.toml | grep package-name

# Test import
python -c "import package_name; print('OK')"
```

---

## Debugging Tips

### 1. Check Logs

**FastAPI logs to console:**
```bash
uv run python main.py
# Watch console for errors, warnings
```

**Add debug logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)
logger.debug(f"State: {state.segments}")
```

### 2. Test Isolation

**Run single test with print statements:**
```python
def test_debug():
    result = function(input)
    print(f"Result: {result}")  # Shows in pytest output with -s
    assert result == expected

# Run with output
uv run pytest tests/test_file.py::test_debug -s
```

### 3. Interactive Testing

```python
# In Python REPL
from orchestrator import create_job_state
state = create_job_state()
state.next_segment_id()  # 'A'
state.next_segment_id()  # 'B'
```

### 4. Common Errors

**Error:** `ModuleNotFoundError: No module named 'utils'`
- **Fix:** Run from project root: `cd /Users/Ilia_Kiselev/Documents/segment2explain_poc`

**Error:** `TypeError: expected str, bytes or os.PathLike object, not NoneType`
- **Fix:** Check that state.image is set before using

**Error:** WebSocket messages not reaching frontend
- **Fix:** Check `backend/ws.py` diff logic and `frontend/src/hooks/useWebSocket.ts` message handler

---

## Current Implementation Status

### ✅ Phase 0: Complete

- uv setup
- pyproject.toml + ruff.toml
- pytest structure (13 tests passing)
- Fixtures directory
- README.md updated

### 🔄 Next: Phase 1 (Foundation)

**Goal:** Create utility modules without breaking existing UI

**Tasks:**
1. Create `utils/` directory
2. Implement `segment_chip_processor.py` + tests
3. Implement `steps_renderer.py` + tests
4. Implement `segments_list_renderer.py` + tests
5. Implement `finding_injector.py` + tests

**Verification:**
```bash
uv run pytest  # All tests pass
uv run ruff check .  # No errors
python -c "from utils.segment_chip_processor import process_segment_chips; print('OK')"
```

---

## Phase-Specific Instructions

### Phase 1: Foundation

**Focus:** Create utility modules, extensive testing

**Key files to create:**
- `utils/__init__.py`
- `utils/segment_chip_processor.py`
- `utils/steps_renderer.py`
- `utils/segments_list_renderer.py`
- `utils/finding_injector.py`
- Corresponding test files

**Quality gates:**
- Each module has >80% test coverage
- All regex patterns tested with edge cases
- HTML output validated for correct structure

### Phase 2: Clickable Chips

**Focus:** UI modification, JavaScript integration

**Key changes:**
- Modify `_format_assistant()` in app.py
- Add CSS via `gr.HTML`
- Add JavaScript bridge components
- Wire bridge to highlight update

**Manual testing required:**
- Upload image, draw box, submit prompt
- Click segment chip → verify highlight

### Phase 3: Two-Stage Pipeline

**Focus:** Complex pipeline restructuring

**Key changes:**
- Split S3 into S3A/S3B/S3C
- Add new prompts to templates.py
- Implement grounding loop
- Chip injection after grounding

**Testing focus:**
- JSON parse failures
- MedSAM3 errors during grounding
- Empty findings
- Progress indicators

### Phase 4: Bidirectional Linking

**Focus:** Interactive HTML, JavaScript bridges

**Key changes:**
- Replace markdown with HTML
- Add step click handlers
- Wire segment → step navigation

**Manual testing required:**
- Click different steps
- Verify correct segments highlight
- Scroll behavior works

### Phase 5: Segments List

**Focus:** Additional UI component

**Key changes:**
- Add segments_list_html component
- Render as table
- Wire click handlers

**Manual testing required:**
- Create 5+ segments
- Click rows
- Verify all interactions

---

## Clinical Context (Important for AI Agents)

### What This System IS

- **Clinical cognition support** for radiologists/trainees
- **Auditable and interactive** image interpretation
- **Evidence-grounded** with pixel-level traceability

### What This System IS NOT

- ❌ NOT autonomous diagnosis
- ❌ NOT for fuzzy/diffuse findings
- ❌ NOT production medical device
- ❌ Research/education/prototyping ONLY

**When writing code:**
- Never claim diagnostic capabilities
- Always enforce evidence grounding
- Maintain "no evidence → no claim" principle
- Include uncertainty in explanations

### No Hardcoded Bias Principle

This application must work on a large variety of inputs. **NEVER** introduce:

- Keyword/pattern lists for intent classification or routing
- Hardcoded anatomy lists or concept hint dictionaries
- Regex-based intent detection or decision branching based on user input patterns

**Let models reason.** If MedGemma misclassifies intent, improve the prompt or model — don't add keyword overrides. Hardcoded patterns bias the system toward specific examples and break on novel inputs (typos, different phrasing, different languages).

**Acceptable patterns:** text normalization (strip HTML, lowercase), JSON output parsing, degeneration detection (filtering model artifacts like `<unused>` tokens).

---

## Resources

### Documentation Files

- `README.md` - Setup, running, demo
- `IMPLEMENTATION_PLAN.md` - Complete plan, all phases
- `.claude/plans/polymorphic-conjuring-rocket.md` - Detailed design
- `tests/fixtures/README.md` - Dataset info
- `prompts/system_prompt.txt` - Evidence rules

### Code Entry Points

- `main.py` - FastAPI entry point
- `backend/pipeline.py` - Pipeline logic (run_job generator)
- `backend/ws.py` - WebSocket adapter (state diffing → incremental messages)
- `frontend/src/stores/jobStore.ts` - Frontend state management
- `orchestrator.py` - Backend state management (JobState)
- `tools/medsam3_tool.py` - Start here for segmentation
- `models/medgemma_torch.py` - Start here for reasoning

### External References

- **VinDr-CXR-VQA:** https://arxiv.org/html/2511.00504v2
- **MedSAM3:** https://huggingface.co/lal-Joey/MedSAM3_v1
- **MedGemma:** https://huggingface.co/google/medgemma-1.5-4b-it
- **uv docs:** https://docs.astral.sh/uv/

---

## Questions?

If you encounter issues or have questions:

1. Check `IMPLEMENTATION_PLAN.md` for phase details
2. Check existing tests for usage examples
3. Check `.claude/plans/` for design decisions
4. Run quality gates to catch common issues

**Remember:** Every phase must pass all quality gates before moving to next phase.

**Quality gates checklist:**
- [ ] `uv run pytest -v` → All tests pass
- [ ] `uv run ruff check .` → No errors
- [ ] `uv run ruff format .` → Code formatted
- [ ] Manual UI test passes (if UI changes)
