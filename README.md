# Segment2Explain — PoC (Mac MPS, all-Torch)

This PoC runs **both models in PyTorch on Apple Silicon (MPS)**:

- **MedSAM3** (segment-everything + concept-guided segmentation) on MPS
- **MedGemma 1.5 4B IT** (multimodal image-text-to-text) on MPS via Hugging Face Transformers

**User experience**
1) Upload a medical image.
2) Ask a question (targeted) or request a description.
3) MedGemma+MedSAM3 pipeline: intent classification → segmentation → filtering → analysis.
4) Interactive results: colored segment overlays, clickable chips, cross-linked hover.

> ⚠️ Not a diagnostic device. Research/education/prototyping only.

---

## 0) Prerequisites

- macOS on Apple Silicon (M4 48GB recommended)
- Python 3.10+ (3.11 recommended)

### Accept MedGemma terms on Hugging Face (required)
The MedGemma Hugging Face repo is gated: you must be logged in and accept **Health AI Developer Foundations terms of use** before you can download weights.

Model page:
- https://huggingface.co/google/medgemma-1.5-4b-it

---

## 1) Setup

```bash
cd segment2explain_poc
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Login to Hugging Face (so gated files can download):
```bash
huggingface-cli login
```

### Merge MedSAM3 weights

The merged MedSAM3 model checkpoint (`models/medsam3-merged/`) is not included in the repository due to its size (~3.1 GB). You must generate it locally by merging LoRA weights into the base SAM3 model.

**Requirements:**
- Hugging Face access to [facebook/sam3](https://huggingface.co/facebook/sam3) (base model, ~3.1 GB download)
- Hugging Face access to [lal-Joey/MedSAM3_v1](https://huggingface.co/lal-Joey/MedSAM3_v1) (LoRA weights, ~12 MB download)
- ~10 GB free disk space (base model cache + merged output)

```bash
uv run python scripts/merge_medsam3_lora.py --output models/medsam3-merged
```

This downloads the base SAM3 model and MedSAM3 LoRA weights from Hugging Face, merges 455 LoRA pairs (rank=16, alpha=32), and saves the merged checkpoint to `models/medsam3-merged/`.

To verify the merge:
```bash
uv run python scripts/merge_medsam3_lora.py --verify models/medsam3-merged
```

---

## 2) Run

```bash
# Start FastAPI backend (port 8001)
uv run python main.py

# In another terminal, start React frontend (port 5173)
cd frontend && npm run dev
```

Open http://localhost:5173 in your browser.

---

## 3) Notes

- Transformers support for Gemma 3 is available from **transformers 4.50.0+**.
- Default dtype on MPS is float16 (override with `MODEL_DTYPE=bfloat16|float16|float32`).
- To use a different MedGemma model id:
```bash
export MEDGEMMA_MODEL_ID=google/medgemma-1.5-4b-it
```

Example dtype override:
```bash
export MODEL_DTYPE=float16
```

---

## 4) Development Setup

For development, we use **uv** for dependency management, **pytest** for testing, and **ruff** for linting/formatting.

### Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install dependencies with uv

```bash
# Install all dependencies (production + development)
uv sync --dev

# Activate the virtual environment (optional)
source .venv/bin/activate
```

### Running tests

```bash
# Run all tests with coverage
uv run pytest

# Run specific test file
uv run pytest tests/test_orchestrator.py

# Run with verbose output
uv run pytest -v
```

### Code quality checks

```bash
# Check code with ruff
uv run ruff check .

# Auto-fix issues where possible
uv run ruff check . --fix

# Format code
uv run ruff format .
```

### Headless pipeline checks (`run_samples.py`)

```bash
# Run all fixture samples headlessly
uv run python run_samples.py

# Run one or more named samples
uv run python run_samples.py cardiomegaly
uv run python run_samples.py cardiomegaly nodule_mass
```

Outputs:
- Per-sample logs in terminal (steps, status, latency)
- Summary JSON at `tests/fixtures/sample_cxr_vqa/run_results.json`

### Frontend (React) tests

```bash
cd frontend && npx playwright test
```

### MedSAM3 quality/tuning notes

- The merged MedSAM3 checkpoint is configured for `target_size=1008` with resize+normalize in `models/medsam3-merged/processor_config.json`.
- Keep `MEDSAM3_AUTO_PREPROCESS=true` for grayscale-like medical images; this enables percentile contrast normalization before segmentation.
- For quality tuning by modality/task, adjust:
  - `MEDSAM3_PRED_IOU_THRESH`
  - `MEDSAM3_CONCEPT_MASK_THRESHOLD`
  - `MEDSAM3_NMS_IOU_THRESH`
  - `MEDSAM3_MIN_MASK_AREA_RATIO`
  - `MEDSAM3_MAX_MASK_AREA_RATIO`
- For CT/MR workflows, apply modality-appropriate windowing/normalization upstream before converting to RGB uint8.

### Project Structure

```
segment2explain_poc/
├── main.py                     # FastAPI entry point (uvicorn)
├── orchestrator.py             # State management (JobState, Step)
├── backend/                    # FastAPI backend
│   ├── main.py                 # FastAPI app, routes, static serving
│   ├── pipeline.py             # Pipeline logic (run_job generator)
│   ├── ws.py                   # WebSocket adapter (state diffing)
│   ├── schemas.py              # Pydantic WS message schemas
│   ├── image_service.py        # Image upload/storage/overlay
│   ├── config.py               # Environment config
│   └── dependencies.py         # Model singletons
├── frontend/                   # React + TypeScript + Vite SPA
│   ├── src/
│   │   ├── components/         # UI components (ChatPanel, ImagePanel, etc.)
│   │   ├── stores/             # Zustand state (jobStore)
│   │   ├── hooks/              # Custom hooks (useWebSocket, useHover)
│   │   └── types/              # TypeScript types
│   └── tests/                  # Playwright e2e tests
├── models/                     # Model wrappers
│   └── medgemma_torch.py
├── tools/                      # Segmentation and measurement tools
│   ├── medsam3_tool.py
│   ├── refined_segmentation.py
│   ├── measure.py
│   └── overlay.py
├── prompts/                    # System prompts and templates
├── utils/                      # Utility modules
├── tests/                      # Backend test suite (pytest)
├── pyproject.toml              # Project metadata and dependencies
└── ruff.toml                   # Linting configuration
```

### Adding new dependencies

```bash
# Production dependency
uv add package-name

# Development dependency
uv add --dev package-name
```

### Clinical Positioning

This system is positioned as **"evidence-grounded interpretation + measurement assistant for clinician-in-loop workflows"**, not autonomous diagnosis.

**Best-fit use cases:**
- Follow-up measurements and progression tracking
- Education and training for residents/fellows
- QA/peer review with audit trails
- Annotation acceleration for research

**NOT intended for:**
- Autonomous diagnosis or triage
- Diffuse/ambiguous findings without clear boundaries
- High-stakes workflows requiring simple, deterministic pipelines

See [tests/fixtures/README.md](tests/fixtures/README.md) for dataset integration strategy (VinDr-CXR-VQA).

---

## 5) Demo script (3 minutes)

1) Upload a chest X-ray image.
2) Ask: "Where is the nodule?" (targeted pipeline) or "Describe this image" (describe pipeline).
3) Watch the pipeline execute: step cards update in real-time, segments appear as colored overlays.
4) Hover segment chips in the answer to highlight corresponding regions on the image.
