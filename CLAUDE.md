# Segment2Explain PoC — Agent Instructions

## Architecture

- **Backend**: FastAPI + WebSocket (`backend/`) — pipeline logic in `backend/pipeline.py`
- **Frontend**: React + TypeScript + Vite (`frontend/`) — SPA with WebSocket connection
- **Entry point**: `main.py` starts FastAPI on port 8000

## UI Verification (REQUIRED)

After making any changes that affect the UI (frontend components, backend WS messages, pipeline), you MUST verify:

1. Start the app: use `/start-app` command
2. Run Playwright tests: `cd frontend && npx playwright test`
3. If the change affects pipeline behavior, run `/test-sample` to verify end-to-end

## No Hardcoded Bias (CRITICAL)

This application must work on a large variety of inputs. **NEVER** introduce solutions that bias behavior toward a specific group of examples:

- **No keyword/pattern lists** for intent classification, routing, or decision-making. Let models (MedGemma, MedSAM3) reason about inputs themselves.
- **No hardcoded anatomy lists** or concept hint dictionaries. The model should decide what to look for.
- **No regex-based intent detection.** If intent classification needs improvement, improve the prompt or model, not add pattern matching.
- **Acceptable:** text normalization (stripping HTML tags, lowercasing), JSON parsing, degeneration detection (filtering model artifacts like `<unused>` tokens).

The distinction: **parsing model output** is fine. **Overriding model decisions** with patterns is not.

## Testing

- Run backend tests: `uv run pytest tests/ -v`
- Run frontend e2e tests: `cd frontend && npx playwright test`
- Run headless sample: `uv run python run_samples.py nodule_mass`
- Always run tests after code changes before reporting completion.

## Key Commands

- `/start-app` — Start FastAPI backend + React dev server
- `/stop-app` — Kill running servers
- `/screenshot` — Take screenshot of React app via Selenium
- `/check-ui` — Verify frontend + backend health + run Playwright tests
- `/test-sample` — Run sample through pipeline headlessly
