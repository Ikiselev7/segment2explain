Start the Segment2Explain FastAPI+React app in the background.

1. Kill any existing processes on the ports:
   ```bash
   pkill -f "uvicorn" 2>/dev/null; pkill -f "vite" 2>/dev/null
   lsof -i :8000 -t 2>/dev/null | xargs kill -9 2>/dev/null
   lsof -i :5173 -t 2>/dev/null | xargs kill -9 2>/dev/null
   ```
2. Wait 1 second
3. Start FastAPI backend: run `uv run python main.py` in the background using the Bash tool with `run_in_background: true`
4. Start React dev server: run `cd frontend && npm run dev` in the background using the Bash tool with `run_in_background: true`
5. Poll `curl -s http://127.0.0.1:8000/api/health` and `curl -s http://localhost:5173` every 2 seconds up to 20 times until both respond
6. Report whether both servers started successfully

- **FastAPI backend**: http://127.0.0.1:8000
- **React frontend**: http://localhost:5173
- Models load lazily on first request, not at startup
