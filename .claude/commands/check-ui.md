Verify the React frontend and FastAPI backend are running and responding correctly.

## Steps

1. Check FastAPI backend health:
   ```bash
   curl -s http://127.0.0.1:8000/health
   ```

2. Check React frontend serves HTML:
   ```bash
   curl -s http://localhost:5173 | head -5
   ```

3. Check WebSocket endpoint exists:
   ```bash
   curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/ws
   ```

4. Run Playwright tests to verify UI components:
   ```bash
   cd frontend && npx playwright test
   ```

Report which components are working and any failures found.
