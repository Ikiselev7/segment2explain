Stop the running Segment2Explain FastAPI+React app.

1. Kill processes: `pkill -f "uvicorn" 2>/dev/null; pkill -f "vite" 2>/dev/null`
2. Verify stopped: `curl -s http://127.0.0.1:8000/health > /dev/null && echo "Backend still running" || echo "Backend stopped"; curl -s http://localhost:5173 > /dev/null && echo "Frontend still running" || echo "Frontend stopped"`
3. Report the result
