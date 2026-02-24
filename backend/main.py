"""FastAPI application for Segment2Explain backend."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.image_service import get_image_bytes, render_heatmap_png, render_overlay, store_image
from backend.schemas import HealthResponse, ImageUploadResponse
from backend.ws import ws_pipeline

logger = logging.getLogger(__name__)

app = FastAPI(title="Segment2Explain API", version="1.0.0")

# CORS for frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    from backend.dependencies import _medgemma, _medsam3

    return HealthResponse(
        status="ok",
        models_loaded={
            "medgemma": _medgemma is not None,
            "medsam3": _medsam3 is not None,
        },
    )


@app.post("/api/upload-image", response_model=ImageUploadResponse)
async def upload_image(file: UploadFile):
    """Upload a medical image for analysis."""
    img_bytes = await file.read()
    image_id, width, height, pixel_spacing = store_image(img_bytes)
    return ImageUploadResponse(
        image_id=image_id,
        width=width,
        height=height,
        url=f"/api/images/{image_id}",
        pixel_spacing_mm=list(pixel_spacing) if pixel_spacing else None,
    )


@app.get("/api/images/{image_id}")
async def get_image(image_id: str):
    """Serve an uploaded image as PNG."""
    img_bytes = get_image_bytes(image_id)
    if img_bytes is None:
        return Response(status_code=404, content="Image not found")
    return Response(content=img_bytes, media_type="image/png")


@app.get("/api/images/{image_id}/overlay")
async def get_overlay(image_id: str, segments: str = ""):
    """Serve overlay PNG with segment contours.

    Query params:
        segments: comma-separated segment IDs to include (empty = all)
    """
    overlay_bytes = render_overlay(image_id)
    if overlay_bytes is None:
        return Response(status_code=404, content="Image not found")
    return Response(content=overlay_bytes, media_type="image/png")


@app.get("/api/images/{image_id}/heatmap/{concept:path}")
async def get_heatmap(image_id: str, concept: str):
    """Serve a colorized attention heatmap PNG for a concept."""
    heatmap_bytes = render_heatmap_png(image_id, concept)
    if heatmap_bytes is None:
        return Response(status_code=404, content="Heatmap not found")
    return Response(content=heatmap_bytes, media_type="image/png")


# WebSocket endpoint
app.websocket("/ws/pipeline")(ws_pipeline)

# Serve frontend static files in production (after API routes so they take priority)
_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=_FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    async def serve_spa(path: str):
        """Serve SPA index.html for all non-API routes."""
        file = _FRONTEND_DIST / path
        if file.is_file():
            return FileResponse(file)
        return FileResponse(_FRONTEND_DIST / "index.html")
