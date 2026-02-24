"""Tests for FastAPI REST endpoints and image service."""

import io

import numpy as np
import pytest
from PIL import Image

from backend.image_service import (
    cleanup_image,
    get_image_array,
    get_image_bytes,
    mask_to_contour_points,
    segment_color_hex,
    store_image,
)


class TestImageService:
    """Tests for image upload, storage, and retrieval."""

    def _make_png_bytes(self, w: int = 64, h: int = 64) -> bytes:
        img = Image.new("RGB", (w, h), color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_store_and_retrieve(self):
        png = self._make_png_bytes(100, 80)
        image_id, w, h, _ps = store_image(png)
        assert w == 100
        assert h == 80
        assert image_id

        arr = get_image_array(image_id)
        assert arr is not None
        assert arr.shape == (80, 100, 3)

        png_out = get_image_bytes(image_id)
        assert png_out is not None
        assert len(png_out) > 0

        cleanup_image(image_id)
        assert get_image_array(image_id) is None

    def test_store_grayscale_converts_to_rgb(self):
        img = Image.new("L", (32, 32), color=128)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_id, w, h, _ps = store_image(buf.getvalue())
        arr = get_image_array(image_id)
        assert arr.shape == (32, 32, 3)
        cleanup_image(image_id)

    def test_nonexistent_image_returns_none(self):
        assert get_image_array("nonexistent") is None
        assert get_image_bytes("nonexistent") is None


class TestMaskToContourPoints:
    """Tests for contour extraction from masks."""

    def test_simple_rectangle(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:50, 10:50] = 1
        polygons = mask_to_contour_points(mask)
        assert len(polygons) == 1  # Single contiguous region
        assert len(polygons[0]) >= 4  # At least 4 corner points
        # All points should be within mask bounds
        for pt in polygons[0]:
            assert 0 <= pt[0] < 64
            assert 0 <= pt[1] < 64

    def test_empty_mask(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        polygons = mask_to_contour_points(mask)
        assert polygons == []

    def test_full_mask(self):
        mask = np.ones((64, 64), dtype=np.uint8)
        polygons = mask_to_contour_points(mask)
        assert len(polygons) == 1
        assert len(polygons[0]) >= 4

    def test_circle_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        cy, cx = 50, 50
        for y in range(100):
            for x in range(100):
                if (x - cx) ** 2 + (y - cy) ** 2 < 30 ** 2:
                    mask[y, x] = 1
        polygons = mask_to_contour_points(mask)
        assert len(polygons) == 1  # Single contiguous region
        assert len(polygons[0]) >= 6  # Circle has many simplified points

    def test_disconnected_regions(self):
        """Two separate rectangles should produce two polygons."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:30] = 1  # Top-left rectangle
        mask[60:80, 60:80] = 1  # Bottom-right rectangle
        polygons = mask_to_contour_points(mask)
        assert len(polygons) == 2
        for polygon in polygons:
            assert len(polygon) >= 4


class TestSegmentColorHex:
    """Tests for segment color generation."""

    def test_returns_hex_string(self):
        color = segment_color_hex(0)
        assert color.startswith("#")
        assert len(color) == 7

    def test_wraps_around(self):
        color0 = segment_color_hex(0)
        color10 = segment_color_hex(10)
        assert color0 == color10  # 10 colors, should wrap

    def test_different_colors(self):
        colors = [segment_color_hex(i) for i in range(5)]
        assert len(set(colors)) == 5  # All different


class TestFastAPIEndpoints:
    """Tests for REST API endpoints."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from backend.main import app

        return TestClient(app)

    def test_health(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "models_loaded" in data

    def test_upload_and_get_image(self, client):
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        resp = client.post("/api/upload-image", files={"file": ("test.png", buf, "image/png")})
        assert resp.status_code == 200
        data = resp.json()
        assert data["width"] == 64
        assert data["height"] == 64
        assert data["image_id"]
        assert data["url"].startswith("/api/images/")

        # Retrieve the image
        resp2 = client.get(data["url"])
        assert resp2.status_code == 200
        assert resp2.headers["content-type"] == "image/png"

    def test_get_nonexistent_image(self, client):
        resp = client.get("/api/images/nonexistent")
        assert resp.status_code == 404

    def test_get_overlay_nonexistent(self, client):
        resp = client.get("/api/images/nonexistent/overlay")
        assert resp.status_code == 404


class TestWebSocketEndpoint:
    """Basic WebSocket connection tests."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from backend.main import app

        return TestClient(app)

    def test_websocket_connect_and_invalid_json(self, client):
        with client.websocket_connect("/ws/pipeline") as ws:
            ws.send_text("not json")
            resp = ws.receive_json()
            assert resp["type"] == "error"

    def test_websocket_missing_image_id(self, client):
        with client.websocket_connect("/ws/pipeline") as ws:
            ws.send_json({"type": "start_job", "prompt": "test"})
            resp = ws.receive_json()
            assert resp["type"] == "error"

    def test_websocket_unknown_message_type(self, client):
        with client.websocket_connect("/ws/pipeline") as ws:
            ws.send_json({"type": "foobar"})
            resp = ws.receive_json()
            assert resp["type"] == "error"

    def test_websocket_image_not_found(self, client):
        with client.websocket_connect("/ws/pipeline") as ws:
            ws.send_json({"type": "start_job", "image_id": "nonexistent", "prompt": "test"})
            resp = ws.receive_json()
            assert resp["type"] == "job_failed"
            assert "not found" in resp["error"].lower()
