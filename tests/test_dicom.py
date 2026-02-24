"""Tests for DICOM parsing, windowing, and pixel spacing extraction."""

from __future__ import annotations

import io

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian

from backend.dicom import is_dicom, parse_dicom


def _make_dicom_bytes(
    rows: int = 64,
    cols: int = 64,
    pixel_spacing: tuple[float, float] | None = (0.5, 0.5),
    bits: int = 16,
    window_center: float | None = None,
    window_width: float | None = None,
    rescale_slope: float | None = None,
    rescale_intercept: float | None = None,
) -> bytes:
    """Create a minimal synthetic DICOM file in-memory."""
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset("test.dcm", {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = bits
    ds.BitsStored = bits
    ds.HighBit = bits - 1
    ds.PixelRepresentation = 0 if bits == 16 else 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    if pixel_spacing is not None:
        ds.PixelSpacing = list(pixel_spacing)

    if window_center is not None:
        ds.WindowCenter = window_center
    if window_width is not None:
        ds.WindowWidth = window_width
    if rescale_slope is not None:
        ds.RescaleSlope = rescale_slope
    if rescale_intercept is not None:
        ds.RescaleIntercept = rescale_intercept

    # Create gradient pixel data
    if bits == 16:
        pixel_data = np.linspace(0, 4095, rows * cols, dtype=np.uint16).reshape(rows, cols)
    else:
        pixel_data = np.linspace(0, 255, rows * cols, dtype=np.uint8).reshape(rows, cols)
    ds.PixelData = pixel_data.tobytes()

    buf = io.BytesIO()
    ds.save_as(buf)
    return buf.getvalue()


class TestIsDicom:
    def test_positive_synthetic(self):
        data = _make_dicom_bytes()
        assert is_dicom(data) is True

    def test_negative_png(self):
        from PIL import Image

        img = Image.new("RGB", (32, 32), (128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        assert is_dicom(buf.getvalue()) is False

    def test_negative_empty(self):
        assert is_dicom(b"") is False

    def test_negative_short(self):
        assert is_dicom(b"short") is False

    def test_negative_jpeg(self):
        from PIL import Image

        img = Image.new("RGB", (32, 32), (128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        assert is_dicom(buf.getvalue()) is False


class TestParseDicom:
    def test_basic_with_pixel_spacing(self):
        data = _make_dicom_bytes(rows=64, cols=64, pixel_spacing=(0.3, 0.4))
        rgb, ps = parse_dicom(data)

        assert rgb.shape == (64, 64, 3)
        assert rgb.dtype == np.uint8
        assert ps is not None
        assert ps == pytest.approx((0.3, 0.4))

    def test_no_pixel_spacing(self):
        data = _make_dicom_bytes(rows=32, cols=32, pixel_spacing=None)
        rgb, ps = parse_dicom(data)

        assert rgb.shape == (32, 32, 3)
        assert rgb.dtype == np.uint8
        assert ps is None

    def test_windowing_applied(self):
        data = _make_dicom_bytes(
            rows=64,
            cols=64,
            window_center=2048.0,
            window_width=4096.0,
        )
        rgb, _ = parse_dicom(data)

        # Should use full range given gradient 0-4095 and window center=2048 width=4096
        assert rgb.min() >= 0
        assert rgb.max() <= 255
        # Should have some spread
        assert rgb.max() > 100

    def test_rescale_slope_intercept(self):
        data = _make_dicom_bytes(
            rows=32,
            cols=32,
            rescale_slope=2.0,
            rescale_intercept=-1000.0,
            window_center=3096.0,  # Center of rescaled range: 2*4095-1000=7190, center=~3595
            window_width=8190.0,
        )
        rgb, _ = parse_dicom(data)

        assert rgb.shape == (32, 32, 3)
        assert rgb.dtype == np.uint8

    def test_output_is_rgb_3_channel(self):
        data = _make_dicom_bytes(rows=16, cols=16)
        rgb, _ = parse_dicom(data)

        # Must be 3-channel RGB
        assert rgb.ndim == 3
        assert rgb.shape[-1] == 3
        # Grayscale DICOM → all channels equal
        assert np.array_equal(rgb[..., 0], rgb[..., 1])
        assert np.array_equal(rgb[..., 1], rgb[..., 2])


class TestImageServiceDicom:
    """Integration: DICOM through image_service.store_image()."""

    def test_store_dicom_returns_pixel_spacing(self):
        from backend.image_service import cleanup_image, get_pixel_spacing, store_image

        data = _make_dicom_bytes(rows=48, cols=48, pixel_spacing=(0.25, 0.25))
        image_id, w, h, ps = store_image(data)

        assert w == 48
        assert h == 48
        assert ps == pytest.approx((0.25, 0.25))
        assert get_pixel_spacing(image_id) == pytest.approx((0.25, 0.25))
        cleanup_image(image_id)

    def test_real_tf_dicom_no_spacing(self):
        """TensorFlow sample CXR DICOM — valid but no pixel spacing."""
        from pathlib import Path

        from backend.image_service import cleanup_image, get_pixel_spacing, store_image

        dcm_path = Path(__file__).parent / "fixtures" / "dicom" / "chest_xray_tf.dcm"
        data = dcm_path.read_bytes()
        image_id, w, h, ps = store_image(data)

        assert w == 1024
        assert h == 1024
        assert ps is None
        assert get_pixel_spacing(image_id) is None
        cleanup_image(image_id)

    def test_real_siim_dicom_end_to_end(self):
        """Full pipeline: real SIIM CXR DICOM → store → measure with mm."""
        from pathlib import Path

        from backend.image_service import cleanup_image, get_pixel_spacing, store_image
        from tools.measure import measure_mask

        dcm_path = Path(__file__).parent / "fixtures" / "dicom" / "siim_pneumothorax.dcm"
        data = dcm_path.read_bytes()
        image_id, w, h, ps = store_image(data)

        assert w == 1024
        assert h == 1024
        assert ps is not None
        assert ps[0] == pytest.approx(0.143, abs=0.001)

        # Measure a 100x100 pixel region
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[462:562, 462:562] = 1
        meas = measure_mask(mask, pixel_spacing=ps)

        assert meas["area_px"] == 10000
        assert meas["area_mm2"] == pytest.approx(10000 * 0.143 * 0.143, abs=0.5)
        assert meas["max_diameter_mm"] > 0
        assert meas["pixel_spacing_mm"] is not None

        cleanup_image(image_id)

    def test_store_png_returns_no_pixel_spacing(self):
        from PIL import Image

        from backend.image_service import cleanup_image, get_pixel_spacing, store_image

        img = Image.new("RGB", (32, 32), (128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        image_id, w, h, ps = store_image(buf.getvalue())
        assert ps is None
        assert get_pixel_spacing(image_id) is None
        cleanup_image(image_id)
