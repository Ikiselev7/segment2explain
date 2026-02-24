"""Tests for _parse_validation_json."""

from backend.pipeline import _parse_validation_json


class TestParseValidationJson:
    """Test parsing of MedGemma validation responses."""

    def test_ok_status(self):
        result = _parse_validation_json('{"status": "ok"}')
        assert result is not None
        assert result["status"] == "ok"

    def test_adjust_status(self):
        result = _parse_validation_json('{"status": "adjust", "bbox_pct": [10, 20, 30, 40]}')
        assert result is not None
        assert result["status"] == "adjust"
        assert result["bbox_pct"] == [10, 20, 30, 40]

    def test_invalid_json(self):
        assert _parse_validation_json("not json at all") is None

    def test_empty_string(self):
        assert _parse_validation_json("") is None

    def test_no_status_field(self):
        assert _parse_validation_json('{"foo": "bar"}') is None

    def test_adjust_without_bbox(self):
        assert _parse_validation_json('{"status": "adjust"}') is None

    def test_adjust_with_bad_bbox(self):
        assert _parse_validation_json('{"status": "adjust", "bbox_pct": "bad"}') is None

    def test_adjust_with_short_bbox(self):
        assert _parse_validation_json('{"status": "adjust", "bbox_pct": [10, 20]}') is None

    def test_with_surrounding_text(self):
        text = 'Based on my analysis, the segment looks correct.\n{"status": "ok"}\nEnd.'
        result = _parse_validation_json(text)
        assert result is not None
        assert result["status"] == "ok"

    def test_adjust_with_surrounding_text(self):
        text = 'The segment is misaligned. Here is my correction:\n{"status": "adjust", "bbox_pct": [25, 30, 75, 80]}'
        result = _parse_validation_json(text)
        assert result is not None
        assert result["status"] == "adjust"
        assert result["bbox_pct"] == [25, 30, 75, 80]

    def test_unknown_status(self):
        assert _parse_validation_json('{"status": "maybe"}') is None

    def test_multiple_json_objects_picks_valid(self):
        text = '{"unrelated": true} and then {"status": "ok"}'
        result = _parse_validation_json(text)
        assert result is not None
        assert result["status"] == "ok"
