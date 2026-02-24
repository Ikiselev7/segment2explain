"""Tests for tool call parsing, stripping, and region extraction."""

from backend.pipeline import _extract_regions_json, _region_to_bbox_px, parse_tool_calls, strip_tool_calls


class TestParseToolCalls:
    """Test extraction of <TOOL_CALL>...</TOOL_CALL> blocks."""

    def test_single_tool_call(self):
        text = 'Some text <TOOL_CALL>{"tool": "medsam3_segment", "label": "opacity", "bbox_px": [10, 20, 100, 200]}</TOOL_CALL>'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["label"] == "opacity"
        assert calls[0]["bbox_px"] == [10, 20, 100, 200]

    def test_multiple_tool_calls(self):
        text = (
            '<TOOL_CALL>{"tool": "medsam3_segment", "label": "a", "bbox_px": [1,2,3,4]}</TOOL_CALL>'
            " middle text "
            '<TOOL_CALL>{"tool": "medsam3_segment", "label": "b", "bbox_px": [5,6,7,8]}</TOOL_CALL>'
        )
        calls = parse_tool_calls(text)
        assert len(calls) == 2
        assert calls[0]["label"] == "a"
        assert calls[1]["label"] == "b"

    def test_no_tool_calls(self):
        text = "Just regular text without any tool calls."
        calls = parse_tool_calls(text)
        assert calls == []

    def test_invalid_json(self):
        text = "<TOOL_CALL>not valid json</TOOL_CALL>"
        calls = parse_tool_calls(text)
        assert calls == []

    def test_missing_bbox(self):
        text = '<TOOL_CALL>{"tool": "medsam3_segment", "label": "test"}</TOOL_CALL>'
        calls = parse_tool_calls(text)
        assert calls == []

    def test_multiline_json(self):
        text = '<TOOL_CALL>{\n  "tool": "medsam3_segment",\n  "label": "heart",\n  "bbox_px": [100, 200, 300, 400]\n}</TOOL_CALL>'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["label"] == "heart"

    def test_empty_string(self):
        assert parse_tool_calls("") == []

    def test_partial_tag(self):
        text = '<TOOL_CALL>{"bbox_px": [1,2,3,4]}'
        calls = parse_tool_calls(text)
        assert calls == []


class TestStripToolCalls:
    """Test removal of <TOOL_CALL>...</TOOL_CALL> blocks from text."""

    def test_strips_single_call(self):
        text = 'Before <TOOL_CALL>{"bbox_px": [1,2,3,4]}</TOOL_CALL> After'
        result = strip_tool_calls(text)
        assert result == "Before  After"

    def test_strips_multiple_calls(self):
        text = "A <TOOL_CALL>x</TOOL_CALL> B <TOOL_CALL>y</TOOL_CALL> C"
        result = strip_tool_calls(text)
        assert result == "A  B  C"

    def test_no_calls_unchanged(self):
        text = "Normal text"
        result = strip_tool_calls(text)
        assert result == "Normal text"

    def test_only_tool_call(self):
        text = '<TOOL_CALL>{"bbox_px": [1,2,3,4]}</TOOL_CALL>'
        result = strip_tool_calls(text)
        assert result == ""

    def test_multiline_call(self):
        text = 'Before\n<TOOL_CALL>{\n"bbox_px": [1,2,3,4]\n}</TOOL_CALL>\nAfter'
        result = strip_tool_calls(text)
        assert "Before" in result
        assert "After" in result
        assert "TOOL_CALL" not in result


class TestExtractRegionsJson:
    """Test extraction of region proposals from MedGemma JSON output."""

    def test_standard_format(self):
        text = '{"regions": [{"label": "heart", "bbox_px": [100, 200, 300, 400]}]}'
        regions = _extract_regions_json(text)
        assert len(regions) == 1
        assert regions[0]["label"] == "heart"
        assert regions[0]["bbox_px"] == [100, 200, 300, 400]

    def test_multiple_regions(self):
        text = '{"regions": [{"label": "a", "bbox_px": [1,2,3,4]}, {"label": "b", "bbox_px": [5,6,7,8]}]}'
        regions = _extract_regions_json(text)
        assert len(regions) == 2

    def test_with_surrounding_text(self):
        text = 'Here is my analysis:\n{"regions": [{"label": "opacity", "bbox_px": [10, 20, 100, 200]}]}\nDone.'
        regions = _extract_regions_json(text)
        assert len(regions) == 1
        assert regions[0]["label"] == "opacity"

    def test_empty_regions(self):
        text = '{"regions": []}'
        regions = _extract_regions_json(text)
        assert regions == []

    def test_no_json(self):
        text = "Just plain text without any JSON."
        regions = _extract_regions_json(text)
        assert regions == []

    def test_empty_string(self):
        assert _extract_regions_json("") == []

    def test_invalid_json(self):
        text = '{"regions": [invalid json here]}'
        regions = _extract_regions_json(text)
        assert regions == []

    def test_missing_bbox(self):
        text = '{"regions": [{"label": "test"}]}'
        regions = _extract_regions_json(text)
        assert regions == []

    def test_bare_array(self):
        text = '[{"label": "nodule", "bbox_px": [50, 60, 70, 80]}]'
        regions = _extract_regions_json(text)
        assert len(regions) == 1
        assert regions[0]["label"] == "nodule"

    def test_bbox_pct_format(self):
        text = '{"regions": [{"label": "heart", "bbox_pct": [30, 40, 70, 80]}]}'
        regions = _extract_regions_json(text)
        assert len(regions) == 1
        assert regions[0]["label"] == "heart"
        assert regions[0]["bbox_pct"] == [30, 40, 70, 80]

    def test_mixed_bbox_formats(self):
        text = '{"regions": [{"label": "a", "bbox_pct": [10, 20, 30, 40]}, {"label": "b", "bbox_px": [100, 200, 300, 400]}]}'
        regions = _extract_regions_json(text)
        assert len(regions) == 2


class TestRegionToBboxPx:
    """Test conversion of region bbox (pct or px) to pixel coordinates."""

    def test_pct_to_px(self):
        region = {"bbox_pct": [25, 50, 75, 100]}
        bbox = _region_to_bbox_px(region, img_w=1000, img_h=2000)
        assert bbox == (250, 1000, 750, 1999)

    def test_px_passthrough(self):
        region = {"bbox_px": [100, 200, 300, 400]}
        bbox = _region_to_bbox_px(region, img_w=1000, img_h=1000)
        assert bbox == (100, 200, 300, 400)

    def test_clamps_to_image_bounds(self):
        region = {"bbox_pct": [0, 0, 150, 150]}
        bbox = _region_to_bbox_px(region, img_w=100, img_h=100)
        assert bbox is not None
        assert bbox[2] <= 99
        assert bbox[3] <= 99

    def test_clamps_negative(self):
        region = {"bbox_pct": [-10, -20, 50, 60]}
        bbox = _region_to_bbox_px(region, img_w=100, img_h=100)
        assert bbox is not None
        assert bbox[0] >= 0
        assert bbox[1] >= 0

    def test_invalid_region(self):
        assert _region_to_bbox_px({}, img_w=100, img_h=100) is None
        assert _region_to_bbox_px({"bbox_pct": "bad"}, img_w=100, img_h=100) is None

    def test_ensures_min_size(self):
        region = {"bbox_pct": [50, 50, 50, 50]}
        bbox = _region_to_bbox_px(region, img_w=100, img_h=100)
        assert bbox is not None
        assert bbox[2] > bbox[0]
        assert bbox[3] > bbox[1]
