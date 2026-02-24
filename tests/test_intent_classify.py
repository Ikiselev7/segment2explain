"""Tests for _parse_classify_json and _parse_concept_match_json."""

from backend.pipeline import _parse_classify_json, _parse_concept_match_json


class TestParseClassifyJson:
    """Test parsing of MedGemma segment identification responses."""

    def test_name_and_description(self):
        result = _parse_classify_json('{"name": "Heart", "description": "Enlarged cardiac silhouette"}')
        assert result is not None
        assert result["name"] == "Heart"
        assert result["description"] == "Enlarged cardiac silhouette"

    def test_name_only(self):
        result = _parse_classify_json('{"name": "Nodule"}')
        assert result is not None
        assert result["name"] == "Nodule"

    def test_invalid_json(self):
        assert _parse_classify_json("not json") is None

    def test_empty_string(self):
        assert _parse_classify_json("") is None

    def test_no_name_field(self):
        assert _parse_classify_json('{"description": "something"}') is None

    def test_empty_name(self):
        assert _parse_classify_json('{"name": ""}') is None

    def test_with_surrounding_text(self):
        text = 'Analysis:\n{"name": "Nodule", "description": "Pulmonary nodule"}\nEnd.'
        result = _parse_classify_json(text)
        assert result is not None
        assert result["name"] == "Nodule"

    def test_multiple_json_picks_valid(self):
        text = '{"unrelated": true} and then {"name": "Effusion", "description": "Pleural effusion"}'
        result = _parse_classify_json(text)
        assert result is not None
        assert result["name"] == "Effusion"

    def test_relevant_true(self):
        result = _parse_classify_json('{"name": "Heart", "description": "enlarged", "relevant": true}')
        assert result is not None
        assert result["relevant"] is True

    def test_relevant_false(self):
        result = _parse_classify_json('{"name": "Lung", "description": "normal", "relevant": false}')
        assert result is not None
        assert result["relevant"] is False

    def test_relevant_string_coercion_true(self):
        result = _parse_classify_json('{"name": "Heart", "relevant": "true"}')
        assert result is not None
        assert result["relevant"] is True

    def test_relevant_string_coercion_false(self):
        result = _parse_classify_json('{"name": "Lung", "relevant": "false"}')
        assert result is not None
        assert result["relevant"] is False

    def test_relevant_missing_defaults_absent(self):
        """When relevant field is absent, it should not be in the dict."""
        result = _parse_classify_json('{"name": "Heart", "description": "enlarged"}')
        assert result is not None
        assert "relevant" not in result

    def test_with_reasoning_prefix(self):
        text = 'The overlay matches the cardiac silhouette region.\n{"name": "Heart", "description": "enlarged", "relevant": true}'
        result = _parse_classify_json(text)
        assert result is not None
        assert result["name"] == "Heart"
        assert result["relevant"] is True


class TestParseConceptMatchJson:
    """Test parsing of concept-level match responses."""

    def test_basic_match(self):
        text = '{"matched": ["heart"], "not_matched": ["nodule"]}'
        result = _parse_concept_match_json(text)
        assert result is not None
        assert result["matched"] == ["heart"]
        assert result["not_matched"] == ["nodule"]

    def test_all_matched(self):
        text = '{"matched": ["heart", "right lung"], "not_matched": []}'
        result = _parse_concept_match_json(text)
        assert result is not None
        assert len(result["matched"]) == 2
        assert len(result["not_matched"]) == 0

    def test_all_unmatched(self):
        text = '{"matched": [], "not_matched": ["nodule", "mass"]}'
        result = _parse_concept_match_json(text)
        assert result is not None
        assert len(result["matched"]) == 0
        assert len(result["not_matched"]) == 2

    def test_with_reasoning_prefix(self):
        text = 'The heart concept matches but nodule does not.\n{"matched": ["heart"], "not_matched": ["nodule"]}'
        result = _parse_concept_match_json(text)
        assert result is not None
        assert result["matched"] == ["heart"]

    def test_empty_string(self):
        assert _parse_concept_match_json("") is None

    def test_invalid_json(self):
        assert _parse_concept_match_json("not json at all") is None

    def test_missing_not_matched_key(self):
        """Both keys must be present."""
        text = '{"matched": ["heart"]}'
        assert _parse_concept_match_json(text) is None

    def test_missing_matched_key(self):
        text = '{"not_matched": ["nodule"]}'
        assert _parse_concept_match_json(text) is None
