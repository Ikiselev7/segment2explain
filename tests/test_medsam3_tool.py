"""Tests for MedSAM3Tool and concept-prep helpers."""

from unittest.mock import MagicMock

import numpy as np

from backend.pipeline import _clean_r2_stream, _extract_concepts_from_bullets, _extract_thinking, _parse_concepts_json, _prepare_medsam3_concepts, _strip_medgemma_thinking


class TestParseConceptsJson:
    """Test parsing of MedGemma concepts suggestion responses."""

    def test_valid_array(self):
        result = _parse_concepts_json('["pleural effusion", "cardiomegaly"]')
        assert result == ["pleural effusion", "cardiomegaly"]

    def test_with_surrounding_text(self):
        text = 'Here are concepts:\n["opacity", "nodule", "effusion"]\nDone.'
        result = _parse_concepts_json(text)
        assert result == ["opacity", "nodule", "effusion"]

    def test_empty_string(self):
        assert _parse_concepts_json("") is None

    def test_invalid_json(self):
        assert _parse_concepts_json("not json") is None

    def test_non_string_array(self):
        # Array of non-strings should not match
        assert _parse_concepts_json("[1, 2, 3]") is None

    def test_empty_array(self):
        result = _parse_concepts_json("[]")
        assert result == []

    def test_single_concept(self):
        result = _parse_concepts_json('["nodule"]')
        assert result == ["nodule"]

    def test_code_fence_json(self):
        text = '```json\n["heart", "lung", "rib"]\n```'
        result = _parse_concepts_json(text)
        assert result == ["heart", "lung", "rib"]

    def test_code_fence_multiline(self):
        text = '```json\n[\n  "pleural effusion",\n  "cardiomegaly",\n  "opacity"\n]\n```'
        result = _parse_concepts_json(text)
        assert result == ["pleural effusion", "cardiomegaly", "opacity"]


    def test_thinking_prefix_with_json_object(self):
        """MedGemma may emit <unused94>thought ... before JSON."""
        text = (
            '<unused94>thought\nThe user wants nodules.\n'
            '1. Concepts: nodule, mass\n'
            '<unused94>end_of_thought\n'
            '{"concepts": ["nodule", "mass"]}'
        )
        result = _parse_concepts_json(text)
        assert result == ["nodule", "mass"]

    def test_thinking_prefix_no_json_output(self):
        """If thinking consumes all tokens and no JSON, return None."""
        text = '<unused94>thought\nThe user wants to find...\n1. Something\n2. More thinking'
        result = _parse_concepts_json(text)
        assert result is None

    def test_thinking_prefix_with_bare_array(self):
        text = '<unused94>thought\nThinking...\n<unused94>\n["heart", "cardiomegaly"]'
        result = _parse_concepts_json(text)
        assert result == ["heart", "cardiomegaly"]


class TestExtractConceptsFromBullets:
    """Test fallback bullet-point concept extraction."""

    def test_concepts_section_with_bullets(self):
        """Extract concepts from a CONCEPTS section with bullet points."""
        text = (
            "1. REASON: The user wants a general description.\n"
            "2. HYPOTHESES: Various abnormalities.\n"
            "3. CONCEPTS: I need to segment major structures.\n"
            "   - Heart (cardiac silhouette)\n"
            "   - Right Lung\n"
            "   - Left Lung\n"
            "   - Diaphragm\n"
        )
        result = _extract_concepts_from_bullets(text)
        assert result is not None
        assert "Heart" in result
        assert "Right Lung" in result
        assert "Left Lung" in result
        assert "Diaphragm" in result

    def test_bullets_without_concepts_header(self):
        """Extract bullets even without a CONCEPTS header."""
        text = "- heart\n- right lung\n- left lung"
        result = _extract_concepts_from_bullets(text)
        assert result == ["heart", "right lung", "left lung"]

    def test_strips_parenthetical_annotations(self):
        """Parenthetical annotations like '(cardiac silhouette)' are stripped."""
        text = "CONCEPTS:\n- Heart (cardiac silhouette)\n- Pleura (as a general area)"
        result = _extract_concepts_from_bullets(text)
        assert result is not None
        assert "Heart" in result
        assert "Pleura" in result
        # Parenthetical text should be stripped
        assert not any("cardiac" in c for c in result)

    def test_no_bullets_returns_none(self):
        text = "Just some plain text with no bullet points."
        assert _extract_concepts_from_bullets(text) is None

    def test_empty_string(self):
        assert _extract_concepts_from_bullets("") is None

    def test_long_bullet_items_excluded(self):
        """Bullet items with >5 words are excluded."""
        text = "- heart\n- this is a very long concept that should be skipped entirely"
        result = _extract_concepts_from_bullets(text)
        assert result == ["heart"]

    def test_parse_concepts_json_falls_back_to_bullets(self):
        """_parse_concepts_json uses bullet fallback when no JSON found."""
        text = (
            "1. REASON: General description.\n"
            "3. CONCEPTS:\n"
            "   - Heart\n"
            "   - Right Lung\n"
            "   - Left Lung\n"
        )
        result = _parse_concepts_json(text)
        assert result is not None
        assert "Heart" in result
        assert "Right Lung" in result

    def test_json_preferred_over_bullets(self):
        """When JSON is present, it's used instead of bullets."""
        text = (
            "Some bullets:\n- heart\n- lung\n"
            '{"concepts": ["nodule", "mass"]}'
        )
        result = _parse_concepts_json(text)
        assert result == ["nodule", "mass"]


class TestStripMedgemmaThinking:
    """Tests for _strip_medgemma_thinking helper."""

    def test_strips_thinking_prefix(self):
        text = '<unused94>thought\nSome reasoning.\n<unused94>end_of_thought\n{"concepts": ["x"]}'
        result = _strip_medgemma_thinking(text)
        assert "<unused94>" not in result
        assert '{"concepts": ["x"]}' in result

    def test_no_thinking_tokens(self):
        text = '{"concepts": ["nodule"]}'
        result = _strip_medgemma_thinking(text)
        assert result == '{"concepts": ["nodule"]}'

    def test_empty_string(self):
        assert _strip_medgemma_thinking("") == ""

    def test_only_thinking(self):
        text = '<unused94>thought\nJust thinking here.'
        result = _strip_medgemma_thinking(text)
        assert "unused94" not in result
        assert "Just thinking here." in result


class TestExtractThinking:
    """Tests for _extract_thinking helper."""

    def test_native_thinking_tokens(self):
        text = (
            '<unused94>thought\nCardiomegaly means enlarged heart.\n'
            'end_of_thought\n'
            '{"concepts": ["heart", "cardiac silhouette"]}'
        )
        thinking, cleaned = _extract_thinking(text)
        assert "enlarged heart" in thinking
        assert '{"concepts"' in cleaned
        assert "<unused" not in cleaned

    def test_free_text_before_json(self):
        text = 'Looking for nodule or mass in lung fields.\n{"concepts": ["nodule", "mass"]}'
        thinking, cleaned = _extract_thinking(text)
        assert "nodule or mass" in thinking
        assert '{"concepts"' in cleaned

    def test_both_native_and_free_text(self):
        text = (
            '<unused94>thought\nFirst thought.\n'
            'end_of_thought\n'
            'Second thought.\n'
            '{"concepts": ["heart"]}'
        )
        thinking, cleaned = _extract_thinking(text)
        assert "First thought" in thinking
        assert "Second thought" in thinking
        assert '{"concepts"' in cleaned

    def test_no_thinking(self):
        text = '{"concepts": ["heart"]}'
        thinking, cleaned = _extract_thinking(text)
        assert thinking == ""
        assert '{"concepts"' in cleaned

    def test_empty_input(self):
        thinking, cleaned = _extract_thinking("")
        assert thinking == ""
        assert cleaned == ""


class TestCleanR2Stream:
    """Tests for _clean_r2_stream degeneration detection."""

    def test_clean_text_passes_through(self):
        text = "FINDINGS:\n- Segment A: cardiac silhouette\n\nIMPRESSION: Normal."
        cleaned, stop = _clean_r2_stream(text)
        assert cleaned == text
        assert stop is False

    def test_stops_on_unused_token(self):
        text = "Good analysis here.<unused95>garbage stuff"
        cleaned, stop = _clean_r2_stream(text)
        assert cleaned == "Good analysis here."
        assert stop is True

    def test_stops_on_json_code_block_with_object(self):
        text = 'IMPRESSION: Cardiomegaly.\n\nSafety note: Not a diagnosis.```json { "findings": { "junk" } }'
        cleaned, stop = _clean_r2_stream(text)
        assert "Cardiomegaly" in cleaned
        assert "```json" not in cleaned
        assert stop is True

    def test_stops_on_json_code_block_with_array(self):
        text = 'Safety note: Not a diagnosis.\n```json\n[{"box_2d": [1, 2, 3, 4]}]\n```'
        cleaned, stop = _clean_r2_stream(text)
        assert "Safety note" in cleaned
        assert "box_2d" not in cleaned
        assert stop is True

    def test_stops_on_bare_code_fence_with_json(self):
        text = 'IMPRESSION: Done.\n\n```\n[{"label": "test"}]'
        cleaned, stop = _clean_r2_stream(text)
        assert "IMPRESSION: Done." in cleaned
        assert "label" not in cleaned
        assert stop is True

    def test_empty_string(self):
        cleaned, stop = _clean_r2_stream("")
        assert cleaned == ""
        assert stop is False

    def test_unused_at_very_start(self):
        text = "<unused94>thought something"
        cleaned, stop = _clean_r2_stream(text)
        assert cleaned == ""
        assert stop is True

    def test_strips_trailing_whitespace(self):
        text = "Analysis complete.  \n\n<unused95>"
        cleaned, stop = _clean_r2_stream(text)
        assert cleaned == "Analysis complete."
        assert stop is True


class TestPrepareMedSam3Concepts:
    """Test conversion from raw LLM concepts to MedSAM3-ready prompts."""

    def test_splits_compound_and_removes_generic_terms(self):
        raw = ["Nodule/Mass", "right lower lung zone", "finding"]
        result = _prepare_medsam3_concepts("Where is the nodule/mass in the right lower lung?", raw)

        assert "nodule" in result
        assert "mass" in result
        assert "right lower lung zone" in result
        assert "finding" not in result

    def test_no_fallback_when_concepts_empty(self):
        """When MedGemma provides no concepts, return empty — no heuristic fallback."""
        result = _prepare_medsam3_concepts("Is there cardiomegaly?", None)
        assert result == []

    def test_deduplicates_and_limits_output(self):
        raw = ["heart", "Heart", "heart and lungs", "cardiomegaly", "right lung"]
        result = _prepare_medsam3_concepts("Where is cardiomegaly?", raw, max_concepts=4)

        assert result.count("heart") == 1
        assert len(result) <= 4

    def test_passes_model_concepts_directly(self):
        """MedGemma concepts are passed through without hardcoded expansion."""
        raw = ["cardiomegaly"]
        result = _prepare_medsam3_concepts("any prompt", raw)
        assert result == ["cardiomegaly"]
        # No hardcoded expansion to "heart", "cardiac silhouette" etc.
        assert "heart" not in result


class TestMedSAM3ToolInit:
    """Test MedSAM3Tool initialization (without loading models)."""

    def test_init_sets_defaults(self):
        from tools.medsam3_tool import MedSAM3Tool

        tool = MedSAM3Tool.__new__(MedSAM3Tool)
        tool.checkpoint = "models/medsam3-merged"
        tool.device = "cpu"
        tool._concept_model = None
        tool._concept_processor = None
        assert tool.checkpoint == "models/medsam3-merged"
        assert tool.device == "cpu"

    def test_segment_concepts_returns_list_with_concept(self):
        """Test segment_concepts with mocked model."""
        import torch

        from tools.medsam3_tool import MedSAM3Tool

        tool = MedSAM3Tool.__new__(MedSAM3Tool)
        tool.checkpoint = "models/medsam3-merged"
        tool.device = "cpu"

        # Mock concept model and processor
        mock_model = MagicMock()
        mock_processor = MagicMock()

        # Mock get_vision_features
        mock_model.get_vision_features.return_value = torch.zeros(1, 64, 256)

        # Mock processor call for images
        mock_processor.return_value = {
            "pixel_values": torch.zeros(1, 3, 224, 224),
            "original_sizes": torch.tensor([[100, 100]]),
        }

        # Mock post_process
        mask = torch.zeros(100, 100, dtype=torch.uint8)
        mask[20:60, 20:60] = 1
        mock_processor.post_process_instance_segmentation.return_value = [
            {
                "masks": [mask],
                "scores": [torch.tensor(0.92)],
                "boxes": [torch.tensor([20.0, 20.0, 60.0, 60.0])],
            }
        ]

        tool._concept_model = mock_model
        tool._concept_processor = mock_processor

        img = np.zeros((100, 100, 3), dtype=np.uint8) + 128
        results = tool.segment_concepts(img, ["nodule"])

        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]["concept"] == "nodule"
        assert "mask" in results[0]
        assert "bbox" in results[0]
        assert "score" in results[0]

    def test_segment_concepts_empty_concepts_list(self):
        """Test segment_concepts with empty concepts list."""
        import torch

        from tools.medsam3_tool import MedSAM3Tool

        tool = MedSAM3Tool.__new__(MedSAM3Tool)
        tool.checkpoint = "models/medsam3-merged"
        tool.device = "cpu"

        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_model.get_vision_features.return_value = torch.zeros(1, 64, 256)
        mock_processor.return_value = {
            "pixel_values": torch.zeros(1, 3, 224, 224),
            "original_sizes": torch.tensor([[100, 100]]),
        }

        tool._concept_model = mock_model
        tool._concept_processor = mock_processor

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        results = tool.segment_concepts(img, [])
        assert results == []


class TestMedSAM3QualityHelpers:
    """Test MedSAM3 quality helper behavior."""

    def test_preprocess_medical_image_stretches_grayscale_contrast(self):
        """Grayscale-like RGB input should get percentile contrast stretching."""
        from tools.medsam3_tool import _preprocess_medical_image

        grad = np.tile(np.linspace(100, 140, 128, dtype=np.uint8), (128, 1))
        img = np.stack([grad, grad, grad], axis=-1)

        out = _preprocess_medical_image(img)

        assert out.dtype == np.uint8
        assert out.shape == img.shape
        assert int(out.min()) <= 5
        assert int(out.max()) >= 250
