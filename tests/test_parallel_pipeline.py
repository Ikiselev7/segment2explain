"""Tests for parallel pipeline (ANSWER → SELECT → SEG → LINK)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.pipeline import (
    _cross_reference_prescan_aliases,
    _dedup_concepts,
    _extract_contextual_aliases,
    _fuzzy_validate_concepts,
    _normalize_concept_entries,
    _normalize_term_to_vocabulary,
    _parse_select_json,
    _prescan_vocab_terms,
    _validate_concept_aliases,
    run_parallel_job,
)


class TestNormalizeTermToVocabulary:
    """Tests for _normalize_term_to_vocabulary helper."""

    VOCAB = [
        "heart", "cardiac silhouette", "left lung", "right lung", "lung",
        "aorta", "pleural effusion", "lung nodule", "nodule", "consolidation",
        "opacity", "mass", "pneumothorax", "edema", "rib fracture",
    ]

    def test_exact_match(self):
        assert _normalize_term_to_vocabulary("heart", self.VOCAB) == "heart"

    def test_exact_match_case_insensitive(self):
        assert _normalize_term_to_vocabulary("Heart", self.VOCAB) == "heart"
        assert _normalize_term_to_vocabulary("PLEURAL EFFUSION", self.VOCAB) == "pleural effusion"

    def test_plural_to_singular(self):
        assert _normalize_term_to_vocabulary("lungs", self.VOCAB) == "lung"

    def test_singular_to_plural(self):
        # "nodule" exists; "nodules" → should match "nodule"
        assert _normalize_term_to_vocabulary("nodules", self.VOCAB) == "nodule"

    def test_containment_longest_match(self):
        # "cardiac" is contained in "cardiac silhouette"
        assert _normalize_term_to_vocabulary("cardiac", self.VOCAB) == "cardiac silhouette"

    def test_containment_in_term(self):
        # "left lung field" contains "left lung"
        assert _normalize_term_to_vocabulary("left lung field", self.VOCAB) == "left lung"

    def test_word_overlap(self):
        # "lung opacity" shares "lung" and "opacity" individually, but "opacity" is exact
        result = _normalize_term_to_vocabulary("pulmonary nodule", self.VOCAB)
        assert result == "nodule"  # "nodule" contained in term

    def test_no_match_returns_original(self):
        assert _normalize_term_to_vocabulary("cardiomegaly", self.VOCAB) == "cardiomegaly"

    def test_empty_term(self):
        assert _normalize_term_to_vocabulary("", self.VOCAB) == ""


class TestNormalizeConceptEntries:
    """Tests for _normalize_concept_entries helper."""

    def test_normalizes_terms_to_vocabulary(self):
        entries = [
            {"term": "lungs", "aliases": ["pulmonary fields"]},
            {"term": "heart", "aliases": []},
        ]
        result = _normalize_concept_entries(entries)
        assert result[0]["term"] == "lung"
        assert "lungs" in result[0]["aliases"]  # original added to aliases
        assert result[1]["term"] == "heart"

    def test_exact_match_no_alias_added(self):
        entries = [{"term": "heart", "aliases": ["cardiac silhouette"]}]
        result = _normalize_concept_entries(entries)
        assert result[0]["term"] == "heart"
        assert result[0]["aliases"] == ["cardiac silhouette"]  # unchanged

    def test_unknown_term_kept_as_is(self):
        entries = [{"term": "cardiomegaly", "aliases": []}]
        result = _normalize_concept_entries(entries)
        assert result[0]["term"] == "cardiomegaly"

    def test_original_term_not_duplicated_in_aliases(self):
        # If original term is already in aliases, don't add again
        entries = [{"term": "lungs", "aliases": ["lungs", "bilateral"]}]
        result = _normalize_concept_entries(entries)
        assert result[0]["aliases"].count("lungs") == 1


class TestValidateConceptAliases:
    """Tests for _validate_concept_aliases helper."""

    def test_keeps_matching_concepts(self):
        concepts = [
            {"term": "heart", "aliases": ["cardiac silhouette"]},
            {"term": "nodule", "aliases": ["round opacity"]},
        ]
        answer = "The heart shows mild cardiomegaly. A round opacity is seen."
        result = _validate_concept_aliases(concepts, answer)
        assert len(result) == 2

    def test_filters_non_matching_concepts(self):
        concepts = [
            {"term": "heart", "aliases": ["cardiac silhouette"]},
            {"term": "pleural effusion", "aliases": ["fluid collection"]},
        ]
        answer = "The heart appears normal. No abnormalities seen."
        result = _validate_concept_aliases(concepts, answer)
        assert len(result) == 1
        assert result[0]["term"] == "heart"

    def test_alias_match_keeps_concept(self):
        concepts = [
            {"term": "cardiac silhouette", "aliases": ["heart"]},
        ]
        answer = "The heart is enlarged."
        result = _validate_concept_aliases(concepts, answer)
        assert len(result) == 1

    def test_filters_aliases_not_in_text(self):
        concepts = [
            {"term": "heart", "aliases": ["cardiac silhouette", "cardiac shadow"]},
        ]
        answer = "The heart and cardiac silhouette are enlarged."
        result = _validate_concept_aliases(concepts, answer)
        assert len(result) == 1
        assert "cardiac silhouette" in result[0]["aliases"]
        assert "cardiac shadow" not in result[0]["aliases"]

    def test_case_insensitive(self):
        concepts = [{"term": "Heart", "aliases": ["Cardiac Silhouette"]}]
        answer = "The heart shows cardiac silhouette enlargement."
        result = _validate_concept_aliases(concepts, answer)
        assert len(result) == 1

    def test_empty_concepts(self):
        result = _validate_concept_aliases([], "some text")
        assert result == []

    def test_empty_answer(self):
        concepts = [{"term": "heart", "aliases": []}]
        result = _validate_concept_aliases(concepts, "")
        assert result == []


class TestPrescanVocabTerms:
    """Tests for _prescan_vocab_terms helper."""

    def test_finds_single_term(self):
        terms = _prescan_vocab_terms("The heart appears enlarged.")
        assert "heart" in terms

    def test_finds_multiple_terms(self):
        terms = _prescan_vocab_terms("The heart and left lung show abnormalities.")
        assert "heart" in terms
        assert "left lung" in terms

    def test_longer_match_preferred(self):
        # "pleural effusion" should match, not just "effusion"
        terms = _prescan_vocab_terms("There is a pleural effusion on the right side.")
        assert "pleural effusion" in terms

    def test_case_insensitive(self):
        terms = _prescan_vocab_terms("HEART is enlarged. The AORTA appears normal.")
        assert "heart" in terms
        assert "aorta" in terms

    def test_no_matches(self):
        terms = _prescan_vocab_terms("The image quality is poor.")
        # May match some common words, but not domain-specific
        # Just verify it returns a list
        assert isinstance(terms, list)

    def test_empty_text(self):
        terms = _prescan_vocab_terms("")
        assert terms == []

    def test_word_boundary_matching(self):
        # "heart" should NOT match inside "sweetheart"
        terms = _prescan_vocab_terms("My sweetheart is fine.")
        assert "heart" not in terms

    def test_multi_word_terms(self):
        terms = _prescan_vocab_terms("The cardiac silhouette is prominent.")
        assert "cardiac silhouette" in terms

    def test_deduplication(self):
        # Same term appearing multiple times should only appear once
        terms = _prescan_vocab_terms("The heart is big. The heart is enlarged.")
        assert terms.count("heart") == 1

    def test_partial_streaming_text(self):
        # Simulates incremental scanning during streaming
        partial = "The heart appears"
        terms1 = _prescan_vocab_terms(partial)
        assert "heart" in terms1

        full = "The heart appears enlarged with pleural effusion."
        terms2 = _prescan_vocab_terms(full)
        assert "heart" in terms2
        assert "pleural effusion" in terms2


class TestExtractContextualAliases:
    """Tests for _extract_contextual_aliases helper."""

    def test_single_word_concept_finds_phrases(self):
        answer = "The heart borders seem prominent along both sides."
        aliases = _extract_contextual_aliases("heart", answer, set())
        alias_lower = [a.lower() for a in aliases]
        assert any("heart borders" in a for a in alias_lower)

    def test_concept_itself_not_in_aliases(self):
        answer = "The heart appears enlarged."
        aliases = _extract_contextual_aliases("heart", answer, set())
        assert "heart" not in [a.lower() for a in aliases]

    def test_skips_other_segment_concept(self):
        answer = "The cardiac silhouette and heart borders are prominent."
        aliases = _extract_contextual_aliases("heart", answer, {"cardiac silhouette"})
        alias_lower = [a.lower() for a in aliases]
        assert "cardiac silhouette" not in alias_lower

    def test_multi_word_concept(self):
        answer = "There is a large pleural effusion on the left side."
        aliases = _extract_contextual_aliases("pleural effusion", answer, set())
        alias_lower = [a.lower() for a in aliases]
        assert any("large pleural effusion" in a for a in alias_lower)

    def test_no_aliases_when_no_extra_context(self):
        answer = "The cardiac silhouette is noted."
        aliases = _extract_contextual_aliases("cardiac silhouette", answer, set())
        # "cardiac silhouette" itself is excluded — should find nearby phrases or nothing
        assert "cardiac silhouette" not in [a.lower() for a in aliases]

    def test_empty_answer(self):
        aliases = _extract_contextual_aliases("heart", "", set())
        assert aliases == []

    def test_short_root_words_ignored(self):
        # Concept "rib" has root "rib" which is only 3 chars — should be skipped
        answer = "The rib cage appears normal."
        aliases = _extract_contextual_aliases("rib", answer, set())
        assert aliases == []

    def test_multiple_occurrences(self):
        answer = "The heart borders are enlarged. The heart shadow is prominent."
        aliases = _extract_contextual_aliases("heart", answer, set())
        alias_lower = [a.lower() for a in aliases]
        assert any("heart borders" in a for a in alias_lower)
        assert any("heart shadow" in a for a in alias_lower)

    def test_skips_comma_list_fragments(self):
        answer = "Bones include ribs, scapulae, sternum, and vertebral bodies."
        aliases = _extract_contextual_aliases("sternum", answer, set())
        # Phrases containing commas (list fragments) should be filtered
        for a in aliases:
            assert "," not in a

    def test_strips_markdown_bold(self):
        answer = "The **left lung** shows opacity in the lower lobe."
        aliases = _extract_contextual_aliases("left lung", answer, set())
        # No markdown markers in aliases
        for a in aliases:
            assert "**" not in a

    def test_skips_phrases_containing_other_concept(self):
        answer = "The cardiac silhouette and heart borders are both enlarged."
        # "heart" should NOT produce aliases containing "cardiac silhouette" (another segment concept)
        aliases = _extract_contextual_aliases("heart", answer, {"cardiac silhouette"})
        alias_lower = [a.lower() for a in aliases]
        for a in alias_lower:
            assert "cardiac silhouette" not in a


class TestCrossReferencePrescanAliases:
    """Tests for _cross_reference_prescan_aliases helper."""

    def test_containment_match(self):
        result = _cross_reference_prescan_aliases(
            prescan_terms=["opacity", "lung opacity"],
            segment_concepts={"A": "lung opacity"},
            answer_text="There is opacity in the lung. Lung opacity is noted.",
        )
        assert "lung opacity" in result
        assert "opacity" in result["lung opacity"]

    def test_no_match_for_unrelated(self):
        result = _cross_reference_prescan_aliases(
            prescan_terms=["consolidation"],
            segment_concepts={"A": "heart"},
            answer_text="There is consolidation and heart enlargement.",
        )
        # "consolidation" has no word overlap with "heart"
        assert "heart" not in result

    def test_word_overlap_match(self):
        result = _cross_reference_prescan_aliases(
            prescan_terms=["left lung"],
            segment_concepts={"A": "lung"},
            answer_text="The left lung and lung fields appear normal.",
        )
        assert "lung" in result
        assert "left lung" in result["lung"]

    def test_skips_used_concepts(self):
        result = _cross_reference_prescan_aliases(
            prescan_terms=["heart", "lung"],
            segment_concepts={"A": "heart", "B": "lung"},
            answer_text="The heart and lung appear normal.",
        )
        # Both are used as segment concepts, nothing unused
        assert result == {}

    def test_skips_terms_not_in_answer(self):
        result = _cross_reference_prescan_aliases(
            prescan_terms=["edema"],
            segment_concepts={"A": "lung"},
            answer_text="The lung appears normal.",
        )
        assert result == {}


class TestParseSelectJson:
    """Tests for _parse_select_json helper."""

    def test_valid_json(self):
        result = _parse_select_json('{"concepts": ["heart", "left lung"]}')
        assert result == ["heart", "left lung"]

    def test_empty_concepts(self):
        result = _parse_select_json('{"concepts": []}')
        assert result == []

    def test_invalid_json(self):
        result = _parse_select_json("not json")
        assert result == []

    def test_empty_string(self):
        result = _parse_select_json("")
        assert result == []

    def test_wrapped_in_markdown(self):
        result = _parse_select_json('```json\n{"concepts": ["heart"]}\n```')
        assert result == ["heart"]

    def test_truncated_json_salvage(self):
        """Should extract concepts from truncated/incomplete JSON."""
        truncated = '{"concepts": ["heart", "left lung", "pleural effusion", "nod'
        result = _parse_select_json(truncated)
        assert "heart" in result
        assert "left lung" in result
        assert "pleural effusion" in result


class TestFuzzyValidateConcepts:
    """Tests for _fuzzy_validate_concepts helper."""

    def test_exact_match(self):
        result = _fuzzy_validate_concepts(
            ["heart", "left lung"],
            "The heart appears enlarged. The left lung is clear.",
        )
        assert "heart" in result
        assert "left lung" in result

    def test_rejects_absent_term(self):
        result = _fuzzy_validate_concepts(
            ["heart", "pneumothorax"],
            "The heart appears enlarged.",
        )
        assert "heart" in result
        assert "pneumothorax" not in result

    def test_fuzzy_root_word(self):
        result = _fuzzy_validate_concepts(
            ["pleural effusion"],
            "There is effusion noted in the pleural space.",
        )
        assert "pleural effusion" in result

    def test_word_boundary(self):
        """Should not match partial words."""
        result = _fuzzy_validate_concepts(
            ["rib"],
            "The terrible finding was described.",
        )
        assert "rib" not in result

    def test_rejects_negated_concept(self):
        """Should reject concepts that only appear after negation words."""
        result = _fuzzy_validate_concepts(
            ["pneumothorax", "heart"],
            "The heart appears enlarged. No pneumothorax is seen.",
        )
        assert "heart" in result
        assert "pneumothorax" not in result

    def test_rejects_no_evidence_of(self):
        result = _fuzzy_validate_concepts(
            ["consolidation"],
            "There is no evidence of consolidation.",
        )
        assert "consolidation" not in result

    def test_keeps_positive_even_with_some_negation(self):
        """If a concept appears both negated and non-negated, keep it."""
        result = _fuzzy_validate_concepts(
            ["effusion"],
            "Previously there was no effusion. Now effusion is present.",
        )
        assert "effusion" in result

    def test_rejects_ruled_out(self):
        result = _fuzzy_validate_concepts(
            ["pneumothorax"],
            "Pneumothorax was ruled out on prior imaging.",
        )
        assert "pneumothorax" not in result

    def test_rejects_unlikely(self):
        result = _fuzzy_validate_concepts(
            ["consolidation"],
            "Consolidation is unlikely.",
        )
        assert "consolidation" not in result

    def test_rejects_absent(self):
        result = _fuzzy_validate_concepts(
            ["pleural effusion"],
            "Pleural effusion is absent bilaterally.",
        )
        assert "pleural effusion" not in result


class TestDedupConcepts:
    """Tests for _dedup_concepts helper."""

    def test_no_duplicates(self):
        result = _dedup_concepts(["heart", "left lung"])
        assert result == ["heart", "left lung"]

    def test_removes_generic_when_specific_exists(self):
        result = _dedup_concepts(["lung nodule", "nodule"])
        assert "lung nodule" in result
        assert "nodule" not in result

    def test_independent_concepts_kept(self):
        result = _dedup_concepts(["heart", "cardiac silhouette"])
        assert "heart" in result
        assert "cardiac silhouette" in result

    def test_lung_containment(self):
        result = _dedup_concepts(["left lung", "lung"])
        assert "left lung" in result
        assert "lung" not in result

    def test_exact_duplicate(self):
        result = _dedup_concepts(["heart", "heart"])
        assert result == ["heart"]


class TestRunParallelJob:
    """Tests for run_parallel_job generator."""

    def _make_mock_medgemma(self, answer_text, select_json):
        """Create a mock MedGemma with chat_stream_with_cache + chat_continue_cached.

        select_json: JSON for the SELECT step (concept selection), e.g. '{"concepts": ["heart"]}'.
        """
        mock = MagicMock()

        def fake_stream_with_cache(**kwargs):
            yield answer_text

        def fake_continue_cached(**kwargs):
            yield select_json

        mock.chat_stream_with_cache.side_effect = fake_stream_with_cache
        mock.chat_continue_cached.side_effect = fake_continue_cached
        mock.json_logits_processor.return_value = None
        mock.extract_concept_heatmaps.return_value = {}
        mock.extract_concept_heatmaps_gradcam.return_value = {}
        return mock

    def test_no_image_returns_error(self):
        outputs = list(run_parallel_job(image=None, user_prompt="test"))
        assert len(outputs) == 1
        chat = outputs[0][0]
        assert any("upload" in str(m.get("content", "")).lower() for m in chat)

    def test_parallel_pipeline_produces_steps_and_segments(self):
        """Full parallel pipeline with mocked models."""
        img = np.zeros((64, 64, 3), dtype=np.uint8) + 128
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:44, 18:40] = 1

        fake_seg_results = [
            {"mask": mask, "bbox": (18, 20, 39, 43), "score": 0.9,
             "concept": "heart", "total_score": 0.8, "area_pct": 10.0},
        ]

        mock_medgemma = self._make_mock_medgemma(
            answer_text="The heart appears mildly enlarged, suggesting cardiomegaly.",
            select_json='{"concepts": ["heart"]}',
        )

        with (
            patch("backend.pipeline.refined_segment", return_value=fake_seg_results),
            patch("backend.pipeline.get_medsam3", return_value=MagicMock()),
            patch("backend.pipeline.get_medgemma", return_value=mock_medgemma),
        ):
            outputs = list(run_parallel_job(
                image=img, user_prompt="Is there cardiomegaly?"
            ))

        assert len(outputs) > 3  # S1 + ANSWER streaming + SEG + CONCEPTS + link

        final_debug = outputs[-1][4]
        assert final_debug.get("segmentation_mode") == "parallel"
        assert "ANSWER_raw" in final_debug
        assert "heart" in final_debug["ANSWER_raw"].lower()
        assert "CONCEPTS_list" in final_debug
        assert "SEG_segment_count" in final_debug

        # Should have concept links
        concept_links = final_debug.get("_concept_links", [])
        assert len(concept_links) > 0
        assert concept_links[0]["concept"] == "heart"

    def test_parallel_pipeline_no_vocab_matches(self):
        """Pipeline should gracefully handle answer with no vocabulary matches."""
        img = np.zeros((64, 64, 3), dtype=np.uint8) + 128

        mock_medgemma = self._make_mock_medgemma(
            answer_text="The image is unremarkable.",
            select_json='{"concepts": []}',
        )

        with (
            patch("backend.pipeline.get_medsam3", return_value=MagicMock()),
            patch("backend.pipeline.get_medgemma", return_value=mock_medgemma),
        ):
            outputs = list(run_parallel_job(
                image=img, user_prompt="Describe this image"
            ))

        assert len(outputs) >= 3
        final_debug = outputs[-1][4]
        assert final_debug.get("CONCEPTS_list") == []

    def test_parallel_pipeline_select_finds_concepts(self):
        """SELECT step should drive segmentation via xgrammar enum."""
        img = np.zeros((64, 64, 3), dtype=np.uint8) + 128
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:30, 10:30] = 1

        mock_medgemma = self._make_mock_medgemma(
            answer_text="The heart appears normal. The left lung is clear.",
            select_json='{"concepts": ["heart", "left lung"]}',
        )

        with (
            patch("backend.pipeline.refined_segment", return_value=[
                {"mask": mask, "bbox": (10, 10, 29, 29), "score": 0.9,
                 "concept": "heart", "total_score": 0.8, "area_pct": 5.0},
            ]),
            patch("backend.pipeline.get_medsam3", return_value=MagicMock()),
            patch("backend.pipeline.get_medgemma", return_value=mock_medgemma),
        ):
            outputs = list(run_parallel_job(
                image=img, user_prompt="Is the heart normal?"
            ))

        final_debug = outputs[-1][4]
        # Pre-scanned terms still recorded for debug
        prescan = final_debug.get("_prescan_terms", [])
        assert "heart" in prescan
        # CONCEPTS_list should have terms (from SELECT step)
        concepts = final_debug.get("CONCEPTS_list", [])
        assert "heart" in concepts
        # SELECT raw should be recorded
        assert "SELECT_raw" in final_debug

    def test_parallel_pipeline_answer_failure(self):
        """If ANSWER step fails, pipeline should return gracefully."""
        img = np.zeros((64, 64, 3), dtype=np.uint8) + 128

        def failing_stream(**kwargs):
            raise RuntimeError("CUDA out of memory")

        mock_medgemma = MagicMock()
        mock_medgemma.chat_stream_with_cache.side_effect = failing_stream
        mock_medgemma.extract_concept_heatmaps_gradcam.return_value = {}

        with (
            patch("backend.pipeline.get_medsam3", return_value=MagicMock()),
            patch("backend.pipeline.get_medgemma", return_value=mock_medgemma),
        ):
            outputs = list(run_parallel_job(
                image=img, user_prompt="test"
            ))

        # Should have yielded something before the error
        assert len(outputs) >= 2
        # Should contain error info
        final_chat = outputs[-1][0]
        assert any("failed" in str(m.get("content", "")).lower() for m in final_chat)

    def test_parallel_pipeline_select_cache_failure_graceful(self):
        """If SELECT step cache fails, pipeline returns early with no concepts."""
        img = np.zeros((64, 64, 3), dtype=np.uint8) + 128

        def fake_stream_with_cache(**kwargs):
            yield "The heart appears enlarged."

        def failing_continue(**kwargs):
            raise RuntimeError("Cache mismatch")

        mock_medgemma = MagicMock()
        mock_medgemma.chat_stream_with_cache.side_effect = fake_stream_with_cache
        mock_medgemma.chat_continue_cached.side_effect = failing_continue
        mock_medgemma.json_logits_processor.return_value = None
        mock_medgemma.extract_concept_heatmaps_gradcam.return_value = {}

        with (
            patch("backend.pipeline.get_medsam3", return_value=MagicMock()),
            patch("backend.pipeline.get_medgemma", return_value=mock_medgemma),
        ):
            outputs = list(run_parallel_job(
                image=img, user_prompt="Is the heart OK?"
            ))

        final_debug = outputs[-1][4]
        # SELECT failed → no concepts → pipeline ends early
        assert final_debug.get("CONCEPTS_list") == []
        assert final_debug.get("SELECT_raw") == ""

    def test_parallel_pipeline_select_before_seg(self):
        """Verify SELECT runs before SEG in the new flow."""
        img = np.zeros((64, 64, 3), dtype=np.uint8) + 128
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:30, 10:30] = 1

        call_order = []

        def fake_stream_with_cache(**kwargs):
            yield "The heart is enlarged."

        def fake_continue_cached(**kwargs):
            call_order.append("SELECT")
            yield '{"concepts": ["heart"]}'

        def fake_refined_segment(*args, **kwargs):
            call_order.append("SEG")
            return [
                {"mask": mask, "bbox": (10, 10, 29, 29), "score": 0.9,
                 "concept": "heart", "total_score": 0.8, "area_pct": 5.0},
            ]

        mock_medgemma = MagicMock()
        mock_medgemma.chat_stream_with_cache.side_effect = fake_stream_with_cache
        mock_medgemma.chat_continue_cached.side_effect = fake_continue_cached
        mock_medgemma.json_logits_processor.return_value = None
        mock_medgemma.extract_concept_heatmaps.return_value = {}
        mock_medgemma.extract_concept_heatmaps_gradcam.return_value = {}

        with (
            patch("backend.pipeline.refined_segment", side_effect=fake_refined_segment),
            patch("backend.pipeline.get_medsam3", return_value=MagicMock()),
            patch("backend.pipeline.get_medgemma", return_value=mock_medgemma),
        ):
            list(run_parallel_job(
                image=img, user_prompt="Is the heart OK?"
            ))

        # SELECT should run before SEG
        assert call_order == ["SELECT", "SEG"]


class TestConceptLinkedMessageSchema:
    """Tests for ConceptLinkedMessage schema."""

    def test_concept_linked_message_serialization(self):
        from backend.schemas import ConceptLinkedMessage

        msg = ConceptLinkedMessage(
            concept="heart",
            segment_id="A",
            aliases=["cardiac silhouette"],
            color="#e6194b",
        )
        data = msg.model_dump()
        assert data["type"] == "concept_linked"
        assert data["concept"] == "heart"
        assert data["segment_id"] == "A"
        assert data["aliases"] == ["cardiac silhouette"]
        assert data["color"] == "#e6194b"

    def test_concept_linked_message_defaults(self):
        from backend.schemas import ConceptLinkedMessage

        msg = ConceptLinkedMessage(concept="lung", segment_id="B")
        data = msg.model_dump()
        assert data["aliases"] == []
        assert data["color"] == "#3b82f6"
