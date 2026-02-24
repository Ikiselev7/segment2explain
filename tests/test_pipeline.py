"""Tests for unified pipeline behavior (R1 → SEG → F → M → R2)."""

from unittest.mock import MagicMock, patch

import numpy as np

from backend.pipeline import run_job


class TestUnifiedPipeline:
    """Pipeline should always use R1 → refined segmentation → identify → match → R2."""

    def test_pipeline_uses_refined_segment(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8) + 128
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:44, 18:40] = 1

        fake_results = [
            {"mask": mask, "bbox": (18, 20, 39, 43), "score": 0.9, "concept": "nodule",
             "total_score": 0.8, "area_pct": 10.0}
        ]

        responses = iter(
            [
                'Looking for nodule or mass in lung fields.\n{"concepts": ["nodule", "mass", "right lung"]}',
                'The overlay highlights a round opacity in the right lung.\n{"name": "nodule", "description": "pulmonary nodule"}',
                '{"matched": ["nodule", "mass", "right lung"], "not_matched": []}',
                "FINDINGS:\n- Segment A: test finding.",
            ]
        )

        def fake_chat_stream(**_kwargs):
            yield next(responses)

        mock_medgemma = MagicMock()
        mock_medgemma.chat_stream.side_effect = fake_chat_stream
        mock_medgemma.extract_concept_heatmaps.return_value = {}
        mock_medgemma.extract_concept_heatmaps_gradcam.return_value = {}

        with (
            patch("backend.pipeline.refined_segment", return_value=fake_results) as mock_refined,
            patch("backend.pipeline.get_medsam3", return_value=MagicMock()),
            patch("backend.pipeline.get_medgemma", return_value=mock_medgemma),
        ):
            outputs = list(
                run_job(
                    image=img,
                    user_prompt="Where is the nodule/mass?",
                    compare_baseline=False,
                    state=None,
                )
            )

        assert outputs
        assert mock_refined.call_count == 1

        # Check concepts were passed to refined_segment
        call_concepts = mock_refined.call_args.args[1]
        assert "nodule" in call_concepts
        assert "mass" in call_concepts

        final_debug = outputs[-1][4]
        assert final_debug["segmentation_mode"] == "medsam3_refined_segment"

    def test_thinking_stored_in_debug(self):
        """R1 thinking text should be extracted and stored in debug."""
        img = np.zeros((64, 64, 3), dtype=np.uint8) + 128
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:44, 18:40] = 1

        fake_results = [
            {"mask": mask, "bbox": (18, 20, 39, 43), "score": 0.9, "concept": "heart",
             "total_score": 0.8, "area_pct": 10.0}
        ]

        responses = iter(
            [
                'Cardiomegaly means enlarged heart. Need to assess heart size.\n{"concepts": ["heart", "cardiac silhouette"]}',
                'The overlay shows the cardiac silhouette which is enlarged.\n{"name": "heart", "description": "enlarged"}',
                '{"matched": ["heart", "cardiac silhouette"], "not_matched": []}',
                "FINDINGS:\n- Segment A: test finding.",
            ]
        )

        def fake_chat_stream(**_kwargs):
            yield next(responses)

        mock_medgemma = MagicMock()
        mock_medgemma.chat_stream.side_effect = fake_chat_stream
        mock_medgemma.extract_concept_heatmaps_gradcam.return_value = {}

        with (
            patch("backend.pipeline.refined_segment", return_value=fake_results),
            patch("backend.pipeline.get_medsam3", return_value=MagicMock()),
            patch("backend.pipeline.get_medgemma", return_value=mock_medgemma),
        ):
            outputs = list(
                run_job(
                    image=img,
                    user_prompt="Is there cardiomegaly?",
                    compare_baseline=False,
                    state=None,
                )
            )

        assert outputs
        final_debug = outputs[-1][4]
        assert "R1_thinking" in final_debug
        assert "enlarged heart" in final_debug["R1_thinking"]
        # Thinking should appear in chat as a reasoning message
        all_chat = outputs[-1][0]
        reasoning_msgs = [m for m in all_chat if m.get("reasoning")]
        assert len(reasoning_msgs) >= 1
        assert "enlarged heart" in reasoning_msgs[0]["content"]

    def test_validation_stores_validated_fields(self):
        """Identify step should store validated_name and validated_desc on segment."""
        img = np.zeros((64, 64, 3), dtype=np.uint8) + 128
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:44, 18:40] = 1

        fake_results = [
            {"mask": mask, "bbox": (18, 20, 39, 43), "score": 0.9, "concept": "heart",
             "total_score": 0.8, "area_pct": 10.0}
        ]

        responses = iter(
            [
                'Need to assess heart.\n{"concepts": ["heart"]}',
                'The overlay correctly highlights the heart region.\n{"name": "cardiac silhouette", "description": "enlarged heart"}',
                '{"matched": ["heart"], "not_matched": []}',
                "FINDINGS:\n- Segment A: The heart appears enlarged.",
            ]
        )

        def fake_chat_stream(**_kwargs):
            yield next(responses)

        mock_medgemma = MagicMock()
        mock_medgemma.chat_stream.side_effect = fake_chat_stream
        mock_medgemma.extract_concept_heatmaps.return_value = {}
        mock_medgemma.extract_concept_heatmaps_gradcam.return_value = {}

        with (
            patch("backend.pipeline.refined_segment", return_value=fake_results),
            patch("backend.pipeline.get_medsam3", return_value=MagicMock()),
            patch("backend.pipeline.get_medgemma", return_value=mock_medgemma),
        ):
            outputs = list(
                run_job(
                    image=img,
                    user_prompt="Is there cardiomegaly?",
                    compare_baseline=False,
                    state=None,
                )
            )

        assert outputs
        final_debug = outputs[-1][4]
        assert final_debug["classified_count"] == 1
        assert "fallback_image_only" not in final_debug
        seg_data = final_debug["_segment_data"]["A"]
        assert seg_data["label"] == "cardiac silhouette"
        assert seg_data["description"] == "enlarged heart"

    def test_empty_seg_results_triggers_fallback(self):
        """When refined_segment returns empty list, should fall back to image-only."""
        img = np.zeros((64, 64, 3), dtype=np.uint8) + 128

        responses = iter(
            [
                'Checking heart size.\n{"concepts": ["heart"]}',
                "Based on the image, the heart appears normal in size…",
            ]
        )

        def fake_chat_stream(**_kwargs):
            yield next(responses)

        mock_medgemma = MagicMock()
        mock_medgemma.chat_stream.side_effect = fake_chat_stream
        mock_medgemma.extract_concept_heatmaps_gradcam.return_value = {}

        with (
            patch("backend.pipeline.refined_segment", return_value=[]),
            patch("backend.pipeline.get_medsam3", return_value=MagicMock()),
            patch("backend.pipeline.get_medgemma", return_value=mock_medgemma),
        ):
            outputs = list(
                run_job(
                    image=img,
                    user_prompt="Is there cardiomegaly?",
                    compare_baseline=False,
                    state=None,
                )
            )

        assert outputs
        final_debug = outputs[-1][4]
        assert final_debug.get("fallback_image_only") is True

    def test_match_rejects_unmatched_concept(self):
        """When match returns concept as not_matched, segment should be removed → fallback."""
        img = np.zeros((64, 64, 3), dtype=np.uint8) + 128
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:44, 18:40] = 1

        fake_results = [
            {"mask": mask, "bbox": (18, 20, 39, 43), "score": 0.9, "concept": "nodule",
             "total_score": 0.8, "area_pct": 10.0}
        ]

        responses = iter(
            [
                'Looking for nodules.\n{"concepts": ["nodule"]}',
                'This region appears to be normal lung tissue.\n'
                '{"name": "lung parenchyma", "description": "normal lung tissue"}',
                '{"matched": [], "not_matched": ["nodule"]}',
                "Based on the image, no definite nodule is identified.",
            ]
        )

        def fake_chat_stream(**_kwargs):
            yield next(responses)

        mock_medgemma = MagicMock()
        mock_medgemma.chat_stream.side_effect = fake_chat_stream
        mock_medgemma.extract_concept_heatmaps_gradcam.return_value = {}

        with (
            patch("backend.pipeline.refined_segment", return_value=fake_results),
            patch("backend.pipeline.get_medsam3", return_value=MagicMock()),
            patch("backend.pipeline.get_medgemma", return_value=mock_medgemma),
            patch("backend.pipeline.ITERATIVE_REFINEMENT_ENABLED", False),
        ):
            outputs = list(
                run_job(
                    image=img,
                    user_prompt="Where is the nodule?",
                    compare_baseline=False,
                    state=None,
                )
            )

        assert outputs
        final_debug = outputs[-1][4]
        # Segment was rejected (unmatched concept), so fallback to image-only
        assert final_debug.get("fallback_image_only") is True
        assert final_debug["classified_count"] == 0

    def test_identify_reasoning_in_chat(self):
        """Identify reasoning should appear in chat with reasoning=True."""
        img = np.zeros((64, 64, 3), dtype=np.uint8) + 128
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:44, 18:40] = 1

        fake_results = [
            {"mask": mask, "bbox": (18, 20, 39, 43), "score": 0.9, "concept": "heart",
             "total_score": 0.8, "area_pct": 10.0}
        ]

        responses = iter(
            [
                'Assess heart.\n{"concepts": ["heart"]}',
                'The overlay correctly shows the cardiac silhouette in the mediastinum.\n'
                '{"name": "heart", "description": "cardiac silhouette"}',
                '{"matched": ["heart"], "not_matched": []}',
                "FINDINGS:\n- Segment A: Normal heart.",
            ]
        )

        def fake_chat_stream(**_kwargs):
            yield next(responses)

        mock_medgemma = MagicMock()
        mock_medgemma.chat_stream.side_effect = fake_chat_stream
        mock_medgemma.extract_concept_heatmaps.return_value = {}
        mock_medgemma.extract_concept_heatmaps_gradcam.return_value = {}

        with (
            patch("backend.pipeline.refined_segment", return_value=fake_results),
            patch("backend.pipeline.get_medsam3", return_value=MagicMock()),
            patch("backend.pipeline.get_medgemma", return_value=mock_medgemma),
        ):
            outputs = list(
                run_job(
                    image=img,
                    user_prompt="Is there cardiomegaly?",
                    compare_baseline=False,
                    state=None,
                )
            )

        assert outputs
        all_chat = outputs[-1][0]
        # Identify reasoning should be in chat with reasoning flag
        filter_reasoning = [m for m in all_chat if m.get("reasoning") and "F" in m.get("content", "") and "cardiac" in m.get("content", "").lower()]
        assert len(filter_reasoning) >= 1

    def test_describe_query_goes_through_unified_pipeline(self):
        """A describe-style query should use the same R1→SEG→R2 path."""
        img = np.zeros((64, 64, 3), dtype=np.uint8) + 128
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:50, 10:50] = 1

        fake_results = [
            {"mask": mask, "bbox": (10, 10, 49, 49), "score": 0.9, "concept": "heart",
             "total_score": 0.8, "area_pct": 25.0}
        ]

        responses = iter(
            [
                'General description. Segment major structures.\n{"concepts": ["heart", "right lung", "left lung"]}',
                'The overlay shows the heart region correctly.\n{"name": "heart", "description": "cardiac silhouette"}',
                '{"matched": ["heart", "right lung", "left lung"], "not_matched": []}',
                "FINDINGS:\n- Segment A: The heart appears normal.",
            ]
        )

        def fake_chat_stream(**_kwargs):
            yield next(responses)

        mock_medgemma = MagicMock()
        mock_medgemma.chat_stream.side_effect = fake_chat_stream
        mock_medgemma.extract_concept_heatmaps.return_value = {}
        mock_medgemma.extract_concept_heatmaps_gradcam.return_value = {}

        with (
            patch("backend.pipeline.refined_segment", return_value=fake_results) as mock_refined,
            patch("backend.pipeline.get_medsam3", return_value=MagicMock()),
            patch("backend.pipeline.get_medgemma", return_value=mock_medgemma),
        ):
            outputs = list(
                run_job(
                    image=img,
                    user_prompt="Describe this chest X-ray",
                    compare_baseline=False,
                    state=None,
                )
            )

        assert outputs
        assert mock_refined.call_count == 1
        final_debug = outputs[-1][4]
        assert final_debug["segmentation_mode"] == "medsam3_refined_segment"
        assert "intent" not in final_debug


class TestIterativeRefinement:
    """Tests for the iterative re-segmentation loop."""

    def test_iterative_one_round(self):
        """M1 reports unmatched concept → SEG2 fires → F → M2 matches → R2."""
        img = np.zeros((64, 64, 3), dtype=np.uint8) + 128
        mask_heart = np.zeros((64, 64), dtype=np.uint8)
        mask_heart[20:44, 18:40] = 1
        mask_nodule = np.zeros((64, 64), dtype=np.uint8)
        mask_nodule[5:15, 40:55] = 1

        # Initial SEG returns only heart
        initial_results = [
            {"mask": mask_heart, "bbox": (18, 20, 39, 43), "score": 0.9,
             "concept": "heart", "area_pct": 10.0}
        ]
        # SEG2 re-segment returns nodule
        reseg_results = [
            {"mask": mask_nodule, "bbox": (40, 5, 54, 14), "score": 0.8,
             "concept": "nodule", "area_pct": 3.0}
        ]

        responses = iter(
            [
                # R1
                'Heart and nodule.\n{"concepts": ["heart", "nodule"]}',
                # F(A) identify heart
                'This is the cardiac silhouette.\n{"name": "heart", "description": "cardiac silhouette"}',
                # M1: heart matched, nodule not
                '{"matched": ["heart"], "not_matched": ["nodule"]}',
                # F(B) identify nodule (after SEG2)
                'Small round opacity.\n{"name": "nodule", "description": "pulmonary nodule"}',
                # M2: all matched
                '{"matched": ["heart", "nodule"], "not_matched": []}',
                # R2
                "FINDINGS:\n- Segment A: Heart. Segment B: Nodule.",
            ]
        )

        def fake_chat_stream(**_kwargs):
            yield next(responses)

        mock_medgemma = MagicMock()
        mock_medgemma.chat_stream.side_effect = fake_chat_stream
        mock_medgemma.extract_concept_heatmaps.return_value = {}
        mock_medgemma.extract_concept_heatmaps_gradcam.return_value = {}

        with (
            patch("backend.pipeline.refined_segment", return_value=initial_results),
            patch("backend.pipeline.refined_segment_with_negatives", return_value=reseg_results) as mock_reseg,
            patch("backend.pipeline.get_medsam3", return_value=MagicMock()),
            patch("backend.pipeline.get_medgemma", return_value=mock_medgemma),
        ):
            outputs = list(
                run_job(
                    image=img,
                    user_prompt="Show heart and nodule",
                    compare_baseline=False,
                    state=None,
                )
            )

        assert outputs
        final_debug = outputs[-1][4]
        # SEG2 was called
        assert mock_reseg.call_count == 1
        # Negative boxes should include heart bbox
        neg_boxes = mock_reseg.call_args.kwargs.get("negative_boxes") or mock_reseg.call_args[0][3]
        assert len(neg_boxes) >= 1
        # Both segments should survive
        assert final_debug["classified_count"] == 2
        assert "fallback_image_only" not in final_debug
        # Both match rounds recorded
        assert "match_M1" in final_debug
        assert "match_M2" in final_debug

    def test_refinement_disabled_skips_reseg(self):
        """With ITERATIVE_REFINEMENT_ENABLED=False, no SEG2/M2 steps created."""
        img = np.zeros((64, 64, 3), dtype=np.uint8) + 128
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:44, 18:40] = 1

        fake_results = [
            {"mask": mask, "bbox": (18, 20, 39, 43), "score": 0.9, "concept": "heart",
             "area_pct": 10.0}
        ]

        responses = iter(
            [
                'Check heart and lungs.\n{"concepts": ["heart", "right lung"]}',
                'Cardiac silhouette.\n{"name": "heart", "description": "cardiac silhouette"}',
                '{"matched": ["heart"], "not_matched": ["right lung"]}',
                "Based on the image, the heart appears enlarged.",
            ]
        )

        def fake_chat_stream(**_kwargs):
            yield next(responses)

        mock_medgemma = MagicMock()
        mock_medgemma.chat_stream.side_effect = fake_chat_stream
        mock_medgemma.extract_concept_heatmaps.return_value = {}
        mock_medgemma.extract_concept_heatmaps_gradcam.return_value = {}

        with (
            patch("backend.pipeline.refined_segment", return_value=fake_results),
            patch("backend.pipeline.refined_segment_with_negatives") as mock_reseg,
            patch("backend.pipeline.get_medsam3", return_value=MagicMock()),
            patch("backend.pipeline.get_medgemma", return_value=mock_medgemma),
            patch("backend.pipeline.ITERATIVE_REFINEMENT_ENABLED", False),
        ):
            outputs = list(
                run_job(
                    image=img,
                    user_prompt="Is there cardiomegaly?",
                    compare_baseline=False,
                    state=None,
                )
            )

        assert outputs
        # No re-segmentation should have happened
        assert mock_reseg.call_count == 0
        final_debug = outputs[-1][4]
        # Only M1, no M2
        assert "match_M1" in final_debug
        assert "match_M2" not in final_debug

    def test_reseg_empty_results_stops_loop(self):
        """If re-segmentation finds nothing, loop stops gracefully."""
        img = np.zeros((64, 64, 3), dtype=np.uint8) + 128
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:44, 18:40] = 1

        fake_results = [
            {"mask": mask, "bbox": (18, 20, 39, 43), "score": 0.9, "concept": "heart",
             "area_pct": 10.0}
        ]

        responses = iter(
            [
                'Heart and nodule.\n{"concepts": ["heart", "nodule"]}',
                'Cardiac silhouette.\n{"name": "heart", "description": "cardiac silhouette"}',
                '{"matched": ["heart"], "not_matched": ["nodule"]}',
                # Fallback R2 (nodule segment removed, but heart stays)
                "FINDINGS:\n- Segment A: Heart appears normal.",
            ]
        )

        def fake_chat_stream(**_kwargs):
            yield next(responses)

        mock_medgemma = MagicMock()
        mock_medgemma.chat_stream.side_effect = fake_chat_stream
        mock_medgemma.extract_concept_heatmaps.return_value = {}
        mock_medgemma.extract_concept_heatmaps_gradcam.return_value = {}

        with (
            patch("backend.pipeline.refined_segment", return_value=fake_results),
            patch("backend.pipeline.refined_segment_with_negatives", return_value=[]),
            patch("backend.pipeline.get_medsam3", return_value=MagicMock()),
            patch("backend.pipeline.get_medgemma", return_value=mock_medgemma),
        ):
            outputs = list(
                run_job(
                    image=img,
                    user_prompt="Show heart and nodule",
                    compare_baseline=False,
                    state=None,
                )
            )

        assert outputs
        final_debug = outputs[-1][4]
        # Heart segment should survive (it was matched)
        assert final_debug["classified_count"] == 1
        assert "fallback_image_only" not in final_debug

    def test_wrong_segment_removed_immediately_after_match(self):
        """When M1 says concept not_matched, the old wrong segment is removed
        from state before re-segmentation starts (visible in intermediate yields)."""
        img = np.zeros((64, 64, 3), dtype=np.uint8) + 128
        mask_heart = np.zeros((64, 64), dtype=np.uint8)
        mask_heart[20:44, 18:40] = 1
        mask_rib = np.zeros((64, 64), dtype=np.uint8)
        mask_rib[5:15, 40:55] = 1
        mask_nodule = np.zeros((64, 64), dtype=np.uint8)
        mask_nodule[8:18, 42:52] = 1

        # Initial SEG returns heart + rib (rib was supposed to be nodule)
        initial_results = [
            {"mask": mask_heart, "bbox": (18, 20, 39, 43), "score": 0.9,
             "concept": "heart", "area_pct": 10.0},
            {"mask": mask_rib, "bbox": (40, 5, 54, 14), "score": 0.7,
             "concept": "nodule", "area_pct": 3.0},
        ]
        # SEG2 finds the real nodule
        reseg_results = [
            {"mask": mask_nodule, "bbox": (42, 8, 51, 17), "score": 0.8,
             "concept": "nodule", "area_pct": 2.5},
        ]

        responses = iter(
            [
                # R1
                'Heart and nodule.\n{"concepts": ["heart", "nodule"]}',
                # F(A) identify heart
                '{"name": "heart", "description": "cardiac silhouette"}',
                # F(B) identify rib (wrong — was supposed to be nodule)
                '{"name": "rib", "description": "costal bone"}',
                # M1: heart matched, nodule NOT matched
                '{"matched": ["heart"], "not_matched": ["nodule"]}',
                # F(C) identify real nodule (after SEG2)
                '{"name": "nodule", "description": "pulmonary nodule"}',
                # M2: all matched
                '{"matched": ["nodule"], "not_matched": []}',
                # R2
                "FINDINGS:\n- Segment A: Heart. Segment C: Nodule.",
            ]
        )

        def fake_chat_stream(**_kwargs):
            yield next(responses)

        mock_medgemma = MagicMock()
        mock_medgemma.chat_stream.side_effect = fake_chat_stream
        mock_medgemma.extract_concept_heatmaps.return_value = {}
        mock_medgemma.extract_concept_heatmaps_gradcam.return_value = {}

        with (
            patch("backend.pipeline.refined_segment", return_value=initial_results),
            patch("backend.pipeline.refined_segment_with_negatives", return_value=reseg_results),
            patch("backend.pipeline.get_medsam3", return_value=MagicMock()),
            patch("backend.pipeline.get_medgemma", return_value=mock_medgemma),
        ):
            outputs = list(
                run_job(
                    image=img,
                    user_prompt="Show heart and nodule",
                    compare_baseline=False,
                    state=None,
                )
            )

        assert outputs
        final_debug = outputs[-1][4]
        final_meas = outputs[-1][3]

        # Only 2 segments should survive: heart (A) + nodule (C)
        # Rib (B) should have been removed after M1
        assert final_debug["classified_count"] == 2
        assert "A" in final_meas  # heart
        assert "B" not in final_meas  # rib — removed
        assert "C" in final_meas  # nodule from SEG2

        # Verify B was removed in an intermediate yield (not just at end)
        # After M1 yield, B should already be gone from meas
        m1_yields = [
            o for o in outputs
            if "match_M1" in o[4] and "match_M2" not in o[4]
        ]
        if m1_yields:
            m1_meas = m1_yields[-1][3]
            assert "B" not in m1_meas, "Segment B should be removed right after M1"
