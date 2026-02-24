"""Environment-variable configuration for the Segment2Explain pipeline."""

from __future__ import annotations

import os


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# Model IDs
MEDGEMMA_MODEL_ID = os.getenv("MEDGEMMA_MODEL_ID", "google/medgemma-1.5-4b-it")
MEDSAM3_MODEL_ID = os.getenv("MEDSAM3_MODEL_ID", "models/medsam3-merged")

# MedSAM3 auto-segment parameters
MEDSAM3_AUTO_PREPROCESS = _env_bool("MEDSAM3_AUTO_PREPROCESS", True)
MEDSAM3_POINTS_PER_BATCH = int(os.getenv("MEDSAM3_POINTS_PER_BATCH", "64"))
MEDSAM3_PRED_IOU_THRESH = float(os.getenv("MEDSAM3_PRED_IOU_THRESH", "0.8"))
MEDSAM3_STABILITY_SCORE_THRESH = float(os.getenv("MEDSAM3_STABILITY_SCORE_THRESH", "0.9"))
MEDSAM3_CROPS_N_LAYERS = int(os.getenv("MEDSAM3_CROPS_N_LAYERS", "1"))
MEDSAM3_CROPS_NMS_THRESH = float(os.getenv("MEDSAM3_CROPS_NMS_THRESH", "0.7"))
MEDSAM3_NMS_IOU_THRESH = float(os.getenv("MEDSAM3_NMS_IOU_THRESH", "0.5"))
MEDSAM3_MIN_MASK_AREA_RATIO = float(os.getenv("MEDSAM3_MIN_MASK_AREA_RATIO", "0.0002"))
MEDSAM3_MAX_MASK_AREA_RATIO = float(os.getenv("MEDSAM3_MAX_MASK_AREA_RATIO", "0.98"))
MEDSAM3_MAX_MASKS = int(os.getenv("MEDSAM3_MAX_MASKS", "30"))
MEDSAM3_CONCEPT_THRESHOLD = float(os.getenv("MEDSAM3_CONCEPT_THRESHOLD", "0.5"))
MEDSAM3_CONCEPT_MASK_THRESHOLD = float(os.getenv("MEDSAM3_CONCEPT_MASK_THRESHOLD", "0.5"))
MEDSAM3_CONCEPT_MAX_MASKS_PER_CONCEPT = int(os.getenv("MEDSAM3_CONCEPT_MAX_MASKS_PER_CONCEPT", "5"))

# Feature toggles
REFINED_SEG_ENABLED = _env_bool("REFINED_SEG_ENABLED", True)
XGRAMMAR_ENABLED = _env_bool("XGRAMMAR_ENABLED", True)

# Pipeline limits
MAX_REGIONS = int(os.getenv("MAX_REGIONS", "3"))
MAX_VALIDATE_ROUNDS = int(os.getenv("MAX_VALIDATE_ROUNDS", "1"))

# Iterative refinement
MAX_REFINEMENT_ITERATIONS = int(os.getenv("MAX_REFINEMENT_ITERATIONS", "2"))
ITERATIVE_REFINEMENT_ENABLED = _env_bool("ITERATIVE_REFINEMENT_ENABLED", True)

# Parallel (quick) mode
PARALLEL_MODE_ENABLED = _env_bool("PARALLEL_MODE_ENABLED", True)

# Attention-based spatial priors for MedSAM3 (uses MedGemma attention heatmaps)
ATTENTION_PRIOR_ENABLED = _env_bool("ATTENTION_PRIOR_ENABLED", True)
# Mode: "box" (spatial prior box only), "overlay" (alpha channel), "both" (overlay + box)
ATTENTION_PRIOR_MODE = os.getenv("ATTENTION_PRIOR_MODE", "both")
