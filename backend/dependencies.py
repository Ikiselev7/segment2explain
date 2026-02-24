"""Lazy model singletons for MedGemma and MedSAM3."""

from __future__ import annotations

import logging

from backend.config import MEDGEMMA_MODEL_ID, MEDSAM3_MODEL_ID
from models.medgemma_torch import MedGemmaTorch
from tools.medsam3_tool import MedSAM3Tool

logger = logging.getLogger(__name__)

_medgemma: MedGemmaTorch | None = None
_medsam3: MedSAM3Tool | None = None


def get_medgemma() -> MedGemmaTorch:
    global _medgemma
    if _medgemma is None:
        logger.info("Initializing MedGemma (model=%s)…", MEDGEMMA_MODEL_ID)
        _medgemma = MedGemmaTorch(model_id=MEDGEMMA_MODEL_ID)
    return _medgemma


def get_medsam3() -> MedSAM3Tool:
    global _medsam3
    if _medsam3 is None:
        logger.info("Initializing MedSAM3 (model=%s)…", MEDSAM3_MODEL_ID)
        _medsam3 = MedSAM3Tool(checkpoint=MEDSAM3_MODEL_ID)
    return _medsam3
