"""Tests for generation token-id handling in models/medgemma_torch.py."""

from types import SimpleNamespace

from models.medgemma_torch import MedGemmaTorch


class TestGenerationTokenIds:
    """Test pad/eos token-id resolution for generation kwargs."""

    def test_uses_tokenizer_pad_token_when_available(self):
        model = MedGemmaTorch.__new__(MedGemmaTorch)
        model.processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=1))
        model.model = SimpleNamespace(config=SimpleNamespace(pad_token_id=None, eos_token_id=None))

        pad_id, eos_id = model._get_generation_token_ids()

        assert pad_id == 0
        assert eos_id == 1

    def test_falls_back_to_eos_when_pad_missing(self):
        model = MedGemmaTorch.__new__(MedGemmaTorch)
        model.processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=None, eos_token_id=1))
        model.model = SimpleNamespace(config=SimpleNamespace(pad_token_id=None, eos_token_id=None))

        pad_id, eos_id = model._get_generation_token_ids()

        assert pad_id == 1
        assert eos_id == 1
