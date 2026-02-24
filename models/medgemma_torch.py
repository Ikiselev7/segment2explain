from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Generator
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, DynamicCache, StoppingCriteria, TextIteratorStreamer

try:
    import xgrammar as xgr
    from xgrammar.contrib.hf import LogitsProcessor as XGRLogitsProcessor

    _XGRAMMAR_AVAILABLE = True
except ImportError:
    _XGRAMMAR_AVAILABLE = False
    xgr = None  # type: ignore[assignment]
    XGRLogitsProcessor = None  # type: ignore[assignment,misc]


logger = logging.getLogger(__name__)


class _StopOnFlag(StoppingCriteria):
    """Stopping criteria that stops generation when a flag is set.

    Used to signal the generate thread to stop early when the consumer
    (pipeline) breaks out of the streaming loop (e.g. on degeneration).
    This allows the thread to finish quickly so the KV cache can be stored.
    """

    def __init__(self) -> None:
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return self._stop.is_set()


def _pick_dtype() -> torch.dtype:
    """
    Pick dtype for MedGemma.  MPS + float16 produces pad-only output
    with MedGemma 1.5 4B, so we default to float32 on MPS.
    Override with MODEL_DTYPE=bfloat16|float16|float32.
    """
    override = (os.getenv("MODEL_DTYPE") or "").lower().strip()
    if override in ("bf16", "bfloat16"):
        return torch.bfloat16
    if override in ("fp16", "float16", "half"):
        return torch.float16
    if override in ("fp32", "float32"):
        return torch.float32

    # float16 on MPS causes MedGemma to emit only <pad> tokens
    return torch.float32


@dataclass
class MedGemmaTorch:
    model_id: str = "google/medgemma-1.5-4b-it"
    device: str | None = None
    dtype: torch.dtype | None = None
    pad_token_id: int | None = None
    eos_token_id: int | None = None
    _grammar_compiler: object | None = field(default=None, init=False, repr=False)
    _compiled_grammars: dict = field(default_factory=dict, init=False, repr=False)
    _last_cache: DynamicCache | None = field(default=None, init=False, repr=False)
    _last_cache_messages: list[dict] | None = field(default=None, init=False, repr=False)
    _last_cache_assistant_text: str | None = field(default=None, init=False, repr=False)
    _last_attention_capture: object | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        if self.dtype is None:
            self.dtype = _pick_dtype()

        logger.info("Loading MedGemma model=%s device=%s dtype=%s", self.model_id, self.device, self.dtype)
        t0 = time.perf_counter()

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            dtype=self.dtype,
        )
        self.model.to(self.device)
        self.model.eval()
        self.pad_token_id, self.eos_token_id = self._get_generation_token_ids()

        dt = time.perf_counter() - t0
        param_count = sum(p.numel() for p in self.model.parameters()) / 1e6
        logger.info("MedGemma loaded in %.1fs (%.0fM params)", dt, param_count)
        logger.debug("MedGemma generation ids: pad_token_id=%s eos_token_id=%s", self.pad_token_id, self.eos_token_id)

        # Initialize xgrammar if available
        if _XGRAMMAR_AVAILABLE:
            self._init_grammar_compiler()

    def _init_grammar_compiler(self) -> None:
        """Initialize xgrammar compiler and pre-compile JSON schemas."""
        try:
            tokenizer = self.processor.tokenizer
            vocab_size = getattr(tokenizer, "vocab_size", len(tokenizer))
            tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
            self._grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
            logger.info("xgrammar compiler initialized (vocab_size=%d)", vocab_size)
        except Exception as e:
            logger.warning("xgrammar init failed, constrained decoding disabled: %s", e)
            self._grammar_compiler = None

    def compile_json_schema(self, schema: dict | str, cache_key: str | None = None) -> object | None:
        """Compile a JSON schema for constrained decoding. Returns compiled grammar or None."""
        if self._grammar_compiler is None:
            return None
        if cache_key and cache_key in self._compiled_grammars:
            return self._compiled_grammars[cache_key]
        try:
            schema_str = json.dumps(schema) if isinstance(schema, dict) else schema
            compiled = self._grammar_compiler.compile_json_schema(schema_str)
            if cache_key:
                self._compiled_grammars[cache_key] = compiled
            logger.debug("xgrammar compiled schema: %s", cache_key or schema_str[:100])
            return compiled
        except Exception as e:
            logger.warning("xgrammar schema compilation failed: %s", e)
            return None

    def json_logits_processor(self, compiled_grammar: object) -> list:
        """Create a fresh LogitsProcessor list for one generate() call."""
        if compiled_grammar is None or XGRLogitsProcessor is None:
            return []
        return [XGRLogitsProcessor(compiled_grammar)]

    def _get_generation_token_ids(self) -> tuple[int | None, int | None]:
        """Resolve generation token IDs and avoid implicit pad->eos fallback warnings."""
        tokenizer = getattr(self.processor, "tokenizer", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)

        model_config = getattr(self.model, "config", None)
        if eos_token_id is None:
            eos_token_id = getattr(model_config, "eos_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(model_config, "pad_token_id", None)
        if pad_token_id is None and eos_token_id is not None:
            pad_token_id = eos_token_id

        return pad_token_id, eos_token_id

    def chat_stream(
        self,
        user_content: str,
        images: list[Image.Image] | None = None,
        system_prompt: str | None = None,
        max_new_tokens: int = 1024,
        do_sample: bool = False,
        temperature: float = 0.2,
        logits_processor: list | None = None,
        repetition_penalty: float | None = None,
        capture_attention: bool = False,
    ) -> Generator[str, None, None]:
        """
        Stream text deltas using TextIteratorStreamer.

        If ``capture_attention`` is True, installs hooks to capture attention
        from generated tokens to image tokens.  After the generator is
        exhausted, the capture is available via ``extract_heatmaps_via_cache()``.
        """
        n_images = len(images) if images else 0
        prompt_preview = user_content[:120].replace("\n", " ")
        logger.info(
            "MedGemma request: images=%d max_tokens=%d system=%s capture_attn=%s prompt='%s…'",
            n_images, max_new_tokens, bool(system_prompt), capture_attention, prompt_preview,
        )
        t0 = time.perf_counter()

        content = []
        if images:
            for im in images:
                content.append({"type": "image", "image": im.convert("RGB")})
        content.append({"type": "text", "text": user_content})

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        messages.append({"role": "user", "content": content})

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
        logger.debug("MedGemma tokenized input: %d tokens", input_len)

        # Setup inline attention capture if requested
        capture = None
        if capture_attention and images:
            capture = self._setup_attention_capture(inputs["input_ids"][0])

        # Move tensors to device and cast pixel_values if present.
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if "pixel_values" in inputs and inputs["pixel_values"].dtype != self.dtype:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.dtype)

        streamer = TextIteratorStreamer(
            tokenizer=self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            streamer=streamer,
            logits_processor=logits_processor,
            repetition_penalty=repetition_penalty,
        )
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        error_holder: list[BaseException | None] = [None]
        generated_output: list[torch.Tensor | None] = [None]

        def _generate() -> None:
            try:
                result = self.model.generate(**gen_kwargs)
                generated_output[0] = result
            except Exception as e:
                error_holder[0] = e
                streamer.end()

        full_response: list[str] = []
        thread = threading.Thread(target=_generate)
        thread.start()
        try:
            for text in streamer:
                if error_holder[0] is not None:
                    raise error_holder[0]
                if text:
                    full_response.append(text)
                    yield text
        finally:
            thread.join(timeout=60)
            if thread.is_alive():
                logger.warning("chat_stream: generate thread still alive after timeout")

            # Store attention capture results (K_image from prefill)
            if capture is not None:
                capture.remove()
                if capture.has_k_image:
                    self._last_attention_capture = capture
                    logger.info(
                        "Attention capture stored: %d layers with K_image",
                        len(capture._k_image),
                    )
                else:
                    capture.clear()

        dt = time.perf_counter() - t0
        response_text = "".join(full_response)
        response_preview = response_text[:200].replace("\n", " ")
        logger.info(
            "MedGemma response: %d chars in %.1fs | '%s…'",
            len(response_text), dt, response_preview,
        )

    def chat_stream_multiturn(
        self,
        messages: list[dict],
        max_new_tokens: int = 1024,
        do_sample: bool = False,
        temperature: float = 0.2,
    ) -> Generator[str, None, None]:
        """
        Stream text deltas from a multi-turn conversation.

        Messages follow the HF chat format with content as list-of-dicts:
          [{"role": "system", "content": [{"type": "text", "text": "..."}]},
           {"role": "user", "content": [{"type": "image", "image": <PIL>}, {"type": "text", "text": "..."}]},
           {"role": "assistant", "content": [{"type": "text", "text": "..."}]},
           ...]
        """
        n_msgs = len(messages)
        logger.info("MedGemma multiturn request: %d messages, max_tokens=%d", n_msgs, max_new_tokens)
        t0 = time.perf_counter()

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
        logger.debug("MedGemma multiturn tokenized input: %d tokens", input_len)

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if "pixel_values" in inputs and inputs["pixel_values"].dtype != self.dtype:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.dtype)

        streamer = TextIteratorStreamer(
            tokenizer=self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            streamer=streamer,
        )
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        error_holder: list[BaseException | None] = [None]

        def _generate() -> None:
            try:
                self.model.generate(**gen_kwargs)
            except Exception as e:
                error_holder[0] = e
                streamer.end()

        full_response: list[str] = []
        thread = threading.Thread(target=_generate)
        thread.start()
        try:
            for text in streamer:
                if error_holder[0] is not None:
                    raise error_holder[0]
                if text:
                    full_response.append(text)
                    yield text
        finally:
            thread.join(timeout=60)
            if thread.is_alive():
                logger.warning("chat_stream_multiturn: generate thread still alive after timeout")

        dt = time.perf_counter() - t0
        response_text = "".join(full_response)
        response_preview = response_text[:200].replace("\n", " ")
        logger.info(
            "MedGemma multiturn response: %d chars in %.1fs | '%s…'",
            len(response_text), dt, response_preview,
        )

    def chat_stream_with_cache(
        self,
        user_content: str,
        images: list[Image.Image] | None = None,
        system_prompt: str | None = None,
        max_new_tokens: int = 1024,
        do_sample: bool = False,
        temperature: float = 0.2,
        repetition_penalty: float | None = None,
        capture_attention: bool = False,
    ) -> Generator[str, None, None]:
        """Stream text deltas and build a KV cache for continuation.

        Same as chat_stream() but creates a DynamicCache that is mutated
        in-place during generate(). After completion the cache and original
        messages are stored as ``_last_cache`` and ``_last_cache_messages``
        for use by ``chat_continue_cached()``.

        If ``capture_attention`` is True, also captures attention for later
        heatmap extraction via ``extract_heatmaps_via_cache()``.
        """
        n_images = len(images) if images else 0
        prompt_preview = user_content[:120].replace("\n", " ")
        logger.info(
            "MedGemma cached request: images=%d max_tokens=%d system=%s capture_attn=%s prompt='%s…'",
            n_images, max_new_tokens, bool(system_prompt), capture_attention, prompt_preview,
        )
        t0 = time.perf_counter()

        content: list[dict] = []
        if images:
            for im in images:
                content.append({"type": "image", "image": im.convert("RGB")})
        content.append({"type": "text", "text": user_content})

        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        messages.append({"role": "user", "content": content})

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
        logger.debug("MedGemma cached tokenized input: %d tokens", input_len)

        # Setup inline attention capture if requested
        capture = None
        if capture_attention and images:
            capture = self._setup_attention_capture(inputs["input_ids"][0])

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if "pixel_values" in inputs and inputs["pixel_values"].dtype != self.dtype:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.dtype)

        streamer = TextIteratorStreamer(
            tokenizer=self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        cache = DynamicCache()
        stop_flag = _StopOnFlag()

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            streamer=streamer,
            repetition_penalty=repetition_penalty,
            past_key_values=cache,
            stopping_criteria=[stop_flag],
        )
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        error_holder: list[BaseException | None] = [None]
        generated_output: list[torch.Tensor | None] = [None]

        def _generate() -> None:
            try:
                result = self.model.generate(**gen_kwargs)
                generated_output[0] = result
            except Exception as e:
                error_holder[0] = e
                streamer.end()

        full_response: list[str] = []
        thread = threading.Thread(target=_generate)
        thread.start()
        try:
            for text in streamer:
                if error_holder[0] is not None:
                    raise error_holder[0]
                if text:
                    full_response.append(text)
                    yield text
        finally:
            # Signal the generate thread to stop (e.g. consumer broke on degeneration).
            # This lets the thread finish quickly so we can safely store the cache.
            stop_flag.stop()
            thread.join(timeout=30)
            thread_alive = thread.is_alive()
            if thread_alive:
                logger.warning("chat_stream_with_cache: generate thread still alive after stop signal + 30s")

            # Only store cache if the generation thread has finished.
            seq_len = cache.get_seq_length() if hasattr(cache, "get_seq_length") else 0
            if seq_len > 0 and not thread_alive:
                self._last_cache = cache
                self._last_cache_messages = messages
                logger.info("chat_stream_with_cache: cache stored, seq_len=%d", seq_len)
            elif thread_alive:
                logger.warning("chat_stream_with_cache: cache NOT stored (thread still running)")

            # Store attention capture results (K_image from prefill)
            if capture is not None:
                capture.remove()
                if capture.has_k_image and not thread_alive:
                    self._last_attention_capture = capture
                    logger.info(
                        "Attention capture stored: %d layers with K_image",
                        len(capture._k_image),
                    )
                else:
                    capture.clear()

            dt = time.perf_counter() - t0
            response_text = "".join(full_response)
            # Store assistant text for cache continuation (used by extract_heatmaps_via_cache)
            self._last_cache_assistant_text = response_text
            if error_holder[0]:
                logger.warning(
                    "MedGemma cached response (error): %d chars in %.1fs, err=%s",
                    len(response_text), dt, error_holder[0],
                )
            else:
                logger.info(
                    "MedGemma cached response: %d chars in %.1fs, cache_seq_len=%d | '%s…'",
                    len(response_text), dt, seq_len, response_text[:200].replace("\n", " "),
                )

    def chat_continue_cached(
        self,
        user_content: str,
        assistant_response: str,
        max_new_tokens: int = 500,
        do_sample: bool = False,
        temperature: float = 0.2,
        logits_processor: list | None = None,
        repetition_penalty: float | None = None,
    ) -> Generator[str, None, None]:
        """Continue a conversation using the KV cache from chat_stream_with_cache().

        Uses two tokenizations to isolate only the truly new tokens:
        1. base_messages (system + user + assistant response) — matches what the cache covers
        2. full_messages (base + new user instruction + gen prompt)
        new_ids = full_tokens[base_len:] — only the new user message + gen prompt

        Falls back to a regular (uncached) chat_stream() call if the cache is
        unavailable or the token alignment fails.
        """
        cache = self._last_cache
        prev_messages = self._last_cache_messages

        if cache is None or prev_messages is None:
            logger.warning("chat_continue_cached: no cache available, falling back to uncached")
            yield from self.chat_stream(
                user_content=user_content, images=None, system_prompt=None,
                max_new_tokens=max_new_tokens, do_sample=do_sample,
                logits_processor=logits_processor,
            )
            return

        cache_len = cache.get_seq_length()
        logger.info(
            "MedGemma continue cached: cache_seq_len=%d new_prompt='%s…'",
            cache_len, user_content[:120].replace("\n", " "),
        )
        t0 = time.perf_counter()

        # Build base messages: everything the cache already covers
        # (system + user with image + assistant response)
        base_messages = list(prev_messages)
        base_messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_response}]})

        # Build full messages: base + new user instruction
        full_messages = list(base_messages)
        full_messages.append({"role": "user", "content": [{"type": "text", "text": user_content}]})

        try:
            # Tokenize base (no gen prompt — assistant is last speaker)
            base_inputs = self.processor.apply_chat_template(
                base_messages,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            base_len = base_inputs["input_ids"].shape[-1]

            # Tokenize full (with gen prompt — model will generate)
            full_inputs = self.processor.apply_chat_template(
                full_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            full_ids = full_inputs["input_ids"].to(self.device)

            # Extract only truly new tokens (new user turn + gen prompt)
            new_ids = full_ids[:, base_len:]

            if new_ids.shape[-1] == 0:
                logger.warning("chat_continue_cached: no new tokens after slicing (base=%d, full=%d), falling back", base_len, full_ids.shape[-1])
                self._last_cache = None
                yield from self.chat_stream(
                    user_content=user_content, images=None, system_prompt=None,
                    max_new_tokens=max_new_tokens, do_sample=do_sample,
                    logits_processor=logits_processor,
                )
                return

            # Attention mask covers cache positions + new token positions
            n_new = new_ids.shape[-1]
            attn_mask = torch.ones(1, cache_len + n_new, device=self.device, dtype=torch.long)

            # Explicit cache_position: new tokens occupy positions [cache_len, cache_len+n_new)
            cache_position = torch.arange(cache_len, cache_len + n_new, device=self.device)

            logger.info(
                "MedGemma continue: base_text=%d full_text=%d new=%d cached=%d cache_pos=[%d,%d)",
                base_len, full_ids.shape[-1], n_new, cache_len,
                cache_len, cache_len + n_new,
            )

            streamer = TextIteratorStreamer(
                tokenizer=self.processor.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            gen_kwargs = dict(
                input_ids=new_ids,
                attention_mask=attn_mask,
                past_key_values=cache,
                cache_position=cache_position,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
                streamer=streamer,
                logits_processor=logits_processor,
                repetition_penalty=repetition_penalty,
            )
            gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

            error_holder: list[BaseException | None] = [None]

            def _generate() -> None:
                try:
                    self.model.generate(**gen_kwargs)
                except Exception as e:
                    error_holder[0] = e
                    streamer.end()

            full_response: list[str] = []
            thread = threading.Thread(target=_generate)
            thread.start()
            try:
                for text in streamer:
                    if error_holder[0] is not None:
                        raise error_holder[0]
                    if text:
                        full_response.append(text)
                        yield text
            finally:
                thread.join(timeout=60)
                if thread.is_alive():
                    logger.warning("chat_continue_cached: generate thread still alive after timeout")

            dt = time.perf_counter() - t0
            response_text = "".join(full_response)
            logger.info(
                "MedGemma continue response: %d chars in %.1fs | '%s…'",
                len(response_text), dt, response_text[:200].replace("\n", " "),
            )

        except Exception as e:
            logger.warning("chat_continue_cached failed: %s, falling back to uncached", e)
            self._last_cache = None
            yield from self.chat_stream(
                user_content=user_content, images=None, system_prompt=None,
                max_new_tokens=max_new_tokens, do_sample=do_sample,
                logits_processor=logits_processor,
            )

        # Keep cache alive — extract_heatmaps_via_cache() may need it.
        # Pipeline clears it explicitly via invalidate_cache() when done.

    def invalidate_cache(self) -> None:
        """Explicitly invalidate the KV cache and associated state.

        Called by the pipeline after PRIOR step (or when cache is no longer needed).
        """
        self._last_cache = None
        self._last_cache_messages = None
        self._last_cache_assistant_text = None
        if self._last_attention_capture is not None:
            self._last_attention_capture.clear()
            self._last_attention_capture = None

    def extract_concept_heatmaps(
        self,
        image: Image.Image,
        concepts: list[str],
        layer_fraction: float = 0.5,
    ) -> dict[str, np.ndarray]:
        """Extract attention-based spatial heatmaps for medical concepts.

        Performs a single forward pass with hooks on attention layers to capture
        how concept text tokens attend to image tokens. Memory-efficient: scores
        are accumulated per-layer and tensors freed immediately.

        Uses pre-RoPE Q,K for content-based (semantic) attention — appropriate
        for spatial prior extraction where we want "what" not "where in sequence".

        Args:
            image: PIL Image to analyze.
            concepts: Concept strings to localize (e.g. ["cardiac silhouette", "left lung"]).
            layer_fraction: Fraction of later layers to use (0.5 = last half).

        Returns:
            dict mapping concept name → 2D heatmap (n_patches_h, n_patches_w).
            Values in [0, 1], higher = more attention.
        """
        from models.attention_prior import (
            AttentionAccumulator,
            find_concept_token_positions,
            find_image_token_positions,
        )

        if not concepts:
            return {}

        t0 = time.perf_counter()

        # Build prompt containing all concepts
        concept_list = ", ".join(concepts)
        prompt = f"Locate these structures in the image: {concept_list}"

        messages: list[dict] = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image.convert("RGB")},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_ids_flat = inputs["input_ids"][0]
        input_ids_list = input_ids_flat.tolist()

        # Find image token positions
        image_positions = find_image_token_positions(self.processor, input_ids_flat)
        if not image_positions:
            logger.warning("extract_concept_heatmaps: no image tokens found")
            return {}

        # Find concept token positions for each concept
        concept_token_positions: dict[str, list[int]] = {}
        for concept in concepts:
            positions = find_concept_token_positions(
                self.processor.tokenizer, input_ids_list, concept,
            )
            if positions:
                concept_token_positions[concept] = positions
            else:
                logger.debug("extract_concept_heatmaps: tokens not found for '%s'", concept)

        if not concept_token_positions:
            logger.warning("extract_concept_heatmaps: no concept tokens located")
            return {}

        logger.info(
            "extract_concept_heatmaps: %d image tokens, concept tokens: %s",
            len(image_positions),
            {c: len(p) for c, p in concept_token_positions.items()},
        )

        # Determine which layers to hook (last fraction)
        n_layers = len(self.model.model.language_model.layers)
        start_layer = int(n_layers * (1 - layer_fraction))

        # Setup accumulator and hooks
        accumulator = AttentionAccumulator(image_positions, concept_token_positions)
        accumulator.register(self.model.model.language_model, start_layer, n_layers)

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if "pixel_values" in inputs and inputs["pixel_values"].dtype != self.dtype:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.dtype)

        try:
            with torch.no_grad():
                self.model(**inputs)
        finally:
            accumulator.remove()

        # Convert accumulated scores to 2D heatmaps
        n_image = len(image_positions)
        n_side = int(n_image**0.5)
        if n_side * n_side != n_image:
            # Try to find closest grid dimensions
            for h in range(n_side, 0, -1):
                if n_image % h == 0:
                    n_patches_h, n_patches_w = h, n_image // h
                    break
            else:
                n_patches_h = n_patches_w = n_side
        else:
            n_patches_h = n_patches_w = n_side

        heatmaps = accumulator.get_heatmaps(n_patches_h, n_patches_w)

        dt = time.perf_counter() - t0
        logger.info(
            "extract_concept_heatmaps: %d/%d heatmaps in %.1fs (layers %d-%d of %d)",
            len(heatmaps), len(concepts), dt, start_layer, n_layers, n_layers,
        )

        return heatmaps

    def extract_concept_heatmaps_gradcam(
        self,
        image: Image.Image,
        concepts: list[str],
    ) -> dict[str, np.ndarray]:
        """GradCAM-based spatial heatmaps for medical concepts.

        Performs a forward pass through the full model (including ViT), then
        backpropagates from concept token hidden states to image token
        embeddings.  The gradient × activation product (classic GradCAM)
        gives per-patch attribution — which image regions contribute most
        to each concept's representation.

        More principled than raw attention: captures the cumulative effect
        across ALL transformer layers, not just per-layer attention weights.

        Args:
            image: PIL Image to analyze.
            concepts: Concept strings to localize.

        Returns:
            dict mapping concept → 2D heatmap in [0, 1].
        """
        from models.attention_prior import (
            find_concept_token_positions,
            find_image_token_positions,
        )

        if not concepts:
            return {}

        t0 = time.perf_counter()

        # Build prompt (same as extract_concept_heatmaps)
        concept_list = ", ".join(concepts)
        prompt = f"Locate these structures in the image: {concept_list}"

        messages: list[dict] = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image.convert("RGB")},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_ids_flat = inputs["input_ids"][0]
        input_ids_list = input_ids_flat.tolist()

        # Find positions
        image_positions = find_image_token_positions(self.processor, input_ids_flat)
        if not image_positions:
            logger.warning("gradcam: no image tokens found")
            return {}

        concept_token_positions: dict[str, list[int]] = {}
        for concept in concepts:
            pos = find_concept_token_positions(
                self.processor.tokenizer, input_ids_list, concept,
            )
            if pos:
                concept_token_positions[concept] = pos
            else:
                logger.debug("gradcam: tokens not found for '%s'", concept)

        if not concept_token_positions:
            logger.warning("gradcam: no concept tokens located")
            return {}

        logger.info(
            "gradcam: %d image tokens, concepts: %s",
            len(image_positions),
            {c: len(p) for c, p in concept_token_positions.items()},
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if "pixel_values" in inputs and inputs["pixel_values"].dtype != self.dtype:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.dtype)

        # ---- GradCAM: forward + backward ----
        # Disable parameter gradients — only track image embeddings.
        # This reduces backward memory from ~full model to ~computation path only.
        param_requires_grad: dict[str, bool] = {}
        for name, p in self.model.named_parameters():
            param_requires_grad[name] = p.requires_grad
            p.requires_grad_(False)

        # Hook the language model input to make image embeddings require grad
        image_embed_holder: list[torch.Tensor | None] = [None]
        language_model = self.model.model.language_model

        def _pre_hook(_module, args, kwargs):
            embeds = kwargs.get("inputs_embeds")
            if embeds is not None and image_embed_holder[0] is None:
                img_part = embeds[:, image_positions, :].detach().clone().requires_grad_(True)
                new_embeds = embeds.detach().clone()
                new_embeds[:, image_positions, :] = img_part
                kwargs["inputs_embeds"] = new_embeds
                image_embed_holder[0] = img_part
            return args, kwargs

        hook_handle = language_model.register_forward_pre_hook(
            _pre_hook, with_kwargs=True,
        )

        try:
            with torch.enable_grad():
                outputs = self.model(**inputs, output_hidden_states=True)

            image_embeds = image_embed_holder[0]
            if image_embeds is None:
                logger.warning("gradcam: image embeddings not captured by hook")
                return {}

            # Last hidden state: (B, S, D)
            hidden_states = outputs.hidden_states[-1]

            # Grid dimensions
            n_image = len(image_positions)
            n_side = int(n_image**0.5)
            if n_side * n_side != n_image:
                for h_dim in range(n_side, 0, -1):
                    if n_image % h_dim == 0:
                        n_patches_h, n_patches_w = h_dim, n_image // h_dim
                        break
                else:
                    n_patches_h = n_patches_w = n_side
            else:
                n_patches_h = n_patches_w = n_side

            n_expected = n_patches_h * n_patches_w
            heatmaps: dict[str, np.ndarray] = {}

            for concept, positions in concept_token_positions.items():
                # Target: sum of concept hidden states (scalar)
                concept_hidden = hidden_states[0, positions, :]  # (n_concept, D)
                target = concept_hidden.sum()

                # Gradient: ∂target / ∂image_embeds
                grads = torch.autograd.grad(
                    target, image_embeds, retain_graph=True,
                )[0]  # (1, n_image, D)

                # GradCAM: gradient × activation, sum across features, ReLU
                cam = (grads[0] * image_embeds[0].detach()).sum(dim=-1)  # (n_image,)
                cam = torch.relu(cam).cpu().float()

                # Normalize to [0, 1]
                vmin, vmax = cam.min(), cam.max()
                cam = (cam - vmin) / (vmax - vmin + 1e-8)

                if len(cam) == n_expected:
                    heatmaps[concept] = cam.numpy().reshape(n_patches_h, n_patches_w)
                else:
                    logger.warning(
                        "gradcam: size mismatch %d vs %dx%d for '%s'",
                        len(cam), n_patches_h, n_patches_w, concept,
                    )

            dt = time.perf_counter() - t0
            logger.info(
                "gradcam: %d/%d heatmaps in %.1fs (grid %dx%d)",
                len(heatmaps), len(concepts), dt,
                n_patches_h, n_patches_w,
            )

            return heatmaps

        except Exception as e:
            logger.warning("gradcam failed: %s", e)
            return {}

        finally:
            hook_handle.remove()
            image_embed_holder[0] = None
            # Restore parameter grad state
            for name, p in self.model.named_parameters():
                if name in param_requires_grad:
                    p.requires_grad_(param_requires_grad[name])

    # ------------------------------------------------------------------
    # Inline attention capture (during generation, no extra forward pass)
    # ------------------------------------------------------------------

    def _setup_attention_capture(self, input_ids: torch.Tensor, layer_fraction: float = 0.5):
        """Create and register a GenerationAttentionCapture on the model.

        Called internally by chat_stream / chat_stream_with_cache when
        ``capture_attention=True``.  Returns the capture object or None.
        """
        from models.attention_prior import (
            GenerationAttentionCapture,
            find_image_token_positions,
        )

        image_positions = find_image_token_positions(self.processor, input_ids)
        if not image_positions:
            logger.warning("_setup_attention_capture: no image tokens found")
            return None

        n_layers = len(self.model.model.language_model.layers)
        start_layer = int(n_layers * (1 - layer_fraction))

        capture = GenerationAttentionCapture(image_positions)
        capture.register(self.model.model.language_model, start_layer, n_layers)
        logger.info(
            "Attention capture registered: %d image tokens, layers %d-%d of %d",
            len(image_positions), start_layer, n_layers, n_layers,
        )
        return capture

    def extract_heatmaps_via_cache(
        self,
        concepts: list[str],
    ) -> dict[str, np.ndarray]:
        """Extract heatmaps using stored K_image + KV cache locate-prompt pass.

        After a generation with ``capture_attention=True``, the prefill phase
        stores pre-RoPE K at image positions.  This method:

        1. Builds a "Locate these structures: X, Y, Z" prompt
        2. Tokenizes it as a standalone user turn
        3. Processes the new tokens through the model using the KV cache
           (no ViT re-encoding — image is already represented in the cache)
        4. Hooks q_proj to capture pre-RoPE Q at concept token positions
        5. Computes Q_concept · K_image per layer → heatmaps

        Both Q and K are pre-RoPE, giving content-based (semantic) attention.

        Returns:
            dict mapping concept → 2D numpy heatmap in [0, 1], or empty dict.
        """
        from models.attention_prior import find_concept_token_positions

        capture = self._last_attention_capture
        cache = self._last_cache

        if capture is None or not capture.has_k_image:
            logger.debug("extract_heatmaps_via_cache: no K_image available")
            return {}
        if cache is None:
            logger.debug("extract_heatmaps_via_cache: no KV cache available")
            return {}
        if not concepts:
            return {}

        t0 = time.perf_counter()

        n_image = len(capture.image_positions)
        start_layer, end_layer = capture.layer_range

        # ---- Build locate prompt and tokenize as continuation ----
        concept_list = ", ".join(concepts)
        locate_text = f"Locate these structures in the image: {concept_list}"

        # Tokenize as a standalone user turn to get the new tokens.
        # We use a minimal conversation with just this user turn to get
        # properly formatted tokens (with turn markers).
        locate_messages: list[dict] = [
            {"role": "user", "content": [{"type": "text", "text": locate_text}]},
        ]

        try:
            locate_inputs = self.processor.apply_chat_template(
                locate_messages,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            new_ids = locate_inputs["input_ids"]
            n_new = new_ids.shape[-1]
            if n_new == 0:
                logger.warning("extract_heatmaps_via_cache: no tokens from locate prompt")
                return {}

            # Find concept token positions within locate prompt tokens
            new_ids_list = new_ids[0].tolist()
            concept_token_positions: dict[str, list[int]] = {}
            for concept in concepts:
                positions = find_concept_token_positions(
                    self.processor.tokenizer, new_ids_list, concept,
                )
                if positions:
                    concept_token_positions[concept] = positions
                else:
                    logger.debug("extract_heatmaps_via_cache: tokens not found for '%s'", concept)

            if not concept_token_positions:
                logger.warning("extract_heatmaps_via_cache: no concept tokens located")
                return {}

            logger.info(
                "extract_heatmaps_via_cache: %d new tokens, concepts: %s",
                n_new, {c: len(p) for c, p in concept_token_positions.items()},
            )

            # ---- Hook q_proj, forward the locate tokens with KV cache ----
            q_buffers: dict[int, torch.Tensor] = {}
            q_hooks: list[torch.utils.hooks.RemovableHook] = []
            language_model = self.model.model.language_model

            for layer_idx in range(start_layer, end_layer):
                attn = language_model.layers[layer_idx].self_attn

                def _make_q_hook(lidx: int):
                    def hook(_module, _input, output):
                        q_buffers[lidx] = output.detach()
                    return hook

                q_hooks.append(attn.q_proj.register_forward_hook(_make_q_hook(layer_idx)))

            # Forward pass: new tokens only, using existing KV cache for context
            cache_len = cache.get_seq_length()
            attn_mask = torch.ones(1, cache_len + n_new, device=self.device, dtype=torch.long)
            cache_position = torch.arange(cache_len, cache_len + n_new, device=self.device)

            new_ids_device = new_ids.to(self.device)

            with torch.no_grad():
                self.model(
                    input_ids=new_ids_device,
                    attention_mask=attn_mask,
                    past_key_values=cache,
                    cache_position=cache_position,
                )

            # Remove hooks
            for h in q_hooks:
                h.remove()

            # ---- Compute Q_concept · K_image per layer ----
            scores: dict[str, torch.Tensor | None] = {c: None for c in concept_token_positions}
            n_layers_accumulated: dict[str, int] = {c: 0 for c in concept_token_positions}

            for layer_idx in range(start_layer, end_layer):
                q_raw = q_buffers.get(layer_idx)
                k_img_cpu = capture._k_image.get(layer_idx)
                if q_raw is None or k_img_cpu is None:
                    continue

                attn_mod = language_model.layers[layer_idx].self_attn
                nh = attn_mod.config.num_attention_heads
                nkv = attn_mod.config.num_key_value_heads
                hd = attn_mod.config.head_dim

                B, S, _ = q_raw.shape
                q = q_raw.view(B, S, nh, hd).transpose(1, 2)  # (B, nh, S, hd)

                k_img = k_img_cpu.to(q.device)
                if nh != nkv:
                    k_img = k_img.repeat_interleave(nh // nkv, dim=1)

                for concept, q_positions in concept_token_positions.items():
                    q_concept = q[:, :, q_positions, :]  # (B, nh, n_concept, hd)
                    attn_scores = torch.matmul(q_concept, k_img.transpose(-1, -2))
                    attn_scores = attn_scores / (hd ** 0.5)
                    attn_scores = torch.softmax(attn_scores, dim=-1)
                    avg = attn_scores.mean(dim=(0, 1, 2)).cpu().float()  # (n_img,)

                    if scores[concept] is None:
                        scores[concept] = avg
                    else:
                        scores[concept] += avg
                    n_layers_accumulated[concept] += 1

                del q, k_img

            q_buffers.clear()

            # ---- Build heatmaps ----
            n_side = int(n_image**0.5)
            if n_side * n_side != n_image:
                for h_dim in range(n_side, 0, -1):
                    if n_image % h_dim == 0:
                        n_patches_h, n_patches_w = h_dim, n_image // h_dim
                        break
                else:
                    n_patches_h = n_patches_w = n_side
            else:
                n_patches_h = n_patches_w = n_side

            n_expected = n_patches_h * n_patches_w
            heatmaps: dict[str, np.ndarray] = {}

            for concept, raw_scores in scores.items():
                if raw_scores is None or n_layers_accumulated[concept] == 0:
                    continue
                avg = raw_scores / n_layers_accumulated[concept]
                vmin, vmax = avg.min(), avg.max()
                avg = (avg - vmin) / (vmax - vmin + 1e-8)
                if len(avg) == n_expected:
                    heatmaps[concept] = avg.numpy().reshape(n_patches_h, n_patches_w)
                else:
                    logger.warning(
                        "extract_heatmaps_via_cache: size mismatch %d vs %dx%d for '%s'",
                        len(avg), n_patches_h, n_patches_w, concept,
                    )

            dt = time.perf_counter() - t0
            logger.info(
                "extract_heatmaps_via_cache: %d/%d heatmaps in %.1fs "
                "(layers %d-%d, %d new tokens, grid %dx%d)",
                len(heatmaps), len(concepts), dt,
                start_layer, end_layer, n_new,
                n_patches_h, n_patches_w,
            )

            return heatmaps

        except Exception as e:
            logger.warning("extract_heatmaps_via_cache failed: %s", e)
            return {}
