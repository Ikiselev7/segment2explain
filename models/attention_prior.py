"""Attention-based spatial priors for MedSAM3 segmentation.

Extracts attention heatmaps from MedGemma for medical concepts,
converts them to spatial prompts (bounding boxes, points) for MedSAM3.

Memory-efficient: attention scores are accumulated per-layer and
discarded immediately — full attention matrices are never stored.
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token position finding
# ---------------------------------------------------------------------------


def _find_subseq(haystack: list[int], needle: list[int]) -> list[int]:
    """Find first occurrence of needle subsequence in haystack.

    Returns list of indices where the subsequence starts, or empty list.
    """
    n = len(needle)
    if n == 0:
        return []
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return list(range(i, i + n))
    return []


def find_concept_token_positions(
    tokenizer,
    input_ids: list[int],
    concept: str,
) -> list[int]:
    """Find token positions of a concept string within the full input_ids.

    Tries encoding with/without leading space (BPE boundary effects).
    Falls back to searching for individual significant words.
    """
    for prefix in ("", " "):
        tokens = tokenizer.encode(prefix + concept, add_special_tokens=False)
        positions = _find_subseq(input_ids, tokens)
        if positions:
            return positions

    # Fallback: try individual significant words (>= 4 chars)
    words = [w for w in concept.split() if len(w) >= 4]
    all_positions: list[int] = []
    for word in words:
        for prefix in ("", " "):
            tokens = tokenizer.encode(prefix + word, add_special_tokens=False)
            positions = _find_subseq(input_ids, tokens)
            if positions:
                all_positions.extend(positions)
                break
    return all_positions


def find_image_token_positions(
    processor,
    input_ids: torch.Tensor,
) -> list[int]:
    """Find positions of image placeholder tokens in the input sequence.

    Tries multiple approaches:
    1. processor.image_token_id attribute
    2. model config image_token_index
    3. Tokenizer special token lookup
    """
    image_token_id: int | None = None

    # 1. Processor attribute (Gemma3Processor has this)
    image_token_id = getattr(processor, "image_token_id", None)

    # 2. Model config
    if image_token_id is None:
        config = getattr(processor, "config", None)
        if config is not None:
            image_token_id = getattr(config, "image_token_index", None)

    # 3. Tokenizer lookup
    if image_token_id is None:
        tokenizer = getattr(processor, "tokenizer", processor)
        unk_id = getattr(tokenizer, "unk_token_id", -1)
        for tok_name in ("<image>", "<image_soft_token>", "<img>"):
            try:
                tid = tokenizer.convert_tokens_to_ids(tok_name)
                if tid != unk_id and tid >= 0:
                    image_token_id = tid
                    break
            except Exception:
                continue

    if image_token_id is None:
        logger.warning("Cannot find image token ID from processor")
        return []

    # Find positions in input_ids
    ids = input_ids.flatten() if isinstance(input_ids, torch.Tensor) else input_ids
    if isinstance(ids, torch.Tensor):
        positions = (ids == image_token_id).nonzero(as_tuple=True)[0].tolist()
    else:
        positions = [i for i, t in enumerate(ids) if t == image_token_id]

    return positions


# ---------------------------------------------------------------------------
# Heatmap → spatial prior conversion
# ---------------------------------------------------------------------------


def heatmap_to_box(
    heatmap: np.ndarray,
    img_h: int,
    img_w: int,
    threshold_pct: float = 20.0,
    min_area_pct: float = 1.0,
    expand_ratio: float = 0.15,
) -> tuple[int, int, int, int] | None:
    """Convert attention heatmap to a bounding box in pixel coordinates.

    Thresholds the heatmap (top percentile), finds the largest connected
    component, and returns its bbox with optional expansion.

    Returns (x0, y0, x1, y1) or None if no significant region found.
    """
    from scipy import ndimage

    if heatmap.ndim != 2 or heatmap.size == 0:
        return None

    # Threshold: keep top percentile
    threshold = np.percentile(heatmap, 100 - threshold_pct)
    binary = (heatmap >= threshold).astype(np.uint8)

    # Find connected components
    labeled, n_components = ndimage.label(binary)
    if n_components == 0:
        return None

    # Find largest component
    largest_label = 0
    largest_size = 0
    for i in range(1, n_components + 1):
        size = int(np.sum(labeled == i))
        if size > largest_size:
            largest_label = i
            largest_size = size

    # Check minimum area
    total_pixels = heatmap.shape[0] * heatmap.shape[1]
    if largest_size / total_pixels * 100 < min_area_pct:
        return None

    mask = labeled == largest_label
    ys, xs = np.where(mask)

    # Convert patch grid to pixel coordinates
    h_scale = img_h / heatmap.shape[0]
    w_scale = img_w / heatmap.shape[1]

    x0 = int(xs.min() * w_scale)
    y0 = int(ys.min() * h_scale)
    x1 = int((xs.max() + 1) * w_scale)
    y1 = int((ys.max() + 1) * h_scale)

    # Expand box slightly for coverage
    bw = x1 - x0
    bh = y1 - y0
    x0 = max(0, int(x0 - bw * expand_ratio))
    y0 = max(0, int(y0 - bh * expand_ratio))
    x1 = min(img_w, int(x1 + bw * expand_ratio))
    y1 = min(img_h, int(y1 + bh * expand_ratio))

    return (x0, y0, x1, y1)


def heatmap_to_points(
    heatmap: np.ndarray,
    img_h: int,
    img_w: int,
    n_positive: int = 3,
    n_negative: int = 2,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Convert attention heatmap to positive and negative point prompts.

    Positive: top-K peaks. Negative: low-scoring regions far from positives.
    Returns (positive_points, negative_points) in pixel coordinates.
    """
    if heatmap.ndim != 2 or heatmap.size == 0:
        return [], []

    flat = heatmap.flatten()
    h_scale = img_h / heatmap.shape[0]
    w_scale = img_w / heatmap.shape[1]

    # Positive: top-K peaks
    top_indices = np.argsort(flat)[-n_positive:][::-1]
    pos_points = []
    for idx in top_indices:
        row, col = divmod(int(idx), heatmap.shape[1])
        px = int((col + 0.5) * w_scale)
        py = int((row + 0.5) * h_scale)
        pos_points.append((px, py))

    # Negative: bottom-K values, skipping those too close to positive points
    bottom_indices = np.argsort(flat)[: n_negative * 3]
    neg_points = []
    min_dist_x = img_w * 0.1
    min_dist_y = img_h * 0.1
    for idx in bottom_indices:
        row, col = divmod(int(idx), heatmap.shape[1])
        px = int((col + 0.5) * w_scale)
        py = int((row + 0.5) * h_scale)
        too_close = any(
            abs(px - ppx) < min_dist_x and abs(py - ppy) < min_dist_y
            for ppx, ppy in pos_points
        )
        if not too_close:
            neg_points.append((px, py))
            if len(neg_points) >= n_negative:
                break

    return pos_points, neg_points


def apply_heatmap_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    floor: float = 0.3,
) -> np.ndarray:
    """Apply attention heatmap as intensity modulation to an image.

    Regions with high attention remain bright; low-attention regions are
    dimmed toward *floor* brightness. The heatmap is resized to match the
    image and used as a per-pixel multiplier on each RGB channel.

    Args:
        image: RGB uint8 image (H, W, 3).
        heatmap: 2-D float heatmap in [0, 1] (arbitrary resolution).
        floor: Minimum brightness multiplier (0.0 = full black, 1.0 = no change).

    Returns:
        Modified RGB uint8 image with the same shape as *image*.
    """
    h, w = image.shape[:2]
    # Resize heatmap to image dimensions
    alpha = cv2.resize(heatmap.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    # Map [0,1] → [floor, 1.0]
    alpha = floor + alpha * (1.0 - floor)
    # Apply per-pixel modulation
    modulated = image.astype(np.float32) * alpha[:, :, np.newaxis]
    return np.clip(modulated, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Hook-based attention accumulation
# ---------------------------------------------------------------------------


class AttentionAccumulator:
    """Accumulates concept→image attention scores across transformer layers.

    Registers hooks on q_proj, k_proj, and self_attn modules. For each layer:
    1. q_proj hook captures raw Q projection output
    2. k_proj hook captures raw K projection output
    3. self_attn post-hook computes attention[concept_tokens → image_tokens],
       averages across heads, and adds to the running sum.

    All per-layer tensors are freed immediately after processing.
    Uses pre-RoPE Q,K which captures content-based (semantic) attention
    — suitable for spatial prior extraction.
    """

    def __init__(
        self,
        image_positions: list[int],
        concept_positions: dict[str, list[int]],
    ) -> None:
        self.image_positions = image_positions
        self.concept_positions = concept_positions
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._q_buffers: dict[int, torch.Tensor] = {}
        self._k_buffers: dict[int, torch.Tensor] = {}
        self.scores: dict[str, torch.Tensor | None] = {c: None for c in concept_positions}
        self.n_layers: dict[str, int] = {c: 0 for c in concept_positions}

    def register(
        self,
        language_model: torch.nn.Module,
        start_layer: int,
        end_layer: int,
    ) -> None:
        """Register hooks on layers[start_layer:end_layer]."""
        layers = language_model.layers
        for layer_idx in range(start_layer, end_layer):
            attn = layers[layer_idx].self_attn
            nh = attn.config.num_attention_heads
            nkv = attn.config.num_key_value_heads
            hd = attn.config.head_dim

            self._hooks.append(
                attn.q_proj.register_forward_hook(self._make_q_hook(layer_idx))
            )
            self._hooks.append(
                attn.k_proj.register_forward_hook(self._make_k_hook(layer_idx))
            )
            self._hooks.append(
                attn.register_forward_hook(
                    self._make_post_hook(layer_idx, nh, nkv, hd)
                )
            )

    def remove(self) -> None:
        """Remove all registered hooks and free buffers."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._q_buffers.clear()
        self._k_buffers.clear()

    def get_heatmaps(self, n_patches_h: int, n_patches_w: int) -> dict[str, np.ndarray]:
        """Get accumulated attention as 2D heatmaps per concept."""
        heatmaps: dict[str, np.ndarray] = {}
        n_expected = n_patches_h * n_patches_w
        for concept, raw_scores in self.scores.items():
            if raw_scores is None or self.n_layers[concept] == 0:
                continue
            avg = raw_scores / self.n_layers[concept]
            # Normalize to [0, 1]
            vmin, vmax = avg.min(), avg.max()
            avg = (avg - vmin) / (vmax - vmin + 1e-8)
            if len(avg) == n_expected:
                heatmaps[concept] = avg.numpy().reshape(n_patches_h, n_patches_w)
            else:
                logger.warning(
                    "Heatmap size mismatch: got %d, expected %dx%d=%d for '%s'",
                    len(avg), n_patches_h, n_patches_w, n_expected, concept,
                )
        return heatmaps

    # -- Hook factories (closures capture layer metadata) --

    def _make_q_hook(self, layer_idx: int):
        def hook(_module, _input, output):
            self._q_buffers[layer_idx] = output.detach()
        return hook

    def _make_k_hook(self, layer_idx: int):
        def hook(_module, _input, output):
            self._k_buffers[layer_idx] = output.detach()
        return hook

    def _make_post_hook(self, layer_idx: int, nh: int, nkv: int, hd: int):
        def hook(_module, _input, _output):
            q_raw = self._q_buffers.pop(layer_idx, None)
            k_raw = self._k_buffers.pop(layer_idx, None)
            if q_raw is None or k_raw is None:
                return

            with torch.no_grad():
                B, S, _ = q_raw.shape

                # Reshape: (B, S, H*D) → (B, H, S, D)
                q = q_raw.view(B, S, nh, hd).transpose(1, 2)
                # K may have different head count (GQA)
                k = k_raw.view(B, S, nkv, hd).transpose(1, 2)

                # Slice K to image positions only: (B, nkv, n_img, D)
                k_img = k[:, :, self.image_positions, :]

                # Expand K for GQA (repeat KV heads to match Q heads)
                if nh != nkv:
                    n_rep = nh // nkv
                    k_img = k_img.repeat_interleave(n_rep, dim=1)

                # For each concept, compute attention scores
                for concept, q_positions in self.concept_positions.items():
                    # Slice Q to concept token positions: (B, nh, n_concept, D)
                    q_concept = q[:, :, q_positions, :]

                    # Attention: (B, nh, n_concept, n_img)
                    attn = torch.matmul(q_concept, k_img.transpose(-1, -2))
                    attn = attn / math.sqrt(hd)
                    attn = torch.softmax(attn, dim=-1)

                    # Average across batch, heads, concept tokens → (n_img,)
                    avg = attn.mean(dim=(0, 1, 2)).cpu().float()

                    if self.scores[concept] is None:
                        self.scores[concept] = avg
                    else:
                        self.scores[concept] += avg
                    self.n_layers[concept] += 1

                # Free immediately
                del q, k, k_img, q_raw, k_raw

        return hook


# ---------------------------------------------------------------------------
# Inline attention capture during generation
# ---------------------------------------------------------------------------


class GenerationAttentionCapture:
    """Captures pre-RoPE K at image positions during model.generate() prefill.

    Hooks into k_proj during the prefill phase to store K at image token
    positions for each hooked layer.  After generation, the stored K_image
    is used by ``extract_heatmaps_via_cache()`` together with a lightweight
    locate-prompt forward pass to produce spatial heatmaps — without
    re-encoding the image through the ViT.
    """

    def __init__(self, image_positions: list[int]) -> None:
        self.image_positions = image_positions
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

        # Per-layer K_image from prefill: {layer_idx: (B, nkv, n_img, hd)}
        self._k_image: dict[int, torch.Tensor] = {}

        # Temporary K buffer for current forward pass
        self._k_buf: dict[int, torch.Tensor] = {}

        self._prefill_done: bool = False
        self._last_layer_idx: int = -1
        self._start_layer: int = 0

    @property
    def has_k_image(self) -> bool:
        """Whether pre-RoPE K at image positions is available."""
        return bool(self._k_image)

    @property
    def layer_range(self) -> tuple[int, int]:
        """(start_layer, end_layer) of hooked layers."""
        return (self._start_layer, self._last_layer_idx + 1)

    def register(
        self,
        language_model: torch.nn.Module,
        start_layer: int,
        end_layer: int,
    ) -> None:
        """Register hooks on layers[start_layer:end_layer]."""
        layers = language_model.layers
        self._last_layer_idx = end_layer - 1
        self._start_layer = start_layer

        for layer_idx in range(start_layer, end_layer):
            attn = layers[layer_idx].self_attn
            nkv = attn.config.num_key_value_heads
            hd = attn.config.head_dim

            self._hooks.append(
                attn.k_proj.register_forward_hook(
                    self._make_k_hook(layer_idx, nkv, hd)
                )
            )

    def remove(self) -> None:
        """Remove hooks and free temporary buffers.

        Preserves ``_k_image`` for use by the cached heatmap extraction path.
        Call ``clear()`` to free K_image when done.
        """
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._k_buf.clear()

    def clear(self) -> None:
        """Free all stored data (call after heatmap extraction)."""
        self._k_image.clear()

    # -- Hook factories --

    def _make_k_hook(self, layer_idx: int, nkv: int, hd: int):
        def hook(_module, _input, output):
            if self._prefill_done:
                return  # Only capture during prefill
            with torch.no_grad():
                k_raw = output.detach()
                B, S, _ = k_raw.shape
                if S > 1:  # Prefill (multi-token)
                    k = k_raw.view(B, S, nkv, hd).transpose(1, 2)
                    self._k_image[layer_idx] = k[:, :, self.image_positions, :].cpu()
                    del k

                if layer_idx == self._last_layer_idx and S > 1:
                    self._prefill_done = True

                del k_raw
        return hook
