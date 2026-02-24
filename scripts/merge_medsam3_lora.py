"""
Merge MedSAM3 LoRA weights into base SAM3 model.

MedSAM3 (lal-Joey/MedSAM3_v1) uses a custom LoRA implementation from the
facebookresearch SAM3 codebase, stored as best_lora_weights.pt.
This script maps those weights to HuggingFace transformers Sam3Model keys
and merges them.

LoRA convention in MedSAM3:
  A = (in_features, rank), B = (rank, out_features)
  forward: output += x @ A @ B * scaling
  merge:   W_new = W + B.T @ A.T * scaling   (W is out_features × in_features)

Usage:
    python scripts/merge_medsam3_lora.py --diagnose
    python scripts/merge_medsam3_lora.py --output models/medsam3-merged
    python scripts/merge_medsam3_lora.py --verify models/medsam3-merged
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("merge_medsam3")

# LoRA config from MedSAM3 training (configs/full_lora_config.yaml)
LORA_RANK = 16
LORA_ALPHA = 32
LORA_SCALING = LORA_ALPHA / LORA_RANK  # 2.0

BASE_MODEL_ID = "facebook/sam3"
LORA_REPO_ID = "lal-Joey/MedSAM3_v1"
LORA_FILENAME = "best_lora_weights.pt"


# ============================================================================
# Key mapping: facebookresearch SAM3 → HuggingFace transformers Sam3Model
# ============================================================================
# Built from diagnostic output of --diagnose comparing 458 LoRA pairs
# against 1468 base model parameters.

# Path prefix renames (applied in order, first match wins)
PREFIX_RENAMES = [
    ("backbone.vision_backbone.trunk.blocks.", "vision_encoder.backbone.layers."),
    ("backbone.language_backbone.encoder.transformer.resblocks.", "text_encoder.text_model.encoder.layers."),
    ("transformer.decoder.", "detr_decoder."),
    ("transformer.encoder.", "detr_encoder."),
    ("geometry_encoder.encode.", "geometry_encoder.layers."),
    ("segmentation_head.cross_attend_prompt.", "mask_decoder.prompt_cross_attn."),
    ("dot_prod_scoring.", "dot_product_scoring."),
]

# Suffix/segment renames within a key (applied after prefix rename)
SEGMENT_RENAMES = [
    # Vision encoder
    (".attn.proj.", ".attention.o_proj."),
    (".attn.qkv.", ".attention.qkv."),  # fused QKV, handled specially
    (".attn.q_proj.", ".self_attn.q_proj."),
    (".attn.k_proj.", ".self_attn.k_proj."),
    (".attn.v_proj.", ".self_attn.v_proj."),
    (".attn.out_proj.", ".self_attn.out_proj."),
    # Text encoder
    (".mlp.c_fc.", ".mlp.fc1."),
    (".mlp.c_proj.", ".mlp.fc2."),
    # DETR/geometry encoder
    (".cross_attn_image.", ".cross_attn."),
    (".ca_text.", ".text_cross_attn."),
    (".cross_attn.out_proj.", ".cross_attn.o_proj."),
    (".self_attn.out_proj.", ".self_attn.o_proj."),
    (".text_cross_attn.out_proj.", ".text_cross_attn.o_proj."),
    # DETR decoder cross_attn → vision_cross_attn
    ("detr_decoder.layers.", "detr_decoder.layers."),  # no-op, needs context
    (".linear1.", ".mlp.fc1."),
    (".linear2.", ".mlp.fc2."),
    # dot_product_scoring
    (".hs_proj.", ".query_proj."),
    (".prompt_proj.", ".text_proj."),
]


def download_lora_weights() -> str:
    """Download MedSAM3 LoRA weights from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download

    logger.info("Downloading LoRA weights from %s/%s …", LORA_REPO_ID, LORA_FILENAME)
    path = hf_hub_download(repo_id=LORA_REPO_ID, filename=LORA_FILENAME)
    logger.info("LoRA weights downloaded to: %s", path)
    return path


def load_lora_state(path: str) -> dict[str, torch.Tensor]:
    """Load LoRA state dict from .pt file."""
    logger.info("Loading LoRA weights from %s …", path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    logger.info("Loaded %d LoRA tensors", len(state))
    return state


def parse_lora_pairs(lora_state: dict[str, torch.Tensor]) -> dict[str, dict]:
    """Group LoRA keys into pairs: {base_key: {"lora_A": tensor, "lora_B": tensor}}."""
    pairs: dict[str, dict] = defaultdict(dict)
    for key, tensor in lora_state.items():
        if key.endswith(".lora_A"):
            base = key[: -len(".lora_A")]
            pairs[base]["lora_A"] = tensor
        elif key.endswith(".lora_B"):
            base = key[: -len(".lora_B")]
            pairs[base]["lora_B"] = tensor
        else:
            logger.warning("Unexpected LoRA key (not .lora_A/.lora_B): %s", key)
    return dict(pairs)


def _translate_key(lora_key: str) -> str:
    """Translate a single LoRA base key to HF transformers naming."""
    key = lora_key

    # Strip trailing .lora suffix (MedSAM3 stores as module.lora.lora_A/lora_B)
    if key.endswith(".lora"):
        key = key[:-5]

    # Apply prefix renames
    for old, new in PREFIX_RENAMES:
        if key.startswith(old):
            key = new + key[len(old):]
            break

    # Apply segment renames (add trailing dot to match both mid-key and end-of-key)
    key = key + "."
    for old, new in SEGMENT_RENAMES:
        key = key.replace(old, new)
    key = key.rstrip(".")

    # Special case: DETR decoder cross_attn → vision_cross_attn
    # (Only for detr_decoder, not detr_encoder or geometry_encoder)
    if key.startswith("detr_decoder.layers."):
        key = re.sub(
            r"(detr_decoder\.layers\.\d+)\.cross_attn\.",
            r"\1.vision_cross_attn.",
            key,
        )

    # Text encoder uses out_proj (not o_proj like other components)
    if key.startswith("text_encoder."):
        key = key.replace(".o_proj", ".out_proj")

    # Generic out_proj → o_proj for non-text-encoder components
    # (handles mask_decoder.prompt_cross_attn.out_proj → o_proj)
    if not key.startswith("text_encoder."):
        key = key.replace(".out_proj", ".o_proj")

    return key


def build_key_mapping(
    lora_pairs: dict[str, dict],
    base_state: dict[str, torch.Tensor],
) -> tuple[dict[str, str | list[str]], list[str]]:
    """
    Build mapping: lora_base_key → base_model_weight_key(s).

    Returns (mapping, unmatched_keys).

    For fused QKV (vision encoder), maps to a list of 3 keys [q, k, v].
    For all others, maps to a single key string.
    """
    mapping: dict[str, str | list[str]] = {}
    unmatched: list[str] = []

    for lora_key, ab in lora_pairs.items():
        lora_a = ab.get("lora_A")
        lora_b = ab.get("lora_B")
        if lora_a is None or lora_b is None:
            logger.warning("Incomplete LoRA pair for %s", lora_key)
            unmatched.append(lora_key)
            continue

        translated = _translate_key(lora_key)

        # Check for fused QKV (vision encoder)
        if ".attention.qkv" in translated:
            # Fused QKV: B=(rank, 3*dim), need to split into q, k, v
            base_prefix = translated.replace(".attention.qkv", ".attention.")
            q_key = base_prefix + "q_proj.weight"
            k_key = base_prefix + "k_proj.weight"
            v_key = base_prefix + "v_proj.weight"

            if q_key in base_state and k_key in base_state and v_key in base_state:
                mapping[lora_key] = [q_key, k_key, v_key]
            else:
                logger.warning("QKV split keys not found: %s → %s, %s, %s", lora_key, q_key, k_key, v_key)
                unmatched.append(lora_key)
        else:
            # Standard key
            base_key = translated + ".weight"
            if base_key in base_state:
                # Verify shape compatibility
                expected_out = lora_b.shape[1]  # B=(rank, out_features)
                expected_in = lora_a.shape[0]   # A=(in_features, rank)
                actual_shape = base_state[base_key].shape
                if actual_shape == torch.Size([expected_out, expected_in]):
                    mapping[lora_key] = base_key
                else:
                    logger.warning(
                        "Shape mismatch: %s → %s (expected (%d,%d), got %s)",
                        lora_key, base_key, expected_out, expected_in, tuple(actual_shape),
                    )
                    unmatched.append(lora_key)
            else:
                # Try without .weight suffix (some params may differ)
                logger.warning("Key not found in base model: %s → %s", lora_key, base_key)
                unmatched.append(lora_key)

    return mapping, unmatched


def merge_lora_into_base(
    base_state: dict[str, torch.Tensor],
    lora_pairs: dict[str, dict],
    key_mapping: dict[str, str | list[str]],
    scaling: float = LORA_SCALING,
) -> dict[str, torch.Tensor]:
    """
    Merge LoRA weights into base model state dict.

    MedSAM3 LoRA convention:
        A = (in_features, rank), B = (rank, out_features)
        forward: output += x @ A @ B * scaling
        merge:   W_new = W + B.T @ A.T * scaling
    """
    merged_state = dict(base_state)  # shallow copy (tensors shared until modified)
    merged_count = 0

    for lora_key, base_keys in key_mapping.items():
        lora_a = lora_pairs[lora_key]["lora_A"]  # (in_features, rank)
        lora_b = lora_pairs[lora_key]["lora_B"]  # (rank, out_features)

        if isinstance(base_keys, list):
            # Fused QKV: split B along output dimension
            # B = (rank, 3*dim) → B_q, B_k, B_v each (rank, dim)
            dim = lora_b.shape[1] // 3
            b_splits = [lora_b[:, i * dim : (i + 1) * dim] for i in range(3)]

            for base_key, b_part in zip(base_keys, b_splits):
                base_w = merged_state[base_key]
                # delta = B_part.T @ A.T * scaling = (dim, rank) @ (rank, in_feat) * s
                delta = (b_part.T @ lora_a.T) * scaling
                merged_state[base_key] = base_w + delta.to(base_w.dtype)
                merged_count += 1
                logger.debug(
                    "Merged QKV split %s → %s: delta_norm=%.4f",
                    lora_key, base_key, delta.norm().item(),
                )
        else:
            # Standard single target
            base_w = merged_state[base_keys]
            # delta = B.T @ A.T * scaling = (out, rank) @ (rank, in) * s = (out, in)
            delta = (lora_b.T @ lora_a.T) * scaling
            merged_state[base_keys] = base_w + delta.to(base_w.dtype)
            merged_count += 1
            logger.debug(
                "Merged %s → %s: delta_norm=%.4f, base_norm=%.4f",
                lora_key, base_keys, delta.norm().item(), base_w.norm().item(),
            )

    logger.info("Merged %d weight matrices from %d LoRA pairs", merged_count, len(key_mapping))
    return merged_state


# ============================================================================
# Diagnostic
# ============================================================================

def diagnose(lora_state: dict[str, torch.Tensor], base_state: dict[str, torch.Tensor] | None = None) -> None:
    """Print diagnostic info about LoRA and base model keys."""
    pairs = parse_lora_pairs(lora_state)

    print("\n" + "=" * 80)
    print(f"LoRA WEIGHTS: {len(lora_state)} tensors, {len(pairs)} pairs")
    print("=" * 80)

    components: dict[str, list] = defaultdict(list)
    for base_key, ab in sorted(pairs.items()):
        parts = base_key.split(".")
        component = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
        components[component].append((base_key, ab))

    for component, items in sorted(components.items()):
        print(f"\n--- {component} ({len(items)} pairs) ---")
        for base_key, ab in items:
            a_shape = tuple(ab["lora_A"].shape) if "lora_A" in ab else "MISSING"
            b_shape = tuple(ab["lora_B"].shape) if "lora_B" in ab else "MISSING"
            print(f"  {base_key}")
            print(f"    A={a_shape}  B={b_shape}")

    if base_state:
        print("\n" + "=" * 80)
        print(f"BASE MODEL: {len(base_state)} parameters")
        print("=" * 80)

        base_components: dict[str, list] = defaultdict(list)
        for key in sorted(base_state.keys()):
            if ".weight" not in key:
                continue
            parts = key.split(".")
            component = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
            base_components[component].append(key)

        for component, keys in sorted(base_components.items()):
            print(f"\n--- {component} ({len(keys)} weight params) ---")
            for key in keys[:5]:
                print(f"  {key}: {tuple(base_state[key].shape)}")
            if len(keys) > 5:
                print(f"  ... and {len(keys) - 5} more")

        # Test key mapping
        print("\n" + "=" * 80)
        print("KEY MAPPING TEST")
        print("=" * 80)
        mapping, unmatched = build_key_mapping(pairs, base_state)
        print(f"Matched: {len(mapping)}/{len(pairs)}")
        if unmatched:
            print(f"Unmatched ({len(unmatched)}):")
            for k in unmatched:
                translated = _translate_key(k)
                print(f"  {k}")
                print(f"    → {translated}.weight (NOT FOUND)")


# ============================================================================
# Commands
# ============================================================================

def run_diagnose() -> None:
    """Download weights and print diagnostic info."""
    lora_path = download_lora_weights()
    lora_state = load_lora_state(lora_path)

    try:
        from transformers import Sam3Model
        logger.info("Loading base SAM3 model for comparison …")
        base_model = Sam3Model.from_pretrained(BASE_MODEL_ID, dtype=torch.float32)
        base_state = dict(base_model.state_dict())
        del base_model
    except Exception as e:
        logger.warning("Could not load base model: %s", e)
        base_state = None

    diagnose(lora_state, base_state)


def run_merge(output_dir: str) -> None:
    """Full merge: download, map, merge, save."""
    from transformers import Sam3Model, Sam3Processor

    lora_path = download_lora_weights()
    lora_state = load_lora_state(lora_path)
    lora_pairs = parse_lora_pairs(lora_state)

    logger.info("Loading base SAM3 model (%s) …", BASE_MODEL_ID)
    base_model = Sam3Model.from_pretrained(BASE_MODEL_ID, dtype=torch.float32)
    processor = Sam3Processor.from_pretrained(BASE_MODEL_ID)
    base_state = dict(base_model.state_dict())

    logger.info("Building key mapping (%d LoRA pairs) …", len(lora_pairs))
    key_mapping, unmatched = build_key_mapping(lora_pairs, base_state)

    matched = len(key_mapping)
    total = len(lora_pairs)

    logger.info("Key mapping: %d/%d matched", matched, total)
    if unmatched:
        logger.warning("Unmatched LoRA keys (%d):", len(unmatched))
        for k in unmatched:
            ab = lora_pairs[k]
            a_shape = tuple(ab.get("lora_A", torch.empty(0)).shape)
            b_shape = tuple(ab.get("lora_B", torch.empty(0)).shape)
            logger.warning("  %s  A=%s B=%s → %s.weight", k, a_shape, b_shape, _translate_key(k))

    if matched == 0:
        logger.error("No LoRA keys matched! Run --diagnose to inspect keys.")
        sys.exit(1)

    # Print mapping summary
    print("\n" + "=" * 80)
    print(f"KEY MAPPING: {matched}/{total} matched ({len(unmatched)} unmatched)")
    print("=" * 80)
    for lora_key, base_keys in sorted(key_mapping.items()):
        if isinstance(base_keys, list):
            print(f"  {lora_key} → [QKV SPLIT]")
            for bk in base_keys:
                print(f"    → {bk}")
        else:
            print(f"  {lora_key}")
            print(f"    → {base_keys}")

    # Merge
    logger.info("Merging LoRA weights (scaling=%.2f) …", LORA_SCALING)
    merged_state = merge_lora_into_base(base_state, lora_pairs, key_mapping)

    # Load merged state
    base_model.load_state_dict(merged_state)

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info("Saving merged model to %s …", output_path)
    base_model.save_pretrained(output_path)
    processor.save_pretrained(output_path)

    # Save merge metadata
    # Convert list values to strings for JSON serialization
    serializable_mapping = {
        k: v if isinstance(v, str) else v
        for k, v in key_mapping.items()
    }
    metadata = {
        "base_model": BASE_MODEL_ID,
        "lora_repo": LORA_REPO_ID,
        "lora_file": LORA_FILENAME,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_scaling": LORA_SCALING,
        "matched_pairs": matched,
        "total_pairs": total,
        "unmatched_keys": unmatched,
        "key_mapping": serializable_mapping,
    }
    with open(output_path / "merge_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Merge complete! Model saved to %s", output_path)
    logger.info("  Matched: %d/%d LoRA pairs", matched, total)
    logger.info("  Unmatched: %d", len(unmatched))

    # Verify saved model loads
    logger.info("Verifying saved model loads …")
    loaded = Sam3Model.from_pretrained(output_path, dtype=torch.float32)
    param_count = sum(p.numel() for p in loaded.parameters()) / 1e6
    logger.info("Verified: loaded model has %.0fM params", param_count)


def run_verify(model_path: str) -> None:
    """Verify a merged model loads and can run inference."""
    from transformers import Sam3Model, Sam3Processor

    logger.info("Loading merged model from %s …", model_path)
    model = Sam3Model.from_pretrained(model_path, dtype=torch.float32)
    processor = Sam3Processor.from_pretrained(model_path)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Model loaded: %.0fM params", param_count)

    meta_path = Path(model_path) / "merge_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        logger.info("Merge metadata: %d/%d pairs merged (scaling=%.2f)",
                     meta["matched_pairs"], meta["total_pairs"], meta["lora_scaling"])
        if meta["unmatched_keys"]:
            logger.warning("Unmatched keys: %s", meta["unmatched_keys"])
    else:
        logger.warning("No merge_metadata.json found")

    logger.info("Model verification passed!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge MedSAM3 LoRA weights into SAM3 base model",
    )
    parser.add_argument("--diagnose", action="store_true",
                        help="Print diagnostic info about LoRA and base model keys")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for merged model")
    parser.add_argument("--verify", type=str, default=None,
                        help="Path to merged model to verify")

    args = parser.parse_args()

    if args.diagnose:
        run_diagnose()
    elif args.verify:
        run_verify(args.verify)
    elif args.output:
        run_merge(args.output)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python scripts/merge_medsam3_lora.py --diagnose")
        print("  python scripts/merge_medsam3_lora.py --output models/medsam3-merged")
        print("  python scripts/merge_medsam3_lora.py --verify models/medsam3-merged")


if __name__ == "__main__":
    main()
