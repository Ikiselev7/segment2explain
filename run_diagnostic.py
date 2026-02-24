"""Diagnostic runner: captures ALL intermediate MedGemma + MedSAM3 results for analysis."""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback

import numpy as np
from PIL import Image

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

logger = logging.getLogger("diagnostic")

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "tests", "fixtures", "sample_cxr_vqa")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "diagnostic_output")


def _render_annotated(annotated_img) -> Image.Image | None:
    """Render build_annotated_image() output to a PIL Image.

    build_annotated_image() returns (base_image_np, [(mask, label), ...]) or None.
    We render masks as colored overlays using overlay_multiple_masks.
    """
    if annotated_img is None:
        return None

    if isinstance(annotated_img, Image.Image):
        return annotated_img
    if isinstance(annotated_img, np.ndarray):
        return Image.fromarray(annotated_img)

    # Tuple format: (img_np, annotations)
    if isinstance(annotated_img, tuple) and len(annotated_img) == 2:
        base_img, annotations = annotated_img
        if base_img is None:
            return None
        if not annotations:
            return Image.fromarray(base_img)

        # Separate masks from bboxes (bboxes are tuples of ints)
        mask_pairs = []
        for item, label in annotations:
            if isinstance(item, np.ndarray):
                mask_pairs.append((item, label))
            # Skip bbox annotations for diagnostic images

        if not mask_pairs:
            return Image.fromarray(base_img)

        from tools.overlay import overlay_multiple_masks
        return overlay_multiple_masks(base_img, mask_pairs)

    return None


def run_diagnostic(sample: dict) -> dict:
    """Run pipeline on a sample and capture all intermediate state."""
    from backend.pipeline import run_job

    name = sample["name"]
    img_path = os.path.join(FIXTURES_DIR, sample["image_file"])
    prompt = sample["prompt"]

    logger.info("=" * 70)
    logger.info("SAMPLE: %s | prompt='%s'", name, prompt)

    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)
    img_h, img_w = img_np.shape[:2]
    logger.info("  image: %dx%d", img_w, img_h)

    result = {
        "name": name,
        "prompt": prompt,
        "image_size": [img_w, img_h],
        "gt_bbox": sample.get("gt_bbox"),
        "gt_finding": sample.get("gt_finding"),
    }

    t0 = time.perf_counter()
    all_yields = []

    try:
        for i, (chat, steps_html, annotated_img, meas_json, debug_json) in enumerate(
            run_job(image=img_np, user_prompt=prompt, compare_baseline=False, state=None)
        ):
            # Capture every yield
            yield_data = {
                "yield_idx": i,
                "chat_last": chat[-1] if chat else None,
                "n_chat": len(chat),
                "meas_keys": list(meas_json.keys()) if meas_json else [],
                "debug_keys": list(debug_json.keys()) if debug_json else [],
            }

            # Capture key debug values
            for key in ["R1_raw", "R1_concepts_raw", "R1_concepts",
                        "R1_proposed_bbox", "segmentation_mode",
                        "SEG_segment_count", "R2_raw",
                        "classified_count"]:
                if key in debug_json:
                    val = debug_json[key]
                    if isinstance(val, (list, dict, str, int, float, bool, type(None))):
                        yield_data[f"debug_{key}"] = val

            # Capture filter results
            for k, v in debug_json.items():
                if k.startswith("filter_F"):
                    yield_data[f"debug_{k}"] = v

            # Capture segment data if present
            seg_data = debug_json.get("_segment_data", {})
            if seg_data:
                yield_data["segments"] = {
                    sid: {
                        "label": s["label"],
                        "bbox": s["bbox"],
                        "description": s.get("description", ""),
                        "created_by_step": s["created_by_step"],
                        "color_idx": s.get("color_idx", 0),
                    }
                    for sid, s in seg_data.items()
                }

            # Save overlay images at key points
            if annotated_img is not None:
                # Save at key transitions
                has_new_debug = any(
                    k in debug_json
                    for k in ["R1_concepts", "SEG_segment_count",
                              "classified_count", "R2_raw"]
                )
                if has_new_debug or i == 0:
                    try:
                        overlay_img = _render_annotated(annotated_img)
                        if overlay_img is not None:
                            overlay_path = os.path.join(
                                OUTPUT_DIR, f"{name}_yield{i:02d}.png"
                            )
                            overlay_img.save(overlay_path)
                            yield_data["saved_overlay"] = overlay_path
                    except Exception as e:
                        yield_data["overlay_error"] = str(e)

            all_yields.append(yield_data)

        # Extract final results
        last_debug = {}
        last_meas = {}
        last_chat = []
        for y in reversed(all_yields):
            if "debug_R1_raw" in y:
                result["R1_raw"] = y["debug_R1_raw"]
            if "debug_R1_concepts_raw" in y:
                result["R1_concepts_raw"] = y["debug_R1_concepts_raw"]
            if "debug_R1_concepts" in y:
                result["R1_concepts"] = y["debug_R1_concepts"]
            if "debug_R1_proposed_bbox" in y:
                result["R1_proposed_bbox"] = y["debug_R1_proposed_bbox"]
            if "debug_SEG_segment_count" in y:
                result["SEG_segment_count"] = y["debug_SEG_segment_count"]
            if "debug_classified_count" in y:
                result["classified_count"] = y["debug_classified_count"]
            if "debug_R2_raw" in y:
                result["R2_raw"] = y["debug_R2_raw"]
            if "segments" in y:
                result["final_segments"] = y["segments"]

        # Collect all filter results
        filter_results = {}
        for y in all_yields:
            for k, v in y.items():
                if k.startswith("debug_filter_F"):
                    filter_results[k.replace("debug_", "")] = v
        if filter_results:
            result["filter_results"] = filter_results

        # Final answer
        final_yield = all_yields[-1] if all_yields else {}
        if final_yield.get("chat_last"):
            result["final_answer"] = final_yield["chat_last"].get("content", "")

        result["status"] = "success"
        result["n_yields"] = len(all_yields)

    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        logger.error("FAILED: %s", traceback.format_exc())

    result["time_s"] = round(time.perf_counter() - t0, 1)
    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(FIXTURES_DIR, "samples.json")) as f:
        samples = json.load(f)

    # Allow selecting specific sample(s)
    if len(sys.argv) > 1:
        names = sys.argv[1:]
        samples = [s for s in samples if s["name"] in names]
        if not samples:
            logger.error("No samples matched: %s", names)
            sys.exit(1)

    all_results = []
    for sample in samples:
        r = run_diagnostic(sample)
        all_results.append(r)

        # Print summary for this sample
        logger.info("-" * 50)
        logger.info("SAMPLE: %s", r["name"])
        logger.info("  Status: %s (%.1fs)", r.get("status"), r.get("time_s", 0))
        if "R1_concepts_raw" in r:
            logger.info("  R1 concepts (raw): %s", r["R1_concepts_raw"])
        if "R1_concepts" in r:
            logger.info("  R1 concepts (prepared): %s", r["R1_concepts"])
        if "R1_proposed_bbox" in r:
            logger.info("  R1 proposed bbox: %s", r["R1_proposed_bbox"])
        if "SEG_segment_count" in r:
            logger.info("  SEG segments: %d", r["SEG_segment_count"])
        if "classified_count" in r:
            logger.info("  Classified segments: %d", r["classified_count"])
        if "final_segments" in r:
            for sid, seg in r["final_segments"].items():
                logger.info("    Seg %s: label='%s' bbox=%s step=%s desc='%s'",
                            sid, seg["label"], seg["bbox"],
                            seg["created_by_step"], seg.get("description", "")[:60])
        if "filter_results" in r:
            for fk, fv in r["filter_results"].items():
                logger.info("  %s: %s", fk, json.dumps(fv)[:120])
        if "final_answer" in r:
            logger.info("  Answer (first 300): %s", r["final_answer"][:300])
        logger.info("-" * 50)

    # Save all results
    results_path = os.path.join(OUTPUT_DIR, "diagnostic_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
