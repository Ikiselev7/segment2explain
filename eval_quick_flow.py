"""Evaluate Quick (parallel) pipeline on all available images + DICOMs.

Saves per-case output to eval_output/:
  - {name}_overlay.png   — source image with segment overlays
  - {name}_report.txt    — answer text, concepts, segments, concept links
"""
from __future__ import annotations

import json
import logging
import os
import time
import traceback

import numpy as np
from PIL import Image

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger("eval_quick")

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "tests", "fixtures")
SAMPLE_DIR = os.path.join(FIXTURES_DIR, "sample_cxr_vqa")
DICOM_DIR = os.path.join(FIXTURES_DIR, "dicom")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "eval_output")


def load_image_rgb(path: str) -> tuple[np.ndarray, str]:
    """Load image from path, handle DICOM. Returns (rgb_uint8, format_info)."""
    with open(path, "rb") as f:
        raw = f.read()

    from backend.dicom import is_dicom, parse_dicom

    if is_dicom(raw):
        img_np, pixel_spacing = parse_dicom(raw)
        info = f"DICOM, pixel_spacing={pixel_spacing}"
    else:
        pil = Image.open(path)
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        img_np = np.array(pil)
        info = f"PNG/JPG {pil.mode}"

    return img_np, info


def render_overlay_image(state) -> Image.Image | None:
    """Render overlay with filled segments from pipeline state."""
    from tools.overlay import overlay_multiple_masks

    if state.image is None:
        return None

    mask_label_pairs = []
    color_indices = []
    for sid, seg in state.segments.items():
        mask = seg["mask"]
        label = seg.get("validated_name") or seg["label"]
        cidx = seg.get("color_idx", 0)
        mask_label_pairs.append((mask, label))
        color_indices.append(cidx)

    if not mask_label_pairs:
        return Image.fromarray(state.image)

    return overlay_multiple_masks(state.image, mask_label_pairs, color_indices=color_indices)


def run_case(name: str, img_np: np.ndarray, prompt: str, pixel_spacing=None) -> dict:
    """Run parallel pipeline on one case, save outputs."""
    from backend.pipeline import run_parallel_job
    from orchestrator import create_job_state

    logger.info("=" * 60)
    logger.info("CASE: %s | prompt='%s'", name, prompt)
    logger.info("  image: %dx%d", img_np.shape[1], img_np.shape[0])

    state = create_job_state()
    if pixel_spacing:
        state.pixel_spacing = pixel_spacing

    t0 = time.perf_counter()
    result = {
        "name": name,
        "prompt": prompt,
        "status": "unknown",
        "error": None,
        "answer": None,
        "prescan_terms": [],
        "concepts_list": [],
        "segments": {},
        "concept_links": [],
        "time_s": 0.0,
    }

    try:
        last_chat = []
        last_meas = {}
        last_debug = {}

        for chat, steps_html, annotated_img, meas_json, debug_json in run_parallel_job(
            image=img_np,
            user_prompt=prompt,
            state=state,
        ):
            last_chat = chat
            last_meas = meas_json
            last_debug = debug_json

        # Extract answer
        assistant_msgs = [
            m for m in last_chat
            if isinstance(m, dict) and m.get("role") == "assistant"
        ]
        if assistant_msgs:
            result["answer"] = assistant_msgs[-1].get("content", "")

        result["prescan_terms"] = last_debug.get("_prescan_terms", [])
        result["concepts_list"] = last_debug.get("CONCEPTS_list", [])
        result["concept_links"] = last_debug.get("_concept_links", [])
        result["select_raw"] = last_debug.get("SELECT_raw", "")

        # Segment info
        for sid, seg in state.segments.items():
            result["segments"][sid] = {
                "label": seg.get("validated_name") or seg["label"],
                "concept": seg.get("concept", ""),
                "bbox": list(seg["bbox"]),
                "measurements": last_meas.get(sid, {}),
            }

        result["status"] = "success"
        result["debug_keys"] = [k for k in last_debug.keys() if not k.startswith("_")]

        # Save overlay image
        overlay = render_overlay_image(state)
        if overlay:
            overlay_path = os.path.join(OUTPUT_DIR, f"{name}_overlay.png")
            overlay.save(overlay_path)
            logger.info("  Saved overlay: %s", overlay_path)

        # Save source image (no overlay) for comparison
        source_path = os.path.join(OUTPUT_DIR, f"{name}_source.png")
        Image.fromarray(img_np).save(source_path)

    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"
        logger.error("FAILED: %s", traceback.format_exc())

    result["time_s"] = round(time.perf_counter() - t0, 1)

    # Save text report
    report_path = os.path.join(OUTPUT_DIR, f"{name}_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Case: {name}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Status: {result['status']}\n")
        f.write(f"Time: {result['time_s']}s\n")
        f.write(f"\n{'='*60}\nANSWER:\n{'='*60}\n")
        f.write(result["answer"] or "(no answer)")
        f.write(f"\n\n{'='*60}\nPRE-SCAN TERMS:\n{'='*60}\n")
        f.write(", ".join(result["prescan_terms"]) if result["prescan_terms"] else "(none)")
        f.write(f"\n\n{'='*60}\nCONCEPTS LIST (sent to MedSAM3):\n{'='*60}\n")
        f.write(", ".join(result["concepts_list"]) if result["concepts_list"] else "(none)")
        f.write(f"\n\n{'='*60}\nSEGMENTS:\n{'='*60}\n")
        for sid, seg in result["segments"].items():
            f.write(f"\n  [{sid}] {seg['label']}\n")
            f.write(f"    concept: {seg['concept']}\n")
            f.write(f"    bbox: {seg['bbox']}\n")
            meas = seg.get("measurements", {})
            if meas:
                f.write(f"    area_px: {meas.get('area_px', 0)}\n")
                if meas.get("area_mm2"):
                    f.write(f"    area_mm2: {meas['area_mm2']}\n")
                if meas.get("max_diameter_mm"):
                    f.write(f"    max_diameter_mm: {meas['max_diameter_mm']}\n")
        f.write(f"\n\n{'='*60}\nCONCEPT LINKS (aliases → segment):\n{'='*60}\n")
        for cl in result.get("concept_links", []):
            f.write(f"\n  [{cl.get('segment_id', '?')}] {cl.get('concept', '?')}\n")
            f.write(f"    aliases: {cl.get('aliases', [])}\n")
            f.write(f"    color: {cl.get('color', '')}\n")
        f.write(f"\n\n{'='*60}\nSELECT RAW (MedGemma JSON):\n{'='*60}\n")
        f.write(result.get("select_raw", "(none)"))
        if result.get("error"):
            f.write(f"\n\n{'='*60}\nERROR:\n{'='*60}\n")
            f.write(result["error"])

    logger.info("  Saved report: %s", report_path)
    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Define all cases
    cases = []

    # 1. Sample CXR-VQA images
    with open(os.path.join(SAMPLE_DIR, "samples.json")) as f:
        samples = json.load(f)

    for s in samples:
        img_path = os.path.join(SAMPLE_DIR, s["image_file"])
        cases.append({
            "name": s["name"],
            "image_path": img_path,
            "prompt": s["prompt"],
        })

    # 2. DICOM files
    if os.path.isdir(DICOM_DIR):
        for fname in sorted(os.listdir(DICOM_DIR)):
            if fname.endswith((".dcm", ".DCM")):
                cases.append({
                    "name": f"dicom_{fname.replace('.dcm', '').replace('.DCM', '')}",
                    "image_path": os.path.join(DICOM_DIR, fname),
                    "prompt": "What are the findings in this chest X-ray?",
                })

    logger.info("Total cases: %d", len(cases))

    results = []
    for case in cases:
        img_np, format_info = load_image_rgb(case["image_path"])
        logger.info("  Format: %s, shape: %s", format_info, img_np.shape)

        r = run_case(case["name"], img_np, case["prompt"])
        results.append(r)
        logger.info(
            "RESULT: %s → %s in %.1fs | segments=%d | answer_len=%d",
            r["name"], r["status"], r["time_s"],
            len(r["segments"]),
            len(r["answer"] or ""),
        )

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY:")
    for r in results:
        icon = "OK" if r["status"] == "success" else "FAIL"
        logger.info(
            "  [%s] %s (%.1fs) segs=%d prescan=%s concepts=%s links=%d",
            icon, r["name"], r["time_s"],
            len(r["segments"]),
            r["prescan_terms"][:5],
            r["concepts_list"][:5],
            len(r.get("concept_links", [])),
        )

    # Save summary JSON
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
