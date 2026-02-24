"""Run the pipeline headlessly on VinDr-CXR-VQA samples to validate end-to-end."""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback

import numpy as np
from PIL import Image

# Load .env before anything else
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

logger = logging.getLogger("run_samples")

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "tests", "fixtures", "sample_cxr_vqa")


def load_samples() -> list[dict]:
    with open(os.path.join(FIXTURES_DIR, "samples.json")) as f:
        return json.load(f)


def run_one_sample(sample: dict, *, mode: str = "sequential") -> dict:
    """Run the pipeline on a single sample, return result dict.

    Args:
        mode: "sequential" (Evidence) or "parallel" (Quick).
    """
    from backend.pipeline import run_job, run_parallel_job

    name = sample["name"]
    img_path = os.path.join(FIXTURES_DIR, sample["image_file"])
    prompt = sample["prompt"]
    gt_bbox = sample.get("gt_bbox")

    logger.info("=" * 60)
    logger.info("SAMPLE: %s | prompt='%s' | mode=%s", name, prompt, mode)
    logger.info("  gt_bbox=%s  image=%s", gt_bbox, sample["image_file"])

    # Load image as numpy RGB uint8
    img = Image.open(img_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_np = np.array(img)

    logger.info("  image loaded: %dx%d", img_np.shape[1], img_np.shape[0])

    t0 = time.perf_counter()
    result = {
        "name": name,
        "prompt": prompt,
        "gt_bbox": gt_bbox,
        "status": "unknown",
        "error": None,
        "segments": [],
        "steps": [],
        "answer": None,
        "time_s": 0.0,
    }

    try:
        last_chat = []
        last_steps = ""
        last_meas = {}
        last_debug = {}

        if mode == "parallel":
            gen = run_parallel_job(image=img_np, user_prompt=prompt, state=None)
        else:
            gen = run_job(image=img_np, user_prompt=prompt, compare_baseline=False, state=None)

        for chat, steps_html, annotated_img, meas_json, debug_json in gen:
            last_chat = chat
            last_steps = steps_html
            last_meas = meas_json
            last_debug = debug_json

        # Extract final assistant answer
        assistant_msgs = [
            m for m in last_chat
            if isinstance(m, dict) and m.get("role") == "assistant"
        ]
        if assistant_msgs:
            result["answer"] = assistant_msgs[-1].get("content", "")[:500]

        result["segments"] = list(last_meas.keys()) if last_meas else []
        result["status"] = "success"
        result["measurements"] = last_meas
        result["debug_keys"] = list(last_debug.keys()) if last_debug else []

    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"
        logger.error("FAILED: %s", traceback.format_exc())

    result["time_s"] = round(time.perf_counter() - t0, 1)
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run pipeline on CXR-VQA samples")
    parser.add_argument("names", nargs="*", help="Sample names to run (default: all)")
    parser.add_argument(
        "--mode", choices=["sequential", "parallel"], default="sequential",
        help="Pipeline mode: sequential (Evidence) or parallel (Quick)",
    )
    args = parser.parse_args()

    samples = load_samples()
    logger.info("Loaded %d samples from %s", len(samples), FIXTURES_DIR)

    if args.names:
        samples = [s for s in samples if s["name"] in args.names]
        if not samples:
            logger.error("No samples matched: %s", args.names)
            sys.exit(1)

    mode = args.mode
    logger.info("Pipeline mode: %s", mode)

    results = []
    for sample in samples:
        r = run_one_sample(sample, mode=mode)
        results.append(r)
        logger.info(
            "RESULT: %s → %s in %.1fs | segments=%s | answer='%s'",
            r["name"], r["status"], r["time_s"],
            r["segments"],
            (r["answer"] or "")[:120],
        )

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY:")
    for r in results:
        status_icon = "OK" if r["status"] == "success" else "FAIL"
        logger.info(
            "  [%s] %s (%.1fs) segments=%s",
            status_icon, r["name"], r["time_s"], r["segments"],
        )

    # Save results
    results_path = os.path.join(FIXTURES_DIR, "run_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", results_path)

    # Exit with error if any failed
    if any(r["status"] != "success" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
