"""Quick test: run one case to debug KV cache continuation."""
import logging
import os
import numpy as np
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s", datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "tests", "fixtures")
SAMPLE_DIR = os.path.join(FIXTURES_DIR, "sample_cxr_vqa")

# Load one image
img_path = os.path.join(SAMPLE_DIR, "cardiomegaly.png")
pil = Image.open(img_path).convert("RGB")
img_np = np.array(pil)

print(f"Image: {img_np.shape}")

from backend.pipeline import run_parallel_job
from orchestrator import create_job_state

state = create_job_state()

for chat, steps_html, annotated_img, meas_json, debug_json in run_parallel_job(
    image=img_np,
    user_prompt="Where is the Cardiomegaly?",
    state=state,
):
    pass

print(f"\nDone. Segments: {len(state.segments)}")
print(f"Debug keys: {list(debug_json.keys())}")
print(f"SELECT raw: {debug_json.get('SELECT_raw', 'N/A')[:200]}")
