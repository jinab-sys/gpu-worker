"""
Configuration for GPU Worker.
All settings controllable via environment variables.
"""

import os


# ─── Server ───────────────────────────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
WORKERS = int(os.getenv("WORKERS", "1"))  # Keep 1 for GPU work

# ─── GPU ──────────────────────────────────────────────────────────────────────
DEVICE = os.getenv("DEVICE", "cuda:0")
DTYPE = "bfloat16"  # Best for A100

# ─── Models ───────────────────────────────────────────────────────────────────
MODELS_DIR = os.getenv("MODELS_DIR", "./models")
IMAGE_MODEL_ID = os.getenv("IMAGE_MODEL_ID", "black-forest-labs/FLUX.2-klein-9B")
VIDEO_MODEL_ID = os.getenv("VIDEO_MODEL_ID", "Lightricks/LTX-2")
# When LTX 2.3 is ready in diffusers, just change above to:
# VIDEO_MODEL_ID = "Lightricks/LTX-2.3"

# ─── Storage ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
DEFAULT_BUCKET = os.getenv("DEFAULT_BUCKET", "generated")

# ─── Redis ────────────────────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# ─── Worker Identity (for multi-GPU setups) ───────────────────────────────────
WORKER_ID = os.getenv("WORKER_ID", f"gpu-worker-{os.getpid()}")

# ─── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_IMAGE_WIDTH = 768
DEFAULT_IMAGE_HEIGHT = 1360  # 9:16
DEFAULT_VIDEO_WIDTH = 432
DEFAULT_VIDEO_HEIGHT = 768   # 9:16
DEFAULT_VIDEO_FPS = 25
DEFAULT_VIDEO_STEPS = 25
DEFAULT_VIDEO_CFG = 3.0
DEFAULT_IMAGE_STEPS = 4
DEFAULT_IMAGE_CFG = 1.0

DEFAULT_NEGATIVE_PROMPT = (
    "morphing, distortion, warping, flicker, jitter, stutter, "
    "shaky camera, erratic motion, temporal artifacts, frame blending, "
    "low quality, jpeg artifacts, text, watermark, logo, cartoon, anime, CGI"
)

# ─── Duration to frames mapping ──────────────────────────────────────────────
# LTX-2 at 25fps. Frames must satisfy (frames - 1) % 8 == 0
DURATION_TO_FRAMES = {
    3: 73,
    5: 121,
    8: 193,
    10: 241,
    12: 297,
    15: 369,
    20: 489,
}

def seconds_to_frames(seconds: int, fps: int = 25) -> int:
    """Convert duration to valid LTX-2 frame count."""
    if seconds in DURATION_TO_FRAMES:
        return DURATION_TO_FRAMES[seconds]
    # Calculate nearest valid frame count: (n-1) % 8 == 0
    raw = int(seconds * fps)
    adjusted = ((raw - 1) // 8) * 8 + 1
    return max(adjusted, 9)  # minimum 9 frames
