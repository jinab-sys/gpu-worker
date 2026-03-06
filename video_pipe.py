"""
LTX-2.3 Video Pipeline — wraps official ltx-pipelines for FastAPI.
Uses fp8 quantization + single-stage generation.
~2.5 min for 5-second video on A100 80GB.

Model paths are configured below — update if you move files.
"""

import os
import sys
import torch
import logging
from typing import Optional
from PIL import Image

import config

logger = logging.getLogger("video_pipe")

# ─── Model paths ──────────────────────────────────────────────────────────────
CHECKPOINT_PATH = os.getenv(
    "LTX_CHECKPOINT",
    "/home/shahbaz_ali/models/ltx-2.3-fp8/ltx-2.3-22b-dev-fp8.safetensors",
)
GEMMA_ROOT = os.getenv(
    "LTX_GEMMA_ROOT",
    "/home/shahbaz_ali/models/gemma",
)
SPATIAL_UPSAMPLER_PATH = os.getenv(
    "LTX_SPATIAL_UPSAMPLER",
    "/home/shahbaz_ali/models/ltx-2/ltx-2-spatial-upscaler-x2-1.0.safetensors",
)
DISTILLED_LORA_PATH = os.getenv(
    "LTX_DISTILLED_LORA",
    "/home/shahbaz_ali/models/ltx-2.3/ltx-2.3-22b-distilled-lora-384.safetensors",
)

# Ensure LTX-2 packages are importable
LTX2_REPO = "/home/shahbaz_ali/LTX-2"
if LTX2_REPO not in sys.path:
    sys.path.insert(0, os.path.join(LTX2_REPO, "packages", "ltx-pipelines", "src"))
    sys.path.insert(0, os.path.join(LTX2_REPO, "packages", "ltx-core", "src"))


def _round32(x: int) -> int:
    return round(x / 32) * 32


class VideoPipeline:
    def __init__(self):
        self.device = config.DEVICE
        self.pipeline = None
        self._loaded = False

    def load(self):
        """Load LTX-2.3 pipeline with fp8 quantization."""
        if self._loaded:
            return

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline

        logger.info(f"Loading LTX-2.3 fp8 from {CHECKPOINT_PATH}...")
        logger.info(f"Gemma root: {GEMMA_ROOT}")

        self.pipeline = TI2VidOneStagePipeline(
            checkpoint_path=CHECKPOINT_PATH,
            gemma_root=GEMMA_ROOT,
            loras=[],
            fp8transformer=True,
        )

        allocated = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"LTX-2.3 loaded. VRAM: {allocated:.1f}GB")
        self._loaded = True

    def unload(self):
        """Free VRAM."""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
        torch.cuda.empty_cache()
        self._loaded = False
        logger.info("LTX-2.3 unloaded")

    def text_to_video(
        self,
        prompt: str,
        negative_prompt: str = config.DEFAULT_NEGATIVE_PROMPT,
        width: int = config.DEFAULT_VIDEO_WIDTH,
        height: int = config.DEFAULT_VIDEO_HEIGHT,
        duration_seconds: int = 5,
        frame_rate: float = config.DEFAULT_VIDEO_FPS,
        num_inference_steps: int = config.DEFAULT_VIDEO_STEPS,
        guidance_scale: float = config.DEFAULT_VIDEO_CFG,
        seed: Optional[int] = None,
        output_path: str = "output.mp4",
    ) -> str:
        """Generate video from text prompt."""
        if not self._loaded:
            self.load()

        width = _round32(width)
        height = _round32(height)
        num_frames = config.seconds_to_frames(duration_seconds, int(frame_rate))

        logger.info(
            f"T2V: {num_frames} frames @ {width}x{height}, "
            f"{num_inference_steps} steps, cfg={guidance_scale}"
        )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        video, audio = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed or 42,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            output_path=output_path,
        )

        logger.info(f"Video saved: {output_path}")
        return output_path

    def image_to_video(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = config.DEFAULT_NEGATIVE_PROMPT,
        width: int = config.DEFAULT_VIDEO_WIDTH,
        height: int = config.DEFAULT_VIDEO_HEIGHT,
        duration_seconds: int = 5,
        frame_rate: float = config.DEFAULT_VIDEO_FPS,
        num_inference_steps: int = config.DEFAULT_VIDEO_STEPS,
        guidance_scale: float = config.DEFAULT_VIDEO_CFG,
        seed: Optional[int] = None,
        output_path: str = "output.mp4",
    ) -> str:
        """Generate video from starting image + text prompt."""
        if not self._loaded:
            self.load()

        width = _round32(width)
        height = _round32(height)
        num_frames = config.seconds_to_frames(duration_seconds, int(frame_rate))

        # Save image to temp file for the pipeline
        image = image.resize((width, height), Image.LANCZOS)
        temp_img_path = output_path.replace(".mp4", "_input.png")
        os.makedirs(os.path.dirname(temp_img_path) or ".", exist_ok=True)
        image.save(temp_img_path)

        logger.info(
            f"I2V: {num_frames} frames @ {width}x{height}, "
            f"{num_inference_steps} steps, cfg={guidance_scale}"
        )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        video, audio = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed or 42,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            images=[(temp_img_path, 0, 1.0)],
            output_path=output_path,
        )

        # Clean up temp image
        try:
            os.remove(temp_img_path)
        except OSError:
            pass

        logger.info(f"Video saved: {output_path}")
        return output_path

    @property
    def is_loaded(self) -> bool:
        return self._loaded