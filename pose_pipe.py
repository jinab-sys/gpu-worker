"""
LTX-2 Pose Transfer Pipeline (IC-LoRA).
Takes a reference video's motion and applies it to a new subject.

NOTE: IC-LoRA uses the ltx-pipelines package (Lightricks' native code).
If diffusers adds IC-LoRA support later, this can be simplified.
For now, this wraps the native ICLoraPipeline.
"""

import os
import torch
import logging
from typing import Optional, Literal
from PIL import Image

import config

logger = logging.getLogger("pose_pipe")


class PosePipeline:
    def __init__(self):
        self.device = config.DEVICE
        self.pipeline = None
        self._loaded = False

    def load(self):
        """
        Load LTX-2 IC-LoRA pipeline.
        Uses the native ltx-pipelines package for IC-LoRA support.
        """
        if self._loaded:
            return

        try:
            from ltx_pipelines.iclora_pipeline import ICLoraPipeline

            logger.info("Loading IC-LoRA pipeline...")

            # IC-LoRA uses the distilled checkpoint
            checkpoint_path = os.path.join(
                config.MODELS_DIR, "ltx-2-19b-distilled-fp8.safetensors"
            )

            # Fall back to searching common locations
            if not os.path.exists(checkpoint_path):
                for search_dir in [
                    config.MODELS_DIR,
                    "/mnt/models",
                    os.path.expanduser("~/.cache/huggingface"),
                ]:
                    for root, dirs, files in os.walk(search_dir):
                        for f in files:
                            if "distilled" in f and f.endswith(".safetensors"):
                                checkpoint_path = os.path.join(root, f)
                                break

            self.pipeline = ICLoraPipeline(
                checkpoint_path=checkpoint_path,
                enable_fp8=True,
            )
            self._loaded = True
            logger.info("IC-LoRA pipeline loaded")

        except ImportError:
            logger.warning(
                "ltx-pipelines not installed. Install with: "
                "pip install git+https://github.com/Lightricks/LTX-2.git"
            )
            raise RuntimeError(
                "IC-LoRA requires ltx-pipelines. "
                "Install: pip install git+https://github.com/Lightricks/LTX-2.git"
            )

    def unload(self):
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
        torch.cuda.empty_cache()
        self._loaded = False

    def transfer_pose(
        self,
        reference_video_path: str,
        subject_image: Optional[Image.Image] = None,
        prompt: str = "",
        control_mode: Literal["pose", "depth", "canny"] = "pose",
        width: int = config.DEFAULT_VIDEO_WIDTH,
        height: int = config.DEFAULT_VIDEO_HEIGHT,
        duration_seconds: int = 5,
        frame_rate: float = config.DEFAULT_VIDEO_FPS,
        guidance_scale: float = 3.0,
        ic_lora_strength: float = 1.0,
        seed: Optional[int] = None,
        output_path: str = "output.mp4",
    ) -> str:
        """
        Transfer motion from reference video to a new subject.

        Args:
            reference_video_path: Path to the motion reference video
            subject_image: PIL Image of the subject (optional, for I2V mode)
            prompt: Style/appearance prompt (motion comes from reference)
            control_mode: "pose", "depth", or "canny"
            width: Output width
            height: Output height
            duration_seconds: Output duration
            frame_rate: Output FPS
            guidance_scale: CFG scale
            ic_lora_strength: IC-LoRA influence (1.0 = full control)
            seed: Random seed
            output_path: Where to save output

        Returns:
            Path to generated video
        """
        if not self._loaded:
            self.load()

        from ltx_core.model.video_vae import TilingConfig
        from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT
        from ltx_pipelines.utils.media_io import encode_video

        num_frames = config.seconds_to_frames(duration_seconds, int(frame_rate))

        # Select IC-LoRA model based on control mode
        ic_lora_map = {
            "pose": "ltx-2-19b-ic-lora-pose-control.safetensors",
            "depth": "ltx-2-19b-ic-lora-depth-control.safetensors",
            "canny": "ltx-2-19b-ic-lora-canny-control.safetensors",
        }

        # Union model works for all modes
        union_lora = "ltx-2-19b-ic-lora-union-ref0.5.safetensors"
        ic_lora_path = os.path.join(config.MODELS_DIR, "loras", union_lora)

        if not os.path.exists(ic_lora_path):
            # Try mode-specific LoRA
            specific = ic_lora_map.get(control_mode)
            ic_lora_path = os.path.join(config.MODELS_DIR, "loras", specific)

        logger.info(
            f"Pose transfer: {control_mode} mode, "
            f"{num_frames} frames @ {width}x{height}"
        )

        tiling_config = TilingConfig()

        # Prepare images list (subject image as first frame)
        images = None
        if subject_image is not None:
            subject_image = subject_image.resize((width, height), Image.LANCZOS)
            images = [subject_image]

        with torch.inference_mode():
            video, audio = self.pipeline(
                prompt=prompt,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                seed=seed or 42,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                guidance_scale=guidance_scale,
                images=images,
                reference_video_path=reference_video_path,
                ic_lora_path=ic_lora_path,
                ic_lora_strength=ic_lora_strength,
                control_mode=control_mode,
                tiling_config=tiling_config,
            )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        encode_video(
            video=video,
            fps=frame_rate,
            audio=audio,
            audio_sample_rate=44100,
            output_path=output_path,
        )

        logger.info(f"Pose transfer video saved: {output_path}")
        return output_path

    @property
    def is_loaded(self) -> bool:
        return self._loaded
