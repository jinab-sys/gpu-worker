"""
LTX-2 Video Pipeline.
Handles image-to-video, text-to-video, with two-stage upscaling.
~90 seconds for 5-second video on A100.
"""

import os
import torch
import logging
from typing import Optional
from PIL import Image

import config

logger = logging.getLogger("video_pipe")


class VideoPipeline:
    def __init__(self):
        self.device = config.DEVICE
        self.pipe = None
        self.pipe_i2v = None
        self.upsample_pipe = None
        self._loaded = False

    def load(self):
        """Load LTX-2 models into VRAM with CPU offloading."""
        if self._loaded:
            return

        from diffusers import (
            LTX2Pipeline,
            LTX2ImageToVideoPipeline,
            LTX2LatentUpsamplePipeline,
        )
        from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel

        logger.info(f"Loading {config.VIDEO_MODEL_ID}...")

        # Text-to-video pipeline
        self.pipe = LTX2Pipeline.from_pretrained(
            config.VIDEO_MODEL_ID,
            torch_dtype=torch.bfloat16,
            cache_dir=config.MODELS_DIR,
        )
        self.pipe.enable_sequential_cpu_offload(device=self.device)

        # Image-to-video pipeline (shares components)
        self.pipe_i2v = LTX2ImageToVideoPipeline.from_pretrained(
            config.VIDEO_MODEL_ID,
            torch_dtype=torch.bfloat16,
            cache_dir=config.MODELS_DIR,
        )
        self.pipe_i2v.enable_sequential_cpu_offload(device=self.device)

        # Latent upsampler (stage 2)
        try:
            upsampler_model = LTX2LatentUpsamplerModel.from_pretrained(
                config.VIDEO_MODEL_ID,
                subfolder="latent_upsampler",
                torch_dtype=torch.bfloat16,
                cache_dir=config.MODELS_DIR,
            )
            self.upsample_pipe = LTX2LatentUpsamplePipeline(
                vae=self.pipe.vae,
                latent_upsampler=upsampler_model,
            )
            self.upsample_pipe.enable_model_cpu_offload(device=self.device)
            logger.info("LTX-2 upsampler loaded")
        except Exception as e:
            logger.warning(f"Upsampler not available ({e}), using single-stage")
            self.upsample_pipe = None

        # Enable VAE tiling to prevent OOM during decode
        self.pipe.vae.enable_tiling()

        self._loaded = True
        logger.info("LTX-2 loaded and ready")

    def unload(self):
        """Free VRAM."""
        for p in [self.pipe, self.pipe_i2v, self.upsample_pipe]:
            if p:
                del p
        self.pipe = self.pipe_i2v = self.upsample_pipe = None
        torch.cuda.empty_cache()
        self._loaded = False
        logger.info("LTX-2 unloaded")

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
        """Generate video from text prompt only."""
        if not self._loaded:
            self.load()

        num_frames = config.seconds_to_frames(duration_seconds, int(frame_rate))
        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)

        logger.info(
            f"T2V: {num_frames} frames @ {width}x{height}, "
            f"{num_inference_steps} steps, cfg={guidance_scale}"
        )

        # Stage 1: Generate
        video_latent, audio_latent = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="latent",
            return_dict=False,
        )

        # Stage 2: Upscale if available
        video_latent = self._upscale_latent(video_latent)

        # Decode and save
        return self._decode_and_save(
            video_latent, audio_latent, frame_rate, output_path
        )

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
        """Generate video from a starting image + text prompt."""
        if not self._loaded:
            self.load()

        num_frames = config.seconds_to_frames(duration_seconds, int(frame_rate))
        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)

        # Resize image to match video dimensions
        image = image.resize((width, height), Image.LANCZOS)

        logger.info(
            f"I2V: {num_frames} frames @ {width}x{height}, "
            f"{num_inference_steps} steps, cfg={guidance_scale}"
        )

        # Stage 1: Generate
        video_latent, audio_latent = self.pipe_i2v(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="latent",
            return_dict=False,
        )

        # Stage 2: Upscale
        video_latent = self._upscale_latent(video_latent)

        # Decode and save
        return self._decode_and_save(
            video_latent, audio_latent, frame_rate, output_path
        )

    def _upscale_latent(self, video_latent):
        """Apply stage 2 latent upsampling if available."""
        if self.upsample_pipe is None:
            return video_latent

        try:
            logger.info("Stage 2: Upscaling latent...")

            # Load distilled LoRA for stage 2
            self.pipe.load_lora_weights(
                config.VIDEO_MODEL_ID,
                adapter_name="stage_2_distilled",
                weight_name="ltx-2-19b-distilled-lora-384.safetensors",
            )
            self.pipe.set_adapters("stage_2_distilled", 1.0)

            upscaled = self.upsample_pipe(
                latents=video_latent,
                output_type="latent",
                return_dict=False,
            )[0]

            # Unload LoRA after stage 2
            self.pipe.unload_lora_weights()

            return upscaled
        except Exception as e:
            logger.warning(f"Upscale failed ({e}), using base output")
            return video_latent

    def _decode_and_save(
        self,
        video_latent,
        audio_latent,
        frame_rate: float,
        output_path: str,
    ) -> str:
        """Decode latents to pixels and save as MP4."""
        from diffusers.pipelines.ltx2.export_utils import encode_video

        logger.info("Decoding video...")

        # Decode video
        video = self.pipe.vae.decode(video_latent, return_dict=False)[0]

        # Decode audio
        audio = None
        if audio_latent is not None:
            try:
                audio = self.pipe.vae.decode_audio(audio_latent)
            except Exception:
                pass

        # Save
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        encode_video(
            video=video,
            fps=frame_rate,
            audio=audio,
            audio_sample_rate=44100,
            output_path=output_path,
        )

        logger.info(f"Video saved: {output_path}")
        return output_path

    @property
    def is_loaded(self) -> bool:
        return self._loaded
