"""
FLUX.2 Klein 9B Image Pipeline.
Handles text-to-image and image editing with reference images.
~3 seconds per image on A100.
"""

import os
import torch
import logging
from typing import Optional
from PIL import Image

import config

logger = logging.getLogger("image_pipe")


class ImagePipeline:
    def __init__(self):
        self.device = config.DEVICE
        self.pipe = None
        self._loaded = False

    def load(self):
        """Load FLUX.2 Klein model into VRAM."""
        if self._loaded:
            return

        from diffusers import Flux2KleinPipeline

        logger.info(f"Loading {config.IMAGE_MODEL_ID}...")

        self.pipe = Flux2KleinPipeline.from_pretrained(
            config.IMAGE_MODEL_ID,
            torch_dtype=torch.bfloat16,
            cache_dir=config.MODELS_DIR,
        )
        self.pipe.to(self.device)

        self._loaded = True
        logger.info("FLUX.2 Klein loaded and ready")

    def unload(self):
        """Free VRAM."""
        if self.pipe:
            del self.pipe
            self.pipe = None
        torch.cuda.empty_cache()
        self._loaded = False
        logger.info("FLUX.2 Klein unloaded")

    def generate(
        self,
        prompt: str,
        width: int = config.DEFAULT_IMAGE_WIDTH,
        height: int = config.DEFAULT_IMAGE_HEIGHT,
        steps: int = config.DEFAULT_IMAGE_STEPS,
        guidance_scale: float = config.DEFAULT_IMAGE_CFG,
        seed: Optional[int] = None,
        reference_image: Optional[Image.Image] = None,
        output_path: Optional[str] = None,
    ) -> Image.Image:
        """
        Generate an image.

        Args:
            prompt: Text description
            width: Output width (default 768)
            height: Output height (default 1360 for 9:16)
            steps: Inference steps (4 for distilled, 25-50 for base)
            guidance_scale: CFG (1.0 for distilled, 4.0 for base)
            seed: Random seed for reproducibility
            reference_image: PIL Image for editing mode
            output_path: Save path (optional, also returns the Image)

        Returns:
            PIL Image
        """
        if not self._loaded:
            self.load()

        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)
        else:
            generator.seed()

        kwargs = dict(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=generator,
        )

        if reference_image is not None:
            kwargs["image"] = reference_image

        result = self.pipe(**kwargs)
        image = result.images[0]

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            image.save(output_path, quality=95)
            logger.info(f"Image saved: {output_path}")

        return image

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def vram_usage_gb(self) -> float:
        if not self._loaded:
            return 0.0
        return torch.cuda.memory_allocated(0) / 1e9
