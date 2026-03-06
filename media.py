"""
Media processing helpers.
"""

import os
import uuid
from datetime import datetime
from PIL import Image


def generate_output_filename(prefix: str, ext: str = "png") -> str:
    """Generate a unique timestamped filename."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    return f"{prefix}_{ts}_{short_id}.{ext}"


def resize_image_to_fit(image: Image.Image, width: int, height: int) -> Image.Image:
    """Resize and center-crop an image to exact dimensions."""
    target_ratio = width / height
    img_ratio = image.width / image.height

    if img_ratio > target_ratio:
        # Image is wider — crop sides
        new_width = int(image.height * target_ratio)
        left = (image.width - new_width) // 2
        image = image.crop((left, 0, left + new_width, image.height))
    else:
        # Image is taller — crop top/bottom
        new_height = int(image.width / target_ratio)
        top = (image.height - new_height) // 2
        image = image.crop((0, top, image.width, top + new_height))

    return image.resize((width, height), Image.LANCZOS)


def extract_last_frame(video_path: str) -> str:
    """Extract the last frame from a video file. Returns path to image."""
    import imageio.v3 as iio

    frames = iio.imread(video_path, plugin="pyav")
    last_frame = Image.fromarray(frames[-1])

    out_path = video_path.rsplit(".", 1)[0] + "_last_frame.png"
    last_frame.save(out_path)
    return out_path


def extract_first_frame(video_path: str) -> str:
    """Extract the first frame from a video file. Returns path to image."""
    import imageio.v3 as iio

    frames = iio.imread(video_path, plugin="pyav")
    first_frame = Image.fromarray(frames[0])

    out_path = video_path.rsplit(".", 1)[0] + "_first_frame.png"
    first_frame.save(out_path)
    return out_path
