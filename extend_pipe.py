"""
Video Extension Pipeline.
Extends a video by using the last frame as the start of a new clip,
then concatenates the clips.
"""

import os
import logging
from typing import Optional
from PIL import Image

from pipelines.video_pipe import VideoPipeline
from utils.media import extract_last_frame

import config

logger = logging.getLogger("extend_pipe")


def extend_video(
    video_pipe: VideoPipeline,
    source_video_path: str,
    prompt: str,
    extend_seconds: int = 5,
    negative_prompt: str = config.DEFAULT_NEGATIVE_PROMPT,
    width: int = config.DEFAULT_VIDEO_WIDTH,
    height: int = config.DEFAULT_VIDEO_HEIGHT,
    frame_rate: float = config.DEFAULT_VIDEO_FPS,
    num_inference_steps: int = config.DEFAULT_VIDEO_STEPS,
    guidance_scale: float = config.DEFAULT_VIDEO_CFG,
    seed: Optional[int] = None,
    output_path: str = "extended.mp4",
) -> str:
    """
    Extend a video by generating a continuation from its last frame.

    Args:
        video_pipe: Loaded VideoPipeline instance
        source_video_path: Path to the original video
        prompt: Continuation prompt
        extend_seconds: How many seconds to add
        output_path: Where to save the combined result

    Returns:
        Path to the extended video
    """
    # Extract last frame
    last_frame_path = extract_last_frame(source_video_path)
    last_frame = Image.open(last_frame_path)

    # Generate continuation
    extension_path = output_path.replace(".mp4", "_ext.mp4")
    video_pipe.image_to_video(
        image=last_frame,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        duration_seconds=extend_seconds,
        frame_rate=frame_rate,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        output_path=extension_path,
    )

    # Concatenate original + extension using ffmpeg
    concat_videos(source_video_path, extension_path, output_path)

    # Cleanup
    if os.path.exists(extension_path):
        os.unlink(extension_path)
    if os.path.exists(last_frame_path):
        os.unlink(last_frame_path)

    logger.info(f"Extended video saved: {output_path}")
    return output_path


def concat_videos(video1: str, video2: str, output: str):
    """Concatenate two videos using ffmpeg."""
    import subprocess
    import tempfile

    # Create concat file
    concat_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    )
    concat_file.write(f"file '{os.path.abspath(video1)}'\n")
    concat_file.write(f"file '{os.path.abspath(video2)}'\n")
    concat_file.close()

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_file.name,
        "-c", "copy",
        output,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    os.unlink(concat_file.name)

    if result.returncode != 0:
        # Fallback: re-encode if copy fails
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_file.name,
            "-c:v", "libx264", "-crf", "18",
            "-c:a", "aac",
            output,
        ]
        subprocess.run(cmd, capture_output=True)

    logger.info(f"Concatenated {video1} + {video2} -> {output}")
