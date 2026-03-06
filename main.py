"""
GPU Worker — Production FastAPI Server
Endpoints for image generation, video generation, pose transfer.
Redis job queue. Webhook callbacks. Full parameter control.

Usage:
    python main.py

Environment:
    REDIS_URL=redis://localhost:6379/0
    SUPABASE_URL=https://xxx.supabase.co
    SUPABASE_KEY=your-key
    LOAD_IMAGE_MODEL=true   # set false on video-only worker
    LOAD_VIDEO_MODEL=true   # set false on image-only worker
"""

import os
import uuid
import time
import random
import asyncio
import logging
from datetime import datetime
from typing import Optional, Literal
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import redis.asyncio as aioredis

import config
from pipelines.image_pipe import ImagePipeline
from pipelines.video_pipe import VideoPipeline
from utils.storage import download_file_sync, upload_to_supabase, cleanup_temp
from utils.webhook import fire_webhook
from utils.media import generate_output_filename

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")

# ─── Feature flags ────────────────────────────────────────────────────────────

LOAD_IMAGE_MODEL = os.getenv("LOAD_IMAGE_MODEL", "true").lower() == "true"
LOAD_VIDEO_MODEL = os.getenv("LOAD_VIDEO_MODEL", "true").lower() == "true"

# ─── Global state ─────────────────────────────────────────────────────────────

image_pipe = ImagePipeline()
video_pipe = VideoPipeline()
redis_client: Optional[aioredis.Redis] = None
jobs: dict[str, dict] = {}

# ─── GPU Semaphore ────────────────────────────────────────────────────────────
# Only 1 job runs on the GPU at a time. All others queue and wait.
# This prevents OOM errors when many requests arrive simultaneously.
# To scale to 1000 videos/day: add more GPU worker instances behind a load balancer.
gpu_semaphore = asyncio.Semaphore(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client

    try:
        redis_client = aioredis.from_url(config.REDIS_URL, decode_responses=True)
        await redis_client.ping()
        logger.info(f"Redis connected: {config.REDIS_URL}")
    except Exception as e:
        logger.warning(f"Redis unavailable ({e}), using in-memory jobs only")
        redis_client = None

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    logger.info(f"Loading models (image={LOAD_IMAGE_MODEL}, video={LOAD_VIDEO_MODEL})...")

    if LOAD_IMAGE_MODEL:
        image_pipe.load()
        logger.info("Image model loaded")
    else:
        logger.info("Image model skipped (LOAD_IMAGE_MODEL=false)")

    if LOAD_VIDEO_MODEL:
        video_pipe.load()
        logger.info("Video model loaded")
    else:
        logger.info("Video model skipped (LOAD_VIDEO_MODEL=false)")

    logger.info("All models loaded. Server ready.")
    yield

    if LOAD_IMAGE_MODEL:
        image_pipe.unload()
    if LOAD_VIDEO_MODEL:
        video_pipe.unload()
    if redis_client:
        await redis_client.close()
    logger.info("Shutdown complete")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="GPU Worker",
    version="1.0.0",
    description="Image & video generation service. FLUX.2 Klein + LTX-2.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ─── Request Models ───────────────────────────────────────────────────────────

class ImageEditRequest(BaseModel):
    reference_image_url: str = Field(description="URL to reference/face image")
    prompt: str = Field(description="What to generate")
    width: int = Field(default=config.DEFAULT_IMAGE_WIDTH)
    height: int = Field(default=config.DEFAULT_IMAGE_HEIGHT)
    steps: int = Field(default=config.DEFAULT_IMAGE_STEPS, ge=1, le=50)
    guidance_scale: float = Field(default=config.DEFAULT_IMAGE_CFG, ge=0.0, le=20.0)
    seed: Optional[int] = Field(default=None)
    output_bucket: str = Field(default=config.DEFAULT_BUCKET)
    output_path: Optional[str] = Field(default=None)
    webhook_url: Optional[str] = Field(default=None)


class ImageGenerateRequest(BaseModel):
    prompt: str
    width: int = Field(default=config.DEFAULT_IMAGE_WIDTH)
    height: int = Field(default=config.DEFAULT_IMAGE_HEIGHT)
    steps: int = Field(default=config.DEFAULT_IMAGE_STEPS, ge=1, le=50)
    guidance_scale: float = Field(default=config.DEFAULT_IMAGE_CFG, ge=0.0, le=20.0)
    seed: Optional[int] = None
    output_bucket: str = config.DEFAULT_BUCKET
    output_path: Optional[str] = None
    webhook_url: Optional[str] = None


class VideoI2VRequest(BaseModel):
    image_url: str = Field(description="URL to starting frame image")
    prompt: str = Field(description="Motion/scene description")
    negative_prompt: str = Field(default=config.DEFAULT_NEGATIVE_PROMPT)
    width: int = Field(default=config.DEFAULT_VIDEO_WIDTH)
    height: int = Field(default=config.DEFAULT_VIDEO_HEIGHT)
    duration_seconds: int = Field(default=5, ge=3, le=20)
    frame_rate: float = Field(default=config.DEFAULT_VIDEO_FPS)
    num_inference_steps: int = Field(default=config.DEFAULT_VIDEO_STEPS, ge=4, le=50)
    guidance_scale: float = Field(default=config.DEFAULT_VIDEO_CFG, ge=0.0, le=10.0)
    seed: Optional[int] = None
    output_bucket: str = config.DEFAULT_BUCKET
    output_path: Optional[str] = None
    webhook_url: Optional[str] = None


class VideoT2VRequest(BaseModel):
    prompt: str
    negative_prompt: str = config.DEFAULT_NEGATIVE_PROMPT
    width: int = Field(default=config.DEFAULT_VIDEO_WIDTH)
    height: int = Field(default=config.DEFAULT_VIDEO_HEIGHT)
    duration_seconds: int = Field(default=5, ge=3, le=20)
    frame_rate: float = Field(default=config.DEFAULT_VIDEO_FPS)
    num_inference_steps: int = Field(default=config.DEFAULT_VIDEO_STEPS, ge=4, le=50)
    guidance_scale: float = Field(default=config.DEFAULT_VIDEO_CFG, ge=0.0, le=10.0)
    seed: Optional[int] = None
    output_bucket: str = config.DEFAULT_BUCKET
    output_path: Optional[str] = None
    webhook_url: Optional[str] = None


class VideoPoseRequest(BaseModel):
    reference_video_url: str = Field(description="URL to motion reference video")
    subject_image_url: Optional[str] = Field(default=None)
    prompt: str = Field(description="Style/appearance")
    control_mode: Literal["pose", "depth", "canny"] = Field(default="pose")
    width: int = Field(default=config.DEFAULT_VIDEO_WIDTH)
    height: int = Field(default=config.DEFAULT_VIDEO_HEIGHT)
    duration_seconds: int = Field(default=5, ge=3, le=20)
    frame_rate: float = Field(default=config.DEFAULT_VIDEO_FPS)
    guidance_scale: float = Field(default=3.0)
    ic_lora_strength: float = Field(default=1.0, ge=0.0, le=2.0)
    seed: Optional[int] = None
    output_bucket: str = config.DEFAULT_BUCKET
    output_path: Optional[str] = None
    webhook_url: Optional[str] = None


class VideoExtendRequest(BaseModel):
    video_url: str = Field(description="URL to video to extend")
    prompt: str = Field(description="Continuation description")
    negative_prompt: str = config.DEFAULT_NEGATIVE_PROMPT
    extend_seconds: int = Field(default=5, ge=3, le=20)
    num_inference_steps: int = Field(default=config.DEFAULT_VIDEO_STEPS)
    guidance_scale: float = Field(default=config.DEFAULT_VIDEO_CFG)
    seed: Optional[int] = None
    output_bucket: str = config.DEFAULT_BUCKET
    output_path: Optional[str] = None
    webhook_url: Optional[str] = None


class ModelReloadRequest(BaseModel):
    model: Literal["image", "video"]
    model_id: Optional[str] = None


# ─── Response ─────────────────────────────────────────────────────────────────

class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str = ""
    queue_position: int = 0


# ─── Job Management ───────────────────────────────────────────────────────────

def create_job(job_type: str) -> str:
    job_id = uuid.uuid4().hex[:12]
    queued = len([j for j in jobs.values() if j["status"] in ("queued", "processing")])
    jobs[job_id] = {
        "job_id": job_id,
        "type": job_type,
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "worker_id": config.WORKER_ID,
        "queue_position": queued,
    }
    return job_id


def update_job(job_id: str, **kwargs):
    if job_id in jobs:
        jobs[job_id].update(kwargs)


async def save_job_to_redis(job_id: str):
    if redis_client:
        await redis_client.hset(f"job:{job_id}", mapping={
            k: str(v) if v is not None else ""
            for k, v in jobs.get(job_id, {}).items()
        })
        await redis_client.expire(f"job:{job_id}", 86400)


# ─── Guard helpers ────────────────────────────────────────────────────────────

def _require_image():
    if not LOAD_IMAGE_MODEL:
        raise HTTPException(503, "Image model not loaded on this worker. Send to port 8000.")

def _require_video():
    if not LOAD_VIDEO_MODEL:
        raise HTTPException(503, "Video model not loaded on this worker. Send to port 8001.")


# ─── Background task runners ─────────────────────────────────────────────────

async def _run_image_edit(job_id: str, req: ImageEditRequest):
    temp_files = []
    try:
        async with gpu_semaphore:
            update_job(job_id, status="processing", started_at=datetime.now().isoformat())
            await save_job_to_redis(job_id)
            start = time.time()
            seed = req.seed if req.seed is not None else random.randint(0, 2**53)

            ref_path = download_file_sync(req.reference_image_url, suffix=".jpg")
            temp_files.append(ref_path)
            from PIL import Image
            ref_image = Image.open(ref_path).convert("RGB")

            local_path = os.path.join(config.OUTPUT_DIR, generate_output_filename("img_edit", "png"))
            await asyncio.to_thread(
                image_pipe.generate,
                prompt=req.prompt, width=req.width, height=req.height,
                steps=req.steps, guidance_scale=req.guidance_scale,
                seed=seed, reference_image=ref_image, output_path=local_path,
            )

            remote_path = req.output_path or f"images/{os.path.basename(local_path)}"
            result_url = upload_to_supabase(local_path, req.output_bucket, remote_path)
            elapsed = round(time.time() - start, 2)
            update_job(job_id, status="completed", result_url=result_url or local_path,
                       local_path=local_path, seed=seed, elapsed_seconds=elapsed,
                       completed_at=datetime.now().isoformat())
            await save_job_to_redis(job_id)
            if req.webhook_url:
                await fire_webhook(req.webhook_url, jobs[job_id])
            logger.info(f"Job {job_id} completed in {elapsed}s")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        update_job(job_id, status="failed", error=str(e))
        await save_job_to_redis(job_id)
        if req.webhook_url:
            await fire_webhook(req.webhook_url, jobs[job_id])
    finally:
        for f in temp_files:
            cleanup_temp(f)


async def _run_image_generate(job_id: str, req: ImageGenerateRequest):
    try:
        async with gpu_semaphore:
            update_job(job_id, status="processing", started_at=datetime.now().isoformat())
            await save_job_to_redis(job_id)
            start = time.time()
            seed = req.seed if req.seed is not None else random.randint(0, 2**53)

            local_path = os.path.join(config.OUTPUT_DIR, generate_output_filename("img_t2i", "png"))
            await asyncio.to_thread(
                image_pipe.generate,
                prompt=req.prompt, width=req.width, height=req.height,
                steps=req.steps, guidance_scale=req.guidance_scale,
                seed=seed, output_path=local_path,
            )

            remote_path = req.output_path or f"images/{os.path.basename(local_path)}"
            result_url = upload_to_supabase(local_path, req.output_bucket, remote_path)
            elapsed = round(time.time() - start, 2)
            update_job(job_id, status="completed", result_url=result_url or local_path,
                       local_path=local_path, seed=seed, elapsed_seconds=elapsed,
                       completed_at=datetime.now().isoformat())
            await save_job_to_redis(job_id)
            if req.webhook_url:
                await fire_webhook(req.webhook_url, jobs[job_id])
            logger.info(f"Job {job_id} completed in {elapsed}s")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        update_job(job_id, status="failed", error=str(e))
        await save_job_to_redis(job_id)
        if req.webhook_url:
            await fire_webhook(req.webhook_url, jobs[job_id])


async def _run_video_i2v(job_id: str, req: VideoI2VRequest):
    temp_files = []
    try:
        async with gpu_semaphore:
            update_job(job_id, status="processing", started_at=datetime.now().isoformat())
            await save_job_to_redis(job_id)
            start = time.time()
            seed = req.seed if req.seed is not None else random.randint(0, 2**53)

            img_path = download_file_sync(req.image_url, suffix=".png")
            temp_files.append(img_path)
            from PIL import Image
            image = Image.open(img_path).convert("RGB")

            local_path = os.path.join(config.OUTPUT_DIR, generate_output_filename("vid_i2v", "mp4"))
            await asyncio.to_thread(
                video_pipe.image_to_video,
                image=image, prompt=req.prompt, negative_prompt=req.negative_prompt,
                width=req.width, height=req.height, duration_seconds=req.duration_seconds,
                frame_rate=req.frame_rate, num_inference_steps=req.num_inference_steps,
                guidance_scale=req.guidance_scale, seed=seed, output_path=local_path,
            )

            remote_path = req.output_path or f"videos/{os.path.basename(local_path)}"
            result_url = upload_to_supabase(local_path, req.output_bucket, remote_path)
            elapsed = round(time.time() - start, 2)
            update_job(job_id, status="completed", result_url=result_url or local_path,
                       local_path=local_path, seed=seed, elapsed_seconds=elapsed,
                       completed_at=datetime.now().isoformat())
            await save_job_to_redis(job_id)
            if req.webhook_url:
                await fire_webhook(req.webhook_url, jobs[job_id])
            logger.info(f"I2V job {job_id} completed in {elapsed}s")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        update_job(job_id, status="failed", error=str(e))
        await save_job_to_redis(job_id)
        if req.webhook_url:
            await fire_webhook(req.webhook_url, jobs[job_id])
    finally:
        for f in temp_files:
            cleanup_temp(f)


async def _run_video_t2v(job_id: str, req: VideoT2VRequest):
    try:
        async with gpu_semaphore:
            update_job(job_id, status="processing", started_at=datetime.now().isoformat())
            await save_job_to_redis(job_id)
            start = time.time()
            seed = req.seed if req.seed is not None else random.randint(0, 2**53)

            local_path = os.path.join(config.OUTPUT_DIR, generate_output_filename("vid_t2v", "mp4"))
            await asyncio.to_thread(
                video_pipe.text_to_video,
                prompt=req.prompt, negative_prompt=req.negative_prompt,
                width=req.width, height=req.height, duration_seconds=req.duration_seconds,
                frame_rate=req.frame_rate, num_inference_steps=req.num_inference_steps,
                guidance_scale=req.guidance_scale, seed=seed, output_path=local_path,
            )

            remote_path = req.output_path or f"videos/{os.path.basename(local_path)}"
            result_url = upload_to_supabase(local_path, req.output_bucket, remote_path)
            elapsed = round(time.time() - start, 2)
            update_job(job_id, status="completed", result_url=result_url or local_path,
                       local_path=local_path, seed=seed, elapsed_seconds=elapsed,
                       completed_at=datetime.now().isoformat())
            await save_job_to_redis(job_id)
            if req.webhook_url:
                await fire_webhook(req.webhook_url, jobs[job_id])
            logger.info(f"T2V job {job_id} completed in {elapsed}s")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        update_job(job_id, status="failed", error=str(e))
        await save_job_to_redis(job_id)
        if req.webhook_url:
            await fire_webhook(req.webhook_url, jobs[job_id])


async def _run_video_pose(job_id: str, req: VideoPoseRequest):
    temp_files = []
    try:
        async with gpu_semaphore:
            update_job(job_id, status="processing", started_at=datetime.now().isoformat())
            await save_job_to_redis(job_id)
            start = time.time()
            seed = req.seed if req.seed is not None else random.randint(0, 2**53)

            from pipelines.pose_pipe import PosePipeline
            pose_pipe = PosePipeline()
            pose_pipe.load()

            ref_video_path = download_file_sync(req.reference_video_url, suffix=".mp4")
            temp_files.append(ref_video_path)

            subject_image = None
            if req.subject_image_url:
                from PIL import Image
                subj_path = download_file_sync(req.subject_image_url, suffix=".png")
                temp_files.append(subj_path)
                subject_image = Image.open(subj_path).convert("RGB")

            local_path = os.path.join(config.OUTPUT_DIR, generate_output_filename("vid_pose", "mp4"))
            await asyncio.to_thread(
                pose_pipe.transfer_pose,
                reference_video_path=ref_video_path, subject_image=subject_image,
                prompt=req.prompt, control_mode=req.control_mode,
                width=req.width, height=req.height, duration_seconds=req.duration_seconds,
                frame_rate=req.frame_rate, guidance_scale=req.guidance_scale,
                ic_lora_strength=req.ic_lora_strength, seed=seed, output_path=local_path,
            )
            pose_pipe.unload()

            remote_path = req.output_path or f"videos/{os.path.basename(local_path)}"
            result_url = upload_to_supabase(local_path, req.output_bucket, remote_path)
            elapsed = round(time.time() - start, 2)
            update_job(job_id, status="completed", result_url=result_url or local_path,
                       local_path=local_path, seed=seed, elapsed_seconds=elapsed,
                       completed_at=datetime.now().isoformat())
            await save_job_to_redis(job_id)
            if req.webhook_url:
                await fire_webhook(req.webhook_url, jobs[job_id])

    except Exception as e:
        logger.error(f"Pose job {job_id} failed: {e}", exc_info=True)
        update_job(job_id, status="failed", error=str(e))
        await save_job_to_redis(job_id)
        if req.webhook_url:
            await fire_webhook(req.webhook_url, jobs[job_id])
    finally:
        for f in temp_files:
            cleanup_temp(f)


async def _run_video_extend(job_id: str, req: VideoExtendRequest):
    temp_files = []
    try:
        async with gpu_semaphore:
            update_job(job_id, status="processing", started_at=datetime.now().isoformat())
            await save_job_to_redis(job_id)
            start = time.time()
            seed = req.seed if req.seed is not None else random.randint(0, 2**53)

            from pipelines.extend_pipe import extend_video
            src_path = download_file_sync(req.video_url, suffix=".mp4")
            temp_files.append(src_path)

            local_path = os.path.join(config.OUTPUT_DIR, generate_output_filename("vid_ext", "mp4"))
            await asyncio.to_thread(
                extend_video,
                video_pipe=video_pipe, source_video_path=src_path,
                prompt=req.prompt, extend_seconds=req.extend_seconds,
                negative_prompt=req.negative_prompt,
                num_inference_steps=req.num_inference_steps,
                guidance_scale=req.guidance_scale, seed=seed, output_path=local_path,
            )

            remote_path = req.output_path or f"videos/{os.path.basename(local_path)}"
            result_url = upload_to_supabase(local_path, req.output_bucket, remote_path)
            elapsed = round(time.time() - start, 2)
            update_job(job_id, status="completed", result_url=result_url or local_path,
                       local_path=local_path, seed=seed, elapsed_seconds=elapsed,
                       completed_at=datetime.now().isoformat())
            await save_job_to_redis(job_id)
            if req.webhook_url:
                await fire_webhook(req.webhook_url, jobs[job_id])

    except Exception as e:
        logger.error(f"Extend job {job_id} failed: {e}", exc_info=True)
        update_job(job_id, status="failed", error=str(e))
        await save_job_to_redis(job_id)
        if req.webhook_url:
            await fire_webhook(req.webhook_url, jobs[job_id])
    finally:
        for f in temp_files:
            cleanup_temp(f)


# ─── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    queued = len([j for j in jobs.values() if j["status"] == "queued"])
    processing = len([j for j in jobs.values() if j["status"] == "processing"])
    return {
        "status": "healthy",
        "worker_id": config.WORKER_ID,
        "gpu": gpu_name,
        "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
        "vram_allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 1),
        "image_model": config.IMAGE_MODEL_ID if image_pipe.is_loaded else "not loaded",
        "video_model": config.VIDEO_MODEL_ID if video_pipe.is_loaded else "not loaded",
        "queued_jobs": queued,
        "processing_jobs": processing,
        "total_jobs": len(jobs),
    }


@app.post("/image/edit", response_model=JobResponse)
async def image_edit(req: ImageEditRequest):
    _require_image()
    job_id = create_job("image_edit")
    asyncio.create_task(_run_image_edit(job_id, req))
    q = jobs[job_id]["queue_position"]
    return JobResponse(job_id=job_id, status="queued", message=f"Position {q} in queue", queue_position=q)


@app.post("/image/generate", response_model=JobResponse)
async def image_generate(req: ImageGenerateRequest):
    _require_image()
    job_id = create_job("image_generate")
    asyncio.create_task(_run_image_generate(job_id, req))
    q = jobs[job_id]["queue_position"]
    return JobResponse(job_id=job_id, status="queued", message=f"Position {q} in queue", queue_position=q)


@app.post("/video/i2v", response_model=JobResponse)
async def video_i2v(req: VideoI2VRequest):
    _require_video()
    job_id = create_job("video_i2v")
    asyncio.create_task(_run_video_i2v(job_id, req))
    q = jobs[job_id]["queue_position"]
    return JobResponse(job_id=job_id, status="queued", message=f"Position {q} in queue", queue_position=q)


@app.post("/video/t2v", response_model=JobResponse)
async def video_t2v(req: VideoT2VRequest):
    _require_video()
    job_id = create_job("video_t2v")
    asyncio.create_task(_run_video_t2v(job_id, req))
    q = jobs[job_id]["queue_position"]
    return JobResponse(job_id=job_id, status="queued", message=f"Position {q} in queue", queue_position=q)


@app.post("/video/pose", response_model=JobResponse)
async def video_pose(req: VideoPoseRequest):
    _require_video()
    job_id = create_job("video_pose")
    asyncio.create_task(_run_video_pose(job_id, req))
    q = jobs[job_id]["queue_position"]
    return JobResponse(job_id=job_id, status="queued", message=f"Position {q} in queue", queue_position=q)


@app.post("/video/extend", response_model=JobResponse)
async def video_extend(req: VideoExtendRequest):
    _require_video()
    job_id = create_job("video_extend")
    asyncio.create_task(_run_video_extend(job_id, req))
    q = jobs[job_id]["queue_position"]
    return JobResponse(job_id=job_id, status="queued", message=f"Position {q} in queue", queue_position=q)


@app.get("/job/{job_id}")
async def get_job(job_id: str):
    if job_id in jobs:
        return jobs[job_id]
    if redis_client:
        data = await redis_client.hgetall(f"job:{job_id}")
        if data:
            return data
    raise HTTPException(404, "Job not found")


@app.get("/jobs")
async def list_jobs(status: Optional[str] = None, job_type: Optional[str] = None, limit: int = 50):
    result = list(jobs.values())
    if status:
        result = [j for j in result if j.get("status") == status]
    if job_type:
        result = [j for j in result if j.get("type") == job_type]
    return sorted(result, key=lambda x: x.get("created_at", ""), reverse=True)[:limit]


@app.post("/admin/reload")
async def reload_model(req: ModelReloadRequest):
    if req.model == "image":
        image_pipe.unload()
        if req.model_id:
            config.IMAGE_MODEL_ID = req.model_id
        image_pipe.load()
        return {"status": "reloaded", "model": config.IMAGE_MODEL_ID}
    elif req.model == "video":
        video_pipe.unload()
        if req.model_id:
            config.VIDEO_MODEL_ID = req.model_id
        video_pipe.load()
        return {"status": "reloaded", "model": config.VIDEO_MODEL_ID}
    raise HTTPException(400, "model must be 'image' or 'video'")


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT, workers=config.WORKERS)
