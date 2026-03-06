# GPU Worker — AI Content Generation Service

Production-grade FastAPI server for generating images and videos using
FLUX.2 Klein 9B and LTX-2 19B. Designed to run on GCP A100 spot instances.

No ComfyUI. Pure Python. Direct GPU access.

## APIs

| Endpoint | Method | Description |
|---|---|---|
| `/image/edit` | POST | Face reference + prompt → new image (FLUX.2 Klein) |
| `/image/generate` | POST | Text prompt → image (FLUX.2 Klein) |
| `/video/i2v` | POST | Starting frame + prompt → video (LTX-2) |
| `/video/t2v` | POST | Text prompt → video (LTX-2) |
| `/video/pose` | POST | Reference video + subject → pose transfer (IC-LoRA) |
| `/video/extend` | POST | Existing video → extended video (chain clips) |
| `/job/{id}` | GET | Check job status and result URL |
| `/jobs` | GET | List jobs with filters |
| `/health` | GET | GPU status, VRAM, model info |
| `/admin/reload` | POST | Hot-swap models (e.g., LTX-2 → LTX-2.3) |

All endpoints return a `job_id` immediately. Generation runs async.
Results delivered via webhook callback to your main backend.

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR_ORG/gpu-worker.git
cd gpu-worker

# Install dependencies
pip install -r requirements.txt
pip install git+https://github.com/huggingface/diffusers.git

# For IC-LoRA pose transfer (optional):
pip install git+https://github.com/Lightricks/LTX-2.git

# Set environment
cp .env.example .env
# Edit .env with your Supabase and Redis URLs

# Run (models download automatically on first start, ~30 min)
python main.py
```

## Environment Variables

```bash
# Required
REDIS_URL=redis://localhost:6379/0

# Supabase (for file storage)
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=your-service-role-key

# Models (persistent disk recommended)
MODELS_DIR=/mnt/models
OUTPUT_DIR=./outputs

# GPU
DEVICE=cuda:0

# Optional: override default models
IMAGE_MODEL_ID=black-forest-labs/FLUX.2-klein-9B
VIDEO_MODEL_ID=Lightricks/LTX-2
# VIDEO_MODEL_ID=Lightricks/LTX-2.3  # When ready
```

## Usage Examples

### Generate an image from a face reference
```bash
curl -X POST http://GPU_IP:8000/image/edit \
  -H "Content-Type: application/json" \
  -d '{
    "reference_image_url": "https://your-bucket.supabase.co/faces/aisha.jpg",
    "prompt": "Professional business portrait, modern Dubai office, golden hour",
    "width": 768,
    "height": 1360,
    "webhook_url": "https://your-backend.com/webhooks/generation"
  }'
```

### Generate a video from an image
```bash
curl -X POST http://GPU_IP:8000/video/i2v \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://your-bucket.supabase.co/generated/aisha_photo.png",
    "prompt": "Close-up shot. She turns toward camera with a warm smile. Hair moves gently in breeze. Camera steady, shallow depth of field.",
    "duration_seconds": 5,
    "num_inference_steps": 25,
    "guidance_scale": 3.0,
    "webhook_url": "https://your-backend.com/webhooks/generation"
  }'
```

### Pose transfer
```bash
curl -X POST http://GPU_IP:8000/video/pose \
  -H "Content-Type: application/json" \
  -d '{
    "reference_video_url": "https://your-bucket.supabase.co/references/dance.mp4",
    "subject_image_url": "https://your-bucket.supabase.co/generated/aisha_portrait.png",
    "prompt": "A woman in elegant attire, soft studio lighting",
    "control_mode": "pose",
    "duration_seconds": 5,
    "webhook_url": "https://your-backend.com/webhooks/generation"
  }'
```

### Check job status
```bash
curl http://GPU_IP:8000/job/abc123def456
```

### Hot-swap to LTX-2.3
```bash
curl -X POST http://GPU_IP:8000/admin/reload \
  -H "Content-Type: application/json" \
  -d '{"model": "video", "model_id": "Lightricks/LTX-2.3"}'
```

## Webhook Payload

When a job completes, the worker POSTs to your `webhook_url`:

```json
{
  "job_id": "abc123def456",
  "type": "video_i2v",
  "status": "completed",
  "result_url": "https://your-bucket.supabase.co/generated/videos/aisha_v1.mp4",
  "seed": 42,
  "elapsed_seconds": 92.5,
  "completed_at": "2026-03-05T14:30:00"
}
```

On failure:
```json
{
  "job_id": "abc123def456",
  "status": "failed",
  "error": "CUDA out of memory"
}
```

## GCP Deployment

### Spot instance with persistent disk:

1. Create a 500GB persistent SSD for models
2. Create an A100 40GB spot instance
3. Attach the persistent disk
4. Set the startup script to `startup.sh`

Models download once to the persistent disk (~150GB).
If the spot instance is preempted, models survive.
On restart, just mounts the disk and starts serving (~2 min).

## Architecture

```
Your Backend (cheap VM)
    │
    ├── POST /image/edit ──→ GPU Worker ──→ Supabase ──→ webhook
    ├── POST /video/i2v  ──→ GPU Worker ──→ Supabase ──→ webhook
    └── GET /job/{id}    ──→ GPU Worker
```

## Scaling

| GPUs | Videos/day (5s) | Images/day | Cost/month (spot) |
|------|-----------------|------------|-------------------|
| 1× A100 | ~960 | ~28,800 | ~$600 |
| 2× A100 | ~1,920 | ~57,600 | ~$1,200 |
| 3× A100 | ~2,880 | ~86,400 | ~$1,800 |
