#!/bin/bash
# GCP Startup Script — runs automatically when the spot instance boots.
# Mount persistent disk, pull latest code, start the worker.

set -e

MODELS_DISK="/dev/sdb"
MODELS_MOUNT="/mnt/models"
REPO_DIR="/opt/gpu-worker"
REPO_URL="${GPU_WORKER_REPO:-https://github.com/YOUR_ORG/gpu-worker.git}"

echo "[startup] Starting GPU worker setup..."

# ── Mount persistent disk (models survive instance preemption) ────────────────
if [ -b "$MODELS_DISK" ]; then
    mkdir -p "$MODELS_MOUNT"
    if ! mountpoint -q "$MODELS_MOUNT"; then
        mount "$MODELS_DISK" "$MODELS_MOUNT" || {
            echo "[startup] Formatting and mounting disk..."
            mkfs.ext4 -F "$MODELS_DISK"
            mount "$MODELS_DISK" "$MODELS_MOUNT"
        }
    fi
    echo "[startup] Persistent disk mounted at $MODELS_MOUNT"
else
    echo "[startup] No persistent disk found, using local storage"
    MODELS_MOUNT="$REPO_DIR/models"
    mkdir -p "$MODELS_MOUNT"
fi

# ── Pull latest code ──────────────────────────────────────────────────────────
if [ -d "$REPO_DIR" ]; then
    cd "$REPO_DIR"
    git pull origin main || true
else
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
    pip install -r requirements.txt
    pip install git+https://github.com/huggingface/diffusers.git
fi

# ── Create output directory ───────────────────────────────────────────────────
mkdir -p "$REPO_DIR/outputs"

# ── Start the server ──────────────────────────────────────────────────────────
echo "[startup] Starting GPU worker..."
cd "$REPO_DIR"

export MODELS_DIR="$MODELS_MOUNT"
export OUTPUT_DIR="$REPO_DIR/outputs"
export REDIS_URL="${REDIS_URL:-redis://10.0.0.2:6379/0}"
export SUPABASE_URL="${SUPABASE_URL}"
export SUPABASE_KEY="${SUPABASE_KEY}"

# Run with nohup so it survives SSH disconnection
nohup python main.py > /var/log/gpu-worker.log 2>&1 &

echo "[startup] GPU worker started (PID: $!)"
echo "[startup] Logs: tail -f /var/log/gpu-worker.log"
