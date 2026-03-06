"""
Storage utilities — download from URLs, upload to Supabase.
"""

import os
import uuid
import logging
import tempfile
from typing import Optional

import httpx

logger = logging.getLogger("storage")

_supabase_client = None


def get_supabase():
    """Lazy-init Supabase client."""
    global _supabase_client
    if _supabase_client is None:
        from config import SUPABASE_URL, SUPABASE_KEY
        if SUPABASE_URL and SUPABASE_KEY:
            from supabase import create_client
            _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase_client


async def download_file(url: str, suffix: str = ".jpg") -> str:
    """
    Download a file from URL to a temp path.
    Returns the local file path.
    """
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()

    # Determine suffix from URL or content-type
    if "." in url.split("/")[-1].split("?")[0]:
        ext = "." + url.split("/")[-1].split("?")[0].rsplit(".", 1)[-1]
        if len(ext) > 6:
            ext = suffix
    else:
        ext = suffix

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(resp.content)
    tmp.close()
    logger.debug(f"Downloaded {url} -> {tmp.name} ({len(resp.content)} bytes)")
    return tmp.name


def download_file_sync(url: str, suffix: str = ".jpg") -> str:
    """Synchronous version for use inside pipeline threads."""
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()

    if "." in url.split("/")[-1].split("?")[0]:
        ext = "." + url.split("/")[-1].split("?")[0].rsplit(".", 1)[-1]
        if len(ext) > 6:
            ext = suffix
    else:
        ext = suffix

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(resp.content)
    tmp.close()
    return tmp.name


def upload_to_supabase(
    local_path: str,
    bucket: str,
    remote_path: str,
) -> Optional[str]:
    """
    Upload a file to Supabase Storage.
    Returns the public URL or None if Supabase is not configured.
    """
    sb = get_supabase()
    if not sb:
        logger.warning("Supabase not configured, skipping upload")
        return None

    try:
        with open(local_path, "rb") as f:
            data = f.read()

        # Determine content type
        if remote_path.endswith(".mp4"):
            content_type = "video/mp4"
        elif remote_path.endswith(".png"):
            content_type = "image/png"
        elif remote_path.endswith(".jpg") or remote_path.endswith(".jpeg"):
            content_type = "image/jpeg"
        else:
            content_type = "application/octet-stream"

        sb.storage.from_(bucket).upload(
            remote_path,
            data,
            file_options={"content-type": content_type},
        )

        url = sb.storage.from_(bucket).get_public_url(remote_path)
        logger.info(f"Uploaded to Supabase: {remote_path} -> {url}")
        return url

    except Exception as e:
        logger.error(f"Supabase upload failed: {e}")
        return None


def cleanup_temp(path: str):
    """Remove temp file if it exists."""
    try:
        if path and os.path.exists(path) and "/tmp" in path:
            os.unlink(path)
    except Exception:
        pass
