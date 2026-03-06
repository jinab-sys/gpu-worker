"""
Webhook callbacks — notify the main backend when jobs complete.
"""

import logging
import httpx

logger = logging.getLogger("webhook")


async def fire_webhook(
    url: str,
    payload: dict,
    retries: int = 3,
) -> bool:
    """
    Send a POST to the webhook URL with the job result.
    Retries on failure. Returns True if delivered successfully.
    """
    if not url:
        return False

    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code < 400:
                    logger.info(f"Webhook delivered to {url}")
                    return True
                else:
                    logger.warning(
                        f"Webhook {url} returned {resp.status_code} "
                        f"(attempt {attempt + 1}/{retries})"
                    )
        except Exception as e:
            logger.warning(
                f"Webhook {url} failed: {e} (attempt {attempt + 1}/{retries})"
            )

    logger.error(f"Webhook {url} failed after {retries} attempts")
    return False
