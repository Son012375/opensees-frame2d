"""In-process cache for chat-driven section-change previews.

The Phase B chat tool ``propose_section_change`` writes a ``preview_id``
here so the frontend can fetch the full ``updated_model`` via HTTP. The
chat streaming layer rejects ``updated_model`` in NDJSON events (see
``chat/streaming.FORBIDDEN_KEYS``) — keeping the heavy payload out of
the chat stream and behind a regular JSON endpoint preserves that guard
while still letting the existing rec-diff modal Apply path consume it.

Entries expire after :data:`_CHAT_PREVIEW_TTL_SEC` (mirrors the
analysis_context_cache TTL). The cache and its lock live here (rather
than in main_simple) so chat tools can write without pulling
main_simple's import surface.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

chat_preview_cache: dict[str, dict] = {}
_CHAT_PREVIEW_TTL_SEC = 30 * 60  # 30 minutes — matches analysis context TTL
_CHAT_PREVIEW_LOCK = threading.Lock()


def _purge_expired_previews_locked() -> None:
    """Drop entries past their TTL. Caller must hold the lock."""
    now = time.time()
    expired = [
        k for k, v in chat_preview_cache.items()
        if v.get("expires_at", 0) < now
    ]
    for k in expired:
        chat_preview_cache.pop(k, None)


def put_preview(preview_id: str, entry: dict) -> None:
    """Insert a preview. Opportunistically purges expired entries first."""
    now = time.time()
    with _CHAT_PREVIEW_LOCK:
        _purge_expired_previews_locked()
        stored = dict(entry)
        stored["expires_at"] = now + _CHAT_PREVIEW_TTL_SEC
        stored["created_at"] = now
        chat_preview_cache[preview_id] = stored


def get_preview(preview_id: str) -> Optional[dict]:
    """TTL-aware read. Evicts on expiry, returns the entry otherwise."""
    now = time.time()
    with _CHAT_PREVIEW_LOCK:
        entry = chat_preview_cache.get(preview_id)
        if entry is None:
            return None
        if entry.get("expires_at", 0) < now:
            chat_preview_cache.pop(preview_id, None)
            return None
        return entry


def ever_existed(preview_id: str) -> bool:
    """Was this preview id ever in the cache (regardless of TTL)?

    Lets the GET endpoint distinguish 404 (never existed — typo / wrong
    server) from 410 (existed but expired — user took too long to click
    Apply). Returns True only when the entry is currently present — we
    don't keep an audit trail of evicted ids since the cost / leak risk
    isn't worth it.
    """
    with _CHAT_PREVIEW_LOCK:
        return preview_id in chat_preview_cache


def clear_cache() -> None:
    """Test helper. Drops all entries."""
    with _CHAT_PREVIEW_LOCK:
        chat_preview_cache.clear()
