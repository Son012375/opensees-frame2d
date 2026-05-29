"""Provider-isolated audit log for KDS chat evidence.

The chat orchestrator intentionally keeps citation quotes out of
provider-visible history. This module preserves those quotes for audit
and research workflows in a separate store that is never read by
``ChatOrchestrator._provider_messages``.

The in-memory view is TTL-bound for live API queries. A JSONL append-only
copy is also written for durable provenance; disk write failures are
logged but never allowed to break the chat tool.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)

chat_audit_log: dict[str, list[dict]] = {}
_CHAT_AUDIT_TTL_SEC = 30 * 60
_CHAT_AUDIT_MAX_PER_ANALYSIS = 200
_CHAT_AUDIT_LOCK = threading.Lock()


def _default_log_path() -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "data"
        / "chat_audit"
        / "evidence_audit.jsonl"
    )


def _log_path() -> Path:
    raw = os.environ.get("CHAT_AUDIT_LOG_PATH")
    return Path(raw) if raw else _default_log_path()


def _purge_expired_locked(now: Optional[float] = None) -> None:
    """Drop expired records from all analysis buckets.

    Caller must hold :data:`_CHAT_AUDIT_LOCK`.
    """
    now = time.time() if now is None else now
    for analysis_id, bucket in list(chat_audit_log.items()):
        kept = [r for r in bucket if r.get("expires_at", 0) >= now]
        if kept:
            chat_audit_log[analysis_id] = kept
        else:
            chat_audit_log.pop(analysis_id, None)


def _append_jsonl_locked(record: dict) -> None:
    """Best-effort durable append.

    Caller must hold :data:`_CHAT_AUDIT_LOCK`.
    """
    try:
        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001 - audit must not break chat
        logger.warning("failed to append chat audit JSONL: %s", exc, exc_info=True)


def append_audit(record: dict) -> str:
    """Store one evidence audit record and return its ``audit_id``.

    The caller supplies the provenance payload. This function adds
    ``audit_id``, ``created_at``, and ``expires_at`` before storing.
    """
    if not isinstance(record, dict):
        raise TypeError(f"audit record must be a dict, got {type(record).__name__}")

    now = time.time()
    stored = copy.deepcopy(record)
    audit_id = "audit_" + uuid.uuid4().hex[:16]
    stored["audit_id"] = audit_id
    stored["created_at"] = now
    stored["expires_at"] = now + _CHAT_AUDIT_TTL_SEC

    analysis_id = str(stored.get("analysis_id") or "")
    with _CHAT_AUDIT_LOCK:
        _purge_expired_locked(now)
        bucket = chat_audit_log.setdefault(analysis_id, [])
        bucket.append(stored)
        if len(bucket) > _CHAT_AUDIT_MAX_PER_ANALYSIS:
            del bucket[: len(bucket) - _CHAT_AUDIT_MAX_PER_ANALYSIS]
        _append_jsonl_locked(stored)

    return audit_id


def query_audit(
    analysis_id: str,
    *,
    member_id: Optional[int] = None,
    turn: Optional[int] = None,
) -> list[dict]:
    """Return TTL-live audit records for an analysis.

    ``member_id`` and ``turn`` are optional filters. Records are returned
    in append order.
    """
    with _CHAT_AUDIT_LOCK:
        _purge_expired_locked()
        bucket = list(chat_audit_log.get(str(analysis_id), []))

    out: list[dict] = []
    for record in bucket:
        if member_id is not None and record.get("member_id") != member_id:
            continue
        if turn is not None and record.get("turn") != turn:
            continue
        out.append(copy.deepcopy(record))
    return out


def clear_cache() -> None:
    """Test helper. Drops only the TTL-live in-memory cache."""
    with _CHAT_AUDIT_LOCK:
        chat_audit_log.clear()
