"""Append-only audit log for AI-generated §10 종합검토의견 (역할 #3 narrator).

Every time the Narrative Interpreter calls the LLM (whether the candidate is
applied or rejected/fallen-back), one provenance record is appended here so the
AI-drafted prose is traceable — model, prompt hash, the prose-safe facts that
were fed in, the produced text, and the apply/fallback outcome.

Mirrors ``webapp/.../chat_audit_log.py``: durable JSONL with size rotation and
an aged-backup retention sweep. Disk write failures are logged and swallowed —
the audit must never break analysis.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()

# Durable JSONL bounds (env-overridable; read once at import).
#   - MAX_BYTES   <= 0 disables rotation (unbounded single file).
#   - BACKUP_COUNT <= 0 rotates by truncation (no .1/.2 backups kept).
#   - RETENTION_DAYS <= 0 disables the aged-backup sweep.
_MAX_BYTES = int(os.environ.get("NARRATION_AUDIT_MAX_BYTES", str(20 * 1024 * 1024)))
_BACKUP_COUNT = int(os.environ.get("NARRATION_AUDIT_BACKUP_COUNT", "5"))
_RETENTION_DAYS = float(os.environ.get("NARRATION_AUDIT_RETENTION_DAYS", "180"))


def _default_log_path() -> Path:
    # core/ -> mcp-server/ -> repo root; data/ lives at the repo root.
    return Path(__file__).resolve().parents[2] / "data" / "narration_audit" / "narration_audit.jsonl"


def _log_path() -> Path:
    raw = os.environ.get("NARRATION_AUDIT_LOG_PATH")
    return Path(raw) if raw else _default_log_path()


def _rotate_if_needed_locked(path: Path) -> None:
    """Size-based rollover for the durable JSONL file (caller holds _LOCK)."""
    if _MAX_BYTES <= 0 or not path.exists():
        return
    if path.stat().st_size < _MAX_BYTES:
        return
    backups = _BACKUP_COUNT
    if backups <= 0:
        path.unlink()
        return
    for i in range(backups - 1, 0, -1):
        src = Path(f"{path}.{i}")
        dst = Path(f"{path}.{i + 1}")
        if src.exists():
            if dst.exists():
                dst.unlink()
            src.rename(dst)
    first = Path(f"{path}.1")
    if first.exists():
        first.unlink()
    path.rename(first)


def _sweep_retention_locked(path: Path, now: float | None = None) -> None:
    """Delete numeric-suffix backups older than retention (caller holds _LOCK)."""
    if _RETENTION_DAYS <= 0:
        return
    now = time.time() if now is None else now
    cutoff = now - _RETENTION_DAYS * 86400.0
    prefix = path.name + "."
    parent = path.parent
    if not parent.exists():
        return
    for entry in parent.iterdir():
        name = entry.name
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix):]
        if not suffix.isdigit():
            continue
        try:
            if entry.stat().st_mtime < cutoff:
                entry.unlink()
        except OSError:
            continue


def append_narration_audit(record: dict) -> None:
    """Best-effort durable append of one narration provenance record.

    Adds ``created_at`` (epoch seconds). Never raises — disk failures are logged.
    """
    if not isinstance(record, dict):
        return
    stored = dict(record)
    stored.setdefault("created_at", time.time())
    try:
        path = _log_path()
        with _LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            _rotate_if_needed_locked(path)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(stored, ensure_ascii=False, default=str) + "\n")
            _sweep_retention_locked(path)
    except Exception as exc:  # noqa: BLE001 — audit must not break analysis
        logger.warning("failed to append narration audit JSONL: %s", exc, exc_info=True)
