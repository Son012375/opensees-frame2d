"""Tests for the provider-isolated KDS chat evidence audit log."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webapp" / "backend"))

from app.services import chat_audit_log as audit  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_audit(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAT_AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    audit.clear_cache()
    yield
    audit.clear_cache()


def _record(
    analysis_id: str = "analysis_a",
    *,
    member_id: int = 5,
    turn: int = 1,
    quote: str = "Shear strength provisions.",
) -> dict:
    return {
        "analysis_id": analysis_id,
        "member_id": member_id,
        "turn": turn,
        "session_id": "chat_test",
        "member": {"member_id": member_id, "section": "H-300x300"},
        "trigger": {
            "status": "NG",
            "governing_ratio": "shear",
            "issue_type": "shear_exceeded",
            "ratios": {"shear": 1.2, "interaction": 0.5},
        },
        "query": {
            "query_id": "q1",
            "query_text": "shear column H-300x300",
            "topic": "member_shear",
            "limit_state": "shear_strength",
        },
        "rag_used": True,
        "evidence": [{
            "doc_id": "KDS 41 31 00",
            "clause": "G2",
            "title": "shear",
            "quote": quote,
            "score": 0.9,
        }],
        "warnings": [],
    }


def test_append_query_and_jsonl_durable_copy(tmp_path):
    path = tmp_path / "audit.jsonl"
    audit_id = audit.append_audit(
        _record(quote="\ud55c\uae00 quote for durability"),
    )

    records = audit.query_audit("analysis_a")
    assert len(records) == 1
    assert records[0]["audit_id"] == audit_id
    assert records[0]["created_at"] > 0
    assert records[0]["expires_at"] > records[0]["created_at"]
    assert records[0]["evidence"][0]["quote"] == "\ud55c\uae00 quote for durability"

    raw = path.read_text(encoding="utf-8")
    line = json.loads(raw)
    assert line["audit_id"] == audit_id
    assert line["evidence"][0]["quote"] == "\ud55c\uae00 quote for durability"
    assert "\\ud55c" not in raw


def test_query_filters_by_analysis_member_and_turn():
    audit.append_audit(_record("analysis_a", member_id=5, turn=1))
    audit.append_audit(_record("analysis_a", member_id=6, turn=2))
    audit.append_audit(_record("analysis_b", member_id=5, turn=1))

    assert len(audit.query_audit("analysis_a")) == 2
    assert [r["member_id"] for r in audit.query_audit("analysis_a", member_id=5)] == [5]
    assert [r["turn"] for r in audit.query_audit("analysis_a", turn=2)] == [2]
    assert len(audit.query_audit("analysis_b")) == 1


def test_ttl_purge_removes_expired_records(monkeypatch):
    monkeypatch.setattr(audit, "_CHAT_AUDIT_TTL_SEC", -1)
    audit.append_audit(_record("analysis_expired"))

    assert audit.query_audit("analysis_expired") == []


def test_bucket_cap_drops_oldest(monkeypatch):
    monkeypatch.setattr(audit, "_CHAT_AUDIT_MAX_PER_ANALYSIS", 2)
    audit.append_audit(_record("analysis_cap", member_id=1))
    audit.append_audit(_record("analysis_cap", member_id=2))
    audit.append_audit(_record("analysis_cap", member_id=3))

    records = audit.query_audit("analysis_cap")
    assert [r["member_id"] for r in records] == [2, 3]


def test_clear_cache_only_clears_live_view(tmp_path):
    audit.append_audit(_record("analysis_clear"))
    audit.clear_cache()

    assert audit.query_audit("analysis_clear") == []
    assert os.environ["CHAT_AUDIT_LOG_PATH"]
    assert (tmp_path / "audit.jsonl").exists()


# ---------------------------------------------------------------------------
# JSONL rotation + retention (operational hardening)
# ---------------------------------------------------------------------------

def test_size_rotation_creates_backups_and_honors_count(tmp_path, monkeypatch):
    # max_bytes tiny so every append after the first rotates; keep 2 backups.
    monkeypatch.setattr(audit, "_CHAT_AUDIT_MAX_BYTES", 10)
    monkeypatch.setattr(audit, "_CHAT_AUDIT_BACKUP_COUNT", 2)

    base = tmp_path / "audit.jsonl"
    for member in (1, 2, 3, 4):
        audit.append_audit(_record("analysis_rot", member_id=member))

    assert base.exists()
    assert Path(f"{base}.1").exists()
    assert Path(f"{base}.2").exists()
    # backup count is capped at 2 -> .3 must never appear
    assert not Path(f"{base}.3").exists()
    # live file holds only the most recent record
    last = json.loads(base.read_text(encoding="utf-8").strip())
    assert last["member_id"] == 4


def test_rotation_with_zero_backups_truncates(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "_CHAT_AUDIT_MAX_BYTES", 10)
    monkeypatch.setattr(audit, "_CHAT_AUDIT_BACKUP_COUNT", 0)

    base = tmp_path / "audit.jsonl"
    audit.append_audit(_record("analysis_trunc", member_id=1))
    audit.append_audit(_record("analysis_trunc", member_id=2))

    assert base.exists()
    assert not Path(f"{base}.1").exists()
    lines = [l for l in base.read_text(encoding="utf-8").splitlines() if l]
    assert len(lines) == 1
    assert json.loads(lines[0])["member_id"] == 2


def test_rotation_disabled_when_max_bytes_nonpositive(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "_CHAT_AUDIT_MAX_BYTES", 0)
    base = tmp_path / "audit.jsonl"
    for member in (1, 2, 3):
        audit.append_audit(_record("analysis_unbounded", member_id=member))

    assert not Path(f"{base}.1").exists()
    lines = [l for l in base.read_text(encoding="utf-8").splitlines() if l]
    assert len(lines) == 3


def test_retention_sweep_deletes_aged_numeric_backups_only(tmp_path, monkeypatch):
    # No rotation during the append (big cap); retention deletes old backups.
    monkeypatch.setattr(audit, "_CHAT_AUDIT_MAX_BYTES", 50 * 1024 * 1024)
    monkeypatch.setattr(audit, "_CHAT_AUDIT_RETENTION_DAYS", 1)

    base = tmp_path / "audit.jsonl"
    aged_backup = Path(f"{base}.1")
    non_numeric = Path(f"{base}.bak")
    aged_backup.write_text("old\n", encoding="utf-8")
    non_numeric.write_text("keep\n", encoding="utf-8")
    old = time.time() - 2 * 86400
    os.utime(aged_backup, (old, old))
    os.utime(non_numeric, (old, old))

    audit.append_audit(_record("analysis_retain"))

    assert not aged_backup.exists(), "aged numeric backup must be swept"
    assert non_numeric.exists(), "non-numeric file must be preserved"
    assert base.exists(), "live base file is never deleted by retention"


def test_retention_disabled_keeps_aged_backups(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "_CHAT_AUDIT_MAX_BYTES", 50 * 1024 * 1024)
    monkeypatch.setattr(audit, "_CHAT_AUDIT_RETENTION_DAYS", 0)

    base = tmp_path / "audit.jsonl"
    aged_backup = Path(f"{base}.1")
    aged_backup.write_text("old\n", encoding="utf-8")
    old = time.time() - 999 * 86400
    os.utime(aged_backup, (old, old))

    audit.append_audit(_record("analysis_noretain"))

    assert aged_backup.exists(), "retention<=0 disables the sweep"
