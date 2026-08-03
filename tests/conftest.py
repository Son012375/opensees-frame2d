"""Shared pytest fixtures.

Narration sidecar isolation: the §10 narrator now has a durable facts-hash
cache (narration_cache.py, C2′) and an append-only audit log (narration_audit.py,
C3) that both default to repo-root/data/. Without isolation, a real narrator
built in one test (build_claude_narrator exposes prompt_hash, so caching activates)
would read/write the real data dir and could return a candidate cached by a
prior test for the same facts+prompt — cross-test contamination. This autouse
fixture redirects both to a per-test tmp dir so every test starts clean. Tests
that need specific behaviour override these env vars themselves (monkeypatch wins).
"""
import pytest


@pytest.fixture(autouse=True)
def _isolate_narration_sidecars(tmp_path, monkeypatch):
    monkeypatch.setenv("NARRATION_CACHE_DIR", str(tmp_path / "narration_cache"))
    monkeypatch.setenv("NARRATION_AUDIT_LOG_PATH", str(tmp_path / "narration_audit.jsonl"))
    monkeypatch.delenv("NARRATION_CACHE_DISABLE", raising=False)
    yield
