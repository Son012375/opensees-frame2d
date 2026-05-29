"""Phase A.1 chat-router smoke tests.

NoopProvider only (Phase A.3 adds the real Ollama path). Covers the
route + NDJSON contract + session lifecycle + a few invariants that
would catch wiring regressions when later phases extend the route.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webapp" / "backend"))
sys.path.insert(0, str(ROOT / "mcp-server"))

# Force the Noop provider in case the dev env exports something else.
os.environ["CHAT_LLM_PROVIDER"] = "noop"

from app.main_simple import app  # noqa: E402
from app.services.chat_session import (  # noqa: E402
    _CHAT_SESSION_LOCK,
    chat_session_cache,
)
from app.services.chat_audit_log import (  # noqa: E402
    append_audit,
    clear_cache as clear_audit_cache,
)


@pytest.fixture(autouse=True)
def _isolate_chat_sessions(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAT_AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    # Default to open/operator mode; token-gated tests opt in via setenv.
    monkeypatch.delenv("DEMO_AUTH_TOKEN", raising=False)
    clear_audit_cache()
    with _CHAT_SESSION_LOCK:
        chat_session_cache.clear()
    yield
    clear_audit_cache()
    with _CHAT_SESSION_LOCK:
        chat_session_cache.clear()


@pytest.fixture
def client():
    return TestClient(app)


def _parse_ndjson(body: str) -> list[dict]:
    return [json.loads(line) for line in body.split("\n") if line]


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

def test_create_session_returns_id_provider_and_ttl(client):
    r = client.post("/api/v2/chat/sessions", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"].startswith("chat_")
    assert body["provider"] == "noop"
    assert body["expires_at"] > 0
    assert body["analysis_id"] is None


def test_create_session_optional_analysis_id_round_trips(client):
    r = client.post(
        "/api/v2/chat/sessions",
        json={"analysis_id": "analysis_test_001"},
    )
    sid = r.json()["session_id"]
    g = client.get(f"/api/v2/chat/sessions/{sid}")
    assert g.status_code == 200
    assert g.json()["analysis_id"] == "analysis_test_001"


def test_debug_get_unknown_session_returns_404(client):
    r = client.get("/api/v2/chat/sessions/chat_doesnotexist")
    assert r.status_code == 404


def test_get_audit_endpoint_returns_filtered_records(client):
    append_audit({
        "analysis_id": "analysis_audit_001",
        "member_id": 5,
        "turn": 2,
        "session_id": "chat_test",
        "member": {"member_id": 5, "section": "H-300x300"},
        "trigger": {
            "status": "NG",
            "governing_ratio": "shear",
            "issue_type": "shear_exceeded",
            "ratios": {"shear": 1.2},
        },
        "query": {"query_text": "shear", "topic": "member_shear"},
        "rag_used": True,
        "evidence": [{"doc_id": "KDS 14 31 10", "clause": "4.3.2.1.2",
                      "quote": "Shear strength provisions."}],
        "warnings": [],
    })
    append_audit({
        "analysis_id": "analysis_audit_001",
        "member_id": 6,
        "turn": 2,
        "member": {"member_id": 6},
        "trigger": {"status": "OK", "ratios": {}},
        "query": {},
        "rag_used": False,
        "evidence": [],
        "warnings": [],
    })

    # Default: quote bodies REDACTED (sensitive), provenance metadata kept.
    r = client.get("/api/v2/chat/audit/analysis_audit_001?member_id=5")
    assert r.status_code == 200
    body = r.json()
    assert body["analysis_id"] == "analysis_audit_001"
    assert body["member_id"] == 5
    assert body["turn"] is None
    assert body["include_quotes"] is False
    assert len(body["records"]) == 1
    ev = body["records"][0]["evidence"][0]
    assert ev["quote"] is None, "quote must be redacted by default"
    assert ev["doc_id"] == "KDS 14 31 10"   # metadata survives redaction
    assert ev["clause"] == "4.3.2.1.2"


def test_get_audit_endpoint_include_quotes_returns_full(client):
    append_audit({
        "analysis_id": "analysis_audit_iq",
        "member_id": 5, "turn": 1,
        "member": {"member_id": 5}, "trigger": {"ratios": {}},
        "query": {}, "rag_used": True,
        "evidence": [{"doc_id": "KDS 14 31 10", "clause": "4.3.2.1.2",
                      "quote": "Shear strength provisions."}],
        "warnings": [],
    })
    r = client.get(
        "/api/v2/chat/audit/analysis_audit_iq?include_quotes=true"
    )
    assert r.status_code == 200
    body = r.json()
    assert body["include_quotes"] is True
    assert body["records"][0]["evidence"][0]["quote"] == "Shear strength provisions."


def test_get_audit_endpoint_empty_records_is_200(client):
    # No DEMO_AUTH_TOKEN in the test env -> operator/open mode.
    r = client.get("/api/v2/chat/audit/missing_analysis")
    assert r.status_code == 200
    assert r.json()["records"] == []
    assert r.json()["access_mode"] == "operator"


# ---------------------------------------------------------------------------
# Audit access control (DEMO_AUTH_TOKEN set -> gated)
# ---------------------------------------------------------------------------

def _audit_record(analysis_id: str, *, session_id: str, quote: str = "Q") -> dict:
    return {
        "analysis_id": analysis_id,
        "member_id": 5,
        "turn": 1,
        "session_id": session_id,
        "member": {"member_id": 5},
        "trigger": {"status": "NG", "ratios": {"shear": 1.2}},
        "query": {"query_text": "shear"},
        "rag_used": True,
        "evidence": [{"doc_id": "KDS 14 31 10", "clause": "4.3.2.1.2",
                      "quote": quote}],
        "warnings": [],
    }


def _live_session(sid: str, analysis_id=None) -> None:
    now = time.time()
    with _CHAT_SESSION_LOCK:
        chat_session_cache[sid] = {
            "session_id": sid,
            "analysis_id": analysis_id,
            "history": [],
            "provider": "noop",
            "created_at": now,
            "expires_at": now + 3600,
        }


def test_audit_token_set_no_credentials_returns_403(client, monkeypatch):
    monkeypatch.setenv("DEMO_AUTH_TOKEN", "secret")
    append_audit(_audit_record("a_gated", session_id="chat_owner"))
    r = client.get("/api/v2/chat/audit/a_gated")
    assert r.status_code == 403


def test_audit_operator_token_default_redacts_quotes(client, monkeypatch):
    """fix #1: operators still get redacted quotes unless include_quotes."""
    monkeypatch.setenv("DEMO_AUTH_TOKEN", "secret")
    append_audit(_audit_record("a_op", session_id="chat_x"))
    r = client.get("/api/v2/chat/audit/a_op?token=secret")
    assert r.status_code == 200
    body = r.json()
    assert body["access_mode"] == "operator"
    assert body["records"][0]["evidence"][0]["quote"] is None


def test_audit_operator_token_include_quotes_returns_full(client, monkeypatch):
    monkeypatch.setenv("DEMO_AUTH_TOKEN", "secret")
    append_audit(_audit_record("a_op2", session_id="chat_x"))
    r = client.get("/api/v2/chat/audit/a_op2?token=secret&include_quotes=true")
    assert r.status_code == 200
    assert r.json()["records"][0]["evidence"][0]["quote"] == "Q"


def test_audit_owning_session_metadata_then_quotes(client, monkeypatch):
    monkeypatch.setenv("DEMO_AUTH_TOKEN", "secret")
    sid = "chat_owner_ok"
    _live_session(sid, analysis_id="a_own")
    append_audit(_audit_record("a_own", session_id=sid))

    r = client.get(f"/api/v2/chat/audit/a_own?session_id={sid}")
    assert r.status_code == 200
    body = r.json()
    assert body["access_mode"] == "session"
    assert body["records"][0]["evidence"][0]["quote"] is None  # default redacts
    assert body["records"][0]["trigger"]["ratios"] == {"shear": 1.2}  # metadata kept

    r2 = client.get(
        f"/api/v2/chat/audit/a_own?session_id={sid}&include_quotes=true"
    )
    assert r2.json()["records"][0]["evidence"][0]["quote"] == "Q"


def test_audit_non_owning_session_returns_403(client, monkeypatch):
    """fix #2: a live session bound to the analysis is NOT enough — the
    record's server-stamped session_id must match the requester."""
    monkeypatch.setenv("DEMO_AUTH_TOKEN", "secret")
    sid = "chat_pretender"
    _live_session(sid, analysis_id="a_victim")  # bound, but didn't write the record
    append_audit(_audit_record("a_victim", session_id="chat_real_owner"))

    r = client.get(f"/api/v2/chat/audit/a_victim?session_id={sid}")
    assert r.status_code == 403


def test_audit_unknown_session_returns_403(client, monkeypatch):
    monkeypatch.setenv("DEMO_AUTH_TOKEN", "secret")
    append_audit(_audit_record("a_unknown", session_id="chat_real"))
    r = client.get("/api/v2/chat/audit/a_unknown?session_id=chat_nonexistent")
    assert r.status_code == 403


# ---------------------------------------------------------------------------
# NDJSON stream contract
# ---------------------------------------------------------------------------

def test_post_message_streams_status_token_done_in_that_order(client):
    sid = client.post("/api/v2/chat/sessions", json={}).json()["session_id"]
    r = client.post(
        "/api/v2/chat/messages",
        json={"session_id": sid, "message": "ping"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")

    events = _parse_ndjson(r.text)
    types = [e["type"] for e in events]
    # status first, done last, at least one token between
    assert types[0] == "status"
    assert types[-1] == "done"
    assert "token" in types

    # status carries provider name so the UI can show "noop unconfigured"
    assert events[0]["provider"] == "noop"

    # done summarises the turn — Phase A.1 means rounds == 0
    done = events[-1]
    assert done["rounds"] == 0
    assert done["total_tokens"] >= 1
    assert done["ms_total"] >= 0


def test_post_message_unknown_session_returns_404(client):
    r = client.post(
        "/api/v2/chat/messages",
        json={"session_id": "chat_doesnotexist", "message": "ping"},
    )
    assert r.status_code == 404


def test_post_message_blank_text_returns_400(client):
    sid = client.post("/api/v2/chat/sessions", json={}).json()["session_id"]
    r = client.post(
        "/api/v2/chat/messages",
        json={"session_id": sid, "message": "   "},
    )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# History + ui_context handling
# ---------------------------------------------------------------------------

def test_history_appended_user_then_assistant(client):
    sid = client.post("/api/v2/chat/sessions", json={}).json()["session_id"]
    client.post(
        "/api/v2/chat/messages",
        json={"session_id": sid, "message": "hello"},
    )
    g = client.get(f"/api/v2/chat/sessions/{sid}").json()
    roles = [h["role"] for h in g["history"]]
    assert roles == ["user", "assistant"]
    assert g["history"][0]["content"] == "hello"


def test_ui_context_stored_on_user_turn_but_not_leaked_in_debug(client):
    """``ui_context`` is captured in history (so Phase A.2 tools can read
    it) but the debug endpoint redacts it so a noisy frontend payload
    doesn't dominate operator output."""
    sid = client.post("/api/v2/chat/sessions", json={}).json()["session_id"]
    client.post(
        "/api/v2/chat/messages",
        json={
            "session_id": sid,
            "message": "이 부재",
            "ui_context": {"selected_element_ids": [42]},
        },
    )
    # debug shape is role + content only
    g = client.get(f"/api/v2/chat/sessions/{sid}").json()
    assert all(set(h.keys()) == {"role", "content"} for h in g["history"])
    # but the raw cache still has it for future tools to consume
    with _CHAT_SESSION_LOCK:
        raw = chat_session_cache[sid]["history"][0]
    assert raw["ui_context"] == {"selected_element_ids": [42]}


# ---------------------------------------------------------------------------
# Forbidden-key guard reaches the route (regression safety net)
# ---------------------------------------------------------------------------

def test_forbidden_key_guard_propagates_through_route():
    """If a future orchestrator change emits ``model_json`` in a tool_result,
    ``encode_event`` raises and the route doesn't swallow it. This is the
    invariant that keeps the model JSON off the wire (see
    ``core.chat.streaming.FORBIDDEN_KEYS``)."""
    from core.chat.streaming import encode_event

    with pytest.raises(ValueError, match="forbidden keys"):
        encode_event("tool_result", {
            "round": 1, "tool": "preview_section_change",
            "result": {"updated_model": {"nodes": []}},
            "ms": 5,
        })


def test_router_mount_does_not_break_main_simple_imports(client):
    """If chat_router's MCP_SERVER_PATH injection or services imports
    broke, ``TestClient(app)`` would fail to construct. Reaching this
    test means the import chain is healthy. We additionally hit the
    OpenAPI doc to confirm the chat routes are actually registered."""
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    assert "/api/v2/chat/sessions" in paths
    assert "/api/v2/chat/messages" in paths
    assert "/api/v2/chat/audit/{analysis_id}" in paths
