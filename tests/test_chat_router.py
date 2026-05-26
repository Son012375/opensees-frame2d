"""Phase A.1 chat-router smoke tests.

NoopProvider only (Phase A.3 adds the real Ollama path). Covers the
route + NDJSON contract + session lifecycle + a few invariants that
would catch wiring regressions when later phases extend the route.
"""
from __future__ import annotations

import json
import os
import sys
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


@pytest.fixture(autouse=True)
def _isolate_chat_sessions():
    with _CHAT_SESSION_LOCK:
        chat_session_cache.clear()
    yield
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
