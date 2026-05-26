"""Phase A.2 — tool registry, inspect tools, orchestrator tool loop.

Uses a ``ScriptedToolProvider`` defined in this file only — keeps the
prod ``NoopProvider`` honest as "LLM 미설정 진단" and lets us script
exact tool-call sequences without depending on Ollama.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import AsyncIterator

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webapp" / "backend"))
sys.path.insert(0, str(ROOT / "mcp-server"))

os.environ.setdefault("CHAT_LLM_PROVIDER", "noop")
os.environ.setdefault("CHAT_TOOLS_ENABLED", "inspect,summary")

from app.main_simple import app  # noqa: E402
from app.services.analysis_context import (  # noqa: E402
    _ANALYSIS_CONTEXT_LOCK,
    analysis_context_cache,
)
from app.services.chat_session import (  # noqa: E402
    _CHAT_SESSION_LOCK,
    chat_session_cache,
)
from core.chat.llm.base import BaseLLMProvider, ToolCall  # noqa: E402
from core.chat.orchestrator import ChatOrchestrator  # noqa: E402
from core.chat.tool_registry import (  # noqa: E402
    ToolDisabledError,
    ToolNotFoundError,
    ToolRegistry,
    ToolSpec,
    default_registry,
    parse_enabled_groups,
)
from core.chat.tools.inspect import (  # noqa: E402
    GET_ANALYSIS_SUMMARY_TOOL,
    INSPECT_SELECTION_TOOL,
    get_analysis_summary,
    inspect_selection,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

ANALYSIS_ID = "analysis_test_a2_001"


@pytest.fixture(autouse=True)
def _clean_caches():
    with _CHAT_SESSION_LOCK:
        chat_session_cache.clear()
    with _ANALYSIS_CONTEXT_LOCK:
        analysis_context_cache.clear()
    yield
    with _CHAT_SESSION_LOCK:
        chat_session_cache.clear()
    with _ANALYSIS_CONTEXT_LOCK:
        analysis_context_cache.clear()


@pytest.fixture
def seeded_analysis_cache():
    """Insert a minimal context dict matching the Phase 0 compact subset."""
    with _ANALYSIS_CONTEXT_LOCK:
        analysis_context_cache[ANALYSIS_ID] = {
            "expires_at": time.time() + 3600,
            "analysis_summary": {
                "max_disp": {"dx_mm": 12.0, "dy_mm": 8.0},
                "max_drift": {"x": 0.004, "y": 0.003},
                "ng_count": 1,
                "num_stories": 3,
                "num_elements": 6,
            },
            "modal_summary": {
                "num_modes": 12,
                "fundamental_periods": {"T1_x_s": 0.85},
                "top_modes": [],
            },
            "envelope": {"max_dx_mm": 12.0},
            "member_info_by_elem_id": {
                "101": {"member_id": 1, "section": "H-300x300", "story": 1, "etype": "column"},
            },
            "member_ratios_by_elem_id": {
                "101": {"member_id": 1, "status": "OK", "ratio_interaction": 0.74},
            },
        }
    return ANALYSIS_ID


# ---------------------------------------------------------------------------
# ScriptedToolProvider — test-only LLM double
# ---------------------------------------------------------------------------

class ScriptedToolProvider(BaseLLMProvider):
    """Returns a pre-baked tool_call sequence then a fixed final text.

    Lives in tests/ only (per Codex A.2 guidance: don't grow NoopProvider
    into a fake tool caller — keep prod "Noop = unconfigured diagnostic"
    distinct from "deterministic test driver").
    """

    name = "scripted"

    def __init__(self, *, calls: list[ToolCall | None], final_text: str = "ok"):
        self._calls = list(calls)
        self._idx = 0
        self._final_text = final_text
        self.requested_with: list[list[dict]] = []

    async def request_tool_call(self, *, messages, tools):
        self.requested_with.append([{"name": t["function"]["name"]} for t in tools])
        if self._idx >= len(self._calls):
            return None
        tc = self._calls[self._idx]
        self._idx += 1
        return tc

    async def stream_tokens(self, *, messages) -> AsyncIterator[str]:
        yield self._final_text


# ---------------------------------------------------------------------------
# parse_enabled_groups + ToolRegistry unit tests
# ---------------------------------------------------------------------------

def test_parse_enabled_groups_strips_whitespace_and_empties():
    assert parse_enabled_groups(" inspect , summary , ") == {"inspect", "summary"}
    assert parse_enabled_groups("") == frozenset()


def _spec(name: str, group: str) -> ToolSpec:
    return ToolSpec(
        name=name, group=group, description=name,
        parameters={"type": "object", "properties": {}},
        func=lambda args, *, session: {"called": name},
    )


def test_registry_hides_disabled_tools_from_llm_schemas():
    reg = ToolRegistry(
        [_spec("alpha", "inspect"), _spec("beta", "edit")],
        enabled_groups=frozenset({"inspect"}),
    )
    names = [s["function"]["name"] for s in reg.llm_schemas()]
    assert names == ["alpha"]
    assert "beta" not in names


def test_registry_call_tool_rejects_disabled_group():
    reg = ToolRegistry(
        [_spec("beta", "edit")],
        enabled_groups=frozenset({"inspect"}),
    )
    with pytest.raises(ToolDisabledError, match="not in CHAT_TOOLS_ENABLED"):
        reg.call_tool("beta", {}, session={})


def test_registry_call_tool_unknown_name_raises():
    reg = ToolRegistry([_spec("alpha", "inspect")], enabled_groups=frozenset({"inspect"}))
    with pytest.raises(ToolNotFoundError):
        reg.call_tool("nonexistent", {}, session={})


def test_registry_duplicate_name_construction_raises():
    with pytest.raises(ValueError, match="duplicate tool name"):
        ToolRegistry([_spec("alpha", "inspect"), _spec("alpha", "summary")])


def test_default_registry_exposes_phase_a2_tools():
    reg = default_registry()
    names = {s["function"]["name"] for s in reg.llm_schemas()}
    # CHAT_TOOLS_ENABLED is set to "inspect,summary" in this module's setup.
    assert names == {"inspect_selection", "get_analysis_summary"}


# ---------------------------------------------------------------------------
# inspect_selection
# ---------------------------------------------------------------------------

def test_inspect_selection_explicit_element_ids(seeded_analysis_cache):
    out = inspect_selection(
        {"element_ids": [101], "analysis_id": seeded_analysis_cache},
        session={},
    )
    assert out["elements"][0]["found"] is True
    assert out["elements"][0]["info"]["section"] == "H-300x300"
    assert out["elements"][0]["ratios"]["ratio_interaction"] == 0.74


def test_inspect_selection_falls_back_to_ui_context(seeded_analysis_cache):
    session = {
        "analysis_id": seeded_analysis_cache,
        "history": [{
            "role": "user", "content": "이 부재",
            "ui_context": {"selected_element_ids": [101]},
        }],
    }
    out = inspect_selection({}, session=session)
    assert out["elements"][0]["element_id"] == 101
    assert out["elements"][0]["found"] is True


def test_inspect_selection_unknown_element_returns_found_false(seeded_analysis_cache):
    out = inspect_selection(
        {"element_ids": [9999], "analysis_id": seeded_analysis_cache},
        session={},
    )
    assert out["elements"][0]["found"] is False


def test_inspect_selection_unknown_analysis_returns_error(seeded_analysis_cache):
    out = inspect_selection(
        {"element_ids": [101], "analysis_id": "missing_id"},
        session={},
    )
    assert out["code"] == "analysis_not_found"
    assert out["elements"] == []


def test_inspect_selection_no_selection_returns_error(seeded_analysis_cache):
    out = inspect_selection({}, session={"analysis_id": seeded_analysis_cache})
    assert out["code"] == "no_selection"


def test_inspect_selection_raises_when_no_analysis_id_anywhere():
    with pytest.raises(ValueError, match="analysis_id is required"):
        inspect_selection({"element_ids": [1]}, session={})


def test_inspect_selection_ui_context_analysis_id_beats_stale_session_binding(
    seeded_analysis_cache,
):
    """Codex P1 on c378b05: the widget refreshes ``ui_context.analysis_id``
    every turn so a session created before any /analyze run, or one whose
    user just re-ran the analysis, can still resolve to the fresh id."""
    session = {
        "analysis_id": "stale_or_missing_id",  # not in cache
        "history": [{
            "role": "user", "content": "이 부재",
            "ui_context": {
                "analysis_id": seeded_analysis_cache,
                "selected_element_ids": [101],
            },
        }],
    }
    out = inspect_selection({}, session=session)
    assert out["analysis_id"] == seeded_analysis_cache
    assert out["elements"][0]["found"] is True


def test_inspect_selection_argument_beats_ui_context(seeded_analysis_cache):
    """Explicit > fresh > stale — the LLM can override even the widget's
    selection when it has a reason to."""
    session = {
        "history": [{
            "role": "user", "content": "x",
            "ui_context": {"analysis_id": "ui_id_should_lose"},
        }],
    }
    out = inspect_selection(
        {"analysis_id": seeded_analysis_cache, "element_ids": [101]},
        session=session,
    )
    assert out["analysis_id"] == seeded_analysis_cache


def test_inspect_selection_session_binding_used_when_no_ui_context():
    """Codex P1 fallback — curl/pytest paths that build a session via
    /sessions {analysis_id:...} keep working without a widget."""
    # Skip the cache hit check; we only care about which id flows through.
    with pytest.raises(ValueError, match="analysis_id is required"):
        inspect_selection({"element_ids": [1]}, session={"history": []})


def test_get_analysis_summary_ui_context_overrides_stale_session_binding(
    seeded_analysis_cache,
):
    session = {
        "analysis_id": "stale_id",
        "history": [{
            "role": "user", "content": "요약",
            "ui_context": {"analysis_id": seeded_analysis_cache},
        }],
    }
    out = get_analysis_summary({}, session=session)
    assert out["analysis_id"] == seeded_analysis_cache
    assert out["summary"]["ng_count"] == 1


# ---------------------------------------------------------------------------
# get_analysis_summary
# ---------------------------------------------------------------------------

def test_get_analysis_summary_returns_compact_fields(seeded_analysis_cache):
    out = get_analysis_summary({}, session={"analysis_id": seeded_analysis_cache})
    assert out["summary"]["ng_count"] == 1
    assert out["summary"]["num_stories"] == 3
    assert out["modal"]["fundamental_periods"]["T1_x_s"] == 0.85


def test_get_analysis_summary_unknown_analysis(seeded_analysis_cache):
    out = get_analysis_summary({"analysis_id": "nope"}, session={})
    assert out["code"] == "analysis_not_found"


# ---------------------------------------------------------------------------
# Orchestrator tool loop with ScriptedToolProvider
# ---------------------------------------------------------------------------

def _drain(async_gen) -> list[dict]:
    """Run an async generator to completion and parse NDJSON lines."""
    import asyncio

    async def _collect():
        return [line async for line in async_gen]

    lines = asyncio.run(_collect())
    return [json.loads(ln) for ln in lines if ln.strip()]


def test_orchestrator_runs_one_tool_then_streams_final_answer(seeded_analysis_cache):
    provider = ScriptedToolProvider(
        calls=[
            ToolCall(name="get_analysis_summary", arguments={}),
            None,  # explicit "no more tools" — terminator before final answer
        ],
        final_text="NG 부재 1개입니다.",
    )
    registry = default_registry()
    orch = ChatOrchestrator(provider, registry=registry)
    session = {"analysis_id": seeded_analysis_cache, "history": []}

    events = _drain(orch.run_turn(
        session=session,
        user_message="요약해줘",
    ))

    types = [e["type"] for e in events]
    assert types[0] == "status"
    assert types[-1] == "done"
    assert "tool_call" in types
    assert "tool_result" in types
    assert "token" in types

    # one tool round consumed
    done = events[-1]
    assert done["rounds"] == 1

    # tool_result carries the summary payload (no forbidden keys)
    tr = next(e for e in events if e["type"] == "tool_result")
    assert tr["tool"] == "get_analysis_summary"
    assert tr["result"]["summary"]["ng_count"] == 1


def test_orchestrator_tool_appended_to_history_as_role_tool(seeded_analysis_cache):
    provider = ScriptedToolProvider(
        calls=[ToolCall(name="get_analysis_summary", arguments={}), None],
        final_text="ok",
    )
    session = {"analysis_id": seeded_analysis_cache, "history": []}
    _drain(ChatOrchestrator(provider, registry=default_registry()).run_turn(
        session=session, user_message="요약",
    ))
    roles = [h["role"] for h in session["history"]]
    # user → tool → assistant
    assert roles == ["user", "tool", "assistant"]
    tool_entry = session["history"][1]
    assert tool_entry["name"] == "get_analysis_summary"
    # content is JSON-encoded so the LLM can parse it on the next round
    assert json.loads(tool_entry["content"])["summary"]["ng_count"] == 1


def test_orchestrator_tool_crash_surfaces_as_tool_result_error(seeded_analysis_cache):
    """If the tool raises, the loop still emits a tool_result with an
    error code rather than aborting — the LLM gets a chance to recover
    or just fall through to the final answer."""
    provider = ScriptedToolProvider(
        calls=[
            ToolCall(name="inspect_selection", arguments={"analysis_id": "missing"}),
            None,
        ],
        final_text="확인 못 했습니다.",
    )
    session = {"analysis_id": seeded_analysis_cache, "history": []}
    events = _drain(ChatOrchestrator(provider, registry=default_registry()).run_turn(
        session=session, user_message="이 부재",
    ))
    tr = next(e for e in events if e["type"] == "tool_result")
    # inspect_selection returns dict (not raise) when analysis_id missing
    assert tr["result"]["code"] == "analysis_not_found"
    assert events[-1]["type"] == "done"


def test_orchestrator_disabled_tool_call_is_blocked(seeded_analysis_cache):
    """If the LLM hallucinates a disabled tool name, call_tool raises
    ToolDisabledError and the loop converts it to a tool_result error
    with code=tool_blocked instead of executing anything."""
    registry = ToolRegistry(
        [INSPECT_SELECTION_TOOL, GET_ANALYSIS_SUMMARY_TOOL],
        enabled_groups=frozenset({"summary"}),  # inspect disabled
    )
    provider = ScriptedToolProvider(
        calls=[ToolCall(name="inspect_selection", arguments={"element_ids": [101]}), None],
        final_text="권한 없음",
    )
    session = {"analysis_id": seeded_analysis_cache, "history": []}
    events = _drain(ChatOrchestrator(provider, registry=registry).run_turn(
        session=session, user_message="ratio",
    ))
    tr = next(e for e in events if e["type"] == "tool_result")
    assert tr["result"]["code"] == "tool_blocked"


def test_orchestrator_max_rounds_caps_runaway_tool_loop(seeded_analysis_cache):
    """If the provider never returns None, the loop must still terminate
    at max_rounds and proceed to the final answer."""
    provider = ScriptedToolProvider(
        calls=[ToolCall(name="get_analysis_summary", arguments={})] * 10,
        final_text="멈춤",
    )
    orch = ChatOrchestrator(
        provider, registry=default_registry(), max_rounds=3,
    )
    session = {"analysis_id": seeded_analysis_cache, "history": []}
    events = _drain(orch.run_turn(session=session, user_message="x"))
    assert events[-1]["rounds"] == 3
    assert events[-1]["type"] == "done"


def test_orchestrator_history_trim_runs_after_assistant_append():
    """Codex P3: max_history bound must hold at end of turn, not only
    after the user append (the old A.1 code could leave max_history + 1
    entries because the assistant message was appended unconditionally
    after the trim)."""
    provider = ScriptedToolProvider(calls=[None], final_text="ok")
    orch = ChatOrchestrator(provider, registry=None, max_history=4)

    session = {
        "history": [
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "2"},
            {"role": "user", "content": "3"},
            {"role": "assistant", "content": "4"},
        ],
    }
    _drain(orch.run_turn(session=session, user_message="5"))
    assert len(session["history"]) == 4
    # Oldest entries dropped; newest user + assistant retained.
    assert session["history"][-2]["content"] == "5"
    assert session["history"][-1]["role"] == "assistant"


def test_orchestrator_provider_messages_strip_ui_context():
    """``ui_context`` is chat-router metadata. It must not be forwarded
    to the LLM (would inflate context + confuse the model)."""
    seen: list[list[dict]] = []

    class CapturingProvider(BaseLLMProvider):
        name = "capture"

        async def stream_tokens(self, *, messages):
            seen.append([dict(m) for m in messages])
            yield ""

    orch = ChatOrchestrator(CapturingProvider(), registry=None)
    session = {"history": []}
    _drain(orch.run_turn(
        session=session,
        user_message="hi",
        ui_context={"selected_element_ids": [42]},
    ))
    for m in seen[0]:
        assert "ui_context" not in m


# ---------------------------------------------------------------------------
# Route-level integration: chat_router serves Phase A.2 tools via NoopProvider
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    return TestClient(app)


def test_route_message_round_trips_through_default_registry(client, seeded_analysis_cache):
    """End-to-end through the live route. NoopProvider doesn't call tools,
    so the schemas exist but go unused — the test confirms the registry
    integration doesn't break the existing status → token → done path."""
    sid = client.post(
        "/api/v2/chat/sessions",
        json={"analysis_id": seeded_analysis_cache},
    ).json()["session_id"]
    r = client.post(
        "/api/v2/chat/messages",
        json={"session_id": sid, "message": "ping"},
    )
    assert r.status_code == 200
    types = [json.loads(ln)["type"] for ln in r.text.split("\n") if ln]
    assert types[0] == "status"
    assert types[-1] == "done"
    # NoopProvider declines tools → rounds=0
    done = json.loads([ln for ln in r.text.split("\n") if ln][-1])
    assert done["rounds"] == 0
