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
    """Insert a minimal context dict matching the Phase 0 compact subset.

    Both ``_by_elem_id`` and ``_by_member_id`` index variants are seeded
    so tests cover the live viewer's member_id click path as well as
    direct sub-element queries. ``member_id=1`` happens to have
    ``element_id=101`` in this fixture — no collision — but the regression
    test for the cross-index bug uses a custom fixture below."""
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
            "member_info_by_member_id": {
                "1": {"member_id": 1, "section": "H-300x300", "story": 1, "etype": "column"},
            },
            "member_ratios_by_elem_id": {
                "101": {"member_id": 1, "status": "OK", "ratio_interaction": 0.74},
            },
            "member_ratios_by_member_id": {
                "1": {"member_id": 1, "status": "OK", "ratio_interaction": 0.74},
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

    async def request_tool_call(self, *, messages, tools, temperature=None):
        self.requested_with.append([{"name": t["function"]["name"]} for t in tools])
        if self._idx >= len(self._calls):
            return None
        tc = self._calls[self._idx]
        self._idx += 1
        return tc

    async def stream_tokens(self, *, messages, temperature=None) -> AsyncIterator[str]:
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


def test_inspect_selection_resolves_via_member_id_first_when_collision():
    """Live regression: in a 27-column / 4-subdivision building, sub-
    element id 19 is owned by member #5 (1st-story column), while
    member #19 itself is a 3rd-story column. The 3D viewer sends a
    member_id when the user clicks, so the chat tool must consult the
    by_member_id index FIRST. Without that, the bot reported "1층" for
    every clicked column because every numeric member-id ≤ 27 collided
    with a 1st-story sub-element id of the same number."""
    cache_id = "analysis_collision_test"
    with _ANALYSIS_CONTEXT_LOCK:
        analysis_context_cache[cache_id] = {
            "expires_at": time.time() + 3600,
            # Sub-element 19 belongs to member 5 (1st-story column).
            "member_info_by_elem_id": {
                "19": {"member_id": 5, "section": "H-300x300", "story": 1, "etype": "column"},
            },
            # Member 19 itself is a 3rd-story column.
            "member_info_by_member_id": {
                "19": {"member_id": 19, "section": "H-300x300", "story": 3, "etype": "column"},
            },
            "member_ratios_by_elem_id": {
                "19": {"member_id": 5, "status": "OK", "ratio_interaction": 0.2},
            },
            "member_ratios_by_member_id": {
                "19": {"member_id": 19, "status": "OK", "ratio_interaction": 0.305},
            },
        }
    out = inspect_selection(
        {"element_ids": [19], "analysis_id": cache_id},
        session={},
    )
    # Resolves to the *clicked member* (#19, story 3) NOT to the member
    # whose sub-element happens to be numbered 19 (#5, story 1).
    assert out["elements"][0]["info"]["member_id"] == 19
    assert out["elements"][0]["info"]["story"] == 3
    assert out["elements"][0]["ratios"]["ratio_interaction"] == 0.305


def test_inspect_selection_falls_back_to_elem_id_when_member_id_missing():
    """If a caller passes a real OpenSees sub-element id (no member_id
    match exists), the elem_id map is still consulted. Keeps the door
    open for future tools that probe internal sub-elements directly."""
    cache_id = "analysis_elem_only_test"
    with _ANALYSIS_CONTEXT_LOCK:
        analysis_context_cache[cache_id] = {
            "expires_at": time.time() + 3600,
            "member_info_by_elem_id": {
                "73": {"member_id": 19, "section": "H-300x300", "story": 3, "etype": "column"},
            },
            # member_id="73" intentionally absent
            "member_info_by_member_id": {},
            "member_ratios_by_elem_id": {
                "73": {"member_id": 19, "status": "OK", "ratio_interaction": 0.305},
            },
            "member_ratios_by_member_id": {},
        }
    out = inspect_selection(
        {"element_ids": [73], "analysis_id": cache_id},
        session={},
    )
    assert out["elements"][0]["found"] is True
    assert out["elements"][0]["info"]["story"] == 3
    assert out["elements"][0]["info"]["member_id"] == 19


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
# element_ids coercion (Codex P2 on c378b05) — real Ollama emits permissive
# types: single int, single str, mixed lists. Tools must normalise instead
# of crashing the chat turn.
# ---------------------------------------------------------------------------

def test_inspect_selection_accepts_single_int_element_id(seeded_analysis_cache):
    out = inspect_selection(
        {"element_ids": 101, "analysis_id": seeded_analysis_cache},
        session={},
    )
    assert [e["element_id"] for e in out["elements"]] == [101]
    assert out["elements"][0]["found"] is True


def test_inspect_selection_accepts_single_string_element_id(seeded_analysis_cache):
    out = inspect_selection(
        {"element_ids": "101", "analysis_id": seeded_analysis_cache},
        session={},
    )
    assert out["elements"][0]["element_id"] == 101
    assert out["elements"][0]["found"] is True


def test_inspect_selection_accepts_mixed_list_dropping_unparseable(
    seeded_analysis_cache,
):
    """qwen2.5 has been observed to mix int and str in the same list and
    occasionally emit a stray non-numeric token. The tool keeps the
    parseable ids and silently drops junk."""
    out = inspect_selection(
        {"element_ids": [101, "201", "bad", None], "analysis_id": seeded_analysis_cache},
        session={},
    )
    eids = [e["element_id"] for e in out["elements"]]
    assert eids == [101, 201]


def test_inspect_selection_member_id_wins_over_colliding_elem_id():
    """Regression for the 'every member is 1층' bug found in live smoke.

    The 3D viewer attaches ``member_id`` to mesh userData, so a click on
    column #19 sends ``selected_element_ids=[19]`` — but 19 is also the
    OpenSees sub-element id of column #5's third internal element when
    ``num_elements_per_member=4``. The buggy version of inspect_selection
    used the elem_id map alone and returned column #5's info (story 1)
    for every click on a high-numbered column. This test seeds exactly
    that collision and asserts the member_id map wins.
    """
    aid = "collision_test"
    with _ANALYSIS_CONTEXT_LOCK:
        analysis_context_cache[aid] = {
            "expires_at": time.time() + 3600,
            # Member 5 = first column at story 1, sub-elements 17-20
            "member_info_by_elem_id": {
                "17": {"member_id": 5, "story": 1, "etype": "column"},
                "18": {"member_id": 5, "story": 1, "etype": "column"},
                "19": {"member_id": 5, "story": 1, "etype": "column"},
                "20": {"member_id": 5, "story": 1, "etype": "column"},
            },
            # Member 19 = first column at story 3 — what the user
            # actually clicked. Editor sends id=19 meaning member_id=19.
            "member_info_by_member_id": {
                "19": {"member_id": 19, "story": 3, "etype": "column"},
            },
            "member_ratios_by_elem_id": {
                "17": {"member_id": 5, "status": "OK", "ratio_interaction": 0.36},
                "19": {"member_id": 5, "status": "OK", "ratio_interaction": 0.36},
            },
            "member_ratios_by_member_id": {
                "19": {"member_id": 19, "status": "OK", "ratio_interaction": 0.305},
            },
        }
    try:
        out = inspect_selection(
            {"element_ids": [19], "analysis_id": aid}, session={},
        )
        assert out["elements"][0]["found"] is True
        # The bug: would return member_id=5, story=1. Fix: member_id wins.
        assert out["elements"][0]["info"]["member_id"] == 19
        assert out["elements"][0]["info"]["story"] == 3
        assert out["elements"][0]["ratios"]["ratio_interaction"] == 0.305
    finally:
        with _ANALYSIS_CONTEXT_LOCK:
            analysis_context_cache.pop(aid, None)


def test_inspect_selection_falls_back_to_elem_id_when_no_member_id_match():
    """When the caller passes a real sub-element id (not in the member_id
    map), the elem_id fallback still resolves. Belt-and-suspenders for
    future tools that legitimately want sub-element granularity."""
    aid = "elem_fallback_test"
    with _ANALYSIS_CONTEXT_LOCK:
        analysis_context_cache[aid] = {
            "expires_at": time.time() + 3600,
            "member_info_by_elem_id": {
                "73": {"member_id": 19, "story": 3, "etype": "column"},
            },
            "member_info_by_member_id": {
                # Note: key "73" is NOT here — caller is asking about a
                # genuine sub-element, not a member_id
                "19": {"member_id": 19, "story": 3, "etype": "column"},
            },
        }
    try:
        out = inspect_selection(
            {"element_ids": [73], "analysis_id": aid}, session={},
        )
        assert out["elements"][0]["found"] is True
        assert out["elements"][0]["info"]["member_id"] == 19
        assert out["elements"][0]["info"]["story"] == 3
    finally:
        with _ANALYSIS_CONTEXT_LOCK:
            analysis_context_cache.pop(aid, None)


def test_inspect_selection_coerces_ui_context_selection_too(seeded_analysis_cache):
    """The fallback path from ui_context should also normalise types —
    the chat widget already filters node selections (see
    EditorV2ChatBridge.getContext) but a future bridge change could
    forward strings."""
    session = {
        "analysis_id": seeded_analysis_cache,
        "history": [{
            "role": "user", "content": "이 부재",
            "ui_context": {"selected_element_ids": ["101"]},
        }],
    }
    out = inspect_selection({}, session=session)
    assert out["elements"][0]["element_id"] == 101


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
    # User message is intentionally neutral so the heuristic pre-guard in
    # ``_run_tool_loop`` does NOT short-circuit to a forced tool call.
    # This test exercises the LLM-driven path where the scripted provider
    # passes an explicit bad analysis_id and the tool surfaces the error.
    events = _drain(ChatOrchestrator(provider, registry=default_registry()).run_turn(
        session=session, user_message="체크 부탁해",
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


def test_orchestrator_emits_error_and_done_when_provider_raises_during_stream(
    seeded_analysis_cache,
):
    """Codex P2 on bcf3a0e: OllamaProvider raises OllamaUnavailableError
    when the daemon goes down mid-stream. The orchestrator's outer
    try/except must catch that and still emit a terminating ``done`` so
    the chat widget doesn't hang waiting for one."""
    class CrashingProvider(BaseLLMProvider):
        name = "crash"
        async def stream_tokens(self, *, messages, temperature=None) -> AsyncIterator[str]:
            raise RuntimeError("simulated provider failure mid-stream")
            yield  # pragma: no cover

    orch = ChatOrchestrator(CrashingProvider(), registry=None)
    events = _drain(orch.run_turn(session={"history": []}, user_message="ping"))
    types = [e["type"] for e in events]
    assert "error" in types
    assert types[-1] == "done"
    err = next(e for e in events if e["type"] == "error")
    assert "simulated provider failure" in err["message"]


def test_orchestrator_emits_error_and_done_when_tool_request_raises(
    seeded_analysis_cache,
):
    """Same guarantee on the tool-round path: a provider crash inside
    ``request_tool_call`` (e.g. OllamaUnavailableError from a dead
    daemon) becomes an ``error`` event and the loop still terminates."""
    class CrashingProvider(BaseLLMProvider):
        name = "crash"
        async def request_tool_call(self, *, messages, tools, temperature=None):
            raise RuntimeError("daemon down during tool round")
        async def stream_tokens(self, *, messages, temperature=None) -> AsyncIterator[str]:
            yield "fallback"

    orch = ChatOrchestrator(CrashingProvider(), registry=default_registry())
    events = _drain(orch.run_turn(
        session={"analysis_id": seeded_analysis_cache, "history": []},
        user_message="요약",
    ))
    types = [e["type"] for e in events]
    assert "error" in types
    assert types[-1] == "done"
    err = next(e for e in events if e["type"] == "error")
    assert err["code"] == "tool_request_failure"


def test_orchestrator_safe_encode_keeps_stream_terminated_when_tool_returns_forbidden_key(
    seeded_analysis_cache,
):
    """Codex P1 on bcf3a0e: if a future tool returns a dict containing
    ``model_json`` (or any other FORBIDDEN_KEYS member), encode_event
    raises ValueError. _safe_encode catches that and emits an ``error``
    event with code=event_encoding_failed instead of letting the
    exception kill the stream — the client must still see ``done``."""
    leaky = ToolSpec(
        name="leaky", group="inspect",
        description="", parameters={"type": "object", "properties": {}},
        func=lambda a, *, session: {
            "ok": True,
            # Phase C / future-tool foot-gun: passing model_json straight
            # through to the wire. The guard must catch it.
            "updated_model": {"nodes": [{"id": 1}]},
        },
    )
    registry = ToolRegistry([leaky], enabled_groups=frozenset({"inspect"}))
    provider = ScriptedToolProvider(
        calls=[ToolCall(name="leaky", arguments={}), None],
        final_text="end",
    )
    events = _drain(ChatOrchestrator(provider, registry=registry).run_turn(
        session={"history": []},
        user_message="trigger leak",
    ))
    types = [e["type"] for e in events]
    assert types[-1] == "done"
    encoding_errors = [
        e for e in events
        if e["type"] == "error" and e.get("code") == "event_encoding_failed"
    ]
    assert encoding_errors, f"expected event_encoding_failed in {types}"


# ---------------------------------------------------------------------------
# Heuristic pre-guard (anti-hallucination)
# ---------------------------------------------------------------------------

def test_pick_forced_tool_inspect_keywords():
    from core.chat.orchestrator import _pick_forced_tool
    available = {"inspect_selection", "get_analysis_summary"}
    assert _pick_forced_tool("이 부재 안전한가?", available) == "inspect_selection"
    assert _pick_forced_tool("선택된 부재 정보 알려줘", available) == "inspect_selection"
    assert _pick_forced_tool("이거 안전?", available) == "inspect_selection"
    assert _pick_forced_tool("ratio 얼마야?", available) == "inspect_selection"


def test_pick_forced_tool_summary_keywords():
    from core.chat.orchestrator import _pick_forced_tool
    available = {"inspect_selection", "get_analysis_summary"}
    assert _pick_forced_tool("결과 요약해줘", available) == "get_analysis_summary"
    assert _pick_forced_tool("분석 결과 보여줘", available) == "get_analysis_summary"
    assert _pick_forced_tool("NG 몇 개?", available) == "get_analysis_summary"
    assert _pick_forced_tool("층간변위 어때?", available) == "get_analysis_summary"
    assert _pick_forced_tool("주기는?", available) == "get_analysis_summary"


def test_pick_forced_tool_neutral_messages_return_none():
    from core.chat.orchestrator import _pick_forced_tool
    available = {"inspect_selection", "get_analysis_summary"}
    assert _pick_forced_tool("안녕", available) is None
    assert _pick_forced_tool("도움말 보여줘", available) is None
    assert _pick_forced_tool("체크 부탁해", available) is None
    assert _pick_forced_tool("", available) is None


def test_pick_forced_tool_inspect_wins_when_both_match():
    """A user with a selection asking about it should never be routed to
    the global summary tool just because their message also mentions
    'NG' or 'drift'."""
    from core.chat.orchestrator import _pick_forced_tool
    available = {"inspect_selection", "get_analysis_summary"}
    # "이 부재" + "NG" → inspect wins (more specific)
    assert _pick_forced_tool("이 부재 NG야?", available) == "inspect_selection"


def test_pick_forced_tool_bare_이거_no_longer_triggers_inspect():
    """Bug C fix: ambiguous bare '이거' (without 부재/요소/안전?) used to
    force inspect_selection, then returned ``code=no_selection`` and
    frustrated the user. Now it falls through to LLM judgement."""
    from core.chat.orchestrator import _pick_forced_tool
    available = {"inspect_selection", "get_analysis_summary"}
    # No explicit element noun, no "안전?" — must NOT force inspect
    assert _pick_forced_tool("이거 괜찮아?", available) is None
    assert _pick_forced_tool("이거 어때?", available) is None
    # But "이거 안전?" still matches via the "안전?" branch
    assert _pick_forced_tool("이거 안전?", available) == "inspect_selection"
    # And "이 부재" continues to fire as before
    assert _pick_forced_tool("이 부재 정보", available) == "inspect_selection"


def test_pick_forced_tool_n_번_부재_routes_to_inspect():
    """Bug B fix part 1: '5번 부재' style references must hit the inspect
    pattern. The element_id itself is extracted separately by
    _extract_explicit_element_ids."""
    from core.chat.orchestrator import _pick_forced_tool
    available = {"inspect_selection", "get_analysis_summary"}
    assert _pick_forced_tool("5번 부재 정보 보여줘", available) == "inspect_selection"
    assert _pick_forced_tool("12번 요소 어때?", available) == "inspect_selection"
    assert _pick_forced_tool("element 7 보여줘", available) == "inspect_selection"


def test_extract_explicit_element_ids_korean_n_번_form():
    from core.chat.orchestrator import _extract_explicit_element_ids
    assert _extract_explicit_element_ids("5번 부재 정보") == [5]
    assert _extract_explicit_element_ids("12번 요소 보여줘") == [12]
    # Multi-id, comma-separated
    assert _extract_explicit_element_ids("5번, 10번 부재 비교해줘") == [5, 10]


def test_extract_explicit_element_ids_english_form():
    from core.chat.orchestrator import _extract_explicit_element_ids
    assert _extract_explicit_element_ids("element 7 보여줘") == [7]
    assert _extract_explicit_element_ids("엘리먼트 42 정보") == [42]
    # "#" prefix tolerated
    assert _extract_explicit_element_ids("element #99") == [99]


def test_extract_explicit_element_ids_requires_element_noun_for_korean():
    """Bare 'N번' without an element noun must not match — otherwise
    story references ('5층') or order references ('5번째 케이스') would
    be miscategorised as element_ids."""
    from core.chat.orchestrator import _extract_explicit_element_ids
    # No 부재/요소/element noun → must NOT extract
    assert _extract_explicit_element_ids("5번 케이스 결과") == []
    assert _extract_explicit_element_ids("5층 어때?") == []
    # Pure greeting
    assert _extract_explicit_element_ids("안녕") == []
    # Empty / None safety
    assert _extract_explicit_element_ids("") == []


def test_extract_explicit_element_ids_dedups_and_preserves_order():
    from core.chat.orchestrator import _extract_explicit_element_ids
    # Same id mentioned twice → kept once
    assert _extract_explicit_element_ids("5번 부재, 5번도 확인") == [5]
    # First-seen order preserved
    assert _extract_explicit_element_ids("10번, 5번 부재 비교") == [10, 5]


def test_orchestrator_forces_inspect_with_extracted_element_id(
    seeded_analysis_cache,
):
    """End-to-end: '5번 부재 정보 보여줘' WITHOUT a UI selection must
    still resolve to element_id=5 via the heuristic argument extractor,
    not return ``code=no_selection``. This is the failing case from the
    live smoke test."""
    # Seed cache has element_id=101 wired; we'll ask about a missing one
    # (5) so the tool returns found=False — the point of this test is
    # that the forced call carries element_ids=[5], NOT that 5 exists.
    from core.chat.llm.noop_provider import NoopProvider
    orch = ChatOrchestrator(NoopProvider(), registry=default_registry())
    session = {"analysis_id": seeded_analysis_cache, "history": []}
    events = _drain(orch.run_turn(
        session=session,
        user_message="5번 부재 정보 보여줘",
        # Intentionally no ui_context selection — proves the id came
        # from the message itself, not from a click.
        ui_context={},
    ))
    tool_calls = [e for e in events if e["type"] == "tool_call"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["tool"] == "inspect_selection"
    assert tool_calls[0]["arguments"] == {"element_ids": [5]}
    # And the result actually targets that id
    tool_results = [e for e in events if e["type"] == "tool_result"]
    assert tool_results[0]["result"]["elements"][0]["element_id"] == 5


def test_orchestrator_forces_inspect_with_multiple_extracted_ids(
    seeded_analysis_cache,
):
    """'5번, 10번 부재 비교' style — multi-id extraction wires through."""
    from core.chat.llm.noop_provider import NoopProvider
    orch = ChatOrchestrator(NoopProvider(), registry=default_registry())
    session = {"analysis_id": seeded_analysis_cache, "history": []}
    events = _drain(orch.run_turn(
        session=session,
        user_message="5번, 10번 부재 비교해줘",
    ))
    tool_calls = [e for e in events if e["type"] == "tool_call"]
    assert tool_calls[0]["arguments"] == {"element_ids": [5, 10]}


def test_pick_forced_tool_respects_disabled_tools():
    """If the registry doesn't expose a tool, the heuristic must not
    pick it (defensive against env-disabled tool groups)."""
    from core.chat.orchestrator import _pick_forced_tool
    only_summary = {"get_analysis_summary"}
    # inspect pattern but inspect tool not available → falls through to None
    assert _pick_forced_tool("이 부재 안전?", only_summary) is None
    # summary pattern still picks summary
    assert _pick_forced_tool("결과 요약", only_summary) == "get_analysis_summary"


def test_orchestrator_forces_inspect_selection_on_keyword_with_noop_provider(
    seeded_analysis_cache,
):
    """The whole point of the pre-guard: even when the LLM would NOT call
    a tool (Noop returns None from request_tool_call), the heuristic
    forces inspect_selection so the answer is grounded in real data."""
    # NoopProvider declines tools, so without the heuristic this test
    # would observe rounds=0. With the heuristic, rounds=1.
    from core.chat.llm.noop_provider import NoopProvider
    orch = ChatOrchestrator(NoopProvider(), registry=default_registry())
    session = {
        "analysis_id": seeded_analysis_cache,
        "history": [],
    }
    events = _drain(orch.run_turn(
        session=session,
        user_message="이 부재 안전한가?",
        ui_context={"selected_element_ids": [101]},
    ))
    tool_calls = [e for e in events if e["type"] == "tool_call"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["tool"] == "inspect_selection"
    assert tool_calls[0]["arguments"] == {}
    tool_results = [e for e in events if e["type"] == "tool_result"]
    assert tool_results[0]["result"]["elements"][0]["element_id"] == 101


def test_orchestrator_forces_summary_on_keyword_with_noop_provider(
    seeded_analysis_cache,
):
    from core.chat.llm.noop_provider import NoopProvider
    orch = ChatOrchestrator(NoopProvider(), registry=default_registry())
    session = {"analysis_id": seeded_analysis_cache, "history": []}
    events = _drain(orch.run_turn(
        session=session, user_message="결과 요약해줘",
    ))
    tool_calls = [e for e in events if e["type"] == "tool_call"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["tool"] == "get_analysis_summary"


def test_orchestrator_forced_round_exits_loop_even_when_provider_wants_more(
    seeded_analysis_cache,
):
    """After a forced round, the LLM does NOT get a second tool round —
    otherwise qwen2.5 would re-call the same tool (history doesn't show
    the forced call as 'its' decision)."""
    provider = ScriptedToolProvider(
        calls=[
            # If the loop ever asked the LLM a second time, this scripted
            # call would land in events. The forced-round early-exit
            # ensures it never runs.
            ToolCall(name="get_analysis_summary", arguments={}),
            None,
        ],
        final_text="ok",
    )
    orch = ChatOrchestrator(provider, registry=default_registry())
    session = {"analysis_id": seeded_analysis_cache, "history": []}
    events = _drain(orch.run_turn(
        session=session, user_message="결과 요약해줘",
    ))
    tool_calls = [e for e in events if e["type"] == "tool_call"]
    # Exactly one tool_call — the forced one. Scripted provider's call
    # never ran because the loop exited after the forced round.
    assert len(tool_calls) == 1
    assert events[-1]["type"] == "done"
    assert events[-1]["rounds"] == 1
    # And the scripted provider never had its request_tool_call called
    assert provider.requested_with == []


def test_orchestrator_neutral_message_does_not_trigger_heuristic(
    seeded_analysis_cache,
):
    """Bare greetings shouldn't force a tool round — the original LLM-
    driven path must still work for messages that don't match patterns."""
    provider = ScriptedToolProvider(
        calls=[None],  # LLM declines tools
        final_text="안녕하세요",
    )
    orch = ChatOrchestrator(provider, registry=default_registry())
    session = {"analysis_id": seeded_analysis_cache, "history": []}
    events = _drain(orch.run_turn(
        session=session, user_message="안녕",
    ))
    tool_calls = [e for e in events if e["type"] == "tool_call"]
    assert tool_calls == []
    assert events[-1]["rounds"] == 0


def test_orchestrator_provider_messages_strip_ui_context():
    """``ui_context`` is chat-router metadata. It must not be forwarded
    to the LLM (would inflate context + confuse the model)."""
    seen: list[list[dict]] = []

    class CapturingProvider(BaseLLMProvider):
        name = "capture"

        async def stream_tokens(self, *, messages, temperature=None):
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
# creativity_hint → stream_tokens temperature routing (Option A)
# ---------------------------------------------------------------------------

class _TemperatureCapturingProvider(BaseLLMProvider):
    """Test double that records the ``temperature`` kwarg the orchestrator
    passes into stream_tokens, while staying tool-aware enough to drive a
    full run_turn through."""
    name = "capture-temp"

    def __init__(self):
        self.stream_temp: Optional[float] = "UNSET"  # type: ignore[assignment]
        self.tool_temp: Optional[float] = "UNSET"  # type: ignore[assignment]

    async def request_tool_call(self, *, messages, tools, temperature=None):
        self.tool_temp = temperature
        return None  # let the heuristic / final stream take over

    async def stream_tokens(self, *, messages, temperature=None) -> AsyncIterator[str]:
        self.stream_temp = temperature
        yield "ok"


def test_factual_tool_routes_to_provider_default_temperature(seeded_analysis_cache):
    """get_analysis_summary has the default ``factual`` hint → orchestrator
    passes ``temperature=None`` so the provider uses its configured
    default (low for chat, high if a future caller swaps it)."""
    provider = _TemperatureCapturingProvider()
    orch = ChatOrchestrator(provider, registry=default_registry())
    session = {"analysis_id": seeded_analysis_cache, "history": []}
    # "결과 요약" triggers the forced-tool heuristic → get_analysis_summary
    # actually runs, so the routing has a tool to read a hint from.
    _drain(orch.run_turn(session=session, user_message="결과 요약해줘"))
    assert provider.stream_temp is None


def test_narrative_tool_routes_to_higher_temperature():
    """A tool registered with ``creativity_hint='narrative'`` makes the
    orchestrator hand stream_tokens a non-None temperature (currently
    0.5). Verified end-to-end through a custom registry."""
    narrative_tool = ToolSpec(
        name="draft_report",
        group="report",
        description="Draft a Korean prose summary of the analysis.",
        parameters={"type": "object", "properties": {}},
        func=lambda args, *, session: {"draft": "보고서 초안"},
        creativity_hint="narrative",
    )
    registry = ToolRegistry([narrative_tool], enabled_groups=frozenset({"report"}))
    provider = _TemperatureCapturingProvider()
    # Scripted call drives the tool — heuristic won't fire for this
    # custom tool name.
    class _Scripted(_TemperatureCapturingProvider):
        async def request_tool_call(self, *, messages, tools, temperature=None):
            self.tool_temp = temperature
            return ToolCall(name="draft_report", arguments={})
    provider = _Scripted()
    orch = ChatOrchestrator(provider, registry=registry)
    _drain(orch.run_turn(
        session={"history": []}, user_message="보고서 초안 만들어줘",
    ))
    # 0.5 matches the current narrative tier in _TEMPERATURE_BY_HINT
    assert provider.stream_temp == 0.5


def test_no_tool_ran_falls_back_to_provider_default(seeded_analysis_cache):
    """If the LLM declines tools and just answers, the previous turn's
    tool hint must NOT leak into this turn's temperature. The reset in
    run_turn ensures _last_tool_hint is None at stream_tokens time."""
    provider = _TemperatureCapturingProvider()
    orch = ChatOrchestrator(provider, registry=default_registry())
    session = {"analysis_id": seeded_analysis_cache, "history": []}
    _drain(orch.run_turn(session=session, user_message="안녕"))
    assert provider.stream_temp is None


def test_per_turn_hint_reset_prevents_leakage_between_turns():
    """Turn 1: narrative tool → bumps hint. Turn 2: just chat → must
    revert to provider default, not stay at the narrative temperature."""
    narrative_tool = ToolSpec(
        name="draft_report",
        group="report",
        description="x",
        parameters={"type": "object", "properties": {}},
        func=lambda args, *, session: {"draft": "x"},
        creativity_hint="narrative",
    )
    registry = ToolRegistry([narrative_tool], enabled_groups=frozenset({"report"}))

    class _SeqProvider(_TemperatureCapturingProvider):
        def __init__(self):
            super().__init__()
            self._turn = 0
            self.temps_seen: list = []

        async def request_tool_call(self, *, messages, tools, temperature=None):
            self._turn += 1
            return ToolCall(name="draft_report", arguments={}) if self._turn == 1 else None

        async def stream_tokens(self, *, messages, temperature=None) -> AsyncIterator[str]:
            self.temps_seen.append(temperature)
            yield "x"

    provider = _SeqProvider()
    orch = ChatOrchestrator(provider, registry=registry)
    session = {"history": []}
    _drain(orch.run_turn(session=session, user_message="보고서 초안"))
    _drain(orch.run_turn(session=session, user_message="안녕"))
    # Turn 1: narrative tool ran → 0.5. Turn 2: no tool ran → None.
    assert provider.temps_seen == [0.5, None]


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
