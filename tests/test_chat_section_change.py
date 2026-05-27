"""Phase B — propose_section_change tool, preview cache, force routing.

Covers:
    * resolve_section_name — exact match, shorthand normalization, family
      mismatch, catalog miss, ambiguity.
    * synthesize_user_directed_candidate — member_id-only target, no
      element_id, bare-string confidence, source metadata.
    * propose_section_change end-to-end — compact-cache rejection,
      preview_id issuance, tool_result keeps FORBIDDEN_KEYS clean.
    * GET /chat-preview/{preview_id} — 200, 404, expiration.
    * Orchestrator force routing — _FORCE_COMMAND_RE positive/negative,
      _extract_section_target, end-to-end forced call with extracted args.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import AsyncIterator, Optional

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webapp" / "backend"))
sys.path.insert(0, str(ROOT / "mcp-server"))

os.environ.setdefault("CHAT_LLM_PROVIDER", "noop")
# Match test_chat_tools.py's default exactly. Every orchestrator test
# in this file constructs the registry explicitly via
# ``_phaseb_registry()`` instead of relying on the env-driven
# ``default_registry()`` — so the 'edit' group doesn't need to be in
# this default. Keeping the default in lockstep with the other chat
# test modules avoids cross-module env pollution (pytest collects
# alphabetically, so this module imports before test_chat_tools.py
# and a stricter setdefault here would survive and break that one).
os.environ.setdefault("CHAT_TOOLS_ENABLED", "inspect,summary")

from app.main_simple import app  # noqa: E402
from app.services.analysis_context import (  # noqa: E402
    _ANALYSIS_CONTEXT_LOCK,
    analysis_context_cache,
)
from app.services.chat_preview_cache import (  # noqa: E402
    chat_preview_cache,
    clear_cache as clear_preview_cache,
    get_preview,
)
from app.services.chat_session import (  # noqa: E402
    _CHAT_SESSION_LOCK,
    chat_session_cache,
)
from core.chat.llm.base import BaseLLMProvider, ToolCall  # noqa: E402
from core.chat.llm.noop_provider import NoopProvider  # noqa: E402
from core.chat.orchestrator import (  # noqa: E402
    ChatOrchestrator,
    _extract_section_target,
    _FORCE_COMMAND_RE,
    _pick_forced_tool,
)
from core.chat.streaming import (  # noqa: E402
    EVENT_TOOL_RESULT,
    _scan_forbidden,
    encode_event,
)
from core.chat.tool_registry import ToolRegistry  # noqa: E402
from core.chat.tools.inspect import (  # noqa: E402
    GET_ANALYSIS_SUMMARY_TOOL,
    INSPECT_SELECTION_TOOL,
)
from core.chat.tools.section_change import (  # noqa: E402
    PROPOSE_SECTION_CHANGE_TOOL,
    SectionResolutionError,
    propose_section_change,
    resolve_section_name,
    synthesize_user_directed_candidate,
)
from core.recommendation.section_catalog import clear_cache as clear_catalog_cache


ANALYSIS_ID = "analysis_phaseb_001"


@pytest.fixture(autouse=True)
def _clean_state():
    with _CHAT_SESSION_LOCK:
        chat_session_cache.clear()
    with _ANALYSIS_CONTEXT_LOCK:
        analysis_context_cache.clear()
    clear_preview_cache()
    # Force fresh section catalog so tests don't depend on a prior
    # Supabase-warmed cache from another module.
    clear_catalog_cache()
    yield
    with _CHAT_SESSION_LOCK:
        chat_session_cache.clear()
    with _ANALYSIS_CONTEXT_LOCK:
        analysis_context_cache.clear()
    clear_preview_cache()
    clear_catalog_cache()


def _build_v2_model() -> dict:
    """Minimal V2 model with three column elements at three stories.

    Elements are listed in id order so member_id (1-based sort position)
    equals the id directly — keeps test assertions readable.
    """
    return {
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "z": 0.0,    "story": 0},
            {"id": 2, "x": 0.0, "y": 0.0, "z": 3000.0, "story": 1},
            {"id": 3, "x": 0.0, "y": 0.0, "z": 6000.0, "story": 2},
            {"id": 4, "x": 0.0, "y": 0.0, "z": 9000.0, "story": 3},
        ],
        "elements": [
            {"id": 1, "node_i": 1, "node_j": 2,
             "elem_type": "column", "section": "H-300x300"},
            {"id": 2, "node_i": 2, "node_j": 3,
             "elem_type": "column", "section": "H-300x300"},
            {"id": 3, "node_i": 3, "node_j": 4,
             "elem_type": "column", "section": "H-300x300"},
        ],
    }


def _seed_full_context(aid: str = ANALYSIS_ID) -> str:
    """Insert a context entry tagged ``context_kind="full"`` with model_json."""
    with _ANALYSIS_CONTEXT_LOCK:
        analysis_context_cache[aid] = {
            "expires_at": time.time() + 3600,
            "context_kind": "full",
            "model_json": _build_v2_model(),
            "analysis_summary": {"ng_count": 0, "num_stories": 3, "num_elements": 3},
            "modal_summary": None,
            "member_info_by_elem_id": {},
            "member_info_by_member_id": {},
            "member_ratios_by_elem_id": {},
            "member_ratios_by_member_id": {},
            "candidates_by_id": {},
        }
    return aid


def _seed_compact_context(aid: str = "analysis_compact_001") -> str:
    with _ANALYSIS_CONTEXT_LOCK:
        analysis_context_cache[aid] = {
            "expires_at": time.time() + 3600,
            "context_kind": "compact",
            # NB: no model_json — compact contexts don't carry it
            "analysis_summary": {"ng_count": 0, "num_stories": 3, "num_elements": 3},
        }
    return aid


# ---------------------------------------------------------------------------
# resolve_section_name
# ---------------------------------------------------------------------------

def test_resolve_section_exact_column_match():
    """A column gets the H-square sub-ladder; H-300x300 is in it."""
    assert resolve_section_name("H-300x300", "column") == "H-300x300"


def test_resolve_section_exact_beam_match():
    """A beam gets the H-wide sub-ladder; H-300x150 is in it."""
    assert resolve_section_name("H-300x150", "beam") == "H-300x150"


def test_resolve_section_shorthand_column_uniquely_resolves():
    """'H-400' + column → fallback ladder has exactly one H-400xN square
    entry (H-400x400) so the shorthand normalizes deterministically."""
    assert resolve_section_name("H-400", "column") == "H-400x400"


def test_resolve_section_shorthand_beam_uniquely_resolves():
    """'H-400' + beam → fallback wide-H ladder has exactly one H-400xN
    entry (H-400x200)."""
    assert resolve_section_name("H-400", "beam") == "H-400x200"


def test_resolve_section_family_mismatch_column_with_wide():
    """H-200x100 is a wide H (beam-only). Asking for it as a column must
    fail with the available column options listed for the user."""
    with pytest.raises(SectionResolutionError) as exc:
        resolve_section_name("H-200x100", "column")
    assert exc.value.options, "options must be populated for chat to surface"
    # All options should be column-eligible (square H — h == b)
    for name in exc.value.options:
        body = name.split("-", 1)[1]
        h, b = body.split("x")[:2]
        assert h == b, f"{name} leaked into column suggestions"


def test_resolve_section_family_mismatch_beam_with_square():
    """H-300x300 is a square H (column-only). Asking for it as a beam
    must fail with wide-H suggestions."""
    with pytest.raises(SectionResolutionError):
        resolve_section_name("H-300x300", "beam")


def test_resolve_section_catalog_miss_lists_alternatives():
    with pytest.raises(SectionResolutionError) as exc:
        resolve_section_name("H-9999x9999", "column")
    assert exc.value.options, "user needs alternatives to act on"


def test_resolve_section_unknown_family_format():
    with pytest.raises(SectionResolutionError, match="형식을 인식할 수 없습니다"):
        resolve_section_name("not-a-section", "column")


def test_resolve_section_empty_input_raises():
    with pytest.raises(SectionResolutionError):
        resolve_section_name("", "column")


def test_resolve_section_shorthand_ambiguity_raises(monkeypatch):
    """When shorthand could resolve to multiple ladder entries, raise
    with the matches as options so the user can pick one. We monkey-
    patch list_family_ladder to inject the ambiguity (the fallback
    catalog has unique base prefixes by construction)."""
    from core.recommendation import section_catalog
    from core.recommendation.section_catalog import SectionEntry

    fake_ladder = [
        SectionEntry("H-400x200", "H", 400.0, 200.0, 80.0),
        SectionEntry("H-400x250", "H", 400.0, 250.0, 95.0),
        SectionEntry("H-400x300", "H", 400.0, 300.0, 130.0),
    ]
    monkeypatch.setattr(
        section_catalog, "list_family_ladder",
        lambda family, member_type: list(fake_ladder)
    )
    # get_section_metadata also lives in section_catalog and is imported
    # locally by resolve_section_name — monkey-patch it on the same module.
    monkeypatch.setattr(
        section_catalog, "get_section_metadata",
        lambda name: None,  # force shorthand path
    )

    with pytest.raises(SectionResolutionError, match="여러 단면") as exc:
        resolve_section_name("H-400", "beam")
    assert set(exc.value.options) == {
        "H-400x200", "H-400x250", "H-400x300",
    }


# ---------------------------------------------------------------------------
# synthesize_user_directed_candidate
# ---------------------------------------------------------------------------

def test_synthesize_returns_candidate_without_element_id():
    """KEY INVARIANT: target.element_id must NOT be set — otherwise
    apply_candidate_to_model would mutate whichever V2 element happens
    to have the raw id matching member_id (a different member entirely
    once num_elements_per_member > 1)."""
    model = _build_v2_model()
    cand, meta = synthesize_user_directed_candidate(model, member_id=2, target_section="H-400x400")

    assert cand.target.get("member_id") == 2
    assert "element_id" not in cand.target, (
        "target.element_id must not be set — apply_candidate.py:139 prefers it "
        "over member_id and would mutate the wrong element."
    )
    assert cand.element_id is None
    assert cand.member_id == 2


def test_synthesize_uses_bare_string_confidence_not_dataclass():
    """RetrofitCandidate.confidence is typed as str (LOW/MEDIUM/HIGH)
    — passing a Confidence(method=..., score=...) dataclass would
    crash to_dict() with a type error."""
    model = _build_v2_model()
    cand, _ = synthesize_user_directed_candidate(model, 1, "H-400x400")
    assert isinstance(cand.confidence, str)
    assert cand.confidence in {"low", "medium", "high"}


def test_synthesize_carries_chat_command_metadata():
    """The source provenance is what lets downstream tooling distinguish
    user-directed changes from auto-generated candidates."""
    model = _build_v2_model()
    cand, _ = synthesize_user_directed_candidate(model, 1, "H-400x400")
    assert cand.metadata.get("source") == "chat_command"
    assert cand.metadata.get("user_directed") is True


def test_synthesize_proposed_change_is_replace_section():
    model = _build_v2_model()
    cand, _ = synthesize_user_directed_candidate(model, 1, "H-350x350")
    assert cand.proposed_change["operation"] == "replace_section"
    assert cand.proposed_change["from"] == "H-300x300"
    assert cand.proposed_change["to"] == "H-350x350"
    assert cand.proposed_change["applicable"] is True


def test_synthesize_rejects_no_op_change():
    """If target == current, reject — applying would still raise inside
    apply_candidate_to_model, but catching here gives a friendlier error."""
    model = _build_v2_model()
    with pytest.raises(ValueError, match="이미"):
        synthesize_user_directed_candidate(model, 1, "H-300x300")


def test_synthesize_unknown_member_id_raises():
    model = _build_v2_model()
    with pytest.raises(ValueError, match="찾을 수 없습니다"):
        synthesize_user_directed_candidate(model, member_id=999, target_section="H-400x400")


def test_synthesize_propagates_section_resolution_error():
    model = _build_v2_model()
    with pytest.raises(SectionResolutionError):
        synthesize_user_directed_candidate(model, 1, "H-300x300x9999")


# ---------------------------------------------------------------------------
# propose_section_change end-to-end (no LLM)
# ---------------------------------------------------------------------------

def test_propose_section_change_returns_preview_id_and_ui_action():
    aid = _seed_full_context()
    result = propose_section_change(
        {"member_id": 2, "target_section": "H-400x400"},
        session={"analysis_id": aid, "history": []},
    )
    assert "preview_id" in result
    assert result["ui_action"] == "open_diff_preview"
    assert result["ui_payload"]["preview_id"] == result["preview_id"]
    assert result["diff_summary"]["section_from"] == "H-300x300"
    assert result["diff_summary"]["section_to"] == "H-400x400"


def test_propose_section_change_result_contains_no_forbidden_keys():
    """The tool_result must pass the chat-stream FORBIDDEN_KEYS guard —
    no model_json, updated_model, member_forces, etc. at any depth."""
    aid = _seed_full_context()
    result = propose_section_change(
        {"member_id": 1, "target_section": "H-350x350"},
        session={"analysis_id": aid, "history": []},
    )
    leaks = list(_scan_forbidden(result))
    assert leaks == [], f"chat result leaks FORBIDDEN_KEYS at: {leaks}"

    # And the encoder itself doesn't reject the result when wrapped in a
    # tool_result event (defensive — exercises the real serializer path).
    line = encode_event(EVENT_TOOL_RESULT, {
        "round": 0, "tool": "propose_section_change",
        "result": result, "ms": 1,
    })
    assert line.endswith("\n")


def test_propose_section_change_stages_updated_model_in_preview_cache():
    """The heavy payload lives in the preview cache, not in the tool result.
    Verifies the dual-write contract: result has preview_id only, the
    cache holds updated_model + diff + candidate."""
    aid = _seed_full_context()
    result = propose_section_change(
        {"member_id": 3, "target_section": "H-350x350"},
        session={"analysis_id": aid, "history": []},
    )
    entry = get_preview(result["preview_id"])
    assert entry is not None
    assert entry["analysis_id"] == aid
    # The cache entry IS allowed to hold updated_model (it's never
    # streamed via chat events). Verify the contract holds.
    assert isinstance(entry["updated_model"], dict)
    assert entry["updated_model"]["elements"][2]["section"] == "H-350x350"
    # Original cached model untouched (deep-copy invariant)
    with _ANALYSIS_CONTEXT_LOCK:
        original = analysis_context_cache[aid]["model_json"]
    assert original["elements"][2]["section"] == "H-300x300"


def test_propose_section_change_rejects_compact_context():
    """compact contexts only carry the chat-tool subset — they lack
    model_json, so applying anything would crash. Surface as a discrete
    error code so the chat router can ask the user to re-run analysis."""
    aid = _seed_compact_context()
    result = propose_section_change(
        {"member_id": 1, "target_section": "H-400x400"},
        session={"analysis_id": aid, "history": []},
    )
    assert result["code"] == "compact_context_rejected"


def test_propose_section_change_missing_target_section_errors():
    aid = _seed_full_context()
    result = propose_section_change(
        {"member_id": 1},
        session={"analysis_id": aid, "history": []},
    )
    assert result["code"] == "target_section_required"


def test_propose_section_change_falls_back_to_ui_selection():
    """When the LLM omits member_id, the tool reads the latest UI
    selection from session history — same fallback pattern as
    inspect_selection."""
    aid = _seed_full_context()
    session = {
        "analysis_id": aid,
        "history": [
            {"role": "user", "content": "단면 H-400으로",
             "ui_context": {"selected_element_ids": [2]}},
        ],
    }
    result = propose_section_change(
        {"target_section": "H-400x400"}, session=session,
    )
    assert "preview_id" in result
    assert result["diff_summary"]["member_id"] == 2


def test_propose_section_change_no_member_id_anywhere_errors():
    aid = _seed_full_context()
    result = propose_section_change(
        {"target_section": "H-400x400"},
        session={"analysis_id": aid, "history": []},
    )
    assert result["code"] == "member_id_required"


def test_propose_section_change_unknown_analysis_errors():
    result = propose_section_change(
        {"member_id": 1, "target_section": "H-400x400", "analysis_id": "missing"},
        session={"history": []},
    )
    assert result["code"] == "analysis_not_found"


def test_propose_section_change_section_resolution_error_surfaces_options():
    """User typed an off-catalog section — tool returns options the chat
    layer should show as candidate alternatives."""
    aid = _seed_full_context()
    result = propose_section_change(
        {"member_id": 1, "target_section": "H-9999x9999"},
        session={"analysis_id": aid, "history": []},
    )
    assert result["code"] == "section_resolution_failed"
    assert result["options"], "user needs alternative names to choose from"


def test_propose_section_change_noop_change_surfaces_synthesis_failed():
    aid = _seed_full_context()
    result = propose_section_change(
        {"member_id": 1, "target_section": "H-300x300"},
        session={"analysis_id": aid, "history": []},
    )
    assert result["code"] == "synthesis_failed"


# ---------------------------------------------------------------------------
# GET /api/v2/recommendations/chat-preview/{preview_id}
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    return TestClient(app)


def test_chat_preview_endpoint_round_trips(client):
    aid = _seed_full_context()
    result = propose_section_change(
        {"member_id": 2, "target_section": "H-400x400"},
        session={"analysis_id": aid, "history": []},
    )
    pid = result["preview_id"]
    r = client.get(f"/api/v2/recommendations/chat-preview/{pid}")
    assert r.status_code == 200
    body = r.json()
    assert body["preview_id"] == pid
    assert body["analysis_id"] == aid
    # HTTP endpoint MAY carry updated_model (it's not a chat stream).
    assert body["updated_model"]["elements"][1]["section"] == "H-400x400"
    assert body["diff"]["operation"] == "replace_section"
    assert body["diff"]["changed_member_count"] == 1


def test_chat_preview_endpoint_404_for_unknown_id(client):
    r = client.get("/api/v2/recommendations/chat-preview/chat_prev_zzzzzzzzzzzz")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"].lower()


def test_chat_preview_endpoint_404_after_expiry(client):
    """Manually expire the entry by rewriting its expires_at."""
    aid = _seed_full_context()
    result = propose_section_change(
        {"member_id": 1, "target_section": "H-400x400"},
        session={"analysis_id": aid, "history": []},
    )
    pid = result["preview_id"]
    # Force-expire
    chat_preview_cache[pid]["expires_at"] = time.time() - 60
    r = client.get(f"/api/v2/recommendations/chat-preview/{pid}")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Force routing: _FORCE_COMMAND_RE + _extract_section_target
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("msg", [
    "5번 부재 단면을 H-400x400으로 변경해줘",
    "12번 부재 단면을 H-350으로 바꿔",
    "단면을 H-400x400으로 바꿔",
    "단면을 SHS-200x200x6으로 변경",
    "단면 H-500으로 교체",
])
def test_force_command_re_positive(msg):
    assert _FORCE_COMMAND_RE.search(msg), f"expected match: {msg!r}"


@pytest.mark.parametrize("msg", [
    "5번 부재 안전한가?",
    "5번 부재 정보 보여줘",
    "결과 요약해줘",
    "단면이 뭐야?",            # question about, no change verb
    "이 부재 정보",
    "NG 몇 개?",
    "안녕",
])
def test_force_command_re_negative(msg):
    assert _FORCE_COMMAND_RE.search(msg) is None, f"false positive: {msg!r}"


def test_pick_forced_tool_prefers_command_over_inspect():
    """'5번 부재 단면을 H-400으로 변경' matches BOTH inspect and command
    patterns — command must win because the change verb is unambiguous."""
    available = {"inspect_selection", "get_analysis_summary",
                 "propose_section_change"}
    assert _pick_forced_tool(
        "5번 부재 단면을 H-400x400으로 변경해줘", available,
    ) == "propose_section_change"


def test_pick_forced_tool_falls_back_when_command_tool_absent():
    """If propose_section_change isn't enabled, command-matching messages
    still route via the inspect heuristic (for '5번 부재' style)."""
    available = {"inspect_selection", "get_analysis_summary"}
    # Falls through to inspect (the '5번 부재' part still matches inspect_re)
    assert _pick_forced_tool(
        "5번 부재 단면을 H-400x400으로 변경해줘", available,
    ) == "inspect_selection"


def test_extract_section_target_variants():
    assert _extract_section_target("H-400x400으로 바꿔") == "H-400x400"
    assert _extract_section_target("h-400으로") == "H-400"
    assert _extract_section_target("H400으로") == "H-400"          # normalize dash
    assert _extract_section_target("shs-200x200x6") == "SHS-200x200x6"
    assert _extract_section_target("그냥 인사") is None
    assert _extract_section_target("") is None
    assert _extract_section_target(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Orchestrator end-to-end: forced propose_section_change with extracted args
# ---------------------------------------------------------------------------

def _drain(async_gen) -> list[dict]:
    import asyncio

    async def _collect():
        return [line async for line in async_gen]

    lines = asyncio.run(_collect())
    return [json.loads(ln) for ln in lines if ln.strip()]


def _phaseb_registry() -> ToolRegistry:
    """Registry with edit group enabled regardless of process env state."""
    return ToolRegistry(
        [INSPECT_SELECTION_TOOL, GET_ANALYSIS_SUMMARY_TOOL,
         PROPOSE_SECTION_CHANGE_TOOL],
        enabled_groups=frozenset({"inspect", "summary", "edit"}),
    )


def test_orchestrator_force_calls_propose_section_change_with_extracted_args():
    """End-to-end: '5번 부재 단면을 H-400x400으로 변경해줘' → forced
    propose_section_change with member_id=5 + target_section=H-400x400."""
    aid = _seed_full_context()
    # Make sure member_id=5 exists in the seeded model. The default
    # _build_v2_model has 3 elements — extend so the heuristic test can
    # target a higher member.
    with _ANALYSIS_CONTEXT_LOCK:
        model = analysis_context_cache[aid]["model_json"]
        for i in range(4, 8):
            model["nodes"].append({
                "id": i, "x": float(i) * 1000.0, "y": 0.0, "z": 0.0, "story": 0,
            })
        # Element 4 connects node1-node2 (just for completeness — not
        # exercised by this test), element 5 too. We only need member_id
        # 5 to resolve via the 1-based sort-position rule, which means
        # the element with the 5th-smallest id.
        model["elements"].append({
            "id": 4, "node_i": 1, "node_j": 2,
            "elem_type": "column", "section": "H-300x300",
        })
        model["elements"].append({
            "id": 5, "node_i": 2, "node_j": 3,
            "elem_type": "column", "section": "H-300x300",
        })

    orch = ChatOrchestrator(NoopProvider(), registry=_phaseb_registry())
    events = _drain(orch.run_turn(
        session={"analysis_id": aid, "history": []},
        user_message="5번 부재 단면을 H-400x400으로 변경해줘",
    ))

    tool_calls = [e for e in events if e["type"] == "tool_call"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["tool"] == "propose_section_change"
    assert tool_calls[0]["arguments"] == {
        "member_id": 5, "target_section": "H-400x400",
    }

    tool_results = [e for e in events if e["type"] == "tool_result"]
    assert "preview_id" in tool_results[0]["result"]
    assert tool_results[0]["result"]["ui_action"] == "open_diff_preview"


def test_orchestrator_force_drops_when_target_section_missing():
    """'5번 부재 단면 바꿔줘' has no section — drop the force so the LLM
    can ask 'which section?'. Without this drop the tool would return
    target_section_required and the user would have to guess what to
    fix."""
    aid = _seed_full_context()
    # Scripted provider that declines tools when asked → mimics qwen
    # asking a clarifying question.
    class _DeclineProvider(BaseLLMProvider):
        name = "decline"

        def __init__(self):
            self.asked = False

        async def request_tool_call(self, *, messages, tools, temperature=None):
            self.asked = True
            return None

        async def stream_tokens(self, *, messages, temperature=None) -> AsyncIterator[str]:
            yield "어느 단면으로?"

    provider = _DeclineProvider()
    orch = ChatOrchestrator(provider, registry=_phaseb_registry())
    events = _drain(orch.run_turn(
        session={"analysis_id": aid, "history": []},
        user_message="5번 부재 단면 변경해줘",
    ))
    tool_calls = [e for e in events if e["type"] == "tool_call"]
    # No forced call — but the orchestrator still asked the LLM for a tool
    # decision (which declined). Either way, 0 tool calls happened.
    assert tool_calls == []
    assert provider.asked is True, (
        "When force drops, the orchestrator MUST fall back to LLM routing"
    )


def test_orchestrator_force_uses_ui_selection_when_member_id_implicit():
    """'단면을 H-400x400으로 변경' has section but no member_id text —
    the forced call should omit member_id (so the tool reads the UI
    selection on its own). End-to-end: with a UI selection on member 2,
    the resulting preview targets member 2."""
    aid = _seed_full_context()
    orch = ChatOrchestrator(NoopProvider(), registry=_phaseb_registry())
    events = _drain(orch.run_turn(
        session={"analysis_id": aid, "history": []},
        user_message="단면을 H-400x400으로 변경해줘",
        ui_context={"selected_element_ids": [2]},
    ))
    tool_calls = [e for e in events if e["type"] == "tool_call"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["arguments"] == {"target_section": "H-400x400"}
    tool_results = [e for e in events if e["type"] == "tool_result"]
    # Tool resolved member_id from the UI selection
    assert tool_results[0]["result"]["diff_summary"]["member_id"] == 2
