"""Phase A.2 read-only inspect tools.

Both tools read ``app.services.analysis_context.analysis_context_cache``
populated by ``/api/v2/analyze`` (see Phase 0 Step 0-2's compact
subset). Neither mutates state.

The cache import is deferred to first call: this module is loaded from
``mcp-server/core/chat/tools/`` and the webapp's path injection happens
when chat_router imports it from FastAPI startup. Lazy import keeps
``core.chat`` usable from non-webapp callers too (CLI, scripts).
"""
from __future__ import annotations

from typing import Any

from ..tool_registry import ToolSpec


# ---------------------------------------------------------------------------
# Cache + selection resolution helpers
# ---------------------------------------------------------------------------

def _get_cache():
    """Returns ``(cache_dict, threading.Lock)``. Lazy to avoid module-load
    coupling between mcp-server and webapp."""
    from app.services.analysis_context import (
        _ANALYSIS_CONTEXT_LOCK,
        analysis_context_cache,
    )
    return analysis_context_cache, _ANALYSIS_CONTEXT_LOCK


def _resolve_analysis_id(arguments: dict, session: dict) -> str:
    """Resolve the analysis the tool should target.

    Precedence — explicit > fresh > stale:

        1. ``arguments.analysis_id`` — LLM was given a specific id.
        2. Latest ``ui_context.analysis_id`` from session history — the
           widget refreshes this every turn, so it tracks the user's
           currently-open analysis even if the session was created
           before any analysis ran or the user re-ran ``/api/v2/analyze``
           and got a new ``currentJobId``.
        3. ``session.analysis_id`` — the binding set at
           ``POST /sessions``. Fallback for the no-widget case
           (e.g. curl / pytest).

    Raises ``ValueError`` if none of the three is set so a misconfigured
    session surfaces a clear error instead of a silent empty result.
    """
    aid = arguments.get("analysis_id")
    if aid:
        return aid
    # The chat widget attaches ui_context to every /messages payload and
    # the orchestrator stows it on the user-turn history entry. Reading
    # in reverse means the freshest analysis id wins — important when
    # the user re-runs /api/v2/analyze mid-session.
    for entry in reversed(session.get("history") or []):
        ui_ctx = entry.get("ui_context") or {}
        ui_aid = ui_ctx.get("analysis_id")
        if ui_aid:
            return ui_aid
    aid = session.get("analysis_id")
    if not aid:
        raise ValueError(
            "analysis_id is required: pass it in the tool call, attach it "
            "to ui_context, or bind the chat session via POST /sessions "
            "{analysis_id: ...}"
        )
    return aid


def _resolve_element_ids(arguments: dict, session: dict) -> list[int]:
    """Explicit ``element_ids`` wins. Otherwise fall back to the latest
    ``ui_context.selected_element_ids`` the widget attached to a user
    turn (the EditorV2ChatBridge selection)."""
    explicit = arguments.get("element_ids")
    if explicit:
        return [int(e) for e in explicit]
    history = session.get("history") or []
    for entry in reversed(history):
        ui_ctx = entry.get("ui_context") or {}
        sel = ui_ctx.get("selected_element_ids") or []
        if sel:
            return [int(e) for e in sel]
    return []


# ---------------------------------------------------------------------------
# inspect_selection
# ---------------------------------------------------------------------------

def inspect_selection(arguments: dict, *, session: dict) -> dict:
    """Look up section/material/story/ratios for one or more elements."""
    aid = _resolve_analysis_id(arguments, session)
    cache, lock = _get_cache()
    with lock:
        ctx = cache.get(aid)
    if ctx is None:
        return {
            "error": f"unknown analysis_id: {aid}",
            "code": "analysis_not_found",
            "elements": [],
        }

    eids = _resolve_element_ids(arguments, session)
    if not eids:
        return {
            "error": "no element selected — click an element in the 3D viewer or pass element_ids",
            "code": "no_selection",
            "elements": [],
        }

    member_info = ctx.get("member_info_by_elem_id") or {}
    member_ratios = ctx.get("member_ratios_by_elem_id") or {}

    elements: list[dict[str, Any]] = []
    for eid in eids:
        key = str(eid)
        info = member_info.get(key)
        ratios = member_ratios.get(key)
        if info is None and ratios is None:
            elements.append({"element_id": eid, "found": False})
            continue
        elements.append({
            "element_id": eid,
            "found": True,
            "info": info or {},
            "ratios": ratios or {},
        })
    return {"analysis_id": aid, "elements": elements}


INSPECT_SELECTION_TOOL = ToolSpec(
    name="inspect_selection",
    group="inspect",
    description=(
        "Inspect one or more elements from the most recent analysis. Returns "
        "section, material, story, and design-check ratios (interaction / "
        "shear / axial / bending) for each. If element_ids is omitted, uses "
        "the user's current 3D-viewer selection."
    ),
    parameters={
        "type": "object",
        "properties": {
            "element_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Element ids to inspect. If omitted, uses the latest UI selection.",
            },
            "analysis_id": {
                "type": "string",
                "description": "Override the session-bound analysis_id. Usually omitted.",
            },
        },
    },
    func=inspect_selection,
)


# ---------------------------------------------------------------------------
# get_analysis_summary
# ---------------------------------------------------------------------------

def get_analysis_summary(arguments: dict, *, session: dict) -> dict:
    """Compact overview: max disp / drift / force, NG count, modal periods."""
    aid = _resolve_analysis_id(arguments, session)
    cache, lock = _get_cache()
    with lock:
        ctx = cache.get(aid)
    if ctx is None:
        return {
            "error": f"unknown analysis_id: {aid}",
            "code": "analysis_not_found",
        }
    return {
        "analysis_id": aid,
        "summary": ctx.get("analysis_summary") or {},
        "modal": ctx.get("modal_summary") or {},
    }


GET_ANALYSIS_SUMMARY_TOOL = ToolSpec(
    name="get_analysis_summary",
    group="summary",
    description=(
        "Compact overview of the current analysis: max displacements, drifts, "
        "force envelope, NG member count, number of stories, plus the first "
        "three modal periods and mass participation. Use this when the user "
        "asks 'how did it go', '결과 요약', 'NG 부재 몇 개' etc."
    ),
    parameters={
        "type": "object",
        "properties": {
            "analysis_id": {
                "type": "string",
                "description": "Override the session-bound analysis_id. Usually omitted.",
            },
        },
    },
    func=get_analysis_summary,
)
