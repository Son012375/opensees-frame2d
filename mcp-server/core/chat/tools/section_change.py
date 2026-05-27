"""Phase B edit tool: ``propose_section_change``.

Generates a *preview* of a single-member section change and stages the
resulting ``updated_model`` in :mod:`app.services.chat_preview_cache`.
The chat tool result returns only the preview_id, a human-readable diff
summary, and a ``ui_action`` hint — never the ``updated_model`` itself,
which would trip the chat-stream FORBIDDEN_KEYS guard (see
``mcp-server/core/chat/streaming.py``).

Apply happens client-side: the chat widget reads ``ui_action ==
"open_diff_preview"``, calls
``window.EditorV2ChatBridge.openDiffPreview({preview_id})`` which
fetches the staged model via ``GET /api/v2/recommendations/chat-preview/
{preview_id}`` and feeds it into the existing rec-diff modal. The
modal's [Apply] button runs the existing ``applyRecDiff()`` path
unchanged — Phase B adds no new mutation code on the chat side.

Member identification
---------------------
``member_id`` is the 1-based sort-position id the analyzer assigns and
the 3D viewer attaches to clicks. We pass it through ``target.member_id``
only — never ``target.element_id`` — so ``apply_candidate_to_model``
resolves via the sort-position lookup (``_find_element_by_member_id``)
that matches the analyzer's index. Setting ``target.element_id`` would
cause the executor to mutate the V2 element whose raw ``id`` field
matches numerically, which is a different member entirely once
``num_elements_per_member > 1``.
"""
from __future__ import annotations

import uuid
from typing import Optional

from ..tool_registry import ToolSpec


# ---------------------------------------------------------------------------
# Lazy cache imports (mirror inspect.py pattern — keeps mcp-server/core
# loadable without webapp being on the path)
# ---------------------------------------------------------------------------

def _get_analysis_context_fn():
    from app.services.analysis_context import _get_analysis_context
    return _get_analysis_context


def _get_preview_put_fn():
    from app.services.chat_preview_cache import put_preview
    return put_preview


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class SectionResolutionError(ValueError):
    """Raised when target_section can't be uniquely resolved against the catalog.

    ``options`` carries alternative section names the user might pick —
    surfaced to the chat answer so the user has an actionable next step
    instead of just "not found".
    """

    def __init__(self, reason: str, options: Optional[list[str]] = None):
        super().__init__(reason)
        self.reason = reason
        self.options = list(options or [])


# ---------------------------------------------------------------------------
# Section validation / normalization
# ---------------------------------------------------------------------------

def _normalize_member_type(elem_type: str) -> str:
    """V2 model elem_type ('column'/'beam'/'brace') → ladder member_type.

    The section catalog only knows two sub-ladders for the H family
    (column = square; beam = wide). Braces aren't a Phase B target —
    they fall through to 'beam' here so a future brace change at least
    routes to a non-empty ladder rather than KeyError-ing in
    list_family_ladder.
    """
    if elem_type == "column":
        return "column"
    return "beam"


def resolve_section_name(target: str, member_type: str) -> str:
    """Validate + normalize ``target`` against the section catalog.

    Resolution order:
      1. Exact catalog match → verify member_type compatibility
         (column-only ladder for H-square, beam-only for H-wide).
      2. Shorthand path ("H-400" without second dim): look for entries
         in the member_type's preferred ladder whose name starts with
         ``target + "x"``. Unique match → normalized name; zero or
         multiple → raise with suggestions.

    Raises
    ------
    SectionResolutionError
        With ``options`` populated for the chat layer to surface to the
        user.
    """
    from core.recommendation.section_catalog import (
        get_section_metadata,
        list_family_ladder,
        parse_family,
    )

    if not target or not isinstance(target, str):
        raise SectionResolutionError("단면명이 비어있습니다.")

    target = target.strip()
    family = parse_family(target)
    if family is None:
        raise SectionResolutionError(
            f"'{target}' — 단면명 형식을 인식할 수 없습니다. "
            "예) H-400x400, SHS-200x200x6"
        )

    ladder = list_family_ladder(family, member_type)
    if not ladder:
        raise SectionResolutionError(
            f"카탈로그에 {family} 패밀리({member_type})가 비어 있습니다.",
        )
    ladder_names = {e.name for e in ladder}

    meta = get_section_metadata(target)
    if meta is not None:
        if target in ladder_names:
            return target
        # Catalogued but wrong sub-ladder (e.g. H-square asked for beam).
        suggestions = [e.name for e in ladder[:5]]
        raise SectionResolutionError(
            f"'{target}'은(는) {member_type} 부재에 적합하지 않습니다. "
            f"가능한 단면: {', '.join(suggestions)}",
            options=suggestions,
        )

    # Shorthand: "H-400" → look for "H-400x..." in the preferred ladder.
    prefix = target + "x"
    matches = [e.name for e in ladder if e.name.startswith(prefix)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        suggestions = [e.name for e in ladder[:5]]
        raise SectionResolutionError(
            f"카탈로그에 '{target}'에 해당하는 단면이 없습니다. "
            f"가능한 단면 예: {', '.join(suggestions)}",
            options=suggestions,
        )
    raise SectionResolutionError(
        f"'{target}'은(는) 여러 단면과 매칭됩니다. 정확한 단면명을 지정해 주세요. "
        f"후보: {', '.join(matches[:5])}",
        options=matches,
    )


# ---------------------------------------------------------------------------
# Candidate synthesis
# ---------------------------------------------------------------------------

def synthesize_user_directed_candidate(
    model_json: dict,
    member_id: int,
    target_section: str,
):
    """Return ``(candidate, preview_meta)`` for a user-directed section change.

    ``member_id`` is the 1-based sort-position id that the analyzer +
    3D viewer use. ``target_section`` may be exact ("H-400x400") or
    shorthand ("H-400") — the latter is only accepted when it
    uniquely resolves inside the member's family ladder.

    Raises
    ------
    SectionResolutionError
        Target section is ambiguous / missing from catalog.
    ValueError
        Member not found, target equals current, or invalid input.
    """
    from core.recommendation import (
        ActionType,
        ChangeOperation,
        Confidence,
        RetrofitCandidate,
    )
    from core.recommendation.apply_candidate import _find_element_by_member_id

    elements = model_json.get("elements") or []
    elem = _find_element_by_member_id(elements, member_id)
    if elem is None:
        raise ValueError(
            f"부재 #{member_id}를 모델에서 찾을 수 없습니다 "
            f"(모델 요소 총 {len(elements)}개)."
        )

    current = elem.get("section")
    elem_type = elem.get("elem_type") or ""
    member_type = _normalize_member_type(elem_type)

    # May raise SectionResolutionError — propagated to the chat layer
    # so the user gets a clear next step.
    to_section = resolve_section_name(target_section, member_type)

    if current == to_section:
        raise ValueError(
            f"부재 #{member_id}는 이미 '{to_section}' 단면입니다 — 변경 없음."
        )

    candidate = RetrofitCandidate(
        candidate_id=f"chat_{uuid.uuid4().hex[:8]}",
        issue_id="user_directed",
        action_type=ActionType.INCREASE_SECTION,
        description=f"부재 #{member_id}: {current} → {to_section} (사용자 지시)",
        member_id=member_id,
        # Intentionally NOT setting element_id. apply_candidate.py's
        # _resolve_target_element falls back to _find_element_by_member_id
        # when element_id is None — which gives the same lookup used here.
        element_id=None,
        confidence=Confidence.MEDIUM,
        target={
            "member_id": member_id,
            "member_type": elem_type,
        },
        proposed_change={
            "operation": ChangeOperation.REPLACE_SECTION,
            "from": current,
            "to": to_section,
            "applicable": True,
            "reason": "user_directed",
            "requires_user_selection": False,
        },
        metadata={
            "source": "chat_command",
            "user_directed": True,
            "actor": "user",
        },
    )

    preview_meta = {
        "member_id": member_id,
        "member_type": elem_type,
        "section_from": current,
        "section_to": to_section,
    }
    return candidate, preview_meta


# ---------------------------------------------------------------------------
# Session helpers (copies of inspect.py logic — kept local so the two
# tool files stay independently auditable)
# ---------------------------------------------------------------------------

def _resolve_analysis_id(arguments: dict, session: dict) -> str:
    aid = arguments.get("analysis_id")
    if aid:
        return aid
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


def _resolve_member_id_from_session(session: dict) -> Optional[int]:
    """Fallback when the LLM omits ``member_id`` — read the latest
    UI selection from history. Returns the first selected id (the rec
    apply path is single-member by design)."""
    history = session.get("history") or []
    for entry in reversed(history):
        ui_ctx = entry.get("ui_context") or {}
        sel = ui_ctx.get("selected_element_ids")
        if isinstance(sel, list) and sel:
            try:
                return int(sel[0])
            except (TypeError, ValueError):
                continue
    return None


# ---------------------------------------------------------------------------
# Chat tool: propose_section_change
# ---------------------------------------------------------------------------

def propose_section_change(arguments: dict, *, session: dict) -> dict:
    """Generate a section-change preview and stage it for the rec diff modal.

    Required arguments:
        ``member_id`` (int): the 1-based sort-position id the 3D viewer
            sends. Falls back to the latest UI selection if omitted.
        ``target_section`` (str): exact ("H-400x400") or shorthand
            ("H-400") if it uniquely resolves.

    Optional:
        ``analysis_id`` (str): override session-bound id.

    Returns a dict with at least ``preview_id``, ``diff_summary``,
    ``ui_action``, and ``ui_payload``. Never returns ``updated_model``
    (FORBIDDEN_KEYS guard would refuse to serialize it).
    """
    from core.recommendation import apply_candidate_to_model
    from core.recommendation.apply_candidate import InapplicableCandidateError

    # ---- 1. Parse arguments ------------------------------------------------
    raw_mid = arguments.get("member_id")
    member_id: Optional[int] = None
    if raw_mid is not None:
        try:
            member_id = int(raw_mid)
        except (TypeError, ValueError):
            return {
                "error": "member_id는 정수여야 합니다.",
                "code": "bad_arguments",
            }
    if member_id is None:
        member_id = _resolve_member_id_from_session(session)
    if member_id is None:
        return {
            "error": "변경할 부재를 알 수 없습니다. '5번 부재' 같이 번호를 "
                     "명시하거나, 3D 뷰어에서 먼저 부재를 클릭해 주세요.",
            "code": "member_id_required",
        }

    target_section = (arguments.get("target_section") or "").strip()
    if not target_section:
        return {
            "error": "target_section (변경할 단면명) 인자가 필요합니다. "
                     "예) 'H-400x400'",
            "code": "target_section_required",
        }

    # ---- 2. Resolve analysis context + guard against compact cache -------
    try:
        aid = _resolve_analysis_id(arguments, session)
    except ValueError as exc:
        return {"error": str(exc), "code": "analysis_id_required"}

    get_ctx = _get_analysis_context_fn()
    ctx = get_ctx(aid)
    if ctx is None:
        return {
            "error": f"분석 컨텍스트 '{aid}'가 만료되었거나 없습니다. "
                     "좌측 패널에서 분석을 다시 실행해 주세요.",
            "code": "analysis_not_found",
        }
    if ctx.get("context_kind") != "full":
        return {
            "error": "이 분석은 compact 컨텍스트만 가지고 있어 단면 변경을 "
                     "적용할 수 없습니다. V2 node-element 경로 "
                     "(/api/v2/analyze)로 다시 실행해 주세요.",
            "code": "compact_context_rejected",
        }

    model_json = ctx.get("model_json")
    if not isinstance(model_json, dict):
        return {
            "error": "분석 컨텍스트에 model_json이 없습니다.",
            "code": "model_json_missing",
        }

    # ---- 3. Synthesize user-directed candidate ----------------------------
    try:
        candidate, preview_meta = synthesize_user_directed_candidate(
            model_json, member_id, target_section,
        )
    except SectionResolutionError as exc:
        return {
            "error": str(exc),
            "code": "section_resolution_failed",
            "options": exc.options,
        }
    except ValueError as exc:
        return {"error": str(exc), "code": "synthesis_failed"}

    # ---- 4. Apply to a deep copy (NOT to the cached model) ----------------
    try:
        updated_model, diff = apply_candidate_to_model(model_json, candidate)
    except InapplicableCandidateError as exc:
        return {
            "error": f"단면 변경 적용 실패: {exc}",
            "code": "apply_failed",
        }

    # ---- 5. Stage in preview cache; chat result carries preview_id only --
    rows = list(diff.changed_members)
    primary = rows[0] if rows else {}
    diff_summary = {
        "member_id": preview_meta["member_id"],
        "member_type": preview_meta["member_type"],
        "section_from": preview_meta["section_from"],
        "section_to": preview_meta["section_to"],
        "story": primary.get("story"),
        "member_label": primary.get("member_label", ""),
        "changed_member_count": diff.changed_member_count,
    }

    preview_id = f"chat_prev_{uuid.uuid4().hex[:12]}"
    put_preview = _get_preview_put_fn()
    put_preview(preview_id, {
        "analysis_id": aid,
        "updated_model": updated_model,
        "diff": {
            "candidate_id": diff.candidate_id,
            "operation": diff.operation,
            "reason": diff.reason,
            "changed_member_count": diff.changed_member_count,
            "changed_members": diff.changed_members,
        },
        "candidate": candidate.to_dict(),
        "preview_meta": preview_meta,
    })

    return {
        "preview_id": preview_id,
        "analysis_id": aid,
        "diff_summary": diff_summary,
        "ui_action": "open_diff_preview",
        "ui_payload": {"preview_id": preview_id, "source": "chat"},
        "next_step_hint": (
            "미리보기 모달이 열립니다. 부재/단면 변경 내역을 확인하고 "
            "[Apply to editor] 버튼을 눌러야 모델에 반영되고 재해석이 "
            "시작됩니다. 사용자가 직접 누르기 전까지는 적용되지 않습니다."
        ),
    }


PROPOSE_SECTION_CHANGE_TOOL = ToolSpec(
    name="propose_section_change",
    group="edit",
    description=(
        "Generate a *preview* of a single-member section change. "
        "Stages the modified model server-side and signals the UI to open "
        "the diff-preview modal so the user can review and Apply. "
        "DOES NOT mutate the model or trigger reanalysis directly — Apply "
        "only happens when the user clicks the modal's [Apply] button. "
        "Use when the user explicitly asks to change a member's section "
        "(e.g. '5번 부재 단면을 H-400x400으로 바꿔줘'). "
        "Always pass both member_id (integer) and target_section "
        "(catalog name like 'H-400x400'). After this tool returns, your "
        "reply should be exactly one short sentence telling the user the "
        "preview modal is open — NEVER claim the change has been applied."
    ),
    parameters={
        "type": "object",
        "properties": {
            "member_id": {
                "type": "integer",
                "description": (
                    "1-based member id (same id the 3D viewer sends on click). "
                    "Extract from the user's '5번 부재' / 'member 5' phrasing. "
                    "If omitted the tool falls back to the latest UI selection."
                ),
            },
            "target_section": {
                "type": "string",
                "description": (
                    "Target section name. Exact preferred "
                    "(e.g. 'H-400x400', 'SHS-200x200x6'). Shorthand 'H-400' "
                    "is accepted only when it uniquely resolves in the "
                    "member's family ladder."
                ),
            },
            "analysis_id": {
                "type": "string",
                "description": (
                    "Override session-bound analysis_id. Usually omitted."
                ),
            },
        },
        "required": ["target_section"],
    },
    func=propose_section_change,
    creativity_hint="factual",
)
