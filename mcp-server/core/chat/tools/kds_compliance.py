"""Phase D KDS compliance tool: ``explain_member_compliance``.

Looks up the selected (or explicitly id'd) member's design-check status
and runs the existing KDS-RAG pipeline (``make_kds_query`` →
``get_default_kds_retriever`` → ``_retrieve_evidence``) to fetch
validated KDS/AISC evidence. Returns evidence + a short member summary;
the chat LLM is expected to synthesise the Korean answer by quoting the
evidence verbatim.

This is the *first* chat tool that consumes the RAG layer — until now it
was only wired into the recommendation explainer. Reusing
``_retrieve_evidence`` (rather than re-implementing retrieve+validate)
means the same citation-validator + AISC temporary-reference dedupe runs
on the chat path.
"""
from __future__ import annotations

import logging
from typing import Optional

from ..tool_registry import ToolSpec


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports — keep mcp-server/core loadable without webapp on path.
# ---------------------------------------------------------------------------

def _get_analysis_context_fn():
    from app.services.analysis_context import _get_analysis_context
    return _get_analysis_context


def _get_default_retriever():
    from core.kds_rag import get_default_kds_retriever
    return get_default_kds_retriever()


def _retrieve_evidence_fn():
    from core.recommendation.explainer import _retrieve_evidence
    return _retrieve_evidence


def _make_kds_query_fn():
    from core.kds_rag import make_kds_query
    return make_kds_query


def _append_audit_fn():
    from app.services.chat_audit_log import append_audit
    return append_audit


# ---------------------------------------------------------------------------
# Session / argument helpers (mirror inspect.py / section_change.py)
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


def _coerce_int(raw) -> Optional[int]:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _resolve_target_id(arguments: dict, session: dict) -> Optional[int]:
    """Explicit ``member_id`` wins (typed by the user — '5번 부재 …'),
    otherwise fall back to the latest UI selection (single-member).
    Returns ``None`` when neither is present."""
    explicit = _coerce_int(arguments.get("member_id"))
    if explicit is not None:
        return explicit
    for entry in reversed(session.get("history") or []):
        ui_ctx = entry.get("ui_context") or {}
        sel = ui_ctx.get("selected_element_ids")
        if isinstance(sel, list) and sel:
            mid = _coerce_int(sel[0])
            if mid is not None:
                return mid
    return None


# ---------------------------------------------------------------------------
# Issue type mapping — VOCAB-LIMITED.
# ---------------------------------------------------------------------------
# ``make_kds_query`` only knows the keys in ``ISSUE_TYPE_KEYWORDS``
# (pipeline.py). Sending any other string degrades the query text to
# "<member_type> <section>" with no domain keywords → Voyage rerank
# scores collapse. So we map onto the existing vocab even for the
# reference-only ("이 부재 안전한가") path.
_OK_REFERENCE_ISSUE_TYPE = "missing_design_check"

# P3 — map the governing strength-side ratio onto its own retrieval
# bucket instead of collapsing axial/flexure/interaction all into
# ``strength_exceeded``. Each value is a real key in pipeline.py's
# ISSUE_TYPE_KEYWORDS / ISSUE_TYPE_TO_TOPIC / ISSUE_TYPE_TO_LIMIT_STATE
# maps — sending anything outside that vocabulary would strip the query
# of domain keywords and collapse Voyage rerank scores, so this dict and
# those maps MUST stay in lockstep.
_STRENGTH_ISSUE_TYPE_BY_RATIO: dict[str, str] = {
    "interaction": "strength_exceeded",
    "axial": "axial_exceeded",
    "bending": "flexure_exceeded",
}


def _governing_issue_type(ratios: dict) -> tuple[str, str]:
    """Return ``(issue_type, governing_ratio_name)``.

    - shear governs when shear>1.0 AND shear>interaction → shear_exceeded
    - otherwise the largest strength-side ratio decides the bucket:
      interaction → strength_exceeded, axial → axial_exceeded,
      bending → flexure_exceeded (P3 — finer than the old single
      ``strength_exceeded`` bucket).
    - all OK → missing_design_check (used as reference-only sentinel)
    """
    status = (ratios.get("status") or "OK").upper()
    r_shear = float(ratios.get("ratio_shear") or 0.0)
    r_inter = float(ratios.get("ratio_interaction") or 0.0)
    r_axial = float(ratios.get("ratio_axial") or 0.0)
    r_bend = float(ratios.get("ratio_bending") or 0.0)

    if status != "NG":
        # All checks passed (or no NG flag) — caller wants reference, not
        # diagnosis. Route through missing_design_check whose vocabulary
        # ("설계 검토 / 부재 검정") still gives a useful retrieval signal.
        return _OK_REFERENCE_ISSUE_TYPE, "ok"

    if r_shear > 1.0 and r_shear > r_inter:
        return "shear_exceeded", "shear"
    # Pick the largest of the strength-side ratios so the summary can
    # tell the user *which* limit dominates, then route the query through
    # that ratio's dedicated vocabulary bucket.
    candidates = {
        "interaction": r_inter,
        "axial": r_axial,
        "bending": r_bend,
    }
    governing = max(candidates, key=candidates.get) or "interaction"
    issue_type = _STRENGTH_ISSUE_TYPE_BY_RATIO.get(governing, "strength_exceeded")
    return issue_type, governing


def _aisc_proxy_standards(evidence: list[dict]) -> list[str]:
    """Standard ids in ``evidence`` that are AISC temporary references.

    The corpus uses AISC 360 chunks as stand-ins until KDS 14 31 00 /
    KDS 41 31 00 are ingested. Deriving the proxy state here — from the
    same ``doc_id`` strings the collapsible renders — gives the summary
    note and the audit record ONE source of truth, so all surfaces flag
    the proxy consistently (and it auto-empties once the real corpus
    lands, with no code change). Order-preserving dedup.
    """
    seen: list[str] = []
    for ev in evidence:
        doc = (ev.get("doc_id") or "").strip()
        if doc.upper().startswith("AISC") and doc not in seen:
            seen.append(doc)
    return seen


_ETYPE_KOREAN = {
    "column": "기둥",
    "beam_x": "X방향 보",
    "beam_y": "Y방향 보",
    "beam": "보",
    "brace": "가새",
}

_GOVERNING_KOREAN = {
    "shear": "전단(shear)",
    "interaction": "조합응력(P+M interaction)",
    "axial": "축력(axial)",
    "bending": "휨(bending)",
    "ok": "검토 통과",
}


def _format_member_block(summary: dict) -> str:
    """Render the fixed top section of every compliance response.

    Same shape regardless of rag_used so the chat widget renders a
    consistent member-id + ratio summary above any KDS quote block.
    """
    mid = summary.get("member_id")
    etype_ko = _ETYPE_KOREAN.get((summary.get("type") or "").strip(), "부재")
    section = summary.get("section") or "(단면 미상)"
    status = (summary.get("status") or "OK").upper()
    story = summary.get("story")
    ratios = summary.get("ratios") or {}
    gov = (summary.get("governing_ratio") or "").lower()
    gov_ko = _GOVERNING_KOREAN.get(gov, gov or "지배 항목 미상")
    story_part = f"{story}층 " if story is not None else ""

    def _fmt(r):
        try:
            return f"{float(r):.3f}"
        except (TypeError, ValueError):
            return "—"

    ratio_lines = (
        f"  • 조합응력(interaction): {_fmt(ratios.get('interaction'))}\n"
        f"  • 전단(shear): {_fmt(ratios.get('shear'))}\n"
        f"  • 축력(axial): {_fmt(ratios.get('axial'))}\n"
        f"  • 휨(bending): {_fmt(ratios.get('bending'))}"
    )

    if status == "NG":
        verdict = (
            f"부재 #{mid} ({story_part}{etype_ko}, {section}) — **NG (불합격)**.\n"
            f"지배 항목: {gov_ko} (해당 ratio가 1.0을 초과)"
        )
    else:
        verdict = (
            f"부재 #{mid} ({story_part}{etype_ko}, {section}) — **OK (합격)**.\n"
            "모든 설계 검토 항목이 1.0 이하입니다."
        )
    return f"{verdict}\n\n설계 검토 ratio:\n{ratio_lines}"


def _render_compliance_summary(
    summary: dict,
    *,
    evidence_count: int,
    rag_used: bool,
    aisc_proxy_standards: Optional[list[str]] = None,
) -> str:
    """The 'always-shown' top half of a compliance response.

    Member info + ratios + a one-line footer telling the user how many
    KDS/AISC citations are available (rendered into a collapsible by
    the chat widget — see :func:`_render_compliance_collapsible`).

    Kept short on purpose: the user said long answers feel noisy, so
    only the diagnostic essentials live here. The verbose evidence
    quote block sits behind the toggle.

    When ``aisc_proxy_standards`` is non-empty, an extra always-visible
    line flags that some citations are AISC temporary references. The
    collapsible already carries a fuller AISC disclaimer; surfacing it
    here too keeps the proxy state consistent for a user who never
    expands the toggle (and for the audit record — same derivation).
    """
    parts = [_format_member_block(summary)]
    if rag_used and evidence_count > 0:
        parts.append(
            f"\n\n📖 KDS/AISC 설계기준 근거 {evidence_count}건이 첨부됐습니다 "
            "(아래 '근거 자료 펼치기'를 클릭).\n"
            # Trust-calibration label (Codex P2). Lives in the
            # always-visible summary so the user sees the advisory caveat
            # without expanding the toggle — the citations are retrieval
            # results against the current corpus (which still mixes AISC
            # temporary references), not an authoritative KDS verdict.
            "※ 현재 코퍼스 기준 참고 근거이며 최종 설계판단은 아닙니다."
        )
        if aisc_proxy_standards:
            parts.append(
                "\n※ 인용 근거에 AISC 360 임시 참조가 포함되어 있습니다 "
                "(KDS 14 31 00 / 41 31 00 원문 확보 후 교체 예정)."
            )
    else:
        parts.append(
            "\n\n⚠️ KDS 근거 자료가 연결되지 않았습니다 "
            "(인덱스 미설정 또는 검색 결과 없음). "
            "현재 설계검토 수치만 안내합니다."
        )
    return "".join(parts)


def _render_compliance_collapsible(
    evidence: list,
    warnings: list,
    rag_used: bool,
) -> Optional[str]:
    """The 'click-to-expand' bottom half — evidence quotes verbatim.

    Returns ``None`` when there is nothing to collapse (no evidence) so
    the orchestrator can skip emitting the EVENT_COLLAPSIBLE event in
    that case. Anti-hallucination invariant identical to the summary
    half: this text is server-rendered, never touched by an LLM.
    """
    if not (rag_used and evidence):
        return None
    parts: list[str] = []
    for i, ev in enumerate(evidence, start=1):
        doc = (ev.get("doc_id") or "").strip()
        clause = (ev.get("clause") or "").strip()
        title = (ev.get("title") or "").strip()
        quote = (ev.get("quote") or "").strip()
        try:
            score = float(ev.get("score") or 0.0)
            score_str = f"  (relevance score: {score:.2f})"
        except (TypeError, ValueError):
            score_str = ""
        header = f"[{i}] {doc}"
        if clause:
            header += f" §{clause}"
        if title:
            header += f" — {title}"
        header += f"{score_str}\n"
        parts.append(header)
        parts.append(f"> {quote}\n\n")

    aisc_warn = next(
        (w for w in warnings if "aisc_temporary_reference" in w),
        None,
    )
    if aisc_warn:
        parts.append(
            "⚠️ 위 AISC 360 인용은 임시 참조입니다. KDS 14 31 00 / "
            "KDS 41 31 00 원문 코퍼스 확보 후 교체 검증 예정.\n"
        )
    return "".join(parts)




def _build_member_summary(
    member_id: int,
    info: dict,
    ratios: dict,
    governing_ratio: str,
) -> dict:
    return {
        "member_id": member_id,
        "type": info.get("etype"),
        "section": info.get("section"),
        "material": info.get("material"),
        "story": info.get("story"),
        "status": ratios.get("status", "OK"),
        "governing_ratio": governing_ratio,
        "ratios": {
            "interaction": ratios.get("ratio_interaction", 0),
            "shear": ratios.get("ratio_shear", 0),
            "axial": ratios.get("ratio_axial", 0),
            "bending": ratios.get("ratio_bending", 0),
        },
    }


def _current_turn(session: dict) -> int:
    """Return the 1-based chat turn count including the current user turn."""
    return sum(
        1 for h in (session.get("history") or [])
        if h.get("role") == "user"
    )


def _write_evidence_audit(
    *,
    analysis_id: str,
    member_id: int,
    session: dict,
    info: dict,
    member_summary: dict,
    issue_type: str,
    rag_context: dict,
    evidence: list[dict],
    rag_used: bool,
    warnings: list[str],
    aisc_proxy_standards: list[str],
) -> None:
    """Best-effort provenance write outside provider-visible history."""
    try:
        query = _make_kds_query_fn()(rag_context).to_dict()
    except Exception as exc:  # noqa: BLE001 - audit is best effort
        query = {"error": f"{type(exc).__name__}: {exc}"}

    try:
        append_audit = _append_audit_fn()
        append_audit({
            "analysis_id": analysis_id,
            "member_id": member_id,
            "turn": _current_turn(session),
            "session_id": session.get("session_id"),
            "member": {
                "member_id": member_id,
                "type": info.get("etype"),
                "section": info.get("section"),
                "material": info.get("material"),
                "story": info.get("story"),
            },
            "trigger": {
                "status": member_summary.get("status"),
                "governing_ratio": member_summary.get("governing_ratio"),
                "issue_type": issue_type,
                "ratios": dict(member_summary.get("ratios") or {}),
            },
            "query": query,
            "rag_used": rag_used,
            # Structured proxy state so audit/query can answer "was this
            # KDS-grounded or an AISC proxy?" without string-matching the
            # warnings list. Same derivation as the summary note — one
            # source of truth across surfaces.
            "evidence_provenance": {
                "has_aisc_proxy": bool(aisc_proxy_standards),
                "aisc_proxy_standards": list(aisc_proxy_standards),
            },
            "evidence": [
                {
                    "doc_id": ev.get("doc_id"),
                    "clause": ev.get("clause"),
                    "title": ev.get("title"),
                    "quote": ev.get("quote"),
                    "score": ev.get("score"),
                }
                for ev in evidence
            ],
            "warnings": list(warnings or []),
        })
    except Exception as exc:  # noqa: BLE001 - never break user-facing tool
        logger.warning(
            "failed to write KDS chat evidence audit: %s",
            exc,
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# Tool handler
# ---------------------------------------------------------------------------

def explain_member_compliance(arguments: dict, *, session: dict) -> dict:
    """Return KDS/AISC evidence + member summary for a single member.

    Selection priority:
      1. ``arguments.member_id`` (explicit number the LLM extracted from
         the user's "5번 부재 ..." phrasing).
      2. Latest ``ui_context.selected_element_ids[0]`` from session
         history — the 3D viewer attaches member_id to mesh userData so
         a click arrives here as a member_id under the legacy key name.

    Never raises — every failure mode returns a structured ``{error, code}``
    dict so the orchestrator can summarise it to the user.
    """
    # ---- 1. analysis_id + cache lookup -----------------------------------
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

    # ---- 2. member_id resolution -----------------------------------------
    member_id = _resolve_target_id(arguments, session)
    if member_id is None:
        return {
            "error": "근거를 찾을 부재를 알 수 없습니다. '5번 부재' 같이 "
                     "번호를 명시하거나, 3D 뷰어에서 먼저 부재를 클릭해 주세요.",
            "code": "no_selection",
        }

    info_by_member = ctx.get("member_info_by_member_id") or {}
    ratios_by_member = ctx.get("member_ratios_by_member_id") or {}
    info_by_elem = ctx.get("member_info_by_elem_id") or {}
    ratios_by_elem = ctx.get("member_ratios_by_elem_id") or {}

    key = str(member_id)
    # member_id is preferred (3D viewer sends member_id under the
    # selected_element_ids key — see inspect.py:146). elem_id is a
    # fallback for callers that genuinely pass an OpenSees sub-element id.
    info = info_by_member.get(key) or info_by_elem.get(key)
    ratios = ratios_by_member.get(key) or ratios_by_elem.get(key)
    if info is None and ratios is None:
        return {
            "error": f"부재 #{member_id}를 분석 결과에서 찾을 수 없습니다.",
            "code": "member_not_found",
        }
    info = info or {}
    ratios = ratios or {}

    # ---- 3. issue_type + RAG context ------------------------------------
    issue_type, governing_ratio = _governing_issue_type(ratios)
    member_summary = _build_member_summary(
        member_id, info, ratios, governing_ratio,
    )

    rag_context = {
        "issue_type": issue_type,
        "target": {
            "member_type": info.get("etype"),
            "section": info.get("section"),
            "material": info.get("material"),
        },
        # Stable discriminator so identical calls share a query_id.
        # Both ids must be strings (or absent) — make_kds_query feeds them
        # into "|".join, which raises if any element is None.
        "issue_id": f"chat_compliance_m{member_id}",
        "candidate_id": "",
    }

    # ---- 4. retrieve + validate ------------------------------------------
    retriever = _get_default_retriever()
    retrieve = _retrieve_evidence_fn()
    raw_top_k = _coerce_int(arguments.get("top_k")) or 3
    top_k = max(1, min(raw_top_k, 10))

    evidence_objs, warnings, rag_used = retrieve(
        rag_context, retriever, top_k=top_k,
    )
    evidence = [e.to_dict() for e in evidence_objs]
    # One AISC-proxy derivation shared by the audit record + summary note
    # so both surfaces flag the proxy state identically.
    aisc_proxy_standards = _aisc_proxy_standards(evidence)
    _write_evidence_audit(
        analysis_id=aid,
        member_id=member_id,
        session=session,
        info=info,
        member_summary=member_summary,
        issue_type=issue_type,
        rag_context=rag_context,
        evidence=evidence,
        rag_used=rag_used,
        warnings=warnings,
        aisc_proxy_standards=aisc_proxy_standards,
    )

    # ---- 5. Always-deterministic response, split into two parts --------
    # The chat widget renders ``mandatory_response_summary`` inline (always
    # visible) and ``mandatory_response_collapsible`` inside a click-to-
    # expand <details> element. Splitting keeps short answers feeling
    # short — the user complained that the unified block was always long
    # because evidence quotes dominate. Anti-hallucination invariant
    # unchanged: both halves are server-rendered, no LLM ever touches them.
    return {
        "analysis_id": aid,
        "member_summary": member_summary,
        "kds_evidence": evidence,
        "rag_used": rag_used,
        "warnings": warnings,
        "answer_hint": (
            "mandatory_response_summary + mandatory_response_collapsible를 "
            "그대로 사용자에게 전달합니다 (LLM 합성 우회 — 환각 방지)."
        ),
        "mandatory_response_summary": _render_compliance_summary(
            member_summary,
            evidence_count=len(evidence),
            rag_used=rag_used,
            aisc_proxy_standards=aisc_proxy_standards,
        ),
        "mandatory_response_collapsible": _render_compliance_collapsible(
            evidence, warnings, rag_used,
        ),
    }


EXPLAIN_MEMBER_COMPLIANCE_TOOL = ToolSpec(
    name="explain_member_compliance",
    group="kds",
    description=(
        "선택된 (또는 member_id로 지정된) 부재의 설계기준 근거를 KDS/AISC "
        "코퍼스에서 검색해 quote + 조항 + 점수를 반환. '왜 NG?', '이 부재 "
        "설계기준', 'KDS 어느 조항', '근거', '왜 안전한가?' 같은 질문에 "
        "사용. 인덱스가 비어있거나 키가 없으면 rag_used=false + 빈 "
        "evidence를 반환하므로 호출 측은 그 케이스를 사용자에게 그대로 "
        "안내하면 됩니다 (조항 발명 금지)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "member_id": {
                "type": "integer",
                "description": (
                    "1-based member id (3D 뷰어가 클릭에 실어 보내는 같은 "
                    "id). 사용자의 'N번 부재' 표현에서 추출. 생략 시 "
                    "최근 UI 선택을 사용."
                ),
            },
            "top_k": {
                "type": "integer",
                "description": "반환할 evidence 수 (default 3, max 10).",
            },
            "analysis_id": {
                "type": "string",
                "description": "session-bound analysis_id 오버라이드. 보통 생략.",
            },
        },
    },
    func=explain_member_compliance,
    creativity_hint="factual",
)
