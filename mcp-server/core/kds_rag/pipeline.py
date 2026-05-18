"""KDS-RAG pipeline: query construction + ref enrichment + payload adapter.

This module wires :class:`CodeReference` (citation-side, from
``core.recommendation``) to :class:`KDSChunk` (source-side, here).

Three layers:

    1. Query building
        build_kds_queries_for_issue(issue) -> [KDSRetrievalQuery]
        build_kds_queries_for_candidate(candidate, issue) -> [KDSRetrievalQuery]

    2. Chunk → reference adapter
        chunk_to_code_reference(chunk, base_ref) -> CodeReference

    3. Enrichment
        enrich_code_refs_with_kds(issues, candidates, retriever, top_k)
            → mutates ``code_refs`` on each issue/candidate (in-place)
              and returns a summary dict including warnings + counts.

        enrich_recommendation_payload_with_kds(payload, retriever, top_k)
            → operates on the JSON-shaped payload returned by
              ``build_recommendation_payload``. Does NOT touch the
              recommendation pipeline itself.

Nothing here calls an LLM or a real vector DB. The
:class:`InMemoryKDSRetriever` test double is the reference backend.
"""
from __future__ import annotations

import copy
import hashlib
from typing import Any, Iterable, Optional, TYPE_CHECKING

from .citation_validator import (
    MAX_QUOTE_LEN,
    validate_code_reference,
)
from .retriever import KDSRetriever
from .schemas import (
    KDSChunk,
    KDSRetrievalQuery,
    KDSRetrievalResult,
)

if TYPE_CHECKING:
    from ..recommendation.schemas import (  # noqa: F401
        CodeReference, StructuralIssue, RetrofitCandidate,
    )


# ---------------------------------------------------------------------------
# Query building
# ---------------------------------------------------------------------------

def _short_digest(parts: list[str]) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:10]


def _query_text_from_ref(ref: "CodeReference") -> str:
    """Reasonable free-text query built from a CodeReference's hints."""
    pieces: list[str] = []
    if getattr(ref, "query_hint", None):
        pieces.append(str(ref.query_hint))
    else:
        # Fall back to whatever discriminating fields we have.
        for k in ("title", "standard_id", "clause_id", "topic", "limit_state"):
            v = getattr(ref, k, None)
            if v:
                pieces.append(str(v))
    return " ".join(p for p in pieces if p).strip()


def _query_for_ref(
    ref: "CodeReference",
    *,
    issue_id: Optional[str] = None,
    candidate_id: Optional[str] = None,
) -> Optional[KDSRetrievalQuery]:
    """Build a query for a single ref, or ``None`` if it lacks info."""
    qtext = _query_text_from_ref(ref)
    if not qtext:
        return None

    discriminator = "|".join([
        getattr(ref, "standard_id", "") or "",
        getattr(ref, "clause_id", "") or "",
        getattr(ref, "topic", "") or "",
        getattr(ref, "limit_state", "") or "",
        getattr(ref, "material", "") or "",
        issue_id or "",
        candidate_id or "",
        qtext,
    ])
    qid = f"kds_q_{_short_digest([discriminator])}"

    return KDSRetrievalQuery(
        query_id=qid,
        query_text=qtext,
        topic=getattr(ref, "topic", None),
        material=getattr(ref, "material", None),
        limit_state=getattr(ref, "limit_state", None),
        standard_id=getattr(ref, "standard_id", None),
        issue_id=issue_id,
        candidate_id=candidate_id,
    )


# ---------------------------------------------------------------------------
# Context-driven query builder (used by the explain endpoint)
# ---------------------------------------------------------------------------

# Deterministic keyword maps. The explainer hits these *before* the LLM /
# RAG so query text is never empty even when code_refs are missing.
# Keep lists short — Voyage rerank scores collapse with noisy queries.
ISSUE_TYPE_KEYWORDS: dict[str, list[str]] = {
    "strength_exceeded": ["부재 강도", "조합응력 비", "interaction ratio"],
    "shear_exceeded": ["전단강도", "전단 검토", "shear strength"],
    "drift_exceeded": ["층간변위", "허용 층간변위비", "drift limit", "사용성"],
    "missing_design_check": ["설계 검토", "부재 검정"],
    "analysis_warning": ["해석 경고", "수치 안정성"],
    "compression_capacity": ["압축강도", "좌굴", "유효좌굴길이"],
    "tension_capacity": ["인장강도"],
}

ACTION_TYPE_KEYWORDS: dict[str, list[str]] = {
    "increase_section": ["단면 증대", "단면 강성", "내력 증가"],
    "replace_section": ["단면 변경", "부재 단면"],
    "replace_sections_by_story": ["층별 기둥 단면", "층 강성"],
    "change_material": ["강재 등급", "재료 강도"],
    "add_member": ["부재 추가"],
    "add_lateral_resistance": ["횡저항 시스템", "가새", "전단벽", "모멘트골조"],
    "change_support": ["지지 조건"],
    "requires_engineer_review": ["엔지니어 검토"],
    "manual_review": ["수동 검토"],
}

ISSUE_TYPE_TO_LIMIT_STATE: dict[str, str] = {
    "strength_exceeded": "strength",
    "shear_exceeded": "shear_strength",
    "drift_exceeded": "drift_limit",
    "compression_capacity": "compression_strength",
    "tension_capacity": "tension_strength",
}

ISSUE_TYPE_TO_TOPIC: dict[str, str] = {
    "strength_exceeded": "member_strength",
    "shear_exceeded": "member_shear",
    "drift_exceeded": "story_drift",
    "compression_capacity": "member_compression",
}


def make_kds_query(context: dict) -> KDSRetrievalQuery:
    """Build a KDSRetrievalQuery from an explainer context dict.

    The explainer context (see ``core.recommendation.explainer.build_explanation_context``)
    is a plain dict, not a typed CodeReference / Issue / Candidate, so
    this is the deterministic mapping layer that turns it into a useful
    search query. It never raises and always returns a query, even when
    the context is sparse — the resulting ``query_text`` falls back to
    the action_type or "kds search" rather than an empty string so the
    upstream retriever can still produce a deterministic (possibly empty)
    result instead of failing.
    """
    issue_type = context.get("issue_type")
    action_type = context.get("action_type")
    target = context.get("target") or {}
    proposed = context.get("proposed_change") or {}

    keywords: list[str] = []
    keywords.extend(ISSUE_TYPE_KEYWORDS.get(issue_type or "", []))
    keywords.extend(ACTION_TYPE_KEYWORDS.get(action_type or "", []))

    member_type = target.get("member_type")
    if member_type:
        keywords.append(str(member_type))

    section_from = proposed.get("from") or {}
    section_to = proposed.get("to") or {}
    for sec in (section_from, section_to):
        if isinstance(sec, dict):
            sec_id = sec.get("section") or sec.get("section_id")
            if sec_id:
                keywords.append(str(sec_id))

    # Stable de-dup that preserves order.
    seen: set[str] = set()
    dedup: list[str] = []
    for k in keywords:
        s = str(k).strip()
        if s and s not in seen:
            seen.add(s)
            dedup.append(s)

    query_text = " ".join(dedup) or (action_type or "kds search")

    discriminator = "|".join([
        context.get("candidate_id", ""),
        context.get("issue_id", ""),
        issue_type or "",
        action_type or "",
        query_text,
    ])
    qid = f"explain_q_{_short_digest([discriminator])}"

    return KDSRetrievalQuery(
        query_id=qid,
        query_text=query_text,
        topic=ISSUE_TYPE_TO_TOPIC.get(issue_type or ""),
        material=target.get("material"),
        limit_state=ISSUE_TYPE_TO_LIMIT_STATE.get(issue_type or ""),
        standard_id=None,
        issue_id=context.get("issue_id"),
        candidate_id=context.get("candidate_id"),
    )


def build_kds_queries_for_issue(issue: "StructuralIssue") -> list[KDSRetrievalQuery]:
    """One query per CodeReference attached to the issue."""
    out: list[KDSRetrievalQuery] = []
    for ref in (issue.code_refs or []):
        q = _query_for_ref(ref, issue_id=issue.issue_id)
        if q is not None:
            out.append(q)
    return out


def build_kds_queries_for_candidate(
    candidate: "RetrofitCandidate",
    issue: Optional["StructuralIssue"] = None,
) -> list[KDSRetrievalQuery]:
    """Queries from a candidate's own code_refs + (optionally) its issue.

    A candidate inherits the issue's RAG context — if the issue carries
    hints, we include those too. Duplicates by ``query_id`` are removed.
    """
    seen: set[str] = set()
    out: list[KDSRetrievalQuery] = []

    def _push(q: Optional[KDSRetrievalQuery]) -> None:
        if q is None:
            return
        if q.query_id in seen:
            return
        seen.add(q.query_id)
        out.append(q)

    for ref in (candidate.code_refs or []):
        _push(_query_for_ref(
            ref,
            issue_id=candidate.issue_id,
            candidate_id=candidate.candidate_id,
        ))

    if issue is not None:
        for ref in (issue.code_refs or []):
            _push(_query_for_ref(
                ref,
                issue_id=issue.issue_id,
                candidate_id=candidate.candidate_id,
            ))

    return out


# ---------------------------------------------------------------------------
# Chunk → CodeReference adapter
# ---------------------------------------------------------------------------

def chunk_to_code_reference(
    chunk: KDSChunk,
    base_ref: Optional["CodeReference"] = None,
) -> "CodeReference":
    """Build a citation-form CodeReference from a retrieved chunk.

    If ``base_ref`` is provided, its hint fields are preserved when the
    chunk doesn't override them. The chunk's ``text`` becomes the
    ``quote`` directly — even if it's too long. The validator then
    decides whether to accept it. We never silently truncate, since the
    quote-length cap is a guardrail, not a styling choice.
    """
    # Local import — avoid circularity at module load time.
    from ..recommendation.schemas import CodeReference

    quote: Optional[str] = None
    if chunk.text:
        t = chunk.text.strip()
        if t:
            quote = t  # validator catches over-length quotes.

    def _pick(field: str) -> Any:
        v = getattr(chunk, field, None)
        if v is not None and v != "":
            return v
        return getattr(base_ref, field, None) if base_ref else None

    return CodeReference(
        standard_id=chunk.standard_id or (
            base_ref.standard_id if base_ref else ""
        ),
        version=_pick("version"),
        clause_id=_pick("clause_id"),
        title=_pick("title"),
        source_url=_pick("source_url"),
        quote=quote,
        relevance_reason=(
            f"matched chunk {chunk.chunk_id} via "
            f"topic={chunk.topic!r} limit_state={chunk.limit_state!r}"
        ),
        chunk_id=chunk.chunk_id or None,
        query_hint=getattr(base_ref, "query_hint", None) if base_ref else None,
        topic=_pick("topic"),
        material=_pick("material"),
        limit_state=_pick("limit_state"),
        jurisdiction=_pick("jurisdiction") or "KDS",
        confidence=None,  # Caller may set after validation
    )


# ---------------------------------------------------------------------------
# Enrichment — object level
# ---------------------------------------------------------------------------

def _enrich_one(
    holder: Any,
    retriever: KDSRetriever,
    top_k: int,
    queries: list[KDSRetrievalQuery],
    accumulated_warnings: list[str],
) -> dict[str, int]:
    """Mutate ``holder.code_refs`` in place. Returns per-holder counters.

    ``holder`` is a :class:`StructuralIssue` or :class:`RetrofitCandidate`.
    Both have ``code_refs: list[CodeReference]``; we keep the original
    hint refs and *append* the enriched citation-form refs after them.
    Validator runs on the appended ones only.
    """
    counters = {
        "queries": 0, "chunks": 0, "attached": 0,
        "unresolved": 0, "rejected": 0,
    }
    if not queries:
        return counters

    hint_refs = list(holder.code_refs or [])
    new_refs: list = []

    for q in queries:
        counters["queries"] += 1
        try:
            result: KDSRetrievalResult = retriever.retrieve(q, top_k=top_k)
        except Exception as exc:
            accumulated_warnings.append(
                f"retriever_exception: query={q.query_id} err={exc!r}"
            )
            counters["unresolved"] += 1
            continue

        if result.warnings:
            accumulated_warnings.extend(result.warnings)

        if result.is_empty:
            counters["unresolved"] += 1
            continue

        # Pick the base hint ref that originated this query (same
        # standard_id/topic if any).
        base_ref = None
        for ref in hint_refs:
            if (getattr(ref, "topic", None) == q.topic
                    and getattr(ref, "limit_state", None) == q.limit_state):
                base_ref = ref
                break

        for chunk in result.chunks:
            counters["chunks"] += 1
            enriched = chunk_to_code_reference(chunk, base_ref=base_ref)

            ok, reason = validate_code_reference(
                enriched,
                chunk_id=chunk.chunk_id,
                matched_topic=chunk.topic,
                matched_limit_state=chunk.limit_state,
            )
            if ok:
                enriched.confidence = "medium"
                new_refs.append(enriched)
                counters["attached"] += 1
            else:
                accumulated_warnings.append(
                    f"ref_rejected: chunk={chunk.chunk_id} reason={reason}"
                )
                counters["rejected"] += 1

    # Append in place — the original hint refs stay first so the
    # downstream consumer still sees the deterministic hints.
    holder.code_refs = hint_refs + new_refs
    return counters


def enrich_code_refs_with_kds(
    issues: Iterable["StructuralIssue"],
    candidates: Iterable["RetrofitCandidate"],
    retriever: KDSRetriever,
    top_k: int = 3,
) -> dict[str, Any]:
    """Retrieve KDS chunks for every issue / candidate and append refs.

    The objects are mutated in place — pass copies if you need to keep
    the originals.
    """
    warnings: list[str] = []
    totals = {"queries": 0, "chunks": 0, "attached": 0,
              "unresolved": 0, "rejected": 0}
    issue_index: dict[str, "StructuralIssue"] = {}

    issue_list = list(issues)
    for issue in issue_list:
        issue_index[issue.issue_id] = issue
        q = build_kds_queries_for_issue(issue)
        c = _enrich_one(issue, retriever, top_k, q, warnings)
        for k, v in c.items():
            totals[k] += v

    for cand in candidates:
        parent = issue_index.get(cand.issue_id)
        q = build_kds_queries_for_candidate(cand, parent)
        c = _enrich_one(cand, retriever, top_k, q, warnings)
        for k, v in c.items():
            totals[k] += v

    has_unresolved = totals["unresolved"] > 0
    has_rejected = totals["rejected"] > 0

    return {
        "kds_rag_summary": {
            "enabled": True,
            "num_queries": totals["queries"],
            "num_chunks_retrieved": totals["chunks"],
            "num_refs_attached": totals["attached"],
            "num_unresolved": totals["unresolved"],
            "num_refs_rejected": totals["rejected"],
            # "Some usable citation is attached." Says nothing about
            # other queries that may have come back empty.
            "citation_ready": totals["attached"] > 0,
            # "Every query was answered AND every returned ref passed
            # the citation guardrail." Use this to decide whether to
            # surface a "근거 미확인" disclaimer to the user.
            "all_queries_resolved": not has_unresolved and not has_rejected,
            "has_unresolved_queries": has_unresolved,
            "has_rejected_refs": has_rejected,
            "backend": type(retriever).__name__,
            "top_k": top_k,
        },
        "kds_rag_warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Enrichment — payload level (the only adapter into recommendation output)
# ---------------------------------------------------------------------------

def _ref_dicts_from_obj(obj_refs: list) -> list[dict]:
    out: list[dict] = []
    for r in obj_refs or []:
        if hasattr(r, "to_dict"):
            out.append(r.to_dict())
        elif isinstance(r, dict):
            out.append(r)
    return out


def enrich_recommendation_payload_with_kds(
    payload: dict[str, Any],
    retriever: KDSRetriever,
    top_k: int = 3,
) -> dict[str, Any]:
    """Operate on a ``build_recommendation_payload`` output.

    The input ``payload`` must already contain ``issues`` and
    ``recommendation_candidates`` as JSON-shaped dicts. We rebuild the
    typed objects, run enrichment, and return a *new* payload dict —
    the input is not mutated.

    The new payload adds:

        kds_rag_summary  : counts + ``citation_ready``
        kds_rag_warnings : list[str]

    and re-serializes ``issues`` / ``recommendation_candidates`` with
    the enriched ``code_refs`` appended.
    """
    # Local import to keep core packages decoupled at import time.
    from ..recommendation.schemas import (
        CodeReference, RetrofitCandidate, StructuralIssue,
    )

    new_payload = copy.deepcopy(payload)

    issues = [
        _structural_issue_from_dict(d, CodeReference, StructuralIssue)
        for d in new_payload.get("issues", [])
    ]
    candidates = [
        _candidate_from_dict(d, CodeReference, RetrofitCandidate)
        for d in new_payload.get("recommendation_candidates", [])
    ]

    summary = enrich_code_refs_with_kds(issues, candidates, retriever, top_k=top_k)

    new_payload["issues"] = [i.to_dict() for i in issues]
    new_payload["recommendation_candidates"] = [c.to_dict() for c in candidates]
    new_payload["kds_rag_summary"] = summary["kds_rag_summary"]
    new_payload["kds_rag_warnings"] = summary["kds_rag_warnings"]
    return new_payload


# ---------------------------------------------------------------------------
# Local helpers — rebuild typed objects from the JSON payload
# ---------------------------------------------------------------------------

def _ref_from_dict(d: dict, CodeReference) -> Any:
    return CodeReference(
        standard_id=d.get("standard_id", ""),
        version=d.get("version"),
        clause_id=d.get("clause_id"),
        title=d.get("title"),
        source_url=d.get("source_url"),
        quote=d.get("quote"),
        relevance_reason=d.get("relevance_reason"),
        chunk_id=d.get("chunk_id"),
        query_hint=d.get("query_hint"),
        topic=d.get("topic"),
        material=d.get("material"),
        limit_state=d.get("limit_state"),
        jurisdiction=d.get("jurisdiction"),
        confidence=d.get("confidence"),
    )


def _structural_issue_from_dict(d: dict, CodeReference, StructuralIssue) -> Any:
    refs = [_ref_from_dict(r, CodeReference) for r in d.get("code_refs", [])]
    return StructuralIssue(
        issue_id=d["issue_id"],
        issue_type=d["issue_type"],
        severity=d["severity"],
        source=d["source"],
        description=d.get("description", ""),
        member_id=d.get("member_id"),
        element_id=d.get("element_id"),
        governing_combo=d.get("governing_combo"),
        demand_capacity_ratio=d.get("demand_capacity_ratio"),
        status=d.get("status"),
        member_type=d.get("member_type"),
        section=d.get("section"),
        material=d.get("material"),
        story=d.get("story"),
        level=d.get("level"),
        connected_node_ids=list(d.get("connected_node_ids") or []),
        grid_location=d.get("grid_location"),
        coordinates=d.get("coordinates"),
        evidence=dict(d.get("evidence") or {}),
        code_refs=refs,
    )


def _candidate_from_dict(d: dict, CodeReference, RetrofitCandidate) -> Any:
    refs = [_ref_from_dict(r, CodeReference) for r in d.get("code_refs", [])]
    return RetrofitCandidate(
        candidate_id=d["candidate_id"],
        issue_id=d["issue_id"],
        action_type=d["action_type"],
        description=d.get("description", ""),
        member_id=d.get("member_id"),
        element_id=d.get("element_id"),
        expected_effect=d.get("expected_effect", ""),
        tradeoffs=d.get("tradeoffs", ""),
        requires_reanalysis=bool(d.get("requires_reanalysis", True)),
        confidence=d.get("confidence", "low"),
        code_refs=refs,
        target=dict(d.get("target") or {}),
        proposed_change=dict(d.get("proposed_change") or {}),
        metadata=dict(d.get("metadata") or {}),
    )
