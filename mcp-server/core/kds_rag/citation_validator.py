"""Citation guardrail for :class:`CodeReference`.

A reference is *citation-ready* iff it carries enough information for a
reader to verify the source — not just an internal hint. The
recommendation layer's own deterministic refs are NOT citation-ready
because they only carry ``standard_id`` + ``clause_id`` + RAG hints,
no ``source_url`` or ``quote``. After KDS-RAG enriches them with a
``chunk_id`` (kept under ``metadata`` or ``source_url`` substitute) +
optional ``quote``, they become citation-ready.

Rules (deterministic — fixed here so LLM prompts can rely on them)
-----------------------------------------------------------------

ACCEPTED iff ALL of:
    1. ``standard_id`` is set.
    2. ``clause_id`` OR ``title`` is set.
    3. ``source_url`` OR ``chunk_id`` (stored on the ref via the
       ``chunk_id`` extension attribute attached during enrichment) is
       set. A bare in-repo hint is NOT enough.
    4. If ``quote`` is set, ``len(quote) <= MAX_QUOTE_LEN``.
    5. If ``topic`` and the matched chunk's ``topic`` are both set,
       they must agree. The caller must pass the matched chunk's topic
       via ``matched_topic`` / ``matched_limit_state`` so we can detect
       cross-topic drift.

REJECTED otherwise. Each rejection records the *first* failed rule via
``reason`` on the result.

Note: ``CodeReference`` has no built-in ``chunk_id`` field — the
pipeline stores it via a sidecar dict (``ref_metadata``) keyed by the
ref's identity. ``validate_code_reference`` accepts ``chunk_id`` as a
keyword arg so callers can wire either shape.
"""
from __future__ import annotations

from typing import Iterable, Optional, TYPE_CHECKING

from .schemas import CitationValidationResult

if TYPE_CHECKING:
    from ..recommendation.schemas import CodeReference


# Empirically chosen — long enough for a clause summary, short enough to
# discourage copy-pasting whole sections (which would risk copyright).
MAX_QUOTE_LEN = 400

# Minimum quote length when a quote is provided. Empty strings or single
# whitespace blobs should not satisfy "quote is set".
MIN_QUOTE_LEN = 8


def _has(value) -> bool:
    return value is not None and str(value).strip() != ""


def validate_code_reference(
    ref: "CodeReference",
    *,
    chunk_id: Optional[str] = None,
    matched_topic: Optional[str] = None,
    matched_limit_state: Optional[str] = None,
) -> tuple[bool, str]:
    """Validate a single CodeReference. Returns ``(accepted, reason)``.

    ``chunk_id`` may be passed explicitly by the caller (the enrichment
    path does this so it can validate before the chunk_id is stored on
    the ref). If the caller omits it, we fall back to ``ref.chunk_id``
    so post-hoc validators don't need to remember which chunk produced
    the citation.
    """
    if not _has(getattr(ref, "standard_id", None)):
        return False, "missing_standard_id"

    if not _has(getattr(ref, "clause_id", None)) and not _has(getattr(ref, "title", None)):
        return False, "missing_clause_id_and_title"

    has_url = _has(getattr(ref, "source_url", None))
    # Explicit chunk_id arg wins; otherwise fall back to ref.chunk_id.
    effective_chunk_id = chunk_id if _has(chunk_id) else getattr(ref, "chunk_id", None)
    has_chunk = _has(effective_chunk_id)
    if not has_url and not has_chunk:
        return False, "missing_source_url_and_chunk_id"

    quote = getattr(ref, "quote", None)
    if quote is not None:
        q = str(quote).strip()
        if 0 < len(q) < MIN_QUOTE_LEN:
            return False, "quote_too_short"
        if len(q) > MAX_QUOTE_LEN:
            return False, "quote_too_long"
        # Empty / whitespace-only quote → treat as missing, fall through.

    # Cross-topic drift: the deterministic hint topic should agree with
    # the matched chunk's topic. If both are set and disagree, reject.
    ref_topic = getattr(ref, "topic", None)
    if _has(ref_topic) and _has(matched_topic) and ref_topic != matched_topic:
        return False, f"topic_mismatch: hint={ref_topic!r} matched={matched_topic!r}"

    ref_ls = getattr(ref, "limit_state", None)
    if _has(ref_ls) and _has(matched_limit_state) and ref_ls != matched_limit_state:
        return False, (
            f"limit_state_mismatch: hint={ref_ls!r} "
            f"matched={matched_limit_state!r}"
        )

    return True, "ok"


def validate_code_references(
    refs: Iterable["CodeReference"],
    *,
    chunk_ids: Optional[dict[int, str]] = None,
    matched_topics: Optional[dict[int, str]] = None,
    matched_limit_states: Optional[dict[int, str]] = None,
) -> CitationValidationResult:
    """Validate a batch of refs. Index keys in the maps are positional.

    ``chunk_ids[i]`` etc. correspond to ``refs[i]``. Missing entries are
    treated as ``None``.
    """
    chunk_ids = chunk_ids or {}
    matched_topics = matched_topics or {}
    matched_limit_states = matched_limit_states or {}

    accepted: list = []
    rejected: list = []
    reasons: list[str] = []

    for i, ref in enumerate(list(refs)):
        ok, reason = validate_code_reference(
            ref,
            chunk_id=chunk_ids.get(i),
            matched_topic=matched_topics.get(i),
            matched_limit_state=matched_limit_states.get(i),
        )
        if ok:
            accepted.append(ref)
        else:
            rejected.append(ref)
            reasons.append(reason)

    valid = not rejected
    if valid:
        summary = f"{len(accepted)} ref(s) accepted"
    else:
        summary = (
            f"{len(rejected)} ref(s) rejected: " + "; ".join(reasons[:3])
            + ("…" if len(reasons) > 3 else "")
        )
    return CitationValidationResult(
        valid=valid,
        reason=summary,
        accepted_refs=accepted,
        rejected_refs=rejected,
    )
