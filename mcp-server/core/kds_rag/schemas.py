"""Data contracts for KDS-RAG retrieval / citation.

Design notes
------------
* Pure ``@dataclass`` — same rationale as ``core.recommendation.schemas``.
* These dataclasses describe what a future RAG implementation must
  ingest, return, and accept. The interface is fixed *before* any
  vector-DB or embedding-model decision is made.
* ``KDSChunk`` is the only place ingested text lives. ``CodeReference``
  in ``core.recommendation`` is the *citation* form; a ``KDSChunk`` is
  the *source* form. The pipeline maps chunk → reference.
* Do NOT embed real KDS source text in this repo. Test doubles and
  fixtures use short synthetic snippets only.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # Forward declaration to avoid a circular import at runtime —
    # CitationValidationResult only references CodeReference for typing.
    from ..recommendation.schemas import CodeReference  # noqa: F401


# ---------------------------------------------------------------------------
# Source-side: KDSChunk
# ---------------------------------------------------------------------------

@dataclass
class KDSChunk:
    """A retrievable unit of KDS (or analog) source text.

    Field semantics
    ----------------
    chunk_id      Stable id assigned at ingestion time. ``"kds_41_31_00_h1_p72_c0"``
                  style is fine — RAG implementations may also hash text.
    standard_id   "KDS 41 31 00" etc. (with spaces, matching upstream usage)
    version       Free-form version label, e.g. ``"2022-10-11"``.
    effective_date ISO date string, e.g. ``"2022-10-11"``. Optional.
    clause_id     Clause / section number ("§8.2.3", "H1", "G2", …).
    title         Short human label ("허용 층간변위비", "조합응력 상관식").
    text          The chunk text itself. Keep SHORT in tests/fixtures —
                  this repo does not bundle full KDS source.
    source_url    Resolvable URL, if known. ``None`` is acceptable.
    page          Page number in the original PDF.
    section_path  Breadcrumb, e.g. ``["KDS 41 31 00", "8.", "8.2", "8.2.3"]``.
    material      "steel" / "rc" / "concrete" / … — same vocabulary as
                  ``CodeReference.material``.
    topic         Same vocabulary as ``CodeReference.topic``.
    limit_state   Same vocabulary as ``CodeReference.limit_state``.
    jurisdiction  Defaults to "KDS".
    """
    chunk_id: str
    standard_id: str
    text: str
    version: Optional[str] = None
    effective_date: Optional[str] = None
    clause_id: Optional[str] = None
    title: Optional[str] = None
    source_url: Optional[str] = None
    page: Optional[int] = None
    section_path: list[str] = field(default_factory=list)
    material: Optional[str] = None
    topic: Optional[str] = None
    limit_state: Optional[str] = None
    jurisdiction: str = "KDS"

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


# ---------------------------------------------------------------------------
# Query-side
# ---------------------------------------------------------------------------

@dataclass
class KDSRetrievalQuery:
    """A single retrieval request.

    Built deterministically from a :class:`CodeReference`'s search hints
    (``query_hint``, ``topic``, ``material``, ``limit_state``,
    ``standard_id``) — see :mod:`core.kds_rag.pipeline`.
    """
    query_id: str
    query_text: str
    topic: Optional[str] = None
    material: Optional[str] = None
    limit_state: Optional[str] = None
    standard_id: Optional[str] = None
    issue_id: Optional[str] = None
    candidate_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class KDSRetrievalResult:
    """Output of a single retrieve() call.

    A failing retrieval is represented by ``chunks=[]`` + a ``warnings``
    entry — never by raising. Callers can therefore treat retrieval as
    best-effort and continue.
    """
    query: KDSRetrievalQuery
    chunks: list[KDSChunk] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query.to_dict(),
            "chunks": [c.to_dict() for c in self.chunks],
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Citation validation
# ---------------------------------------------------------------------------

@dataclass
class CitationValidationResult:
    """Verdict of the citation guardrail.

    ``valid`` is True iff *every* checked ref passed the guardrail.
    Individual passes / failures are surfaced via ``accepted_refs`` /
    ``rejected_refs`` so callers can split them. ``reason`` is a short
    human-readable summary suitable for logging.
    """
    valid: bool
    reason: str
    accepted_refs: list[Any] = field(default_factory=list)   # CodeReference
    rejected_refs: list[Any] = field(default_factory=list)   # CodeReference

    def to_dict(self) -> dict[str, Any]:
        def _ref(r: Any) -> Any:
            if hasattr(r, "to_dict"):
                return r.to_dict()
            return r
        return {
            "valid": self.valid,
            "reason": self.reason,
            "accepted_refs": [_ref(r) for r in self.accepted_refs],
            "rejected_refs": [_ref(r) for r in self.rejected_refs],
        }
