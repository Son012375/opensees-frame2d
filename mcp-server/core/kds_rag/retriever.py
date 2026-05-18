"""Retriever interface + in-memory test double.

The real RAG implementation (vector DB, embedding model, hybrid
sparse/dense scoring, …) lives outside this repo. What we lock down
*here* is the call shape so that downstream code in
``core.recommendation`` and ``core.kds_rag.pipeline`` can be tested
and wired without picking a backend yet.

Interface contract
------------------
``KDSRetriever`` is a protocol-style ABC with one method:

    retrieve(query: KDSRetrievalQuery, top_k: int = 5) -> KDSRetrievalResult

Implementations MUST:
    * never raise on "no match" — return ``chunks=[]`` + a warning.
    * never return more than ``top_k`` chunks.
    * preserve the input ``query`` on the result.

Implementations MAY:
    * use any scoring strategy.
    * add their own warnings (e.g. "vector backend timed out").

The :class:`InMemoryKDSRetriever` is the reference test double. It uses
simple field-equality + keyword overlap and is intentionally cheap so
unit tests can spin up dozens of fixtures.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional

from .schemas import KDSChunk, KDSRetrievalQuery, KDSRetrievalResult


class KDSRetriever(ABC):
    """Abstract retriever — the only contract real RAG backends must honor."""

    @abstractmethod
    def retrieve(
        self,
        query: KDSRetrievalQuery,
        top_k: int = 5,
    ) -> KDSRetrievalResult:
        """Return up to ``top_k`` chunks ranked by relevance to ``query``."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# In-memory test double
# ---------------------------------------------------------------------------

def _tokenize(text: Optional[str]) -> set[str]:
    if not text:
        return set()
    # Cheap lowercase ascii tokenizer. Sufficient for short synthetic
    # fixtures; real RAG will use a proper tokenizer + embeddings.
    out: set[str] = set()
    buf: list[str] = []
    for ch in str(text).lower():
        if ch.isalnum():
            buf.append(ch)
        else:
            if buf:
                out.add("".join(buf))
                buf.clear()
    if buf:
        out.add("".join(buf))
    return {t for t in out if len(t) > 1}


def _score_chunk(query: KDSRetrievalQuery, chunk: KDSChunk) -> float:
    """Bounded, deterministic, simple — same input → same score every time.

    Component weights are arbitrary but ordered:
        topic / limit_state exact match → strong (3.0 each)
        material match                   → medium (1.5)
        standard_id filter               → 2.0
        keyword overlap (query_text vs   → 0..1 normalized
        chunk.text + title + clause)
    """
    score = 0.0

    if query.topic and chunk.topic and query.topic == chunk.topic:
        score += 3.0
    if query.limit_state and chunk.limit_state and query.limit_state == chunk.limit_state:
        score += 3.0
    if query.material and chunk.material and query.material == chunk.material:
        score += 1.5
    if query.standard_id and chunk.standard_id and query.standard_id == chunk.standard_id:
        score += 2.0

    q_tokens = _tokenize(query.query_text)
    if q_tokens:
        haystack = " ".join(filter(None, [
            chunk.text, chunk.title, chunk.clause_id,
            chunk.topic, chunk.limit_state, chunk.material,
        ]))
        c_tokens = _tokenize(haystack)
        if c_tokens:
            overlap = len(q_tokens & c_tokens)
            score += min(1.0, overlap / max(1, len(q_tokens)))

    return score


class InMemoryKDSRetriever(KDSRetriever):
    """Deterministic in-memory retriever for tests and local development.

    Construct with a list of :class:`KDSChunk`. ``retrieve`` performs
    field-equality + keyword-overlap scoring; ties are broken by
    insertion order so results stay stable across runs.

    NOT a production retriever — there is no embedding step.
    """

    def __init__(self, chunks: Optional[Iterable[KDSChunk]] = None):
        self._chunks: list[KDSChunk] = list(chunks or [])

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    def add(self, chunk: KDSChunk) -> None:
        self._chunks.append(chunk)

    def retrieve(
        self,
        query: KDSRetrievalQuery,
        top_k: int = 5,
    ) -> KDSRetrievalResult:
        if not self._chunks:
            return KDSRetrievalResult(
                query=query,
                chunks=[],
                warnings=["retriever_empty: no chunks indexed"],
            )

        if top_k <= 0:
            return KDSRetrievalResult(
                query=query,
                chunks=[],
                warnings=[f"top_k_invalid: top_k={top_k!r} <= 0"],
            )

        scored: list[tuple[float, int, KDSChunk]] = []
        for idx, chunk in enumerate(self._chunks):
            s = _score_chunk(query, chunk)
            if s > 0.0:
                scored.append((s, idx, chunk))

        # Sort by score desc, then by insertion order (stable tie-break).
        scored.sort(key=lambda t: (-t[0], t[1]))
        picked = [c for _, _, c in scored[:top_k]]
        scores = {c.chunk_id: round(float(s), 6) for s, _, c in scored[:top_k]}

        warnings: list[str] = []
        if not picked:
            warnings.append(
                "no_match: in-memory retriever found no chunks matching "
                f"topic={query.topic!r} material={query.material!r} "
                f"limit_state={query.limit_state!r} "
                f"standard_id={query.standard_id!r}"
            )
        elif len(picked) < min(top_k, len(self._chunks)) and len(picked) < 2:
            # Soft warning: very weak result set.
            warnings.append(
                f"weak_match: only {len(picked)} chunk(s) matched query "
                f"{query.query_id!r}"
            )

        return KDSRetrievalResult(
            query=query,
            chunks=picked,
            warnings=warnings,
            scores=scores,
        )


# ---------------------------------------------------------------------------
# Noop retriever — the explicit "no KDS index configured" backend
# ---------------------------------------------------------------------------

NOOP_WARNING = (
    "kds_rag_unavailable: KDS 검색 인덱스가 설정되지 않았습니다. "
    "VOYAGE_API_KEY와 KDS_RAG_INDEX_PATH 환경변수를 설정하면 활성화됩니다."
)


class NoopKDSRetriever(KDSRetriever):
    """The honest-fallback retriever for environments without a KDS index.

    Always returns an empty result PLUS an explicit warning so the
    /explain endpoint can surface "deterministic-only, no KDS evidence"
    without inventing data. Callers must never see a silent empty:
    if there is no index, say so.
    """

    def retrieve(
        self,
        query: KDSRetrievalQuery,
        top_k: int = 5,
    ) -> KDSRetrievalResult:
        return KDSRetrievalResult(
            query=query,
            chunks=[],
            warnings=[NOOP_WARNING],
            scores={},
        )
