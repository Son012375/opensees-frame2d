"""Voyage-backed KDS retriever skeleton.

This module is Phase 3B's *interface* layer. It contains:

    * VoyageEmbeddingClient — thin wrapper around voyageai SDK.
      Constructor raises a clear RuntimeError when the SDK or API key
      is missing; the CLI build path wants that failure mode.
    * VoyageReranker      — same shape for the reranker.
    * VoyageKDSRetriever  — KDSRetriever implementation that loads a
      JSONL index built by :mod:`core.kds_rag.ingest`, embeds the query
      via Voyage, ranks via cosine similarity, and (optionally) reranks
      via Voyage rerank-2.5.

The retriever is read-only at runtime. Index building lives in
``scripts/build_kds_rag_index.py``. Both code paths share the same
``VoyageEmbeddingClient`` constructor — the retriever simply never calls
``embed_documents``.

Safety invariants:

    * retrieve() NEVER raises. Any external API failure becomes
      ``KDSRetrievalResult(chunks=[], warnings=[...])``.
    * No clause numbers are invented. Only what the index contains is
      returned.
    * No KDS source text is embedded in this module — the index file is
      the only place chunks live.
"""
from __future__ import annotations

import json
import logging
import math
import os
from typing import Any, Optional

from .retriever import KDSRetriever
from .schemas import KDSChunk, KDSRetrievalQuery, KDSRetrievalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Voyage client wrappers
# ---------------------------------------------------------------------------

_VOYAGE_ENV_KEYS = ("VOYAGE_API_KEY", "VOYAGEAI_API_KEY")


def _resolve_api_key(api_key: Optional[str]) -> str:
    if api_key:
        return api_key
    for k in _VOYAGE_ENV_KEYS:
        v = os.environ.get(k)
        if v:
            return v
    raise RuntimeError(
        "VOYAGE_API_KEY (or VOYAGEAI_API_KEY) is not set. "
        "Provide one via --api-key or environment."
    )


def _import_voyageai():
    """Lazy import of the voyageai SDK. Returns module or raises clearly."""
    try:
        import voyageai  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "voyageai package is not installed. "
            "Run `pip install voyageai` to enable Voyage embeddings."
        ) from exc
    return voyageai


class VoyageEmbeddingClient:
    """Wrapper around voyageai.Client for document/query embeddings.

    The constructor resolves the API key eagerly and instantiates the
    underlying SDK client so failures surface at construction time
    (which the index-building CLI wants). The retriever path wraps
    construction in try/except, so endpoint behavior stays soft.
    """

    def __init__(self, api_key: Optional[str] = None,
                 model: str = "voyage-4-large"):
        self.api_key = _resolve_api_key(api_key)
        self.model = model
        voyageai = _import_voyageai()
        # voyageai.Client picks up api_key from env if we don't pass one;
        # we always pass explicitly so behavior is testable.
        self._client = voyageai.Client(api_key=self.api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        result = self._client.embed(
            texts=list(texts),
            model=self.model,
            input_type="document",
        )
        return list(result.embeddings)

    def embed_query(self, text: str) -> list[float]:
        result = self._client.embed(
            texts=[text],
            model=self.model,
            input_type="query",
        )
        return list(result.embeddings[0])


class VoyageReranker:
    """Wrapper around voyageai.Client.rerank for top-k reranking."""

    def __init__(self, api_key: Optional[str] = None,
                 model: str = "rerank-2.5"):
        self.api_key = _resolve_api_key(api_key)
        self.model = model
        voyageai = _import_voyageai()
        self._client = voyageai.Client(api_key=self.api_key)

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        if not documents:
            return []
        result = self._client.rerank(
            query=query,
            documents=list(documents),
            model=self.model,
            top_k=min(top_k, len(documents)),
        )
        # voyageai returns objects with .index and .relevance_score.
        out: list[tuple[int, float]] = []
        for item in (result.results or []):
            idx = int(getattr(item, "index", 0))
            score = float(getattr(item, "relevance_score", 0.0))
            out.append((idx, score))
        return out


# ---------------------------------------------------------------------------
# Cosine similarity (numpy-optional)
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _try_numpy_cosine(query_vec, embeddings):
    """Optional numpy fast path. Returns scores list or None on miss."""
    try:
        import numpy as np  # type: ignore
    except ImportError:
        return None
    if not embeddings:
        return []
    q = np.asarray(query_vec, dtype=np.float32)
    M = np.asarray(embeddings, dtype=np.float32)
    qn = np.linalg.norm(q)
    Mn = np.linalg.norm(M, axis=1)
    denom = Mn * qn
    denom[denom == 0] = 1.0
    scores = (M @ q) / denom
    return scores.tolist()


# ---------------------------------------------------------------------------
# Index format
# ---------------------------------------------------------------------------

# Index file is JSONL: one record per chunk.
#
# Required keys:
#     chunk_id, standard_id, text, embedding
# Optional keys (preserved if present):
#     clause_id, title, version, effective_date, source_url, page,
#     section_path, material, topic, limit_state, jurisdiction
#
# The build script in ``scripts/build_kds_rag_index.py`` writes this
# shape; the retriever validates it lazily (records missing required
# keys are skipped with a warning, never crashing the run).

INDEX_REQUIRED_KEYS = ("chunk_id", "standard_id", "text", "embedding")


def _load_index(path: str) -> tuple[list[KDSChunk], list[list[float]], list[str]]:
    """Read a JSONL index and return (chunks, embeddings, warnings)."""
    chunks: list[KDSChunk] = []
    embeddings: list[list[float]] = []
    warnings: list[str] = []
    line_no = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_no += 1
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.append(f"index_parse_error: line={line_no} err={exc!s}")
                continue
            missing = [k for k in INDEX_REQUIRED_KEYS if k not in rec]
            if missing:
                warnings.append(
                    f"index_record_missing_fields: line={line_no} "
                    f"missing={missing}"
                )
                continue
            chunk = KDSChunk(
                chunk_id=str(rec["chunk_id"]),
                standard_id=str(rec["standard_id"]),
                text=str(rec["text"]),
                version=rec.get("version"),
                effective_date=rec.get("effective_date"),
                clause_id=rec.get("clause_id"),
                title=rec.get("title"),
                source_url=rec.get("source_url"),
                page=rec.get("page"),
                section_path=list(rec.get("section_path") or []),
                material=rec.get("material"),
                topic=rec.get("topic"),
                limit_state=rec.get("limit_state"),
                jurisdiction=rec.get("jurisdiction", "KDS"),
            )
            chunks.append(chunk)
            embeddings.append([float(x) for x in rec["embedding"]])
    return chunks, embeddings, warnings


# ---------------------------------------------------------------------------
# VoyageKDSRetriever
# ---------------------------------------------------------------------------

class VoyageKDSRetriever(KDSRetriever):
    """KDS retriever backed by a Voyage embedding index.

    Construction loads the index file into memory once. ``retrieve``:

        1. Builds a query embedding via the supplied (or default) Voyage
           embedding client.
        2. Computes cosine similarity against every chunk's stored
           embedding.
        3. Takes top ``top_n_dense`` chunks by cosine score.
        4. If a reranker is configured, reorders those chunks via
           Voyage rerank-2.5 and truncates to ``top_k``. Without a
           reranker, simply truncates to ``top_k``.

    Any external call failure (network, auth, malformed response) is
    captured and returned as an empty result + a warning. The
    deterministic Noop fallback in /explain takes over from there.
    """

    def __init__(
        self,
        index_path: str,
        *,
        api_key: Optional[str] = None,
        embedding_model: str = "voyage-4-large",
        rerank_model: Optional[str] = "rerank-2.5",
        top_n_dense: int = 20,
        embedding_client: Optional[VoyageEmbeddingClient] = None,
        reranker: Optional[VoyageReranker] = None,
    ):
        self.index_path = index_path
        self.top_n_dense = max(1, int(top_n_dense))

        # Allow injection (used by tests). When neither is supplied we
        # build a real client — the constructor raises if Voyage SDK or
        # key is missing, which the factory wraps.
        if embedding_client is None:
            embedding_client = VoyageEmbeddingClient(
                api_key=api_key, model=embedding_model,
            )
        self._embedder = embedding_client

        self._reranker: Optional[VoyageReranker] = reranker
        if reranker is None and rerank_model:
            try:
                self._reranker = VoyageReranker(
                    api_key=api_key, model=rerank_model,
                )
            except Exception as exc:  # noqa: BLE001 — rerank is best-effort
                logger.warning(
                    "Voyage reranker init failed (%s); "
                    "retriever will run without reranking",
                    exc,
                )
                self._reranker = None

        self._chunks, self._embeddings, self._load_warnings = _load_index(
            index_path,
        )
        if not self._chunks:
            logger.warning(
                "VoyageKDSRetriever loaded zero chunks from %s",
                index_path,
            )

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    def retrieve(
        self,
        query: KDSRetrievalQuery,
        top_k: int = 5,
    ) -> KDSRetrievalResult:
        warnings = list(self._load_warnings)
        if not self._chunks:
            warnings.append(
                f"voyage_index_empty: index_path={self.index_path}"
            )
            return KDSRetrievalResult(query=query, chunks=[], warnings=warnings)
        if top_k <= 0:
            warnings.append(f"top_k_invalid: top_k={top_k!r}")
            return KDSRetrievalResult(query=query, chunks=[], warnings=warnings)

        # 1) Query embedding.
        try:
            q_vec = self._embedder.embed_query(query.query_text)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"voyage_embed_query_failed: {exc!r}")
            return KDSRetrievalResult(query=query, chunks=[], warnings=warnings)

        # 2) Cosine over the index (numpy fast path optional).
        scores = _try_numpy_cosine(q_vec, self._embeddings)
        if scores is None:
            scores = [_cosine(q_vec, emb) for emb in self._embeddings]

        # 3) Top-N dense.
        idx_score = sorted(
            range(len(scores)),
            key=lambda i: (-scores[i], i),
        )[: self.top_n_dense]

        # 4) Optional rerank.
        if self._reranker is not None and len(idx_score) > 1:
            doc_texts = [self._chunks[i].text for i in idx_score]
            try:
                rerank_pairs = self._reranker.rerank(
                    query.query_text, doc_texts, top_k=min(top_k, len(doc_texts)),
                )
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"voyage_rerank_failed: {exc!r}")
                rerank_pairs = []

            if rerank_pairs:
                # rerank_pairs are (local_index_within_doc_texts, score).
                picked_idx = [idx_score[local] for local, _ in rerank_pairs]
                rerank_scores = {
                    idx_score[local]: float(s) for local, s in rerank_pairs
                }
                final_indices = picked_idx[:top_k]
                out_chunks = [self._chunks[i] for i in final_indices]
                out_scores = {
                    self._chunks[i].chunk_id: rerank_scores.get(
                        i, float(scores[i])
                    )
                    for i in final_indices
                }
                return KDSRetrievalResult(
                    query=query,
                    chunks=out_chunks,
                    warnings=warnings,
                    scores=out_scores,
                )

        # No reranker (or it failed) — return top-k by cosine.
        final_indices = idx_score[:top_k]
        out_chunks = [self._chunks[i] for i in final_indices]
        out_scores = {
            self._chunks[i].chunk_id: float(scores[i]) for i in final_indices
        }
        if not out_chunks:
            warnings.append("voyage_no_match")
        return KDSRetrievalResult(
            query=query,
            chunks=out_chunks,
            warnings=warnings,
            scores=out_scores,
        )


__all__ = [
    "VoyageEmbeddingClient",
    "VoyageReranker",
    "VoyageKDSRetriever",
]
