"""Mock-driven tests for the Voyage-backed KDS RAG path.

Phase 3B lays down the *interface* + skeleton — no live API calls. These
tests lock down:

    - the chunker preserves clause metadata and respects max_chars
    - load_kds_documents skips unknown extensions (PDFs etc.)
    - make_kds_query produces useful search text even when the context
      has no code_refs (action_type / issue_type mappings carry it)
    - NoopKDSRetriever returns an empty result PLUS a warning
    - get_default_kds_retriever falls back to Noop when env vars miss
    - VoyageEmbeddingClient surfaces a clear error when the SDK or key
      is missing
    - VoyageKDSRetriever loads a JSONL index, runs cosine top-k, and
      respects reranker output without ever touching the network
    - reranker reordering is reflected in the result + scores dict
    - any retriever-side exception becomes a warning, not a raise

These tests must not require network access or installed voyageai SDK.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mcp-server"))

import core.kds_rag.voyage_retriever as voyage_module  # noqa: E402
from core.kds_rag import (  # noqa: E402
    KDSChunk,
    KDSRetrievalQuery,
    NOOP_WARNING,
    NoopKDSRetriever,
    get_default_kds_retriever,
    make_kds_query,
)
from core.kds_rag.factory import get_default_kds_retriever as factory_get  # noqa: E402
from core.kds_rag.ingest import (  # noqa: E402
    SUPPORTED_EXTENSIONS,
    build_kds_index,
    chunk_kds_document,
    load_kds_documents,
)
from core.kds_rag.voyage_retriever import (  # noqa: E402
    VoyageKDSRetriever,
)


# ---------------------------------------------------------------------------
# make_kds_query
# ---------------------------------------------------------------------------

class TestMakeKdsQuery:
    def test_drift_issue_maps_to_drift_keywords(self):
        ctx = {
            "candidate_id": "c1",
            "issue_id": "drift_exceeded_story_1",
            "issue_type": "drift_exceeded",
            "action_type": "add_lateral_resistance",
            "target": {"member_type": "column"},
            "proposed_change": {},
        }
        q = make_kds_query(ctx)
        assert isinstance(q, KDSRetrievalQuery)
        text = q.query_text
        # at least one drift-side keyword AND one action-side keyword
        assert any(k in text for k in ("층간변위", "drift"))
        assert any(k in text for k in ("가새", "전단벽", "횡저항 시스템"))
        assert q.limit_state == "drift_limit"
        assert q.topic == "story_drift"

    def test_works_without_code_refs(self):
        # No code_refs in the context — the deterministic mapping should
        # still produce a non-empty, action-aware query.
        ctx = {
            "candidate_id": "c2",
            "issue_id": "strength_exceeded_member_1",
            "issue_type": "strength_exceeded",
            "action_type": "replace_section",
            "target": {},
            "proposed_change": {},
        }
        q = make_kds_query(ctx)
        assert q.query_text
        assert "단면" in q.query_text or "내력" in q.query_text or "강도" in q.query_text

    def test_action_type_keyword_mapping(self):
        ctx = {
            "candidate_id": "c3",
            "issue_id": "x",
            "issue_type": None,
            "action_type": "add_lateral_resistance",
            "target": {},
            "proposed_change": {},
        }
        q = make_kds_query(ctx)
        assert "가새" in q.query_text or "전단벽" in q.query_text \
            or "횡저항 시스템" in q.query_text

    def test_empty_context_still_returns_a_query(self):
        q = make_kds_query({"candidate_id": "c", "issue_id": "i"})
        # No keywords resolved → text falls back to action_type or
        # "kds search"; either way it must be non-empty so a retriever
        # can run.
        assert isinstance(q, KDSRetrievalQuery)
        assert q.query_text

    def test_string_section_from_to_makes_it_into_query(self):
        """Phase 2 candidates carry proposed_change.from/to as plain
        section-name strings (e.g. "H-300x150"). The query builder must
        include those so dense retrieval has the actual member size as a
        search signal — otherwise replace_section queries lose their
        strongest discriminator."""
        ctx = {
            "candidate_id": "c4",
            "issue_id": "iss_strength_exceeded_x",
            "issue_type": "strength_exceeded",
            "action_type": "replace_section",
            "target": {"member_type": "column"},
            "proposed_change": {
                "operation": "replace_section",
                "from": "H-300x150",
                "to": "H-300x300",
                "applicable": True,
            },
        }
        q = make_kds_query(ctx)
        assert "H-300x150" in q.query_text
        assert "H-300x300" in q.query_text

    def test_dict_section_shape_still_supported(self):
        """Legacy/structured candidate shape uses dict for from/to."""
        ctx = {
            "candidate_id": "c5",
            "issue_id": "iss_strength_exceeded_x",
            "action_type": "replace_section",
            "target": {},
            "proposed_change": {
                "from": {"section": "H-400x200", "material": "SS275"},
                "to": {"section_id": "H-400x400"},
            },
        }
        q = make_kds_query(ctx)
        assert "H-400x200" in q.query_text
        assert "H-400x400" in q.query_text

    def test_none_section_does_not_crash(self):
        """Abstract candidates may carry from=None/to=None."""
        ctx = {
            "candidate_id": "c6",
            "issue_id": "iss_drift_exceeded_x",
            "action_type": "add_lateral_resistance",
            "target": {},
            "proposed_change": {
                "from": None,
                "to": None,
                "applicable": False,
            },
        }
        q = make_kds_query(ctx)
        # No section keywords, but the action mapping still fills the
        # query text — so this must not raise and must yield non-empty.
        assert isinstance(q, KDSRetrievalQuery)
        assert q.query_text


# ---------------------------------------------------------------------------
# Noop retriever + factory
# ---------------------------------------------------------------------------

class TestNoopRetriever:
    def test_returns_empty_and_warning(self):
        retriever = NoopKDSRetriever()
        q = KDSRetrievalQuery(query_id="t", query_text="anything")
        result = retriever.retrieve(q, top_k=5)
        assert result.chunks == []
        assert any(NOOP_WARNING in w for w in result.warnings)
        assert result.scores == {}


class TestFactory:
    def test_returns_noop_without_env(self, monkeypatch):
        # Clear every Voyage env var we know about.
        for key in ("VOYAGE_API_KEY", "VOYAGEAI_API_KEY",
                    "KDS_RAG_INDEX_PATH"):
            monkeypatch.delenv(key, raising=False)
        retriever = get_default_kds_retriever()
        assert isinstance(retriever, NoopKDSRetriever)

    def test_returns_noop_when_index_path_missing_on_disk(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VOYAGE_API_KEY", "fake")
        monkeypatch.setenv("KDS_RAG_INDEX_PATH", str(tmp_path / "no_such.jsonl"))
        retriever = get_default_kds_retriever()
        assert isinstance(retriever, NoopKDSRetriever)

    def test_swallows_voyage_init_failure(self, monkeypatch, tmp_path):
        # Force the factory's Voyage path: env present + file exists.
        idx_path = tmp_path / "stub.jsonl"
        idx_path.write_text("", encoding="utf-8")
        monkeypatch.setenv("VOYAGE_API_KEY", "fake")
        monkeypatch.setenv("KDS_RAG_INDEX_PATH", str(idx_path))

        # Patch VoyageKDSRetriever.__init__ to raise so the factory's
        # try/except is exercised.
        class _Boom:
            def __init__(self, *a, **k):
                raise RuntimeError("simulated init failure")

        import core.kds_rag.voyage_retriever as vmod
        monkeypatch.setattr(vmod, "VoyageKDSRetriever", _Boom)
        # Re-import factory so it picks up the patched module via its
        # own lazy import.
        retriever = factory_get()
        assert isinstance(retriever, NoopKDSRetriever)


# ---------------------------------------------------------------------------
# Ingest: loading + chunking
# ---------------------------------------------------------------------------

class TestIngest:
    def test_load_skips_unknown_extensions(self, tmp_path):
        (tmp_path / "ok.txt").write_text(
            "허용 층간변위비는 내진등급에 따라 다르다.",
            encoding="utf-8",
        )
        (tmp_path / "ignore.pdf").write_bytes(b"%PDF-1.4 ...")
        docs = load_kds_documents(str(tmp_path))
        assert len(docs) == 1
        assert docs[0]["doc_id"].startswith("ok")
        # SUPPORTED_EXTENSIONS does NOT include .pdf — assert it
        # to catch a future scope drift.
        assert ".pdf" not in SUPPORTED_EXTENSIONS

    def test_load_reads_json_doc_shape(self, tmp_path):
        payload = {
            "doc_id": "kds_41_17_00_drift",
            "title": "허용 층간변위비",
            "clause": "§8.2.3",
            "standard_id": "KDS 41 17 00",
            "topic": "story_drift",
            "limit_state": "drift_limit",
            "text": (
                "허용 층간변위비는 내진등급에 따라 0.010 hsx (특), "
                "0.015 hsx (I), 0.020 hsx (II)로 한다."
            ),
        }
        (tmp_path / "drift.json").write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        docs = load_kds_documents(str(tmp_path))
        assert len(docs) == 1
        assert docs[0]["standard_id"] == "KDS 41 17 00"
        assert docs[0]["topic"] == "story_drift"

    def test_chunk_preserves_metadata_and_respects_max(self):
        # Multiple paragraphs (\n\n separated) so the chunker can split.
        paragraph = "허용 층간변위비는 내진등급에 따라 다르게 정의된다. " * 8
        doc = {
            "doc_id": "kds_drift",
            "title": "허용 층간변위비",
            "clause": "§8.2.3",
            "standard_id": "KDS 41 17 00",
            "topic": "story_drift",
            "limit_state": "drift_limit",
            "text": "\n\n".join([paragraph] * 6),
        }
        chunks = chunk_kds_document(doc, max_chars=500, overlap=50)
        assert len(chunks) >= 2  # produced multiple chunks
        for ch in chunks:
            assert ch["doc_id"] == "kds_drift"
            assert ch["standard_id"] == "KDS 41 17 00"
            assert ch["topic"] == "story_drift"
            assert ch["limit_state"] == "drift_limit"
            # Body fits roughly the cap (a single oversize paragraph
            # is preserved intact, but multi-paragraph chunks honor it).
            assert len(ch["text"]) <= max(500 + 50, len(paragraph) + 50)

    def test_chunk_picks_up_clause_heading_from_body(self):
        doc = {
            "doc_id": "kds_strength",
            "title": "부재 강도",
            "clause": None,
            "text": (
                "8.1\n\n"
                "부재의 조합응력 비는 1.0을 초과할 수 없다.\n\n"
                "8.2.3\n\n"
                "허용 층간변위비는 내진등급에 따라 다르다."
            ),
        }
        chunks = chunk_kds_document(doc, max_chars=200)
        clauses = [c["clause"] for c in chunks]
        # Both detected clause headings (8.1 → 8.2.3) propagate into
        # subsequent chunks' metadata.
        assert "8.1" in clauses or "8.2.3" in clauses


# ---------------------------------------------------------------------------
# build_kds_index with a fake embedding client
# ---------------------------------------------------------------------------

class _FakeEmbeddingClient:
    """Deterministic embed-by-hash client — no network."""

    def __init__(self, model: str = "fake-voyage"):
        self.model = model
        self.embed_doc_calls = 0
        self.embed_query_calls = 0

    @staticmethod
    def _vec(text: str) -> list[float]:
        # 4-dim "embedding" derived from char-class counts. Deterministic
        # and roughly aligns near-duplicate texts.
        v = [0.0, 0.0, 0.0, 0.0]
        for ch in text:
            if "가" <= ch <= "힣":
                v[0] += 1.0
            elif ch.isalpha():
                v[1] += 1.0
            elif ch.isdigit():
                v[2] += 1.0
            else:
                v[3] += 1.0
        return v

    def embed_documents(self, texts):
        self.embed_doc_calls += 1
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        self.embed_query_calls += 1
        return self._vec(text)


class TestBuildIndex:
    def test_build_index_writes_jsonl_with_embeddings(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "drift.txt").write_text(
            "허용 층간변위비는 내진등급에 따라 다르다.",
            encoding="utf-8",
        )
        (src / "strength.md").write_text(
            "부재의 조합응력 비는 1.0을 초과할 수 없다.",
            encoding="utf-8",
        )
        out = tmp_path / "idx.jsonl"
        client = _FakeEmbeddingClient()
        summary = build_kds_index(
            source_dir=str(src),
            index_path=str(out),
            embedding_client=client,
        )
        assert summary["n_docs"] == 2
        assert summary["n_chunks"] >= 2
        assert summary["dim"] == 4
        assert client.embed_doc_calls >= 1

        # JSONL is parseable and each record has required keys.
        with out.open(encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        assert len(records) == summary["n_chunks"]
        for r in records:
            for k in ("chunk_id", "standard_id", "text", "embedding"):
                assert k in r
            assert len(r["embedding"]) == 4

    def test_build_index_raises_when_source_is_empty(self, tmp_path):
        src = tmp_path / "empty"
        src.mkdir()
        with pytest.raises(RuntimeError, match="No supported"):
            build_kds_index(
                source_dir=str(src),
                index_path=str(tmp_path / "out.jsonl"),
                embedding_client=_FakeEmbeddingClient(),
            )


# ---------------------------------------------------------------------------
# VoyageEmbeddingClient: clear error when SDK / key missing
# ---------------------------------------------------------------------------

class TestVoyageClientErrors:
    def test_missing_api_key_raises_clear_runtime_error(self, monkeypatch):
        for k in ("VOYAGE_API_KEY", "VOYAGEAI_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        with pytest.raises(RuntimeError, match="VOYAGE_API_KEY"):
            voyage_module.VoyageEmbeddingClient(api_key=None)

    def test_missing_sdk_raises_clear_runtime_error(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "fake")

        def _fail_import():
            raise RuntimeError(
                "voyageai package is not installed. "
                "Run `pip install voyageai` to enable Voyage embeddings."
            )
        monkeypatch.setattr(voyage_module, "_import_voyageai", _fail_import)
        with pytest.raises(RuntimeError, match="voyageai package is not installed"):
            voyage_module.VoyageEmbeddingClient(api_key="fake")


# ---------------------------------------------------------------------------
# VoyageKDSRetriever with mocked embedder / reranker
# ---------------------------------------------------------------------------

def _write_index(path: Path, chunks: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False))
            f.write("\n")


def _record(chunk_id: str, text: str, vec: list[float], **extra) -> dict:
    base = {
        "chunk_id": chunk_id,
        "standard_id": "KDS 41 17 00",
        "title": "허용 층간변위비",
        "clause_id": "§8.2.3",
        "topic": "story_drift",
        "limit_state": "drift_limit",
        "jurisdiction": "KDS",
        "text": text,
        "embedding": vec,
    }
    base.update(extra)
    return base


class _StubReranker:
    """Reranker mock that reverses the cosine top order."""

    def __init__(self, reorder: list[int] = None):
        # If reorder is set, returns those local indices in order.
        # Else, reverses the input.
        self.reorder = reorder
        self.calls = 0

    def rerank(self, query, documents, *, top_k=5):
        self.calls += 1
        if self.reorder is not None:
            seq = self.reorder
        else:
            seq = list(reversed(range(len(documents))))
        out = []
        for rank, local_idx in enumerate(seq[:top_k]):
            out.append((local_idx, 1.0 - rank * 0.1))
        return out


class TestVoyageKDSRetriever:
    def test_cosine_top_k_with_mocked_embeddings(self, tmp_path):
        idx = tmp_path / "idx.jsonl"
        _write_index(idx, [
            _record("c0", "drift 층간변위 한도", [1.0, 0.0, 0.0, 0.0]),
            _record("c1", "strength 강도 조합응력", [0.0, 1.0, 0.0, 0.0]),
            _record("c2", "drift 변위 한도", [0.9, 0.1, 0.0, 0.0]),
        ])
        client = _FakeEmbeddingClient()

        # Embedder is injected so no network is touched.
        retriever = VoyageKDSRetriever(
            index_path=str(idx),
            embedding_client=client,
            reranker=None,
            rerank_model=None,
        )
        # Force the embedder's query vector to point at c0/c2 cluster.
        client.embed_query = lambda text: [1.0, 0.0, 0.0, 0.0]  # type: ignore

        q = KDSRetrievalQuery(query_id="t", query_text="층간변위")
        result = retriever.retrieve(q, top_k=2)
        ids = [c.chunk_id for c in result.chunks]
        assert ids[0] == "c0"
        assert "c2" in ids
        assert "c1" not in ids
        # Score dict keyed by chunk_id, every selected chunk has an entry.
        for cid in ids:
            assert cid in result.scores
            assert result.scores[cid] > 0.0

    def test_reranker_reorders_results(self, tmp_path):
        idx = tmp_path / "idx.jsonl"
        _write_index(idx, [
            _record("c0", "first", [1.0, 0.0, 0.0, 0.0]),
            _record("c1", "second", [0.9, 0.1, 0.0, 0.0]),
            _record("c2", "third", [0.8, 0.2, 0.0, 0.0]),
        ])
        client = _FakeEmbeddingClient()
        client.embed_query = lambda text: [1.0, 0.0, 0.0, 0.0]  # type: ignore
        reranker = _StubReranker()  # reverses

        retriever = VoyageKDSRetriever(
            index_path=str(idx),
            embedding_client=client,
            reranker=reranker,
            rerank_model="rerank-mock",
        )
        q = KDSRetrievalQuery(query_id="t", query_text="x")
        result = retriever.retrieve(q, top_k=3)
        # Reranker reversed [c0, c1, c2] → [c2, c1, c0].
        assert [c.chunk_id for c in result.chunks] == ["c2", "c1", "c0"]
        assert reranker.calls == 1
        # Reranker scores reflect in result.scores.
        assert result.scores["c2"] > result.scores["c0"]

    def test_retriever_graceful_on_embed_failure(self, tmp_path):
        idx = tmp_path / "idx.jsonl"
        _write_index(idx, [
            _record("c0", "any", [1.0, 0.0, 0.0, 0.0]),
        ])

        class _BoomClient:
            def embed_documents(self, texts):
                return [[0.0] * 4 for _ in texts]

            def embed_query(self, text):
                raise RuntimeError("network down")

        retriever = VoyageKDSRetriever(
            index_path=str(idx),
            embedding_client=_BoomClient(),
            reranker=None,
            rerank_model=None,
        )
        q = KDSRetrievalQuery(query_id="t", query_text="x")
        result = retriever.retrieve(q, top_k=3)
        assert result.chunks == []
        assert any("voyage_embed_query_failed" in w for w in result.warnings)

    def test_empty_index_returns_warning_not_exception(self, tmp_path):
        idx = tmp_path / "empty.jsonl"
        idx.write_text("", encoding="utf-8")
        client = _FakeEmbeddingClient()
        retriever = VoyageKDSRetriever(
            index_path=str(idx),
            embedding_client=client,
            reranker=None,
            rerank_model=None,
        )
        q = KDSRetrievalQuery(query_id="t", query_text="x")
        result = retriever.retrieve(q, top_k=2)
        assert result.chunks == []
        assert any("voyage_index_empty" in w for w in result.warnings)

    def test_rerank_failure_falls_back_to_cosine(self, tmp_path):
        idx = tmp_path / "idx.jsonl"
        _write_index(idx, [
            _record("c0", "first", [1.0, 0.0, 0.0, 0.0]),
            _record("c1", "second", [0.9, 0.1, 0.0, 0.0]),
        ])
        client = _FakeEmbeddingClient()
        client.embed_query = lambda text: [1.0, 0.0, 0.0, 0.0]  # type: ignore

        class _FailReranker:
            def rerank(self, *a, **k):
                raise RuntimeError("rerank failed")

        retriever = VoyageKDSRetriever(
            index_path=str(idx),
            embedding_client=client,
            reranker=_FailReranker(),
            rerank_model="rerank-mock",
        )
        q = KDSRetrievalQuery(query_id="t", query_text="x")
        result = retriever.retrieve(q, top_k=2)
        # Falls back to cosine ranking — c0 first since it best matches.
        assert [c.chunk_id for c in result.chunks] == ["c0", "c1"]
        assert any("voyage_rerank_failed" in w for w in result.warnings)
