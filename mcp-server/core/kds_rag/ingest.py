"""KDS document ingestion + chunking + index build.

Phase 3B scope: text-only ingestion (.txt, .md, .json). PDF parsing is
deferred to Phase 4 unless the operator already has a text-extracted
version of the standard.

The output index is a JSONL file readable by
:class:`core.kds_rag.voyage_retriever.VoyageKDSRetriever`. Each record
carries enough metadata to round-trip into a ``KDSChunk`` and through
the citation validator.

Build flow (used by ``scripts/build_kds_rag_index.py``):

    load_kds_documents(dir) -> [doc]
        -> chunk_kds_document(doc) -> [chunk]
            -> VoyageEmbeddingClient.embed_documents(batch)
                -> JSONL line {chunk_id, embedding, ...}

This module never calls a network API directly — it depends on
``VoyageEmbeddingClient`` (which itself raises a clear RuntimeError if
the SDK or API key is missing). All chunking is pure-Python and
deterministic so a future migration to a different embedding backend
keeps the same chunk_ids.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = (".txt", ".md", ".json")


def _short_hash(s: str, n: int = 10) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def _doc_id_from_path(path: Path) -> str:
    """Stable doc_id from the filename stem + a short content hash."""
    stem = path.stem
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    return safe or f"doc_{_short_hash(str(path))}"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _doc_from_json(path: Path) -> Optional[dict[str, Any]]:
    """JSON loader.

    Accepted shapes:
        * {"doc_id": ..., "title": ..., "clause": ..., "text": ...}
        * {"chunks": [{...}, ...]} — passthrough (caller chunks again)

    Returns a single doc dict; if the JSON is a list of chunk dicts,
    we synthesize a wrapper doc whose ``text`` is the concatenation
    (with clause headers) so chunking still applies.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("kds_ingest: failed to parse %s: %s", path, exc)
        return None

    if isinstance(data, dict) and "text" in data:
        return {
            "doc_id": str(data.get("doc_id") or _doc_id_from_path(path)),
            "title": data.get("title"),
            "clause": data.get("clause"),
            "text": str(data.get("text") or ""),
            "source_path": str(path),
            "standard_id": data.get("standard_id"),
            "version": data.get("version"),
            "effective_date": data.get("effective_date"),
            "source_url": data.get("source_url"),
            "material": data.get("material"),
            "topic": data.get("topic"),
            "limit_state": data.get("limit_state"),
            "jurisdiction": data.get("jurisdiction"),
        }
    if isinstance(data, dict) and "chunks" in data and isinstance(
        data["chunks"], list
    ):
        parts: list[str] = []
        for c in data["chunks"]:
            if not isinstance(c, dict):
                continue
            head = c.get("clause") or c.get("title") or ""
            txt = c.get("text") or ""
            if head:
                parts.append(f"[{head}] {txt}")
            else:
                parts.append(str(txt))
        return {
            "doc_id": str(data.get("doc_id") or _doc_id_from_path(path)),
            "title": data.get("title"),
            "clause": data.get("clause"),
            "text": "\n\n".join(parts),
            "source_path": str(path),
            "standard_id": data.get("standard_id"),
        }
    if isinstance(data, list):
        return {
            "doc_id": _doc_id_from_path(path),
            "title": None,
            "clause": None,
            "text": "\n\n".join(json.dumps(item, ensure_ascii=False)
                                for item in data),
            "source_path": str(path),
        }
    return None


def load_kds_documents(source_dir: str) -> list[dict[str, Any]]:
    """Recursively load supported files under ``source_dir``.

    Each returned dict carries at least:
        doc_id, title, clause, text, source_path
    plus any optional metadata the file itself exposed (json shape).
    PDF files and other extensions are silently skipped — the caller's
    logger surfaces what was indexed.
    """
    root = Path(source_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(
            f"KDS source directory does not exist: {source_dir}"
        )

    docs: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        if ext == ".json":
            doc = _doc_from_json(path)
            if doc is None:
                continue
            docs.append(doc)
            continue
        # .txt / .md
        try:
            text = _read_text(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("kds_ingest: failed to read %s: %s", path, exc)
            continue
        docs.append({
            "doc_id": _doc_id_from_path(path),
            "title": path.stem,
            "clause": None,
            "text": text,
            "source_path": str(path),
        })
    return docs


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

# Cheap clause-heading detector: lines like "8.2.3", "§8.2.3", or
# "8.2.3 허용 층간변위비". Used as a soft break preference inside chunks.
_CLAUSE_HEADING_RE = re.compile(
    r"^\s*(§?\d+(?:\.\d+){1,4})(?:[\s.\-:]\s*(.*))?$",
    re.MULTILINE,
)


def _split_into_paragraphs(text: str) -> list[str]:
    """Split on blank lines; collapse internal single newlines into spaces."""
    paragraphs: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        block = re.sub(r"\s+", " ", block)
        paragraphs.append(block)
    return paragraphs


def chunk_kds_document(
    doc: dict[str, Any],
    *,
    max_chars: int = 1200,
    overlap: int = 150,
) -> list[dict[str, Any]]:
    """Split ``doc['text']`` into citation-friendly chunks.

    Each returned chunk inherits the doc's metadata and adds:
        chunk_id, text, clause (best-effort), char_start, char_end
    """
    text = (doc.get("text") or "").strip()
    if not text:
        return []
    max_chars = max(100, int(max_chars))
    overlap = max(0, int(overlap))
    if overlap >= max_chars:
        overlap = max_chars // 4

    paragraphs = _split_into_paragraphs(text)
    if not paragraphs:
        return []

    chunks: list[dict[str, Any]] = []
    buf: list[str] = []
    buf_len = 0
    current_clause: Optional[str] = doc.get("clause")

    def _flush(start_offset: int) -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        body = " ".join(buf).strip()
        if not body:
            buf = []
            buf_len = 0
            return
        h = _short_hash(f"{doc.get('doc_id')}|{start_offset}|{body[:64]}")
        chunk = {
            "chunk_id": f"{doc.get('doc_id', 'doc')}_c{len(chunks):04d}_{h}",
            "doc_id": doc.get("doc_id"),
            "standard_id": doc.get("standard_id") or doc.get("doc_id") or "KDS",
            "title": doc.get("title"),
            "clause": current_clause,
            "clause_id": current_clause,
            "text": body,
            "source_path": doc.get("source_path"),
            "version": doc.get("version"),
            "effective_date": doc.get("effective_date"),
            "source_url": doc.get("source_url"),
            "material": doc.get("material"),
            "topic": doc.get("topic"),
            "limit_state": doc.get("limit_state"),
            "jurisdiction": doc.get("jurisdiction"),
            "char_start": start_offset,
            "char_end": start_offset + len(body),
        }
        chunks.append(chunk)
        buf = []
        buf_len = 0

    cursor = 0
    chunk_start = 0
    for p in paragraphs:
        # Pick up clause heading hints from short leading lines.
        m = _CLAUSE_HEADING_RE.match(p)
        if m and len(p) <= 80:
            new_clause = m.group(1)
            # Heading-only paragraph: don't emit it as its own chunk,
            # carry it into the next paragraph's metadata.
            current_clause = new_clause
            cursor += len(p) + 2
            continue

        if buf_len + len(p) + 1 > max_chars and buf:
            _flush(chunk_start)
            chunk_start = cursor
            # Carry overlap from the previous chunk's tail.
            if overlap > 0 and chunks:
                tail = chunks[-1]["text"][-overlap:]
                buf = [tail]
                buf_len = len(tail)

        buf.append(p)
        buf_len += len(p) + 1
        cursor += len(p) + 2

    _flush(chunk_start)
    return chunks


# ---------------------------------------------------------------------------
# Index build
# ---------------------------------------------------------------------------

def build_kds_index(
    source_dir: str,
    index_path: str,
    *,
    embedding_model: str = "voyage-4-large",
    api_key: Optional[str] = None,
    batch_size: int = 32,
    embedding_client: Any = None,
) -> dict[str, Any]:
    """Embed every chunk and write a JSONL index file.

    Returns a summary dict (n_docs, n_chunks, model, dim, index_path).

    Raises RuntimeError when the Voyage SDK or API key is missing — the
    CLI build script wants a clear stop-the-world failure mode here
    (unlike the endpoint, which falls back to Noop).
    """
    docs = load_kds_documents(source_dir)
    if not docs:
        raise RuntimeError(
            f"No supported (.txt/.md/.json) documents found under {source_dir}"
        )

    all_chunks: list[dict[str, Any]] = []
    for doc in docs:
        all_chunks.extend(chunk_kds_document(doc))
    if not all_chunks:
        raise RuntimeError(
            f"Loaded {len(docs)} doc(s) but produced zero chunks "
            "(empty text?)"
        )

    if embedding_client is None:
        # Lazy import keeps voyage SDK optional for library callers.
        from .voyage_retriever import VoyageEmbeddingClient
        embedding_client = VoyageEmbeddingClient(
            api_key=api_key, model=embedding_model,
        )

    # Batch the embedding calls so big indices don't blow Voyage's request
    # size cap. The model name is whatever the supplied client carries.
    out_path = Path(index_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dim: Optional[int] = None
    with out_path.open("w", encoding="utf-8") as f:
        for start in range(0, len(all_chunks), batch_size):
            batch = all_chunks[start: start + batch_size]
            texts = [c["text"] for c in batch]
            vectors = embedding_client.embed_documents(texts)
            if len(vectors) != len(batch):
                raise RuntimeError(
                    f"embedding returned {len(vectors)} vectors for "
                    f"batch of {len(batch)}"
                )
            for chunk, vec in zip(batch, vectors):
                if dim is None:
                    dim = len(vec)
                rec = dict(chunk)
                rec["embedding"] = list(vec)
                f.write(json.dumps(rec, ensure_ascii=False))
                f.write("\n")
    return {
        "n_docs": len(docs),
        "n_chunks": len(all_chunks),
        "model": embedding_model,
        "dim": dim or 0,
        "index_path": str(out_path),
    }


__all__ = [
    "SUPPORTED_EXTENSIONS",
    "load_kds_documents",
    "chunk_kds_document",
    "build_kds_index",
]
