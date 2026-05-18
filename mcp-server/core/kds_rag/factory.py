"""Factory for the default KDS retriever.

The /explain endpoint reads ``get_default_kds_retriever()`` once per
request. The factory picks the most capable retriever it can construct
*without* raising, and falls back silently to the Noop retriever if any
prerequisite is missing:

    1. VoyageKDSRetriever — needs both VOYAGE_API_KEY (or VOYAGEAI_API_KEY)
       and a readable KDS_RAG_INDEX_PATH.
    2. NoopKDSRetriever (default) — always returns empty results with an
       explicit ``kds_rag_unavailable`` warning so the API surface is
       honest about the missing index.

Endpoint-side note: this factory MUST NOT raise. If a future
backend's constructor can fail (e.g. corrupt index, network probe), it
must be wrapped here. The CLI build script (``scripts/build_kds_rag_index.py``)
calls the same client classes directly and is free to surface
``RuntimeError`` so the operator notices the misconfiguration.
"""
from __future__ import annotations

import logging
import os

from .retriever import KDSRetriever, NoopKDSRetriever

logger = logging.getLogger(__name__)


def get_default_kds_retriever() -> KDSRetriever:
    """Return the KDS retriever appropriate for the current environment.

    Never raises. The endpoint relies on this — a failed retriever
    initialization must downgrade to Noop, not crash /explain.
    """
    index_path = os.environ.get("KDS_RAG_INDEX_PATH")
    api_key = (
        os.environ.get("VOYAGE_API_KEY")
        or os.environ.get("VOYAGEAI_API_KEY")
    )

    if index_path and api_key and os.path.exists(index_path):
        try:
            # Lazy import — keeps Voyage SDK optional. If the module is
            # missing or the SDK fails to import we silently fall back.
            from .voyage_retriever import VoyageKDSRetriever
            return VoyageKDSRetriever(index_path=index_path, api_key=api_key)
        except Exception as exc:  # noqa: BLE001 — endpoint safety
            logger.warning(
                "Voyage KDS retriever init failed (%s); "
                "falling back to NoopKDSRetriever",
                exc,
            )

    return NoopKDSRetriever()


__all__ = ["get_default_kds_retriever"]
