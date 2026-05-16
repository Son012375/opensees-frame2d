"""KDS-RAG interface layer (test-double phase).

This package fixes the *contract* between the recommendation pipeline
and a future KDS-RAG retrieval system. It deliberately does NOT:

    * call any LLM
    * connect to a vector DB
    * ingest real KDS source text
    * mutate /api/v2/analyze responses by default

Public surface:
    from core.kds_rag import (
        KDSChunk, KDSRetrievalQuery, KDSRetrievalResult,
        CitationValidationResult, KDSRetriever, InMemoryKDSRetriever,
        build_kds_queries_for_issue,
        build_kds_queries_for_candidate,
        enrich_code_refs_with_kds,
        enrich_recommendation_payload_with_kds,
        validate_code_reference, validate_code_references,
        MAX_QUOTE_LEN,
    )
"""
from __future__ import annotations

from .schemas import (
    KDSChunk,
    KDSRetrievalQuery,
    KDSRetrievalResult,
    CitationValidationResult,
)
from .retriever import KDSRetriever, InMemoryKDSRetriever
from .citation_validator import (
    MAX_QUOTE_LEN,
    validate_code_reference,
    validate_code_references,
)
from .pipeline import (
    build_kds_queries_for_issue,
    build_kds_queries_for_candidate,
    enrich_code_refs_with_kds,
    enrich_recommendation_payload_with_kds,
    chunk_to_code_reference,
)

__all__ = [
    "KDSChunk",
    "KDSRetrievalQuery",
    "KDSRetrievalResult",
    "CitationValidationResult",
    "KDSRetriever",
    "InMemoryKDSRetriever",
    "MAX_QUOTE_LEN",
    "validate_code_reference",
    "validate_code_references",
    "build_kds_queries_for_issue",
    "build_kds_queries_for_candidate",
    "enrich_code_refs_with_kds",
    "enrich_recommendation_payload_with_kds",
    "chunk_to_code_reference",
]
