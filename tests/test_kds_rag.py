"""Tests for the KDS-RAG interface layer (test-double phase).

Scope: schema serialization, in-memory retriever behavior, query
construction from recommendation refs, payload-level enrichment, and
citation-validator guardrails.

Real RAG / vector DB / LLM are NOT covered here — none are
implemented. Fixture chunks are SHORT synthetic snippets; do not
paste real KDS source into this file.

Run:
    pytest tests/test_kds_rag.py -q
"""
from __future__ import annotations

import json
import os
import sys

# mcp-server를 import path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mcp-server"))

from core.kds_rag import (  # noqa: E402
    CitationValidationResult,
    InMemoryKDSRetriever,
    KDSChunk,
    KDSRetrievalQuery,
    KDSRetrievalResult,
    KDSRetriever,
    MAX_QUOTE_LEN,
    build_kds_queries_for_candidate,
    build_kds_queries_for_issue,
    chunk_to_code_reference,
    enrich_code_refs_with_kds,
    enrich_recommendation_payload_with_kds,
    make_kds_query,
    validate_code_reference,
    validate_code_references,
)
from core.recommendation import (  # noqa: E402
    ActionType,
    CodeReference,
    IssueSource,
    IssueType,
    MODE_ANALYZE,
    RetrofitCandidate,
    Severity,
    StructuralIssue,
    build_recommendation_payload,
)


# ============================================================
# Fixture builders (SHORT synthetic text only — no real KDS)
# ============================================================

def _h1_chunk() -> KDSChunk:
    return KDSChunk(
        chunk_id="kds_41_31_00_h1_p72",
        standard_id="KDS 41 31 00",
        version="2022-10-11",
        clause_id="H1",
        title="조합응력 상관식",
        text="Synthetic H1 placeholder text for testing. Combined "
             "axial + bending interaction check per AISC 360 H1.",
        source_url="https://example.invalid/kds-41-31-00#h1",
        page=72,
        section_path=["KDS 41 31 00", "H", "H1"],
        material="steel",
        topic="steel_member_strength",
        limit_state="combined_force_interaction",
    )


def _g2_chunk() -> KDSChunk:
    return KDSChunk(
        chunk_id="kds_41_31_00_g2_p65",
        standard_id="KDS 41 31 00",
        clause_id="G2",
        title="전단강도",
        text="Synthetic G2 placeholder. Shear strength provisions.",
        source_url="https://example.invalid/kds-41-31-00#g2",
        material="steel",
        topic="steel_member_shear",
        limit_state="shear_strength",
    )


def _drift_chunk() -> KDSChunk:
    return KDSChunk(
        chunk_id="kds_41_17_00_8_2_3_p41",
        standard_id="KDS 41 17 00",
        clause_id="§8.2.3",
        title="허용 층간변위비",
        text="Synthetic story-drift placeholder. Cd × δe / IE check.",
        source_url="https://example.invalid/kds-41-17-00#8.2.3",
        topic="seismic_story_drift",
        limit_state="story_drift",
    )


def _long_chunk() -> KDSChunk:
    """A chunk whose text exceeds MAX_QUOTE_LEN to test rejection."""
    return KDSChunk(
        chunk_id="kds_long_quote_p1",
        standard_id="KDS 41 31 00",
        clause_id="H1",
        title="조합응력 상관식 (long)",
        text="x" * (MAX_QUOTE_LEN + 50),
        topic="steel_member_strength",
        limit_state="combined_force_interaction",
    )


def _no_source_chunk() -> KDSChunk:
    """A chunk that, if turned into a ref, has no source_url AND would
    be created without a chunk_id propagated — used to test the
    'missing source' rejection path on the ref itself."""
    return KDSChunk(
        chunk_id="",  # validator receives chunk_id="" → falsy
        standard_id="KDS 41 31 00",
        clause_id="H1",
        title="No source",
        text="Synthetic with no source_url and no chunk_id.",
        topic="steel_member_strength",
        limit_state="combined_force_interaction",
    )


def _h1_hint_ref() -> CodeReference:
    return CodeReference(
        standard_id="KDS 41 31 00",
        clause_id="H1",
        title="조합응력 상관식 (AISC 360 H1)",
        query_hint="KDS 41 31 00 강구조 조합응력 AISC 360 H1 interaction",
        topic="steel_member_strength",
        material="steel",
        limit_state="combined_force_interaction",
        jurisdiction="KDS",
    )


def _drift_hint_ref() -> CodeReference:
    return CodeReference(
        standard_id="KDS 41 17 00",
        clause_id="§8.2.3",
        title="허용 층간변위비",
        query_hint="KDS 41 17 00 허용 층간변위비 inelastic story drift",
        topic="seismic_story_drift",
        limit_state="story_drift",
        jurisdiction="KDS",
    )


# ============================================================
# Schemas
# ============================================================

class TestKDSChunkSerialization:
    def test_serializes_to_dict_and_back_through_json(self):
        c = _h1_chunk()
        d = c.to_dict()
        # JSON-safe
        round_trip = json.loads(json.dumps(d))
        assert round_trip["chunk_id"] == "kds_41_31_00_h1_p72"
        assert round_trip["standard_id"] == "KDS 41 31 00"
        assert round_trip["topic"] == "steel_member_strength"
        # Optional fields with None are dropped
        no_source = KDSChunk(chunk_id="x", standard_id="Y", text="z")
        d2 = no_source.to_dict()
        assert "source_url" not in d2
        assert "clause_id" not in d2

    def test_retrieval_result_to_dict(self):
        q = KDSRetrievalQuery(query_id="q1", query_text="hello")
        r = KDSRetrievalResult(query=q, chunks=[_h1_chunk()], warnings=["w"])
        d = r.to_dict()
        assert d["query"]["query_id"] == "q1"
        assert len(d["chunks"]) == 1
        assert d["warnings"] == ["w"]


# ============================================================
# InMemoryKDSRetriever
# ============================================================

class TestInMemoryRetriever:
    def test_retrieves_by_topic_limit_state_material(self):
        retr = InMemoryKDSRetriever([_h1_chunk(), _g2_chunk(), _drift_chunk()])
        q = KDSRetrievalQuery(
            query_id="q",
            query_text="H1 combined force interaction",
            topic="steel_member_strength",
            material="steel",
            limit_state="combined_force_interaction",
            standard_id="KDS 41 31 00",
        )
        result = retr.retrieve(q, top_k=2)
        assert isinstance(result, KDSRetrievalResult)
        assert result.is_empty is False
        assert result.chunks[0].chunk_id == "kds_41_31_00_h1_p72"
        # G2 has the same standard_id but different topic/limit_state
        assert all(c.chunk_id != "kds_41_17_00_8_2_3_p41" for c in result.chunks)

    def test_respects_top_k(self):
        retr = InMemoryKDSRetriever([_h1_chunk(), _g2_chunk(), _drift_chunk()])
        # Broad query — let everything score, then cap at 2
        q = KDSRetrievalQuery(
            query_id="q", query_text="KDS",
            standard_id="KDS 41 31 00",
        )
        result = retr.retrieve(q, top_k=2)
        assert len(result.chunks) <= 2

    def test_empty_index_returns_warning(self):
        retr = InMemoryKDSRetriever([])
        q = KDSRetrievalQuery(query_id="q", query_text="hi")
        result = retr.retrieve(q, top_k=3)
        assert result.is_empty
        assert any("retriever_empty" in w for w in result.warnings)

    def test_no_match_returns_warning_not_exception(self):
        retr = InMemoryKDSRetriever([_h1_chunk()])
        q = KDSRetrievalQuery(
            query_id="q",
            query_text="unrelated stuff",
            topic="non_existent_topic",
            limit_state="non_existent_state",
        )
        result = retr.retrieve(q, top_k=3)
        assert result.is_empty
        assert any("no_match" in w for w in result.warnings)

    def test_results_are_deterministic_across_calls(self):
        retr = InMemoryKDSRetriever([_h1_chunk(), _g2_chunk()])
        q = KDSRetrievalQuery(
            query_id="q", query_text="KDS 41 31 00",
            standard_id="KDS 41 31 00",
        )
        a = retr.retrieve(q, top_k=2)
        b = retr.retrieve(q, top_k=2)
        assert [c.chunk_id for c in a.chunks] == [c.chunk_id for c in b.chunks]

    def test_is_subclass_of_abstract(self):
        # Contract: real backends will subclass KDSRetriever the same way.
        assert issubclass(InMemoryKDSRetriever, KDSRetriever)


# ============================================================
# Query construction
# ============================================================

class TestQueryConstruction:
    def test_issue_query_uses_query_hint(self):
        issue = StructuralIssue(
            issue_id="iss_strength_m7",
            issue_type=IssueType.STRENGTH_EXCEEDED,
            severity=Severity.ERROR,
            source=IssueSource.DESIGN_CHECK,
            description="…",
            member_id=7,
            code_refs=[_h1_hint_ref()],
        )
        queries = build_kds_queries_for_issue(issue)
        assert len(queries) == 1
        q = queries[0]
        assert q.issue_id == "iss_strength_m7"
        assert q.topic == "steel_member_strength"
        assert q.limit_state == "combined_force_interaction"
        assert q.standard_id == "KDS 41 31 00"
        assert "AISC 360 H1" in q.query_text

    def test_candidate_query_inherits_from_issue(self):
        issue = StructuralIssue(
            issue_id="iss_strength_m7",
            issue_type=IssueType.STRENGTH_EXCEEDED,
            severity=Severity.ERROR,
            source=IssueSource.DESIGN_CHECK,
            description="…",
            code_refs=[_h1_hint_ref()],
        )
        cand = RetrofitCandidate(
            candidate_id="cand_x",
            issue_id="iss_strength_m7",
            action_type=ActionType.INCREASE_SECTION,
            description="…",
            code_refs=[],  # candidate itself has no refs yet
        )
        queries = build_kds_queries_for_candidate(cand, issue)
        assert len(queries) == 1
        q = queries[0]
        assert q.candidate_id == "cand_x"
        assert q.issue_id == "iss_strength_m7"

    def test_candidate_with_no_hints_returns_no_query(self):
        cand = RetrofitCandidate(
            candidate_id="cand_y",
            issue_id="iss_y",
            action_type=ActionType.REQUIRES_ENGINEER_REVIEW,
            description="…",
            code_refs=[],
        )
        queries = build_kds_queries_for_candidate(cand, issue=None)
        assert queries == []

    def test_duplicate_queries_are_deduped(self):
        issue = StructuralIssue(
            issue_id="iss",
            issue_type=IssueType.STRENGTH_EXCEEDED,
            severity=Severity.ERROR,
            source=IssueSource.DESIGN_CHECK,
            description="…",
            code_refs=[_h1_hint_ref()],
        )
        cand = RetrofitCandidate(
            candidate_id="cand",
            issue_id="iss",
            action_type=ActionType.INCREASE_SECTION,
            description="…",
            code_refs=[_h1_hint_ref()],
        )
        # The cand and issue would otherwise yield the same query_id (same
        # discriminator). Dedup keeps only one.
        queries = build_kds_queries_for_candidate(cand, issue)
        assert len(queries) == 1

    def test_make_kds_query_includes_target_section(self):
        # When the caller only describes the member (chat compliance
        # tool — no proposed_change), the current section id must still
        # land in the query text so Voyage retrieval sees the size.
        q = make_kds_query({
            "issue_type": "strength_exceeded",
            "target": {
                "member_type": "column",
                "section": "H-300x300",
                "material": "steel",
            },
        })
        assert "H-300x300" in q.query_text
        assert "column" in q.query_text


# ============================================================
# Citation validator
# ============================================================

class TestCitationValidator:
    def test_accepts_full_ref_with_chunk_id(self):
        ref = chunk_to_code_reference(_h1_chunk())
        ok, reason = validate_code_reference(
            ref, chunk_id=_h1_chunk().chunk_id,
            matched_topic="steel_member_strength",
            matched_limit_state="combined_force_interaction",
        )
        assert ok, reason
        assert reason == "ok"

    def test_rejects_missing_source_and_chunk(self):
        ref = CodeReference(
            standard_id="KDS 41 31 00", clause_id="H1",
            title="조합응력",
        )
        ok, reason = validate_code_reference(ref, chunk_id=None)
        assert not ok
        assert reason == "missing_source_url_and_chunk_id"

    def test_rejects_missing_standard_id(self):
        ref = CodeReference(standard_id="", clause_id="H1",
                            source_url="http://x")
        ok, reason = validate_code_reference(ref, chunk_id=None)
        assert not ok
        assert reason == "missing_standard_id"

    def test_rejects_long_quote(self):
        ref = CodeReference(
            standard_id="KDS 41 31 00", clause_id="H1",
            source_url="http://x",
            quote="x" * (MAX_QUOTE_LEN + 5),
        )
        ok, reason = validate_code_reference(ref, chunk_id=None)
        assert not ok
        assert reason == "quote_too_long"

    def test_rejects_topic_mismatch(self):
        ref = _h1_hint_ref()
        ref.source_url = "http://x"
        ok, reason = validate_code_reference(
            ref, chunk_id=None,
            matched_topic="seismic_story_drift",
        )
        assert not ok
        assert "topic_mismatch" in reason

    def test_batch_validation_returns_split(self):
        good = CodeReference(
            standard_id="KDS 41 31 00", clause_id="H1",
            source_url="http://x",
        )
        bad = CodeReference(standard_id="", clause_id="x")
        result = validate_code_references([good, bad])
        assert isinstance(result, CitationValidationResult)
        assert result.valid is False
        assert len(result.accepted_refs) == 1
        assert len(result.rejected_refs) == 1


# ============================================================
# Chunk → CodeReference adapter
# ============================================================

class TestChunkAdapter:
    def test_short_chunk_yields_quote(self):
        ref = chunk_to_code_reference(_h1_chunk(), base_ref=_h1_hint_ref())
        assert ref.quote is not None
        assert ref.standard_id == "KDS 41 31 00"
        assert ref.clause_id == "H1"
        assert ref.topic == "steel_member_strength"
        # query_hint preserved from the base hint
        assert ref.query_hint is not None

    def test_long_chunk_is_excerpted_to_pass_validator(self):
        """Adapter excerpts oversized chunk text so realistic 1200-char
        retrieval chunks survive ``validate_code_reference``.

        Without this excerpt step the explainer would silently fall back
        to "RAG 미사용" on every long chunk even when Voyage retrieval
        succeeded — see Phase 3B live-smoke regression where all three
        sample chunks were rejected with ``quote_too_long``.
        """
        ref = chunk_to_code_reference(_long_chunk())
        assert ref.quote is not None
        # Excerpt ends with a horizontal ellipsis (U+2026) so the user
        # can tell it's a snippet, and the total length never exceeds
        # the citation cap.
        assert ref.quote.endswith("…")
        assert len(ref.quote) <= MAX_QUOTE_LEN
        # Validator now accepts the ref (chunk_id is present, quote is
        # within cap, source/topic still satisfied).
        ok, reason = validate_code_reference(
            ref, chunk_id="kds_long_quote_p1",
        )
        assert ok, f"expected acceptance after excerpt, got: {reason}"

    def test_short_chunk_text_passes_through_unchanged(self):
        """Chunks already within MAX_QUOTE_LEN must not be touched —
        the excerpt logic only kicks in above the cap.
        """
        ref = chunk_to_code_reference(_h1_chunk())
        original = _h1_chunk().text.strip()
        assert ref.quote == original
        assert not ref.quote.endswith("…")

    def test_long_chunk_excerpt_prefers_korean_sentence_boundary(self):
        """When the head window contains a Korean sentence terminator
        ("다."), the excerpt should cut there instead of mid-word.
        """
        head = (
            "허용 층간변위비는 내진등급에 따라 다르게 정의된다. "
            "특등급은 0.010 × hsx 로 가장 엄격하다."
        )
        tail = " 추가 설명." * 80  # pushes total length well above the cap
        chunk = KDSChunk(
            chunk_id="kds_drift_excerpt_p1",
            standard_id="KDS 41 17 00",
            clause_id="8.2.3",
            title="허용 층간변위비",
            text=head + tail,
            topic="seismic_story_drift",
            limit_state="story_drift",
            source_url="https://example.invalid/kds-41-17-00#8.2.3",
        )
        ref = chunk_to_code_reference(chunk)
        assert ref.quote is not None
        assert len(ref.quote) <= MAX_QUOTE_LEN
        assert ref.quote.endswith("…")
        # Cut should land after a Korean sentence terminator, never
        # mid-sentence. The excerpt MUST contain "정의된다." (the first
        # full sentence) and should not contain the truncated tail
        # phrase that came after the cutoff.
        assert "정의된다." in ref.quote


# ============================================================
# Enrichment — object level
# ============================================================

class TestEnrichCodeRefsWithKds:
    def test_enrich_attaches_refs_to_issue_and_candidate(self):
        retr = InMemoryKDSRetriever([_h1_chunk(), _g2_chunk(), _drift_chunk()])
        issue = StructuralIssue(
            issue_id="iss_strength_m7",
            issue_type=IssueType.STRENGTH_EXCEEDED,
            severity=Severity.ERROR,
            source=IssueSource.DESIGN_CHECK,
            description="…",
            member_id=7,
            member_type="column",
            section="H-300x300",
            code_refs=[_h1_hint_ref()],
        )
        cand = RetrofitCandidate(
            candidate_id="cand_increase",
            issue_id="iss_strength_m7",
            action_type=ActionType.INCREASE_SECTION,
            description="…",
            code_refs=[],
        )
        before_issue_refs = len(issue.code_refs)
        before_cand_refs = len(cand.code_refs)

        summary = enrich_code_refs_with_kds(
            [issue], [cand], retr, top_k=2,
        )

        # New refs appended after the hint refs
        assert len(issue.code_refs) > before_issue_refs
        assert len(cand.code_refs) > before_cand_refs
        s = summary["kds_rag_summary"]
        assert s["enabled"] is True
        assert s["num_queries"] >= 1
        assert s["num_refs_attached"] >= 1
        assert s["citation_ready"] is True
        assert s["backend"] == "InMemoryKDSRetriever"

    def test_no_match_marks_unresolved(self):
        retr = InMemoryKDSRetriever([_drift_chunk()])
        # Issue points at H1 / steel strength, retriever only has drift
        issue = StructuralIssue(
            issue_id="iss_strength_m7",
            issue_type=IssueType.STRENGTH_EXCEEDED,
            severity=Severity.ERROR,
            source=IssueSource.DESIGN_CHECK,
            description="…",
            code_refs=[_h1_hint_ref()],
        )
        summary = enrich_code_refs_with_kds([issue], [], retr, top_k=3)
        s = summary["kds_rag_summary"]
        assert s["num_unresolved"] >= 1
        assert s["citation_ready"] is False

    def test_long_chunk_is_excerpted_and_attached_via_enrichment(self):
        """End-to-end: oversized retrieved chunks must survive the
        enrichment pipeline now that the adapter excerpts the quote.

        Pre-fix this test asserted rejection — but rejecting every long
        chunk meant `citation_ready` was effectively always False once
        the chunker started emitting realistic 1200-char chunks, which
        defeated the entire Voyage RAG path. The excerpt step restores
        the invariant: long chunks attach with a snippet ending in "…".
        """
        retr = InMemoryKDSRetriever([_long_chunk()])
        issue = StructuralIssue(
            issue_id="iss_strength_m7",
            issue_type=IssueType.STRENGTH_EXCEEDED,
            severity=Severity.ERROR,
            source=IssueSource.DESIGN_CHECK,
            description="…",
            code_refs=[_h1_hint_ref()],
        )
        summary = enrich_code_refs_with_kds([issue], [], retr, top_k=3)
        s = summary["kds_rag_summary"]
        assert s["num_refs_attached"] >= 1
        assert s["citation_ready"] is True
        # The attached ref carries the excerpted quote — find it on
        # the mutated issue and verify the snippet shape.
        excerpted_refs = [
            r for r in (issue.code_refs or [])
            if r.quote and r.quote.endswith("…")
        ]
        assert excerpted_refs, (
            f"expected at least one excerpted ref; got "
            f"{[(r.standard_id, r.quote and len(r.quote)) for r in issue.code_refs]}"
        )
        assert all(
            len(r.quote) <= MAX_QUOTE_LEN for r in excerpted_refs
        )


# ============================================================
# Enrichment — payload level
# ============================================================

class TestEnrichRecommendationPayload:
    def _build_payload(self) -> dict:
        dc = {
            "member_check": {
                "members": [{
                    "member_id": 7, "type": "column",
                    "section": "H-300x300",
                    "governing_combo": "1.2DL+1.0LL",
                    "story": 2,
                    "ratios": {"interaction": 1.42, "shear": 0.3},
                    "status": "NG",
                }],
            },
        }
        return build_recommendation_payload(
            design_check=dc, raw_warnings=None,
            analysis_metadata={"member_info": [{
                "member_id": 7, "type": "column", "ni": 11, "nj": 12,
            }]},
            stage="t", mode=MODE_ANALYZE,
        )

    def test_enriches_payload_does_not_mutate_input(self):
        payload = self._build_payload()
        retr = InMemoryKDSRetriever([_h1_chunk(), _g2_chunk()])

        snapshot = json.loads(json.dumps(payload))
        enriched = enrich_recommendation_payload_with_kds(
            payload, retr, top_k=2,
        )

        # Input not mutated
        assert payload == snapshot
        # Output has kds_rag fields
        assert "kds_rag_summary" in enriched
        assert "kds_rag_warnings" in enriched
        s = enriched["kds_rag_summary"]
        assert s["enabled"] is True
        assert s["citation_ready"] in (True, False)
        # Issues have more refs than before (hint + enriched)
        sissue = next(i for i in enriched["issues"]
                      if i["issue_type"] == IssueType.STRENGTH_EXCEEDED)
        before = next(i for i in snapshot["issues"]
                      if i["issue_type"] == IssueType.STRENGTH_EXCEEDED)
        assert len(sissue["code_refs"]) > len(before["code_refs"])

    def test_citation_ready_false_when_retriever_has_no_chunks(self):
        payload = self._build_payload()
        retr = InMemoryKDSRetriever([])  # empty
        enriched = enrich_recommendation_payload_with_kds(payload, retr, top_k=3)
        s = enriched["kds_rag_summary"]
        assert s["citation_ready"] is False
        assert s["num_refs_attached"] == 0
        # Warnings propagated from retriever
        assert any("retriever_empty" in w for w in enriched["kds_rag_warnings"])

    def test_recommendation_payload_default_does_not_include_kds(self):
        """build_recommendation_payload alone must NOT auto-attach KDS."""
        payload = self._build_payload()
        assert "kds_rag_summary" not in payload
        assert "kds_rag_warnings" not in payload


# ============================================================
# chunk_id provenance (CodeReference + payload round-trip)
# ============================================================

class TestChunkIdProvenance:
    def test_chunk_to_code_reference_preserves_chunk_id(self):
        chunk = _h1_chunk()
        ref = chunk_to_code_reference(chunk, base_ref=_h1_hint_ref())
        assert ref.chunk_id == chunk.chunk_id
        # And it serializes
        d = ref.to_dict()
        assert d["chunk_id"] == chunk.chunk_id

    def test_validator_falls_back_to_ref_chunk_id(self):
        """Caller can omit chunk_id and the validator should still find
        it on the ref itself."""
        ref = chunk_to_code_reference(_h1_chunk())
        # Strip source_url so the validator must rely on chunk_id
        ref.source_url = None
        # No explicit chunk_id kwarg → falls back to ref.chunk_id
        ok, reason = validate_code_reference(
            ref,
            matched_topic="steel_member_strength",
            matched_limit_state="combined_force_interaction",
        )
        assert ok, reason

    def test_validator_still_rejects_when_ref_has_no_chunk_id_or_url(self):
        ref = CodeReference(
            standard_id="KDS 41 31 00", clause_id="H1",
            title="조합응력",
        )
        # No chunk_id arg, no ref.chunk_id, no source_url
        ok, reason = validate_code_reference(ref)
        assert not ok
        assert reason == "missing_source_url_and_chunk_id"

    def test_payload_enrichment_round_trip_keeps_chunk_id(self):
        """Build a payload, enrich with KDS, serialize to JSON, and
        confirm every attached citation ref carries chunk_id."""
        dc = {"member_check": {"members": [{
            "member_id": 7, "type": "column", "section": "H-300x300",
            "governing_combo": "1.2DL+1.0LL", "story": 2,
            "ratios": {"interaction": 1.42, "shear": 0.3},
            "status": "NG",
        }]}}
        payload = build_recommendation_payload(
            design_check=dc, raw_warnings=None,
            stage="t", mode=MODE_ANALYZE,
        )
        retr = InMemoryKDSRetriever([_h1_chunk(), _g2_chunk()])
        enriched = enrich_recommendation_payload_with_kds(payload, retr, top_k=3)

        # JSON round-trip — chunk_id must survive
        as_json = json.loads(json.dumps(enriched))

        # At least one attached citation ref on the issue with a chunk_id
        sissue = next(i for i in as_json["issues"]
                      if i["issue_type"] == IssueType.STRENGTH_EXCEEDED)
        chunk_ids_on_issue = [r.get("chunk_id") for r in sissue["code_refs"]
                              if r.get("chunk_id")]
        assert chunk_ids_on_issue, (
            "expected at least one enriched ref with chunk_id on the issue"
        )

        # And on the corresponding candidate
        scand = next(c for c in as_json["recommendation_candidates"]
                     if c["action_type"] == ActionType.INCREASE_SECTION)
        chunk_ids_on_cand = [r.get("chunk_id") for r in scand["code_refs"]
                             if r.get("chunk_id")]
        assert chunk_ids_on_cand, (
            "expected at least one enriched ref with chunk_id on the candidate"
        )


# ============================================================
# Split summary semantics (citation_ready vs all_queries_resolved)
# ============================================================

class TestSummarySemantics:
    def test_summary_fields_present_with_correct_types(self):
        retr = InMemoryKDSRetriever([_h1_chunk()])
        issue = StructuralIssue(
            issue_id="iss_strength_m7",
            issue_type=IssueType.STRENGTH_EXCEEDED,
            severity=Severity.ERROR,
            source=IssueSource.DESIGN_CHECK,
            description="…",
            code_refs=[_h1_hint_ref()],
        )
        summary = enrich_code_refs_with_kds([issue], [], retr, top_k=3)
        s = summary["kds_rag_summary"]
        for k in ("citation_ready", "all_queries_resolved",
                  "has_unresolved_queries", "has_rejected_refs"):
            assert isinstance(s[k], bool), f"{k} should be bool"

    def test_partial_unresolved_separates_citation_ready_from_resolved(self):
        """One query resolves cleanly, another finds nothing.

        Expectation:
            citation_ready = True   (the H1 query worked)
            all_queries_resolved = False  (the drift query did not)
            has_unresolved_queries = True
        """
        # Retriever only knows about H1, NOT drift.
        retr = InMemoryKDSRetriever([_h1_chunk()])

        strength_issue = StructuralIssue(
            issue_id="iss_strength_m7",
            issue_type=IssueType.STRENGTH_EXCEEDED,
            severity=Severity.ERROR,
            source=IssueSource.DESIGN_CHECK,
            description="…",
            code_refs=[_h1_hint_ref()],
        )
        drift_issue = StructuralIssue(
            issue_id="iss_drift_s3_x",
            issue_type=IssueType.DRIFT_EXCEEDED,
            severity=Severity.ERROR,
            source=IssueSource.DESIGN_CHECK,
            description="…",
            code_refs=[_drift_hint_ref()],
        )

        summary = enrich_code_refs_with_kds(
            [strength_issue, drift_issue], [], retr, top_k=3,
        )
        s = summary["kds_rag_summary"]
        assert s["citation_ready"] is True
        assert s["all_queries_resolved"] is False
        assert s["has_unresolved_queries"] is True
        assert s["num_unresolved"] >= 1
        assert s["num_refs_attached"] >= 1

    def test_all_resolved_when_every_query_returns_accepted_ref(self):
        retr = InMemoryKDSRetriever([_h1_chunk()])
        issue = StructuralIssue(
            issue_id="iss_strength_m7",
            issue_type=IssueType.STRENGTH_EXCEEDED,
            severity=Severity.ERROR,
            source=IssueSource.DESIGN_CHECK,
            description="…",
            code_refs=[_h1_hint_ref()],
        )
        summary = enrich_code_refs_with_kds([issue], [], retr, top_k=3)
        s = summary["kds_rag_summary"]
        assert s["citation_ready"] is True
        assert s["all_queries_resolved"] is True
        assert s["has_unresolved_queries"] is False
        assert s["has_rejected_refs"] is False

    def test_rejected_refs_make_all_queries_resolved_false(self):
        """Even if a query returned chunks, a rejected ref invalidates
        all_queries_resolved.

        Uses ``_no_source_chunk`` (empty chunk_id + no source_url) to
        exercise the ``missing_source_url_and_chunk_id`` rejection. The
        long-quote rejection path went away when the adapter learned to
        excerpt — that's covered by the
        ``test_long_chunk_is_excerpted_and_attached_via_enrichment``
        positive case above.
        """
        retr = InMemoryKDSRetriever([_no_source_chunk()])
        issue = StructuralIssue(
            issue_id="iss_strength_m7",
            issue_type=IssueType.STRENGTH_EXCEEDED,
            severity=Severity.ERROR,
            source=IssueSource.DESIGN_CHECK,
            description="…",
            code_refs=[_h1_hint_ref()],
        )
        summary = enrich_code_refs_with_kds([issue], [], retr, top_k=3)
        s = summary["kds_rag_summary"]
        assert s["has_rejected_refs"] is True
        assert s["all_queries_resolved"] is False
