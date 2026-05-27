"""Phase D — explain_member_compliance chat tool.

Covers:
    * T1 Voyage hit (shear-dominant): mock retriever returns 2 shear
      chunks → evidence list of 2, rag_used=True, governing_ratio="shear",
      and the retriever sees a query whose topic reflects shear_exceeded.
    * T2 Voyage hit (strength-dominant): interaction>1.0, shear<1.0 →
      issue_type=strength_exceeded → query.topic="member_strength".
    * T3 Noop fallback: no env vars → real get_default_kds_retriever
      returns Noop → rag_used=False, kds_rag_unavailable warning,
      member_summary still populated.
    * T4 no_selection: empty arguments + no ui_context selection →
      code=no_selection.

The tool is intentionally tested at the handler level (no orchestrator
roundtrip) — the orchestrator branch is exercised by the regex
positive/negative tests at the end of this file.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webapp" / "backend"))
sys.path.insert(0, str(ROOT / "mcp-server"))

os.environ.setdefault("CHAT_LLM_PROVIDER", "noop")
# Match the other chat test modules' default exactly. This test calls
# the kds_compliance handler directly (no orchestrator round-trip via
# ``default_registry()``), so the env-driven group list is irrelevant
# here — keeping the value in lockstep with test_chat_tools.py /
# test_chat_section_change.py avoids cross-module env pollution
# (pytest collects alphabetically; this module loads before the
# others and a different setdefault would survive).
os.environ.setdefault("CHAT_TOOLS_ENABLED", "inspect,summary")

from app.services.analysis_context import (  # noqa: E402
    _ANALYSIS_CONTEXT_LOCK,
    analysis_context_cache,
)
from app.services.chat_session import (  # noqa: E402
    _CHAT_SESSION_LOCK,
    chat_session_cache,
)
from core.chat.orchestrator import (  # noqa: E402
    _FORCE_EXPLAIN_RE,
    _pick_forced_tool,
)
from core.chat.tools import kds_compliance  # noqa: E402
from core.chat.tools.kds_compliance import (  # noqa: E402
    EXPLAIN_MEMBER_COMPLIANCE_TOOL,
    _governing_issue_type,
    explain_member_compliance,
)
from core.kds_rag import (  # noqa: E402
    KDSChunk,
    KDSRetrievalQuery,
    KDSRetrievalResult,
    KDSRetriever,
)


ANALYSIS_ID = "analysis_phased_001"


@pytest.fixture(autouse=True)
def _clean_state():
    with _CHAT_SESSION_LOCK:
        chat_session_cache.clear()
    with _ANALYSIS_CONTEXT_LOCK:
        analysis_context_cache.clear()
    yield
    with _CHAT_SESSION_LOCK:
        chat_session_cache.clear()
    with _ANALYSIS_CONTEXT_LOCK:
        analysis_context_cache.clear()


# ---------------------------------------------------------------------------
# Cache seeding
# ---------------------------------------------------------------------------

def _seed_context(
    member_id: int,
    *,
    status: str,
    ratio_interaction: float = 0.0,
    ratio_shear: float = 0.0,
    ratio_axial: float = 0.0,
    ratio_bending: float = 0.0,
    section: str = "H-300x300",
    etype: str = "column",
    material: str = "SS275",
    story: int = 2,
    aid: str = ANALYSIS_ID,
) -> str:
    """Seed a minimal analysis_context with a single member entry.

    Mirrors the structure produced by ``derive_chat_lookups`` so the
    handler's lookup keys (``member_info_by_member_id`` etc.) hit.
    """
    info = {
        "member_id": member_id,
        "section": section,
        "material": material,
        "story": story,
        "etype": etype,
        "length_mm": 3500.0,
    }
    ratios = {
        "member_id": member_id,
        "status": status,
        "ratio_interaction": ratio_interaction,
        "ratio_shear": ratio_shear,
        "ratio_axial": ratio_axial,
        "ratio_bending": ratio_bending,
    }
    with _ANALYSIS_CONTEXT_LOCK:
        analysis_context_cache[aid] = {
            "expires_at": time.time() + 3600,
            "context_kind": "full",
            "model_json": {"elements": [], "nodes": []},
            "analysis_summary": {"ng_count": 1, "num_stories": 3, "num_elements": 1},
            "member_info_by_elem_id": {},
            "member_info_by_member_id": {str(member_id): info},
            "member_ratios_by_elem_id": {},
            "member_ratios_by_member_id": {str(member_id): ratios},
        }
    return aid


# ---------------------------------------------------------------------------
# Mock retrievers
# ---------------------------------------------------------------------------

class _RecordingRetriever(KDSRetriever):
    """Records the last query it received and returns a canned result."""

    def __init__(self, chunks: list[KDSChunk]):
        self.chunks = chunks
        self.last_query: KDSRetrievalQuery | None = None

    def retrieve(
        self,
        query: KDSRetrievalQuery,
        top_k: int = 5,
    ) -> KDSRetrievalResult:
        self.last_query = query
        return KDSRetrievalResult(
            query=query,
            chunks=list(self.chunks)[:top_k],
            warnings=[],
            scores={c.chunk_id: 0.9 - 0.1 * i for i, c in enumerate(self.chunks)},
        )


def _g2_shear_chunk(idx: int = 0) -> KDSChunk:
    return KDSChunk(
        chunk_id=f"kds_41_31_00_g2_{idx}",
        standard_id="KDS 41 31 00",
        version="2022-10-11",
        clause_id=f"G2-{idx}",
        title="전단강도 검토",
        text="Synthetic G2 placeholder. Shear strength provisions of "
             "AISC 360 G2 / KDS 41 31 00.",
        source_url=f"https://example.invalid/kds-41-31-00#g2-{idx}",
        topic="steel_member_shear",
        limit_state="shear_strength",
        material="steel",
    )


def _h1_strength_chunk(idx: int = 0) -> KDSChunk:
    return KDSChunk(
        chunk_id=f"kds_41_31_00_h1_{idx}",
        standard_id="KDS 41 31 00",
        version="2022-10-11",
        clause_id=f"H1-{idx}",
        title="조합응력 상관식",
        text="Synthetic H1 placeholder. Combined axial + bending "
             "interaction check per AISC 360 H1.",
        source_url=f"https://example.invalid/kds-41-31-00#h1-{idx}",
        topic="steel_member_strength",
        limit_state="combined_force_interaction",
        material="steel",
    )


# ---------------------------------------------------------------------------
# T1 — Voyage hit, shear-dominant
# ---------------------------------------------------------------------------

def test_t1_voyage_hit_shear_returns_evidence_and_governing_shear(monkeypatch):
    """ratio_shear=1.2 > ratio_interaction=0.5 → shear governs.

    The mock retriever asserts the issue_type → topic mapping ran
    (query.topic == 'member_shear') and the response carries 2 evidence
    objects + rag_used=True + mandatory_response that quotes them
    verbatim under doc_id headings (LLM bypass invariant).
    """
    aid = _seed_context(
        5, status="NG",
        ratio_interaction=0.5, ratio_shear=1.2,
        section="H-300x300",
    )
    retriever = _RecordingRetriever([_g2_shear_chunk(0), _g2_shear_chunk(1)])
    monkeypatch.setattr(
        kds_compliance, "_get_default_retriever", lambda: retriever,
    )

    result = explain_member_compliance(
        {"member_id": 5}, session={"analysis_id": aid, "history": []},
    )
    assert "error" not in result, result
    assert result["rag_used"] is True
    assert len(result["kds_evidence"]) == 2
    # Evidence preserves the citation fields verbatim.
    e0 = result["kds_evidence"][0]
    assert e0["doc_id"] == "KDS 41 31 00"
    assert e0["clause"] == "G2-0"
    assert "Shear" in e0["quote"]
    assert e0["score"] == pytest.approx(0.9, abs=1e-6)

    summary = result["member_summary"]
    assert summary["member_id"] == 5
    assert summary["status"] == "NG"
    assert summary["governing_ratio"] == "shear"
    assert summary["section"] == "H-300x300"

    # Toggle UX (Phase D follow-up): explain returns TWO deterministic
    # parts. ``summary`` is the always-visible top half (member info +
    # ratios + a one-line "근거 N건 첨부됨" footer). ``collapsible`` is
    # the bottom half — evidence quotes under <details> on the client.
    # Both are server-rendered; no LLM ever touches either.
    assert "mandatory_response_summary" in result
    assert "mandatory_response_collapsible" in result
    summary_text = result["mandatory_response_summary"]
    coll = result["mandatory_response_collapsible"]
    # Summary stays short and signals evidence is available.
    assert "부재 #5" in summary_text
    assert "근거 2건" in summary_text  # both retrieved chunks counted
    # Evidence quotes live exclusively in the collapsible — keeps the
    # always-visible message short.
    assert "KDS 41 31 00" in coll
    assert "G2-0" in coll
    assert "G2-1" in coll
    assert "Shear strength provisions" in coll
    # And the summary must NOT contain the clause numbers itself —
    # the footer announces the count but not the citations.
    assert "G2-0" not in summary_text

    # Retrieval routing: shear_exceeded → topic=member_shear; the section
    # boost we added in pipeline.py should also surface "H-300x300" in
    # the query text.
    q = retriever.last_query
    assert q is not None
    assert q.topic == "member_shear"
    assert q.limit_state == "shear_strength"
    assert "H-300x300" in q.query_text
    assert "column" in q.query_text


# ---------------------------------------------------------------------------
# T2 — Voyage hit, strength-dominant
# ---------------------------------------------------------------------------

def test_t2_voyage_hit_strength_routes_to_member_strength_topic(monkeypatch):
    """ratio_interaction=1.1 > ratio_shear=0.4 → strength governs.

    The retriever should see a query routed to the 'member_strength'
    topic — confirms the issue_type mapping doesn't accidentally fall
    through to shear when interaction dominates.
    """
    aid = _seed_context(
        7, status="NG",
        ratio_interaction=1.1, ratio_shear=0.4,
        ratio_axial=0.6, ratio_bending=0.8,
        section="H-400x400",
    )
    retriever = _RecordingRetriever([_h1_strength_chunk(0)])
    monkeypatch.setattr(
        kds_compliance, "_get_default_retriever", lambda: retriever,
    )

    result = explain_member_compliance(
        {"member_id": 7}, session={"analysis_id": aid, "history": []},
    )
    assert result["rag_used"] is True
    assert len(result["kds_evidence"]) == 1

    summary = result["member_summary"]
    assert summary["status"] == "NG"
    # interaction is the largest of the strength-side ratios.
    assert summary["governing_ratio"] == "interaction"

    q = retriever.last_query
    assert q is not None
    assert q.topic == "member_strength"
    assert q.limit_state == "strength"
    # Section keyword from pipeline boost must land in query text.
    assert "H-400x400" in q.query_text


# ---------------------------------------------------------------------------
# T3 — Noop fallback (no env vars set)
# ---------------------------------------------------------------------------

def test_t3_noop_fallback_returns_member_summary_only(monkeypatch):
    """With no VOYAGE_API_KEY / KDS_RAG_INDEX_PATH, get_default_kds_retriever
    returns Noop. The tool must still succeed with member_summary
    populated, rag_used=False, and the kds_rag_unavailable warning.

    Anti-hallucination contract: when rag_used=False the tool ALSO
    returns ``mandatory_response`` (a fully-rendered Korean answer)
    so the orchestrator can bypass LLM narration. Verifies the
    template carries the member id + 'KDS 근거 자료가 연결되지 않'
    framing and contains NO fabricated clause references.
    """
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.delenv("VOYAGEAI_API_KEY", raising=False)
    monkeypatch.delenv("KDS_RAG_INDEX_PATH", raising=False)

    aid = _seed_context(
        3, status="NG",
        ratio_interaction=1.05, ratio_shear=0.3,
        section="H-350x350",
    )

    result = explain_member_compliance(
        {"member_id": 3}, session={"analysis_id": aid, "history": []},
    )
    assert "error" not in result
    assert result["rag_used"] is False
    assert result["kds_evidence"] == []
    assert any("kds_rag_unavailable" in w for w in result["warnings"])
    # member_summary is still real — the user gets ratios even without RAG.
    assert result["member_summary"]["status"] == "NG"
    assert result["member_summary"]["section"] == "H-350x350"

    # Anti-hallucination: the deterministic answer is rendered server
    # side and handed to the orchestrator. No LLM ever sees this prompt.
    # No-evidence branch: only the summary field carries text — the
    # collapsible is None because there's nothing to put behind a
    # toggle.
    assert "mandatory_response_summary" in result
    text = result["mandatory_response_summary"]
    assert result.get("mandatory_response_collapsible") is None
    assert "근거 자료가 연결되지 않" in text
    assert "부재 #3" in text
    assert "H-350x350" in text
    # The whole point of this branch is "no fabricated clauses." The
    # template must not contain any specific KDS / AISC reference.
    import re as _re
    assert not _re.search(r"KDS\s+\d{2}\s+\d{2}\s+\d{2}", text), (
        "deterministic template leaked a KDS clause reference: " + text
    )
    assert not _re.search(r"AISC\s+\d{3}", text), (
        "deterministic template leaked an AISC clause reference: " + text
    )


# ---------------------------------------------------------------------------
# T4 — no selection
# ---------------------------------------------------------------------------

def test_t4_no_selection_returns_no_selection_code():
    """member_id absent AND history has no ui_context.selected_element_ids."""
    aid = _seed_context(1, status="OK")
    result = explain_member_compliance(
        {}, session={"analysis_id": aid, "history": []},
    )
    assert result.get("code") == "no_selection"
    assert "부재" in result.get("error", "")


# ---------------------------------------------------------------------------
# Extra invariants worth locking down
# ---------------------------------------------------------------------------

def test_falls_back_to_ui_selection_when_member_id_omitted(monkeypatch):
    """Latest ui_context.selected_element_ids[0] is used as member_id when
    the LLM/UI omits an explicit argument. Mirrors propose_section_change
    fallback semantics."""
    aid = _seed_context(
        4, status="NG",
        ratio_interaction=1.05, ratio_shear=0.2,
    )
    retriever = _RecordingRetriever([_h1_strength_chunk(0)])
    monkeypatch.setattr(
        kds_compliance, "_get_default_retriever", lambda: retriever,
    )
    session = {
        "analysis_id": aid,
        "history": [
            {"role": "user", "content": "왜 NG?",
             "ui_context": {"selected_element_ids": [4]}},
        ],
    }
    result = explain_member_compliance({}, session=session)
    assert result["member_summary"]["member_id"] == 4


def test_member_not_found_returns_specific_code(monkeypatch):
    """member_id=999 isn't in the cache → discrete error code so the
    chat layer can phrase a helpful message rather than crash on a KeyError."""
    aid = _seed_context(1, status="OK")
    result = explain_member_compliance(
        {"member_id": 999}, session={"analysis_id": aid, "history": []},
    )
    assert result.get("code") == "member_not_found"


def test_analysis_not_found_returns_specific_code():
    """analysis_id refers to an evicted/missing cache entry."""
    result = explain_member_compliance(
        {"member_id": 1, "analysis_id": "missing"},
        session={"history": []},
    )
    assert result.get("code") == "analysis_not_found"


def test_governing_issue_type_ok_routes_to_missing_design_check():
    """status=OK → use missing_design_check sentinel so the query still
    carries domain keywords through to RAG."""
    issue_type, governing = _governing_issue_type({
        "status": "OK", "ratio_interaction": 0.4, "ratio_shear": 0.3,
    })
    assert issue_type == "missing_design_check"
    assert governing == "ok"


def test_tool_spec_is_in_kds_group_and_factual():
    """Defensive: ensure the ToolSpec metadata stays as the orchestrator
    expects (group=kds, creativity=factual)."""
    assert EXPLAIN_MEMBER_COMPLIANCE_TOOL.group == "kds"
    assert EXPLAIN_MEMBER_COMPLIANCE_TOOL.creativity_hint == "factual"


# ---------------------------------------------------------------------------
# Force-routing regex coverage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("msg", [
    "5번 부재 왜 NG야?",
    "이 부재 왜 안전한가?",
    "왜 NG?",
    "왜 안전한가요?",
    "이 부재 설계기준은 뭐야?",
    "근거가 뭐예요?",
    "KDS 어느 조항?",
    "AISC 조항?",
    "조항 알려줘",
    # Korean copula variants — '안전한가'만 잡으면 자연스러운 회화체
    # ("안전해?", "안전합니까?")가 inspect로 빠지는 라이브 smoke 회귀.
    "이 부재 안전해?",
    "이 부재 안전합니까?",
    "이 부재 안전할까?",
])
def test_force_explain_re_positive(msg):
    assert _FORCE_EXPLAIN_RE.search(msg), f"expected match: {msg!r}"


@pytest.mark.parametrize("msg", [
    "5번 부재 정보 보여줘",
    "결과 요약해줘",
    "단면이 뭐야?",
    "안녕",
    "층간변위 알려줘",
])
def test_force_explain_re_negative(msg):
    assert _FORCE_EXPLAIN_RE.search(msg) is None, f"false positive: {msg!r}"


def test_pick_forced_tool_prefers_explain_over_inspect():
    """'5번 부재 왜 NG?' matches BOTH inspect ('\\d+번 부재') and explain
    ('왜 NG'). explain must win — inspect_selection has no RAG hook so
    the user would lose the KDS quote otherwise."""
    available = {
        "inspect_selection", "get_analysis_summary",
        "propose_section_change", "explain_member_compliance",
    }
    assert _pick_forced_tool(
        "5번 부재 왜 NG야?", available,
    ) == "explain_member_compliance"


def test_pick_forced_tool_still_prefers_command_over_explain():
    """Command verbs win over both inspect and explain — a user asking
    to *change* a section doesn't want a KDS dump first."""
    available = {
        "inspect_selection", "get_analysis_summary",
        "propose_section_change", "explain_member_compliance",
    }
    assert _pick_forced_tool(
        "5번 부재를 H-400x400으로 바꿔줘", available,
    ) == "propose_section_change"


def test_pick_forced_tool_explain_falls_back_when_tool_absent():
    """If explain_member_compliance isn't enabled, the message routes to
    inspect (the '\\d+번 부재' fallback) — defence in depth."""
    available = {"inspect_selection", "get_analysis_summary"}
    assert _pick_forced_tool(
        "5번 부재 왜 NG?", available,
    ) == "inspect_selection"


# ---------------------------------------------------------------------------
# Orchestrator-level: mandatory_response bypass
# ---------------------------------------------------------------------------

def test_orchestrator_hybrid_discards_lying_prefix_keeps_deterministic_block(monkeypatch):
    """Hybrid invariant: the LLM IS called for the prefix, but its
    output is DISCARDED if it contains a fabricated clause pattern
    (`KDS XX XX XX`, `AISC 360`, `§N.N`). The deterministic block from
    ``mandatory_response`` is ALWAYS emitted intact.

    This is the failure mode we saw live in 2026-05-27: qwen2.5:3b
    ignored rule 7 and invented "KDS 41-31-00" / "§8.2-1". The hybrid
    design accepts a short prose intro from the LLM when (and only
    when) it stays out of the clause-citation zone.
    """
    import asyncio
    import json as _json
    from typing import AsyncIterator
    from core.chat.llm.base import BaseLLMProvider
    from core.chat.orchestrator import ChatOrchestrator
    from core.chat.tool_registry import ToolRegistry
    from core.chat.tools.inspect import (
        GET_ANALYSIS_SUMMARY_TOOL,
        INSPECT_SELECTION_TOOL,
    )

    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.delenv("VOYAGEAI_API_KEY", raising=False)
    monkeypatch.delenv("KDS_RAG_INDEX_PATH", raising=False)

    aid = _seed_context(
        5, status="NG",
        ratio_interaction=0.4, ratio_shear=1.3,
        section="H-300x300",
    )

    class _LyingProvider(BaseLLMProvider):
        """Pretends to be qwen2.5:3b that ignores rule 7 and
        fabricates a clause number in its prefix."""
        name = "lying"

        def __init__(self):
            self.stream_called = False

        async def request_tool_call(self, *, messages, tools, temperature=None):
            return None  # tool routing is handled by force-routing

        async def stream_tokens(
            self, *, messages, temperature=None,
        ) -> AsyncIterator[str]:
            self.stream_called = True
            # Fabricated citation — orchestrator must discard.
            yield "이 부재는 KDS 41 31 00 §7.3.2에 따라 NG입니다 (가짜 인용)"

    provider = _LyingProvider()
    registry = ToolRegistry(
        [INSPECT_SELECTION_TOOL, GET_ANALYSIS_SUMMARY_TOOL,
         EXPLAIN_MEMBER_COMPLIANCE_TOOL],
        enabled_groups=frozenset({"inspect", "summary", "kds"}),
    )
    orch = ChatOrchestrator(provider, registry=registry)

    async def _collect():
        return [
            ln async for ln in orch.run_turn(
                session={"analysis_id": aid, "history": []},
                user_message="5번 부재 왜 NG?",
            )
        ]

    lines = asyncio.run(_collect())
    events = [_json.loads(ln) for ln in lines if ln.strip()]
    tokens = [e for e in events if e["type"] == "token"]

    # Hybrid: LLM IS invoked (no full bypass) ...
    assert provider.stream_called is True, (
        "hybrid mode must call LLM for the prefix — full bypass is wrong"
    )
    # ... but the lying prefix was discarded so the user never sees the
    # fabricated citation.
    assert tokens, "no token events emitted"
    combined = "".join(t["text"] for t in tokens)
    assert "§7.3.2" not in combined, (
        f"fabricated §7.3.2 leaked through despite guard: {combined!r}"
    )
    assert "가짜 인용" not in combined, (
        f"lying prefix not discarded: {combined!r}"
    )
    assert "KDS 41 31 00" not in combined
    # Deterministic block is always intact.
    assert "근거 자료가 연결되지 않" in combined
    assert "부재 #5" in combined


def test_orchestrator_hybrid_keeps_clean_prefix(monkeypatch):
    """Negative companion: when the LLM behaves and writes a clean,
    short prose intro (no clause patterns), it IS preserved and
    prepended to the deterministic summary. Noop path (no evidence) so
    no collapsible event."""
    import asyncio
    import json as _json
    from typing import AsyncIterator
    from core.chat.llm.base import BaseLLMProvider
    from core.chat.orchestrator import ChatOrchestrator
    from core.chat.tool_registry import ToolRegistry
    from core.chat.tools.inspect import (
        GET_ANALYSIS_SUMMARY_TOOL,
        INSPECT_SELECTION_TOOL,
    )

    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.delenv("VOYAGEAI_API_KEY", raising=False)
    monkeypatch.delenv("KDS_RAG_INDEX_PATH", raising=False)

    aid = _seed_context(
        5, status="NG",
        ratio_interaction=0.4, ratio_shear=1.3,
        section="H-300x300",
    )

    CLEAN_PREFIX = "5번 부재가 전단력 과다로 NG 판정되었습니다. 주요 근거는 다음과 같습니다:"

    class _GoodProvider(BaseLLMProvider):
        name = "good"

        async def request_tool_call(self, *, messages, tools, temperature=None):
            return None

        async def stream_tokens(
            self, *, messages, temperature=None,
        ) -> AsyncIterator[str]:
            yield CLEAN_PREFIX

    registry = ToolRegistry(
        [INSPECT_SELECTION_TOOL, GET_ANALYSIS_SUMMARY_TOOL,
         EXPLAIN_MEMBER_COMPLIANCE_TOOL],
        enabled_groups=frozenset({"inspect", "summary", "kds"}),
    )
    orch = ChatOrchestrator(_GoodProvider(), registry=registry)

    async def _collect():
        return [
            ln async for ln in orch.run_turn(
                session={"analysis_id": aid, "history": []},
                user_message="5번 부재 왜 NG?",
            )
        ]

    lines = asyncio.run(_collect())
    events = [_json.loads(ln) for ln in lines if ln.strip()]
    combined = "".join(e["text"] for e in events if e["type"] == "token")
    # Clean prefix preserved AND deterministic summary follows.
    assert combined.startswith(CLEAN_PREFIX)
    assert "\n\n" in combined  # separator between prefix and summary
    assert "부재 #5" in combined
    assert "근거 자료가 연결되지 않" in combined  # summary contains the no-RAG notice
    # Noop path: no evidence → no collapsible event.
    assert not any(e["type"] == "collapsible" for e in events)


def test_orchestrator_hybrid_emits_collapsible_event_when_evidence_present(monkeypatch):
    """Toggle UX invariant: when the tool returns a non-empty
    ``mandatory_response_collapsible`` (evidence list non-empty), the
    orchestrator emits it as a SEPARATE EVENT_COLLAPSIBLE so the chat
    widget can render <details>. The visible token stream stays short
    (prefix + summary), and the evidence text MUST NOT appear inside
    any token event — only inside the collapsible event."""
    import asyncio
    import json as _json
    from typing import AsyncIterator
    from core.chat.llm.base import BaseLLMProvider
    from core.chat.orchestrator import ChatOrchestrator
    from core.chat.tool_registry import ToolRegistry
    from core.chat.tools import kds_compliance
    from core.chat.tools.inspect import (
        GET_ANALYSIS_SUMMARY_TOOL,
        INSPECT_SELECTION_TOOL,
    )

    aid = _seed_context(
        5, status="NG",
        ratio_interaction=0.4, ratio_shear=1.3,
        section="H-300x300",
    )
    # Mock retriever returns 2 evidence chunks → collapsible non-empty.
    retriever = _RecordingRetriever([_g2_shear_chunk(0), _g2_shear_chunk(1)])
    monkeypatch.setattr(
        kds_compliance, "_get_default_retriever", lambda: retriever,
    )

    class _QuietProvider(BaseLLMProvider):
        name = "quiet"

        async def request_tool_call(self, *, messages, tools, temperature=None):
            return None

        async def stream_tokens(self, *, messages, temperature=None) -> AsyncIterator[str]:
            yield "5번 부재 NG."  # clean short prefix

    registry = ToolRegistry(
        [INSPECT_SELECTION_TOOL, GET_ANALYSIS_SUMMARY_TOOL,
         EXPLAIN_MEMBER_COMPLIANCE_TOOL],
        enabled_groups=frozenset({"inspect", "summary", "kds"}),
    )
    orch = ChatOrchestrator(_QuietProvider(), registry=registry)

    async def _collect():
        return [
            ln async for ln in orch.run_turn(
                session={"analysis_id": aid, "history": []},
                user_message="5번 부재 왜 NG?",
            )
        ]

    events = [_json.loads(ln) for ln in asyncio.run(_collect()) if ln.strip()]
    tokens = [e for e in events if e["type"] == "token"]
    collapsibles = [e for e in events if e["type"] == "collapsible"]

    # Exactly one collapsible event with the evidence quote inside it.
    assert len(collapsibles) == 1
    coll = collapsibles[0]
    assert coll.get("summary_label")
    coll_text = coll["text"]
    assert "G2-0" in coll_text
    assert "Shear strength provisions" in coll_text

    # Token stream MUST NOT carry the quote — otherwise the toggle UX
    # is pointless (the long block would still be visible by default).
    token_text = "".join(t["text"] for t in tokens)
    assert "Shear strength provisions" not in token_text
    assert "G2-0" not in token_text
    # But the summary footer "근거 N건 첨부됐습니다" IS in the visible
    # token stream so the user knows the toggle is there.
    assert "근거 2건" in token_text
    assert "부재 #5" in token_text


def test_orchestrator_hybrid_discards_too_long_prefix(monkeypatch):
    """Prefix that exceeds _MAX_PREFIX_CHARS is treated the same as a
    fabrication — the LLM is rambling beyond '1-2 sentences', and
    silent discard is safer than letting a long generation through
    unvetted."""
    import asyncio
    import json as _json
    from typing import AsyncIterator
    from core.chat.llm.base import BaseLLMProvider
    from core.chat.orchestrator import ChatOrchestrator, _MAX_PREFIX_CHARS
    from core.chat.tool_registry import ToolRegistry
    from core.chat.tools.inspect import (
        GET_ANALYSIS_SUMMARY_TOOL,
        INSPECT_SELECTION_TOOL,
    )

    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.delenv("VOYAGEAI_API_KEY", raising=False)
    monkeypatch.delenv("KDS_RAG_INDEX_PATH", raising=False)

    aid = _seed_context(5, status="NG", ratio_shear=1.3)
    LONG = "X" * (_MAX_PREFIX_CHARS + 50)

    class _RamblingProvider(BaseLLMProvider):
        name = "rambler"

        async def request_tool_call(self, *, messages, tools, temperature=None):
            return None

        async def stream_tokens(self, *, messages, temperature=None) -> AsyncIterator[str]:
            yield LONG

    orch = ChatOrchestrator(_RamblingProvider(), registry=ToolRegistry(
        [INSPECT_SELECTION_TOOL, GET_ANALYSIS_SUMMARY_TOOL,
         EXPLAIN_MEMBER_COMPLIANCE_TOOL],
        enabled_groups=frozenset({"inspect", "summary", "kds"}),
    ))

    async def _collect():
        return [
            ln async for ln in orch.run_turn(
                session={"analysis_id": aid, "history": []},
                user_message="5번 부재 왜 NG?",
            )
        ]

    lines = asyncio.run(_collect())
    events = [_json.loads(ln) for ln in lines if ln.strip()]
    combined = "".join(e["text"] for e in events if e["type"] == "token")
    assert LONG not in combined  # rambling discarded
    assert "부재 #5" in combined  # deterministic block intact


def test_pop_mandatory_response_strips_evidence_from_history():
    """Direct unit test on the helper: returns ``(summary, collapsible)``
    and removes ``kds_evidence`` from the history entry so the LLM that
    runs next for the prefix can't see the quotes (and therefore can't
    paraphrase them into fabrications)."""
    import json as _json
    from core.chat.orchestrator import _pop_mandatory_response

    history = [
        {"role": "user", "content": "왜 NG?"},
        {
            "role": "tool",
            "name": "explain_member_compliance",
            "content": _json.dumps({
                "member_summary": {"member_id": 5, "status": "NG"},
                "kds_evidence": [
                    {"doc_id": "AISC 360-22", "clause": "G2", "quote": "Vn = 0.6 Fy Aw Cv1"},
                ],
                "warnings": ["aisc_temporary_reference: ..."],
                "mandatory_response_summary": "summary text",
                "mandatory_response_collapsible": "deterministic evidence block",
            }),
        },
    ]
    summary, collapsible = _pop_mandatory_response(history)
    assert summary == "summary text"
    assert collapsible == "deterministic evidence block"
    # kds_evidence stripped, member_summary + warnings preserved
    sanitized = _json.loads(history[-1]["content"])
    assert "kds_evidence" not in sanitized
    assert "mandatory_response_summary" not in sanitized
    assert "mandatory_response_collapsible" not in sanitized
    assert sanitized["member_summary"]["member_id"] == 5
    assert sanitized["warnings"]  # warnings stay (no clause text)


def test_pop_mandatory_response_legacy_single_field_still_works():
    """Backwards-compat: callers that still emit the old single
    ``mandatory_response`` field get treated as summary-only (no
    collapsible). Lets us add the field gradually to other tools."""
    import json as _json
    from core.chat.orchestrator import _pop_mandatory_response

    history = [
        {
            "role": "tool",
            "name": "some_other_tool",
            "content": _json.dumps({
                "mandatory_response": "legacy single text",
            }),
        },
    ]
    summary, collapsible = _pop_mandatory_response(history)
    assert summary == "legacy single text"
    assert collapsible is None


def test_orchestrator_streams_normally_when_no_mandatory_response(monkeypatch):
    """Negative: when the tool does NOT carry mandatory_response (e.g.
    a normal inspect_selection turn), the LLM IS asked to narrate."""
    import asyncio
    import json as _json
    from typing import AsyncIterator
    from core.chat.llm.base import BaseLLMProvider
    from core.chat.orchestrator import ChatOrchestrator
    from core.chat.tool_registry import ToolRegistry
    from core.chat.tools.inspect import (
        GET_ANALYSIS_SUMMARY_TOOL,
        INSPECT_SELECTION_TOOL,
    )

    aid = _seed_context(2, status="OK")

    class _RecordingProvider(BaseLLMProvider):
        name = "recording"

        def __init__(self):
            self.stream_called = False

        async def request_tool_call(self, *, messages, tools, temperature=None):
            return None

        async def stream_tokens(
            self, *, messages, temperature=None,
        ) -> AsyncIterator[str]:
            self.stream_called = True
            yield "this is the LLM's narration"

    provider = _RecordingProvider()
    registry = ToolRegistry(
        [INSPECT_SELECTION_TOOL, GET_ANALYSIS_SUMMARY_TOOL],
        enabled_groups=frozenset({"inspect", "summary"}),
    )
    orch = ChatOrchestrator(provider, registry=registry)

    async def _collect():
        return [
            ln async for ln in orch.run_turn(
                session={"analysis_id": aid, "history": []},
                user_message="2번 부재 정보 보여줘",
            )
        ]

    asyncio.run(_collect())
    assert provider.stream_called is True, (
        "orchestrator bypassed LLM stream on a normal (no-mandatory) tool "
        "result — bypass condition too loose"
    )
