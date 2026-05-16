# KDS-RAG + LLM Guardrails

> **Status:** test-double phase. Real RAG retrieval, vector DB, and LLM
> calls are **not** implemented yet. The rules below are fixed so that
> when those layers do land, they don't quietly drift from the
> deterministic contract.

## 1. Scope

This document constrains how an LLM (and the surrounding RAG pipeline)
may use the recommendation data contract:

- `core.recommendation.schemas.StructuralIssue`
- `core.recommendation.schemas.RetrofitCandidate`
- `core.recommendation.schemas.CodeReference`
- `core.kds_rag.schemas.KDSChunk` / `KDSRetrievalQuery` / `KDSRetrievalResult`
- `core.kds_rag.schemas.CitationValidationResult`

The deterministic layers (recommendation extractor + candidate generator
+ citation validator) are the **source of truth** for numerical
judgement. The LLM is a *narrator*, not a judge.

## 2. Hard rules for the LLM

The LLM MUST NOT:

1. **Perform structural calculations.** No re-deriving D/C ratios, no
   replacing values from `evidence` / `proposed_change`, no inventing
   new ratios.
2. **Fill `proposed_change.to`.** The "to" section / material / system
   is left `null` on purpose until an optimizer or a human picks one.
   The LLM may *suggest* candidates verbally, but must say
   "needs engineer selection" and must not write into the field.
3. **Treat `requires_reanalysis: true` as advisory.** Every typed
   candidate carries this flag. The LLM's narrative MUST tell the user
   that any change requires a re-analysis pass.
4. **Cite refs that aren't `citation_ready`.** A `CodeReference` is
   citation-ready only when the citation validator
   (`core.kds_rag.citation_validator.validate_code_reference`) accepts
   it. Hint-only refs (with `query_hint` but no `source_url` /
   `chunk_id`) MUST be described as "근거 미확인" / "reference unverified".
5. **Truncate or paraphrase a long quote and present it as the quote.**
   If a chunk's text is too long (> `MAX_QUOTE_LEN = 400`), the
   pipeline leaves `quote = None`. The LLM must not fabricate a quote
   to fill that gap.
6. **Mix topics across refs.** If a candidate's `code_refs` contains a
   shear-strength ref and a story-drift ref, the LLM must keep their
   roles separate in the narration — never blend them.
7. **Present a candidate as a final design.** Every narration must
   end with "재해석 + 엔지니어 검토 필요" or equivalent. The data model
   already encodes this; the prose must match.

## 3. Hard rules for the RAG pipeline

The retrieval layer MUST:

1. **Never raise on empty match.** Return `KDSRetrievalResult(chunks=[],
   warnings=[…])` instead. The caller treats retrieval as best-effort.
2. **Honor `top_k`.** Real retrievers may return fewer chunks but never
   more than `top_k`.
3. **Preserve the query on the result.** Useful for audit trails and
   for the LLM to phrase "I searched for *X* and found …".
4. **Annotate chunks with topic / limit_state / material whenever
   known.** Cross-topic drift detection in
   `validate_code_reference` relies on these.
5. **Never write into `CodeReference.source_url` with an unverifiable
   URL.** If the chunk's `source_url` is unknown, leave it `None` and
   rely on `chunk_id`.

## 4. Citation-ready definition

A single `CodeReference` passes the citation guardrail iff
`validate_code_reference(ref, chunk_id=...)` returns `(True, "ok")`.
The current rule set:

1. `standard_id` is set.
2. `clause_id` OR `title` is set.
3. `source_url` OR `chunk_id` is set. The `chunk_id` may be passed
   explicitly as a keyword arg (the enrichment path does this) **or**
   stored on the ref as `ref.chunk_id` — the validator falls back to
   the ref's own value if the kwarg is omitted. An internal hint
   (`query_hint`, `topic`, …) is NOT enough.
4. If `quote` is set: `MIN_QUOTE_LEN <= len(quote) <= MAX_QUOTE_LEN`.
5. If both `ref.topic` and the matched chunk's `topic` are set, they
   must match. Same for `limit_state`.

### 4.1 `chunk_id` is an internal citation key

`CodeReference.chunk_id` is an **internal audit / provenance pointer**
back to the `KDSChunk` that produced the citation. It's not a URL and
must not be displayed to users as one. When `chunk_id` is the only
"source" present (no `source_url`), the LLM narration must phrase
attribution as "내부 인용 (chunk X)" / "internal reference (chunk X)",
never as a public link.

### 4.2 Batch-level summary semantics

The `kds_rag_summary` block exposes **four** boolean flags. Read them
carefully — they are NOT redundant:

| flag | meaning | true when |
|------|---------|-----------|
| `citation_ready` | At least one usable citation is attached. Says **nothing** about queries that came back empty. | `num_refs_attached > 0` |
| `all_queries_resolved` | Every query was answered AND every returned ref passed the guardrail. The strict "all green" flag. | `num_unresolved == 0` AND `num_refs_rejected == 0` |
| `has_unresolved_queries` | At least one query returned empty. | `num_unresolved > 0` |
| `has_rejected_refs` | At least one returned ref failed the guardrail. | `num_refs_rejected > 0` |

**LLM consequences:**

- `citation_ready=true` is permission to **cite the refs that are
  present**. It is NOT permission to claim "all sources verified".
- If `all_queries_resolved=false`, the narration MUST disclose which
  parts of the response remain unverified — typically by emitting a
  "근거 미확인 (unresolved)" / "근거 거부됨 (rejected)" disclaimer for
  the affected issues / candidates.
- The LLM should prefer the `all_queries_resolved` flag when deciding
  whether to phrase the answer with high confidence.

## 5. Examples

### 5.1 OK narration

> "부재 7 (column, H-300x300) 의 H1 상관비 1.42 > 1.0 이므로 단면 증가
> 후보가 자동 생성되었습니다. 적용 단면은 엔지니어 선정 + 재해석으로
> 확정해야 합니다. 근거: KDS 41 31 00 §H1 (chunk kds_41_31_00_h1_p72)."

### 5.2 NOT OK narration

> "엔진 분석 결과 H-350x350 으로 교체하면 D/C 0.9 가 됩니다." ← 계산
> 수행 / `proposed_change.to` 채움 / 재해석 언급 누락 — 모두 금지.

### 5.3 Unverified ref handling

If `kds_rag_summary.citation_ready` is `False` and `num_refs_attached`
is 0, the narration must say:

> "관련 KDS 조항 검색 결과를 확인하지 못해 근거를 첨부할 수 없었습니다.
> 엔지니어가 KDS 41 31 00 §H1 (조합응력 상관식) 을 직접 확인해 주세요."

## 6. Test hooks (current phase)

- `tests/test_kds_rag.py` exercises every guardrail listed above.
- The reference retriever is `InMemoryKDSRetriever` — pure Python,
  deterministic scoring, suitable for CI.
- `citation_validator` rejection reasons are part of the contract and
  greppable in logs (`ref_rejected: chunk=… reason=…`).

## 7. What we deliberately don't do yet

- ❌ Hit a real KDS API or vector DB
- ❌ Embed KDS source text in this repo
- ❌ Call an LLM
- ❌ Fill `proposed_change.to`
- ❌ Auto-attach KDS refs to `/api/v2/analyze` response (callers must
  opt in via `enrich_recommendation_payload_with_kds`)

These constraints are intentional and should not be relaxed without an
explicit design discussion.
