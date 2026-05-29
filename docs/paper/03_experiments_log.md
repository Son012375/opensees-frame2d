# Experiments Log

This file records experiment evidence in a form that can later be converted
into a paper method/results section. Keep entries factual and include the
limitation with the result.

## E1. Provider-visible history excludes evidence quotes

- Date: 2026-05-29
- PR/commit: PR #2, later merged to main as `4f6460c`
- Method: run a hybrid KDS compliance chat turn, then inspect provider messages
  derived from session history.
- Metric: presence or absence of evidence quote body in provider-visible
  content.
- Result: quote/collapsible body absent from provider-visible messages; summary
  remains present.
- Test evidence: `tests/test_chat_kds_tool.py`
- Supports: strong architectural invariant.
- Limitation: proves exclusion for current orchestration path and test fixtures;
  should be maintained with regression tests as the chat stack changes.

## E2. KDS-RAG retrieval golden-set benchmark

- Date: 2026-05-29
- Source document: `docs/kds_rag_benchmark.md`
- Dataset: `data/kds_eval/golden_set.json`, five scenarios covering shear,
  flexure, interaction, axial, and drift.
- Deterministic retriever: InMemory over `data/kds_sample`.
- Live retriever: Voyage index with KDS 14 31 10 and KDS 41 17 00 sample corpus.
- Metrics: recall@1, recall@3, precision@1, MRR.
- Result: deterministic 5/5/5; live 5/5/5, MRR 1.000; AISC proxy 0/5.
- Supports: lightweight regression evidence for routing and corpus ingestion.
- Limitation: small golden set; not enough for broad retrieval-quality claims.

## E3. Axial retrieval precision fix

- Date: 2026-05-29
- Source document: `docs/kds_rag_benchmark.md`
- Observation: axial query initially ranked the interaction chunk above the
  compression chunk in live Voyage retrieval.
- Fix: add compression-buckling discriminators to the axial query seed.
- Result: axial compression clause ranked first in the live run.
- Supports: query-seed design matters for embedding retrieval.
- Limitation: axial tension and compression are not yet separate issue types
  because the chat tool only has absolute axial ratio, not sign.

## E4. qwen2.5:14b prefix consistency probe

- Date: 2026-05-29
- Source document: `docs/kds_prefix_consistency.md`
- Model: qwen2.5:14b, temperature approximately 0.1.
- Dataset: fixed shear-NG member summary, 12 user phrasings, 3 repeats.
- Metric: sanitizer classification: kept, empty, fabricated, toolong.
- Result: kept 36/36; empty 0; fabricated 0; toolong 0.
- Supports: current prompt and member summary produce stable clause-free
  prefixes in this fixed condition.
- Limitation: not evidence for weakening the sanitizer; not a broad model
  evaluation.

## E5. R3 provenance audit log

- Date: 2026-05-29
- PR/commit: PR #2, later merged to main as `4f6460c`
- Method: after KDS retrieval, write audit record with analysis, member, ratios,
  governing issue type, query, evidence metadata, quote text, and warnings.
- Result: audit records can be queried by analysis/member/turn while provider
  history remains quote-free.
- Supports: reproducibility and provenance-chain claims.
- Limitation: initial PR #2 version was research/debug grade until PR #4 added
  retention and access controls.

## E6. All-violations summary

- Date: 2026-05-29
- PR/commit: PR #3, merged to main as `4653694`
- Method: list all ratios above 1.0 in descending order and mark governing
  ratio in summary and audit trigger.
- Result: multiple simultaneous NG causes are preserved in user answer and
  provenance record.
- Supports: practical explainability improvement.
- Limitation: retrieval remains single-governing-query by design; not a
  multi-query evidence retrieval system.

## E7. Operational audit hardening

- Date: 2026-05-29
- PR/commit: PR #4, merged to main as `ee50bf8`
- Method: add JSONL size rotation, retention sweep, operator/session access
  control, and quote-tier redaction.
- Test result: 80 targeted tests passed; full suite 721 passed except a
  pre-existing unrelated matplotlib/Tk flaky failure that passed in isolation.
- Supports: operational-readiness claim for the audit endpoint.
- Limitation: `DEMO_AUTH_TOKEN` is a shared operator token, not a full user
  identity system.

## Entry template

```md
## E?. Title

- Date:
- PR/commit:
- Scenario or dataset:
- Method:
- Metric:
- Result:
- Test evidence:
- Supports:
- Limitation:
```
