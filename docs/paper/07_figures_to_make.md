# Figures and Tables to Make

## Figure 1. System architecture

Purpose:

Show how the user question, orchestrator, deterministic tool, retriever,
evidence renderer, provider-visible history, and audit log are separated.

Source:

- `docs/paper/01_system_overview.md`

Paper message:

The LLM is not the citation author.

## Figure 2. P1 evidence isolation

Purpose:

Before/after diagram showing evidence quote re-entry risk and the fixed path.

Content:

- before: `EVENT_COLLAPSIBLE` text appended to assistant history
- after: collapsible sent to UI only; history stores prefix plus summary

Paper message:

Multi-turn citation leakage is blocked by history construction.

## Figure 3. Provenance chain

Purpose:

Show the chain:

`member ratio -> governing ratio -> issue_type -> KDS query -> evidence clause`

Source:

- `appendix_audit_schema.md`
- `chat_audit_log.py`

Paper message:

The system preserves reproducibility outside the LLM context.

## Figure 4. Audit access matrix

Purpose:

Summarize operator/session/anonymous access and quote redaction behavior.

Rows:

- operator, `include_quotes=false`
- operator, `include_quotes=true`
- owning session, `include_quotes=false`
- owning session, `include_quotes=true`
- no token and no owned session
- wrong session

Paper message:

Provenance is queryable without making quote exposure the default.

## Table 1. Failure modes and mitigations

Source:

- `05_failure_cases_and_fixes.md`

Columns:

- failure mode
- risk
- fix
- test
- remaining limitation

## Table 2. Retrieval benchmark

Source:

- `docs/kds_rag_benchmark.md`
- `data/kds_eval/golden_set.json`

Columns:

- case
- issue type
- expected clause
- top-1 clause
- rank
- hit@1
- hit@3

## Table 3. Claim tiers

Source:

- `00_research_questions.md`

Purpose:

Separate strong architectural claims from preliminary evaluation results.

## Screenshots to capture later

- chat answer with summary and collapsed evidence
- expanded evidence block
- audit endpoint response with quote redacted
- audit endpoint response with quote included for owning session or operator

## Missing data for future figures

- larger golden set results
- expert-labeled retrieval accuracy
- user-study comparison of inline evidence vs collapsible evidence
- latency or token-cost measurements for prefix-only vs full-answer LLM modes
