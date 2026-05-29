# Paper Material Index

This directory is a working notebook for future papers about the Phase D
KDS-RAG chatbot. It is not the paper itself. Its job is to preserve claims,
evidence, experiments, limitations, and figure ideas while the final thesis
angle is still undecided.

## How to use this folder

When a new experiment, PR review, or design change lands, add one short entry
to the relevant file instead of rewriting the whole story.

Minimum record for any result:

- date
- git commit or PR
- exact command or scenario
- dataset or fixed fixture
- metric
- result
- limitation
- whether the result can support a paper claim

## Files

| File | Purpose |
|------|---------|
| `00_research_questions.md` | Candidate paper angles and claim boundaries |
| `01_system_overview.md` | System architecture and component responsibilities |
| `02_design_decisions.md` | Design choices that may become contribution claims |
| `03_experiments_log.md` | Experiment ledger with method/result/limitation |
| `04_results_summary.md` | Short, citable result table and interpretation |
| `05_failure_cases_and_fixes.md` | Failure mode -> fix -> test chains |
| `06_limitations.md` | What the current system cannot yet claim |
| `07_figures_to_make.md` | Figure/table candidates for the eventual paper |
| `appendix_audit_schema.md` | Audit log schema, access policy, and provenance notes |

## Existing source documents

Use these as primary project evidence:

- `docs/kds_rag_benchmark.md`
- `docs/kds_prefix_consistency.md`
- `docs/kds_rag_llm_guardrails.md`
- `docs/kds_rag_sample_corpus.md`
- `docs/kds_rag_smoke.md`
- `docs/phase_d_review_roadmap.md`
- `DEPLOY.md`
- `data/kds_eval/golden_set.json`
- `scripts/kds_rag_benchmark.py`
- `scripts/kds_prefix_consistency.py`

## Current safest paper stance

The safest current claim is not "the chatbot makes correct design decisions."
The safer claim is:

> A structural-design chat interface can reduce citation hallucination risk by
> keeping code evidence out of provider-visible LLM history, rendering citations
> deterministically on the server, and preserving a separate provenance audit
> trail from member ratios to retrieved code clauses.

The current retrieval and prefix numbers are useful supporting evidence, but
they are lightweight regression evidence, not broad performance proof.
