# Research Questions and Claim Boundaries

## Candidate paper angles

### A. Applied system paper

Working title:

> A KDS-RAG Design-Review Chatbot for Frame Analysis Results

Main question:

> Can structural analysis results be connected to code-grounded conversational
> review while keeping the LLM out of final numerical and citation authority?

Best evidence:

- deterministic design-check summary
- server-rendered evidence block
- KDS query routing by member status and issue type
- operator-safe evidence audit log

Risk:

- needs a usability or engineer-review study before claiming field usefulness.

### B. Anti-hallucination architecture paper

Working title:

> Provider-Isolated Evidence Rendering for Code-Grounded Engineering Chat

Main question:

> Does separating LLM narration from deterministic evidence rendering reduce
> opportunities for citation fabrication across multi-turn chat?

Best evidence:

- P1 fix: evidence quotes removed from provider-visible history
- prefix sanitizer: fabricated code references discarded
- collapsible evidence emitted only to the user UI
- audit store holds quotes outside LLM history

Risk:

- current tests prove an invariant, not a probabilistic reduction across many
  models and prompts.

### C. Provenance and reproducibility paper

Working title:

> From Member Ratios to Code Clauses: Provenance Logging for RAG-Assisted
> Structural Review

Main question:

> Can a chat-based design-review system preserve a reproducible chain from
> structural result to retrieved clause without exposing evidence to the LLM?

Best evidence:

- R3 audit schema
- session-bound access control
- quote redaction policy
- retention and rotation policy from PR #4

Risk:

- audit logging is operationally stronger now, but still lacks a long-term
  database, per-user identity, and deletion workflow.

### D. Retrieval-routing paper

Working title:

> Limit-State-Aware Query Routing for KDS Structural Code Retrieval

Main question:

> Does splitting issue types by governing limit state improve clause retrieval
> quality for KDS design review?

Best evidence:

- P3 vocabulary split: shear, axial, flexure, interaction, drift
- R1 deterministic and live golden-set benchmark
- axial precision fix through query-side discriminators

Risk:

- current golden set is small. This should be framed as a regression benchmark
  until expanded.

## Claim tiers

### Strong enough now

- The system keeps verbatim evidence quotes out of provider-visible chat
  history by construction.
- The user-facing evidence block is rendered deterministically by the server,
  not authored by the LLM.
- Audit records preserve the chain from analysis/member/ratio to query and
  retrieved evidence outside LLM history.
- The audit read endpoint now enforces operator or owning-session access and
  redacts quotes by default.

### Preliminary only

- The current KDS retrieval route performs well on the five-case golden set.
- qwen2.5:14b produced clean prefixes in the fixed shear-NG probe.
- The all-violations summary improves practical explainability.

### Do not claim yet

- Do not claim compliance decisions are fully automated.
- Do not claim general KDS retrieval accuracy from the five-case benchmark.
- Do not claim user trust or productivity improvement without a user study.
- Do not claim full security or privacy without real user identity and storage
  governance.
- Do not claim final engineering suitability without expert validation.
