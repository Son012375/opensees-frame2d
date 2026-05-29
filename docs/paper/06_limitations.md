# Limitations

This file protects the eventual paper from overclaiming.

## Evaluation limitations

- The R1 retrieval benchmark uses a five-case golden set. It is useful for
  regression and routing checks, not for general accuracy claims.
- The live retrieval result depends on the current sample corpus and Voyage
  index state.
- Prefix consistency was measured on one fixed shear-NG member summary. It does
  not cover every issue type, language style, model, or temperature.
- There is no controlled user study yet for trust, comprehension, speed, or
  design-review quality.
- There is no independent structural-engineer labeling study yet.

## System limitations

- The chatbot is a review assistant, not a final compliance authority.
- The deterministic design-check layer still defines the ratios available to
  the chat tool; missing or simplified checks cannot be repaired by RAG.
- Retrieval routes by the governing issue type. It does not yet run separate
  evidence queries for every exceeded ratio.
- Axial tension and compression are not fully separated because the current
  ratio interface does not carry enough sign/limit-state detail.
- Audit storage is JSONL plus in-memory TTL, not a full governed audit database.
- Shared deployment currently depends on `DEMO_AUTH_TOKEN`, not per-user
  identity.

## Corpus limitations

- The current KDS sample corpus is small and targeted.
- Clause coverage should be expanded before broad KDS compliance claims.
- Any OCR/PDF ingestion pipeline needs separate validation for citation
  accuracy, clause boundaries, and table extraction quality.

## Security and governance limitations

- Quote redaction and session ownership reduce exposure but are not a full
  privacy program.
- Retention deletes aged rotated backups, not line-level records by
  `analysis_id`.
- Operational deployments must set `DEMO_AUTH_TOKEN`; unset means local/dev
  open mode.
- Audit records may contain sensitive structural metadata even when quotes are
  redacted.

## Statements to avoid

- "The system proves KDS compliance."
- "The LLM never hallucinates."
- "Retrieval accuracy is 100%."
- "The audit log is fully secure."
- "The current results generalize to all KDS clauses."

## Safer statements

- "The system constrains the LLM to narration and renders citations
  deterministically."
- "The current tests lock the invariant that quote bodies do not enter
  provider-visible history."
- "The five-case benchmark is a lightweight regression set for current routing
  behavior."
- "The audit endpoint is hardened for shared deployments using token/session
  access and default quote redaction."
