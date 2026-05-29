# Results Summary

Use this file as the compact source for result tables in a paper draft.

## Current results

| Area | Result | Evidence | Claim strength |
|------|--------|----------|----------------|
| History isolation | Evidence quote bodies absent from provider-visible messages | `tests/test_chat_kds_tool.py` | Strong invariant |
| Retrieval golden set | Deterministic 5/5/5; live 5/5/5; MRR 1.000 | `docs/kds_rag_benchmark.md` | Lightweight regression |
| AISC proxy removal | AISC temporary proxy 0/5 in R1 benchmark | `docs/kds_rag_benchmark.md` | Strong for tested set |
| Prefix consistency | kept 36/36, empty 0, fabricated 0, toolong 0 | `docs/kds_prefix_consistency.md` | Fixed-condition probe |
| Provenance audit | ratio -> issue_type -> query -> evidence recorded | `chat_audit_log.py`, tests | Strong system feature |
| Audit hardening | rotation, retention, session/operator access, default quote redaction | PR #4, `DEPLOY.md` | Strong implementation feature |
| All-violations summary | all ratios above 1.0 shown and audited | PR #3 tests | Strong implementation feature |

## Interpretation

The strongest result is architectural: the system can show evidence to the
user and store it for audit without placing quote bodies back into the LLM's
provider-visible history. This supports a safer engineering-chat design pattern.

The retrieval and prefix results should be presented as regression evidence.
They show that the current routing and prompt behave well under controlled
conditions, but they do not establish general model or corpus performance.

## Paper-safe wording

Use:

> In the tested Phase D implementation, verbatim evidence quotes are excluded
> from provider-visible chat history while remaining available through a
> separately access-controlled audit log.

Avoid:

> The system eliminates hallucinations.

Use:

> The five-case golden set passed both deterministic and live retrieval checks.

Avoid:

> KDS retrieval accuracy is 100%.

Use:

> qwen2.5:14b produced valid prefixes in a fixed shear-NG probe, with the
> sanitizer retained as the hard guard.

Avoid:

> qwen2.5:14b is reliable enough without sanitization.
