# Appendix: Evidence Audit Schema

The audit log exists to preserve provenance without placing verbatim evidence
quotes back into provider-visible LLM history.

## Record shape

Representative fields:

```json
{
  "audit_id": "uuid-like id",
  "created_at": 0.0,
  "expires_at": 0.0,
  "analysis_id": "analysis id",
  "member_id": 5,
  "turn": 1,
  "session_id": "chat session id",
  "member": {
    "member_id": 5,
    "type": "column",
    "section": "H-...",
    "material": "SS275",
    "story": "..."
  },
  "trigger": {
    "status": "NG",
    "governing_ratio": "shear",
    "issue_type": "shear_exceeded",
    "ratios": {
      "interaction": 0.42,
      "shear": 1.28,
      "axial": 0.0,
      "bending": 0.0
    },
    "exceeded": [
      {"name": "shear", "value": 1.28, "governing": true}
    ]
  },
  "query": {
    "query_text": "...",
    "topic": "member_shear",
    "limit_state": "shear_strength"
  },
  "rag_used": true,
  "evidence": [
    {
      "doc_id": "KDS 14 31 10",
      "clause": "4.3.2.1.2",
      "title": "...",
      "quote": "verbatim quote",
      "score": 0.0
    }
  ],
  "warnings": []
}
```

## Storage policy

In-memory view:

- TTL: 30 minutes
- cap: 200 records per analysis
- purpose: live debugging and endpoint reads

Durable JSONL:

- default path: `data/chat_audit/evidence_audit.jsonl`
- path override: `CHAT_AUDIT_LOG_PATH`
- size cap: `CHAT_AUDIT_MAX_BYTES`, default 50 MB
- backup count: `CHAT_AUDIT_BACKUP_COUNT`, default 5
- retention: `CHAT_AUDIT_RETENTION_DAYS`, default 90
- retention sweep deletes aged numeric-suffix backups only
- live base file is not deleted by retention

## Access policy

| Caller | Metadata | Quote text |
|--------|----------|------------|
| Operator token, `include_quotes=false` | allowed | redacted |
| Operator token, `include_quotes=true` | allowed | allowed |
| Owning session, `include_quotes=false` | allowed | redacted |
| Owning session, `include_quotes=true` | allowed | allowed |
| Unknown session | denied | denied |
| Wrong session | denied | denied |
| No token and no session in token-enabled deployment | denied | denied |

`DEMO_AUTH_TOKEN` unset means local/dev open mode. Shared deployments must set
it.

## Why this is not a P1 regression

P1 prevents evidence quote bodies from entering provider-visible chat history.
The audit log is a separate store. Returning quotes through the audit endpoint
to an authorized operator or owning session does not feed those quotes to the
LLM provider.

## Paper relevance

The schema supports a reproducible chain:

`analysis_id + member_id + turn -> ratios -> issue_type -> query -> evidence`

This chain is the main bridge between engineering result, retrieval behavior,
and user-visible citation evidence.
