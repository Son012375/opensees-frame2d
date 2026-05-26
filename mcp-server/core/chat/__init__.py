"""Chat router internals — orchestrator, tool registry, LLM providers,
and the NDJSON event stream contract used by ``/api/v2/chat/messages``.

Phase 0 ships only :mod:`.streaming` (the wire format). The orchestrator,
tools, and LLM providers land in Phase A.
"""
