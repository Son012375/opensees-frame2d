"""LLM provider implementations for the chat router.

Phase A.1 ships :class:`NoopProvider`. :class:`OllamaProvider` lands
in Phase A.3 — its ``stream_tokens`` will speak to a local
``ollama serve`` instance and the tool-round non-stream JSON path is
added on top of that contract in Phase B.
"""
