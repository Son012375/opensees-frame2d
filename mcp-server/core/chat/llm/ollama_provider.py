"""Ollama HTTP provider for the chat orchestrator (Phase A.3).

Talks to a local or LAN ``ollama serve`` (default ``http://localhost:11434``).
Implements the two :class:`BaseLLMProvider` hooks the orchestrator drives:

  * :meth:`request_tool_call` — non-streaming ``/api/chat`` POST with the
    registry's OpenAI/Ollama-shaped tool schemas. Returns the first
    ``tool_call`` the model produces, or ``None`` if it produced a final
    text response instead. Non-streaming is deliberate: Ollama splits
    ``tool_call`` JSON across SSE chunks unpredictably (Codex plan rev2
    P0-1) and the orchestrator only enters the tool loop in this code
    path, so we don't pay any UX cost.
  * :meth:`stream_tokens` — streaming ``/api/chat`` POST that yields
    ``message.content`` chunks one at a time as the model emits them.
    Matches the look-and-feel of other chat UIs.

Environment
-----------
``OLLAMA_BASE_URL`` (default ``http://localhost:11434``)
``OLLAMA_MODEL``    (default ``qwen2.5:14b`` — see plan §A.3 model choice)
``OLLAMA_TIMEOUT_S`` (default ``120``)
``OLLAMA_TEMPERATURE`` (default ``0.1``) — base temperature applied to
    both /api/chat calls. Tuned for *factual* conversation: the chat tools
    return exact data (member story, ratio, drift) and the LLM's only job
    is to translate that JSON into Korean. Low temperature stops qwen2.5
    from drifting (e.g. answering "1층" for every member regardless of the
    actual ``info.story`` value, or leaking Chinese phrases).

    Callers that need creative prose (future: result-report generation
    where varied phrasing is desirable) should pass ``temperature=`` per
    call instead of bumping this default. ``request_tool_call`` always
    benefits from low temp because tool routing is a discrete decision;
    only the final-answer ``stream_tokens`` call should ever go higher.

A failed transport (daemon down, model not pulled, network unreachable)
raises :class:`OllamaUnavailableError` which the chat router catches at
init time and converts to a Noop fallback so the route stays up.

Test hooks
----------
``client_factory`` callable can be injected to swap the live ``httpx``
client for one backed by :class:`httpx.MockTransport`. Used by
``tests/test_ollama_provider.py`` to script Ollama responses without
requiring a running daemon.
"""
from __future__ import annotations

import json
import logging
import os
from typing import AsyncIterator, Callable, Optional

import httpx

from .base import BaseLLMProvider, ToolCall


logger = logging.getLogger(__name__)


class OllamaUnavailableError(RuntimeError):
    """Raised when the Ollama daemon is unreachable or returns an error.

    The chat router treats this as a soft failure: it logs and falls
    back to :class:`NoopProvider` so the route keeps serving the wire
    contract instead of 500-ing.
    """


def _default_client_factory(timeout_s: float) -> Callable[[], httpx.AsyncClient]:
    return lambda: httpx.AsyncClient(timeout=timeout_s)


class OllamaProvider(BaseLLMProvider):
    """Speaks the Ollama ``/api/chat`` protocol.

    Two-call surface keeps the orchestrator's tool loop and final-answer
    stream cleanly separated. Both calls re-use the configured base URL,
    model name, and timeout.
    """

    name = "ollama"

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: Optional[float] = None,
        temperature: Optional[float] = None,
        client_factory: Optional[Callable[[], httpx.AsyncClient]] = None,
    ):
        self.base_url = (
            base_url or os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434"
        ).rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL") or "qwen2.5:14b"
        self.timeout_s = float(
            timeout_s if timeout_s is not None else os.environ.get("OLLAMA_TIMEOUT_S", "120")
        )
        # Low default so qwen2.5 sticks to the system prompt + reads
        # tool_result fields verbatim. Higher prose-friendly values can
        # be passed per call (see _ollama_options docstring).
        self.temperature = float(
            temperature if temperature is not None
            else os.environ.get("OLLAMA_TEMPERATURE", "0.1")
        )
        self._client_factory = client_factory or _default_client_factory(self.timeout_s)

    def _ollama_options(self, temperature: Optional[float] = None) -> dict:
        """Per-request sampler options passed in both /api/chat payloads.

        Pass ``temperature=`` to override the instance default for a
        single call — used by future report-generation flows that want
        more varied prose than the factual chat default. ``None`` (no
        override) means "use whatever was configured at construction".

        Kept as one helper so future knobs (top_p, num_ctx, ...) can be
        added without touching two call sites.
        """
        return {
            "temperature": temperature if temperature is not None else self.temperature,
        }

    # ------------------------------------------------------------------
    # Public BaseLLMProvider hooks
    # ------------------------------------------------------------------

    async def request_tool_call(
        self,
        *,
        messages: list[dict],
        tools: list[dict],
        temperature: Optional[float] = None,
    ) -> Optional[ToolCall]:
        """Ask Ollama whether the next move is a tool call.

        Returns the first :class:`ToolCall` in the response, or ``None``
        if the model produced a plain-text answer (orchestrator then
        falls through to :meth:`stream_tokens`).

        ``temperature`` overrides ``self.temperature`` for this call. Tool
        routing is a discrete decision so callers rarely want anything
        but the default low value — exposed for symmetry with
        :meth:`stream_tokens`.
        """
        # No tools available → skip the round trip entirely. Saves a
        # network hop when CHAT_TOOLS_ENABLED is empty.
        if not tools:
            return None

        payload = {
            "model": self.model,
            "messages": _to_ollama_messages(messages),
            "tools": tools,
            "stream": False,
            "options": self._ollama_options(temperature),
        }
        try:
            async with self._client_factory() as client:
                resp = await client.post(f"{self.base_url}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except (httpx.HTTPError, OSError) as exc:
            raise OllamaUnavailableError(
                f"Ollama tool-round request failed: {type(exc).__name__}: {exc}"
            ) from exc

        return _parse_first_tool_call(data)

    async def stream_tokens(
        self,
        *,
        messages: list[dict],
        temperature: Optional[float] = None,
    ) -> AsyncIterator[str]:
        """Yield ``message.content`` text as Ollama streams the answer.

        Ollama's streaming format is NDJSON — one JSON object per line.
        We skip the final ``{"done": true}`` line and any malformed
        lines (network can occasionally split a UTF-8 boundary).

        ``temperature`` overrides ``self.temperature`` for this single
        call. Default behaviour is low-temp factual narration; future
        report-generation flows can pass e.g. ``temperature=0.5`` to get
        more varied prose without mutating the provider's default state.
        """
        payload = {
            "model": self.model,
            "messages": _to_ollama_messages(messages),
            "stream": True,
            "options": self._ollama_options(temperature),
        }
        try:
            async with self._client_factory() as client:
                async with client.stream(
                    "POST", f"{self.base_url}/api/chat", json=payload,
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            logger.debug("Ollama stream: skipping non-JSON line: %r", line)
                            continue
                        text = (chunk.get("message") or {}).get("content") or ""
                        if text:
                            yield text
                        if chunk.get("done"):
                            return
        except (httpx.HTTPError, OSError) as exc:
            raise OllamaUnavailableError(
                f"Ollama stream request failed: {type(exc).__name__}: {exc}"
            ) from exc


# ----------------------------------------------------------------------
# Wire-format adapters — kept module-level for direct unit testing
# ----------------------------------------------------------------------

def _to_ollama_messages(messages: list[dict]) -> list[dict]:
    """Convert orchestrator-internal messages to the Ollama wire format.

    The orchestrator stores tool-result entries as ``{"role": "tool",
    "name": ..., "content": ...}`` (the OpenAI shape we keep
    provider-agnostic). Ollama's ``/api/chat`` expects the field to be
    called ``tool_name`` for tool messages — that's what lets qwen2.5
    et al. tie the result back to the originating ``tool_call``.

    Non-tool messages pass through unchanged so the orchestrator never
    has to know which provider it's talking to.
    """
    out: list[dict] = []
    for m in messages:
        if m.get("role") == "tool" and "name" in m:
            converted = {k: v for k, v in m.items() if k != "name"}
            converted["tool_name"] = m["name"]
            out.append(converted)
        else:
            out.append(m)
    return out


# ----------------------------------------------------------------------
# Response parsing — kept module-level for direct unit testing
# ----------------------------------------------------------------------

def _parse_first_tool_call(response: dict) -> Optional[ToolCall]:
    """Extract the first tool_call from an Ollama /api/chat response.

    Ollama's tool_call schema:

        {"message": {"tool_calls": [{"function": {"name": ..., "arguments": {...}}}, ...]}}

    But two real-world wrinkles:

      1. ``arguments`` can come back as a JSON-encoded string (Ollama's
         OpenAI-compatible mode does this for older clients). Parse it
         transparently so the orchestrator always sees a dict.
      2. The whole ``tool_calls`` array can be missing or empty when the
         model decided to answer in plain text — return ``None`` so the
         orchestrator drops out of the loop.
    """
    msg = response.get("message") or {}
    calls = msg.get("tool_calls") or []
    if not calls:
        return None

    first = calls[0]
    function = first.get("function") if isinstance(first, dict) else None
    function = function or {}

    name = function.get("name")
    if not name:
        return None

    arguments = function.get("arguments")
    if isinstance(arguments, str):
        # Ollama sometimes returns the args as a JSON string; an empty
        # string means "no args" rather than a parse failure.
        if not arguments.strip():
            arguments = {}
        else:
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                logger.warning(
                    "Ollama tool_call arguments not parseable JSON: %r", arguments,
                )
                arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}

    return ToolCall(name=name, arguments=arguments)
