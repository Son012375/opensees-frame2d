"""LLM provider abstract base.

A provider does two jobs for the chat orchestrator:

1. ``request_tool_call`` (Phase A.2+) — given the conversation so far and
   a list of available tool schemas, decide whether to call a tool. Non-
   streaming because Ollama's tool_call JSON is unreliable when split
   across SSE chunks. Returning ``None`` means "no more tools, give me
   the final answer."
2. ``stream_tokens`` — yield the final natural-language reply token by
   token. Used after the tool loop terminates.

``messages`` follows the OpenAI/Ollama convention:
``[{"role": "system|user|assistant|tool", "content": "...", "name"?: ...}]``.

The default ``request_tool_call`` returns ``None`` so a provider that
doesn't speak tools (Noop, future text-only fallbacks) is fully usable
without overriding it. Tool-capable providers (Ollama, scripted test
doubles) override the default.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional


@dataclass(frozen=True)
class ToolCall:
    """One LLM-requested tool invocation.

    ``arguments`` is the already-parsed JSON object the LLM emitted. The
    orchestrator passes it straight to ``ToolRegistry.call_tool`` — the
    registry is responsible for validating against the tool's schema.
    """
    name: str
    arguments: dict


class BaseLLMProvider(ABC):
    """Minimal contract every chat LLM provider must satisfy."""

    #: Stable identifier for logging / status events. Subclasses override.
    name: str = "base"

    async def request_tool_call(
        self,
        *,
        messages: list[dict],
        tools: list[dict],
        temperature: Optional[float] = None,
    ) -> Optional[ToolCall]:
        """Decide whether to call a tool this round.

        Default: never request a tool — final answer streams immediately.
        Overridden by tool-capable providers (Ollama in A.3, scripted
        doubles in tests).

        ``temperature`` is an optional per-call sampling override. The
        orchestrator passes it based on the active tool's creativity
        hint (see :attr:`ToolSpec.creativity_hint`). Providers that
        ignore temperature must still accept the kwarg without erroring.
        """
        return None

    @abstractmethod
    def stream_tokens(
        self,
        *,
        messages: list[dict],
        temperature: Optional[float] = None,
    ) -> AsyncIterator[str]:
        """Yield response tokens one at a time.

        Implementations are async generators (``async def`` + ``yield``)
        and may yield a single chunk if the upstream API doesn't stream.

        ``temperature`` is an optional per-call sampling override — see
        :meth:`request_tool_call` for context. Providers without sampling
        control accept the kwarg and ignore it.
        """
