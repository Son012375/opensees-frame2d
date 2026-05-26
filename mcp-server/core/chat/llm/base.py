"""LLM provider abstract base.

Phase A only needs the final-answer streaming path. The tool-call
non-streaming method comes in Phase B alongside ``OllamaProvider``;
keeping it off the base class for now means ``NoopProvider`` doesn't
have to stub a method it can't honour.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator


class BaseLLMProvider(ABC):
    """Minimal contract every chat LLM provider must satisfy.

    Subclasses provide ``stream_tokens`` for the final natural-language
    response. ``messages`` follows the OpenAI/Ollama convention:
    ``[{"role": "system|user|assistant", "content": "..."}, ...]``.
    """

    #: Stable identifier for logging / status events. Subclasses override.
    name: str = "base"

    @abstractmethod
    def stream_tokens(self, *, messages: list[dict]) -> AsyncIterator[str]:
        """Yield response tokens one at a time.

        Implementations are async generators (``async def`` + ``yield``)
        and may yield a single chunk if the upstream API doesn't stream.
        """
