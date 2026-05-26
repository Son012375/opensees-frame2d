"""Diagnostic LLM provider used when no real backend is configured.

Returns a fixed message so the UI surfaces "the LLM is unconfigured"
instead of failing silently. Also the default provider in tests — keeps
the chat router exercisable without an Ollama server.
"""
from __future__ import annotations

from typing import AsyncIterator

from core.chat.llm.base import BaseLLMProvider


class NoopProvider(BaseLLMProvider):
    name = "noop"

    REPLY = (
        "[챗봇 LLM 미설정] 백엔드는 응답하고 있지만 LLM provider가 연결되지 "
        "않았습니다. CHAT_LLM_PROVIDER 환경변수 + OLLAMA_BASE_URL/OLLAMA_MODEL "
        "을 확인하세요."
    )

    async def stream_tokens(self, *, messages: list[dict]) -> AsyncIterator[str]:
        # Single-chunk yield — same contract as a real streaming provider
        # without introducing artificial pseudo-tokens that would confuse
        # token-count metrics.
        yield self.REPLY
