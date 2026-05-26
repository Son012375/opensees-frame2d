"""Diagnostic LLM provider used when no real backend is available.

Returns a fixed message so the UI surfaces "the LLM is unconfigured"
instead of failing silently. Used in three places:

  * Tests — keeps the chat router exercisable without a live Ollama.
  * ``CHAT_LLM_PROVIDER`` unset or set to ``noop`` — local dev default.
  * ``CHAT_LLM_PROVIDER=ollama`` but provider construction failed (e.g.
    daemon down, missing model). The chat router catches the init
    exception and builds a Noop with a context-specific ``message`` so
    the user sees *why* the chat fell back instead of a generic
    placeholder.
"""
from __future__ import annotations

from typing import AsyncIterator, Optional

from core.chat.llm.base import BaseLLMProvider


class NoopProvider(BaseLLMProvider):
    name = "noop"

    REPLY = (
        "[챗봇 LLM 미설정] 백엔드는 응답하고 있지만 LLM provider가 연결되지 "
        "않았습니다. CHAT_LLM_PROVIDER 환경변수 + OLLAMA_BASE_URL/OLLAMA_MODEL "
        "을 확인하세요."
    )

    def __init__(self, message: Optional[str] = None):
        # Falsy values (None or "") keep the default diagnostic message.
        # Any non-empty string overrides — used by the chat router to
        # surface the actual provider init failure ("Ollama 초기화 실패: ...").
        self._message = message if message else self.REPLY

    async def stream_tokens(self, *, messages: list[dict]) -> AsyncIterator[str]:
        # Single-chunk yield — same contract as a real streaming provider
        # without introducing artificial pseudo-tokens that would confuse
        # token-count metrics.
        yield self._message
