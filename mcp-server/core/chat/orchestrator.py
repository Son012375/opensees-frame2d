"""Per-turn chat orchestrator — Phase A.1 (no tool loop).

One ``run_turn`` call emits the NDJSON sequence for a single user
message: ``status`` → 1..N ``token`` → ``done`` (or ``error`` →
``done``). Tool calling arrives in Phase B; this module is intentionally
narrow so the route + stream + history-append contract can be locked in
first.

History is owned by the caller (chat_router holds it inside
``chat_session_cache``). We mutate it in place so the session's
conversation history grows without the router having to know about
message shape.
"""
from __future__ import annotations

import time
from typing import AsyncIterator, Optional

from core.chat.llm.base import BaseLLMProvider
from core.chat.streaming import (
    EVENT_DONE,
    EVENT_ERROR,
    EVENT_STATUS,
    EVENT_TOKEN,
    encode_event,
)


class ChatOrchestrator:
    def __init__(self, llm: BaseLLMProvider, *, max_history: int = 50):
        self.llm = llm
        self.max_history = max_history

    async def run_turn(
        self,
        *,
        history: list[dict],
        user_message: str,
        ui_context: Optional[dict] = None,
    ) -> AsyncIterator[str]:
        """Run one chat turn and yield NDJSON-encoded events.

        Phase A.1 path:
            status(provider) → token+ → done(rounds=0, total_tokens, ms_total)

        If the LLM raises, we emit a single ``error`` event and a
        terminating ``done`` (rounds=0) — the route never leaves the
        client without a terminator.
        """
        t0 = time.monotonic()

        # Append user message and trim oldest history if the cap is hit.
        # `ui_context` is reserved for Phase A.2's tool layer; capture it
        # on the user turn so the orchestrator can read it from history
        # later instead of threading it through every call.
        history.append({
            "role": "user",
            "content": user_message,
            "ui_context": ui_context or {},
        })
        if len(history) > self.max_history:
            del history[: len(history) - self.max_history]

        yield encode_event(EVENT_STATUS, {
            "message": "thinking",
            "provider": self.llm.name,
        })

        # Messages passed to the LLM exclude our private ``ui_context``
        # field — providers only understand role+content.
        provider_messages = [
            {"role": h["role"], "content": h["content"]}
            for h in history
        ]

        full_text = ""
        token_count = 0
        try:
            async for tok in self.llm.stream_tokens(messages=provider_messages):
                if not tok:
                    continue
                full_text += tok
                token_count += 1
                yield encode_event(EVENT_TOKEN, {"text": tok})
        except Exception as exc:  # noqa: BLE001 — provider crash is recoverable
            yield encode_event(EVENT_ERROR, {
                "message": str(exc) or type(exc).__name__,
                "code": "llm_failure",
            })
            ms_total = int((time.monotonic() - t0) * 1000)
            yield encode_event(EVENT_DONE, {
                "rounds": 0,
                "total_tokens": token_count,
                "ms_total": ms_total,
            })
            return

        history.append({"role": "assistant", "content": full_text})

        ms_total = int((time.monotonic() - t0) * 1000)
        yield encode_event(EVENT_DONE, {
            "rounds": 0,  # Phase A.1: no tool rounds yet
            "total_tokens": token_count,
            "ms_total": ms_total,
        })
