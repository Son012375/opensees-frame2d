"""Per-turn chat orchestrator — Phase A.2 (tool loop minimal).

One ``run_turn`` call drives a single user message through this sequence:

    status (provider) → 0..N tool rounds → 1..N tokens → done

Each tool round emits ``tool_call`` and ``tool_result`` events and
appends a ``role=tool`` message to history so the next round / final
answer can read the result. The loop stops when the provider returns
``None`` from ``request_tool_call`` (i.e. "no more tools, give me the
final answer") or when ``max_rounds`` is hit.

History layout
--------------
``session['history']`` is the source of truth. Each entry is one of:

    {"role": "user", "content": str, "ui_context": dict}
    {"role": "assistant", "content": str}
    {"role": "tool", "name": str, "content": str}      # JSON string

The provider only sees role + content (+ name for tool). ``ui_context``
stays out of the provider stream — it's chat-router metadata that tools
can read from history but the LLM shouldn't fixate on.

Phase A.1 → A.2 changes
-----------------------
- Provider gets ``registry`` and ``max_rounds`` (default 5).
- ``run_turn`` signature switches from ``history=...`` to ``session=...``
  so tools can read ``session['analysis_id']`` etc.
- History is trimmed *once* after all appends (P3 fix from Codex review
  of 6112155 — the old "trim after user append only" version could leave
  ``max_history + 1`` entries after the assistant append).
- Provider failures stay non-fatal: an ``error`` event lands before the
  terminating ``done`` so the client never hangs without a terminator.
"""
from __future__ import annotations

import json
import time
from typing import AsyncIterator, Optional

from .llm.base import BaseLLMProvider, ToolCall
from .streaming import (
    EVENT_DONE,
    EVENT_ERROR,
    EVENT_STATUS,
    EVENT_TOKEN,
    EVENT_TOOL_CALL,
    EVENT_TOOL_RESULT,
    encode_event,
)
from .tool_registry import (
    ToolDisabledError,
    ToolNotFoundError,
    ToolRegistry,
)


class ChatOrchestrator:
    def __init__(
        self,
        llm: BaseLLMProvider,
        *,
        registry: Optional[ToolRegistry] = None,
        max_history: int = 50,
        max_rounds: int = 5,
    ):
        self.llm = llm
        self.registry = registry
        self.max_history = max_history
        self.max_rounds = max_rounds

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _provider_messages(self, history: list[dict]) -> list[dict]:
        """Strip ``ui_context`` and surface tool messages with their name.

        Providers (Ollama in A.3, scripted in tests) only know about the
        OpenAI/Ollama message shape. ``ui_context`` is chat-router-only
        metadata and never goes to the LLM.
        """
        out = []
        for h in history:
            if h["role"] == "tool":
                out.append({
                    "role": "tool",
                    "name": h.get("name", ""),
                    "content": h.get("content", ""),
                })
            else:
                out.append({"role": h["role"], "content": h.get("content", "")})
        return out

    def _trim_history(self, history: list[dict]) -> None:
        """Bounded growth (Codex P3). Drop oldest entries after every turn
        so a long conversation can't OOM the session cache."""
        if len(history) > self.max_history:
            del history[: len(history) - self.max_history]

    async def _run_tool_loop(
        self,
        *,
        session: dict,
        history: list[dict],
    ) -> AsyncIterator[tuple[str, int]]:
        """Yield ``(ndjson_line, rounds_consumed_delta)`` for each event in
        the tool-call loop. The orchestrator's main ``run_turn`` adds the
        line straight to the stream and accumulates the rounds counter."""
        if self.registry is None:
            return
        schemas = self.registry.llm_schemas()
        if not schemas:
            return

        for round_idx in range(self.max_rounds):
            try:
                tool_call: Optional[ToolCall] = await self.llm.request_tool_call(
                    messages=self._provider_messages(history),
                    tools=schemas,
                )
            except Exception as exc:  # noqa: BLE001
                yield (
                    encode_event(EVENT_ERROR, {
                        "message": str(exc) or type(exc).__name__,
                        "code": "tool_request_failure",
                    }),
                    0,
                )
                return

            if tool_call is None:
                return

            yield (
                encode_event(EVENT_TOOL_CALL, {
                    "round": round_idx,
                    "tool": tool_call.name,
                    "arguments": tool_call.arguments,
                }),
                1,
            )

            t_tool = time.monotonic()
            try:
                result = self.registry.call_tool(
                    tool_call.name,
                    tool_call.arguments,
                    session=session,
                )
            except (ToolNotFoundError, ToolDisabledError) as exc:
                result = {"error": str(exc), "code": "tool_blocked"}
            except Exception as exc:  # noqa: BLE001
                result = {"error": f"{type(exc).__name__}: {exc}", "code": "tool_crash"}

            yield (
                encode_event(EVENT_TOOL_RESULT, {
                    "round": round_idx,
                    "tool": tool_call.name,
                    "result": result,
                    "ms": int((time.monotonic() - t_tool) * 1000),
                }),
                0,
            )
            history.append({
                "role": "tool",
                "name": tool_call.name,
                "content": json.dumps(result, ensure_ascii=False),
            })

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    async def run_turn(
        self,
        *,
        session: dict,
        user_message: str,
        ui_context: Optional[dict] = None,
    ) -> AsyncIterator[str]:
        """Drive one chat turn and yield NDJSON-encoded events."""
        t0 = time.monotonic()
        history: list[dict] = session.setdefault("history", [])

        history.append({
            "role": "user",
            "content": user_message,
            "ui_context": ui_context or {},
        })

        yield encode_event(EVENT_STATUS, {
            "message": "thinking",
            "provider": self.llm.name,
        })

        rounds = 0
        async for line, delta in self._run_tool_loop(session=session, history=history):
            yield line
            rounds += delta

        full_text = ""
        token_count = 0
        try:
            async for tok in self.llm.stream_tokens(
                messages=self._provider_messages(history),
            ):
                if not tok:
                    continue
                full_text += tok
                token_count += 1
                yield encode_event(EVENT_TOKEN, {"text": tok})
        except Exception as exc:  # noqa: BLE001
            yield encode_event(EVENT_ERROR, {
                "message": str(exc) or type(exc).__name__,
                "code": "llm_failure",
            })

        history.append({"role": "assistant", "content": full_text})
        self._trim_history(history)

        yield encode_event(EVENT_DONE, {
            "rounds": rounds,
            "total_tokens": token_count,
            "ms_total": int((time.monotonic() - t0) * 1000),
        })
