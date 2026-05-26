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


def _safe_encode(event_type: str, payload: dict | None = None) -> str:
    """Encode an event without ever raising.

    ``encode_event`` rejects payloads that carry :data:`FORBIDDEN_KEYS`
    (the forbidden-key guard that keeps ``model_json`` etc. off the
    wire). If the orchestrator's own emit calls or a tool result happen
    to trip the guard, we'd otherwise blow up mid-turn and the client
    would never see a ``done`` terminator — Codex P1 on bcf3a0e.

    Instead we catch and emit an ``error`` event explaining the failure.
    The fallback ``error`` event is built from primitive strings so it
    can never trigger the guard recursively; even so it goes through one
    more ``try``/``except`` so a bug in the encoder itself can't crash
    the stream.
    """
    try:
        return encode_event(event_type, payload)
    except (ValueError, TypeError) as exc:
        try:
            return encode_event(EVENT_ERROR, {
                "message": f"event encoding failed for {event_type!r}: {exc}",
                "code": "event_encoding_failed",
            })
        except Exception:  # noqa: BLE001 — last-resort raw NDJSON
            import json as _json
            return _json.dumps({
                "type": "error",
                "code": "event_encoding_failed",
                "message": "encode_event failed and fallback also failed",
            }) + "\n"


DEFAULT_SYSTEM_PROMPT = """당신은 V2 Editor(OpenSees 기반 구조해석 도구)의 한국어 어시스턴트입니다.

## 절대 규칙
1. **답변은 항상 한국어로만.** 중국어, 영어 문장 금지. 전문 용어가 필요하면 한국어 + 괄호 안 영어 표기(예: 층간변위(drift)).

2. **부재 정보, ratio, 부재 ID, 단면명 같은 수치는 절대로 추측하거나 만들어내지 마세요.** 반드시 도구를 호출해 받은 결과의 값만 사용하세요. 도구를 호출하지 않은 채로 "부재 ID 12345" 같은 임의 숫자를 답변에 포함하면 사용자가 잘못된 판단을 내릴 수 있습니다.

3. 사용자가 다음 표현을 사용하면 답변하기 **전에 반드시 도구를 먼저 호출**하세요:
   - "결과", "요약", "분석 상태", "NG", "층간변위", "변위", "모드", "주기", "고유주기" → `get_analysis_summary` 호출
   - "이 부재", "선택한 부재", "선택된 부재", "부재 정보", "이거", "ratio", "안전한가" → `inspect_selection` 호출 (인자 비워도 됨 — 사용자의 현재 3D 선택을 자동으로 사용)
   - 특정 element 번호 명시 → `inspect_selection({"element_ids":[번호]})`

4. 도구 결과의 `error` 또는 `code` 필드가 있으면 그 내용을 한국어로 사용자에게 그대로 전달하세요:
   - `code=analysis_not_found` → "분석이 만료되었거나 찾을 수 없습니다. 좌측 패널에서 분석을 다시 실행해주세요."
   - `code=no_selection` → "먼저 3D 뷰어에서 부재를 클릭해 선택해주세요."
   - `analysis_id is required` → "아직 분석을 실행하지 않았습니다. 좌측 패널에서 분석을 먼저 실행해주세요."

5. 도구 결과에 들어있지 않은 수치/조항 번호는 추측하지 마세요. 모르면 "해당 정보는 가지고 있지 않습니다"라고 답변.

6. 답변은 간결하게. 불필요한 인사말이나 "도움이 되었으면 좋겠습니다" 같은 사족 금지.

## 도구 사용 예시
- 사용자: "지금 분석 결과 요약해줘"
  → get_analysis_summary 호출 → 결과의 max_drift, ng_count, T1 등을 자연스러운 한국어 문장으로 요약
- 사용자: "지금 선택된 부재 정보 보여줘"
  → inspect_selection 호출(인자 비움)
  → 결과가 `code=no_selection`이면 "먼저 3D 뷰어에서 부재를 클릭해 선택해주세요."
  → 결과가 `code=analysis_not_found`이면 "분석을 먼저 실행해주세요."
  → 정상 결과면 elements[0]의 info.section, ratios.ratio_interaction 등을 한국어로 안내
- 사용자: "안녕"
  → 도구 호출 없이 짧게 인사 + 무엇을 도와줄 수 있는지 한 문장
"""


class ChatOrchestrator:
    def __init__(
        self,
        llm: BaseLLMProvider,
        *,
        registry: Optional[ToolRegistry] = None,
        max_history: int = 50,
        max_rounds: int = 5,
        system_prompt: Optional[str] = None,
    ):
        self.llm = llm
        self.registry = registry
        self.max_history = max_history
        self.max_rounds = max_rounds
        # System prompt is prepended to every provider call so the LLM
        # has a stable instruction header — without it qwen2.5 falls
        # back to Chinese and skips tool calls (Phase A.4 live smoke
        # finding: empty system role meant the model treated every turn
        # as a free-form chat with no awareness of the available tools).
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _provider_messages(self, history: list[dict]) -> list[dict]:
        """Strip ``ui_context``, prepend the system prompt, and surface
        tool messages with their name.

        Providers (Ollama in A.3, scripted in tests) only know about the
        OpenAI/Ollama message shape. ``ui_context`` is chat-router-only
        metadata and never goes to the LLM. The system prompt lands first
        so the model has the language + tool-use contract before reading
        any user turn.
        """
        out: list[dict] = [{"role": "system", "content": self.system_prompt}]
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
                    _safe_encode(EVENT_ERROR, {
                        "message": str(exc) or type(exc).__name__,
                        "code": "tool_request_failure",
                    }),
                    0,
                )
                return

            if tool_call is None:
                return

            yield (
                _safe_encode(EVENT_TOOL_CALL, {
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
                _safe_encode(EVENT_TOOL_RESULT, {
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

        yield _safe_encode(EVENT_STATUS, {
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
                yield _safe_encode(EVENT_TOKEN, {"text": tok})
        except Exception as exc:  # noqa: BLE001
            yield _safe_encode(EVENT_ERROR, {
                "message": str(exc) or type(exc).__name__,
                "code": "llm_failure",
            })

        history.append({"role": "assistant", "content": full_text})
        self._trim_history(history)

        yield _safe_encode(EVENT_DONE, {
            "rounds": rounds,
            "total_tokens": token_count,
            "ms_total": int((time.monotonic() - t0) * 1000),
        })
