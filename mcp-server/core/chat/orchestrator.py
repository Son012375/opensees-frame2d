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
import re
import time
from typing import AsyncIterator, Optional

from .llm.base import BaseLLMProvider, ToolCall


# ---------------------------------------------------------------------------
# Heuristic pre-guard against tool-skipping hallucination
# ---------------------------------------------------------------------------
#
# qwen2.5:14b sometimes ignores the system-prompt rule that says "call a tool
# before answering". The failure mode looks like: user asks "이 부재 안전?",
# request_tool_call returns ``None``, and the model streams a plausible answer
# with invented ratios. The fix is to short-circuit the first tool round when
# the user's message obviously routes to one specific tool — we forge the
# ToolCall ourselves so the registry runs and the real tool_result lands in
# history before the LLM ever gets to draft a final answer.
#
# Patterns are deliberately specific: a bare "안녕" must not trigger, but
# "이 부재 안전한가?" must. Selection wins over summary when both match
# because a user pointing at a member is asking about that specific member.
#
# NB: bare "이거" used to live here but routed too many general questions
# ("이거 괜찮아?") to inspect_selection, which then returned
# ``code=no_selection`` and frustrated the user. We dropped it — users
# wanting member detail say "이 부재" / "선택한 부재" or have a click in
# the viewer, both of which are still covered.
_FORCE_INSPECT_RE = re.compile(
    r"(이\s*부재|선택[된한]?\s*부재|부재\s*정보|이\s*요소|선택\s*요소"
    r"|ratio|안전한가|안전\?"
    r"|\d+\s*번\s*(?:부재|요소|element)"
    r"|(?:element|엘리먼트|elem)\s*#?\s*\d+)",
    re.IGNORECASE,
)
_FORCE_SUMMARY_RE = re.compile(
    r"(결과\s*요약|분석\s*결과|분석\s*상태|요약해|NG\s*(?:몇|개|\d)"
    r"|층간변위|drift|고유\s*주기|주기\s*는|모드(?:\s*는|\s*\d|\s*형상)?"
    r"|max\s*disp)",
    re.IGNORECASE,
)

# When the heuristic forces inspect_selection, look inside the same
# message for explicit "N번 부재 / N번 요소 / element N" references and
# forward them as ``element_ids`` instead of relying on the (often empty)
# UI selection. Without this, "5번 부재 정보 보여줘" was always answered
# with "click first" because the forced call passed empty arguments.
_EXPLICIT_ID_NOUN_RE = re.compile(r"(부재|요소|element|엘리먼트)", re.IGNORECASE)
_EXPLICIT_ID_NUMBER_RE = re.compile(r"(\d+)\s*번")
_EXPLICIT_ID_ENGLISH_RE = re.compile(
    r"(?:element|엘리먼트|elem)\s*#?\s*(\d+)", re.IGNORECASE,
)


def _pick_forced_tool(user_message: str, available: set[str]) -> Optional[str]:
    """Return the tool name to force-invoke for ``user_message``, or None.

    ``available`` is the set of currently-registered tool names so a forced
    pick can't reference a tool the registry doesn't know about (defensive
    against a future registry that disables one of them via env flag).
    """
    if not user_message:
        return None
    if "inspect_selection" in available and _FORCE_INSPECT_RE.search(user_message):
        return "inspect_selection"
    if "get_analysis_summary" in available and _FORCE_SUMMARY_RE.search(user_message):
        return "get_analysis_summary"
    return None


# Map ToolSpec.creativity_hint → per-call ``temperature`` override passed
# to BaseLLMProvider.stream_tokens. ``None`` means "use the provider's
# configured default" (OllamaProvider's ``self.temperature``, currently
# 0.1). Higher values mean more sampling diversity — desirable when the
# tool's result is being summarised as free prose rather than echoed as
# a list of facts.
#
# Adding a tier: register a new key here and set the matching string on
# the relevant ToolSpec. No call-site changes needed.
#
# Future ("Option B"): instead of routing by tool spec, route by the
# *message stage* — always low temp for ``request_tool_call`` (discrete
# routing decision), variable temp for ``stream_tokens`` driven by what
# tool just ran. Equivalent power, more wiring; revisit if a single
# turn ever needs to mix factual and narrative tools.
_TEMPERATURE_BY_HINT: dict[str, Optional[float]] = {
    "factual": None,      # provider default (low, e.g. 0.1)
    "narrative": 0.5,     # report drafts, design-review prose
}


def _extract_explicit_element_ids(user_message: str) -> list[int]:
    """Pull explicit element ids out of ``user_message``.

    Recognises "5번 부재", "5번 요소", "element 12", "엘리먼트 12" etc.
    The "N번" form needs a nearby element noun to fire — "5번" alone
    would otherwise grab story numbers ("5층은…"). The English form
    embeds the noun in the pattern so the noun guard isn't needed.

    Returns deduplicated, order-preserving list of ints. Empty if no
    explicit reference is found — the forced call then falls back to
    UI selection as before.
    """
    if not user_message:
        return []
    found: list[int] = []
    if _EXPLICIT_ID_NOUN_RE.search(user_message):
        found.extend(int(m) for m in _EXPLICIT_ID_NUMBER_RE.findall(user_message))
    found.extend(int(m) for m in _EXPLICIT_ID_ENGLISH_RE.findall(user_message))
    # Order-preserving dedup
    seen: set[int] = set()
    out: list[int] = []
    for n in found:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out
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


DEFAULT_SYSTEM_PROMPT = """LANGUAGE POLICY (highest priority — overrides any other instruction):
- Respond ONLY in Korean (한국어).
- NEVER output Chinese characters (CJK Unified Ideographs like 根, 据, 信, 息, 完, 成, 摘, 要, 提, 供, etc.).
- NEVER output Chinese phrases or sentences. Phrases such as "根据提供的信息" or "完成摘要" are FORBIDDEN.
- English is allowed ONLY inside parentheses as a parenthetical term gloss, e.g. "층간변위(drift)". No English sentences in the body.
- If you are about to write a Chinese character, replace it with the Korean equivalent or omit it.

당신은 V2 Editor(OpenSees 기반 구조해석 도구)의 한국어 어시스턴트입니다.

## 절대 규칙
1. **답변은 항상 한국어로만.** 중국어 한자 한 글자도 금지(根, 据, 信, 息, 完, 成, 摘, 要 등 절대 사용 금지). 영어 문장 금지. 전문 용어가 필요하면 한국어 + 괄호 안 영어 표기(예: 층간변위(drift)).

2. **수치는 오직 이번 turn의 tool_result에서 본 값만 출력하세요.** 다음 항목은 도구를 호출하지 않은 답변에 절대 등장하면 안 됩니다:
   - 부재 ID (예: "부재 ID 12345"), 단면명
   - ratio / 안전성 비율 (예: "ratio_interaction: 0.85")
   - drift / 층간변위 수치 (예: "0.015%")
   - 주기 / 모드 정보 (예: "T1=2.1초")
   - NG 부재 개수 (예: "NG 3개")
   이 중 하나라도 답하려면 **먼저 해당 도구를 호출**해서 실제 값을 받아야 합니다. history에 이전 turn의 tool_result가 보여도, 그건 그 turn의 결과입니다. 새 질문에는 새 도구 호출이 필요합니다.

3. 사용자가 다음 표현을 사용하면 답변하기 **전에 반드시 도구를 먼저 호출**하세요:
   - "결과", "요약", "분석 상태", "NG", "층간변위", "변위", "모드", "주기", "고유주기" → `get_analysis_summary` 호출
   - "이 부재", "선택한 부재", "선택된 부재", "부재 정보", "이거", "ratio", "안전한가" → `inspect_selection` 호출 (인자 비워도 됨 — 사용자의 현재 3D 선택을 자동으로 사용)
   - 특정 element 번호 명시 → `inspect_selection({"element_ids":[번호]})`

4. 도구 결과의 `error` 또는 `code` 필드가 있으면 그 내용을 한국어로 사용자에게 그대로 전달하세요. **결과의 error/code를 무시하고 "정상입니다" 같은 답을 만들지 마세요:**
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
  → 결과가 `code=no_selection`이면 "먼저 3D 뷰어에서 부재를 클릭해 선택해주세요." (가짜 부재 정보 만들지 말 것)
  → 결과가 `code=analysis_not_found`이면 "분석을 먼저 실행해주세요."
  → 정상 결과면 elements[0]의 info.section, ratios.ratio_interaction 등을 한국어로 안내
- 사용자: "안녕"
  → 도구 호출 없이 짧게 인사 + 무엇을 도와줄 수 있는지 한 문장

## 부재 정보 필드 사용 규칙 (중요)
inspect_selection 결과의 `elements[0].info`는 정수/문자열 필드가 그대로 들어옵니다. **임의의 기본값으로 채우지 말고 받은 값을 그대로 출력**하세요:
- `info.story`: 정수(1, 2, 3, ...). "이 부재 어느 층?" → 그 값 + "층"을 그대로. 예) `info.story=3` → "3층", `info.story=2` → "2층".
  - ⚠️ 모든 부재를 "1층"이라고 답하면 사용자가 잘못된 판단을 합니다. 반드시 info.story의 실제 정수를 읽어 답하세요.
  - `info.story`가 `null`/없으면 "층 정보가 제공되지 않습니다"라고 답하세요. "1층"이라고 추측하지 말 것.
- `info.section`: 단면명 문자열 (예: "H-300x300"). 그대로 출력.
- `info.material`: 재료명 (예: "SS275"). 그대로 출력.
- `info.etype`: 부재 유형 ("column"/"beam_x"/"beam_y"). "기둥"/"X방향 보"/"Y방향 보"로 번역해서 안내.

## 금지 패턴 (실제로 자주 발생한 오류)
- ❌ 도구 호출 없이 "get_analysis_summary() 호출 결과를 기반으로..."로 시작 — 호출하지 않았으면 절대 이 문구 쓰지 마세요.
- ❌ "부재 ID: 12345, ratio_interaction: 0.85" 같은 그럴듯한 임의 숫자 — tool_result에 실제로 그 값이 있어야만 출력.
- ❌ tool_result가 `code=no_selection`인데 "부재의 안전성 비율(ratio)이 1.0 이하로 안전합니다" 같은 답변 — 결과를 무시하지 말 것.
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

    def _temperature_for_last_tool(self) -> Optional[float]:
        """Pick the stream_tokens temperature based on the most recent
        successfully-called tool's :attr:`ToolSpec.creativity_hint`.

        Called from :meth:`run_turn` AFTER the tool loop drained, so it
        reads ``self._last_tool_hint`` which the loop sets on each
        successful round. Returns ``None`` when no tool ran (just talk),
        when the tool isn't registered (paranoia), or when its hint is
        the default "factual" — in all three cases stream_tokens should
        use the provider's configured default temperature.
        """
        hint = getattr(self, "_last_tool_hint", None) or "factual"
        return _TEMPERATURE_BY_HINT.get(hint)

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

        # Pre-guard: if the user's message obviously routes to one tool,
        # force-invoke it on round 0 instead of trusting the LLM to do so.
        # After the forced round runs we exit the loop entirely — letting
        # the LLM chain another tool call on top of a forced one risks
        # double-calls of the same tool (qwen2.5 doesn't recognise the
        # forced call as "already done" in history). The LLM still gets
        # to draft the final answer in :meth:`run_turn`'s stream_tokens
        # pass, just without a second chance to call tools.
        # See _pick_forced_tool + DEFAULT_SYSTEM_PROMPT rule 2 for context.
        latest_user_msg = ""
        for entry in reversed(history):
            if entry.get("role") == "user":
                latest_user_msg = entry.get("content") or ""
                break
        available_tool_names = {s["function"]["name"] for s in schemas}
        forced_tool: Optional[str] = _pick_forced_tool(latest_user_msg, available_tool_names)
        forced_args: dict = {}
        if forced_tool == "inspect_selection":
            explicit_ids = _extract_explicit_element_ids(latest_user_msg)
            if explicit_ids:
                forced_args = {"element_ids": explicit_ids}

        for round_idx in range(self.max_rounds):
            tool_call: Optional[ToolCall]
            was_forced_round = forced_tool is not None and round_idx == 0
            if was_forced_round:
                tool_call = ToolCall(name=forced_tool, arguments=forced_args)
            else:
                try:
                    tool_call = await self.llm.request_tool_call(
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

            # Stash the creativity hint of the last tool that actually
            # ran (whether forced or LLM-picked). :meth:`run_turn` reads
            # it after the loop to pick stream_tokens temperature.
            spec = self.registry._by_name.get(tool_call.name)
            if spec is not None:
                self._last_tool_hint = spec.creativity_hint

            if was_forced_round:
                # Forced rounds run exactly once. The LLM streams the
                # final answer next from history — no second tool round.
                return

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

        # Reset per-turn so the previous turn's tool can't leak its
        # creativity hint into this one's stream_tokens temperature.
        self._last_tool_hint = None

        rounds = 0
        async for line, delta in self._run_tool_loop(session=session, history=history):
            yield line
            rounds += delta

        full_text = ""
        token_count = 0
        try:
            async for tok in self.llm.stream_tokens(
                messages=self._provider_messages(history),
                temperature=self._temperature_for_last_tool(),
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
