"""NDJSON event format for ``POST /api/v2/chat/messages``.

The chat router replies with a single ND-JSON stream
(``Content-Type: application/x-ndjson``). Each line is exactly one JSON
object terminated by ``"\\n"``. There is no ``data:`` prefix and no
multi-line framing — that lets the browser read it with
``fetch + ReadableStream`` line-by-line without an SSE parser, and lets
``pytest`` assert on full event sequences with a one-liner split.

Event types
-----------
``tool_call``    LLM requested a tool. Payload: ``{round, tool, arguments}``.
``tool_result``  Tool execution finished. Payload: ``{round, tool, result, ms}``.
``token``        One streamed token of the final natural-language answer.
                 Payload: ``{text}``.
``status``       Transient progress message (e.g. "evaluating 3/12").
                 Payload: ``{message, ...}``.
``error``        Recoverable failure surfaced to the chat UI.
                 Payload: ``{message, code?}``.
``done``         Terminator. Payload: ``{rounds, total_tokens, ms_total}``.

Forbidden keys
--------------
``analysis_context_cache`` carries the full ``model_json``,
``updated_model``, ``case_data``, ``member_forces``, ``building_model``,
and ``seismic_report`` — they're large (KB–MB per event) and a leak
would let the LLM echo the model verbatim, breaking the Phase 0
invariant that mutation goes through ``preview_apply`` only.
:func:`encode_event` walks each payload and raises if any of those keys
appear at any depth, catching orchestrator slips before they reach the
wire.
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Optional


EVENT_TOOL_CALL = "tool_call"
EVENT_TOOL_RESULT = "tool_result"
EVENT_TOKEN = "token"
EVENT_STATUS = "status"
EVENT_ERROR = "error"
EVENT_DONE = "done"

ALLOWED_EVENT_TYPES: frozenset[str] = frozenset({
    EVENT_TOOL_CALL,
    EVENT_TOOL_RESULT,
    EVENT_TOKEN,
    EVENT_STATUS,
    EVENT_ERROR,
    EVENT_DONE,
})


# Payload keys that must never appear in a streaming event at any depth.
# Two reasons: (1) size — these can be hundreds of KB per analysis and
# would dominate the event stream; (2) safety — exposing ``model_json``
# or ``updated_model`` to the LLM's response surface contradicts the
# Phase 0 invariant that LLMs cannot mutate or echo model JSON. The
# orchestrator should slice/summarise before emitting; this layer is the
# last-line guard.
FORBIDDEN_KEYS: frozenset[str] = frozenset({
    "model_json",
    "updated_model",
    "case_data",
    "member_forces",
    "building_model",
    "seismic_report",
})


def _scan_forbidden(obj: Any, path: tuple[str, ...] = ()) -> Iterable[str]:
    """Yield dotted paths to every forbidden key found inside ``obj``."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            here = path + (str(k),)
            if k in FORBIDDEN_KEYS:
                yield ".".join(here)
            yield from _scan_forbidden(v, here)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            yield from _scan_forbidden(item, path + (f"[{i}]",))


def encode_event(event_type: str, payload: Optional[dict] = None) -> str:
    """Serialise one chat event into an NDJSON line (ends with ``"\\n"``).

    The payload is merged at the top level of the line (``{"type": ...,
    **payload}``) so consumers can read fields directly without unwrapping
    a nested ``payload`` envelope.

    Raises
    ------
    ValueError
        If ``event_type`` is not one of :data:`ALLOWED_EVENT_TYPES`, or if
        ``payload`` contains any of :data:`FORBIDDEN_KEYS` at any depth.
    TypeError
        If ``payload`` is not a ``dict`` (or ``None``).
    """
    if event_type not in ALLOWED_EVENT_TYPES:
        raise ValueError(
            f"unknown chat event type: {event_type!r} "
            f"(allowed: {sorted(ALLOWED_EVENT_TYPES)})"
        )
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise TypeError(
            f"event payload must be dict or None, got {type(payload).__name__}"
        )
    if "type" in payload:
        raise ValueError(
            "payload may not contain a 'type' key — it would clash with "
            "the event type discriminator"
        )

    leaks = list(_scan_forbidden(payload))
    if leaks:
        raise ValueError(
            f"chat event {event_type!r} carries forbidden keys at: {leaks}"
        )

    line = json.dumps(
        {"type": event_type, **payload},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return line + "\n"
