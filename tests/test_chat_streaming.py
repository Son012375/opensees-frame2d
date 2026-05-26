"""Unit tests for the chat NDJSON event encoder.

The wire format ships in Phase 0 Step 0-4 so the chat router + chat
widget can agree on a single contract before either side exists. Tests
exercise the encoder directly; the router/orchestrator integration
arrives in Phase A.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mcp-server"))

from core.chat.streaming import (  # noqa: E402
    EVENT_DONE,
    EVENT_ERROR,
    EVENT_STATUS,
    EVENT_TOKEN,
    EVENT_TOOL_CALL,
    EVENT_TOOL_RESULT,
    FORBIDDEN_KEYS,
    encode_event,
)


def test_encode_returns_single_ndjson_line_with_type():
    """One event = one line, terminated by '\\n', parseable as JSON,
    type discriminator at the top level (not nested)."""
    line = encode_event(EVENT_TOKEN, {"text": "분석"})
    assert line.endswith("\n")
    assert line.count("\n") == 1
    parsed = json.loads(line)
    assert parsed == {"type": "token", "text": "분석"}


def test_payload_is_merged_not_nested():
    """Consumers read fields off the top object — no ``{type, payload: {...}}``
    envelope wrapping."""
    line = encode_event(EVENT_TOOL_CALL, {
        "round": 0, "tool": "get_analysis_summary", "arguments": {"analysis_id": "abc"},
    })
    parsed = json.loads(line)
    assert parsed["type"] == "tool_call"
    assert parsed["round"] == 0
    assert parsed["tool"] == "get_analysis_summary"
    assert parsed["arguments"] == {"analysis_id": "abc"}
    assert "payload" not in parsed


def test_payload_none_yields_bare_type_event():
    """``done`` etc. may carry no payload. Bare event must still be valid."""
    line = encode_event(EVENT_DONE)
    parsed = json.loads(line)
    assert parsed == {"type": "done"}


def test_unknown_event_type_raises():
    with pytest.raises(ValueError, match="unknown chat event type"):
        encode_event("magic", {"text": "no"})


def test_payload_must_be_dict():
    with pytest.raises(TypeError, match="payload must be dict"):
        encode_event(EVENT_TOKEN, "raw string is not allowed")  # type: ignore[arg-type]


def test_payload_type_key_clashes_with_discriminator():
    """If the orchestrator passes a payload that itself has a 'type' field
    we'd silently overwrite the event-type discriminator — reject loudly."""
    with pytest.raises(ValueError, match="'type'"):
        encode_event(EVENT_TOOL_RESULT, {"type": "something_else"})


def test_forbidden_top_level_key_raises_with_path():
    """``model_json`` at the root must never leak into a stream event."""
    with pytest.raises(ValueError, match=r"forbidden keys at.*model_json"):
        encode_event(EVENT_TOOL_RESULT, {
            "round": 1, "tool": "preview", "result": {}, "ms": 5,
            "model_json": {"nodes": []},
        })


def test_forbidden_nested_key_raises_with_full_dot_path():
    """Guard walks recursively — orchestrator can't smuggle ``updated_model``
    inside a tool result dict by mistake."""
    with pytest.raises(ValueError, match=r"result\.updated_model"):
        encode_event(EVENT_TOOL_RESULT, {
            "round": 1, "tool": "preview_section_change",
            "result": {
                "diff": {"changed_members": [{"id": 1}]},
                "updated_model": {"nodes": [{"id": 1}]},
            },
            "ms": 12,
        })


def test_forbidden_key_inside_list_caught():
    """Forbidden keys hidden inside a list of dicts (e.g. batch tool
    results) are still flagged."""
    with pytest.raises(ValueError, match="case_data"):
        encode_event(EVENT_TOOL_RESULT, {
            "tool": "list_results",
            "result": {"items": [{"name": "ok"}, {"case_data": {"DL": {}}}]},
            "ms": 1, "round": 0,
        })


def test_concatenated_events_split_to_ndjson_records():
    """Real consumers concatenate events and split on '\\n'. Verify the
    encoder produces a stream that round-trips through that split."""
    events = [
        encode_event(EVENT_TOOL_CALL, {"round": 0, "tool": "ping"}),
        encode_event(EVENT_TOOL_RESULT, {"round": 0, "tool": "ping", "result": {"ok": True}, "ms": 1}),
        encode_event(EVENT_TOKEN, {"text": "안"}),
        encode_event(EVENT_TOKEN, {"text": "녕"}),
        encode_event(EVENT_DONE, {"rounds": 1, "total_tokens": 2, "ms_total": 42}),
    ]
    stream = "".join(events)
    # Split exactly like a browser reading line-by-line; trailing "" after
    # final \n is dropped.
    lines = [ln for ln in stream.split("\n") if ln]
    parsed = [json.loads(ln) for ln in lines]
    assert [p["type"] for p in parsed] == [
        "tool_call", "tool_result", "token", "token", "done",
    ]
    assert parsed[-1]["total_tokens"] == 2


def test_forbidden_keys_set_matches_documented_contract():
    """If someone adds/removes a forbidden key, this test forces a paired
    update to the streaming.py docstring + Phase 0 plan."""
    assert FORBIDDEN_KEYS == frozenset({
        "model_json",
        "updated_model",
        "case_data",
        "member_forces",
        "building_model",
        "seismic_report",
    })


def test_event_type_constants_match_string_values():
    """Constants documented in the module docstring must equal the actual
    on-wire ``type`` field — protects against silent rename drift."""
    assert EVENT_TOOL_CALL == "tool_call"
    assert EVENT_TOOL_RESULT == "tool_result"
    assert EVENT_TOKEN == "token"
    assert EVENT_STATUS == "status"
    assert EVENT_ERROR == "error"
    assert EVENT_DONE == "done"
