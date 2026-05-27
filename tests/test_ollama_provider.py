"""Phase A.3 — OllamaProvider unit tests via httpx.MockTransport.

No live Ollama daemon required. Each test wires a mock client factory
that returns a deterministic ``/api/chat`` response, then asserts the
provider parsed it correctly into ``ToolCall`` instances or yielded the
right ``stream_tokens`` sequence. The wire-format edge cases (arguments
as JSON string, missing tool_calls, malformed stream lines) are the
ones most likely to bite when we point at a real qwen2.5 instance.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mcp-server"))

from core.chat.llm.base import ToolCall  # noqa: E402
from core.chat.llm.ollama_provider import (  # noqa: E402
    OllamaProvider,
    OllamaUnavailableError,
    _parse_first_tool_call,
    _to_ollama_messages,
)


# ---------------------------------------------------------------------------
# Helpers — mock transports + run-async wrapper
# ---------------------------------------------------------------------------

def _provider_with(handler, **kwargs) -> OllamaProvider:
    """Build an OllamaProvider whose AsyncClient uses ``handler`` as its
    transport. Lets each test script exactly one /api/chat response."""
    transport = httpx.MockTransport(handler)
    factory = lambda: httpx.AsyncClient(transport=transport, timeout=5)  # noqa: E731
    return OllamaProvider(
        base_url="http://test-ollama:11434",
        model="qwen2.5:14b",
        timeout_s=5,
        client_factory=factory,
        **kwargs,
    )


def _run(coro):
    return asyncio.run(coro)


async def _drain(async_gen):
    return [item async for item in async_gen]


# ---------------------------------------------------------------------------
# request_tool_call — happy paths
# ---------------------------------------------------------------------------

def test_request_tool_call_empty_tools_short_circuits_without_http():
    """Saves a network hop when CHAT_TOOLS_ENABLED resolves to no tools."""
    calls_seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls_seen.append(request)
        return httpx.Response(200, json={})

    provider = _provider_with(handler)
    result = _run(provider.request_tool_call(messages=[{"role": "user", "content": "x"}], tools=[]))
    assert result is None
    assert calls_seen == []  # no HTTP call made


def test_request_tool_call_returns_tool_call_from_well_formed_response():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is False
        assert body["model"] == "qwen2.5:14b"
        assert body["tools"]  # forwarded
        return httpx.Response(200, json={
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "inspect_selection",
                        "arguments": {"element_ids": [101]},
                    },
                }],
            },
            "done": True,
        })

    provider = _provider_with(handler)
    tc = _run(provider.request_tool_call(
        messages=[{"role": "user", "content": "이 부재"}],
        tools=[{"type": "function", "function": {"name": "inspect_selection"}}],
    ))
    assert isinstance(tc, ToolCall)
    assert tc.name == "inspect_selection"
    assert tc.arguments == {"element_ids": [101]}


def test_request_tool_call_returns_none_when_model_emits_text_instead():
    """Plain-text response → orchestrator drops out of the loop and
    streams the final answer."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "message": {"role": "assistant", "content": "여기서는 도구 안 씁니다."},
            "done": True,
        })

    provider = _provider_with(handler)
    result = _run(provider.request_tool_call(
        messages=[], tools=[{"type": "function", "function": {"name": "x"}}],
    ))
    assert result is None


# ---------------------------------------------------------------------------
# request_tool_call — arguments parsing edge cases (real Ollama wrinkles)
# ---------------------------------------------------------------------------

def test_request_tool_call_arguments_as_json_string_parsed_transparently():
    """Ollama's OpenAI-compatible mode returns arguments as a JSON string;
    the orchestrator should still see a dict."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "message": {
                "tool_calls": [{
                    "function": {
                        "name": "inspect_selection",
                        "arguments": '{"element_ids": [42]}',
                    },
                }],
            },
        })

    provider = _provider_with(handler)
    tc = _run(provider.request_tool_call(messages=[], tools=[{"x": 1}]))
    assert tc.arguments == {"element_ids": [42]}


def test_request_tool_call_arguments_empty_string_becomes_empty_dict():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "message": {"tool_calls": [{"function": {"name": "x", "arguments": ""}}]},
        })

    provider = _provider_with(handler)
    tc = _run(provider.request_tool_call(messages=[], tools=[{"x": 1}]))
    assert tc.arguments == {}


def test_request_tool_call_arguments_invalid_json_logs_and_uses_empty_dict():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "message": {"tool_calls": [{"function": {"name": "x", "arguments": "{not json"}}]},
        })

    provider = _provider_with(handler)
    tc = _run(provider.request_tool_call(messages=[], tools=[{"x": 1}]))
    assert tc.arguments == {}


def test_request_tool_call_returns_none_when_name_missing():
    """No name = unusable tool call. Return None so the orchestrator
    falls through to the text answer instead of trying to dispatch to
    an empty registry entry."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "message": {"tool_calls": [{"function": {"arguments": {}}}]},
        })

    provider = _provider_with(handler)
    assert _run(provider.request_tool_call(messages=[], tools=[{"x": 1}])) is None


# ---------------------------------------------------------------------------
# request_tool_call — transport failures
# ---------------------------------------------------------------------------

def test_request_tool_call_http_5xx_raises_ollama_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server exploded")

    provider = _provider_with(handler)
    with pytest.raises(OllamaUnavailableError, match="tool-round request failed"):
        _run(provider.request_tool_call(messages=[], tools=[{"x": 1}]))


def test_request_tool_call_network_error_raises_ollama_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    provider = _provider_with(handler)
    with pytest.raises(OllamaUnavailableError, match="tool-round request failed"):
        _run(provider.request_tool_call(messages=[], tools=[{"x": 1}]))


# ---------------------------------------------------------------------------
# stream_tokens — streaming NDJSON
# ---------------------------------------------------------------------------

def _ndjson_response(chunks: list[dict]) -> httpx.Response:
    body = "".join(json.dumps(c) + "\n" for c in chunks).encode("utf-8")
    return httpx.Response(200, content=body)


def test_stream_tokens_yields_message_content_chunks_in_order():
    chunks = [
        {"message": {"content": "안"}, "done": False},
        {"message": {"content": "녕"}, "done": False},
        {"message": {"content": "하세요"}, "done": False},
        {"message": {"content": ""}, "done": True},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return _ndjson_response(chunks)

    provider = _provider_with(handler)
    tokens = _run(_drain(provider.stream_tokens(messages=[])))
    assert tokens == ["안", "녕", "하세요"]


def test_stream_tokens_stops_at_done_true_even_if_more_lines_follow():
    chunks = [
        {"message": {"content": "first"}, "done": False},
        {"message": {"content": ""}, "done": True},
        # These lines would be a server bug but must not contaminate output.
        {"message": {"content": "ghost"}, "done": False},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return _ndjson_response(chunks)

    provider = _provider_with(handler)
    tokens = _run(_drain(provider.stream_tokens(messages=[])))
    assert tokens == ["first"]


def test_stream_tokens_skips_malformed_lines_without_breaking_stream():
    body = (
        b'{"message":{"content":"ok1"},"done":false}\n'
        b'not-json-at-all\n'
        b'{"message":{"content":"ok2"},"done":false}\n'
        b'{"message":{"content":""},"done":true}\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    provider = _provider_with(handler)
    tokens = _run(_drain(provider.stream_tokens(messages=[])))
    assert tokens == ["ok1", "ok2"]


def test_stream_tokens_http_error_raises_ollama_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="overloaded")

    provider = _provider_with(handler)
    with pytest.raises(OllamaUnavailableError, match="stream request failed"):
        _run(_drain(provider.stream_tokens(messages=[])))


# ---------------------------------------------------------------------------
# _parse_first_tool_call — module-level helper coverage
# ---------------------------------------------------------------------------

def test_parse_first_tool_call_empty_response_returns_none():
    assert _parse_first_tool_call({}) is None
    assert _parse_first_tool_call({"message": {}}) is None
    assert _parse_first_tool_call({"message": {"tool_calls": []}}) is None


def test_parse_first_tool_call_picks_only_the_first_entry():
    """If the model proposes a sequence (rare), we drive one round at a
    time. The next round can request the next call after seeing the
    first tool_result."""
    out = _parse_first_tool_call({
        "message": {
            "tool_calls": [
                {"function": {"name": "a", "arguments": {"x": 1}}},
                {"function": {"name": "b", "arguments": {"y": 2}}},
            ],
        },
    })
    assert out == ToolCall(name="a", arguments={"x": 1})


# ---------------------------------------------------------------------------
# _to_ollama_messages — name → tool_name rewrite (Codex P1 on bcf3a0e)
# ---------------------------------------------------------------------------

def test_to_ollama_messages_renames_tool_name_field():
    """Ollama's chat API expects ``tool_name`` (not ``name``) on tool
    messages so the model can match a result back to the originating
    tool_call. The orchestrator keeps the internal field as ``name``
    to stay provider-agnostic; the provider rewrites at the wire."""
    out = _to_ollama_messages([
        {"role": "user", "content": "이 부재"},
        {"role": "assistant", "content": ""},
        {"role": "tool", "name": "inspect_selection", "content": '{"elements": []}'},
    ])
    assert out[0] == {"role": "user", "content": "이 부재"}
    assert out[1] == {"role": "assistant", "content": ""}
    # name dropped, tool_name added, content preserved
    assert out[2]["role"] == "tool"
    assert out[2]["tool_name"] == "inspect_selection"
    assert "name" not in out[2]
    assert out[2]["content"] == '{"elements": []}'


def test_to_ollama_messages_leaves_non_tool_entries_untouched():
    inp = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "ping", "extra_metadata": {"keep": True}},
    ]
    assert _to_ollama_messages(inp) == inp


def test_to_ollama_messages_tool_without_name_passes_through():
    """Edge case: a tool entry that somehow lacks ``name`` shouldn't
    get a synthesized ``tool_name`` key out of thin air."""
    inp = [{"role": "tool", "content": "{}"}]
    assert _to_ollama_messages(inp) == inp


def test_request_tool_call_sends_tool_name_not_name():
    """End-to-end check: orchestrator-style messages with ``name`` on
    tool entries arrive at Ollama as ``tool_name``."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"tool_calls": []}})

    provider = _provider_with(handler)
    _run(provider.request_tool_call(
        messages=[
            {"role": "user", "content": "이 부재"},
            {"role": "tool", "name": "inspect_selection", "content": '{"elements": []}'},
        ],
        tools=[{"type": "function", "function": {"name": "inspect_selection"}}],
    ))
    sent = captured["body"]["messages"]
    tool_msg = next(m for m in sent if m["role"] == "tool")
    assert tool_msg["tool_name"] == "inspect_selection"
    assert "name" not in tool_msg


def test_stream_tokens_also_rewrites_tool_messages():
    """The stream path runs after tool rounds, so its history can already
    contain tool entries that need rewriting. Same conversion applies."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=(
            json.dumps({"message": {"content": "x"}, "done": True}) + "\n"
        ).encode("utf-8"))

    provider = _provider_with(handler)
    _run(_drain(provider.stream_tokens(messages=[
        {"role": "tool", "name": "get_analysis_summary", "content": "{}"},
    ])))
    tool_msg = next(m for m in captured["body"]["messages"] if m["role"] == "tool")
    assert tool_msg["tool_name"] == "get_analysis_summary"
    assert "name" not in tool_msg


def test_parse_first_tool_call_non_dict_arguments_coerced_to_empty():
    out = _parse_first_tool_call({
        "message": {"tool_calls": [{"function": {"name": "a", "arguments": 42}}]},
    })
    assert out.arguments == {}


# ---------------------------------------------------------------------------
# Env-driven defaults
# ---------------------------------------------------------------------------

def test_env_defaults_applied_when_no_kwargs(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://lab:11434/")  # trailing slash gets stripped
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:32b")
    monkeypatch.setenv("OLLAMA_TIMEOUT_S", "45")
    monkeypatch.setenv("OLLAMA_TEMPERATURE", "0.15")

    provider = OllamaProvider()
    assert provider.base_url == "http://lab:11434"
    assert provider.model == "qwen2.5:32b"
    assert provider.timeout_s == 45.0
    assert provider.temperature == 0.15


def test_kwargs_beat_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://env:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "from-env")
    monkeypatch.setenv("OLLAMA_TEMPERATURE", "0.9")

    provider = OllamaProvider(
        base_url="http://kw:11434", model="from-kw", timeout_s=10, temperature=0.2,
    )
    assert provider.base_url == "http://kw:11434"
    assert provider.model == "from-kw"
    assert provider.timeout_s == 10.0
    assert provider.temperature == 0.2


def test_temperature_default_is_0_1_when_env_unset(monkeypatch):
    """Default lowered to 0.1 for factual chat (member story, ratio,
    drift — JSON-to-Korean translation that doesn't need creativity).
    Stops qwen2.5 from drifting (e.g. answering "1층" for every member
    regardless of the actual ``info.story`` value). Future report
    generation should pass higher temperature per call instead of
    bumping this default."""
    monkeypatch.delenv("OLLAMA_TEMPERATURE", raising=False)
    provider = OllamaProvider()
    assert provider.temperature == 0.1


def test_stream_tokens_per_call_temperature_overrides_default():
    """Foundation for future report-generation flow: callers can pass a
    higher temperature without mutating the provider's default. Critical
    that the per-call value beats self.temperature, not the other way."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=(
            json.dumps({"message": {"content": "ok"}, "done": True}) + "\n"
        ).encode("utf-8"))

    provider = _provider_with(handler, temperature=0.1)  # base default
    _run(_drain(provider.stream_tokens(
        messages=[{"role": "user", "content": "보고서 써줘"}],
        temperature=0.6,  # per-call override (e.g. report generation)
    )))
    assert captured["body"]["options"]["temperature"] == 0.6


def test_stream_tokens_falls_back_to_instance_temperature_when_no_override():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=(
            json.dumps({"message": {"content": "ok"}, "done": True}) + "\n"
        ).encode("utf-8"))

    provider = _provider_with(handler, temperature=0.15)
    _run(_drain(provider.stream_tokens(
        messages=[{"role": "user", "content": "x"}],
        # no temperature= argument → instance default wins
    )))
    assert captured["body"]["options"]["temperature"] == 0.15


def test_request_tool_call_per_call_temperature_overrides_default():
    """Symmetric override on the tool-routing call too."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"tool_calls": []}, "done": True})

    provider = _provider_with(handler, temperature=0.1)
    _run(provider.request_tool_call(
        messages=[{"role": "user", "content": "x"}],
        tools=[{"type": "function", "function": {"name": "x"}}],
        temperature=0.0,
    ))
    assert captured["body"]["options"]["temperature"] == 0.0


# ---------------------------------------------------------------------------
# Sampler options forwarded on both /api/chat calls
# ---------------------------------------------------------------------------

def test_request_tool_call_payload_includes_options_temperature():
    """The tool-routing call must carry the configured temperature so
    qwen2.5 routes deterministically + stays in Korean."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"tool_calls": []}, "done": True})

    provider = _provider_with(handler, temperature=0.25)
    _run(provider.request_tool_call(
        messages=[{"role": "user", "content": "이 부재"}],
        tools=[{"type": "function", "function": {"name": "inspect_selection"}}],
    ))
    assert captured["body"]["options"]["temperature"] == 0.25


def test_stream_tokens_payload_includes_options_temperature():
    """Same temperature must apply when streaming the final answer —
    that's where the language-drift symptom actually surfaces."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=(
            json.dumps({"message": {"content": "안녕하세요"}, "done": True}) + "\n"
        ).encode("utf-8"))

    provider = _provider_with(handler, temperature=0.4)
    _run(_drain(provider.stream_tokens(messages=[{"role": "user", "content": "안녕"}])))
    assert captured["body"]["options"]["temperature"] == 0.4


# ---------------------------------------------------------------------------
# chat_router fallback — Ollama init failure must keep route serving
# ---------------------------------------------------------------------------

def test_chat_router_falls_back_to_noop_when_ollama_provider_explodes(monkeypatch):
    """The router catches provider construction errors and returns a
    NoopProvider with a friendly placeholder so /messages never 500s
    on a bad CHAT_LLM_PROVIDER env."""
    sys.path.insert(0, str(ROOT / "webapp" / "backend"))
    from app.chat_router import _resolve_llm_provider
    from core.chat.llm.noop_provider import NoopProvider

    monkeypatch.setenv("CHAT_LLM_PROVIDER", "ollama")

    def boom(*a, **kw):
        raise RuntimeError("simulated init crash")

    # Patch OllamaProvider import target inside the router's local namespace.
    import core.chat.llm.ollama_provider as ollama_mod
    monkeypatch.setattr(ollama_mod, "OllamaProvider", boom)

    provider = _resolve_llm_provider()
    assert isinstance(provider, NoopProvider)
    # The fallback message should hint at the failure cause.
    assert "Ollama" in provider.REPLY or "초기화" in provider._message
