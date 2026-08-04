"""Narrative LLM (역할 #3 P1) 테스트 — 게이트웨이 narrator (오프라인/모의).

라이브 호출 없음. 모의 client를 주입해 프롬프트 조립·JSON 파싱·적용/폴백을 검증한다.
실행: cd mcp-server && python -m pytest ../tests/test_narrative_llm.py -v
"""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mcp-server"))

from core.narrative_llm import (
    build_system_prompt,
    extract_json,
    build_claude_narrator,
    make_narrator_from_env,
    _message_text,
    _DEFAULT_MODEL,
)
from core.narrative_interpreter import build_facts, narrate_interpretation


# ============================================================
# 모의 anthropic client
# ============================================================

class _FakeBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeMessage:
    def __init__(self, text):
        self.content = [_FakeBlock(text)]


class _FakeMessages:
    def __init__(self, reply_text, sink):
        self._reply = reply_text
        self._sink = sink

    def create(self, **kwargs):
        self._sink.append(kwargs)
        return _FakeMessage(self._reply)


class _FakeClient:
    def __init__(self, reply_text):
        self.calls = []
        self.messages = _FakeMessages(reply_text, self.calls)


def _moderate_interp():
    return {
        "severity": "moderate",
        "severity_label": {"ko": "보통 — 단면 보강 필요", "en": "Moderate"},
        "drift_interpretation": {"max_ratio": 0.92, "critical_story": 3, "critical_direction": "X"},
        "member_interpretation": {
            "governing_check": "interaction",
            "weakest_link": {"member_id": 214, "type": "beam_x", "story": 3, "ratio": 1.12},
            "ng_by_story": {3: 4}, "ng_by_type": {"beam_x": 4}, "status": "NG",
        },
        "modal_interpretation": None,
        "diagnosis": {"primary_cause_ko": "부재 내력 부족", "contributing_factors_ko": []},
        "suggestions": [{"type": "section_upgrade", "target": "beam_x",
                         "current": "H-400x200", "recommended": "H-450x200"}],
        "summary_ko": "TEMPLATE_KO", "summary_en": "TEMPLATE_EN",
    }


def _moderate_dc():
    return {"summary": {"max_drift_ratio": 0.92, "max_interaction_ratio": 1.12,
                        "ng_stories": 0, "ng_members": 4}}


# ============================================================
# 시스템 프롬프트 조립
# ============================================================

def test_system_prompt_from_asset():
    p = build_system_prompt()  # docs/.../fewshot_examples.json 로드
    assert "문체 규칙" in p
    assert "summary_ko" in p
    assert "JSON" in p
    # few-shot 자산이 로드되면 예시 비율이 들어감
    assert "1.12" in p


def test_system_prompt_fallback_rules():
    p = build_system_prompt(asset={})  # 자산 없음 → 인라인 폴백
    assert "명사형" in p
    assert "facts" in p


# ============================================================
# JSON 파싱
# ============================================================

def test_extract_json_bare():
    assert extract_json('{"summary_ko": "가", "confidence": "high"}')["summary_ko"] == "가"


def test_extract_json_fenced():
    txt = '```json\n{"summary_ko": "나", "confidence": "high"}\n```'
    assert extract_json(txt)["summary_ko"] == "나"


def test_extract_json_with_prose_around():
    txt = '다음은 결과입니다:\n{"summary_ko": "다"}\n이상입니다.'
    assert extract_json(txt)["summary_ko"] == "다"


def test_extract_json_malformed_returns_empty():
    assert extract_json("죄송하지만 JSON이 없습니다") == {}
    assert extract_json("") == {}


def test_message_text_joins_blocks():
    msg = _FakeMessage("hello")
    assert _message_text(msg) == "hello"


# ============================================================
# narrator (모의 client 주입)
# ============================================================

def test_narrator_returns_parsed_candidate_and_sends_request():
    reply = json.dumps({"summary_ko": "요약", "summary_en": "summary",
                        "confidence": "high", "used_only_given_facts": True}, ensure_ascii=False)
    client = _FakeClient(reply)
    narrate = build_claude_narrator(api_key="x", client=client, model=_DEFAULT_MODEL)
    out = narrate({"severity": "safe"})
    assert out["summary_ko"] == "요약"
    # 요청이 올바른 model/system/messages로 나갔는지
    call = client.calls[0]
    assert call["model"] == _DEFAULT_MODEL
    assert "문체 규칙" in call["system"]
    assert call["messages"][0]["role"] == "user"
    assert "facts" in call["messages"][0]["content"]


# ============================================================
# 통합: narrate_interpretation + 게이트웨이 narrator
# ============================================================

def test_integration_clean_candidate_applied():
    interp = _moderate_interp()
    facts = build_facts(interp, _moderate_dc())
    clean = json.dumps({
        "summary_ko": "최대 내력비 1.12(beam_x #214, 3층)로 4개 부재가 부적합으로 검토됨.",
        "summary_en": "Max ratio 1.12 (beam_x #214, Story 3); 4 members NG.",
        "confidence": "high",
    }, ensure_ascii=False)
    narrate = build_claude_narrator(api_key="x", client=_FakeClient(clean))
    out = narrate_interpretation(interp, _moderate_dc(), llm=narrate)
    assert out["summary_ko"].startswith("최대 내력비 1.12")
    assert out["narration_meta"]["fallback"] is False


def test_integration_number_leak_falls_back():
    interp = _moderate_interp()
    leaky = json.dumps({"summary_ko": "최대 내력비 9.99로 부적합.", "confidence": "high"},
                       ensure_ascii=False)
    narrate = build_claude_narrator(api_key="x", client=_FakeClient(leaky))
    out = narrate_interpretation(interp, _moderate_dc(), llm=narrate)
    assert out["summary_ko"] == "TEMPLATE_KO"  # 원본 유지
    assert out["narration_meta"]["reason"] == "number_leak"


def test_integration_garbage_response_falls_back():
    interp = _moderate_interp()
    narrate = build_claude_narrator(api_key="x", client=_FakeClient("죄송합니다 JSON 없음"))
    out = narrate_interpretation(interp, _moderate_dc(), llm=narrate)
    assert out["summary_ko"] == "TEMPLATE_KO"
    assert out["narration_meta"]["fallback"] is True


# ============================================================
# env 기반 팩토리
# ============================================================

def test_make_narrator_none_without_key(monkeypatch):
    monkeypatch.delenv("NARRATOR_API_KEY", raising=False)
    monkeypatch.delenv("KHU_GATEWAY_KEY", raising=False)
    assert make_narrator_from_env() is None


def test_make_narrator_callable_with_key(monkeypatch):
    monkeypatch.setenv("NARRATOR_API_KEY", "dummy-key")
    narrate = make_narrator_from_env()
    assert callable(narrate)  # anthropic 클라이언트 생성(네트워크 호출 아님)
