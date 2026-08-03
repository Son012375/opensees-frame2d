"""Claude narrator — 결과 해설 LLM(역할 #3)의 P1 I/O 계층.

ChatKHU / Mindlogic 게이트웨이의 **Anthropic 네이티브 Messages API**를 통해
claude-opus-4-8 등으로 종합검토의견 산문(summary_ko/en)을 생성한다.

게이트웨이는 공식 `anthropic` SDK에 base_url만 바꿔 끼우면 동작한다:
    base_url = https://factchat-cloud.mindlogic.ai/v1/gateway/claude   (SDK가 /v1/messages 부착)
    인증     = x-api-key (SDK api_key=) 또는 Authorization: Bearer (SDK auth_token=) 둘 다 허용

이 모듈은 `llm(facts) -> 후보(dict)` 콜러블을 만든다. 검증·폴백은
core.narrative_interpreter가 담당하므로, 여기서는 프롬프트 조립 + 호출 + JSON 파싱만 한다.
structured outputs(output_config.format) 미사용 — 프롬프트로 JSON을 요청하고 관대하게 파싱한다
(게이트웨이 기능 의존 회피; 숫자누출 self-check가 어차피 검증).

가이드/few-shot: docs/narrative_interpreter/fewshot_register_ko.md, fewshot_examples.json

Usage:
    from core.narrative_llm import make_narrator_from_env
    from core.narrative_interpreter import narrate_interpretation
    interpretation = narrate_interpretation(interpretation, dc, llm=make_narrator_from_env())
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Callable

# ============================================================
# 기본값 (env로 오버라이드)
# ============================================================

_DEFAULT_BASE_URL = "https://factchat-cloud.mindlogic.ai/v1/gateway/claude"
_DEFAULT_MODEL = "claude-opus-4-8"
# NG(moderate/severe)는 summary_ko/en + diagnosis_narrative_ko/en 4개 필드를 요청하므로
# 1200으로는 절단→파싱실패→empty_output 폴백이 잦다. 4필드 이중언어 여유분 확보.
_DEFAULT_MAX_TOKENS = 2000
# KHU/Mindlogic 게이트웨이는 temperature=1.0(기본) 외 값을 400(BadRequest)으로 거부함
# (0.0/0.2/0.5 모두 실패, 1.0만 통과). 따라서 기본은 '미전송'(게이트웨이 기본 사용).
# 결정성이 필요하고 게이트웨이가 지원하면 NARRATOR_TEMPERATURE 로 명시 설정.
_DEFAULT_TEMPERATURE = None

_ENV_KEY = ("NARRATOR_API_KEY", "KHU_GATEWAY_KEY")
_ENV_BASE_URL = "NARRATOR_BASE_URL"
_ENV_MODEL = "NARRATOR_MODEL"
_ENV_TEMPERATURE = "NARRATOR_TEMPERATURE"

_ASSET_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "docs", "narrative_interpreter", "fewshot_examples.json",
)

# 자산 파일을 못 찾을 때의 인라인 폴백 규칙 (핵심만).
_FALLBACK_RULES = [
    "명사형 종결(~됨/~함/~확인됨/~판단됨/~검토됨)로 쓰는 한국 구조 실무 종합검토의견 문체.",
    "수치(비/숫자/KDS 조항)는 주어진 facts 문자열을 글자 그대로 복사만 한다. facts에 없는 숫자는 절대 쓰지 않는다.",
    "무주어 3인칭, 1인칭·감정·과장·이모지 금지.",
    "길이: facts가 풍부하면 4~6문장으로 다음 흐름에 엮는다 — 구조개요 → 적용기준·하중(특히 지진 밑면전단 V·Cs·R·Ie·SDS) → 해석방법 → 부재검토 → 층간변위 → 동적(질량참여율) → 종합판정·권고. facts가 적으면 2~3문장으로 짧게.",
    "facts에 structure_overview/seismic/loads/analysis_method_ko/mass_participation/member_groups가 있으면 적극 활용하고, 없는 항목은 생략한다(억지로 채우지 않음).",
    "OK는 '…만족하는 것으로 확인됨 → 안전한 것으로 판단됨', NG는 '…초과하여 보강/단면증대/시스템변경이 필요한 것으로 검토됨'.",
    "OK/NG 판정을 뒤집거나 완화하지 않는다. 충실히 못 쓰면 confidence='low'.",
]
_FALLBACK_TONE = {
    "safe": "단정적 긍정 — '…안전한 것으로 판단됨.'",
    "marginal": "긍정 + 조건 — '…기준을 만족하나, 하중 변경 시 재검토 필요.'",
    "moderate": "부적합 + 해결책 — '…단면 증대로 기준을 만족할 수 있는 것으로 판단됨.'",
    "severe": "부적합 + 시스템 차원 — '…구조시스템 차원의 재검토가 요구됨.'",
}


# ============================================================
# 프롬프트 조립
# ============================================================

def _load_asset() -> dict:
    try:
        with open(_ASSET_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def build_system_prompt(asset: dict | None = None) -> str:
    """register 규칙 + 심각도 어조 + few-shot으로 시스템 프롬프트를 조립한다."""
    if asset is None:
        asset = _load_asset()
    rules = asset.get("register_rules") or _FALLBACK_RULES
    tone = asset.get("severity_tone") or _FALLBACK_TONE
    fewshot = asset.get("fewshot") or []

    lines: list[str] = [
        "당신은 한국 구조 실무의 '종합검토의견' 작성자다. "
        "규칙기반 해석기가 이미 확정한 결과(facts)를 받아, 그 사실을 한국어 공학 문체의 산문으로 다듬는다.",
        "당신은 숫자·판정을 새로 만들지 않는다. 받은 facts를 문장으로만 엮는다.",
        "",
        "## 문체 규칙",
    ]
    lines += [f"- {r}" for r in rules]
    lines += ["", "## 심각도별 어조"]
    lines += [f"- {sev}: {desc}" for sev, desc in tone.items()]

    if fewshot:
        lines += ["", "## 예시 (facts → 출력)"]
        for i, ex in enumerate(fewshot, 1):
            f = ex.get("facts", {})
            lines.append(f"[예시 {i}] facts: {json.dumps(f, ensure_ascii=False)}")
            if ex.get("summary_ko"):
                lines.append(f"  summary_ko: {ex['summary_ko']}")
            if ex.get("summary_en"):
                lines.append(f"  summary_en: {ex['summary_en']}")
            # NG 예시에만 존재 — diagnosis_narrative 작성법을 학습시킨다 (B2).
            if ex.get("diagnosis_narrative_ko"):
                lines.append(f"  diagnosis_narrative_ko: {ex['diagnosis_narrative_ko']}")
            if ex.get("diagnosis_narrative_en"):
                lines.append(f"  diagnosis_narrative_en: {ex['diagnosis_narrative_en']}")

    lines += [
        "",
        "## 출력 형식 (반드시 JSON만)",
        '{"summary_ko": "...", "summary_en": "...", "confidence": "high" 또는 "low", "used_only_given_facts": true}',
        "JSON 외 다른 텍스트·코드펜스 설명을 붙이지 말 것. facts만으로 충실히 작성 불가하면 confidence를 \"low\"로.",
    ]
    return "\n".join(lines)


def _user_message(facts: dict) -> str:
    head = (
        "아래 facts만 사용해 종합검토의견 summary를 작성하라. "
        "facts에 없는 숫자/비/KDS 조항은 절대 쓰지 말 것.\n\n"
        "facts:\n" + json.dumps(facts, ensure_ascii=False, indent=2) + "\n\n"
    )
    # B2: NG(moderate/severe)일 때만 진단 서사(diagnosis_narrative)도 함께 요청.
    if facts.get("severity") in ("moderate", "severe"):
        return head + (
            "이 결과는 부적합(NG)이므로, summary와 별도로 부적합의 원인·메커니즘을 "
            "2~3문장으로 설명하는 diagnosis_narrative_ko/en도 작성하라 "
            "(facts.diagnosis/member_groups/drift_by_direction만 근거로, 숫자는 facts 문자열만 복사).\n"
            '출력은 JSON만: {"summary_ko": "...", "summary_en": "...", '
            '"diagnosis_narrative_ko": "...", "diagnosis_narrative_en": "...", '
            '"confidence": "high|low", "used_only_given_facts": true}'
        )
    return head + (
        '출력은 JSON만: {"summary_ko": "...", "summary_en": "...", '
        '"confidence": "high|low", "used_only_given_facts": true}'
    )


# ============================================================
# 응답 파싱
# ============================================================

def _message_text(message: Any) -> str:
    """anthropic Message.content(블록 리스트)에서 text를 이어붙인다."""
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        btype = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
        if btype == "text":
            txt = getattr(block, "text", None) or (block.get("text") if isinstance(block, dict) else None)
            if txt:
                parts.append(txt)
    return "".join(parts)


def extract_json(text: str) -> dict:
    """응답 텍스트에서 첫 JSON 객체를 관대하게 파싱한다. 실패 시 {}.

    코드펜스(```json) 제거 후, 직접 json.loads → 실패 시 첫 '{'~마지막 '}' 추출.
    """
    if not text:
        return {}
    s = text.strip()
    # 코드펜스 제거
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s).strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except ValueError:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(s[start:end + 1])
            return obj if isinstance(obj, dict) else {}
        except ValueError:
            return {}
    return {}


# ============================================================
# narrator 빌더
# ============================================================

def build_claude_narrator(
    *,
    api_key: str,
    base_url: str = _DEFAULT_BASE_URL,
    model: str = _DEFAULT_MODEL,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    temperature: float | None = _DEFAULT_TEMPERATURE,
    client: Any | None = None,
    system_prompt: str | None = None,
) -> Callable[[dict], dict]:
    """facts -> 후보(dict) 콜러블을 만든다.

    client를 주입하면 그대로 사용(테스트용). None이면 anthropic.Anthropic을
    base_url(게이트웨이)로 생성. 인증은 x-api-key(api_key=)로 전송 — 게이트웨이 허용.
    후보 dict에 `_model`을 부착해 호출부(narration_meta)가 출처(AI 고지)를 기록하게 한다.
    """
    if client is None:
        import anthropic  # 지연 임포트 (키 있을 때만)
        client = anthropic.Anthropic(base_url=base_url, api_key=api_key)
    system = system_prompt if system_prompt is not None else build_system_prompt()
    # C3 감사로그용: 시스템 프롬프트(few-shot 포함) 지문. 프롬프트 변경 추적.
    prompt_hash = hashlib.sha256(system.encode("utf-8")).hexdigest()[:16]

    def narrate(facts: dict) -> dict:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": _user_message(facts)}],
        }
        if temperature is not None:  # 게이트웨이 호환: 명시될 때만 전송
            kwargs["temperature"] = temperature
        message = client.messages.create(**kwargs)
        candidate = extract_json(_message_text(message))
        if isinstance(candidate, dict):
            candidate.setdefault("_model", model)          # 출처 추적 (C1 AI 고지용)
            candidate.setdefault("_prompt_hash", prompt_hash)  # 출처 추적 (C3 감사로그용)
        return candidate

    # C2′ facts-hash 캐시가 LLM 호출 전 키를 만들 수 있도록 프롬프트 지문을 노출.
    narrate.prompt_hash = prompt_hash  # type: ignore[attr-defined]
    return narrate


def make_narrator_from_env() -> Callable[[dict], dict] | None:
    """env에 게이트웨이 키가 있으면 narrator를, 없으면 None(=identity 폴백)을 반환."""
    key = None
    for name in _ENV_KEY:
        key = os.environ.get(name)
        if key:
            break
    if not key:
        return None
    base_url = os.environ.get(_ENV_BASE_URL, _DEFAULT_BASE_URL)
    model = os.environ.get(_ENV_MODEL, _DEFAULT_MODEL)
    temp_env = os.environ.get(_ENV_TEMPERATURE)
    temperature = _DEFAULT_TEMPERATURE  # 기본 None(미전송) — 게이트웨이가 1.0 외 거부
    if temp_env not in (None, ""):
        try:
            temperature = float(temp_env)
        except (TypeError, ValueError):
            temperature = _DEFAULT_TEMPERATURE
    try:
        return build_claude_narrator(api_key=key, base_url=base_url, model=model, temperature=temperature)
    except Exception:  # noqa: BLE001 — narrator 생성 실패 시 identity로 폴백
        return None
