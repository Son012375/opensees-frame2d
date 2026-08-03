"""Narrative Interpreter (역할 #3) P0 테스트 — 계약/self-check/폴백.

실행: cd mcp-server && python -m pytest ../tests/test_narrative_interpreter.py -v
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mcp-server"))

from core.narrative_interpreter import (
    narrate_interpretation,
    build_facts,
    build_number_allowlist,
    find_ungrounded_numbers,
    passes_number_check,
    contradicts_verdict,
    apply_narration,
    _norm_num,
)


# ============================================================
# Mock 헬퍼
# ============================================================

def _moderate_interp():
    return {
        "severity": "moderate",
        "severity_label": {"ko": "보통 — 단면 보강 필요", "en": "Moderate — section upgrade needed"},
        "findings": [],
        "drift_interpretation": {
            "max_ratio": 0.92, "critical_story": 3, "critical_direction": "X",
            "soft_story_detected": False, "soft_story_stories": [],
        },
        "member_interpretation": {
            "governing_check": "interaction",
            "weakest_link": {"member_id": 214, "type": "beam_x", "story": 3, "ratio": 1.12},
            "ng_by_story": {3: 4}, "ng_by_type": {"beam_x": 4}, "status": "NG",
        },
        "modal_interpretation": None,
        "diagnosis": {
            "primary_cause": "member_capacity",
            "primary_cause_ko": "부재 내력 부족", "primary_cause_en": "Insufficient member capacity",
            "contributing_factors": [], "contributing_factors_ko": [],
        },
        "suggestions": [{
            "priority": 1, "type": "section_upgrade", "target": "beam_x",
            "current": "H-400x200", "recommended": "H-450x200",
            "message_ko": "...", "message_en": "...", "expected_impact": "medium",
        }],
        "summary_ko": "TEMPLATE_KO",
        "summary_en": "TEMPLATE_EN",
    }


def _moderate_dc():
    return {"summary": {"max_drift_ratio": 0.92, "max_interaction_ratio": 1.12,
                        "ng_stories": 0, "ng_members": 4}}


def _safe_interp():
    return {
        "severity": "safe",
        "severity_label": {"ko": "안전", "en": "Safe"},
        "findings": [],
        "drift_interpretation": {"max_ratio": 0.48, "critical_story": 3, "critical_direction": "X",
                                 "soft_story_detected": False, "soft_story_stories": []},
        "member_interpretation": {"governing_check": "interaction", "weakest_link": None,
                                  "ng_by_story": {}, "ng_by_type": {}, "status": "OK"},
        "modal_interpretation": None,
        "diagnosis": None,
        "suggestions": [],
        "summary_ko": "모든 설계검토 통과. 최대 활용률 62%.",
        "summary_en": "All design checks passed. Max utilization 62%.",
    }


# ============================================================
# build_facts
# ============================================================

def test_build_facts_numbers_are_strings():
    facts = build_facts(_moderate_interp(), _moderate_dc())
    assert isinstance(facts["max_drift_ratio"], str)
    assert isinstance(facts["max_interaction_ratio"], str)
    assert facts["max_drift_ratio"] == "0.92 (3층, X방향)"
    assert facts["max_interaction_ratio"] == "1.12 (X방향 보 #214, 3층)"


def test_build_facts_moderate_fields():
    facts = build_facts(_moderate_interp(), _moderate_dc())
    assert facts["severity"] == "moderate"
    assert facts["governing_member_mode"] == "축력-휨 상관"
    assert facts["ng_members"] == 4 and facts["ng_stories"] == 0
    assert facts["diagnosis"]["primary_cause_ko"] == "부재 내력 부족"
    assert facts["suggestions"] == ["X방향 보 단면 증대: H-400x200 → H-450x200"]
    assert facts["diagnosis"]["ng_by_story_ko"] == "3층 4개"
    assert facts["applicable_codes"] == ["KDS 41 17 00", "KDS 14 31 00"]


def test_build_facts_safe_omits_diagnosis_and_suggestion():
    facts = build_facts(_safe_interp(), {"summary": {"max_drift_ratio": 0.48,
                                                     "max_interaction_ratio": 0.62,
                                                     "ng_stories": 0, "ng_members": 0}})
    assert facts["severity"] == "safe"
    assert "diagnosis" not in facts
    assert "suggestion" not in facts
    assert "governing_member_mode" not in facts  # status OK → 생략
    assert facts["ng_members"] == 0


# ============================================================
# 숫자 정규화 + allowlist
# ============================================================

def test_norm_num():
    assert _norm_num("1.40") == "1.4"
    assert _norm_num("1.0") == "1"
    assert _norm_num("0.85") == "0.85"
    assert _norm_num("00") == "0"
    assert _norm_num("003") == "3"
    assert _norm_num("214") == "214"


def test_allowlist_includes_facts_and_constants():
    facts = build_facts(_moderate_interp(), _moderate_dc())
    allow = build_number_allowlist(facts)
    for n in ("0.92", "3", "1.12", "214", "4", "400", "200", "450", "41", "17", "31", "14", "0", "1"):
        assert n in allow, f"{n} missing from allowlist"


# ============================================================
# 숫자누출 self-check
# ============================================================

def test_grounded_text_passes():
    facts = build_facts(_moderate_interp(), _moderate_dc())
    allow = build_number_allowlist(facts)
    txt = ("설계검토 결과 일부 부재의 내력비가 1.0을 초과하여(최대 1.12, beam_x #214, 3층) "
           "4개 부재가 부적합으로 검토됨. KDS 14 31 00의 강도기준 대비 보강이 필요함.")
    assert passes_number_check(txt, allow)
    assert find_ungrounded_numbers(txt, allow) == []


def test_fabricated_ratio_flagged():
    facts = build_facts(_moderate_interp(), _moderate_dc())
    allow = build_number_allowlist(facts)
    txt = "최대 내력비가 1.45로 산정되어 부적합함."  # 1.45는 facts에 없음
    assert not passes_number_check(txt, allow)
    assert "1.45" in find_ungrounded_numbers(txt, allow)


def test_fabricated_kds_clause_flagged():
    facts = build_facts(_moderate_interp(), _moderate_dc())
    allow = build_number_allowlist(facts)
    txt = "KDS 41 17 05 기준을 적용함."  # 05 → 5, facts에 없음
    assert "5" in find_ungrounded_numbers(txt, allow)


def test_grounded_kds_clause_passes():
    facts = build_facts(_moderate_interp(), _moderate_dc())
    allow = build_number_allowlist(facts)
    txt = "각 층 설계층간변위는 KDS 41 17 00의 허용층간변위 이내임."
    assert passes_number_check(txt, allow)


# ============================================================
# 판정모순 검사
# ============================================================

def test_verdict_contradiction_moderate_safe_phrase():
    assert contradicts_verdict("…구조적으로 안전한 것으로 판단됨.", "moderate", "ko") is True


def test_verdict_no_contradiction_when_safe():
    assert contradicts_verdict("…구조적으로 안전한 것으로 판단됨.", "safe", "ko") is False


def test_verdict_no_contradiction_for_ng_text():
    assert contradicts_verdict("…부적합으로 검토되어 보강이 필요함.", "severe", "ko") is False


# ============================================================
# apply_narration
# ============================================================

def test_apply_clean_candidate_overwrites_summary():
    interp = _moderate_interp()
    facts = build_facts(interp, _moderate_dc())
    candidate = {
        "summary_ko": "설계검토 결과 최대 내력비 1.12(beam_x #214, 3층)로 4개 부재가 부적합으로 검토됨.",
        "summary_en": "Design check finds 4 members non-compliant (max ratio 1.12, beam_x #214, Story 3).",
        "confidence": "high", "used_only_given_facts": True,
    }
    out = apply_narration(interp, candidate, facts)
    assert out["summary_ko"] == candidate["summary_ko"]
    assert out["summary_en"] == candidate["summary_en"]
    assert out["narration_meta"]["fallback"] is False
    assert "summary_ko" in out["narration_meta"]["applied_fields"]


def test_apply_number_leak_falls_back():
    interp = _moderate_interp()
    facts = build_facts(interp, _moderate_dc())
    candidate = {"summary_ko": "최대 내력비 1.45로 부적합.", "summary_en": "max ratio 1.45.",
                 "confidence": "high"}
    out = apply_narration(interp, candidate, facts)
    assert out["summary_ko"] == "TEMPLATE_KO"  # 원본 유지
    assert out["narration_meta"]["fallback"] is True
    assert out["narration_meta"]["reason"] == "number_leak"
    assert "summary_ko" in out["narration_meta"]["leaked"]


def test_apply_low_confidence_falls_back():
    interp = _moderate_interp()
    facts = build_facts(interp, _moderate_dc())
    out = apply_narration(interp, {"summary_ko": "x", "confidence": "low"}, facts)
    assert out["summary_ko"] == "TEMPLATE_KO"
    assert out["narration_meta"]["reason"] == "low_confidence_or_empty"


def test_apply_verdict_contradiction_falls_back():
    interp = _moderate_interp()
    facts = build_facts(interp, _moderate_dc())
    candidate = {"summary_ko": "본 구조물은 구조적으로 안전한 것으로 판단됨.", "confidence": "high"}
    out = apply_narration(interp, candidate, facts)
    assert out["summary_ko"] == "TEMPLATE_KO"
    assert out["narration_meta"]["reason"] == "verdict_contradiction"


# ============================================================
# narrate_interpretation (진입점)
# ============================================================

def test_narrate_identity_when_no_llm():
    interp = _moderate_interp()
    out = narrate_interpretation(interp, _moderate_dc(), llm=None)
    assert out["summary_ko"] == "TEMPLATE_KO"
    assert out["summary_en"] == "TEMPLATE_EN"
    assert out["narration_meta"]["reason"] == "no_llm"
    assert out["narration_meta"]["fallback"] is True


def test_narrate_applies_clean_llm():
    interp = _moderate_interp()
    clean = {
        "summary_ko": "최대 내력비 1.12로 4개 부재 부적합, 단면 H-400x200→H-450x200 보강 필요.",
        "summary_en": "Max ratio 1.12; 4 members NG; upgrade H-400x200 to H-450x200.",
        "confidence": "high",
    }
    out = narrate_interpretation(interp, _moderate_dc(), llm=lambda facts: clean)
    assert out["summary_ko"] == clean["summary_ko"]
    assert out["narration_meta"]["fallback"] is False


def test_narrate_llm_exception_falls_back():
    interp = _moderate_interp()

    def boom(facts):
        raise RuntimeError("model down")

    out = narrate_interpretation(interp, _moderate_dc(), llm=boom)
    assert out["summary_ko"] == "TEMPLATE_KO"
    assert out["narration_meta"]["fallback"] is True
    assert out["narration_meta"]["reason"].startswith("llm_exception")


def test_narrate_empty_interpretation_passthrough():
    assert narrate_interpretation({}, None) == {}
    assert narrate_interpretation(None, None) is None


# ============================================================
# B4: 방향별 비대칭 (drift_by_direction)
# ============================================================

def test_build_facts_drift_by_direction_when_asymmetric():
    interp = _moderate_interp()
    interp["drift_interpretation"]["max_ratio_by_dir"] = {"X": 0.45, "Y": 0.88}
    facts = build_facts(interp, _moderate_dc())
    assert facts["drift_by_direction"] == "X방향 0.45 / Y방향 0.88"


def test_build_facts_no_drift_by_direction_when_symmetric():
    interp = _moderate_interp()
    interp["drift_interpretation"]["max_ratio_by_dir"] = {"X": 0.88, "Y": 0.85}
    facts = build_facts(interp, _moderate_dc())
    assert "drift_by_direction" not in facts  # 격차 0.03 < 0.15


def test_build_facts_no_drift_by_direction_when_negligible():
    interp = _moderate_interp()
    interp["drift_interpretation"]["max_ratio_by_dir"] = {"X": 0.20, "Y": 0.02}
    facts = build_facts(interp, _moderate_dc())
    assert "drift_by_direction" not in facts  # hi 0.20 < 0.3 (변위 자체가 미미)


# ============================================================
# B3: 권고 다건 (suggestions 복수)
# ============================================================

def test_build_facts_suggestions_multiple():
    interp = _moderate_interp()
    interp["suggestions"] = [
        {"type": "section_upgrade", "target": "column", "current": "H-300x300", "recommended": "H-350x350"},
        {"type": "section_upgrade", "target": "beam_x", "current": "H-400x200", "recommended": "H-450x200"},
        {"type": "system_change"},
    ]
    facts = build_facts(interp, _moderate_dc())
    assert facts["suggestions"] == [
        "기둥 단면 증대: H-300x300 → H-350x350",
        "X방향 보 단면 증대: H-400x200 → H-450x200",
        "횡력저항시스템 변경·추가(가새·전단벽 등)",
    ]
    # 단면치수는 allowlist 자기씨앗 — 누출 0 보장
    allow = build_number_allowlist(facts)
    assert passes_number_check(" ".join(facts["suggestions"]), allow)


# ============================================================
# A8: 한계 1문장 (limitations_ko — 숫자 없는 라벨)
# ============================================================

def test_build_facts_limitations_ko():
    dc = {
        "summary": {"max_drift_ratio": 0.92, "max_interaction_ratio": 1.12,
                    "ng_stories": 0, "ng_members": 4},
        "member_check": {"assumptions": [
            # Tier1-1/2: 정정된 가정문구 (K 골조형식별 / LTB 비지지길이 기준)
            "유효좌굴길이계수 K: 기둥 K=1.2 (비가새 모멘트골조 sway, AISC 보수 하한; K=1.0은 비보수)",
            "보 강축 휨내력: 슬래브 연속횡지지 시 Mp, 비지지 시 AISC F2 LTB(Lb/Lp/Lr, Cb=1.0). "
            "기둥 강축 휨은 층레벨 횡지지로 Mp 가정",
            "φ = 0.9 (KDS 41 31 00 표준 강도감소계수)",
        ]},
    }
    facts = build_facts(_moderate_interp(), dc)
    assert facts["limitations_ko"] == [
        "유효좌굴길이계수 K는 골조형식별로 적용함(비가새 모멘트골조는 sway로 K 증가)",
        "보 횡비틀림좌굴(LTB)은 비지지길이 기준 검토(슬래브 연속횡지지 가정 시 소성모멘트)",
    ]
    # 숫자 없는 라벨이어야 함 (자기씨앗 불필요)
    from core.narrative_interpreter import _numbers_in
    assert _numbers_in(" ".join(facts["limitations_ko"])) == []


# ============================================================
# B2: 진단 재료 (soft_story / ng_by_story)
# ============================================================

def test_build_facts_diagnosis_soft_story_and_distribution():
    interp = _moderate_interp()
    interp["severity"] = "severe"
    interp["drift_interpretation"]["soft_story_detected"] = True
    interp["drift_interpretation"]["soft_story_stories"] = [1]
    interp["member_interpretation"]["ng_by_story"] = {1: 6, 2: 3}
    facts = build_facts(interp, _moderate_dc())
    assert "연약층" in facts["diagnosis"]["soft_story_ko"]
    assert facts["diagnosis"]["ng_by_story_ko"] == "1층 6개, 2층 3개"


# ============================================================
# C4: 필드단위 폴백
# ============================================================

def test_apply_partial_fallback_clean_ko_leaky_en():
    interp = _moderate_interp()
    facts = build_facts(interp, _moderate_dc())
    candidate = {
        "summary_ko": "최대 내력비 1.12(beam_x #214, 3층)로 4개 부재가 부적합으로 검토됨.",
        "summary_en": "max ratio 9.99 — fabricated value.",   # 9.99 누출
        "confidence": "high",
    }
    out = apply_narration(interp, candidate, facts)
    assert out["summary_ko"] == candidate["summary_ko"]   # 깨끗 → 적용
    assert out["summary_en"] == "TEMPLATE_EN"             # 누출 → 템플릿
    meta = out["narration_meta"]
    assert meta["fallback"] is True
    assert meta["reason"] == "partial_fallback"
    assert meta["applied_fields"] == ["summary_ko"]
    assert meta["fallback_fields"]["summary_en"] == "number_leak"


def test_apply_partial_fallback_keeps_diagnosis_when_summary_leaks():
    interp = _moderate_interp()
    facts = build_facts(interp, _moderate_dc())
    candidate = {
        "summary_ko": "최대 내력비 7.77로 부적합.",   # 7.77 누출 → 폐기
        "diagnosis_narrative_ko": "3층 X방향 보 4개에서 내력비가 1.0을 초과함.",  # 깨끗 → 적용
        "confidence": "high",
    }
    out = apply_narration(interp, candidate, facts)
    assert out["summary_ko"] == "TEMPLATE_KO"
    assert out["diagnosis_narrative_ko"] == candidate["diagnosis_narrative_ko"]
    assert out["narration_meta"]["applied_fields"] == ["diagnosis_narrative_ko"]


# ============================================================
# C5: ko/en 교차 일관성
# ============================================================

def test_apply_bilingual_mismatch_falls_back():
    interp = _moderate_interp()
    facts = build_facts(interp, _moderate_dc())
    candidate = {
        "summary_ko": "최대 내력비 1.12, 최대 층간변위비 0.92로 검토됨.",  # 1.12, 0.92
        "summary_en": "Max interaction ratio 1.12.",                       # 1.12 만
        "confidence": "high",
    }
    out = apply_narration(interp, candidate, facts)
    assert out["summary_ko"] == "TEMPLATE_KO"   # 둘 다 폐기
    assert out["summary_en"] == "TEMPLATE_EN"
    assert out["narration_meta"]["reason"] == "bilingual_mismatch"
    assert out["narration_meta"]["bilingual_fields"] == ["summary_ko", "summary_en"]


def test_bilingual_consistent_ignores_clause_and_constants():
    from core.narrative_interpreter import _bilingual_consistent, _clause_and_const_numbers
    facts = {"applicable_codes": ["KDS 41 17 00", "KDS 14 31 00"]}
    exclude = _clause_and_const_numbers(facts)
    # ko는 두 조항, en은 한 조항만 인용 — 측정 숫자(1.12)는 동일
    ko = "KDS 41 17 00·14 31 00 적용, 최대 내력비 1.12 이하."
    en = "Per KDS 14 31 00; max ratio 1.12, below 1.0."
    assert _bilingual_consistent(ko, en, exclude) is True


def test_bilingual_consistent_ko_digit_vs_en_word_count():
    # 회귀: 한국어는 개수를 숫자('4개'), 영어는 단어('four')로 적어도
    # 측정값(소수 1.12)이 같으면 일치로 본다 (정수 개수는 비교 대상 아님).
    from core.narrative_interpreter import _bilingual_consistent
    ko = "X방향 보 4개의 내력비가 1.0을 초과(최대 1.12)하여 부적합으로 검토됨."
    en = "Four X-direction beams exceed unity (max ratio 1.12) and are non-compliant."
    assert _bilingual_consistent(ko, en, set()) is True


def test_apply_ko_digit_en_word_not_dropped():
    # 회귀(고위험): few-shot 스타일(한글 '4개' vs 영문 'four')의 깨끗한 후보가
    # 비대칭검사로 통째로 폐기되지 않아야 한다.
    interp = _moderate_interp()
    facts = build_facts(interp, _moderate_dc())
    candidate = {
        "summary_ko": "X방향 보 4개의 내력비가 1.0을 초과(최대 1.12)하여 부적합으로 검토됨.",
        "summary_en": "Four X-direction beams exceed unity (max ratio 1.12) and are non-compliant.",
        "confidence": "high",
    }
    out = apply_narration(interp, candidate, facts)
    assert out["summary_ko"] == candidate["summary_ko"]
    assert out["summary_en"] == candidate["summary_en"]
    assert out["narration_meta"]["fallback"] is False


# ============================================================
# B2 적용 + C3 감사로그
# ============================================================

def test_apply_diagnosis_narrative_applied():
    interp = _moderate_interp()
    facts = build_facts(interp, _moderate_dc())
    candidate = {
        "summary_ko": "최대 내력비 1.12로 4개 부재 부적합.",
        "summary_en": "Max ratio 1.12; 4 members NG.",
        "diagnosis_narrative_ko": "3층 X방향 보 4개에서 내력비가 1.0을 초과함.",
        "diagnosis_narrative_en": "Four Story-3 X beams exceed unity.",
        "confidence": "high",
    }
    out = apply_narration(interp, candidate, facts)
    assert out["diagnosis_narrative_ko"] == candidate["diagnosis_narrative_ko"]
    assert "diagnosis_narrative_ko" in out["narration_meta"]["applied_fields"]
    assert out["narration_meta"]["fallback"] is False


def test_cache_hit_reuses_candidate_without_calling_llm(tmp_path, monkeypatch):
    # C2′ 결정성: 동일 facts+prompt_hash면 2회차는 LLM 미호출, 동일 §10 재사용.
    monkeypatch.setenv("NARRATION_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("NARRATION_CACHE_DISABLE", raising=False)
    calls = {"n": 0}

    def llm(facts):
        calls["n"] += 1
        return {"summary_ko": "최대 내력비 1.12로 4개 부재 부적합.",
                "summary_en": "Max ratio 1.12; 4 NG.",
                "confidence": "high", "_model": "claude-opus-4-8", "_prompt_hash": "ph1"}
    llm.prompt_hash = "ph1"

    interp1 = _moderate_interp()
    out1 = narrate_interpretation(interp1, _moderate_dc(), llm=llm)
    interp2 = _moderate_interp()
    out2 = narrate_interpretation(interp2, _moderate_dc(), llm=llm)

    assert calls["n"] == 1                       # 2회차는 캐시 → LLM 1번만 호출
    assert out2["summary_ko"] == out1["summary_ko"]  # 결정론적 동일
    assert out2["narration_meta"]["cached"] is True
    assert out2["narration_meta"]["fallback"] is False


def test_cache_disabled_env_always_calls_llm(tmp_path, monkeypatch):
    monkeypatch.setenv("NARRATION_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("NARRATION_CACHE_DISABLE", "1")
    calls = {"n": 0}

    def llm(facts):
        calls["n"] += 1
        return {"summary_ko": "최대 내력비 1.12로 4개 부재 부적합.",
                "summary_en": "Max ratio 1.12; 4 NG.", "confidence": "high",
                "_model": "m", "_prompt_hash": "ph1"}
    llm.prompt_hash = "ph1"

    narrate_interpretation(_moderate_interp(), _moderate_dc(), llm=llm)
    narrate_interpretation(_moderate_interp(), _moderate_dc(), llm=llm)
    assert calls["n"] == 2  # 캐시 비활성 → 매번 호출


def test_cache_invalidated_by_prompt_hash_change(tmp_path, monkeypatch):
    monkeypatch.setenv("NARRATION_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("NARRATION_CACHE_DISABLE", raising=False)
    calls = {"n": 0}

    def make_llm(ph):
        def llm(facts):
            calls["n"] += 1
            return {"summary_ko": "최대 내력비 1.12로 4개 부재 부적합.",
                    "summary_en": "Max ratio 1.12; 4 NG.", "confidence": "high",
                    "_model": "m", "_prompt_hash": ph}
        llm.prompt_hash = ph
        return llm

    narrate_interpretation(_moderate_interp(), _moderate_dc(), llm=make_llm("ph1"))
    narrate_interpretation(_moderate_interp(), _moderate_dc(), llm=make_llm("ph2"))
    assert calls["n"] == 2  # 프롬프트 지문이 바뀌면 캐시 무효 → 재호출


def test_no_cache_when_llm_has_no_prompt_hash(tmp_path, monkeypatch):
    # 테스트용 람다(prompt_hash 속성 없음)는 캐시 경로를 타지 않음(하위호환).
    monkeypatch.setenv("NARRATION_CACHE_DIR", str(tmp_path / "cache"))
    calls = {"n": 0}

    def llm(facts):
        calls["n"] += 1
        return {"summary_ko": "최대 내력비 1.12로 4개 부재 부적합.",
                "summary_en": "Max ratio 1.12; 4 NG.", "confidence": "high"}

    out1 = narrate_interpretation(_moderate_interp(), _moderate_dc(), llm=llm)
    narrate_interpretation(_moderate_interp(), _moderate_dc(), llm=llm)
    assert calls["n"] == 2
    assert "cached" not in out1["narration_meta"]


def test_audit_record_written(tmp_path, monkeypatch):
    monkeypatch.setenv("NARRATION_AUDIT_LOG_PATH", str(tmp_path / "aud.jsonl"))
    interp = _moderate_interp()
    clean = {"summary_ko": "최대 내력비 1.12로 4개 부재 부적합.",
             "summary_en": "Max ratio 1.12; 4 NG.",
             "confidence": "high", "_model": "claude-opus-4-8", "_prompt_hash": "deadbeef"}
    narrate_interpretation(interp, _moderate_dc(), llm=lambda f: clean, analysis_id="job-xyz")
    import json
    p = tmp_path / "aud.jsonl"
    assert p.exists()
    rec = json.loads(p.read_text(encoding="utf-8").splitlines()[0])
    assert rec["analysis_id"] == "job-xyz"
    assert rec["model"] == "claude-opus-4-8"
    assert rec["prompt_hash"] == "deadbeef"
    assert rec["fallback"] is False
    assert "facts" in rec
