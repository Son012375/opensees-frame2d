# Phase D (챗봇 KDS-RAG 통합) — 리뷰 & 개선 로드맵

> 대상 PR: `feat(chat): Phase D — KDS compliance tool with hybrid anti-hallucination + collapsible UX` (base `origin/main` `b7bd91e`)
> 작성 근거: Codex 2-관점(실무자/연구자) 코드 리뷰, 2026-05-28

## 총평

**내부 베타 / 실무자 검토 보조 MVP로 merge 가능.** 핵심 방향 — "LLM이 답하는 게 아니라 *결정론적 설계검토 + 검증된 근거 표시 + 제한된 자연어 prefix*" — 가 옳다.

**강점**
- 강제 라우팅(`orchestrator.py`) + 부재별 설계근거 도구(`kds_compliance.py`) + prefix sanitize 조합이 실무 UX에 적합.
- KDS/AISC 인용을 LLM이 생성하지 않고 서버가 결정론적으로 렌더. 긴 근거는 collapsible로 접어 피로도 감소.
- 실패모드 기반 테스트(거짓 prefix 폐기, evidence-only collapsible, history sanitization)가 잘 잠겨 있어 논문/보고서의 검증 절로 활용 가능.

**한계** — 아래 실무자/연구자 관점으로 분리.

---

## 실무자 관점

| # | 항목 | 심각도 | 상태 |
|---|------|--------|------|
| P1 | evidence quote/collapsible body가 provider-visible history로 재유입 | 높음 | ✅ **이 PR에서 해결** |
| P2 | 신뢰 보정 라벨 부재 | 중간 | ✅ **이 PR에서 해결** |
| P3 | `issue_type` vocab-limited 매핑 = 검색용 근사 | 중간 | 🔜 로드맵 |
| P4 | AISC 임시참조 = proxy를 "KDS 준거"로 단정할 위험 | 중간 | 🔜 로드맵 (R4와 연동) |

### P1 — evidence 재유입 (해결됨)
- **근거**: anti-hallucination 보장이 단일 turn에 국한. `_pop_mandatory_response`가 tool 항목의 `kds_evidence`를 strip해도, 렌더된 quote가 assistant history(`full_text`)에 남아 `_provider_messages`를 통해 다음 turn LLM에 재노출 → paraphrase·발명 가능.
- **해결**: `run_turn`에서 provider-visible history에 저장하는 텍스트(`history_text` = prefix + summary)와 사용자에게 보내는 collapsible quote를 분리. collapsible은 `EVENT_COLLAPSIBLE`로만 emit하고 history에는 미저장.
- **성공 기준 (정확히)**: evidence **quote/collapsible body**가 provider-visible 모든 메시지 content에 부재. *모든 KDS 문자열 제거가 아님* — `warnings` 필드의 code-id 힌트("KDS 14 31 00")는 메타 경고로 남으며 완전 제거는 R3/R4 후속.
- **테스트**: `test_assistant_history_excludes_collapsible_evidence` — `_provider_messages(session["history"])` 전체에서 quote body 부재 확인.

### P2 — 신뢰 보정 라벨 (해결됨)
- **근거**: AISC 임시참조가 섞인 코퍼스 결과를 "KDS 준거 검토"로 오인할 위험. 토글을 펼치지 않으면 advisory 성격을 모름.
- **해결**: always-visible summary footer에 한 줄 추가 — "※ 현재 코퍼스 기준 참고 근거이며 최종 설계판단은 아닙니다."
- **작업량**: one-liner, 리팩터 0.

### P3 — issue_type 근사 매핑 (로드맵)
- **근거**: 축력/휨/조합응력/전단 NG를 모두 `strength_exceeded` 한 버킷으로 라우팅. 검색 성능을 위한 근사일 뿐 정밀 조항 구분 아님. ([pipeline.py](../mcp-server/core/kds_rag/pipeline.py) `ISSUE_TYPE_KEYWORDS`)
- **제안**: limit_state별 세분 버킷 추가(vocabulary 확장) 또는 governing ratio를 evidence와 명시 연결. UI에 "근사 매핑" 주석.
- **작업량**: 중 (vocabulary + 회귀 테스트).

### P4 — AISC proxy 단정 위험 (로드맵, R4 연동)
- **근거**: 현재 AISC 360-22 chunk가 KDS 14 31 00 / 41 31 00 자리 임시 참조. ([explainer.py](../mcp-server/core/recommendation/explainer.py) `aisc_temporary_reference` dedupe)
- **제안**: KDS 원문 ingest 후 자동 해소(코드 변경 0). 그 전까지는 P2 라벨 + collapsible 내 경고로 advisory 유지.

---

## 연구자 관점

| # | 항목 | 상태 |
|---|------|------|
| R1 | 실제 KDS 코퍼스 retrieval benchmark 부재 | 🔜 로드맵 |
| R2 | usability 평가 부재 | 🔜 로드맵 |
| R3 | 부재 ratio → clause provenance chain 약함 | 🔜 로드맵 |
| R4 | AISC 임시참조 제거/명시 선언 | 🔜 로드맵 (P4 연동) |

### R1 — Retrieval benchmark
- **근거**: 현재 테스트는 synthetic chunk 중심. 검색 품질 wiring은 검증됐으나 실측 지표 없음.
- **제안**: 실제 KDS PDF 코퍼스로 recall@k / citation accuracy / clause-level precision 측정. 회귀 케이스를 골든셋으로 고정.
- **작업량**: 대 (코퍼스 구축 + 평가 하니스).

### R2 — Usability 평가
- **근거**: 인용 표시 방식이 신뢰 판단에 미치는 영향 미측정.
- **제안**: inline citation vs collapsible, LLM prefix 유무, RAG 미연결 경고 유무를 실무자 대상 비교. 신뢰·과신·검토시간 측정.
- **작업량**: 대 (사용자 연구 설계).

### R3 — Provenance chain
- **근거**: member ratio → issue_type → query → evidence 흐름이 암묵적. 어떤 ratio/limit_state가 어떤 조항을 트리거했는지 추적 약함.
- **제안**: evidence에 트리거 limit_state/ratio를 명시 부착. 감사용 별도 event log(provider-visible history와 분리)로 evidence 영구 보관 — P1 fix와 자연 연동.
- **작업량**: 중.

### R4 — AISC 임시참조 (P4 참조)

---

## 다음 빠른 후속 (이 PR 직후 권장)

1. **R3 audit log**: P1에서 history 밖으로 뺀 evidence를 별도 감사 로그에 저장 → provenance + 재현성.
2. **P3 vocabulary 세분화**: shear 외 strength 하위(축력/휨/조합) 구분 검색.

## 후속 phase (별도)

- R1 retrieval benchmark + 골든셋
- R2 usability study
- KDS 14 31 00 / 41 31 00 원문 코퍼스 ingest → AISC 임시참조 자동 해소
