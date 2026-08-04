# 결과 해설 LLM — 한국어 "종합검토의견" register & few-shot 가이드

> **역할 #3 — Narrative Interpreter (설계 결과 해설 LLM)** 의 프롬프트 자산.
> 대상 모듈(예정): `mcp-server/core/narrative_interpreter.py`
> 모델: `claude-opus-4-8` / v1 범위: `summary_ko` · `summary_en` 만 생성 (한·영 동시)
> 머신리더블 버전: [`fewshot_examples.json`](fewshot_examples.json)

---

## 0. 이 문서의 목적

규칙기반 해석기([result_interpreter.py](../../mcp-server/core/result_interpreter.py))와 결정론적 설계검토([design_check.py](../../mcp-server/core/design_check.py))가 **숫자·판정을 모두 확정한 뒤**, 그 결과를 한국 구조 실무의 **"종합결론/검토의견"** 문체로 자연스럽게 풀어주는 LLM의 시스템 프롬프트에 들어갈 내용이다. LLM은 **숫자·판정을 만들지 않는다** — 받은 facts를 문장으로만 엮는다.

핵심 원칙(이미 `report_ux.md`에 명시): **"LLM is explanation-only, not judgment."**

---

## 1. 우리가 모사하는 섹션

한국 구조계산서/구조검토서 표준 목차:

```
1 설계개요  2 적용기준  3 사용재료  4 하중조건  5 하중조합  6 해석모델
7 부재검토(내력비)  8 층간변위검토  9 고유치/동적해석  10 변위·처짐
(11 기초)  →  12 종합결론 / 검토의견   ★ 이 섹션의 prose만 LLM이 작성
```

1~11장은 표·수식·그래프. **12장만 산문**이며, 앞 장들의 OK/NG를 한국어 공학체로 요약·단정한다 — 이게 우리 `summary_ko/en`이 해야 할 일이다.

---

## 2. 문체 규칙 (register rules) — 시스템 프롬프트 핵심

1. **명사형 종결**: `~됨 / ~함 / ~확인됨 / ~판단됨 / ~사료됨 / ~검토됨 / ~요구됨` (보고서 문어체. "확인되었다"보다 명사형 종결이 더 격식).
2. **판정엔 항상 근거 병기**: 판정 문장에는 기준(KDS 조항) + 수치를 함께 쓴다. **단, 수치는 입력 `facts`의 문자열을 글자 그대로 복사만 한다 — 새 숫자 생성 금지.**
3. **무주어 3인칭**: 작성자("본인", "저희")를 드러내지 않는다. 1인칭·감정·과장·이모지 금지.
4. **OK 2단 구조**: "…(기준)을 만족하는 것으로 확인됨" → "…구조적으로 안전한 것으로 판단됨".
5. **NG 구조**: "…(기준)을 초과하여, …(보강/단면증대/시스템변경)이 필요한 것으로 검토됨".
6. **길이**: 2~3문장. 부재내력 → 층간변위 → (있으면)동적 → 종합판정 순으로 한 흐름에 엮는다.
7. **조항 인용 한계**: KDS **조항 번호를 새로 지어내지 않는다.** facts에 들어온 조항만 인용하고, 없으면 "적용 설계기준" 같은 총칭을 쓴다. (조항 인용의 정본은 결정론적 design_check + Qwen KDS-RAG 몫.)

---

## 3. 우리 심각도 ↔ 판정 어조 매핑

`result_interpreter`의 4단계(최대 활용률 기준: safe `<0.7` / marginal `0.7–1.0` / moderate `1.0–1.3` / severe `>1.3`)를 종합결론 어조에 매핑:

| severity | 의미 | 종합판정 어조 | 마무리 정형구 |
|----------|------|--------------|--------------|
| **safe** | 전 항목 통과, 여유 충분 | 단정적 긍정 | "…구조적으로 안전한 것으로 판단됨." |
| **marginal** | 통과하나 여유 부족 | 긍정 + 조건 단서 | "…기준을 만족하나, 하중 조건 변경 시 재검토가 필요한 것으로 판단됨." |
| **moderate** | 일부 부적합, 보강으로 해소 | 부적합 + 해결책 | "…단면 증대(보강)로 기준을 만족할 수 있는 것으로 판단됨." |
| **severe** | 광범위 부적합 / 연약층 | 부적합 + 시스템 차원 | "…구조시스템 차원의 재검토가 요구됨." |

---

## 4. few-shot 예시 (입력 facts → 출력 summary_ko / summary_en)

> ⚠️ **아래 숫자는 예시(illustrative)일 뿐이다.** 런타임에서 LLM은 입력 `facts`의 문자열을 그대로 복사하며, 예시 숫자를 재사용하거나 새 숫자를 만들지 않는다. 머신리더블 형태는 [`fewshot_examples.json`](fewshot_examples.json) 참조.

입력 `facts` 스키마(요약): `severity`, `severity_label_ko`, `max_drift_ratio`(허용기준 대비, 문자열), `max_interaction_ratio`(문자열), `governing_check`, `governing_failure_mode`, `T1_Ta`, `first_mode_direction`, `ng_stories`(int), `ng_members`(int), `diagnosis.primary_cause_ko`, `suggestions[0]`(type/target/current/recommended).

### 예시 A — safe

```json
{ "severity": "safe",
  "max_interaction_ratio": "0.62 (보 #214, 3층)",
  "max_drift_ratio": "0.48 (3층, X방향)",
  "governing_check": "부재 내력 (0.62)",
  "T1_Ta": "1.05s / 1.10s (비 0.95)", "first_mode_direction": "TRAN-X",
  "ng_stories": 0, "ng_members": 0 }
```
- **summary_ko**: "구조해석 및 설계검토 결과, 전 부재의 최대 내력비는 0.62(보 #214, 3층)로 1.0 이하이며, 최대 층간변위비는 허용기준의 0.48 수준으로 KDS 41 17 00의 허용층간변위 이내로 확인됨. 1차 고유주기(1.05초)는 경험식 추정치(1.10초)와 부합하여 동적 특성도 적정함. 따라서 본 구조물은 적용 설계기준이 요구하는 강도·사용성·내진성능을 모두 만족하여 구조적으로 안전한 것으로 판단됨."
- **summary_en**: "Structural analysis and design checks confirm a maximum member demand-to-capacity ratio of 0.62 (beam #214, Story 3), below 1.0, and a maximum story-drift ratio of 0.48 of the allowable, within the limit of KDS 41 17 00. The fundamental period (1.05 s) agrees with the empirical estimate (1.10 s), indicating adequate dynamic behavior. The structure therefore satisfies the strength, serviceability, and seismic performance required by the applied design code and is judged structurally safe."

### 예시 B — marginal

```json
{ "severity": "marginal",
  "max_interaction_ratio": "0.94 (기둥 #C12, 2층)",
  "max_drift_ratio": "0.88 (2층, Y방향)",
  "governing_check": "부재 내력 (0.94)",
  "T1_Ta": "1.40s / 1.05s (비 1.33)", "first_mode_direction": "TRAN-Y",
  "ng_stories": 0, "ng_members": 0 }
```
- **summary_ko**: "설계검토 결과 전 항목이 허용기준을 만족하나, 최대 내력비가 0.94(기둥 #C12, 2층), 최대 층간변위비가 허용기준의 0.88 수준으로 여유가 크지 않음. 1차 고유주기(1.40초)가 경험식(1.05초) 대비 다소 길어 횡강성 여유가 제한적임. 현 설계는 기준을 만족하나, 하중 조건 변경 시 재검토가 필요한 것으로 판단됨."
- **summary_en**: "All checks satisfy the allowable limits, but with limited margin: the maximum member ratio is 0.94 (column #C12, Story 2) and the maximum story-drift ratio reaches 0.88 of the allowable. The fundamental period (1.40 s) is somewhat longer than the empirical estimate (1.05 s), leaving restricted lateral-stiffness margin. The current design meets the code but would require re-evaluation if the loading conditions change."

### 예시 C — moderate

```json
{ "severity": "moderate",
  "max_interaction_ratio": "1.12 (보 #214, 3층)",
  "max_drift_ratio": "0.92 (3층, X방향)",
  "governing_check": "부재 내력 (1.12)", "governing_failure_mode": "휨-축력 상호작용",
  "ng_stories": 0, "ng_members": 4,
  "diagnosis": { "primary_cause_ko": "부재 내력 부족" },
  "suggestions": [{ "type": "section_upgrade", "target": "보", "current": "H-400x200", "recommended": "H-450x200" }] }
```
- **summary_ko**: "설계검토 결과 층간변위는 허용기준 이내(최대 0.92)이나, 일부 부재의 내력비가 1.0을 초과하여(최대 1.12, 보 #214, 3층) 4개 부재가 부적합으로 검토됨. 주된 원인은 휨-축력 상호작용에 대한 부재 내력 부족임. 해당 부재의 단면을 H-400x200에서 H-450x200으로 증대하면 기준을 만족할 수 있는 것으로 판단됨."
- **summary_en**: "Story drifts remain within the allowable limit (max 0.92), but several members exceed a demand-to-capacity ratio of 1.0 (max 1.12, beam #214, Story 3), so 4 members are found non-compliant. The governing cause is insufficient member capacity under combined flexure–axial interaction. Increasing the affected sections from H-400x200 to H-450x200 is expected to satisfy the criteria."

### 예시 D — severe

```json
{ "severity": "severe",
  "max_interaction_ratio": "1.28 (기둥 #C03, 1층)",
  "max_drift_ratio": "1.45 (1층, X방향)",
  "governing_check": "층간변위 (1.45)",
  "ng_stories": 2, "ng_members": 9,
  "diagnosis": { "primary_cause_ko": "연약층(1층)", "contributing_factors_ko": ["횡강성 부족"] },
  "suggestions": [{ "type": "system_change" }] }
```
- **summary_ko**: "설계검토 결과 1층 X방향 층간변위비가 허용기준을 크게 초과하고(1.45), 2개 층·9개 부재가 부적합으로 검토됨. 1층의 층간변위가 인접 층 대비 집중되어 연약층(soft story) 거동이 지배적이며, 횡강성 부족이 주된 원인으로 판단됨. 단면 보강만으로는 해소가 어려워, 횡력저항시스템의 변경 또는 추가(가새·전단벽 등)를 포함한 구조시스템 차원의 재검토가 요구됨."
- **summary_en**: "The Story-1 X-direction drift ratio substantially exceeds the allowable limit (1.45), and 2 stories and 9 members are non-compliant. Drift concentrates at Story 1 relative to adjacent stories, indicating a governing soft-story mechanism driven primarily by insufficient lateral stiffness. Section upgrades alone are unlikely to resolve this; a system-level revision adding or changing the lateral-force-resisting system (e.g., braces or shear walls) is required."

---

## 5. 판정 문장 은행 (seed sentence bank)

> 출처: **verbatim**은 KISTI ScienceON 공개 학술자료(아래 §6) — 내진성능평가/보강 맥락이라 종결어미·톤이 신축 구조계산서 종합결론과 동일 register. **재구성(reconstructed)**은 KDS 14 31 00 / KDS 41 17 00 용어 + 실무 정형구로 만든 것(저작권 무관, 우리 주제에 정확히 일치). **프롬프트 few-shot에는 재구성·예시문(§4)을 우선 사용**하고, verbatim은 "이 register가 실제임"을 보증하는 근거로 둔다.

### [부재내력-OK] (재구성)
- "검토 결과, 주요 보·기둥 부재의 최대 응력비(소요강도/설계강도)는 1.0 이하로, 전 부재가 KDS 14 31 00의 강도기준을 만족하는 것으로 확인됨."
- "각 부재의 설계강도가 소요강도를 상회하여, 강도 측면에서 구조적으로 안전한 것으로 판단됨."
- "기둥·보 및 가새 부재의 단면 검토 결과 모두 허용기준 이내로, 부재 안전성에 이상이 없는 것으로 검토됨."

### [부재내력-NG]
- *(verbatim)* "1층 기둥하부에서 모멘트가 집중되어 먼저 항복에 도달함." — TRKO201400028832
- *(verbatim)* "…X, Y방향 모두 최대내력 이후에 급격한 내력저하와 더불어 파괴에 이르는 것으로 나타나 내진성능을 만족시키지 못하는 것으로 판정되었다." — JAKO201113253030854
- *(재구성)* "일부 부재에서 응력비가 1.0을 초과하여, 해당 부재의 단면 증대 또는 보강이 필요한 것으로 검토됨."

### [층간변위-OK] (재구성)
- "각 층의 설계층간변위는 KDS 41 17 00에서 규정한 허용층간변위 이내로 검토되어, 횡변위에 대한 사용성 및 내진 요구조건을 만족하는 것으로 판단됨."
- "X·Y 양 방향 모두 최대 층간변위비가 허용기준 이내로 나타나, 횡력저항시스템이 충분한 강성을 확보한 것으로 확인됨."

### [층간변위-NG]
- *(verbatim)* "정규화된 지진파를 이용한 분석결과 1층에서 층간변위비 1.5% 이상을 나타냄." — TRKO201400028832
- *(재구성)* "OO층 X방향 층간변위가 허용층간변위를 초과하여, 횡력저항시스템의 강성 보강이 요구되는 것으로 검토됨."

### [고유주기/동적]
- *(verbatim)* "각 모멘트골조는 FEMA 356의 소성회전각 허용기준을 상회하는 것으로 나타남." — TRKO201400028832
- *(verbatim)* "강재 이력댐퍼를 사용한 결과 규준에서 정하고 있는 밑면 지진력 보다 큰 내력을 나타냈으며, 힌지가 건물전체에 고르게 발생하여 연성능력을 향상시키는 것으로 나타났다." — JAKO201113253030854
- *(재구성)* "고유치 해석 결과 1차 고유주기는 O.OO초로 산정되었으며, 질량참여율이 양 방향 모두 90% 이상으로 동적해석의 유효성을 만족함."
- *(재구성)* "고유주기 및 모드형상 검토 결과 비틀림 거동이 지배적이지 않아, 평면 비정형성에 따른 추가 검토는 불필요한 것으로 판단됨."

### [종합판정-OK] (재구성)
- "이상의 검토 결과, 본 건축물은 적용 설계기준(KDS)에서 요구하는 강도·사용성·내진성능을 모두 만족하여 구조적으로 안전한 것으로 판단됨."
- "검토 결과 본 구조물은 내진등급 OO에 요구되는 내진성능을 확보한 것으로 사료됨."
- "따라서 본 구조물은 제반 설계기준을 만족하며, 구조 안전성에 이상이 없는 것으로 판단됨."

### [종합판정-조건부]
- *(재구성)* "현행 기준 대비 일부 항목이 허용기준을 초과하므로, 해당 부위에 대한 보강을 전제로 구조 안전성을 확보할 수 있는 것으로 판단됨."
- *(출처근접 재구성)* "본 건물은 Y방향은 추가 보강이 불필요하나, X방향은 충분한 내진성능 확보를 위해 추가 보강이 필요한 것으로 검토됨."

### [종합판정-NG]
- *(verbatim)* "최종지진피해 판정결과는 X, Y 방향 모두 '대규모 피해'라는 결과를 얻었다." — JAKO201113253030854
- *(재구성)* "검토 결과 본 구조물은 요구되는 내진성능을 확보하지 못하는 것으로 판정되어, 내진보강이 필요한 것으로 사료됨."

---

## 6. 출처 & 라이선스

**Verbatim 출처 (전부 무료·공개, KISTI ScienceON):**
- 「학교 건축물의 내진성능평가 및 보강설계 사례」 — `JAKO201113253030854`
  (https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=JAKO201113253030854)
- 「내진설계 되지 않은 학교 건물의 내진성능평가 및 보강방법개발」 — `TRKO201400028832`
  (https://scienceon.kisti.re.kr/srch/selectPORSrchReport.do?cn=TRKO201400028832)

**목차/형식 표준 (무료, 공공저작물):**
- 학교시설 내진성능평가 및 보강 매뉴얼(교육부) §4.2.7 층간변위 검토 · §4.2.8 결과의 판정 ← 표준 판정 문구 1차 출처
- 기존시설물(건축물) 내진성능 평가요령(국토안전관리원) — CODIL `OTKCMA130067`
- 국토안전관리원 정밀안전점검 표준서식(HWPX) — data.go.kr `15140381`

**어휘 참고만 (유료, 직접 인용 금지):**
- 《예제로 배우는 강구조 설계》, 《강구조 접합부 설계예제집》(대한건축학회·한국강구조학회·KSEA)

> **라이선스 메모:** 프롬프트에 실제로 투입하는 few-shot(§4)·재구성 문장(§5)은 모두 KDS 용어 기반 자체 작성이라 저작권 무관. verbatim 인용은 register 검증용 근거이며, 시스템 프롬프트에 다량 박아 넣기보다 출처를 남겨 추적성만 확보한다.

**한계:** 강구조 신축 건물 구조계산서의 "응력비-OK / 종합판정-OK" verbatim은 무료 웹에서 확보 실패(유료 예제집·비공개 인허가 문서에 존재). 그래서 OK 계열은 재구성으로 채웠다. 확보된 verbatim은 대부분 내진성능평가(NG·조건부·동적) 맥락이나 종결어미·톤은 동일 register라 유효하다.

---

## 7. 금지사항 (프롬프트에 포함할 anti-hallucination 조항)

- `facts`에 **글자 그대로 존재하지 않는 숫자/비/%/조항번호를 절대 쓰지 않는다.** 필요하면 숫자를 빼고 문장을 쓰거나 문장을 생략한다.
- **OK/NG 판정을 뒤집거나 완화하지 않는다.** severity가 moderate/severe인데 "안전/적합"이라 단정하지 않는다.
- **KDS 조항을 새로 인용하지 않는다.** facts에 없으면 "적용 설계기준"으로 총칭.
- 충실히 작성할 수 없으면 `confidence: "low"` 로 표기 → 호출부가 기존 `_generate_summary()` 템플릿으로 폴백.
