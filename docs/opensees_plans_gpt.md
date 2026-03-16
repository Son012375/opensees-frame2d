# 아키텍처 리뷰: AI Structural Engineer

> **작성일**: 2026-03-11
> **작성자**: Claude (코드베이스 기반 분석)
> **대상 repo**: `opensees-MCP` (35,700줄, Python)

---

## 1. 현재 아키텍처는 이 방향을 지원하는가?

**결론: 80% 지원한다. 핵심 누락 계층은 하나다.**

현재 시스템의 데이터 흐름:

```
[JSON config]  →  BuildingModel  →  LoadGenerator  →  frame_3d  →  결과+리포트
                      ↑                   ↑
                   from_ifc()         Supabase DB
```

제안된 시스템의 데이터 흐름:

```
[IFC + 자연어]  →  LLM 해석  →  structured config  →  BuildingModel  →  ...동일...
                      ↑                                     ↑
                 의도 파악/파라미터 추출              from_ifc() + config merge
```

**차이점은 정확히 하나: `자연어 → structured config` 변환 계층이 없다.**

나머지는 이미 구현되어 있다:

| 구성요소 | 상태 | 근거 |
|----------|------|------|
| IFC → 기하형상 | ✅ 완료 | `ifc_parser.py` — 814줄, 벽/기둥 기반 |
| BuildingModel IR | ✅ 완료 | `building_model.py` — `from_ifc()` + `from_json()` |
| 용도→하중 매핑 | ✅ 완료 | `load_generator.py:120-141` — `USAGE_TO_LIVE_LOAD_KEY` 32개 매핑 |
| 지역→위험계수 | ✅ 완료 | Supabase `hazard_region_values` 2,290건 (229개 시군구) |
| 자동 하중 생성 | ✅ 완료 | `generate_all_loads()` — DL/LL/EQ/Wind + 18개 조합 |
| 가정 추적 | ✅ 완료 | `assumption_tracker.py` — 출처별 분류 |
| 3D 해석 | ✅ 완료 | `frame_3d.py` — 6-DOF, Corotational |
| HTML 리포트 | ✅ 완료 | `visualization_3d.py` — 1,741줄 |
| MCP 도구 13개 | ✅ 완료 | `server.py` — Pydantic 스키마 정의 |
| LLM ↔ 결정론 분리 | ✅ 설계됨 | LLM은 해석에 관여 안함 |

---

## 2. LLM과 결정론적 시스템의 분리가 올바른가?

**올바르다. 이미 코드에서 명확히 분리되어 있다.**

```python
# server.py의 BuildingAnalysisInput — LLM의 출력 경계
class BuildingAnalysisInput(BaseModel):
    config: dict = Field(description="Building configuration")
    # config 내부: stories, bays_x, bays_y, region, site_class, ...

# load_generator.py — 이 안에는 LLM 호출이 전혀 없다
def generate_all_loads(model: BuildingModel) -> dict:
    # DB 조회 → 수식 계산 → 결과 반환
    # "추정"이나 "판단"이 아닌, 오직 DB 값 + KDS 공식
```

**핵심 설계 원칙이 코드에 이미 반영됨:**
- `load_generator.py`에 Claude API import 없음
- `frame_3d.py`에 LLM 관련 코드 없음
- DB 조회 실패 시 `FALLBACK_LIVE_LOADS` 상수 사용 (LLM에 질문하지 않음)
- `assumption_tracker.py`가 모든 값의 출처를 `user_input` / `ifc_detected` / `default` / `fallback`로 분류

**한 가지 주의점**: `webapp/backend/app/core/claude_service.py`에서 Claude를 사용하지만, 이건 **결과 해석**(자연어 설명)용이지 계산에 사용하는 것이 아님. 올바른 분리.

---

## 3. 누락된 핵심 구성요소

### 3.1 자연어 → Structured Config 변환기 (Critical — 미구현)

**현재**: 사용자가 JSON config를 직접 작성해야 함
**필요**: "1층 근생, 2~5층 오피스, 부산 해운대" → `BuildingAnalysisInput.config`

구체적으로 필요한 스키마:

```python
# 현재 BuildingAnalysisInput이 기대하는 config 형태
config = {
    "stories": [
        {"height": 4.0, "usage": "retail"},       # "근생" → "retail"
        {"height": 3.5, "usage": "office"},        # "오피스" → "office"
        {"height": 3.5, "usage": "office"},
        {"height": 3.5, "usage": "office"},
        {"height": 3.5, "usage": "office"},
    ],
    "bays_x": [8.0, 8.0],     # ← IFC에서 추출
    "bays_y": [8.0],           # ← IFC에서 추출
    "region": "부산 해운대구",   # "부산 해운대" → DB 키
    "site_class": "S3",         # 미언급 → 기본값 or LLM이 추론 요청
    "importance": "II",         # 미언급 → 기본값
}
```

**LLM이 추출해야 하는 정보:**

| 자연어 표현 | 추출 결과 | 변환 규칙 |
|-------------|----------|----------|
| "1층은 근린생활시설" | `stories[0].usage = "retail"` | occupancy.json 매핑 |
| "2층부터 5층까지 오피스" | `stories[1:5].usage = "office"` | 범위 해석 |
| "옥상에 기계실" | `stories[-1].usage = "mechanical"` | "옥상" = 최상층 |
| "부산 해운대" | `region = "부산 해운대구"` | 229개 시군구 매칭 |

**이미 존재하는 지원 인프라:**
- `data/mapping/occupancy.json` — 한국어↔DB 키 매핑 32개
- `hazard_region_values` — 229개 시군구 목록 (fuzzy match 가능)
- `assumption_tracker.py` — 추론 결과 추적

### 3.2 IFC + 자연어 충돌 해결 로직 (Partial — 보강 필요)

**현재 코드의 merge 전략** (`building_model.py:202-247`):

```
IFC 기하형상 (height, bays) → 무조건 IFC 우선
사용자 config (usage, region) → 무조건 config 우선
단면 (sections) → config 있으면 config, 없으면 IFC
```

**누락된 충돌 시나리오:**
- IFC에서 6개 층 감지, 사용자가 "5층 건물"이라고 말한 경우
- IFC에서 경간 8m 감지, 사용자가 "6m 경간"이라고 입력한 경우
- 해결 방안: **IFC 기하형상이 항상 우선, 불일치 시 경고 생성**

### 3.3 부재 설계 검토 모듈 (미구현)

**현재**: `server.py:132-195`에 기본 응력비(`σ/fy`)만 존재

**필요한 KDS 41 10 00 검토 항목:**

| 검토 | 현재 | 필요 |
|------|------|------|
| 휨 D/C ratio (Mu/ΦMn) | ✗ | KDS 41 10 00 §6 |
| 전단 D/C ratio (Vu/ΦVn) | ✗ | KDS 41 10 00 §7 |
| 압축+휨 상호작용 | ✗ | KDS 41 10 00 §8 |
| 횡비틀림좌굴 (LTB) | ✗ | KDS 41 10 00 §6.2 |
| 층간변위 한계 | ✗ | KDS 41 17 00 §4.3 |
| P-Delta 안정성 지수 (θ) | ✗ | θ = ΣPΔ/Vh < 0.10 |

### 3.4 도구 오케스트레이션 / 플래너 (미구현)

**현재**: 모든 실행이 고정 시퀀스 (`analyze_building` → 3단계 파이프라인)

**필요**: 결과에 따른 조건부 분기

```
IF 층간변위 > 한계 THEN "기둥 단면 증가 필요" 제안
IF D/C > 1.0 THEN 해당 부재 단면 업그레이드 검색
IF 가정 경고 다수 THEN 사용자에게 확인 요청
```

현재는 이런 루프가 없다. `analyze_building`은 **1회 실행 후 결과 반환**으로 끝남.

---

## 4. 기술적 리스크

### Risk 1: 자연어 파싱의 구조공학 도메인 특수성

"근생"이 `retail`인지 `commercial`인지, "옥상 기계실"이 별도 층인지 옥탑인지 — 구조공학 문맥에서의 해석이 필요. 범용 NLP로는 부족하고 **도메인 프롬프트 + 검증 스키마**가 필수.

**완화 방안**: `occupancy.json`에 한국어 동의어를 충분히 등록하고, LLM 출력을 스키마 검증 후 사용자 확인 단계를 거침.

### Risk 2: 229개 시군구 이름 매칭

"부산 해운대" → "부산광역시 해운대구"로의 fuzzy match가 필요. "해운대"만으로도, "부산 해운대"로도, "해운대구"로도 검색 가능해야.

**완화 방안**: `hazard_region_values`에서 `region_name` 목록 추출 → 유사도 매칭 + 다중 후보 시 사용자 선택.

### Risk 3: IFC 품질 편차

Revit, ArchiCAD, Tekla 등 BIM 소프트웨어별 IFC 출력 품질이 다름. 현재 파서는 Revit IFC4에 최적화되어 있음 (`ifc_parser.py`).

**완화 방안**: IFC 파싱 실패 시 JSON config으로 fallback하는 경로는 이미 존재 (`from_json()`).

### Risk 4: Euler-Bernoulli 한계

현재 solver는 전단변형을 무시하는 Euler-Bernoulli 보 이론 사용. 짧은 보(span/depth < 10)에서 부정확.

**완화 방안**: 벤치마크에서 Midas와 "shear deformation OFF" 조건으로 일치 확인 완료. 향후 Timoshenko 요소 전환 가능.

### Risk 5: 단일 실행 파이프라인의 설계 반복 한계

실무에서는 해석 → 검토 → 단면 변경 → 재해석을 반복. 현재는 1회 실행만 가능.

**완화 방안**: Phase 2에서 설계 반복 루프 구현 (아래 로드맵 참조).

---

## 5. 구체적 질문에 대한 분석

### Q1. 자연어 → Structured Parameter 스키마

**LLM 출력과 LoadGenerator 사이에 존재해야 하는 스키마는 이미 정의되어 있다:**

`BuildingAnalysisInput.config` (`server.py` line 330-357)가 바로 그 스키마다.

```python
# LLM이 생성해야 하는 JSON 출력
{
    "stories": [{"height": float, "usage": str, "dead_load_finish": float}],
    "bays_x": [float],
    "bays_y": [float],
    "region": str,              # 229개 시군구 중 하나
    "site_class": str,          # S1~S5
    "importance": str,          # "특", "I", "II"
    "seismic_system": str,      # "ordinary_moment_frame" 등
    "exposure_category": str,   # "A"~"D"
    "column_section": str,      # "H-400x400x13x21" (IFC에서 옴)
    "beam_x_section": str,      # (IFC에서 옴)
    "auto_combinations": bool,  # true
}
```

**추가로 필요한 것**: IFC에서 자동 채워지는 필드와 LLM이 채워야 하는 필드의 **명시적 구분**.

| 필드 | 출처 | 비고 |
|------|------|------|
| `stories[].height` | **IFC** | IFC에서 자동 추출 |
| `bays_x`, `bays_y` | **IFC** | IFC에서 자동 추출 |
| `column_section`, `beam_section` | **IFC** | IFC에서 감지 가능 |
| `stories[].usage` | **LLM** | 자연어에서 추출 |
| `region` | **LLM** | 자연어에서 추출 |
| `site_class` | **LLM 또는 기본값** | 미언급 시 S3 |
| `importance` | **LLM 또는 기본값** | 미언급 시 II |
| `dead_load_finish` | **LLM 또는 기본값** | 미언급 시 1.0 kN/m² |

### Q2. IFC + 자연어 병합 로직

현재 `building_model.py`의 `from_ifc(ifc_path, config)` 메서드가 이미 병합한다:

```
IFC → 기하형상 (height, bays, sections)
config → 의미 정보 (usage, region, site_class)
```

**우선순위 규칙 (현재 코드 기준):**
1. 사용자 config의 명시적 값 → 최우선
2. IFC 감지 값 → 기하형상 우선
3. BuildingModel 기본값 → fallback

**보강이 필요한 부분:**
- 층수 불일치 경고 (IFC 층수 ≠ 자연어 층수)
- 기하형상 불일치 경고 (예: IFC 8m bay vs 사용자 6m 언급)
- 이들은 `assumption_tracker`에 `conflict` 카테고리로 추가 가능

### Q3. 가정 추적

**이미 구현되어 있다.** `assumption_tracker.py`가 모든 파라미터의 출처를 추적:

```python
# 현재 추적하는 카테고리:
"user_input"    # 사용자가 명시적으로 제공
"ifc_detected"  # IFC에서 자동 감지
"default"       # 시스템 기본값 사용
"calculated"    # 공식으로 산출 (slab self-weight 등)
"fallback"      # DB 조회 실패 시 대체값
```

**추가 필요:**
- `"llm_inferred"` — LLM이 자연어에서 추론한 값 (예: "근생" → "retail")
- `"conflict_resolved"` — IFC/자연어 충돌 시 어느 쪽을 채택했는지

### Q4. 도구 오케스트레이션

**현재: MCP 프로토콜 자체가 오케스트레이터다.**

MCP 설계에서 Claude가 도구를 호출하는 순서를 스스로 결정한다. 13개 도구가 등록되어 있고, Claude는 문맥에 따라:

```
1. get_section_properties("H-400x200x8x13")  ← 단면 확인
2. get_hazard_values("부산 해운대구", "seismic_zone")  ← 지역 위험도
3. analyze_building(config)  ← 전체 해석 실행
```

이 순서로 호출할 수 있다. **별도 플래너가 불필요한 이유**: MCP 프로토콜에서 LLM이 플래너 역할을 이미 수행.

**다만, 현재 `analyze_building`이 내부적으로 DB 조회를 포함**하므로, 사실상 대부분의 경우 단일 도구 호출로 충분:

```
Claude: analyze_building(config)  ← 이 안에서 자동으로:
  → LoadGenerator가 Supabase 조회
  → 지진/풍하중 자동 계산
  → 18개 하중조합 자동 생성
  → 3D 해석 실행
  → HTML 리포트 생성
```

**향후 필요한 오케스트레이션**은 결과 기반 의사결정:
- "해석 결과 D/C > 1.0이면 단면 증가 후 재해석"
- 이건 MCP 도구가 아니라 **Claude의 추론 능력**으로 처리 가능

### Q5. 설계 검토 모듈

**별도 모듈 도입을 권장한다.**

현재 `verification.py`는 평형 검증만 수행(351줄). 부재 설계 검토는 없다.

제안하는 구조:

```
mcp-server/core/design_check.py (신규)
  ├── check_flexure(M, section, material)     → ΦMn, D/C
  ├── check_shear(V, section, material)       → ΦVn, D/C
  ├── check_combined(P, M, section)           → Interaction ratio
  ├── check_story_drift(drift, limit, Ie)     → Pass/Fail
  └── check_stability_index(P, Δ, V, h)      → θ, Pass/Fail
```

이 모듈은 `analyze_building` 결과에 자동 적용되어 리포트에 포함.

---

## 6. 권장 개발 로드맵

기존 코드 위에 **최소 변경으로 최대 효과**를 내는 순서:

### Phase 1: NL 파싱 계층 (1~2주)

**목표**: "1층 근생, 2~5층 오피스, 부산 해운대" → `BuildingAnalysisInput.config`

| 작업 | 변경 파일 | 규모 |
|------|----------|------|
| NL→config 변환 프롬프트 설계 | 신규: `prompts/building_config.py` | ~200줄 |
| 용도 매핑 확장 | `data/mapping/occupancy.json` | 기존 파일 보강 |
| 지역명 fuzzy match | 신규: `core/region_matcher.py` | ~100줄 |
| IFC+NL 충돌 경고 | `building_model.py` 수정 | ~50줄 |
| assumption_tracker에 `llm_inferred` 추가 | `assumption_tracker.py` 수정 | ~20줄 |

**이건 MCP 도구가 아니라 Claude의 프롬프트/시스템 메시지로 구현하는 것이 자연스럽다.** Claude가 `analyze_building` 호출 전에 자연어를 JSON으로 변환하는 역할.

### Phase 2: 설계 검토 모듈 (2~3주)

| 작업 | 변경 파일 | 규모 |
|------|----------|------|
| KDS 부재 설계 검토 | 신규: `core/design_check.py` | ~400줄 |
| 층간변위 한계 검토 | 위 파일에 포함 | — |
| 리포트에 D/C 테이블 추가 | `visualization_3d.py` 수정 | ~200줄 |
| MCP 도구 등록 | `server.py` 수정 | ~50줄 |

### Phase 3: 결과 해석 + 코드 참조 (1~2주)

| 작업 | 변경 파일 | 규모 |
|------|----------|------|
| 결과 요약 프롬프트 | 신규: `prompts/result_interpreter.py` | ~150줄 |
| KDS 조항 참조 생성 | `design_check.py`에 조항 번호 포함 | — |
| 자연어 결과 설명 템플릿 | Claude 시스템 프롬프트 | — |

### Phase 4: 설계 반복 루프 (3~4주)

| 작업 | 변경 파일 | 규모 |
|------|----------|------|
| 단면 업그레이드 검색 | `section_3d.py` 확장 | ~100줄 |
| 재해석 루프 | Claude 오케스트레이션 (MCP 다중 호출) | — |
| 수렴 판정 | `design_check.py` 확장 | ~100줄 |

---

## 7. 아키텍처 개선 제안

### 제안 1: Config Schema를 2단계로 분리

```
IFCConfig (기하형상 전용) ← IFC 파서가 생성
  + NLConfig (의미 정보 전용) ← LLM이 생성
  = BuildingConfig (병합) ← BuildingModel.from_merged()
```

현재는 하나의 `config` dict에 모든 것이 섞여 있어서 출처 추적이 어렵다. 분리하면 assumption_tracker와 자연스럽게 연동.

### 제안 2: 사용자 확인 단계 명시화

```
LLM 추론 결과:
  - 1층: "근생" → retail (LL=5.0 kN/m²)  ← 확인 필요
  - 위치: "부산 해운대" → 부산광역시 해운대구 (Sg=0.5)  ← 확인 필요
  - 지반: 미언급 → S3 (가정)  ← 경고 표시

사용자 확인 → 해석 실행
```

이 패턴은 `assumption_tracker`와 Streamlit UI에 자연스럽게 통합 가능.

### 제안 3: MCP 도구 구조는 현행 유지

`analyze_building`이 이미 "원클릭 해석"을 제공하므로, 별도 오케스트레이터를 만들 필요 없다. Claude가 MCP 프로토콜을 통해 직접 도구를 호출하는 현재 구조가 가장 깔끔하다.

---

## 8. 현재 구현된 코드베이스 상세

### 8.1 디렉터리별 코드 규모 (총 35,657줄)

| 카테고리 | 디렉터리 | 줄 수 | 비고 |
|----------|----------|------:|------|
| **핵심 엔진** | `mcp-server/core/` | **13,727** | solver 4종 + viz 2종 + 하중/IFC/스펙트럼 |
| **MCP 서버** | `mcp-server/` (core 제외) | **3,907** | server.py + 테스트 |
| **프론트엔드** | `streamlit_app/` | **2,439** | NL chat + IFC upload |
| **백엔드** | `webapp/` | **1,493** | FastAPI |
| **테스트** | `tests/` | **3,592** | benchmark 5 cases + stage 테스트 |
| **스크립트** | `scripts/` | **5,524** | DB 적재, 비교 도구 |
| **기타** | `agents/`, `adapters/`, `pipeline/`, `apply_mvp/` | **2,948** | Agent 파이프라인 + MVP |
| **opensees-ai-agent** | `opensees-ai-agent/` | **2,027** | 별도 agent |
| | **합계** | **35,657** | |

### 8.2 핵심 파일 Top 10

| 파일 | 줄 수 | 역할 |
|------|------:|------|
| `visualization.py` | 3,811 | 2D HTML 리포트 |
| `visualization_3d.py` | 1,741 | 3D HTML 리포트 |
| `frame_2d.py` | 1,449 | 2D 프레임 solver |
| `frame_3d.py` | 1,259 | 3D 프레임 solver |
| `server.py` | 1,141 | MCP 서버 (13 tools) |
| `benchmark/cases.py` | 1,069 | 벤치마크 5 cases |
| `ifc_parser.py` | 813 | IFC 파싱 |
| `load_generator.py` | 736 | KDS 자동 하중 |
| `continuous_beam.py` | 673 | 연속보 solver |
| `simple_beam.py` | 667 | 단순보 solver |

### 8.3 MCP 도구 목록 (13개)

| # | 도구명 | 입력 | 출력 |
|---|--------|------|------|
| 1 | `analyze_simple_beam` | span, load, section | 변위, 모멘트, 전단력 |
| 2 | `analyze_continuous_beam` | spans[], loads[], supports | 다경간 해석 결과 |
| 3 | `analyze_frame_2d` | stories[], bays[], loads | 2D 프레임 해석 |
| 4 | `analyze_frame_3d` | stories[], bays_x[], bays_y[], load_cases | 3D 프레임 해석 |
| 5 | `analyze_building` | config (전체 건물 정보) | 자동하중 + 3D 해석 + 리포트 |
| 6 | `get_design_loads` | param_type, keyword | KDS 설계하중 파라미터 |
| 7 | `get_load_combinations` | limit_state | 하중조합 계수 |
| 8 | `get_hazard_values` | region_name, hazard_type | 지역별 위험계수 |
| 9 | `get_design_spectrum` | region, site_class | 설계응답스펙트럼 Sa(T) |
| 10 | `get_section_properties` | section_name | 단면 물성 (A, Ix, Iy, J) |
| 11 | `get_material_properties` | material_name | 재료 물성 (E, fy, G) |
| 12 | `list_available_sections` | table_type | 단면 카탈로그 |
| 13 | `list_available_materials` | — | 재료 카탈로그 |

### 8.4 Supabase DB 현황 (3,048건)

| 테이블 | 건수 | 내용 |
|--------|------|------|
| `load_params` | 712 | 설계하중 파라미터 (DL 234, 내진 258, LL 79, 풍 49, 설 17, 조합 46, 기타) |
| `hazard_region_values` | 2,290 | 229개 시군구 × 10종 (Sg, V₀, 지진구역, PGA 7종) |
| `load_combo` | 46 | ULS 7식 + SLS 8식 하중조합 |

### 8.5 Midas Gen 벤치마크 검증 (완료)

| Case | 유형 | 결과 |
|------|------|------|
| 1 | 2D 단순보 (점하중) | MATCH — 변위, 모멘트 정확 일치 |
| 2 | 2D 1층 포탈프레임 | MATCH — 변위, 반력, 모멘트 일치 |
| 3 | 2D 3층 프레임 | MATCH — 변위, 층간변위, 모멘트 일치 |
| 4 | 3D 2층 프레임 | AGREE (~3%) — 3D 요소 정식화 차이 |
| 5 | 3D 5층 P-Delta | MATCH (<0.1%) — Linear 정확 일치, PDelta ~0.03% |

**총 112개 비교 메트릭**: 100 OK, 12 CHECK (Case 4 3D 차이)

### 8.6 세 가지 AI-Driven 경로

| 경로 | 구현 위치 | 역할 |
|------|-----------|------|
| **MCP 경로** | `mcp-server/` (17.6k줄) | Claude가 실시간으로 구조해석 수행 |
| **Agent 경로** | `agents/` + `pipeline/` (1k줄) | Claude가 KDS PDF를 자동 파싱해서 DB 구축 |
| **Apply 경로** | `apply_mvp/` + `core/` (3k줄) | DB 데이터로 자동 하중조합 → 해석 |

---

## 9. 요약

| 항목 | 평가 |
|------|------|
| **기존 코드의 방향 적합성** | ✅ 80% — 핵심 파이프라인 완성 |
| **LLM/결정론 분리** | ✅ 올바르게 설계됨 |
| **가장 큰 누락** | 자연어 → structured config 변환 계층 |
| **두 번째 누락** | KDS 부재 설계 검토 (D/C ratio) |
| **아키텍처 리스크** | 낮음 — 기존 구조 위에 증분 개발 가능 |
| **로드맵 예상 규모** | Phase 1~3: ~1,200줄 신규 코드 |

**핵심 메시지**: 이 시스템은 이미 `IFC → 하중 → 해석 → 리포트` 파이프라인이 완성되어 있다. "AI Structural Engineer"로 가려면 **앞단(NL 파싱)과 뒷단(설계 검토)만 추가**하면 된다. 중간의 결정론적 엔진은 건드릴 필요 없다.
