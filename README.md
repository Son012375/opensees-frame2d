# OpenSees-MCP: LLM-Integrated Structural Analysis System

자연어 기반 구조해석 시스템 — IFC/자연어 입력 → OpenSeesPy 해석 → KDS 설계검토 → 3D 시각화

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![OpenSeesPy](https://img.shields.io/badge/Engine-OpenSeesPy-orange)](https://openseespydoc.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Claude AI](https://img.shields.io/badge/AI-Claude_API-8B5CF6)](https://www.anthropic.com/)
[![Supabase](https://img.shields.io/badge/DB-Supabase-3ECF8E?logo=supabase)](https://supabase.com/)

---

## 1. Project Overview

### 1.1 Research Goal

사용자가 **한국어 자연어** 또는 **IFC 파일**로 건물 정보를 입력하면,
LLM(Claude)이 이를 구조해석 Config로 변환하고, OpenSeesPy로 해석을 수행하여
**KDS 기준 설계검토 결과**를 자연어로 피드백하는 End-to-End 시스템.

```
[자연어/IFC 입력] → [LLM 파싱] → [Config 생성] → [OpenSeesPy 해석]
    → [KDS/AISC 설계검토] → [3D 시각화 + HTML 리포트] → [자연어 해석 피드백]
```

### 1.2 System Pipeline

| 단계 | 구성요소 | 설명 |
|------|----------|------|
| 입력 | NL Resolver / IFC Parser | 자연어 또는 IFC → BuildingConfig 변환 |
| 하중생성 | Load Generator | KDS 41 12 00 기반 DL/LL/EQ/Wind 자동생성 |
| 해석 | OpenSeesPy (2D/3D) | 정적/동적 해석, P-Delta, 고유치 |
| 검토 | Design Check | KDS 층간변위 + AISC 360 부재강도 |
| 시각화 | 3D Editor + HTML Report | Three.js 인터랙티브 뷰 + 독립 HTML 리포트 |
| 피드백 | Result Interpreter | 심각도/진단/제안 자연어 해석 |

---

## 2. Supported Analysis Types (V2 웹앱 기준)

| 해석 유형 | 상태 | DOF | 핵심 기능 |
|-----------|------|-----|-----------|
| 3D Frame (V2) | Ready | 6 | 양방향 경간, Corotational 비선형, Rigid Diaphragm |
| Building Pipeline | Ready | 6 | IFC(V2)/Manual/NL → 자동하중 → 3D해석 → 설계검토 |
| Response Spectrum | Ready | 6 | KDS 설계응답스펙트럼, CQC/SRSS, RSA/ELF 비교 |

> **참고:** Simple Beam / Continuous Beam / 2D Frame 해석 엔진은 [mcp-server/core/](mcp-server/core/)에 그대로 남아있고 MCP tool로 접근 가능합니다. 웹앱에서는 V2 3D Editor에 집중합니다.

---

## 3. Key Features

### 3.1 Natural Language Input (NL Resolver)
- 한국어 자연어 → Claude API → BuildingIntent 추출 → Config 변환
- 30개 용도 매핑 (한국어 별칭 → canonical key)
- 복합용도 자동 분리 ("근생" → retail)
- 지역/중요도/층고/경간 자동 추론

### 3.2 IFC Integration

#### V1 (Grid-based)
- IFC 파일 업로드 → 벽/기둥 기반 자동 파싱
- 3D 와이어프레임 미리보기 (Three.js)
- 3단계 위자드: Upload → Geometry Preview → Config → Analyze
- 비정형 건물 지원 (L자형/T자형/Setback, Zone 기반)

#### V2 (Node-Element, `v2/node-element` branch)
- **IFC → 노드/요소 직접 추출** (ifcopenshell.geom 기반 글로벌 좌표)
- 비정형 건물 완벽 지원 (경사 부재, 불규칙 평면, setback)
- 요소별 개별 단면/재료/릴리즈 지정
- 보-기둥 접합 자동 검증 + 스냅
- 요소 자동 분할 (중간 노드 감지 시 Midas Gen 방식 분할)

**3D Model Editor (Midas Gen 유사)**:
- 좌측 도구 팔레트 (Geometry / Properties / Display 섹션)
- 노드 추가: 그리드 스냅 + XYZ 좌표 직접 입력 (N키)
- 요소 생성: 노드 2개 선택 → 유형/단면/재료 다이얼로그
- 삭제: 노드/요소 클릭 삭제 (연결 요소 경고)
- 이동: 드래그 (연결 요소 자동 추적) + 더블클릭 좌표 입력
- Beam Release: 6-DOF 육각형 UI (i-end/j-end 개별, 프리셋: Pin i/j/Both/Fixed)
- Support Conditions: 노드별 경계조건 (Fixed/Pinned/Roller + DOF 미리보기)
- Undo/Redo (Ctrl+Z/Y, 30단계)

**3D 뷰어 기능**:
- Wire/Solid Section 토글 (H/SHS/RHS/CHS/L/채널 6종 ExtrudeGeometry + 엣지 윤곽선)
- 노드 라벨: ID only / XYZ 좌표 (고해상도 Sprite)
- 마우스 보조선 (X/Y/Z 점선 가이드) + Ghost Node 미리보기
- 실시간 마우스 좌표 표시 (우측 하단)
- 편집 모드에서 우클릭 회전 + 스크롤 줌 + Shift 패닝

**해석 결과 시각화** (2026-04-15~17):
- SFD/BMD/Axial 3D 다이어그램 (폴리곤 면 + 외곽선 + i/j-end + 고점 라벨)
- BMD/SFD 스케일 슬라이더 (0.1×~10× 로그)
- Display Filter 패널 (Loads DL/LL/EQ/Wind, Story, Member Type, Node#/Member# 토글)
- 하중 화살표 (분포하중 보 위 배열 + 수평하중 층 중심)
- 반력 시각화 (지점 화살표 + kN 수치 라벨 + Values 토글 연동)
- Deformed shape 토글 + 스케일 슬라이더 (1×~500×, Auto)
- 부재력 hover tooltip (N/V/M 최대값, 단면 타입별 필드 분기: H→H,B / SHS→B,t / CHS→D,t)
- 개별 부재 Canvas 다이어그램 (N/V/M × 3, 클릭 시)
- Properties 탭 정리 (Results / Modal / DC)
- Story Drift 탭 ([3D Model] / [Story Drift] 뷰 전환, 수평 bar chart X/Y, KDS 허용치 자동전환 + 수동 오버라이드)
- **Drift Profile 차트** (2026-05-04): 높이방향 X/Y 꺾은선 + 영역 채우기 + 허용치 수직선, 층별 OK/NG 상세 테이블 (1/N + %, Max 행)

**데이터 Export** (2026-04-16~05-04):
- Excel Export (pandas + openpyxl): Wide Pivot 다중 시트 (Displacements / Member_Forces / Reactions / Envelope / Metadata)
- 케이스 그룹별 헤더 색상 (DL=시안, LL=주황, EQ=빨강, Wind=보라, Combo=교차 연녹)
- 2행 MultiIndex 헤더 + freeze pane
- **DXF Export** (ezdxf, 2026-04-20): 층별 평면도 + X/Y 입면도 한 도면에 자동 배치
  - 평면: GL/1F/2F... 수직 적층, 그리드선(중심선) + 축번호(X1/Y1) + 연속/전체 치수선 + 제목 박스
  - 입면: 라인별 수직 LINE(기둥) + 수평 LINE(보) + 층고 치수
  - 레이어 분리 (GRID/COL/BEAM/TEXT/DIM/FRAME) → AutoCAD/QCAD 호환

**편집 도구** (2026-04-15~17):
- 다중 선택 (Ctrl+클릭, 박스 드래그, Story 선택)
- 복사/미러/층 복사 (플로팅 드래그 패널)
- 개별 부재 단면 변경 ("이 부재만" + 재해석)
- 층별 단면 일괄 변경 (Story 체크박스 + [선택 층] 버튼, 하위층 기준, 프리뷰 하이라이트)
- 보-보 교차점 자동 분할 (2D line intersection)
- 프로젝트 저장/불러오기 (.v2proj, 해석 결과 포함, 재해석 불필요)

**다중 단면 타입 지원** (2026-04-17):
- IFC 단면 매핑 확장: SHS/RHS/CHS/L/채널 ProfileName 파싱 → Supabase DB 자동 매칭
- Solid Section(3D) 렌더링: H형강(I자), SHS(사각 중공), RHS, CHS(원형 중공), L(ㄱ자), 채널(ㄷ자)
- Solid/Wire 전환 시 mesh 동기화 (필터/선택/프리뷰 연동)
- API section_type + t_mm 필드 추가
- Solid 모드에서 Deformed shape/DC Colors 자동 동기화 (2026-05-04)
- 헤더 Solid 토글 추가 (3D 뷰 어디서든 즉시 전환 가능)

**해석 결과 ↔ 편집 워크플로우 개선** (2026-05-04):
- 해석 직전 자동 undo 포인트 저장 → Ctrl+Z 한 번으로 편집 모드 복귀 가능
- 결과 화면 우측 상단 "Edit" 복귀 버튼 (오렌지 강조)
- IFC 위자드 이전 단계 이동 시 결과 정리 확인 다이얼로그
- 결과 → 편집 전환 시 메쉬/필터/Deformed/Solid 상태 완전 초기화 (잔재 방지)

**IFC 누락 최상층 자동 감지** (2026-05-04):
- IfcBuildingStorey에 선언되지 않은 최상층 슬래브/지붕 노드를 구조 노드 Z좌표에서 추론
- 0.3m 허용오차로 기존 층과 거리 검증 후 자동 추가, INFO 로그 기록

**향후 개선 예정**:
- 슬래브 하중 분배 개선 (tributary → 2-way)
- RC 단면 (직사각형, 원형) 확장
- IFC 파싱 추가 검증 (CHS/L/채널 Revit IFC)
- Design Check 비H형강 확장 (AISC HSS/Round HSS)

### 3.3 KDS-Based Auto Load Generation
- **고정하중(DL)**: 슬래브 자중 + 마감 + 설비
- **활하중(LL)**: 용도별 KDS 41 12 00 매핑 (712건 DB)
- **지진하중(EQ)**: 등가정적 + 응답스펙트럼해석(RSA)
- **풍하중(Wind)**: qz x Gf x Cp 프로파일
- **하중조합**: KDS 41 17 00 자동 18개 조합 생성

### 3.4 Geometric Nonlinearity
- **2D**: P-Delta (geomTransf 'PDelta') + Newton solver (10단계, fallback)
- **3D**: Corotational (geomTransf 'Corotational')
- Midas Gen과 비교 검증 완료 (drift < 0.25%에서 ~0.03% 차이)

### 3.5 Design Check (KDS + AISC)
- **층간변위**: KDS 41 17 00 (inelastic drift = Cd x delta / IE)
- **부재강도**: AISC 360 (압축 E3, 휨 F2, 전단 G2, 상관 H1)
- OK/NG 판정 + 최대 interaction ratio

### 3.6 3D Building Editor (Web UI)
- Three.js 기반 인터랙티브 3D 뷰어
- 부재 클릭 선택 + 속성 패널
- 설계검토 결과 색상 오버레이 (DC Colors)
- 라이트/다크 테마 전환
- Manual / NL / IFC 3개 입력 탭

### 3.7 Midas Gen Benchmark Verification
- 5개 벤치마크 케이스, 112개 메트릭
- 100 OK, 12 CHECK (3D 요소 정식화 차이 ~3%)
- 선형/비선형 모두 우수한 일치 확인

---

## 3.X Recommendation Pipeline Foundation

> 향후 **KDS-RAG + LLM 기반 "구조 부재 수정 추천 시스템"** 의 토대.
> 이번 단계에서는 **결정론(deterministic) 추출 계층만** 구현되어 있고,
> RAG / 벡터DB / LLM 호출 / 실제 최적화 / 프론트엔드 대규모 변경은 **아직 구현되지 않았습니다.**

### 3.X.1 개요

"해석 결과 → 설계 이슈 → 수정 후보 → KDS 근거" 흐름의 데이터 계약(data contract) 만을 먼저 안정화했습니다.
LLM은 추후 **설명/근거/선택지 정리** 에만 붙고, **계산/판정은 deterministic layer가 담당**합니다.

```
analysis result + design_check + warnings
   ↓
extract_issues()         (deterministic — 이번 단계)
   ↓
StructuralIssue[]
   ↓
generate_candidates()    (placeholder — 이번 단계)
   ↓
RetrofitCandidate[]   ← 향후 KDS-RAG 가 CodeReference 채움
```

### 3.X.2 응답 추가 필드 (additive)

`/api/v2/analyze`, `/api/v2/parse-ifc`, `/api/building/analyze` 응답에 다음 필드가 **추가** 되었습니다.
기존 필드는 그대로 유지되어 V2 Editor 와 V1 fallback 모두 호환됩니다.

| 필드 | 타입 | 설명 |
|------|------|------|
| `warnings` | `list[AnalysisWarning]` | 구조화된 경고 객체 (code/severity/stage/recoverable/detail) |
| `warning_messages` | `list[str]` | 기존 프론트 호환용 문자열 미러 (`"code: msg"`) |
| `issues` | `list[StructuralIssue]` | deterministic 이슈 목록 (strength_exceeded / drift_exceeded / missing_design_check / analysis_warning / shear_exceeded) |
| `recommendation_candidates` | `list[RetrofitCandidate]` | 수정 후보 placeholder. `requires_reanalysis: true` 가 데이터 모델에 인코딩됨 |
| `recommendation_summary` | `dict` | 카운트 + `rag_enabled: false` / `llm_enabled: false` 명시 |

### 3.X.3 핵심 데이터 모델

`mcp-server/core/recommendation/schemas.py` 의 dataclass:

| 모델 | 역할 |
|------|------|
| `AnalysisEnvelope` | 케이스 전체 envelope (변위/부재력/층간변위 최대값) |
| `AnalysisCaseSummary` | 케이스/조합별 요약 |
| `MemberForceSummary` | 부재별 envelope 부재력 (Pu/Mux/Muy/Vu + governing_combo) |
| `MemberDesignCheck` | 부재별 설계검토 (interaction/shear ratio, status, code_refs, raw 원본) |
| `AnalysisWarning` | 구조화된 경고 — `list[str]` 대체 |
| `StructuralIssue` | 이슈 단위. issue_type / severity / evidence / code_refs |
| `RetrofitCandidate` | 수정 후보 placeholder. `requires_reanalysis: true` 기본값 |
| `CodeReference` | KDS-RAG 가 채울 자리 (현재 standard_id/clause_id 만 채움, quote/relevance_reason 은 `None` placeholder) |

### 3.X.4 결정론(deterministic) 규칙

`generate_candidates()` 의 placeholder 매핑:

| 이슈 유형 | action_type |
|-----------|-------------|
| `strength_exceeded`, `shear_exceeded` (interaction > 1.0) | `increase_section` |
| `drift_exceeded` | `add_lateral_resistance` |
| `missing_design_check` | `requires_engineer_review` |
| `analysis_warning` (severity=error) | `requires_engineer_review` |
| 정보 부족 / 미분류 | `requires_engineer_review` |

후보는 모두 `requires_reanalysis: true`, `confidence: low|medium` 이며,
`code_refs` 는 현재 빈 배열입니다. (향후 KDS-RAG 가 채움)

### 3.X.5 명시적 비범위 (NOT in scope)

- ❌ RAG 검색 / 벡터DB
- ❌ LLM 호출 / 프롬프트
- ❌ 실제 단면 최적화 / 자동 설계 확정
- ❌ 프론트엔드 UI 대규모 변경
- ❌ Midas Gen / NX 비교 수치 검증 (별도 축)
- ❌ Render 배포 최적화 (임시 인프라)

### 3.X.6 구현 위치

```
mcp-server/core/recommendation/
├── __init__.py
├── schemas.py              # dataclass 정의
├── issue_extractor.py      # extract_issues()
├── candidate_generator.py  # generate_candidates()
└── pipeline.py             # build_recommendation_payload() — API 통합용
```

테스트: `tests/test_recommendation.py` (26 tests).

---

## 4. Project Structure

```
opensees-MCP/
├── mcp-server/                        # MCP 구조해석 서버
│   ├── server.py                      # MCP tool 정의 (14개 도구)
│   └── core/                          # 해석 엔진 모듈
│       ├── simple_beam.py             # 단순보 해석
│       ├── continuous_beam.py         # 연속보 해석
│       ├── frame_2d.py                # 2D 프레임 (릴리즈 + P-Delta)
│       ├── frame_3d.py                # 3D 프레임 (6-DOF + Corotational)
│       ├── building_model.py          # BuildingModel IR
│       ├── load_generator.py          # KDS 자동 하중 생성
│       ├── ifc_parser.py              # V1 IFC → BuildingModel 파싱
│       ├── ifc_parser_v2.py           # V2 IFC → Node-Element (StructuralModel)
│       ├── structural_model.py        # V2 StructuralModel IR (노드-요소 그래프)
│       ├── visualization_v2.py        # V2 Plotly 3D 뷰어 (독립 HTML)
│       ├── nl_resolver.py             # 자연어 → Config 변환
│       ├── design_check.py            # KDS 층간변위 + AISC 부재강도
│       ├── design_spectrum.py         # KDS 17 10 00 Sa(T) 곡선
│       ├── response_spectrum_analysis.py  # RSA (CQC/SRSS)
│       ├── visualization.py           # 2D HTML 리포트
│       ├── visualization_3d.py        # 3D HTML 리포트
│       ├── result_interpreter.py      # 심각도/진단/제안 해석
│       ├── recommendation/             # 추천 시스템 토대 (deterministic only)
│       │   ├── schemas.py              # AnalysisWarning / StructuralIssue / RetrofitCandidate / CodeReference …
│       │   ├── issue_extractor.py      # 해석/설계검토 → StructuralIssue[]
│       │   ├── candidate_generator.py  # placeholder 후보 생성
│       │   └── pipeline.py             # build_recommendation_payload() (API 통합)
│       └── ...                        # 기타 모듈
│
├── webapp/                            # 웹 애플리케이션 (V2 only)
│   └── backend/
│       ├── app/
│       │   ├── main_simple.py         # FastAPI 앱 (V2 라우트, Excel/DXF Export)
│       │   ├── core/
│       │   │   ├── claude_service.py   # Claude API → BuildingIntent
│       │   │   ├── auth.py            # 데모 인증 미들웨어
│       │   │   └── config.py
│       │   └── models/schemas.py      # JobStatus enum
│       ├── templates/
│       │   └── editor_v2.html         # V2 3D Editor (단일 SPA)
│       └── static/
│           ├── js/editor3d_v2.js      # V2 3D 뷰어 (IFC + 결과 + DXF/Excel)
│           ├── js/v2_edit.js          # V2 편집 도구 (Node/Element/Delete/Move)
│           ├── js/v2_tools.js         # V2 Solid Section + Release/Support
│           └── css/editor_v2.css      # V2 테마 + 편집 UI
│
├── tests/                             # 테스트 스위트
│   ├── benchmark/                     # Midas Gen 비교 (5 cases)
│   └── ...                            # 단위 테스트
│
├── data/                              # 데이터
│   ├── mapping/occupancy.json         # 30개 용도 매핑
│   └── kds_output/                    # KDS 추출/정규화 데이터
│
├── scripts/                           # 유틸리티 스크립트
│
├── Dockerfile                         # Docker 이미지 (Python 3.12)
├── docker-compose.yml                 # 로컬 Docker 실행
└── DEPLOY.md                          # Azure Container Apps 배포 가이드
```

---

## 5. Technical Stack

| 구분 | 기술 | 버전/비고 |
|------|------|-----------|
| Language | Python | 3.12+ |
| Analysis Engine | OpenSeesPy | opensees 0.1.x (3.12+) |
| Web Framework | FastAPI | uvicorn ASGI |
| AI/LLM | Claude API (Anthropic) | claude-sonnet |
| Frontend | Three.js + Vanilla JS | 3D 인터랙티브 뷰어 |
| Database | Supabase (PostgreSQL) | 712건 load_params + 2290건 hazard |
| Design Code | KDS 41 12 00, 41 17 00, 17 10 00 | 한국건설기준 |
| Design Standard | AISC 360 | 강구조 부재강도 |
| IFC Parser | ifcopenshell | 0.8.4+ |
| DXF Export | ezdxf | 1.0.0+ (층별 평면도/입면도) |
| Excel Export | pandas + openpyxl | 2.0+ / 3.1+ |
| Protocol | MCP (Model Context Protocol) | 14개 도구 등록 |
| Deployment | Render.com (Singapore) | Python 3.12.7 / 512MB free plan |

---

## 6. Installation & Run

> **운영 브랜치:** `v2/node-element` (현재 Render에 배포되는 브랜치).
> 아래 명령은 모두 이 브랜치 기준입니다.

### Local Development

```bash
# Python 3.12.7 권장 (CI/Render와 동일)
conda create -n opensees python=3.12.7
conda activate opensees

# 의존성 설치 (Render와 동일한 파일)
pip install -r webapp/backend/requirements.txt

# 환경변수 (.env, 프로젝트 루트)
ANTHROPIC_API_KEY=your-api-key   # 선택 — 없으면 자연어 입력만 비활성화
SUPABASE_URL=your-supabase-url   # 선택 — 없으면 KDS DB fallback (로컬 매핑)
SUPABASE_KEY=your-supabase-key   # 선택

# PYTHONPATH 설정 — webapp/backend 와 mcp-server 둘 다 필요
# (Render는 render.yaml 에서 동일하게 설정합니다)
# Windows PowerShell:
$env:PYTHONPATH = "$PWD;$PWD\mcp-server;$PWD\webapp\backend"
# macOS/Linux:
export PYTHONPATH="$PWD:$PWD/mcp-server:$PWD/webapp/backend"

# 실행
cd webapp/backend
python -m uvicorn app.main_simple:app --host 0.0.0.0 --port 8001
```

접속: http://localhost:8001 → V2 3D Editor 바로 표시 (V1 라우트는 모두 제거됨)

### Running Tests

```bash
# CI와 동일 명령 (benchmark 제외, script-style 테스트는 conftest 에서 제외됨)
pytest tests/ --ignore=tests/benchmark -q
```

`pytest tests/test_ifc_parser_v2.py` 는 ifcopenshell이 설치돼 있으면
`tests/fixtures/minimal_v2_building.ifc` 픽스처를 자동 생성/재사용하여
검증합니다. ifcopenshell이 없는 환경에서도 순수 함수 단위 테스트
(`TestPureHelpers`, `TestModelJsonRoundtrip`)는 항상 실행됩니다.

### Docker

```bash
docker compose up --build
# 접속: http://localhost:8000
```

### Render 배포 (현재 운영)

V2 전용 — `render.yaml` blueprint로 자동 배포 (Python 3.12.7, Singapore region, free plan).

```bash
# v2/node-element 브랜치에 push 하면 Render가 자동 빌드/배포
git push origin v2/node-element
```

**Render 대시보드에서 설정해야 하는 환경변수:**

| Key | 필수 | 설명 |
|-----|------|------|
| `PYTHONPATH` | 자동 | `render.yaml` 이 `/opt/render/project/src/webapp/backend:/opt/render/project/src/mcp-server` 로 설정 |
| `MCP_SERVER_PATH` | 자동 | `render.yaml` 이 `/opt/render/project/src/mcp-server` 로 설정 |
| `ANTHROPIC_API_KEY` | 선택 | 자연어 → BuildingIntent 변환에만 사용. 미설정 시 자연어 탭만 비활성화 |
| `SUPABASE_URL` | 선택 | KDS 하중 DB 조회. 미설정 시 `data/mapping/occupancy.json` 로컬 fallback |
| `SUPABASE_KEY` | 선택 | 위와 동일 |

### Azure Container Apps (대안, 더 큰 메모리 필요시)

[DEPLOY.md](DEPLOY.md) 참조

---

## 6.1 Known Limitations

### CI 검증 범위
GitHub Actions (`.github/workflows/ci.yml`) 는 push/PR 마다 다음을 검증합니다:

- `webapp/backend/requirements.txt` 가 Python 3.12.7 환경에서 설치 가능한지
- 각 패키지가 실제 import 이름으로 로딩되는지 (예: `python-multipart` → `multipart`)
- `core.ops_compat` 가 Linux에서 정상 동작하는지 (opensees 0.1.x 호환)
- `pytest tests/` (benchmark 제외) 전체 통과
- `app.main_simple:app` 임포트 + `uvicorn` 부팅 + `/health` 응답

CI가 **검증하지 않는 부분**:
- 실제 구조 해석 결과의 수치적 정확성 (Midas 비교는 수동, `tests/benchmark/` 참조)
- 프론트엔드 동작 (Three.js, 편집 UI)
- IFC 파서가 실제 Revit 출력물에서 작동하는지 (자체 생성한 미니멀 IFC만 검증)

### IFC 파서 테스트 범위
- `tests/fixtures/minimal_ifc.py` 가 4 기둥 + 2 보의 1층 IFC4 파일을
  ifcopenshell.api 로 생성합니다.
- 픽스처 IFC는 ifcopenshell.api 의 유닛 변환 버그로 노드 좌표가
  부분적으로 mm 단위로 남아있어, **disconnect / snap 통합 테스트는 픽스처에서 검증하지 않습니다**.
  대신 `TestPureHelpers` 가 m 단위 좌표로 같은 로직 경로를 직접 테스트합니다.
- 실제 Revit IFC 호환성은 별도 수동 검증 필요.

### Render free plan 한계
- 메모리 512MB — `numpy + scipy + matplotlib + ifcopenshell + opensees + pandas`
  가 모두 cold start에 로드되어 첫 요청에 ~25-30s 소요 가능.
- 디스크는 컨테이너에 종속되며 sleep / redeploy 시 초기화됨.
- 무료 플랜은 동시 요청 수가 제한되므로 큰 모델 해석 중에는 다른 요청이 대기열에 들어갈 수 있음.

### Jobs 저장은 영속적이지 않음
- 분석 결과 (`jobs_db` 메모리 + `webapp/backend/jobs/<job_id>/*.json`,
  `report.html`) 는 모두 워커의 로컬 디스크에만 저장됩니다.
- 서버 재시작 / Render sleep / 재배포 시 모든 job이 사라집니다.
- API 응답 (`/api/jobs/{job_id}/report`, `/api/building/{job_id}`,
  `/api/export/excel/{job_id}`) 가 404 를 반환하면 **해석을 다시 실행**해야 합니다.
  404 응답 메시지에 동일한 안내가 포함되어 있습니다.
- 외부 DB / S3 backing 은 후속 작업.

### Partial Failure Surface (`warnings` field)
`/api/v2/analyze`, `/api/v2/parse-ifc`, `/api/building/analyze` 응답에는
`warnings: list[str]` 필드가 포함됩니다. Design check, RSA, HTML 리포트,
IFC viewer 생성 등 보조 단계가 실패하더라도 핵심 응답은 success 로
유지되며, 실패 원인이 이 배열에 기록됩니다.

---

## 7. Benchmark Results (vs Midas Gen)

| Case | 모델 | 메트릭 | 결과 |
|------|------|--------|------|
| 1 | 1-bay 2D Linear | 20 | ALL OK |
| 2 | 2-bay 3-story 2D | 24 | ALL OK |
| 3 | 2D P-Delta | 16 | ALL OK |
| 4 | 3D Linear | 12/12 OK, 12 CHECK | 요소 정식화 차이 ~3% |
| 5 | 5-story 3D P-Delta | 40 | ALL OK (max diff 0.05%) |

**Total: 112 metrics — 100 OK, 12 CHECK**

---

## 8. Sign Convention

| 구분 | 규약 | 설명 |
|------|------|------|
| 전단력 V | V > 0 | 좌측 절단면에서 상향 |
| 모멘트 M | M > 0 | Sagging (하부 인장) |
| 축력 N | N > 0 | 인장 (+), 압축 (-) |

---

## 9. License

MIT License

---

## 10. References

- [OpenSeesPy Documentation](https://openseespydoc.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [KDS 한국건설기준](https://www.kcsc.re.kr/)
- [AISC 360 — Specification for Structural Steel Buildings](https://www.aisc.org/)
- [Three.js Documentation](https://threejs.org/docs/)
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)
