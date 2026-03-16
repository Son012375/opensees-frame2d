# OpenSees-MCP Claude 가이드

> **프로젝트 상세 문서:** `.claude/PROJECT_CONTEXT.md`

## Quick Start

새 세션에서 컨텍스트 복원:
```
/read .claude/PROJECT_CONTEXT.md
```

## 현재 상태 요약

| 해석 유형 | 상태 |
|-----------|------|
| Simple Beam | ✅ Ready |
| Continuous Beam | ✅ Ready |
| Frame 2D | ✅ Ready (릴리즈 + P-Delta 지원) |
| Frame 3D | ✅ Ready (릴리즈 + Corotational 기하비선형 지원) |
| Building Pipeline | ✅ Ready (IFC/JSON → 자동하중 → 3D해석) |
| NL Resolver | ✅ Ready (자연어 → Config 변환, 38 tests) |
| Design Check | ✅ Ready (KDS 층간변위 + AISC 부재강도, 16 tests) |

## 구현 스테이지

| Stage | 내용 | 상태 |
|-------|------|------|
| 1 | 리포트 품질 강화 (warnings, envelope, equilibrium) | ✅ |
| 2 | 가정 확인 (Assumption Confirmation) | ✅ |
| 3 | 부재 릴리즈 (2D `-release` / 3D `equalDOF`) | ✅ |
| 4 | P-Delta (2D PDelta / 3D Corotational) | ✅ |
| 5 | 메타데이터 + 리포트 명확성 개선 | ✅ |
| 6 | 3D 전역 응답 검증 + 건물 파이프라인 통합 | ✅ |
| 7 | Midas Gen 벤치마크 준비 (5 cases, 105 metrics) | ✅ |
| Phase 1 | 자연어 → 구조해석 Config 변환 (NL Resolver) | ✅ |

## 핵심 파일

| 파일 | 설명 |
|------|------|
| `mcp-server/core/frame_2d.py` | 2D 프레임 해석 엔진 |
| `mcp-server/core/frame_3d.py` | 3D 프레임 해석 엔진 |
| `mcp-server/core/building_model.py` | BuildingModel IR (IFC/JSON → 해석) |
| `mcp-server/core/load_generator.py` | KDS 자동 하중 생성 |
| `mcp-server/core/ifc_parser.py` | IFC 파싱 (벽/기둥 기반) |
| `mcp-server/core/design_spectrum.py` | KDS 설계응답스펙트럼 |
| `mcp-server/core/visualization.py` | HTML 리포트 생성 (2D, ~3800줄) |
| `mcp-server/core/visualization_3d.py` | HTML 리포트 생성 (3D, ~1500줄) |
| `mcp-server/core/sign_convention.py` | 부호규약 변환 |
| `mcp-server/core/assumption_tracker.py` | 가정 확인 모듈 |
| `mcp-server/core/nl_resolver.py` | 자연어 → Config 변환 (용도/지역/층/경간) |
| `mcp-server/core/design_check.py` | KDS/AISC 설계검토 (drift + member strength) |
| `tests/benchmark/` | Midas Gen 벤치마크 (5 cases) |
| `webapp/backend/app/main_simple.py` | FastAPI 앱 |

## 부호규약

- **V > 0:** 좌측면 상향 (↑)
- **M > 0:** Sagging (하부 인장)
- **변환:** `V_textbook = -V_opensees`, `M_textbook = -M_opensees`

## 실행

```bash
cd webapp/backend
python -m uvicorn app.main_simple:app --port 8001
```

## 진행 예정

1. 3.2.2 빌딩 프레임 해석 노션 문서화
2. Phase 3 방향 재검토 (NL 결과 해석 or KDS 준거 검토)
3. 고유치해석 결과 통합

---

## KDS Agent Team 워크플로우

> KDS(한국건설기준) PDF에서 하중/계수 파라미터를 추출하여 Supabase DB에 적재하는 Agent Team 파이프라인

### 사전 준비

```bash
# PDF를 이미지로 변환 (Vision 추출용)
python scripts/pdf_to_images.py --pdf "PDF경로" --output data/kds_images/KDS_41_12_00
```

### 팀 구성

| 역할 | 담당 | 입력 | 출력 |
|------|------|------|------|
| **Lead** | 전체 조율, 태스크 분배, 결과 종합 | 사용자 지시 | 최종 보고서 |
| **Document Analyst** | 문서 메타데이터 추출 | PDF 이미지 (목차 등) | `data/kds_output/01_document_meta.json` |
| **Table Extractor** | 표 데이터 Vision 추출 | PDF 이미지 + 메타 | `data/kds_output/02_tables_extracted/*.json` |
| **Normalizer** | 키 매핑, 단위 변환 | 추출된 표 JSON | `data/kds_output/03_normalized.json` |
| **Validator** | 범위 검증, 중복 탐지 | 정규화 데이터 | `data/kds_output/04_validation_report.json` |
| **DB Loader** | Supabase 적재 | 검증 통과 데이터 | `data/kds_output/05_load_result.json` |

### 파일 기반 데이터 교환

- 모든 Agent 출력은 `data/kds_output/` 디렉터리에 JSON으로 저장
- 파일명 접두사(`01_`, `02_`, ...)가 파이프라인 순서
- PDF 이미지는 `data/kds_images/{KDS_CODE}/page_NNN.png` 형식

### 용도 키 매핑 (occupancy_key)

전체 매핑은 `data/mapping/occupancy.json` 참조. 주요 매핑:

| 한국어 | key | 비고 |
|--------|-----|------|
| 사무실/업무시설 | `office` | |
| 주거/아파트/공동주택 | `residential` | |
| 소매점/판매시설 | `retail` | |
| 집회시설 | `assembly` | 고정/이동좌석 구분 |
| 창고 | `storage` | 경량/중량 구분 |
| 주차장 | `parking` | 승용차/트럭 구분 |
| 병원 | `hospital` | 병실/수술실 구분 |
| 학교/교육시설 | `school` | |
| 도서관 | `library` | 열람실/서고 구분 |
| 복도/계단 | `corridor` | |

### 단위 규칙

- **기본 단위:** kN/m² (모든 하중값)
- kgf/m² → kN/m²: ×0.00981
- kPa = kN/m²: 1:1
- tf/m² → kN/m²: ×9.81

### 출처 메타데이터 (모든 레코드 필수)

```json
{
  "source": {
    "code_id": "KDS 41 12 00",
    "code_version": "2022-10-11",
    "clause_id": "3.1",
    "table_id": "표 3.1-1"
  }
}
```

### 검증 규칙

| param_type | 범위 | 단위 |
|------------|------|------|
| live_load | 0.5 ~ 25.0 | kN/m² |
| dead_load | 0.1 ~ 50.0 | kN/m² |
| wind_speed | 20.0 ~ 50.0 | m/s |
| snow_load | 0.3 ~ 5.0 | kN/m² |

- confidence < 0.7 → `needs_review: true`
- 필수 필드: param_type, primary_key, value, unit, source.code_id

### Supabase 적재

```bash
# Dry-run (미리보기)
python scripts/supabase_loader.py --mode dry_run --input data/kds_output/03_normalized.json

# 실제 적재
python scripts/supabase_loader.py --mode upsert --input data/kds_output/03_normalized.json
```

### 고정하중 (Dead Load) 출처 구조

> **조사 결과 (2026-02-25):** 건축 고정하중 단위중량 데이터의 출처가 3단계로 분산되어 있음

```
KDS 41 12 00 §2 (기준 본문)
  → "재료의 밀도, 단위체적중량, 조합중량으로 산정" (원칙만, 표 없음)
    → 건축구조기준 및 해설 2024 (대한건축학회)
      → 해설에서 참조 지시: "해표 2.1~2.13을 참조할 수 있다"
        → 건축물의 하중기준 및 해설 (대한건축학회, 2000) ← 실제 표 원본
```

#### 데이터 소재

| 자료 | 내용 | 상태 |
|------|------|------|
| **건축물의 하중기준 및 해설** (대한건축학회, 2000) | 해표 2.1~2.13: 재료 단위체적중량 + 마감재 조합중량 (완전한 표) | 🔜 입수 예정 |
| **KDS 24 12 21** (교량 설계하중) 표 4.2-1 | 기본 재료 단위체적중량 (콘크리트, 강재 등) | ✅ Phase 1 적재 완료 (14건 적재, 2건 review 대기) |

#### 적재 전략

- **Phase 1 (2026-02-25 완료):** KDS 24 12 21 기본 재료 단위중량 14건 → `param_type: "dead_load"`, `param_subtype: "unit_weight"`, `code_id: "KDS 24 12 21"`
  - 철근콘크리트: 24.0 (건축 기준 조정, 교량 원본 24.5)
  - 경량콘크리트 1종/2종: review 대기 (건축 기준 확인 필요)
  - 교량 특화 4종(연철, 아스팔트 포장재, 도상, 석탄) 제외
  - 정규화 파일: `data/kds_output/03_dead_load_normalized.json`
- **Phase 2 (예정):** 건축물의 하중기준 및 해설 입수 후 → 마감재 조합중량 추가 → `code_id: "KBC-DL-2000"` (또는 적절한 출처 코드)
- 기본 재료 단위중량은 물리 상수이므로 교량/건축 기준 간 값 동일 → 학술적 문제 없음
