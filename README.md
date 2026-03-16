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

## 2. Supported Analysis Types

| 해석 유형 | 상태 | DOF | 핵심 기능 |
|-----------|------|-----|-----------|
| Simple Beam | Ready | 3 | 단순지지/캔틸레버/고정단, 분포/집중/조합하중 |
| Continuous Beam | Ready | 3 | 다경간, 내부 힌지, SFD 불연속 처리 |
| 2D Frame | Ready | 3 | 다층/다경간, 릴리즈, P-Delta, Envelope |
| 3D Frame | Ready | 6 | 양방향 경간, Corotational 비선형, Rigid Diaphragm |
| Building Pipeline | Ready | 6 | IFC/JSON → 자동하중 → 3D해석 → 설계검토 |

---

## 3. Key Features

### 3.1 Natural Language Input (NL Resolver)
- 한국어 자연어 → Claude API → BuildingIntent 추출 → Config 변환
- 30개 용도 매핑 (한국어 별칭 → canonical key)
- 복합용도 자동 분리 ("근생" → retail)
- 지역/중요도/층고/경간 자동 추론

### 3.2 IFC Integration
- IFC 파일 업로드 → 벽/기둥 기반 자동 파싱
- 3D 와이어프레임 미리보기 (Three.js)
- 3단계 위자드: Upload → Geometry Preview → Config → Analyze

### 3.3 KDS-Based Auto Load Generation
- **고정하중(DL)**: 슬래브 자중 + 마감 + 설비
- **활하중(LL)**: 용도별 KDS 41 12 00 매핑 (712건 DB)
- **지진하중(EQ)**: 등가정적, KDS 17 10 00 설계응답스펙트럼
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

## 4. Project Structure

```
opensees-MCP/
├── mcp-server/                        # MCP 구조해석 서버
│   ├── server.py                      # MCP tool 정의 (14개 도구)
│   └── core/                          # 해석 엔진 모듈 (20개)
│       ├── simple_beam.py             # 단순보 해석
│       ├── continuous_beam.py         # 연속보 해석
│       ├── frame_2d.py                # 2D 프레임 (릴리즈 + P-Delta)
│       ├── frame_3d.py                # 3D 프레임 (6-DOF + Corotational)
│       ├── building_model.py          # BuildingModel IR
│       ├── load_generator.py          # KDS 자동 하중 생성
│       ├── ifc_parser.py              # IFC → BuildingModel 파싱
│       ├── nl_resolver.py             # 자연어 → Config 변환
│       ├── design_check.py            # KDS 층간변위 + AISC 부재강도
│       ├── design_spectrum.py         # KDS 17 10 00 Sa(T) 곡선
│       ├── visualization.py           # 2D HTML 리포트 (~3800줄)
│       ├── visualization_3d.py        # 3D HTML 리포트 (~1400줄)
│       ├── result_interpreter.py      # 심각도/진단/제안 해석
│       ├── assumption_tracker.py      # 가정 확인 모듈
│       ├── section_3d.py              # 3D 단면 물성 (A, Ix, Iy, J)
│       ├── sign_convention.py         # 부호규약 변환
│       ├── ops_compat.py              # OpenSeesPy 호환 (3.8/3.12+)
│       ├── kds_loads.py               # KDS 하중 파라미터
│       └── verification.py            # 수치 검증
│
├── webapp/                            # 웹 애플리케이션
│   └── backend/
│       ├── app/
│       │   ├── main_simple.py         # FastAPI 앱 (Building API 포함)
│       │   └── core/
│       │       ├── claude_service.py   # Claude API 서비스
│       │       └── config.py
│       ├── templates/
│       │   ├── editor.html            # 3D Building Editor
│       │   ├── home.html              # 메인 페이지
│       │   └── base.html              # 베이스 템플릿
│       └── static/
│           ├── js/editor3d.js         # Three.js 3D 뷰어 (~1650줄)
│           └── css/editor.css         # 테마 시스템 (~960줄)
│
├── tests/                             # 테스트 스위트
│   ├── test_nl_resolver.py            # NL 변환 (38 tests)
│   ├── test_design_check.py           # 설계검토 (16 tests)
│   ├── test_stage5_metadata.py        # 메타데이터 (29 tests)
│   ├── test_stage6_building_nonlinear.py  # 건물 비선형 (33 tests)
│   ├── test_result_interpreter.py     # 결과 해석
│   └── benchmark/                     # Midas Gen 비교 (5 cases, 112 metrics)
│       ├── cases.py
│       ├── compare.py
│       └── run_benchmarks.py
│
├── data/                              # 데이터
│   ├── mapping/occupancy.json         # 30개 용도 매핑
│   └── kds_output/                    # KDS 추출/정규화 데이터
│       ├── 03_*_normalized.json       # 정규화 데이터셋 (16종)
│       └── midas_input_reference.json # Midas 비교 참조값
│
├── scripts/                           # 유틸리티 스크립트
│   ├── supabase_loader.py             # Supabase DB 적재
│   ├── load_hazard_regions.py         # 지역 위험도 적재
│   ├── load_seismic_pga.py            # 지진 PGA 적재
│   ├── pdf_to_images.py               # PDF → 이미지 변환
│   └── normalize_*.py                 # KDS 데이터 정규화
│
├── docs/                              # 프로젝트 문서
├── adapters/                          # 어댑터 모듈
├── agents/                            # AI 에이전트
├── pipeline/                          # 데이터 파이프라인
├── streamlit_app/                     # Streamlit UI 프로토타입
└── opensees-ai-agent/                 # VIKTOR 기반 에이전트
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
| Protocol | MCP (Model Context Protocol) | 14개 도구 등록 |

---

## 6. Sign Convention

시각화에는 **교과서/MIDAS 부호규약**이 적용됩니다:

| 구분 | 규약 | 설명 |
|------|------|------|
| 전단력 V | V > 0 | 좌측 절단면에서 상향 |
| 모멘트 M | M > 0 | Sagging (하부 인장) |
| 축력 N | N > 0 | 인장 (+), 압축 (-) |

변환: `V_textbook = -V_opensees`, `M_textbook = -M_opensees`

---

## 7. KDS Database (Supabase)

| 테이블 | 건수 | 내용 |
|--------|------|------|
| load_params | 712 | DL 234, LL 79, Seismic 258, Wind 49, Snow 17, Combo 46 |
| hazard_region_values | 2,290 | 229개 시군구 x 10종 (snow, wind, seismic, PGA) |

### KDS Agent Team Pipeline
```
[PDF] → pdf_to_images.py → [PNG Images]
     → Document Analyst → Table Extractor → Normalizer → Validator → DB Loader
     → Supabase (load_params / hazard_region_values)
```

---

## 8. Benchmark Results (vs Midas Gen)

| Case | 모델 | 메트릭 | 결과 |
|------|------|--------|------|
| 1 | 1-bay 2D Linear | 20 | ALL OK |
| 2 | 2-bay 3-story 2D | 24 | ALL OK |
| 3 | 2D P-Delta | 16 | ALL OK |
| 4 | 3D Linear | 12/12 OK, 12 CHECK | 요소 정식화 차이 ~3% |
| 5 | 5-story 3D P-Delta | 40 | ALL OK (max diff 0.05%) |

**Total: 112 metrics — 100 OK, 12 CHECK** (선형/비선형 모두 우수한 일치)

---

## 9. Installation & Run

### Prerequisites
```bash
# Python 3.12+ 환경
conda create -n opensees python=3.12
conda activate opensees
```

### Install
```bash
# 의존성 설치
cd webapp/backend
pip install -r requirements.txt

# MCP 서버 의존성
cd ../../mcp-server
pip install -r requirements.txt
```

### Environment Variables
```bash
# .env 파일 생성
ANTHROPIC_API_KEY=your-api-key
SUPABASE_URL=your-supabase-url
SUPABASE_KEY=your-supabase-key
```

### Run
```bash
cd webapp/backend
python -m uvicorn app.main_simple:app --host 0.0.0.0 --port 8001
```

접속: http://localhost:8001

---

## 10. API Endpoints

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/` | 메인 페이지 |
| GET | `/editor` | 3D Building Editor |
| POST | `/api/building/analyze` | 빌딩 해석 (3D) |
| POST | `/api/building/parse-ifc` | IFC 파싱 |
| POST | `/api/building/sections` | 단면 목록 조회 |
| POST | `/api/building/materials` | 재료 목록 조회 |
| POST | `/api/claude/parse-building` | 자연어 → BuildingIntent |
| POST | `/api/claude/resolve-config` | Intent → Config 변환 |
| GET | `/api/jobs/{job_id}/report` | HTML 리포트 |
| POST | `/api/jobs` | 2D Frame 해석 |
| POST | `/api/simple-beam/jobs` | 단순보 해석 |
| POST | `/api/continuous-beam/jobs` | 연속보 해석 |

---

## 11. Implementation Stages

| Stage | 내용 | 테스트 |
|-------|------|--------|
| 1 | 리포트 품질 (warnings, envelope, equilibrium) | - |
| 2 | 가정 확인 (Assumption Confirmation) | - |
| 3 | 부재 릴리즈 (2D `-release` / 3D `equalDOF`) | 29/29 |
| 4 | P-Delta (2D PDelta / 3D Corotational) | 40/40 |
| 5 | 메타데이터 + 리포트 명확성 개선 | 29/29 |
| 6 | 3D 전역 응답 검증 + 건물 파이프라인 | 33/33 |
| 7 | Midas Gen 벤치마크 (5 cases, 112 metrics) | 112/112 |
| Phase 1 | 자연어 → Config (NL Resolver) | 38/38 |
| Phase 2 | 설계검토 (Design Check) | 16/16 |
| Phase 3 | 3D Building Editor (Web UI) | Manual |

---

## 12. License

MIT License

---

## 13. References

- [OpenSeesPy Documentation](https://openseespydoc.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [KDS 한국건설기준](https://www.kcsc.re.kr/)
- [AISC 360 — Specification for Structural Steel Buildings](https://www.aisc.org/)
- [Three.js Documentation](https://threejs.org/docs/)
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)
