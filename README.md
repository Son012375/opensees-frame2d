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
| Response Spectrum | Ready | 6 | KDS 설계응답스펙트럼, CQC/SRSS, RSA/ELF 비교 |

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
- Wire/Solid Section 토글 (H형강 ExtrudeGeometry + 엣지 윤곽선)
- 노드 라벨: ID only / XYZ 좌표 (고해상도 Sprite)
- 마우스 보조선 (X/Y/Z 점선 가이드) + Ghost Node 미리보기
- 실시간 마우스 좌표 표시 (우측 하단)
- 편집 모드에서 우클릭 회전 + 스크롤 줌 + Shift 패닝

**해석 결과 시각화** (2026-04-15 추가):
- SFD/BMD/Axial 3D 다이어그램 (폴리곤 면 + 외곽선 + i/j-end + 고점 라벨)
- Display Filter 패널 (Loads DL/LL/EQ/Wind, Story, Member Type 토글)
- 하중 화살표 (분포하중 보 위 배열 + 수평하중 층 중심)
- 반력 시각화 (지점 화살표 + kN 수치 라벨)
- Deformed shape 토글 + 스케일 슬라이더 (1×~500×, Auto)
- 부재력 hover tooltip (N/V/M 최대값)
- 개별 부재 Canvas 다이어그램 (N/V/M × 3, 클릭 시)
- Properties 탭 정리 (Results / Modal / DC)

**편집 도구** (2026-04-15 추가):
- 다중 선택 (Ctrl+클릭, 박스 드래그, Story 선택)
- 복사/미러/층 복사 (플로팅 드래그 패널)
- 개별 부재 단면 변경 ("이 부재만" + 재해석)
- 보-보 교차점 자동 분할 (2D line intersection)
- 프로젝트 저장/불러오기 (.v2proj, 해석 결과 포함, 재해석 불필요)

**향후 개선 예정**:
- BMD 스케일 조절 슬라이더
- Export (CSV/Excel — 부재력, 변위, 반력)
- IFC 단면 매핑 확장 (Box/Pipe/C채널)
- 슬래브 하중 분배 개선 (tributary → 2-way)
- RC 단면 (직사각형, 원형) 확장

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
│       └── ...                        # 기타 모듈
│
├── webapp/                            # 웹 애플리케이션
│   └── backend/
│       ├── app/
│       │   ├── main_simple.py         # FastAPI 앱
│       │   └── core/
│       │       ├── claude_service.py   # Claude API 서비스
│       │       ├── auth.py            # 데모 인증 미들웨어
│       │       └── config.py
│       ├── templates/                 # HTML 템플릿
│       │   ├── editor.html            # V1 3D Building Editor
│       │   ├── editor_v2.html         # V2 3D Editor (Node-Element + Edit)
│       │   └── ...
│       └── static/                    # JS/CSS
│           ├── js/editor3d.js         # V1 Three.js 3D 뷰어
│           ├── js/editor3d_v2.js      # V2 3D 뷰어 (V2 IFC + 편집)
│           ├── js/v2_edit.js          # V2 편집 도구 (Node/Element/Delete/Move)
│           ├── css/editor.css         # V1 테마 시스템
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
| Protocol | MCP (Model Context Protocol) | 14개 도구 등록 |
| Deployment | Azure Container Apps | Docker + ACR |

---

## 6. Installation & Run

### Local Development

```bash
# Python 3.12+ 환경
conda create -n opensees python=3.12
conda activate opensees

# 의존성 설치
cd webapp/backend
pip install -r requirements.txt

# 환경변수 (.env)
ANTHROPIC_API_KEY=your-api-key
SUPABASE_URL=your-supabase-url
SUPABASE_KEY=your-supabase-key

# 실행
python -m uvicorn app.main_simple:app --host 0.0.0.0 --port 8001
```

### Docker

```bash
docker compose up --build
# 접속: http://localhost:8000
```

### Azure Deployment

[DEPLOY.md](DEPLOY.md) 참조

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
