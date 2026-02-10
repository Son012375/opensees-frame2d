# OpenSees-MCP 프로젝트 통합 문서

> **최종 업데이트:** 2026-02-10
> **목적:** 새 세션에서 컨텍스트 유지를 위한 통합 레퍼런스

---

## 1. 프로젝트 개요

### 1.1 시스템 목표

**OpenSees-MCP**는 구조공학용 해석 플랫폼으로:
- 사용자가 **한국어 자연어**로 구조물을 설명
- **Claude AI**가 파싱하여 구조해석 입력 생성
- **OpenSeesPy**로 해석 수행
- **인터랙티브 HTML 리포트**로 결과 시각화

### 1.2 지원 해석 유형

| 해석 | 상태 | 설명 |
|------|------|------|
| **Simple Beam** | ✅ Ready | 단순지지/캔틸레버/고정단, 분포/집중하중 |
| **Continuous Beam** | ✅ Ready | 다경간, 내부힌지, SFD 불연속 처리 |
| **Frame 2D** | ✅ Ready | 다층/다경간, 하중조합, Envelope, 층간변위 |
| **Frame 3D** | 🚧 Planned | 3차원 해석, 비틀림 (장기 목표) |

### 1.3 기술 스택

| 구분 | 기술 |
|------|------|
| Backend | FastAPI, Python 3.8 |
| Analysis | OpenSeesPy (elasticBeamColumn) |
| Frontend | Jinja2, HTMX, Plotly.js |
| AI | Claude API (Anthropic) |
| Database | Supabase (KS 표준 단면/재료 DB) |
| Deployment | Render |

---

## 2. 프로젝트 구조

```
opensees-MCP/
├── mcp-server/                    # 구조해석 엔진
│   ├── server.py                  # MCP 서버 메인
│   ├── core/
│   │   ├── simple_beam.py         # 단순보 해석
│   │   ├── continuous_beam.py     # 연속보 해석
│   │   ├── frame_2d.py            # 2D 프레임 해석 (~1200줄)
│   │   ├── visualization.py       # HTML 리포트 생성 (~3500줄)
│   │   ├── sign_convention.py     # 부호규약 변환
│   │   └── verification.py        # 수치 검증
│   ├── tools/
│   │   └── opensees_tools.py      # MCP Tool 정의
│   ├── data/
│   │   ├── sections.json          # 단면 DB
│   │   └── materials.json         # 재료 DB
│   └── tests/
│       └── test_sign_convention.py
│
├── webapp/                        # 웹 애플리케이션
│   ├── start_server.bat           # 서버 시작 (API 키 포함)
│   └── backend/
│       ├── requirements.txt
│       ├── app/
│       │   └── main_simple.py     # FastAPI 앱
│       ├── templates/
│       │   ├── home.html          # 메인 페이지
│       │   ├── simple_beam.html
│       │   ├── continuous_beam.html
│       │   └── index.html         # Frame 2D 입력
│       └── static/
│           ├── css/style.css
│           └── js/main.js
│
├── .claude/
│   └── PROJECT_CONTEXT.md         # 이 문서
│
└── README.md                      # GitHub README
```

---

## 3. 부호규약 (Sign Convention)

### 3.1 핵심 규약

**시각화에는 교과서/MIDAS 부호규약 적용:**

| 구분 | 규약 | 설명 |
|------|------|------|
| 전단력 V | V > 0 | 좌측 절단면에서 **상향** (↑) |
| 모멘트 M | M > 0 | **Sagging** (하부 인장, 오목 상향) |
| 축력 N | N > 0 | 인장 (+), 압축 (-) |

### 3.2 변환 규칙

OpenSees → 교과서:
```python
V_textbook = -V_opensees
M_textbook = -M_opensees
```

**적용 위치:** `visualization.py`에서 시각화 시점에만 변환
**저장 규약:** `frame_2d.py`, `simple_beam.py` 결과는 OpenSees 규약 유지

### 3.3 부재 방향 강제

| 부재 | 규칙 | 검증 함수 |
|------|------|-----------|
| Beam | i=left, j=right (x 증가 방향) | `enforce_beam_direction()` |
| Column | i=bottom, j=top (y 증가 방향) | `enforce_column_direction()` |

**위치:** `sign_convention.py`

---

## 4. Frame 2D 상세

### 4.1 입력 스키마

```python
Frame2DInput(
    stories=[3.5, 3.2, 3.2],        # 층고 (m), 아래→위
    bays=[6.0, 6.0],                 # 경간 (m), 좌→우
    column_section_name="H-300x300x10x15",
    beam_section_name="H-400x200x8x13",
    material_name="SS275",
    supports="fixed",                # "fixed" | "pinned"
    num_elements_per_member=4,       # sub-element 개수
    load_cases={
        "DL": [{"type": "floor", "story": 1, "value": 20.0}],
        "EQX": [{"type": "lateral", "story": 1, "fx": 50.0}],
    },
    load_combinations={
        "1.2DL+1.0EQX": {"DL": 1.2, "EQX": 1.0},
    },
)
```

### 4.2 하중 유형

| type | 파라미터 | 설명 |
|------|----------|------|
| `floor` | story, value (kN/m) | 바닥 분포하중 → 보에 적용 |
| `lateral` | story, fx (kN) | 횡하중 → 각 층 좌측 노드에 적용 |

### 4.3 출력 구조 (Frame2DResult)

```python
Frame2DResult(
    nodes=[{id, x, y}, ...],
    elements=[{id, type, ni, nj}, ...],
    member_info=[{id, type, ni, nj, length_m, location, sub_element_ids}, ...],
    case_results={
        "DL": Frame2DCaseResult(
            nodal_displacements=[{node, dx_mm, dy_mm, rz_rad}, ...],
            element_forces=[{element, N_i_kN, V_i_kN, M_i_kNm, ...}, ...],
            reactions=[{node, RX_kN, RY_kN, MZ_kNm}, ...],
            member_forces=[{member_id, s, N_kN, V_kN, M_kNm}, ...],
            story_data={
                story_displacements: [...],
                story_shears: [...],
            },
            story_drifts=[...],
        ),
        ...
    },
    envelope={
        drift: [...],
        memberforces: [...],
        reactions: [...],
    },
)
```

### 4.4 층전단력 계산 (Phase O)

**이중 검증 방식:**

| 방법 | 설명 | 필드 |
|------|------|------|
| Reaction-based | ΣRX (기초 반력 기반) | `shear_rxn_kN` |
| Element-based (signed) | ΣVx (기둥 전단력 합) | `shear_kN_signed` |
| Element-based (abs) | Σ\|Vx\| | `shear_kN_abs` |

**has_lateral 플래그:** 수평하중 없는 케이스(DL)에서는 "Column Cut Force" 라벨로 표시

---

## 5. HTML 리포트 (visualization.py)

### 5.1 탭 구성

| 탭 | 내용 |
|---|------|
| **Deformation** | 변형 형상, Node/Member 라벨 토글, 클릭→부재력 이동 |
| **Member Forces** | N/V/M 3-subplot, 부재 단부력·극값 테이블, Global Diagram |
| **Reactions** | 반력 테이블, 평형 검증 (ΣFx, ΣFy, ΣM) |
| **Story** | 층변위 프로파일, Story Shear (Method 선택), Drift 판정 |
| **Envelope** | 전 케이스 극값 집계, 클릭→해당 케이스·부재 이동 |
| **Model** | 단위, 재료, 단면, Capabilities 매트릭스 |
| **Export** | CSV, PNG, PDF 인쇄 |

### 5.2 Drift Limit 옵션

- 1/200, 1/300, 1/400, Custom
- OK/NG 판정 + 색상 표시 (green/yellow/red)

### 5.3 Export 기능

| 형식 | 내용 |
|------|------|
| CSV | 절점, 반력, 부재력, 층응답, 극값 |
| PNG | 각 Plotly 차트 (`Plotly.toImage`) |
| PDF | Print 버튼 → A4 landscape |

---

## 6. 개발 현황

### 6.1 완료된 기능

- [x] 단순보 해석 (다양한 지지조건, 하중유형)
- [x] 연속보 해석 (다경간, SFD 불연속 처리)
- [x] 2D Frame 해석 (다층/다경간)
- [x] 하중 조합 및 Envelope 분석
- [x] 층간변위 검토 (사용자 정의 허용기준)
- [x] Story Shear 이중검증 (반력/요소 기반)
- [x] 부호규약 통일 (교과서 규약)
- [x] CSV/PNG/PDF Export
- [x] Claude AI 자연어 입력
- [x] Model 탭 Capabilities 매트릭스

### 6.2 진행 예정

- [ ] 3D Frame 해석 (장기)
- [ ] 부재 단부 릴리즈 (힌지)
- [ ] Rigid offset
- [ ] P-Delta 해석
- [ ] 자중 자동 적용
- [ ] 전단변형 (Timoshenko beam)

### 6.3 제한사항 (Model 탭에 표시됨)

| 기능 | 상태 |
|------|------|
| End release (힌지) | Not supported |
| Rigid offset | Not supported |
| Shear deformation (Timoshenko) | Not supported |
| P-Delta (기하비선형) | Not supported |
| Self-weight 자동 계산 | Not supported |

---

## 7. API 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/` | 메인 페이지 |
| GET | `/simple-beam` | 단순보 입력 |
| GET | `/continuous-beam` | 연속보 입력 |
| GET | `/frame2d` | 2D 골조 입력 |
| POST | `/api/jobs` | Frame 2D 해석 Job 생성 |
| POST | `/api/simple-beam/jobs` | 단순보 해석 Job 생성 |
| POST | `/api/continuous-beam/jobs` | 연속보 해석 Job 생성 |
| GET | `/api/jobs/{job_id}/report` | 해석 결과 리포트 |
| POST | `/api/claude/parse` | 자연어 → JSON 변환 |

---

## 8. MCP Tool 설계 철학

### 8.1 High-Level Workflow 방식

기존 MCP 구현체 조사 결과, **FreeCAD MCP**, **Modelica MCP** 등 엔지니어링 시뮬레이션 MCP들이 High-Level 방식 사용.

| 접근 방식 | Tool 수 | LLM 부담 | 적용 |
|-----------|---------|----------|------|
| **High-Level Workflow** | 5-10개 | 낮음 | 시뮬레이션, 해석 ✅ |
| Low-Level Primitives | 20+개 | 높음 | CAD, 그래픽 |

### 8.2 구현된 Tool

| Tool | 설명 |
|------|------|
| `analyze_simple_beam` | 단순보 해석 |
| `analyze_continuous_beam` | 연속보 해석 |
| `analyze_frame_2d` | 2D 프레임 해석 |
| `get_section_properties` | 단면 정보 조회 |
| `get_material_properties` | 재료 정보 조회 |
| `list_available_sections` | 사용 가능한 단면 목록 |
| `list_available_materials` | 사용 가능한 재료 목록 |

---

## 9. 설치 및 실행

### 9.1 환경 설정

```bash
# Python 3.8 필수 (OpenSeesPy 요구사항)
conda create -n opensees38 python=3.8
conda activate opensees38

cd webapp/backend
pip install -r requirements.txt
```

### 9.2 환경 변수

```bash
set ANTHROPIC_API_KEY=your-api-key
set SUPABASE_URL=your-supabase-url
set SUPABASE_KEY=your-supabase-key
```

### 9.3 서버 실행

```bash
cd webapp/backend
python -m uvicorn app.main_simple:app --host 0.0.0.0 --port 8001
```

또는: `webapp/start_server.bat`

---

## 10. OpenSeesPy 핵심 명령어 (Quick Reference)

### 10.1 모델 정의

```python
ops.wipe()
ops.model('basic', '-ndm', 2, '-ndf', 3)
ops.node(nodeTag, x, y)
ops.fix(nodeTag, 1, 1, 1)  # Ux, Uy, Rz 고정
ops.geomTransf('Linear', transfTag)
```

### 10.2 요소 정의

```python
# 탄성 보-기둥 요소
ops.element('elasticBeamColumn', eleTag, ni, nj, A, E, Iz, transfTag)
```

### 10.3 하중

```python
ops.timeSeries('Constant', tsTag)
ops.pattern('Plain', patternTag, tsTag)
ops.load(nodeTag, Fx, Fy, Mz)
ops.eleLoad('-ele', eleTag, '-type', '-beamUniform', wy)
```

### 10.4 해석

```python
ops.constraints('Transformation')
ops.numberer('RCM')
ops.system('BandGeneral')
ops.test('NormDispIncr', 1e-8, 10)
ops.algorithm('Newton')
ops.integrator('LoadControl', 1.0)
ops.analysis('Static')
ops.analyze(1)
```

### 10.5 결과 출력

```python
ops.nodeDisp(nodeTag, dof)      # 변위
ops.nodeReaction(nodeTag, dof)  # 반력
ops.eleForce(eleTag)            # [N_i, V_i, M_i, N_j, V_j, M_j]
```

---

## 11. 변경 이력

| 날짜 | 내용 |
|------|------|
| 2026-02-10 | 통합 문서 생성, README 갱신 |
| 2026-02-09 | Phase O: Story Shear 이중검증, Model 탭 Capabilities, Envelope 정렬 |
| 2026-02-08 | 부호규약 통일 (sign_convention.py) |
| 2026-02-07 | SFD 불연속 처리 (연속보 point load) |
| 2026-02-04 | Phase K-M: Drift Limit, Envelope, PDF/PNG Export |
| 2026-02-03 | 2D Frame HTML 뷰어 확장 |
| 2026-01-26 | MCP 서버 구축, Simple Beam 구현 |

---

## 12. 외부 참조

- [OpenSeesPy Documentation](https://openseespydoc.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Plotly.js Documentation](https://plotly.com/javascript/)
- [GitHub Repository](https://github.com/Son012375/opensees-frame2d)

---

**이 문서는 새 세션에서 `/read .claude/PROJECT_CONTEXT.md`로 컨텍스트를 복원할 수 있습니다.**
