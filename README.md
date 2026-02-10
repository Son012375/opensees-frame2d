# OpenSees-MCP

OpenSeesPy 기반 구조해석 웹 플랫폼 - Claude AI 자연어 입력 지원

[![Deploy to Render](https://img.shields.io/badge/Deploy-Render-46E3B7?logo=render)](https://render.com)
[![Python 3.8](https://img.shields.io/badge/Python-3.8-blue?logo=python)](https://www.python.org/)
[![OpenSeesPy](https://img.shields.io/badge/Engine-OpenSeesPy-orange)](https://openseespydoc.readthedocs.io/)

## 개요

OpenSees-MCP는 구조공학용 해석 플랫폼으로, 사용자가 한국어 자연어로 구조물을 설명하면 Claude AI가 이를 파싱하여 OpenSeesPy로 해석을 수행하고 결과를 시각화합니다.

## 지원 해석 유형

| 해석 유형 | 상태 | 주요 기능 |
|-----------|------|-----------|
| **단순보 (Simple Beam)** | ✅ Ready | 단순지지/캔틸레버/고정단, 분포/집중/조합하중 |
| **연속보 (Continuous Beam)** | ✅ Ready | 다경간, 내부 힌지, 다양한 하중 패턴 |
| **2D 골조 (Frame 2D)** | ✅ Ready | 다층/다경간, 하중조합, 층간변위, Envelope |
| **3D 골조 (Frame 3D)** | 🚧 Coming Soon | 3차원 해석, 비틀림 |

## 주요 기능

### 해석 기능
- **다양한 하중 유형**: 분포하중, 집중하중, 모멘트, 횡하중(EQ)
- **하중 조합**: DL, LL, EQ 등 하중케이스 선형조합
- **Envelope 분석**: 모든 케이스에서 최대/최소 부재력 추출
- **층간변위 검토**: 사용자 정의 허용기준 (1/200, 1/400 등)

### 시각화
- **변형도**: 원본/변형 형상 중첩, 변위 스케일 조절
- **부재력 다이어그램**: N (축력), V (전단력), M (모멘트)
- **SFD/BMD**: 교과서 부호규약 적용 (V>0: 좌측면 상향, M>0: sagging)
- **Story Response**: 층변위/층전단력 프로파일, 반력/요소 기반 이중검증

### 입출력
- **자연어 입력**: Claude AI가 한국어 설명을 구조해석 입력으로 변환
- **폼 입력**: 직접 파라미터 입력 지원
- **CSV Export**: 노드, 반력, 부재력, 층데이터
- **PNG Export**: 각 다이어그램 이미지 저장
- **PDF Report**: Print 기능으로 전체 리포트 출력

## 프로젝트 구조

```
opensees-MCP/
├── mcp-server/                    # 구조해석 엔진
│   ├── core/
│   │   ├── simple_beam.py         # 단순보 해석
│   │   ├── continuous_beam.py     # 연속보 해석
│   │   ├── frame_2d.py            # 2D 프레임 해석
│   │   ├── visualization.py       # 결과 시각화 (HTML 리포트)
│   │   ├── sign_convention.py     # 부호규약 변환
│   │   └── verification.py        # 수치 검증
│   ├── tools/
│   │   └── opensees_tools.py      # MCP 도구 정의
│   └── tests/
│       └── test_sign_convention.py
│
└── webapp/                        # 웹 애플리케이션
    └── backend/
        ├── app/
        │   └── main_simple.py     # FastAPI 앱
        ├── templates/             # Jinja2 템플릿
        │   ├── home.html          # 메인 페이지
        │   ├── simple_beam.html   # 단순보 입력
        │   ├── continuous_beam.html
        │   └── index.html         # Frame 2D 입력
        └── static/
            ├── css/style.css
            └── js/main.js
```

## 기술 스택

| 구분 | 기술 |
|------|------|
| **Backend** | FastAPI, Python 3.8 |
| **Analysis Engine** | OpenSeesPy (elasticBeamColumn) |
| **Frontend** | Jinja2, HTMX, Plotly.js |
| **AI** | Claude API (Anthropic) |
| **Database** | Supabase (KS 표준 단면/재료 DB) |
| **Deployment** | Render |

## 부호규약 (Sign Convention)

시각화에는 **교과서/MIDAS 부호규약**이 적용됩니다:

| 구분 | 규약 | 설명 |
|------|------|------|
| 전단력 V | V > 0 | 좌측 절단면에서 상향 (↑) |
| 모멘트 M | M > 0 | Sagging (하부 인장) |
| 축력 N | N > 0 | 인장 (+), 압축 (-) |

**변환 규칙** (OpenSees → 교과서):
```python
V_textbook = -V_opensees
M_textbook = -M_opensees
```

## 설치 및 실행

### 1. 환경 설정

```bash
# Conda 환경 생성 (Python 3.8 필수 - OpenSeesPy 요구사항)
conda create -n opensees38 python=3.8
conda activate opensees38

# 의존성 설치
cd webapp/backend
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
# Windows
set ANTHROPIC_API_KEY=your-api-key
set SUPABASE_URL=your-supabase-url
set SUPABASE_KEY=your-supabase-key

# Linux/Mac
export ANTHROPIC_API_KEY=your-api-key
export SUPABASE_URL=your-supabase-url
export SUPABASE_KEY=your-supabase-key
```

### 3. 서버 실행

```bash
cd webapp/backend
python -m uvicorn app.main_simple:app --host 0.0.0.0 --port 8001
```

또는 Windows:
```bash
cd webapp
start_server.bat
```

### 4. 접속

- 로컬: http://localhost:8001

## API 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/` | 메인 페이지 |
| GET | `/simple-beam` | 단순보 입력 페이지 |
| GET | `/continuous-beam` | 연속보 입력 페이지 |
| GET | `/frame2d` | 2D 골조 입력 페이지 |
| POST | `/api/jobs` | 2D Frame 해석 Job 생성 |
| POST | `/api/simple-beam/jobs` | 단순보 해석 Job 생성 |
| POST | `/api/continuous-beam/jobs` | 연속보 해석 Job 생성 |
| GET | `/api/jobs/{job_id}/report` | 해석 결과 리포트 |
| POST | `/api/claude/parse` | 자연어 → 입력 파싱 |

## 배포 (Render)

1. GitHub 저장소 연결
2. Environment Variables 설정:
   - `ANTHROPIC_API_KEY`
   - `SUPABASE_URL`
   - `SUPABASE_KEY`
3. Build Command: `pip install -r webapp/backend/requirements.txt`
4. Start Command: `cd webapp/backend && python -m uvicorn app.main_simple:app --host 0.0.0.0 --port $PORT`

## 개발 현황

### 완료된 기능
- [x] 단순보 해석 (다양한 지지조건, 하중유형)
- [x] 연속보 해석 (다경간, SFD 불연속 처리)
- [x] 2D Frame 해석 (다층/다경간)
- [x] 하중 조합 및 Envelope 분석
- [x] 층간변위 검토 (사용자 정의 허용기준)
- [x] Story Shear 이중검증 (반력/요소 기반)
- [x] 부호규약 통일 (교과서 규약)
- [x] CSV/PNG/PDF Export
- [x] Claude AI 자연어 입력

### 진행 예정
- [ ] 3D Frame 해석
- [ ] 부재 단부 릴리즈 (힌지)
- [ ] Rigid offset
- [ ] P-Delta 해석
- [ ] 자중 자동 적용
- [ ] 전단변형 (Timoshenko beam)

## 제한사항

현재 버전에서 **지원되지 않는** 기능:
- End release (힌지 조인트)
- Rigid offset
- Shear deformation (Timoshenko beam)
- P-Delta (기하비선형)
- Self-weight 자동 계산

## 라이선스

MIT License

## 관련 문서

- [OpenSeesPy Documentation](https://openseespydoc.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Plotly.js Documentation](https://plotly.com/javascript/)
