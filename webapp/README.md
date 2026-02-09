# Frame2D Analysis Web Application

OpenSees 기반 2D 프레임 구조해석 웹 애플리케이션

## 주요 기능

### 1. 입력 모드
- **자연어 입력**: Claude AI가 한국어 설명을 구조해석 입력으로 자동 변환
- **직접 입력**: 폼에서 직접 파라미터 입력

### 2. 구조해석
- OpenSeesPy 기반 2D Frame 선형 해석
- 다층/다경간 골조 지원 (최대 10층 x 5경간)
- 다양한 하중 케이스 및 조합
- 고정단/핀지지 경계조건

### 3. 결과 시각화
- 변형도, 모멘트, 전단력, 축력 다이어그램
- 층별 변위/층간변위 차트
- 부재력 테이블 (Envelope)
- 인터랙티브 Plotly 차트

## 기술 스택

- **Backend**: FastAPI, Python 3.8
- **Analysis**: OpenSeesPy
- **Frontend**: Jinja2, HTMX, Vanilla JS
- **AI**: Claude API (Anthropic)
- **Database**: Supabase (단면/재료 DB)

## 빠른 시작

### 요구사항
- Python 3.8 (OpenSeesPy 호환 필수)
- Conda (권장)

### 설치

```bash
# Conda 환경 생성
conda create -n opensees38 python=3.8
conda activate opensees38

# 의존성 설치
cd webapp/backend
pip install -r requirements.txt
```

### 실행

**Windows - start_server.bat 사용 (권장)**
```bash
cd webapp
start_server.bat
```

**수동 실행**
```bash
set ANTHROPIC_API_KEY=your-api-key
set MCP_SERVER_PATH=d:/son/opensees-MCP/mcp-server
set PYTHONPATH=D:\son\opensees-MCP\webapp\backend
cd webapp/backend
python -m uvicorn app.main_simple:app --host 0.0.0.0 --port 8001
```

### 접속
- 로컬: http://localhost:8001
- 외부 접속: ngrok 사용 (`ngrok http 8001`)

## 아키텍처 (Simple Mode)

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Browser   │────▶│  FastAPI    │────▶│  OpenSees   │
│  (HTMX)     │     │  (API)      │     │ (Analysis)  │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │
       │                   ▼
       │            ┌─────────────┐
       └───────────▶│  Claude AI  │
                    │  (Parsing)  │
                    └─────────────┘
```

## 입력 스키마 (Frame2DInput)

```json
{
  "stories": [3.5, 3.2, 3.2],
  "bays": [6.0, 6.0],
  "column_section": "H-300x300",
  "beam_section": "H-400x200",
  "material_name": "SS275",
  "supports": "fixed",
  "num_elements_per_member": 4,
  "load_cases": {
    "DL": [
      {"type": "floor", "story": 1, "value": 20},
      {"type": "floor", "story": 2, "value": 20}
    ],
    "EQX": [
      {"type": "lateral", "story": 1, "fx": 30},
      {"type": "lateral", "story": 2, "fx": 60}
    ]
  },
  "load_combinations": {
    "1.2DL+1.0EQX": {"DL": 1.2, "EQX": 1.0}
  }
}
```

## API 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/` | 메인 입력 페이지 |
| POST | `/api/jobs` | 새 분석 작업 생성 |
| GET | `/jobs/{job_id}/status` | 작업 상태/결과 페이지 |
| GET | `/api/jobs/{job_id}` | 작업 상태 JSON |
| GET | `/api/claude/status` | Claude API 상태 확인 |
| POST | `/api/claude/parse` | 자연어 → JSON 변환 |

## 디렉토리 구조

```
webapp/
├── start_server.bat        # 서버 시작 스크립트 (API 키 포함)
├── README.md
└── backend/
    ├── requirements.txt
    ├── jobs/               # 작업 결과 저장
    ├── static/
    │   ├── css/style.css
    │   └── js/form.js
    ├── templates/
    │   ├── base.html
    │   ├── index.html      # 입력 페이지
    │   └── job_status.html # 결과 페이지
    └── app/
        ├── main_simple.py  # FastAPI 앱 (Simple Mode)
        ├── core/
        │   ├── config.py
        │   └── claude_service.py  # Claude API 연동
        └── models/
            └── schemas.py  # Pydantic 모델
```

## 환경변수

| 변수 | 설명 |
|------|------|
| `ANTHROPIC_API_KEY` | Claude API 키 (자연어 입력용) |
| `MCP_SERVER_PATH` | mcp-server 경로 |
| `PYTHONPATH` | webapp/backend 경로 |

## 외부 접속 (ngrok)

```bash
# ngrok 설치 후
ngrok http 8001

# 발급된 URL로 외부 접속 가능
# 예: https://xxxx.ngrok-free.app
```

## 배포 옵션

| 플랫폼 | 지원 | 비고 |
|--------|------|------|
| ngrok | ✅ | 로컬 PC 필요, 무료 |
| Railway | ✅ | Docker 필요, $5/월 무료 크레딧 |
| Render | ✅ | Docker 필요, 무료 티어 있음 |
| GitHub Pages | ❌ | 정적 사이트만 지원 |

## 라이선스

MIT License
