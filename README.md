# OpenSees Frame2D Analysis

OpenSeesPy 기반 2D 프레임 구조해석 시스템

## 프로젝트 구성

```
opensees-frame2d/
├── mcp-server/          # MCP 서버 (구조해석 엔진)
│   └── core/
│       ├── frame_2d.py      # 2D 프레임 해석
│       ├── simple_beam.py   # 단순보 해석
│       └── visualization.py # 결과 시각화
│
└── webapp/              # 웹 애플리케이션
    └── backend/
        ├── app/             # FastAPI 앱
        ├── templates/       # HTML 템플릿
        └── static/          # CSS, JS
```

## 주요 기능

### 1. 웹 애플리케이션
- **자연어 입력**: Claude AI가 한국어 설명을 구조해석 입력으로 자동 변환
- **직접 입력**: 폼에서 파라미터 직접 입력
- **결과 시각화**: 변형도, 모멘트, 전단력, 축력 다이어그램

### 2. 구조해석
- 2D Frame 선형 해석 (다층/다경간)
- 단순보, 연속보 해석
- 다양한 하중 케이스 및 조합
- KS 표준 단면/재료 DB (Supabase)

## 기술 스택

| 구분 | 기술 |
|------|------|
| Backend | FastAPI, Python 3.8 |
| Analysis | OpenSeesPy |
| Frontend | Jinja2, HTMX, Plotly |
| AI | Claude API (Anthropic) |
| Database | Supabase |

## 빠른 시작

### 1. 환경 설정

```bash
# Conda 환경 생성 (Python 3.8 필수)
conda create -n opensees38 python=3.8
conda activate opensees38

# 의존성 설치
cd webapp/backend
pip install -r requirements.txt
```

### 2. 서버 실행

**Windows:**
```bash
cd webapp
start_server.bat
```

**수동 실행:**
```bash
set ANTHROPIC_API_KEY=your-api-key
set MCP_SERVER_PATH=/path/to/mcp-server
cd webapp/backend
python -m uvicorn app.main_simple:app --host 0.0.0.0 --port 8001
```

### 3. 접속
- 로컬: http://localhost:8001

## 배포

### Render (무료)
1. GitHub 저장소 연결
2. Environment Variables에 `ANTHROPIC_API_KEY` 설정
3. 자동 배포

자세한 내용은 [webapp/README.md](webapp/README.md) 참조

## 라이선스

MIT License
