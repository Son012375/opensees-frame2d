# OpenSeesPy MCP Server

OpenSeesPy 구조해석을 위한 MCP (Model Context Protocol) 서버

## 프로젝트 구조

```
mcp-server/
├── server.py           # MCP 서버 메인
├── core/
│   ├── __init__.py
│   ├── model.py        # OpenSeesPy 모델링 (opensees-ai-agent에서 가져옴)
│   └── simple_beam.py  # 단순보 해석 로직
├── data/
│   ├── sections.json   # 단면 DB
│   └── materials.json  # 재료 DB
├── tools/
│   ├── __init__.py
│   └── analysis.py     # MCP Tool 정의
├── requirements.txt
└── README.md
```

## 설치

```bash
# 가상환경 활성화
conda activate opensees310

# 의존성 설치
pip install -r requirements.txt
```

## 사용법

### Claude Code에서 사용

`claude_desktop_config.json`에 추가:
```json
{
  "mcpServers": {
    "opensees": {
      "command": "python",
      "args": ["d:/son/opensees-MCP/mcp-server/server.py"]
    }
  }
}
```

## Tool 목록

### 구조물 형태별 해석
- `analyze_simple_beam` - 단순보 해석
- `analyze_cantilever` - 캔틸레버 해석 (예정)
- `analyze_frame` - 라멘 프레임 해석 (예정)
- `analyze_truss` - 트러스 해석 (예정)

### 보조 Tool
- `get_section_properties` - 단면 정보 조회
- `get_material_properties` - 재료 정보 조회

## 참고

- 기반 코드: [viktor-platform/opensees-ai-agent](https://github.com/viktor-platform/opensees-ai-agent)
- OpenSeesPy: https://openseespydoc.readthedocs.io/
